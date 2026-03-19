#!/usr/bin/env python3
"""
SILCS-lite competitive probe saturation MD for pocket mapping.

Runs a short MD simulation with the protein soaked in a mixture of small
probe molecules (benzene, propane, methanol, formamide). Probes compete
for binding sites, generating occupancy-based Grid Free Energy (GFE) maps.

The ligand is REMOVED from the pocket so probes can explore the binding site.

Probe types and their pharmacophore mapping:
  - Benzene (c1ccccc1)   → Hydrophobic (aromatic)
  - Propane (CCC)         → Hydrophobic (aliphatic)
  - Methanol (CO)         → H-bond donor (OH) + acceptor (O)
  - Formamide (NC=O)      → H-bond donor (NH2) + acceptor (C=O)

Output: 3 DX files + binding_site_results.json (same format as map_binding_site.py).
"""

import argparse
import json
import os
import random
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

try:
    from openmm import *
    from openmm.app import *
    from openmm.unit import *
    import numpy as np
    from scipy.spatial import cKDTree
    import rdkit.Chem as Chem
    from openff.toolkit import Molecule
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator
    from pdbfixer import PDBFixer
except ImportError as e:
    print(f"ERROR:Missing dependency: {e}", file=sys.stderr)
    sys.exit(1)

from utils import write_dx, find_hotspots, normalize_grid

# Load OpenMM plugins
_default_plugins = Platform.getDefaultPluginsDirectory()
_bundled_plugins = os.path.join(os.path.dirname(os.path.realpath(sys.executable)), '..', 'lib', 'plugins')
for _pdir in [_default_plugins, _bundled_plugins]:
    _pdir = os.path.normpath(_pdir)
    if os.path.isdir(_pdir):
        try:
            Platform.loadPluginsFromDirectory(_pdir)
        except Exception:
            pass


# ============================================================
# Probe definitions
# ============================================================

PROBES = {
    'benzene':   {'smiles': 'c1ccccc1', 'channel': 'hydrophobic', 'copies': 12},
    'propane':   {'smiles': 'CCC',      'channel': 'hydrophobic', 'copies': 12},
    'methanol':  {'smiles': 'CO',       'channel': 'donor',       'copies': 12},
    'formamide': {'smiles': 'NC=O',     'channel': 'acceptor',    'copies': 12},
}

# Atoms used for occupancy binning per probe type
PROBE_TRACK_ATOMS = {
    'benzene':   'aromatic carbon',   # Ring carbons
    'propane':   'carbon',            # All carbons
    'methanol':  'oxygen',            # OH oxygen (donor + acceptor)
    'formamide': 'nitrogen',          # NH2 nitrogen (donor) — also track O for acceptor
}


def find_ligand_com_and_remove(pdb_path: str, ligand_resname: str, ligand_resnum: int,
                                output_pdb: str) -> Any:
    """Find ligand COM, then save PDB without ligand for probe simulation."""
    from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select

    if pdb_path.lower().endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_path)

    ligand_coords = []
    for chain in structure[0]:
        for residue in chain:
            resname = residue.get_resname().strip()
            resnum = residue.get_id()[1]
            if resname == ligand_resname and resnum == ligand_resnum:
                for atom in residue:
                    ligand_coords.append(atom.get_vector().get_array())

    if not ligand_coords:
        print(f"Error: Ligand {ligand_resname} {ligand_resnum} not found", file=sys.stderr)
        sys.exit(1)

    ligand_com = np.array(ligand_coords).mean(axis=0)

    # Save PDB without ligand
    class NotLigand(Select):
        def accept_residue(self, residue):
            resname = residue.get_resname().strip()
            resnum = residue.get_id()[1]
            if resname == ligand_resname and resnum == ligand_resnum:
                return 0
            return 1

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, NotLigand())

    return ligand_com


def select_platform() -> Tuple[Any, Dict]:
    """Select best available OpenMM platform."""
    for name in ['OpenCL', 'Metal', 'CPU']:
        try:
            platform = Platform.getPlatformByName(name)
            props = {}
            if name == 'OpenCL':
                props = {'Precision': 'single'}
            return platform, props
        except Exception:
            continue
    return Platform.getPlatformByName('CPU'), {}


def prepare_system_with_probes(apo_pdb_path: str, ligand_com: Any,
                               output_dir: str) -> Tuple[Any, Any, Any, Dict]:
    """Prepare solvated system with probe molecules placed around binding site.

    Returns (topology, positions, system, probe_atom_indices).
    """
    print("PROGRESS: Preparing protein with PDBFixer...")

    fixer = PDBFixer(filename=apo_pdb_path)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)

    # Set up force field with OpenFF for probes
    ff = ForceField('amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml')

    # Register probe molecules with OpenFF
    print("PROGRESS: Parameterizing probe molecules with OpenFF Sage 2.3.0...")
    probe_mols = {}
    for name, info in PROBES.items():
        rdmol = Chem.MolFromSmiles(info['smiles'])
        rdmol = Chem.AddHs(rdmol)
        Chem.AllChem.EmbedMolecule(rdmol, randomSeed=42)
        Chem.AllChem.MMFFOptimizeMolecule(rdmol)
        off_mol = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)
        probe_mols[name] = (rdmol, off_mol)

    smirnoff = SMIRNOFFTemplateGenerator(molecules=[m[1] for m in probe_mols.values()], forcefield='openff-2.3.0')
    ff.registerTemplateGenerator(smirnoff.generator)

    # Solvate
    print("PROGRESS: Solvating protein...")
    modeller = Modeller(fixer.topology, fixer.positions)
    modeller.addSolvent(ff, model='tip3p', padding=1.2 * nanometers, ionicStrength=0.15 * molar)

    # Get water positions and find those near the binding site
    print("PROGRESS: Placing probe molecules near binding site...")
    positions = modeller.getPositions(asNumpy=True)
    pos_nm = positions.value_in_unit(nanometers)
    com_nm = ligand_com / 10.0  # A -> nm

    # Find water oxygens near binding site (within 1.5 nm of ligand COM)
    water_oxy_indices = []
    for atom in modeller.topology.atoms():
        if atom.residue.name in ('HOH', 'WAT', 'TIP3') and atom.name == 'O':
            water_oxy_indices.append(atom.index)

    water_oxy_indices = np.array(water_oxy_indices)
    if len(water_oxy_indices) == 0:
        print("Error: No water molecules found after solvation", file=sys.stderr)
        sys.exit(1)

    water_positions = pos_nm[water_oxy_indices]
    distances = np.linalg.norm(water_positions - com_nm, axis=1)

    # Select waters in shell 0.8-1.5 nm from ligand COM for replacement
    replacement_mask = (distances > 0.8) & (distances < 1.5)
    candidate_indices = water_oxy_indices[replacement_mask]

    # Randomly select waters to replace with probes
    rng = random.Random(42)
    probe_atom_indices = {}  # probe_name -> list of atom indices
    waters_to_remove = set()
    probes_placed = {name: 0 for name in PROBES}

    # Shuffle candidates and place probes with clash checking
    candidates = list(candidate_indices)
    rng.shuffle(candidates)

    placed_positions = []  # positions of placed probes for clash checking
    min_probe_distance = 0.35  # nm, minimum distance between probes

    for water_idx in candidates:
        # Check if we still need more probes
        all_done = all(probes_placed[n] >= PROBES[n]['copies'] for n in PROBES)
        if all_done:
            break

        # Pick which probe type needs more copies
        for probe_name in PROBES:
            if probes_placed[probe_name] >= PROBES[probe_name]['copies']:
                continue

            water_pos = pos_nm[water_idx]

            # Check clash with already-placed probes
            if placed_positions:
                min_dist = min(np.linalg.norm(water_pos - pp) for pp in placed_positions)
                if min_dist < min_probe_distance:
                    continue

            # Find the water residue (O + H + H) to remove
            for atom in modeller.topology.atoms():
                if atom.index == water_idx:
                    water_res = atom.residue
                    for wa in water_res.atoms():
                        waters_to_remove.add(wa.index)
                    break

            placed_positions.append(water_pos)
            probes_placed[probe_name] += 1
            break

    print(f"PROGRESS: Probes placed: {probes_placed}")

    # Remove selected waters
    atoms_to_keep = [
        atom for atom in modeller.topology.atoms()
        if atom.index not in waters_to_remove
    ]
    modeller.delete(list(waters_to_remove))

    # Add probe molecules at the positions where waters were removed
    # For simplicity in this implementation, we add probes via the Modeller
    # by adding their topologies and positions at the water replacement sites
    probe_idx = 0
    for probe_name, count in probes_placed.items():
        if count == 0:
            continue
        rdmol, off_mol = probe_mols[probe_name]

        for i in range(count):
            if probe_idx >= len(placed_positions):
                break

            target_pos = placed_positions[probe_idx]

            # Get probe atom positions (from RDKit conformer)
            conf = rdmol.GetConformer()
            probe_positions = []
            for j in range(rdmol.GetNumAtoms()):
                pos = conf.GetAtomPosition(j)
                # Center on target position (convert from A to nm)
                probe_positions.append(
                    Vec3(
                        target_pos[0] + pos.x / 10.0 - conf.GetAtomPosition(0).x / 10.0,
                        target_pos[1] + pos.y / 10.0 - conf.GetAtomPosition(0).y / 10.0,
                        target_pos[2] + pos.z / 10.0 - conf.GetAtomPosition(0).z / 10.0,
                    ) * nanometers
                )

            try:
                modeller.add(
                    off_mol.to_topology().to_openmm(),
                    probe_positions
                )

                # Track the atom indices for this probe
                n_atoms_now = sum(1 for _ in modeller.topology.atoms())
                n_probe_atoms = rdmol.GetNumAtoms()
                start_idx = n_atoms_now - n_probe_atoms
                if probe_name not in probe_atom_indices:
                    probe_atom_indices[probe_name] = []
                probe_atom_indices[probe_name].extend(range(start_idx, n_atoms_now))
            except Exception as e:
                print(f"  Warning: Failed to add {probe_name} copy {i}: {e}")

            probe_idx += 1

    total_probes = sum(probes_placed.values())
    print(f"PROGRESS: System prepared: {total_probes} probe molecules added")

    # Create system
    print("PROGRESS: Building OpenMM system...")
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * nanometers,
        constraints=HBonds,
        hydrogenMass=1.5 * amu,
    )

    # Save system PDB
    system_pdb = os.path.join(output_dir, 'probe_system.pdb')
    with open(system_pdb, 'w') as f:
        PDBFile.writeFile(modeller.topology, modeller.getPositions(), f)
    print(f"PROGRESS: System PDB saved: {system_pdb}")

    return modeller.topology, modeller.getPositions(), system, probe_atom_indices


def run_probe_simulation(topology: Any, positions: Any, system: Any,
                         output_dir: str, production_ns: float = 2.0) -> str:
    """Run abbreviated equilibration + production MD. Returns trajectory path."""
    platform, props = select_platform()
    print(f"PROGRESS: Platform: {platform.getName()}")

    # Integrator: 4fs with HMR
    integrator = LangevinMiddleIntegrator(300 * kelvin, 1.0 / picosecond, 4.0 * femtoseconds)
    integrator.setRandomNumberSeed(42)

    simulation = Simulation(topology, system, integrator, platform, props)
    simulation.context.setPositions(positions)

    # Minimize
    print("PROGRESS: Minimizing energy...")
    simulation.minimizeEnergy(maxIterations=1000)

    # Brief NVT heating
    print("PROGRESS: NVT heating 10K -> 300K...")
    simulation.context.setVelocitiesToTemperature(10 * kelvin, 42)

    n_heat_steps = 25000  # 100ps at 4fs
    temp_increment = (300 - 10) / 10
    for i in range(10):
        temp = (10 + temp_increment * (i + 1)) * kelvin
        integrator.setTemperature(temp)
        simulation.step(n_heat_steps // 10)
        print(f"PROGRESS: Heating... {int((i + 1) * 10)}%")

    # Add barostat for NPT
    system.addForce(MonteCarloBarostat(1.0 * bar, 300 * kelvin, 25))
    simulation.context.reinitialize(preserveState=True)

    # Brief NPT equilibration (200ps)
    print("PROGRESS: NPT equilibration (200ps)...")
    simulation.step(50000)  # 200ps at 4fs

    # Production
    traj_path = os.path.join(output_dir, 'probe_trajectory.dcd')
    n_production_steps = int(production_ns * 1e6 / 4)  # ns -> steps at 4fs
    report_interval = max(250, n_production_steps // 200)  # ~200 frames

    simulation.reporters.append(DCDReporter(traj_path, report_interval))
    simulation.reporters.append(
        StateDataReporter(
            os.path.join(output_dir, 'probe_energy.csv'),
            report_interval * 10,
            step=True, potentialEnergy=True, temperature=True,
            speed=True
        )
    )

    print(f"PROGRESS: Production MD ({production_ns}ns)...")
    steps_done = 0
    chunk = max(1, n_production_steps // 20)
    while steps_done < n_production_steps:
        steps_to_do = min(chunk, n_production_steps - steps_done)
        simulation.step(steps_to_do)
        steps_done += steps_to_do
        pct = int(100 * steps_done / n_production_steps)
        print(f"PROGRESS: Production... {pct}%")

    print(f"PROGRESS: Trajectory saved: {traj_path}")
    return traj_path


def analyze_probe_occupancy(system_pdb: str, traj_path: str,
                            probe_atom_indices: Dict[str, List[int]],
                            ligand_com: Any, padding: float, spacing: float,
                            output_dir: str) -> Tuple[Any, Any, Any, Tuple[int, int, int], Any]:
    """Analyze probe atom occupancy to generate GFE maps.

    Returns (hydro_3d, donor_3d, acceptor_3d, shape, origin).
    """
    import MDAnalysis as mda

    print("PROGRESS: Analyzing probe occupancy...")

    u = mda.Universe(system_pdb, traj_path)

    # Set up grid
    grid_min = ligand_com - padding
    nx = int(np.ceil(2 * padding / spacing)) + 1
    ny = int(np.ceil(2 * padding / spacing)) + 1
    nz = int(np.ceil(2 * padding / spacing)) + 1
    shape = (nx, ny, nz)
    origin = grid_min.tolist()

    # Initialize occupancy grids per channel
    hydro_occ = np.zeros(shape)
    donor_occ = np.zeros(shape)
    acceptor_occ = np.zeros(shape)

    n_frames = len(u.trajectory)
    report_interval = max(1, n_frames // 10)

    for frame_idx, ts in enumerate(u.trajectory):
        if frame_idx % report_interval == 0:
            pct = int(100 * frame_idx / n_frames)
            print(f"PROGRESS: Binning probe positions... {pct}%")

        for probe_name, atom_ids in probe_atom_indices.items():
            if not atom_ids:
                continue

            try:
                # Select probe atoms by index
                probe_atoms = u.atoms[atom_ids]
                positions = probe_atoms.positions  # in Angstroms
            except (IndexError, ValueError):
                continue

            # Bin positions into grid
            indices = ((positions - grid_min) / spacing).astype(int)
            valid = (
                (indices[:, 0] >= 0) & (indices[:, 0] < nx) &
                (indices[:, 1] >= 0) & (indices[:, 1] < ny) &
                (indices[:, 2] >= 0) & (indices[:, 2] < nz)
            )
            valid_idx = indices[valid]

            channel = PROBES[probe_name]['channel']
            for idx in valid_idx:
                if channel == 'hydrophobic':
                    hydro_occ[idx[0], idx[1], idx[2]] += 1
                elif channel == 'donor':
                    donor_occ[idx[0], idx[1], idx[2]] += 1
                elif channel == 'acceptor':
                    acceptor_occ[idx[0], idx[1], idx[2]] += 1

    print("PROGRESS: Binning probe positions... 100%")

    # Convert occupancy to Grid Free Energy: GFE = -kT * ln(rho / rho_bulk)
    # rho = occupancy / (n_frames * voxel_volume)
    # rho_bulk estimated from total occupancy in outer shell
    kT = 0.596  # kcal/mol at 300K
    voxel_vol = spacing ** 3

    def occupancy_to_gfe(occ):
        density = occ / (n_frames * voxel_vol)
        # Estimate bulk density from the 10th percentile of nonzero values
        nonzero = density[density > 0]
        if len(nonzero) < 10:
            return np.zeros(shape)
        bulk = np.percentile(nonzero, 10)
        if bulk <= 0:
            return np.zeros(shape)

        gfe = np.zeros(shape)
        mask = density > 0
        gfe[mask] = -kT * np.log(density[mask] / bulk)
        # Negative GFE = favorable binding, clip positive (unfavorable)
        return np.maximum(0, -gfe)  # Flip sign: positive = favorable

    hydro_3d = occupancy_to_gfe(hydro_occ)
    donor_3d = occupancy_to_gfe(donor_occ)
    acceptor_3d = occupancy_to_gfe(acceptor_occ)

    return hydro_3d, donor_3d, acceptor_3d, shape, origin


def main() -> None:
    parser = argparse.ArgumentParser(description='SILCS-lite probe saturation MD')
    parser.add_argument('--pdb_path', required=True, help='PDB file (protein + ligand)')
    parser.add_argument('--ligand_resname', required=True, help='Ligand residue name')
    parser.add_argument('--ligand_resnum', required=True, type=int, help='Ligand residue number')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--project_name', default=None, help='Project name prefix')
    parser.add_argument('--production_ns', type=float, default=2.0, help='Production MD length (ns)')
    parser.add_argument('--box_padding', type=float, default=8.0, help='Grid padding in Angstroms')
    parser.add_argument('--grid_spacing', type=float, default=0.75, help='Grid spacing in Angstroms')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()

    # Step 1: Find ligand COM and remove ligand from PDB
    print(f"PROGRESS: Loading PDB: {args.pdb_path}")
    apo_pdb = os.path.join(args.output_dir, 'apo_protein.pdb')
    ligand_com = find_ligand_com_and_remove(
        args.pdb_path, args.ligand_resname, args.ligand_resnum, apo_pdb
    )
    print(f"PROGRESS: Ligand COM: {ligand_com}")
    print(f"PROGRESS: Apo protein saved: {apo_pdb}")

    # Step 2: Prepare system with probes
    topology, positions, system, probe_atom_indices = prepare_system_with_probes(
        apo_pdb, ligand_com, args.output_dir
    )

    total_probe_atoms = sum(len(v) for v in probe_atom_indices.values())
    print(f"PROGRESS: Probe atom indices tracked: {total_probe_atoms}")

    # Step 3: Run MD simulation
    traj_path = run_probe_simulation(
        topology, positions, system,
        args.output_dir, args.production_ns
    )

    # Step 4: Analyze probe occupancy
    system_pdb = os.path.join(args.output_dir, 'probe_system.pdb')
    hydro_3d, donor_3d, acceptor_3d, shape, origin = analyze_probe_occupancy(
        system_pdb, traj_path, probe_atom_indices,
        ligand_com, args.box_padding, args.grid_spacing, args.output_dir
    )

    hydro_3d = normalize_grid(hydro_3d)
    donor_3d = normalize_grid(donor_3d)
    acceptor_3d = normalize_grid(acceptor_3d)

    # Step 5: Write DX files
    prefix = f'{args.project_name}_' if args.project_name else ''
    hydro_path = os.path.join(args.output_dir, f'{prefix}hydrophobic.dx')
    donor_path = os.path.join(args.output_dir, f'{prefix}hbond_donor.dx')
    acceptor_path = os.path.join(args.output_dir, f'{prefix}hbond_acceptor.dx')

    print("PROGRESS: Writing DX files...")
    write_dx(hydro_path, hydro_3d, origin, args.grid_spacing, shape)
    write_dx(donor_path, donor_3d, origin, args.grid_spacing, shape)
    write_dx(acceptor_path, acceptor_3d, origin, args.grid_spacing, shape)

    # Find hotspots
    print("PROGRESS: Identifying hotspots...")
    hotspots = []
    hotspots.extend(find_hotspots(hydro_3d, 'hydrophobic', origin, args.grid_spacing, ligand_com))
    hotspots.extend(find_hotspots(donor_3d, 'hbond_donor', origin, args.grid_spacing, ligand_com))
    hotspots.extend(find_hotspots(acceptor_3d, 'hbond_acceptor', origin, args.grid_spacing, ligand_com))

    # Write results JSON
    elapsed = time.time() - start_time
    results = {
        'hydrophobicDx': hydro_path,
        'hbondDonorDx': donor_path,
        'hbondAcceptorDx': acceptor_path,
        'hotspots': hotspots,
        'gridDimensions': list(shape),
        'ligandCom': [round(float(c), 3) for c in ligand_com],
        'method': 'probe',
        'simulationNs': args.production_ns,
        'elapsedSeconds': round(elapsed, 1),
    }

    results_path = os.path.join(args.output_dir, f'{prefix}binding_site_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"PROGRESS: Results written to {results_path}")
    print(f"PROGRESS: Hotspots found: {len(hotspots)}")
    print(f"PROGRESS: Total time: {elapsed:.0f}s")
    print("Done!")


if __name__ == '__main__':
    main()
