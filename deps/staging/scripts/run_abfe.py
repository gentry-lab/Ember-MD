#!/usr/bin/env python3
"""
Absolute Binding Free Energy (ABFE) calculation using alchemical FEP.

Extracts snapshots from an equilibrated MD trajectory, builds alchemical
systems using openmmtools AbsoluteAlchemicalFactory, runs lambda windows
for both complex and solvent legs, and computes ΔG_bind via MBAR.

Progress protocol (stdout):
  RMSD_DATA:{"timeNs":[...],"rmsd":[...]}
  PROGRESS:snapshot:0:complex:3:50
  FEP_RESULT:{"snapshotIndex":0,"deltaG_bind":-8.2,...}
"""

import argparse
import json
import math
import os
import signal
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Graceful shutdown on SIGTERM
_cancelled = False
def _handle_sigterm(signum: int, frame: Any) -> None:
    global _cancelled
    _cancelled = True
    print("SIGTERM received, finishing current window and saving partial results...",
          file=sys.stderr)
signal.signal(signal.SIGTERM, _handle_sigterm)


def progress(snapshot_idx: int, leg: str, window_idx: int, pct: int) -> None:
    """Emit progress line for the frontend."""
    print(f"PROGRESS:snapshot:{snapshot_idx}:{leg}:{window_idx}:{pct}", flush=True)


def get_lambda_schedule(speed_preset: str) -> Tuple[List[float], List[float]]:
    """Return (elec_lambdas, steric_lambdas) for the given preset."""
    if speed_preset == 'fast':
        elec = [1.0, 0.75, 0.5, 0.25, 0.0]
        sterics = [0.9, 0.7, 0.5, 0.0]
    else:  # accurate
        elec = [1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.0]
        sterics = [0.9, 0.75, 0.5, 0.25, 0.0]
    return elec, sterics


def get_sim_lengths(speed_preset: str) -> Tuple[float, float]:
    """Return (equil_ns, prod_ns) per window."""
    if speed_preset == 'fast':
        return 0.25, 0.75
    else:
        return 0.5, 1.5


def detect_ligand_atoms(topology: Any) -> Tuple[Any, Optional[List[int]]]:
    """Find ligand residue and return atom indices. Looks for non-standard residues
    that aren't water/ions."""
    standard_residues = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        'HIE', 'HID', 'HIP', 'CYX', 'ASH', 'GLH',
        # Modified amino acids
        'MSE', 'SEC', 'PCA', 'HYP', 'SEP', 'TPO', 'PTR',
        # Water / ions
        'HOH', 'WAT', 'TIP3', 'TP4', 'OPC', 'SPC', 'SOL',
        'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'Na+', 'Cl-',
        # Caps
        'ACE', 'NME', 'NHE',
    }
    ligand_residues = []
    for residue in topology.residues():
        if residue.name not in standard_residues:
            ligand_residues.append(residue)

    if not ligand_residues:
        return None, None

    # Use the first non-standard residue as ligand
    lig_res = ligand_residues[0]
    atom_indices = [atom.index for atom in lig_res.atoms()]
    return lig_res, atom_indices


def select_boresch_anchors(positions: Any, topology: Any, ligand_atom_indices: List[int]) -> Tuple[List[int], List[int], float, float, float]:
    """Select 3 protein CA atoms and 3 ligand heavy atoms for Boresch restraints.

    Returns: (protein_indices, ligand_indices, r0, theta_A0, theta_B0)
    """
    positions_nm = np.array([[p.x, p.y, p.z] for p in positions])

    # Ligand COM
    lig_pos = positions_nm[ligand_atom_indices]
    lig_com = lig_pos.mean(axis=0)

    # Find protein CA atoms within 8 Angstrom of ligand COM
    ca_indices = []
    for atom in topology.atoms():
        if atom.name == 'CA' and atom.residue.name not in (
            'HOH', 'WAT', 'TIP3', 'TP4', 'OPC', 'NA', 'CL'):
            ca_indices.append(atom.index)

    ca_indices = np.array(ca_indices)
    if len(ca_indices) < 3:
        raise ValueError("Not enough protein CA atoms found for Boresch restraints")

    ca_pos = positions_nm[ca_indices]
    dists = np.linalg.norm(ca_pos - lig_com, axis=1)
    nearby_mask = dists < 0.8  # 8 Angstrom in nm
    nearby_ca = ca_indices[nearby_mask]

    if len(nearby_ca) < 3:
        # Expand search radius
        sorted_idx = np.argsort(dists)
        nearby_ca = ca_indices[sorted_idx[:10]]

    # Pick 3 CA atoms forming non-degenerate triangle
    protein_anchors = _pick_triangle(positions_nm, nearby_ca, min_dist=0.3, min_angle=30.0)

    # Pick 3 ligand heavy atoms (prefer ring atoms)
    all_atoms = list(topology.atoms())
    heavy_lig = []
    for idx in ligand_atom_indices:
        if all_atoms[idx].element.symbol != 'H':
            heavy_lig.append(idx)

    if len(heavy_lig) < 3:
        raise ValueError("Not enough ligand heavy atoms for Boresch restraints")

    ligand_anchors = _pick_triangle(positions_nm, np.array(heavy_lig), min_dist=0.15, min_angle=30.0)

    # Compute reference values
    p_pos = positions_nm[protein_anchors]
    l_pos = positions_nm[ligand_anchors]

    r0 = float(np.linalg.norm(p_pos[2] - l_pos[0]))

    # Angle A: p1-p2-l0
    v1 = p_pos[1] - p_pos[2]
    v2 = l_pos[0] - p_pos[2]
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    theta_A0 = float(np.arccos(np.clip(cos_a, -1, 1)))

    # Angle B: p2-l0-l1
    v1 = p_pos[2] - l_pos[0]
    v2 = l_pos[1] - l_pos[0]
    cos_b = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    theta_B0 = float(np.arccos(np.clip(cos_b, -1, 1)))

    return list(protein_anchors), list(ligand_anchors), r0, theta_A0, theta_B0


def _pick_triangle(positions: Any, candidates: Any, min_dist: float = 0.3, min_angle: float = 30.0) -> List[int]:
    """Pick 3 atoms from candidates forming a non-degenerate triangle."""
    min_angle_rad = min_angle * math.pi / 180.0
    n = len(candidates)
    for i in range(n):
        for j in range(i + 1, n):
            d_ij = np.linalg.norm(positions[candidates[i]] - positions[candidates[j]])
            if d_ij < min_dist:
                continue
            for k in range(j + 1, n):
                d_ik = np.linalg.norm(positions[candidates[i]] - positions[candidates[k]])
                d_jk = np.linalg.norm(positions[candidates[j]] - positions[candidates[k]])
                if d_ik < min_dist or d_jk < min_dist:
                    continue
                # Check angles
                angles_ok = True
                for (a, b, c) in [(i, j, k), (j, i, k), (k, i, j)]:
                    v1 = positions[candidates[b]] - positions[candidates[a]]
                    v2 = positions[candidates[c]] - positions[candidates[a]]
                    cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                    ang = np.arccos(np.clip(cos_ang, -1, 1))
                    if ang < min_angle_rad:
                        angles_ok = False
                        break
                if angles_ok:
                    return [candidates[i], candidates[j], candidates[k]]
    # Fallback: just use first 3
    return [candidates[0], candidates[1], candidates[2]]


def add_boresch_restraints(system: Any, positions: Any, topology: Any, ligand_atom_indices: List[int]) -> Tuple[Any, float]:
    """Add Boresch orientational restraints to the complex system.

    Returns: (restrained_system, dG_restraint_kcal)
    """
    from openmm import CustomBondForce, CustomAngleForce
    from openmm.unit import kilocalories_per_mole, angstroms, radians

    prot_idx, lig_idx, r0, theta_A0, theta_B0 = select_boresch_anchors(
        positions, topology, ligand_atom_indices
    )

    # Distance restraint: k_r * (r - r0)^2
    k_r = 10.0  # kcal/mol/A^2
    k_r_kj = k_r * 4.184 * 100  # convert to kJ/mol/nm^2

    bond_force = CustomBondForce("0.5*k*(r-r0)^2")
    bond_force.addPerBondParameter("k")
    bond_force.addPerBondParameter("r0")
    bond_force.addBond(prot_idx[2], lig_idx[0], [k_r_kj, r0])
    system.addForce(bond_force)

    # Angle restraints
    k_theta = 10.0  # kcal/mol/rad^2
    k_theta_kj = k_theta * 4.184  # convert to kJ/mol/rad^2

    angle_A = CustomAngleForce("0.5*k*(theta-theta0)^2")
    angle_A.addPerAngleParameter("k")
    angle_A.addPerAngleParameter("theta0")
    angle_A.addAngle(prot_idx[1], prot_idx[2], lig_idx[0], [k_theta_kj, theta_A0])
    system.addForce(angle_A)

    angle_B = CustomAngleForce("0.5*k*(theta-theta0)^2")
    angle_B.addPerAngleParameter("k")
    angle_B.addPerAngleParameter("theta0")
    angle_B.addAngle(prot_idx[2], lig_idx[0], lig_idx[1], [k_theta_kj, theta_B0])
    system.addForce(angle_B)

    # Analytical correction for Boresch restraints
    # ΔG_restraint = -kT * ln(V0 * k_r * k_theta^2 / (8 * pi^2 * r0^2 * sin(theta_A) * sin(theta_B)))
    # Simplified: use standard state correction
    kT = 0.592  # kcal/mol at 300K
    V0 = 1660.54  # A^3, standard state volume (1/1661 L converted)
    sin_A = math.sin(theta_A0)
    sin_B = math.sin(theta_B0)
    if sin_A < 0.01:
        sin_A = 0.01
    if sin_B < 0.01:
        sin_B = 0.01

    r0_A = r0 * 10  # nm to Angstrom
    dG_restraint = -kT * math.log(
        (V0 * math.sqrt(k_r * k_theta**2 * k_theta**2)) /
        (8.0 * math.pi**2 * r0_A**2 * sin_A * sin_B)
    )

    print(f"  Boresch anchors: protein {prot_idx}, ligand {lig_idx}", flush=True)
    print(f"  r0={r0_A:.2f} A, theta_A={math.degrees(theta_A0):.1f} deg, theta_B={math.degrees(theta_B0):.1f} deg", flush=True)
    print(f"  dG_restraint = {dG_restraint:.3f} kcal/mol", flush=True)

    return system, dG_restraint


def build_lambda_schedule(elec_lambdas: List[float], steric_lambdas: List[float]) -> List[Tuple[float, float]]:
    """Build deduplicated list of (elec, steric) lambda pairs."""
    pairs = []
    seen = set()
    for el in elec_lambdas:
        p = (el, 1.0)
        if p not in seen:
            seen.add(p)
            pairs.append(p)
    for st in steric_lambdas:
        p = (0.0, st)
        if p not in seen:
            seen.add(p)
            pairs.append(p)
    return pairs


# Force field presets (matches run_md_simulation.py)
FF_PRESETS = {
    'ff19sb-opc': {
        'protein_ff': 'amber/protein.ff19SB.xml',
        'water_ff': 'amber/opc_standard.xml',
        'water_model': 'opc',
    },
    'ff14sb-tip3p': {
        'protein_ff': 'amber14-all.xml',
        'water_ff': 'amber14/tip3p.xml',
        'water_model': 'tip3p',
    },
}


def build_system(pdb_path: str, force_field_preset: str, ligand_sdf: Optional[str] = None) -> Tuple[Any, Any, Any]:
    """Build OpenMM system from PDB, replicating run_md_simulation.py patterns."""
    from openmm.app import ForceField, Modeller, PME, HBonds
    from openmm.unit import nanometers
    from pdbfixer import PDBFixer

    ff_config = FF_PRESETS.get(force_field_preset, FF_PRESETS['ff19sb-opc'])

    # Load and fix PDB
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)

    ff_files = [ff_config['protein_ff'], ff_config['water_ff']]
    forcefield = ForceField(*ff_files)

    # Add ligand parameters if SDF provided
    if ligand_sdf:
        try:
            from openff.toolkit import Molecule
            from openmmforcefields.generators import SMIRNOFFTemplateGenerator
            mol = Molecule.from_file(ligand_sdf)
            smirnoff = SMIRNOFFTemplateGenerator(molecules=[mol], forcefield='openff-2.3.0')
            forcefield.registerTemplateGenerator(smirnoff.generator)
        except Exception as e:
            print(f"  Warning: Could not load ligand SDF for FF: {e}", file=sys.stderr)

    modeller = Modeller(fixer.topology, fixer.positions)

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * nanometers,
        constraints=HBonds,
    )

    return system, modeller.topology, modeller.positions


def build_solvent_system(ligand_sdf: str, force_field_preset: str) -> Tuple[Any, Any, Any, List[int]]:
    """Build a solvent-only system with the ligand in a water box."""
    from openmm.app import ForceField, Modeller, PME, HBonds
    from openmm.unit import nanometers
    from openff.toolkit import Molecule
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator

    ff_config = FF_PRESETS.get(force_field_preset, FF_PRESETS['ff19sb-opc'])

    mol = Molecule.from_file(ligand_sdf)
    smirnoff = SMIRNOFFTemplateGenerator(molecules=[mol], forcefield='openff-2.3.0')

    forcefield = ForceField(ff_config['water_ff'])
    forcefield.registerTemplateGenerator(smirnoff.generator)

    # Create topology from the ligand
    from openff.toolkit.topology import Topology as OFFTopology
    off_top = OFFTopology.from_molecules([mol])
    omm_top = off_top.to_openmm()

    # Get positions from the molecule
    conf = mol.conformers[0]
    positions = conf.to_openmm()

    modeller = Modeller(omm_top, positions)
    modeller.addSolvent(
        forcefield,
        model=ff_config['water_model'],
        padding=1.2 * nanometers,
    )

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * nanometers,
        constraints=HBonds,
    )

    # All ligand atoms (first residue)
    lig_atoms = []
    for residue in modeller.topology.residues():
        for atom in residue.atoms():
            lig_atoms.append(atom.index)
        break  # first residue is the ligand

    return system, modeller.topology, modeller.positions, lig_atoms


def run_alchemical_leg(system: Any, topology: Any, positions: Any, ligand_indices: List[int],
                       elec_lambdas: List[float], steric_lambdas: List[float], equil_ns: float, prod_ns: float,
                       snapshot_idx: int, leg_name: str, platform_name: Optional[str] = None) -> Tuple[float, float]:
    """Run all lambda windows for one leg (complex or solvent).

    Creates the alchemical system and simulation ONCE, then iterates
    over lambda windows by updating parameters on the same Context.
    """
    from openmmtools.alchemy import AbsoluteAlchemicalFactory, AlchemicalRegion, AlchemicalState
    from openmm import LangevinMiddleIntegrator
    from openmm.app import Simulation
    from openmm.unit import kelvin, picoseconds, kilojoules_per_mole

    temperature = 300.0 * kelvin
    timestep = 2.0 * picoseconds / 1000  # 2 fs
    equil_steps = int(equil_ns * 1e6 / 2)
    prod_steps = int(prod_ns * 1e6 / 2)
    collect_interval = 500  # collect every 1ps
    kT = 2.479  # kJ/mol at 300K

    # Build lambda schedule once
    all_lambda_pairs = build_lambda_schedule(elec_lambdas, steric_lambdas)
    n_windows = len(all_lambda_pairs)
    n_states = n_windows

    # Create alchemical system ONCE for the entire leg
    factory = AbsoluteAlchemicalFactory()
    alch_region = AlchemicalRegion(alchemical_atoms=ligand_indices)
    alch_system = factory.create_alchemical_system(system, alch_region)
    alch_state = AlchemicalState.from_system(alch_system)

    # Create simulation ONCE
    integrator = LangevinMiddleIntegrator(temperature, 1.0 / picoseconds, timestep)
    if platform_name:
        from openmm import Platform
        platform = Platform.getPlatformByName(platform_name)
        simulation = Simulation(topology, alch_system, integrator, platform)
    else:
        simulation = Simulation(topology, alch_system, integrator)

    all_u_kn = []

    for win_i, (elec_l, steric_l) in enumerate(all_lambda_pairs):
        if _cancelled:
            break
        pct = int(100 * win_i / n_windows)
        progress(snapshot_idx, leg_name, win_i, pct)
        print(f"  Window {win_i+1}/{n_windows}: elec={elec_l:.2f}, sterics={steric_l:.2f}", flush=True)

        # Set lambda for this window
        alch_state.lambda_electrostatics = elec_l
        alch_state.lambda_sterics = steric_l
        alch_state.apply_to_system(alch_system)

        # Reset positions and velocities for each window (independent sampling)
        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(temperature)

        # Minimize
        simulation.minimizeEnergy(maxIterations=200)

        # Equilibrate
        simulation.step(equil_steps)

        # Production: collect reduced potentials at all lambda states
        n_samples = prod_steps // collect_interval
        u_kn = np.zeros((n_states, n_samples))

        for sample_i in range(n_samples):
            if _cancelled:
                break
            simulation.step(collect_interval)

            # Evaluate energy at all lambda states
            for state_i, (el, st) in enumerate(all_lambda_pairs):
                alch_state.lambda_electrostatics = el
                alch_state.lambda_sterics = st
                alch_state.apply_to_context(simulation.context)
                s = simulation.context.getState(getEnergy=True)
                u_kn[state_i, sample_i] = s.getPotentialEnergy().value_in_unit(kilojoules_per_mole) / kT

            # Restore this window's lambdas
            alch_state.lambda_electrostatics = elec_l
            alch_state.lambda_sterics = steric_l
            alch_state.apply_to_context(simulation.context)

        all_u_kn.append(u_kn)

    progress(snapshot_idx, leg_name, n_windows, 100)

    # Combine and run MBAR
    if len(all_u_kn) == 0:
        return 0.0, float('inf')

    n_per_window = [u.shape[1] for u in all_u_kn]
    total_samples = sum(n_per_window)
    u_kn_full = np.zeros((n_states, total_samples))
    col = 0
    for u in all_u_kn:
        n = u.shape[1]
        u_kn_full[:, col:col+n] = u
        col += n

    N_k = np.array(n_per_window)

    try:
        from pymbar import MBAR
        mbar = MBAR(u_kn_full, N_k)
        results = mbar.compute_free_energy_differences()
        dG = results['Delta_f'][0, -1]
        ddG = results['dDelta_f'][0, -1]
        dG_kcal = float(dG) * 0.592
        ddG_kcal = float(ddG) * 0.592
        return dG_kcal, ddG_kcal
    except Exception as e:
        print(f"  MBAR failed ({e}), falling back to BAR...", file=sys.stderr)
        try:
            from pymbar import BAR
            total_dG = 0.0
            total_ddG2 = 0.0
            for i in range(len(all_u_kn) - 1):
                w_f = all_u_kn[i][i+1, :] - all_u_kn[i][i, :]
                w_r = all_u_kn[i+1][i, :] - all_u_kn[i+1][i+1, :]
                result = BAR(w_f, w_r)
                total_dG += result['Delta_f']
                total_ddG2 += result['dDelta_f']**2
            dG_kcal = float(total_dG) * 0.592
            ddG_kcal = float(math.sqrt(total_ddG2)) * 0.592
            return dG_kcal, ddG_kcal
        except Exception as e2:
            print(f"  BAR also failed: {e2}", file=sys.stderr)
            return 0.0, float('inf')


def detect_platform() -> str:
    """Detect best available OpenMM platform."""
    from openmm import Platform
    for name in ['CUDA', 'OpenCL', 'Metal', 'CPU']:
        try:
            Platform.getPlatformByName(name)
            return name
        except Exception:
            continue
    return 'CPU'


def main() -> None:
    parser = argparse.ArgumentParser(description='ABFE FEP scoring')
    parser.add_argument('--topology', required=True, help='PDB topology file')
    parser.add_argument('--trajectory', required=True, help='DCD trajectory file')
    parser.add_argument('--start_ns', type=float, required=True)
    parser.add_argument('--end_ns', type=float, required=True)
    parser.add_argument('--num_snapshots', type=int, default=5)
    parser.add_argument('--speed_preset', choices=['fast', 'accurate'], default='fast')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--force_field_preset', default='ff19sb-opc')
    parser.add_argument('--ligand_sdf', default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Check for checkpoint
    checkpoint_path = os.path.join(args.output_dir, 'fep_checkpoint.json')
    completed_snapshots = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
            completed_snapshots = checkpoint.get('completed', [])
        print(f"Resuming from checkpoint: {len(completed_snapshots)} snapshots complete", flush=True)

    print("Loading trajectory...", flush=True)
    import MDAnalysis as mda
    u = mda.Universe(args.topology, args.trajectory)

    total_frames = len(u.trajectory)
    timestep_ps = u.trajectory.dt  # ps per frame
    total_time_ns = total_frames * timestep_ps / 1000.0

    # Convert ns to frame indices
    start_frame = max(0, int(args.start_ns * 1000 / timestep_ps))
    end_frame = min(total_frames - 1, int(args.end_ns * 1000 / timestep_ps))

    if end_frame <= start_frame:
        print("ERROR: Invalid time range", file=sys.stderr)
        sys.exit(1)

    # Select evenly spaced snapshot frames
    snapshot_frames = np.linspace(start_frame, end_frame, args.num_snapshots, dtype=int)
    print(f"Snapshot frames: {list(snapshot_frames)} (of {total_frames} total)", flush=True)

    # Detect ligand
    from openmm.app import PDBFile
    pdb = PDBFile(args.topology)
    lig_res, ligand_indices = detect_ligand_atoms(pdb.topology)
    if ligand_indices is None:
        print("ERROR: No ligand detected in the topology. ABFE requires a protein-ligand system.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Ligand: {lig_res.name} ({len(ligand_indices)} atoms)", flush=True)

    # Lambda schedule
    elec_lambdas, steric_lambdas = get_lambda_schedule(args.speed_preset)
    equil_ns, prod_ns = get_sim_lengths(args.speed_preset)

    all_lambda_pairs = build_lambda_schedule(elec_lambdas, steric_lambdas)
    total_windows_per_leg = len(all_lambda_pairs)
    print(f"Lambda windows per leg: {total_windows_per_leg}", flush=True)
    print(f"Per-window: {equil_ns}ns equilibration + {prod_ns}ns production", flush=True)

    platform_name = detect_platform()
    print(f"Platform: {platform_name}", flush=True)

    # Build solvent system once (independent of trajectory snapshots)
    solv_system = solv_top = solv_pos = solv_lig_idx = None
    if args.ligand_sdf:
        print("Building solvent system (once)...", flush=True)
        solv_system, solv_top, solv_pos, solv_lig_idx = build_solvent_system(
            args.ligand_sdf, args.force_field_preset
        )

    results = list(completed_snapshots)

    for snap_i, frame_idx in enumerate(snapshot_frames):
        if _cancelled:
            break

        # Check if already completed
        if any(r['snapshotIndex'] == snap_i for r in results):
            print(f"\nSnapshot {snap_i+1}/{args.num_snapshots}: already complete (checkpoint)", flush=True)
            continue

        frame_time_ns = frame_idx * timestep_ps / 1000.0
        print(f"\n{'='*60}", flush=True)
        print(f"Snapshot {snap_i+1}/{args.num_snapshots} (frame {frame_idx}, t={frame_time_ns:.2f} ns)", flush=True)
        print(f"{'='*60}", flush=True)

        # Extract coordinates from trajectory frame
        u.trajectory[frame_idx]
        tmp_pdb = os.path.join(args.output_dir, f'snapshot_{snap_i}.pdb')
        protein_and_ligand = u.select_atoms('protein or (not resname HOH WAT TIP3 TP4 OPC NA CL Na+ Cl- K MG CA ZN)')
        protein_and_ligand.write(tmp_pdb)

        # === Complex leg ===
        print("\n--- Complex Leg ---", flush=True)
        try:
            complex_system, complex_top, complex_pos = build_system(
                tmp_pdb, args.force_field_preset, args.ligand_sdf
            )

            # Re-detect ligand indices in the rebuilt system
            _, complex_lig_indices = detect_ligand_atoms(complex_top)
            if complex_lig_indices is None:
                print(f"  ERROR: Ligand not found in rebuilt system for snapshot {snap_i}", file=sys.stderr)
                continue

            # Add Boresch restraints
            complex_system, dG_restraint = add_boresch_restraints(
                complex_system, complex_pos, complex_top, complex_lig_indices
            )

            dG_complex, ddG_complex = run_alchemical_leg(
                complex_system, complex_top, complex_pos, complex_lig_indices,
                elec_lambdas, steric_lambdas, equil_ns, prod_ns,
                snap_i, 'complex', platform_name=platform_name,
            )
            print(f"  ΔG_complex = {dG_complex:.3f} ± {ddG_complex:.3f} kcal/mol", flush=True)
        except Exception as e:
            print(f"  Complex leg failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            continue

        if _cancelled:
            break

        # === Solvent leg ===
        print("\n--- Solvent Leg ---", flush=True)
        try:
            if solv_system is None:
                # No ligand SDF — build from extracted PDB (fallback)
                lig_pdb = os.path.join(args.output_dir, f'ligand_{snap_i}.pdb')
                lig_atoms = u.select_atoms(f'resname {lig_res.name}')
                lig_atoms.write(lig_pdb)
                print("  Warning: No ligand SDF provided, solvent leg may fail without proper FF parameters",
                      file=sys.stderr)
                s_sys, s_top, s_pos = build_system(
                    lig_pdb, args.force_field_preset, None
                )
                _, s_lig = detect_ligand_atoms(s_top)
                if s_lig is None:
                    s_lig = list(range(len(lig_atoms)))
                solv_system, solv_top, solv_pos, solv_lig_idx = s_sys, s_top, s_pos, s_lig

            dG_solvent, ddG_solvent = run_alchemical_leg(
                solv_system, solv_top, solv_pos, solv_lig_idx,
                elec_lambdas, steric_lambdas, equil_ns, prod_ns,
                snap_i, 'solvent', platform_name=platform_name,
            )
            print(f"  ΔG_solvent = {dG_solvent:.3f} ± {ddG_solvent:.3f} kcal/mol", flush=True)
        except Exception as e:
            print(f"  Solvent leg failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            continue

        # Compute binding free energy
        dG_bind = dG_complex - dG_solvent - dG_restraint
        uncertainty = math.sqrt(ddG_complex**2 + ddG_solvent**2)

        snap_result = {
            'snapshotIndex': snap_i,
            'frameIndex': int(frame_idx),
            'timeNs': round(frame_time_ns, 3),
            'deltaG_complex': round(dG_complex, 3),
            'deltaG_solvent': round(dG_solvent, 3),
            'deltaG_bind': round(dG_bind, 3),
            'uncertainty': round(uncertainty, 3),
        }
        results.append(snap_result)

        print(f"\n  ΔG_bind = {dG_bind:.3f} ± {uncertainty:.3f} kcal/mol", flush=True)
        print(f"FEP_RESULT:{json.dumps(snap_result)}", flush=True)

        # Save checkpoint
        with open(checkpoint_path, 'w') as f:
            json.dump({'completed': results}, f, indent=2)

    # Compute final statistics
    if len(results) > 0:
        dG_values = [r['deltaG_bind'] for r in results]
        mean_dG = float(np.mean(dG_values))
        sem = float(np.std(dG_values, ddof=1) / math.sqrt(len(dG_values))) if len(dG_values) > 1 else results[0]['uncertainty']
    else:
        mean_dG = 0.0
        sem = float('inf')

    final_result = {
        'snapshots': results,
        'meanDeltaG': round(mean_dG, 3),
        'sem': round(sem, 3),
        'outputDir': args.output_dir,
    }

    result_path = os.path.join(args.output_dir, 'fep_results.json')
    with open(result_path, 'w') as f:
        json.dump(final_result, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"ABFE Complete: ΔG_bind = {mean_dG:.3f} ± {sem:.3f} kcal/mol", flush=True)
    print(f"Results saved to {result_path}", flush=True)
    print(f"{'='*60}", flush=True)

    if _cancelled:
        sys.exit(143)


if __name__ == '__main__':
    main()
