#!/usr/bin/env python3
"""
Compute 3D interaction potential grids around a bound ligand.

Generates hydrophobic, H-bond donor, and H-bond acceptor grids as OpenDX files
and identifies expansion vector hotspots for lead optimization.

Supports two scoring modes:
  - energy: LJ + Coulombic via OpenMM force field (default, requires PDBFixer)
  - distance: Distance-based heuristic scoring (fallback)
"""

import argparse
import json
import os
import sys
import warnings
from typing import Any, List, Optional, Tuple

warnings.filterwarnings('ignore')

from utils import write_dx, find_hotspots, normalize_grid


# Atom classification helpers (used by distance-based scoring)
NONPOLAR_ELEMENTS = {'C', 'S'}
BACKBONE_ATOMS = {'C', 'O', 'N', 'CA'}
HBOND_ACCEPTOR_ELEMENTS = {'O', 'N', 'S'}
HBOND_DONOR_RESIDUES_NH = {
    'ARG', 'ASN', 'GLN', 'HIS', 'LYS', 'SER', 'THR', 'TRP', 'TYR', 'CYS'
}


def is_nonpolar(atom: Any) -> bool:
    """Check if atom is non-polar (C or S, excluding backbone C=O)."""
    elem = atom.element.strip().upper()
    if elem not in NONPOLAR_ELEMENTS:
        return False
    if elem == 'C' and atom.name in ('C', 'O'):
        return False
    return True


def is_hbond_acceptor(atom: Any) -> bool:
    """Check if atom can accept H-bonds (O, N with lone pairs)."""
    elem = atom.element.strip().upper()
    return elem in HBOND_ACCEPTOR_ELEMENTS


def is_hbond_donor(atom: Any) -> bool:
    """Check if atom can donate H-bonds (N-H groups)."""
    elem = atom.element.strip().upper()
    if elem != 'N':
        return False
    if atom.name == 'N':
        return True
    resname = atom.get_parent().get_resname().strip()
    if resname in HBOND_DONOR_RESIDUES_NH:
        return True
    return False


def score_distance_based(
    free_points: Any, free_indices: Any,
    protein_coords: Any, protein_props: Any,
    grid_size: int
) -> Tuple[Any, Any, Any]:
    """Distance-based interaction scoring (original heuristic method)."""
    import numpy as np
    from scipy.spatial import cKDTree

    hydrophobic = np.zeros(grid_size)
    hbond_donor = np.zeros(grid_size)
    hbond_acceptor = np.zeros(grid_size)

    if len(free_points) == 0:
        return hydrophobic, hbond_donor, hbond_acceptor

    # Hydrophobic: sum 1/d^2 for nonpolar atoms in 3.5-5.0 A range
    # Single query at 5.0A, filter by distance (avoids double query_ball_point)
    nonpolar_mask = protein_props[:, 0].astype(bool)
    if np.any(nonpolar_mask):
        nonpolar_coords = protein_coords[nonpolar_mask]
        nonpolar_tree = cKDTree(nonpolar_coords)
        neighbors_5 = nonpolar_tree.query_ball_point(free_points, 5.0)
        for i, nbrs in enumerate(neighbors_5):
            if nbrs:
                dists = np.linalg.norm(nonpolar_coords[nbrs] - free_points[i], axis=1)
                in_range = dists >= 3.5  # already know dists < 5.0
                if np.any(in_range):
                    hydrophobic[free_indices[i]] = np.sum(1.0 / (dists[in_range] ** 2))

    # H-bond donor: protein acceptors within 3.5 A
    acceptor_mask = protein_props[:, 1].astype(bool)
    if np.any(acceptor_mask):
        acceptor_coords = protein_coords[acceptor_mask]
        acceptor_tree = cKDTree(acceptor_coords)
        neighbors = acceptor_tree.query_ball_point(free_points, 3.5)
        for i, nbrs in enumerate(neighbors):
            if nbrs:
                dists = np.linalg.norm(acceptor_coords[nbrs] - free_points[i], axis=1)
                hbond_donor[free_indices[i]] = np.sum((3.5 - dists) / 3.5)

    # H-bond acceptor: protein donors within 3.5 A
    donor_mask = protein_props[:, 2].astype(bool)
    if np.any(donor_mask):
        donor_coords = protein_coords[donor_mask]
        donor_tree = cKDTree(donor_coords)
        neighbors = donor_tree.query_ball_point(free_points, 3.5)
        for i, nbrs in enumerate(neighbors):
            if nbrs:
                dists = np.linalg.norm(donor_coords[nbrs] - free_points[i], axis=1)
                hbond_acceptor[free_indices[i]] = np.sum((3.5 - dists) / 3.5)

    return hydrophobic, hbond_donor, hbond_acceptor


def score_energy_based(
    pdb_path: str, free_points: Any, free_indices: Any, grid_size: int
) -> Optional[Tuple[Any, Any, Any]]:
    """Energy-based interaction scoring using OpenMM force field parameters.

    Vectorized via sparse_distance_matrix — no per-grid-point Python loop.
    Returns (hydrophobic, hbond_donor, hbond_acceptor) or None on failure.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    try:
        from pdbfixer import PDBFixer
        from openmm.app import ForceField, NoCutoff
        from openmm import NonbondedForce
        from openmm import unit as u
    except ImportError as e:
        print(f"PROGRESS: PDBFixer/OpenMM not available: {e}")
        return None

    print("PROGRESS: Extracting force field parameters (PDBFixer + AMBER ff14SB)...")

    try:
        fixer = PDBFixer(filename=pdb_path)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.4)

        ff = ForceField('amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml')
        system = ff.createSystem(fixer.topology, nonbondedMethod=NoCutoff)
    except Exception as e:
        print(f"PROGRESS: Force field setup failed: {e}")
        return None

    prot_coords = np.array(
        [[p.x, p.y, p.z] for p in fixer.positions]
    ) * 10.0  # nm -> A

    n_prot = len(prot_coords)
    charges = np.zeros(n_prot)
    sigmas = np.zeros(n_prot)
    epsilons = np.zeros(n_prot)

    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            for i in range(force.getNumParticles()):
                q, sig, eps = force.getParticleParameters(i)
                charges[i] = q.value_in_unit(u.elementary_charge)
                sigmas[i] = sig.value_in_unit(u.nanometer) * 10
                epsilons[i] = eps.value_in_unit(u.kilojoule_per_mole)
            break

    print(f"PROGRESS: Protein atoms (with H): {n_prot}")

    elements = np.array([
        atom.element.symbol if atom.element else 'X'
        for atom in fixer.topology.atoms()
    ])
    is_nonpolar_arr = (np.abs(charges) < 0.3) & (elements != 'H')

    # Carbon-like probe for LJ combining rules (OPLS-AA sp3 carbon)
    PROBE_SIGMA = 3.4
    PROBE_EPSILON = 0.36

    cutoff_lj = 6.0
    cutoff_elec = 10.0

    print("PROGRESS: Computing energy-based interaction potentials (vectorized)...")

    # Build sparse distance matrix — one C-level call replaces the Python loop
    free_tree = cKDTree(free_points)
    prot_tree = cKDTree(prot_coords)
    sdm = free_tree.sparse_distance_matrix(prot_tree, cutoff_elec, output_type='coo_matrix')
    row = sdm.row       # free-point index
    col = sdm.col       # protein-atom index
    dist = np.maximum(sdm.data, 0.5)

    # --- Electrostatic: Coulomb with e=4r, scatter-add ---
    phi_contrib = charges[col] / (4.0 * dist)
    phi = np.zeros(len(free_points))
    np.add.at(phi, row, phi_contrib)

    hbond_donor_fp = np.zeros(len(free_points))
    hbond_acceptor_fp = np.zeros(len(free_points))
    neg = phi < 0
    pos = phi > 0
    hbond_donor_fp[neg] = -phi[neg]
    hbond_acceptor_fp[pos] = phi[pos]

    # --- Hydrophobic: LJ from nonpolar atoms within cutoff_lj ---
    hydrophobic_fp = np.zeros(len(free_points))
    lj_mask = is_nonpolar_arr[col] & (dist < cutoff_lj)
    if np.any(lj_mask):
        lj_row = row[lj_mask]
        lj_col = col[lj_mask]
        lj_dist = dist[lj_mask]

        sig_c = (PROBE_SIGMA + sigmas[lj_col]) / 2
        eps_c = np.sqrt(PROBE_EPSILON * np.maximum(epsilons[lj_col], 0))

        valid = eps_c > 0
        if np.any(valid):
            ratio = sig_c[valid] / lj_dist[valid]
            lj_energy = 4.0 * eps_c[valid] * (ratio**12 - ratio**6)
            attractive = lj_energy < 0
            if np.any(attractive):
                np.add.at(hydrophobic_fp, lj_row[valid][attractive], -lj_energy[attractive])

    # Scatter back into full grid
    hydrophobic = np.zeros(grid_size)
    hbond_donor = np.zeros(grid_size)
    hbond_acceptor = np.zeros(grid_size)
    hydrophobic[free_indices] = hydrophobic_fp
    hbond_donor[free_indices] = hbond_donor_fp
    hbond_acceptor[free_indices] = hbond_acceptor_fp

    print("PROGRESS: Scoring grid points... 100%")
    return hydrophobic, hbond_donor, hbond_acceptor


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute binding site interaction maps')
    parser.add_argument('--pdb_path', required=True, help='PDB file path')
    parser.add_argument('--ligand_resname', required=True, help='Ligand residue name')
    parser.add_argument('--ligand_resnum', required=True, type=int, help='Ligand residue number')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--box_padding', type=float, default=8.0, help='Box padding in Angstroms')
    parser.add_argument('--grid_spacing', type=float, default=0.75, help='Grid spacing in Angstroms')
    parser.add_argument('--project_name', default=None, help='Project name prefix for output files')
    parser.add_argument('--scoring', choices=['energy', 'distance'], default='energy',
                        help='Scoring method: energy (LJ+Coulomb, default) or distance (heuristic)')
    args = parser.parse_args()

    try:
        import numpy as np
        from scipy.spatial import cKDTree
    except ImportError as e:
        print(f"Error: Missing required package: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"PROGRESS: Loading PDB: {args.pdb_path}")

    if args.pdb_path.lower().endswith('.cif'):
        from Bio.PDB import MMCIFParser
        parser_pdb = MMCIFParser(QUIET=True)
    else:
        from Bio.PDB import PDBParser
        parser_pdb = PDBParser(QUIET=True)
    structure = parser_pdb.get_structure('complex', args.pdb_path)
    model = structure[0]

    protein_coords = []
    protein_props = []
    ligand_coords = []

    for chain in model:
        for residue in chain:
            resname = residue.get_resname().strip()
            resnum = residue.get_id()[1]

            if resname in ('HOH', 'WAT', 'TIP3', 'TIP4', 'NA', 'CL', 'SOL', 'K', 'MG', 'CA', 'ZN'):
                continue

            is_ligand = (resname == args.ligand_resname and resnum == args.ligand_resnum)

            for atom in residue:
                coord = atom.get_vector().get_array()
                if is_ligand:
                    ligand_coords.append(coord)
                else:
                    protein_coords.append(coord)
                    protein_props.append((
                        is_nonpolar(atom),
                        is_hbond_acceptor(atom),
                        is_hbond_donor(atom),
                    ))

    if len(ligand_coords) == 0:
        print(f"Error: Ligand {args.ligand_resname} {args.ligand_resnum} not found", file=sys.stderr)
        sys.exit(1)

    protein_coords = np.array(protein_coords)
    ligand_coords = np.array(ligand_coords)
    protein_props = np.array(protein_props)

    print(f"PROGRESS: Protein atoms: {len(protein_coords)}, Ligand atoms: {len(ligand_coords)}")

    ligand_com = ligand_coords.mean(axis=0)

    padding = args.box_padding
    spacing = args.grid_spacing
    grid_min = ligand_com - padding
    grid_max = ligand_com + padding

    nx = int(np.ceil((grid_max[0] - grid_min[0]) / spacing)) + 1
    ny = int(np.ceil((grid_max[1] - grid_min[1]) / spacing)) + 1
    nz = int(np.ceil((grid_max[2] - grid_min[2]) / spacing)) + 1

    print(f"PROGRESS: Grid dimensions: {nx} x {ny} x {nz} = {nx * ny * nz} points")

    x = np.linspace(grid_min[0], grid_min[0] + (nx - 1) * spacing, nx)
    y = np.linspace(grid_min[1], grid_min[1] + (ny - 1) * spacing, ny)
    z = np.linspace(grid_min[2], grid_min[2] + (nz - 1) * spacing, nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    print("PROGRESS: Filtering occupied grid points...")
    vdw_cutoff = 1.4
    all_atom_coords = np.vstack([protein_coords, ligand_coords])
    all_atom_tree = cKDTree(all_atom_coords)
    nearest_dist, _ = all_atom_tree.query(grid_points)
    free_mask = nearest_dist > vdw_cutoff

    n_free = np.sum(free_mask)
    print(f"PROGRESS: Free grid points: {n_free} / {len(grid_points)}")

    free_indices = np.where(free_mask)[0]
    free_points = grid_points[free_indices]

    print(f"PROGRESS: Scoring mode: {args.scoring}")

    result = None
    if args.scoring == 'energy':
        result = score_energy_based(args.pdb_path, free_points, free_indices, len(grid_points))
        if result is None:
            print("PROGRESS: Energy scoring failed, falling back to distance-based")

    if result is not None:
        hydrophobic, hbond_donor, hbond_acceptor = result
    else:
        hydrophobic, hbond_donor, hbond_acceptor = score_distance_based(
            free_points, free_indices, protein_coords, protein_props, len(grid_points)
        )

    hydrophobic = normalize_grid(hydrophobic)
    hbond_donor = normalize_grid(hbond_donor)
    hbond_acceptor = normalize_grid(hbond_acceptor)

    shape = (nx, ny, nz)
    hydro_3d = hydrophobic.reshape(shape)
    donor_3d = hbond_donor.reshape(shape)
    acceptor_3d = hbond_acceptor.reshape(shape)

    os.makedirs(args.output_dir, exist_ok=True)
    origin = grid_min.tolist()

    prefix = f'{args.project_name}_' if args.project_name else ''
    hydro_path = os.path.join(args.output_dir, f'{prefix}hydrophobic.dx')
    donor_path = os.path.join(args.output_dir, f'{prefix}hbond_donor.dx')
    acceptor_path = os.path.join(args.output_dir, f'{prefix}hbond_acceptor.dx')

    print("PROGRESS: Writing DX files...")
    write_dx(hydro_path, hydro_3d, origin, spacing, shape)
    write_dx(donor_path, donor_3d, origin, spacing, shape)
    write_dx(acceptor_path, acceptor_3d, origin, spacing, shape)

    print("PROGRESS: Identifying hotspots...")
    hotspots = []
    hotspots.extend(find_hotspots(hydro_3d, 'hydrophobic', origin, spacing, ligand_com))
    hotspots.extend(find_hotspots(donor_3d, 'hbond_donor', origin, spacing, ligand_com))
    hotspots.extend(find_hotspots(acceptor_3d, 'hbond_acceptor', origin, spacing, ligand_com))

    results = {
        'hydrophobicDx': hydro_path,
        'hbondDonorDx': donor_path,
        'hbondAcceptorDx': acceptor_path,
        'hotspots': hotspots,
        'gridDimensions': [nx, ny, nz],
        'ligandCom': [round(float(c), 3) for c in ligand_com],
    }

    results_path = os.path.join(args.output_dir, f'{prefix}binding_site_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"PROGRESS: Results written to {results_path}")
    print(f"PROGRESS: Hotspots found: {len(hotspots)}")
    print("Done!")


if __name__ == '__main__':
    main()
