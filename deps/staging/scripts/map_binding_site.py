#!/usr/bin/env python3
"""
Compute 3D interaction potential grids around a bound ligand.

Generates hydrophobic, H-bond donor, and H-bond acceptor grids as OpenDX files
and identifies expansion vector hotspots for lead optimization.
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')


# Atom classification helpers
NONPOLAR_ELEMENTS = {'C', 'S'}
BACKBONE_ATOMS = {'C', 'O', 'N', 'CA'}
HBOND_ACCEPTOR_ELEMENTS = {'O', 'N', 'S'}
HBOND_DONOR_RESIDUES_NH = {
    'ARG', 'ASN', 'GLN', 'HIS', 'LYS', 'SER', 'THR', 'TRP', 'TYR', 'CYS'
}


def is_nonpolar(atom):
    """Check if atom is non-polar (C or S, excluding backbone C=O)."""
    elem = atom.element.strip().upper()
    if elem not in NONPOLAR_ELEMENTS:
        return False
    # Exclude backbone carbonyl carbon
    if elem == 'C' and atom.name in ('C', 'O'):
        return False
    return True


def is_hbond_acceptor(atom):
    """Check if atom can accept H-bonds (O, N with lone pairs)."""
    elem = atom.element.strip().upper()
    return elem in HBOND_ACCEPTOR_ELEMENTS


def is_hbond_donor(atom):
    """Check if atom can donate H-bonds (N-H groups)."""
    elem = atom.element.strip().upper()
    if elem != 'N':
        return False
    # Backbone NH
    if atom.name == 'N':
        return True
    # Sidechain NH groups
    resname = atom.get_parent().get_resname().strip()
    if resname in HBOND_DONOR_RESIDUES_NH:
        return True
    return False


def write_dx(filepath, data, origin, spacing, shape):
    """Write a 3D grid in OpenDX format."""
    import numpy as np
    nx, ny, nz = shape
    flat = data.flatten(order='C')
    # Batch format: reshape into rows of 3, write with numpy
    n_full_rows = len(flat) // 3
    remainder = len(flat) % 3
    with open(filepath, 'w') as f:
        f.write(f'object 1 class gridpositions counts {nx} {ny} {nz}\n')
        f.write(f'origin {origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}\n')
        f.write(f'delta {spacing:.6f} 0.000000 0.000000\n')
        f.write(f'delta 0.000000 {spacing:.6f} 0.000000\n')
        f.write(f'delta 0.000000 0.000000 {spacing:.6f}\n')
        f.write(f'object 2 class gridconnections counts {nx} {ny} {nz}\n')
        f.write(f'object 3 class array type double rank 0 items {len(flat)} data follows\n')
        if n_full_rows > 0:
            rows = flat[:n_full_rows * 3].reshape(-1, 3)
            lines = '\n'.join(f'{r[0]:.6f} {r[1]:.6f} {r[2]:.6f}' for r in rows)
            f.write(lines)
            f.write('\n')
        if remainder > 0:
            f.write(' '.join(f'{v:.6f}' for v in flat[n_full_rows * 3:]))
            f.write('\n')
        f.write('attribute "dep" string "positions"\n')
        f.write('object "regular positions regular connections" class field\n')
        f.write('component "positions" value 1\n')
        f.write('component "connections" value 2\n')
        f.write('component "data" value 3\n')


def main():
    parser = argparse.ArgumentParser(description='Compute binding site interaction maps')
    parser.add_argument('--pdb_path', required=True, help='PDB file path')
    parser.add_argument('--ligand_resname', required=True, help='Ligand residue name')
    parser.add_argument('--ligand_resnum', required=True, type=int, help='Ligand residue number')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--box_padding', type=float, default=8.0, help='Box padding in Angstroms')
    parser.add_argument('--grid_spacing', type=float, default=0.75, help='Grid spacing in Angstroms')
    args = parser.parse_args()

    try:
        import numpy as np
        from scipy.spatial import cKDTree
        from scipy import ndimage
        from Bio.PDB import PDBParser
    except ImportError as e:
        print(f"Error: Missing required package: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"PROGRESS: Loading PDB: {args.pdb_path}")

    if args.pdb_path.lower().endswith('.cif'):
        from Bio.PDB import MMCIFParser
        parser_pdb = MMCIFParser(QUIET=True)
    else:
        parser_pdb = PDBParser(QUIET=True)
    structure = parser_pdb.get_structure('complex', args.pdb_path)
    model = structure[0]

    # Separate protein and ligand atoms
    protein_coords = []
    protein_props = []  # (is_nonpolar, is_acceptor, is_donor)
    ligand_coords = []

    for chain in model:
        for residue in chain:
            resname = residue.get_resname().strip()
            resnum = residue.get_id()[1]

            # Skip water and ions
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

    # Compute ligand center of mass
    ligand_com = ligand_coords.mean(axis=0)

    # Build grid
    padding = args.box_padding
    spacing = args.grid_spacing
    grid_min = ligand_com - padding
    grid_max = ligand_com + padding

    nx = int(np.ceil((grid_max[0] - grid_min[0]) / spacing)) + 1
    ny = int(np.ceil((grid_max[1] - grid_min[1]) / spacing)) + 1
    nz = int(np.ceil((grid_max[2] - grid_min[2]) / spacing)) + 1

    print(f"PROGRESS: Grid dimensions: {nx} x {ny} x {nz} = {nx * ny * nz} points")

    # Generate grid points
    x = np.linspace(grid_min[0], grid_min[0] + (nx - 1) * spacing, nx)
    y = np.linspace(grid_min[1], grid_min[1] + (ny - 1) * spacing, ny)
    z = np.linspace(grid_min[2], grid_min[2] + (nz - 1) * spacing, nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # Build KD-trees
    protein_tree = cKDTree(protein_coords)

    # Exclude grid points inside protein or ligand vdW radii (1.4 A)
    # Single combined tree avoids two separate queries
    print("PROGRESS: Filtering occupied grid points...")
    vdw_cutoff = 1.4
    all_atom_coords = np.vstack([protein_coords, ligand_coords])
    all_atom_tree = cKDTree(all_atom_coords)
    nearest_dist, _ = all_atom_tree.query(grid_points)
    free_mask = nearest_dist > vdw_cutoff

    n_free = np.sum(free_mask)
    print(f"PROGRESS: Free grid points: {n_free} / {len(grid_points)}")

    # Initialize grids
    hydrophobic = np.zeros(len(grid_points))
    hbond_donor = np.zeros(len(grid_points))
    hbond_acceptor = np.zeros(len(grid_points))

    # Score free grid points
    print("PROGRESS: Scoring interaction potentials...")
    free_indices = np.where(free_mask)[0]
    free_points = grid_points[free_indices]

    if len(free_points) > 0:
        # Hydrophobic: sum 1/d^2 for nonpolar atoms within 3.5-5.0 A
        nonpolar_mask = protein_props[:, 0].astype(bool)
        if np.any(nonpolar_mask):
            nonpolar_coords = protein_coords[nonpolar_mask]
            nonpolar_tree = cKDTree(nonpolar_coords)
            neighbors_5 = nonpolar_tree.query_ball_point(free_points, 5.0)
            neighbors_35 = nonpolar_tree.query_ball_point(free_points, 3.5)
            for i, (n5, n35) in enumerate(zip(neighbors_5, neighbors_35)):
                # Atoms in 3.5-5.0 range
                in_range = set(n5) - set(n35)
                if in_range:
                    dists = np.linalg.norm(nonpolar_coords[list(in_range)] - free_points[i], axis=1)
                    hydrophobic[free_indices[i]] = np.sum(1.0 / (dists ** 2))

        # H-bond donor: protein acceptors within 3.5 A, score (3.5-d)/3.5
        acceptor_mask = protein_props[:, 1].astype(bool)
        if np.any(acceptor_mask):
            acceptor_coords = protein_coords[acceptor_mask]
            acceptor_tree = cKDTree(acceptor_coords)
            neighbors = acceptor_tree.query_ball_point(free_points, 3.5)
            for i, nbrs in enumerate(neighbors):
                if nbrs:
                    dists = np.linalg.norm(acceptor_coords[nbrs] - free_points[i], axis=1)
                    hbond_donor[free_indices[i]] = np.sum((3.5 - dists) / 3.5)

        # H-bond acceptor: protein donors within 3.5 A, score (3.5-d)/3.5
        donor_mask = protein_props[:, 2].astype(bool)
        if np.any(donor_mask):
            donor_coords = protein_coords[donor_mask]
            donor_tree = cKDTree(donor_coords)
            neighbors = donor_tree.query_ball_point(free_points, 3.5)
            for i, nbrs in enumerate(neighbors):
                if nbrs:
                    dists = np.linalg.norm(donor_coords[nbrs] - free_points[i], axis=1)
                    hbond_acceptor[free_indices[i]] = np.sum((3.5 - dists) / 3.5)

    # Normalize to [0, 1]
    def normalize(arr):
        vmax = arr.max()
        if vmax > 0:
            return arr / vmax
        return arr

    hydrophobic = normalize(hydrophobic)
    hbond_donor = normalize(hbond_donor)
    hbond_acceptor = normalize(hbond_acceptor)

    # Reshape to 3D
    shape = (nx, ny, nz)
    hydro_3d = hydrophobic.reshape(shape)
    donor_3d = hbond_donor.reshape(shape)
    acceptor_3d = hbond_acceptor.reshape(shape)

    # Write DX files
    os.makedirs(args.output_dir, exist_ok=True)
    origin = grid_min.tolist()

    hydro_path = os.path.join(args.output_dir, 'hydrophobic.dx')
    donor_path = os.path.join(args.output_dir, 'hbond_donor.dx')
    acceptor_path = os.path.join(args.output_dir, 'hbond_acceptor.dx')

    print("PROGRESS: Writing DX files...")
    write_dx(hydro_path, hydro_3d, origin, spacing, shape)
    write_dx(donor_path, donor_3d, origin, spacing, shape)
    write_dx(acceptor_path, acceptor_3d, origin, spacing, shape)

    # Find hotspots: threshold at 70th percentile, cluster, extract centroids
    print("PROGRESS: Identifying hotspots...")
    hotspots = []

    def find_hotspots(grid_3d, channel_name, grid_origin, grid_spacing):
        nonzero = grid_3d[grid_3d > 0]
        if len(nonzero) == 0:
            return []
        threshold = np.percentile(nonzero, 70)
        binary = grid_3d > threshold
        labeled, num_features = ndimage.label(binary)
        if num_features == 0:
            return []

        results = []
        for label_id in range(1, num_features + 1):
            cluster_mask = labeled == label_id
            cluster_scores = grid_3d[cluster_mask]
            score = float(cluster_scores.mean())

            # Centroid in grid indices
            centroid_idx = ndimage.center_of_mass(grid_3d, labeled, label_id)
            # Convert to real coordinates
            pos = [
                grid_origin[0] + centroid_idx[0] * grid_spacing,
                grid_origin[1] + centroid_idx[1] * grid_spacing,
                grid_origin[2] + centroid_idx[2] * grid_spacing,
            ]
            # Direction vector from ligand COM
            direction = [pos[0] - ligand_com[0], pos[1] - ligand_com[1], pos[2] - ligand_com[2]]
            mag = np.sqrt(sum(d ** 2 for d in direction))
            if mag > 0:
                direction = [d / mag for d in direction]
            results.append({
                'type': channel_name,
                'position': [round(p, 3) for p in pos],
                'direction': [round(d, 3) for d in direction],
                'score': round(score, 4),
            })

        # Sort by score descending, keep top 5
        results.sort(key=lambda h: h['score'], reverse=True)
        return results[:5]

    hotspots.extend(find_hotspots(hydro_3d, 'hydrophobic', origin, spacing))
    hotspots.extend(find_hotspots(donor_3d, 'hbond_donor', origin, spacing))
    hotspots.extend(find_hotspots(acceptor_3d, 'hbond_acceptor', origin, spacing))

    # Write results JSON
    results = {
        'hydrophobicDx': hydro_path,
        'hbondDonorDx': donor_path,
        'hbondAcceptorDx': acceptor_path,
        'hotspots': hotspots,
        'gridDimensions': [nx, ny, nz],
        'ligandCom': [round(float(c), 3) for c in ligand_com],
    }

    results_path = os.path.join(args.output_dir, 'binding_site_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"PROGRESS: Results written to {results_path}")
    print(f"PROGRESS: Hotspots found: {len(hotspots)}")
    print("Done!")


if __name__ == '__main__':
    main()
