#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
GIST (Grid Inhomogeneous Solvation Theory) water thermodynamics analysis.

Analyzes water behavior from an MD trajectory to identify thermodynamically
unfavorable hydration sites. Maps water density and energy to hydrophobic,
H-bond donor, and H-bond acceptor channels.

Primary method: cpptraj GIST (from AmberTools).
Fallback: MDAnalysis-based water density analysis.

Output: 3 DX files + binding_site_results.json (same format as map_binding_site.py).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

from utils import write_dx, read_dx, find_hotspots, find_ligand_com, normalize_grid


def run_cpptraj_gist(pdb_path: str, traj_path: str, center: Any,
                     nx: int, ny: int, nz: int, spacing: float,
                     output_dir: str) -> Optional[Dict[str, Any]]:
    """Run cpptraj GIST analysis. Returns dict of grid data or None."""
    import numpy as np

    cpptraj = shutil.which('cpptraj')
    if not cpptraj:
        print("PROGRESS: cpptraj not found, using density fallback")
        return None

    print("PROGRESS: Running cpptraj GIST analysis...")

    gist_prefix = os.path.join(output_dir, 'gist')
    input_file = os.path.join(output_dir, 'gist_input.in')

    with open(input_file, 'w') as f:
        f.write(f'parm {pdb_path}\n')
        f.write(f'trajin {traj_path}\n')
        f.write(f'autoimage\n')
        f.write(f'gist gridcntr {center[0]:.3f} {center[1]:.3f} {center[2]:.3f} ')
        f.write(f'griddim {nx} {ny} {nz} gridspacn {spacing:.3f} ')
        f.write(f'out {gist_prefix}\n')
        f.write('run\n')

    try:
        result = subprocess.run(
            [cpptraj, '-i', input_file],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            print(f"PROGRESS: cpptraj failed: {result.stderr[:200]}")
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"PROGRESS: cpptraj execution error: {e}")
        return None

    # Parse GIST output DX files
    gist_data = {}
    expected_files = {
        'gO': f'{gist_prefix}-gO.dx',
        'Esw': f'{gist_prefix}-Esw-norm.dx',
        'Eww': f'{gist_prefix}-Eww-norm.dx',
        'dTStrans': f'{gist_prefix}-dTStrans-norm.dx',
        'dTSorient': f'{gist_prefix}-dTSorient-norm.dx',
    }

    for key, fpath in expected_files.items():
        if os.path.exists(fpath):
            data, _origin, _sp, shape = read_dx(fpath)
            gist_data[key] = data
            print(f"PROGRESS: Loaded GIST {key}: {shape}")
        else:
            alt = fpath.replace('-norm', '-dens')
            if os.path.exists(alt):
                data, _origin, _sp, shape = read_dx(alt)
                gist_data[key] = data
                print(f"PROGRESS: Loaded GIST {key} (alt): {shape}")

    if 'gO' not in gist_data:
        print("PROGRESS: GIST output files not found")
        return None

    return gist_data


def compute_water_density(pdb_path: str, traj_path: str, center: Any,
                          nx: int, ny: int, nz: int, spacing: float) -> Optional[Any]:
    """Fallback: compute 3D water oxygen density from MDAnalysis."""
    import numpy as np

    try:
        import MDAnalysis as mda
    except ImportError:
        print("PROGRESS: MDAnalysis not available for density fallback")
        return None

    print("PROGRESS: Computing water density with MDAnalysis...")

    u = mda.Universe(pdb_path, traj_path)
    waters = u.select_atoms('name OW or (resname WAT HOH TIP3 and name O)')

    if len(waters) == 0:
        print("PROGRESS: No water oxygens found")
        return None

    origin = center - np.array([nx, ny, nz]) * spacing / 2.0
    density = np.zeros((nx, ny, nz))

    n_frames = len(u.trajectory)
    report_interval = max(1, n_frames // 10)

    for frame_idx, ts in enumerate(u.trajectory):
        if frame_idx % report_interval == 0:
            pct = int(100 * frame_idx / n_frames)
            print(f"PROGRESS: Binning water positions... {pct}%")

        positions = waters.positions
        indices = ((positions - origin) / spacing).astype(int)
        valid = (
            (indices[:, 0] >= 0) & (indices[:, 0] < nx) &
            (indices[:, 1] >= 0) & (indices[:, 1] < ny) &
            (indices[:, 2] >= 0) & (indices[:, 2] < nz)
        )
        valid_idx = indices[valid]
        for idx in valid_idx:
            density[idx[0], idx[1], idx[2]] += 1

    print("PROGRESS: Binning water positions... 100%")

    voxel_vol = spacing ** 3
    density /= (n_frames * voxel_vol)
    return density


def get_protein_electrostatic_grid(pdb_path: str, grid_points: Any,
                                   shape: Tuple[int, int, int]) -> Any:
    """Compute electrostatic potential from protein at each grid point."""
    import numpy as np
    from scipy.spatial import cKDTree

    try:
        from pdbfixer import PDBFixer
        from openmm.app import ForceField, NoCutoff
        from openmm import NonbondedForce
        from openmm import unit as u
    except ImportError:
        return np.zeros(len(grid_points))

    print("PROGRESS: Computing protein electrostatic potential for decomposition...")

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
        print(f"PROGRESS: Electrostatic decomposition failed: {e}")
        return np.zeros(len(grid_points))

    prot_coords = np.array([[p.x, p.y, p.z] for p in fixer.positions]) * 10.0
    n_prot = len(prot_coords)
    charges = np.zeros(n_prot)

    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            for i in range(force.getNumParticles()):
                q, _, _ = force.getParticleParameters(i)
                charges[i] = q.value_in_unit(u.elementary_charge)
            break

    tree = cKDTree(prot_coords)
    phi = np.zeros(len(grid_points))
    cutoff = 10.0

    neighbors = tree.query_ball_point(grid_points, cutoff)
    for i, nbrs in enumerate(neighbors):
        if nbrs:
            nbrs_arr = np.array(nbrs)
            dists = np.linalg.norm(prot_coords[nbrs_arr] - grid_points[i], axis=1)
            dists = np.maximum(dists, 0.5)
            phi[i] = np.sum(charges[nbrs_arr] / (4.0 * dists))

    return phi


def main() -> None:
    parser = argparse.ArgumentParser(description='GIST water thermodynamics analysis')
    parser.add_argument('--pdb_path', required=True, help='System PDB (with ligand)')
    parser.add_argument('--trajectory_path', required=True, help='DCD trajectory')
    parser.add_argument('--ligand_resname', required=True, help='Ligand residue name')
    parser.add_argument('--ligand_resnum', required=True, type=int, help='Ligand residue number')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--box_padding', type=float, default=8.0, help='Box padding in Angstroms')
    parser.add_argument('--grid_spacing', type=float, default=0.75, help='Grid spacing in Angstroms')
    parser.add_argument('--project_name', default=None, help='Project name prefix')
    args = parser.parse_args()

    import numpy as np

    print(f"PROGRESS: Loading PDB: {args.pdb_path}")
    print(f"PROGRESS: Trajectory: {args.trajectory_path}")

    ligand_com = find_ligand_com(args.pdb_path, args.ligand_resname, args.ligand_resnum)
    print(f"PROGRESS: Ligand COM: {ligand_com}")

    padding = args.box_padding
    spacing = args.grid_spacing
    grid_min = ligand_com - padding
    grid_max = ligand_com + padding

    nx = int(np.ceil((grid_max[0] - grid_min[0]) / spacing)) + 1
    ny = int(np.ceil((grid_max[1] - grid_min[1]) / spacing)) + 1
    nz = int(np.ceil((grid_max[2] - grid_min[2]) / spacing)) + 1
    shape = (nx, ny, nz)

    print(f"PROGRESS: Grid dimensions: {nx} x {ny} x {nz} = {nx * ny * nz} points")

    os.makedirs(args.output_dir, exist_ok=True)
    origin = grid_min.tolist()

    x = np.linspace(grid_min[0], grid_min[0] + (nx - 1) * spacing, nx)
    y = np.linspace(grid_min[1], grid_min[1] + (ny - 1) * spacing, ny)
    z = np.linspace(grid_min[2], grid_min[2] + (nz - 1) * spacing, nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # Try cpptraj GIST first
    gist_data = run_cpptraj_gist(
        args.pdb_path, args.trajectory_path,
        ligand_com, nx, ny, nz, spacing, args.output_dir
    )

    bulk_density = 0.0334  # TIP3P at 300K: ~0.0334 molecules/A^3

    if gist_data is not None and 'gO' in gist_data:
        print("PROGRESS: Decomposing GIST data into pharmacophore channels...")

        gO = gist_data['gO']
        dG_excess = np.zeros(shape)
        if 'Esw' in gist_data and 'Eww' in gist_data:
            Esw = gist_data['Esw']
            Eww = gist_data['Eww']
            dTS = np.zeros(shape)
            if 'dTStrans' in gist_data:
                dTS += gist_data['dTStrans']
            if 'dTSorient' in gist_data:
                dTS += gist_data['dTSorient']
            dG_excess = Esw + Eww - dTS
        else:
            dG_excess = np.maximum(0, bulk_density - gO)

        phi = get_protein_electrostatic_grid(args.pdb_path, grid_points, shape)
        phi_3d = phi.reshape(shape)

        hydro_3d = np.maximum(0, dG_excess) * np.minimum(gO / bulk_density, 2.0)
        neg_phi = np.maximum(0, -phi_3d)
        donor_3d = np.maximum(0, dG_excess) * neg_phi
        pos_phi = np.maximum(0, phi_3d)
        acceptor_3d = np.maximum(0, dG_excess) * pos_phi

    else:
        print("PROGRESS: Using water density fallback...")
        water_density = compute_water_density(
            args.pdb_path, args.trajectory_path,
            ligand_com, nx, ny, nz, spacing
        )

        if water_density is None:
            print("Error: Could not compute water density", file=sys.stderr)
            sys.exit(1)

        phi = get_protein_electrostatic_grid(args.pdb_path, grid_points, shape)
        phi_3d = phi.reshape(shape)

        hydro_3d = np.maximum(0, bulk_density - water_density)
        donor_3d = water_density * np.maximum(0, -phi_3d)
        acceptor_3d = water_density * np.maximum(0, phi_3d)

    hydro_3d = normalize_grid(hydro_3d)
    donor_3d = normalize_grid(donor_3d)
    acceptor_3d = normalize_grid(acceptor_3d)

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
        'method': 'solvation',
    }

    results_path = os.path.join(args.output_dir, f'{prefix}binding_site_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"PROGRESS: Results written to {results_path}")
    print(f"PROGRESS: Hotspots found: {len(hotspots)}")
    print("Done!")


if __name__ == '__main__':
    main()
