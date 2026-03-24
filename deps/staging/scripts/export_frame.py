#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
Export a single frame from an MD trajectory as a PDB file.

Useful for extracting representative structures or specific timepoints.
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(description='Export a single frame from trajectory')
    parser.add_argument('--topology', required=True, help='Topology file (PDB)')
    parser.add_argument('--trajectory', required=True, help='Trajectory file (DCD)')
    parser.add_argument('--frame', type=int, required=True, help='Frame index (0-based)')
    parser.add_argument('--output', required=True, help='Output PDB path')
    parser.add_argument('--strip_waters', action='store_true', help='Remove waters and ions')
    args = parser.parse_args()

    try:
        import MDAnalysis as mda
    except ImportError as e:
        print(f"Error: Missing required package: {e}", file=sys.stderr)
        print("Install with: pip install MDAnalysis", file=sys.stderr)
        sys.exit(1)

    # Load universe
    u = mda.Universe(args.topology, args.trajectory)
    n_frames = len(u.trajectory)

    if args.frame < 0 or args.frame >= n_frames:
        print(f"Error: Frame {args.frame} out of range (0-{n_frames - 1})", file=sys.stderr)
        sys.exit(1)

    # Apply PBC transformations to center protein in view
    from utils import select_ligand_atoms, apply_pbc_transforms
    protein = u.select_atoms('protein')
    ligand = select_ligand_atoms(u)
    apply_pbc_transforms(u, protein, ligand)

    # Go to specified frame
    u.trajectory[args.frame]

    # Select atoms to write
    if args.strip_waters:
        atoms_to_write = u.select_atoms('not (resname WAT HOH TIP3 TIP4 NA CL SOL K MG)')
    else:
        atoms_to_write = u.atoms

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write PDB
    atoms_to_write.write(args.output)

    print(f"Exported frame {args.frame} to: {args.output}")
    print(f"  Atoms: {len(atoms_to_write)}")


if __name__ == '__main__':
    main()
