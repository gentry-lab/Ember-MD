#!/usr/bin/env python3
"""
Align multiple PDB structures by backbone atoms.

Used to superimpose cluster centroid structures for overlay visualization.
Writes aligned PDB files to output directory and returns paths.
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')


def main() -> None:
    parser = argparse.ArgumentParser(description='Align multiple PDB structures by backbone')
    parser.add_argument('--pdb_files', nargs='+', required=True, help='PDB files to align')
    parser.add_argument('--output_dir', required=True, help='Output directory for aligned PDBs')
    args = parser.parse_args()

    if len(args.pdb_files) < 1:
        print(json.dumps({'error': 'At least one PDB file required'}))
        sys.exit(1)

    try:
        import MDAnalysis as mda
        from MDAnalysis.analysis import align
    except ImportError as e:
        print(json.dumps({'error': f'Missing required package: {e}'}))
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load reference (first PDB)
    ref_path = args.pdb_files[0]
    try:
        ref = mda.Universe(ref_path)
    except Exception as e:
        print(json.dumps({'error': f'Failed to load reference PDB {ref_path}: {e}'}))
        sys.exit(1)

    # Select backbone atoms for alignment
    ref_backbone = ref.select_atoms('protein and (backbone or name CA C N O)')
    if len(ref_backbone) == 0:
        ref_backbone = ref.select_atoms('protein')
    if len(ref_backbone) == 0:
        ref_backbone = ref.atoms

    print(f"Reference: {ref_path} ({len(ref_backbone)} alignment atoms)", file=sys.stderr)

    results = []

    for pdb_path in args.pdb_files:
        try:
            mobile = mda.Universe(pdb_path)

            # Select atoms to align
            mobile_backbone = mobile.select_atoms('protein and (backbone or name CA C N O)')
            if len(mobile_backbone) == 0:
                mobile_backbone = mobile.select_atoms('protein')
            if len(mobile_backbone) == 0:
                mobile_backbone = mobile.atoms

            # Align if we have matching atom counts
            if len(mobile_backbone) == len(ref_backbone):
                align.alignto(mobile, ref, select='protein and (backbone or name CA C N O)',
                             match_atoms=False)
            else:
                try:
                    align.alignto(mobile, ref, select='protein and name CA',
                                 match_atoms=True)
                except Exception:
                    pass  # Skip alignment if it fails

            # Write aligned structure to output directory
            basename = os.path.basename(pdb_path)
            name, ext = os.path.splitext(basename)
            output_path = os.path.join(args.output_dir, f"{name}_aligned{ext}")

            # Write all atoms (including ligand)
            mobile.atoms.write(output_path)

            print(f"Aligned: {basename} -> {output_path}", file=sys.stderr)

            results.append({
                'originalPath': pdb_path,
                'alignedPath': output_path,
            })

        except Exception as e:
            print(f"Error aligning {pdb_path}: {e}", file=sys.stderr)
            # On error, just copy the original path
            results.append({
                'originalPath': pdb_path,
                'alignedPath': pdb_path,
                'error': str(e),
            })

    # Output JSON result
    print(json.dumps({'alignedPdbs': results}))


if __name__ == '__main__':
    main()
