#!/usr/bin/env python
# Copyright (c) 2026 Ember Contributors. MIT License.
"""Generate a 2D structure thumbnail from SDF using shared depiction helpers."""
import argparse
import sys

from ligand_torsion_utils import load_canonical_ligand_mol, render_png_data_url


def generate_thumbnail(sdf_path, size=300):
    """Generate a 2D thumbnail image from an SDF file."""
    try:
        mol = load_canonical_ligand_mol(sdf_path)
        return render_png_data_url(mol, size=size)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description='Generate 2D thumbnail from SDF')
    parser.add_argument('--sdf_file', required=True, help='Input SDF file')
    parser.add_argument('--size', type=int, default=300, help='Image size in pixels')
    args = parser.parse_args()

    result = generate_thumbnail(args.sdf_file, args.size)
    if result:
        print(result)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
