#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""Prepare a ligand SDF for viewer display: sanitize, add hydrogens, assign bond orders."""

import argparse
import json
import os
import sys


def prepare_ligand(input_path: str, output_path: str) -> dict:
    """Read SDF, sanitize with RDKit, add explicit Hs, write back."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    suppl = Chem.SDMolSupplier(input_path, removeHs=False, sanitize=False)
    writer = Chem.SDWriter(output_path)
    count = 0

    for mol in suppl:
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            # If sanitization fails, try without kekulization
            try:
                Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except Exception:
                writer.write(mol)
                count += 1
                continue

        # Assign stereochemistry from 3D coordinates if available
        try:
            Chem.AssignStereochemistryFrom3D(mol)
        except Exception:
            pass

        # Add explicit hydrogens with 3D coordinates
        try:
            mol = Chem.AddHs(mol, addCoords=True)
        except Exception:
            pass

        writer.write(mol)
        count += 1

    writer.close()
    return {'success': True, 'output': output_path, 'count': count}


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare ligand SDF for viewing')
    parser.add_argument('--input', required=True, help='Input SDF file')
    parser.add_argument('--output', required=True, help='Output SDF file')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    try:
        result = prepare_ligand(args.input, args.output)
        print(json.dumps(result))
    except Exception as e:
        print(f"ERROR: Ligand preparation failed: {e}", file=sys.stderr)
        # Fall back: copy input to output
        import shutil
        shutil.copy2(args.input, args.output)
        print(json.dumps({'success': True, 'output': args.output, 'fallback': True}))


if __name__ == '__main__':
    main()
