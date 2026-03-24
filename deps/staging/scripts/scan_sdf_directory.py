#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
Scan a directory for SDF files and extract SMILES and properties.

Usage:
    python scan_sdf_directory.py --directory <path> --output_dir <path>

Output:
    JSON array of molecules: [{ filename, smiles, qed, saScore, sdfPath }, ...]
"""

import argparse
import json
import os
import sys
import shutil
from typing import Any, Dict, List, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED
except ImportError:
    print(json.dumps([]))
    sys.exit(0)


def get_smiles_from_sdf(sdf_path: str) -> str:
    """Extract SMILES from SDF file."""
    try:
        # Try reading as regular SDF
        if sdf_path.endswith('.gz'):
            import gzip
            with gzip.open(sdf_path, 'rt') as f:
                suppl = Chem.ForwardSDMolSupplier(f)
                for mol in suppl:
                    if mol is not None:
                        return Chem.MolToSmiles(mol)
        else:
            suppl = Chem.SDMolSupplier(sdf_path)
            for mol in suppl:
                if mol is not None:
                    return Chem.MolToSmiles(mol)
    except Exception as e:
        print(f"Warning: Could not read {sdf_path}: {e}", file=sys.stderr)
    return ""


def calculate_properties(sdf_path: str) -> Tuple[float, float]:
    """Calculate QED and SA score from SDF file."""
    qed_score = 0.5  # Default
    sa_score = 3.0   # Default

    try:
        if sdf_path.endswith('.gz'):
            import gzip
            with gzip.open(sdf_path, 'rt') as f:
                suppl = Chem.ForwardSDMolSupplier(f)
                for mol in suppl:
                    if mol is not None:
                        qed_score = QED.qed(mol)
                        # SA score requires sascorer module, fall back to estimate
                        from utils import calculate_sa_score
                        sa_score = calculate_sa_score(mol)
                        break
        else:
            suppl = Chem.SDMolSupplier(sdf_path)
            for mol in suppl:
                if mol is not None:
                    qed_score = QED.qed(mol)
                    from utils import calculate_sa_score
                    sa_score = calculate_sa_score(mol)
                    break
    except Exception as e:
        print(f"Warning: Could not calculate properties for {sdf_path}: {e}", file=sys.stderr)

    return qed_score, sa_score


def scan_directory(directory: str, output_dir: str) -> List[Dict[str, Any]]:
    """Scan directory for SDF files and extract information."""
    molecules = []

    # Find all SDF files
    sdf_files = []
    for f in os.listdir(directory):
        if f.endswith('.sdf') or f.endswith('.sdf.gz'):
            sdf_files.append(f)

    sdf_files.sort()

    # Create output directory for copied files
    os.makedirs(output_dir, exist_ok=True)

    for sdf_file in sdf_files:
        sdf_path = os.path.join(directory, sdf_file)

        # Get base filename without extension
        if sdf_file.endswith('.sdf.gz'):
            filename = sdf_file[:-7]
        else:
            filename = sdf_file[:-4]

        # Extract SMILES
        smiles = get_smiles_from_sdf(sdf_path)

        # Calculate properties
        qed, sa_score = calculate_properties(sdf_path)

        # Copy SDF to output directory (use uncompressed for GNINA)
        dest_sdf = os.path.join(output_dir, f"{filename}.sdf")

        if sdf_file.endswith('.gz'):
            # Decompress
            import gzip
            with gzip.open(sdf_path, 'rb') as f_in:
                with open(dest_sdf, 'wb') as f_out:
                    f_out.write(f_in.read())
        else:
            shutil.copy2(sdf_path, dest_sdf)

        molecules.append({
            'filename': filename,
            'smiles': smiles,
            'qed': round(qed, 4),
            'saScore': round(sa_score, 2),
            'sdfPath': dest_sdf,
        })

    return molecules


def main() -> None:
    parser = argparse.ArgumentParser(description='Scan directory for SDF files')
    parser.add_argument('--directory', required=True, help='Directory to scan')
    parser.add_argument('--output_dir', required=True, help='Output directory for processed files')
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(json.dumps({'error': f'Directory not found: {args.directory}'}))
        sys.exit(1)

    molecules = scan_directory(args.directory, args.output_dir)
    print(json.dumps(molecules))


if __name__ == '__main__':
    main()
