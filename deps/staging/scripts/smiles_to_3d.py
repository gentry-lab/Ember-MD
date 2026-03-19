#!/usr/bin/env python3
"""
Convert SMILES CSV to 3D SDF files using RDKit ETKDG.

Usage:
    python smiles_to_3d.py --input_csv <path> --output_dir <path>

Input CSV format:
    Must have a "smiles" column (case-insensitive)
    Optional "name" column for molecule names

Output:
    - SDF files in output_dir
    - JSON array of molecules: [{ filename, smiles, qed, saScore, sdfPath }, ...]
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, QED
except ImportError:
    print("ERROR: RDKit not installed", file=sys.stderr)
    sys.exit(1)


def generate_3d_conformer(mol: Any, max_attempts: int = 10) -> Any:
    """Generate 3D conformer using ETKDG."""
    # Add hydrogens
    mol = Chem.AddHs(mol)

    # Try ETKDG first
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.maxIterations = max_attempts

    result = AllChem.EmbedMolecule(mol, params)

    if result == -1:
        # Fall back to random coordinates
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            return None

    # Optimize geometry
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass  # Keep unoptimized conformer

    return mol


def calculate_properties(mol: Any) -> Tuple[float, float]:
    """Calculate QED and SA score."""
    qed_score = 0.5
    sa_score = 3.0

    try:
        qed_score = QED.qed(mol)
    except Exception:
        pass

    from utils import calculate_sa_score
    sa_score = calculate_sa_score(mol, default=sa_score)

    return qed_score, sa_score


def process_smiles_csv(input_csv: str, output_dir: str) -> List[Dict[str, Any]]:
    """Process SMILES CSV and generate 3D SDF files."""
    molecules = []

    os.makedirs(output_dir, exist_ok=True)

    # Read CSV and find columns
    with open(input_csv, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        # Find SMILES column (case-insensitive)
        headers = reader.fieldnames
        smiles_col = None
        name_col = None

        for h in headers:
            h_lower = h.lower().strip()
            if h_lower == 'smiles':
                smiles_col = h
            elif h_lower in ('name', 'id', 'molecule_name', 'mol_name', 'compound'):
                name_col = h

        if smiles_col is None:
            print("ERROR: No 'smiles' column found in CSV", file=sys.stderr)
            sys.exit(1)

        rows = list(reader)

    total = len(rows)
    print(f"Processing {total} SMILES from {input_csv}")

    for i, row in enumerate(rows):
        smiles = row[smiles_col].strip()

        # Get name or generate one
        if name_col and row.get(name_col):
            name = row[name_col].strip()
            # Sanitize name for filename
            name = "".join(c if c.isalnum() or c in '-_' else '_' for c in name)
        else:
            name = f"mol_{i+1:04d}"

        # Progress update
        print(f"PROGRESS: {i+1}/{total} - Converting {name}")
        sys.stdout.flush()

        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  WARNING: Invalid SMILES: {smiles}", file=sys.stderr)
            continue

        # Generate 3D conformer
        mol_3d = generate_3d_conformer(mol)
        if mol_3d is None:
            print(f"  WARNING: Could not generate 3D for: {name}", file=sys.stderr)
            continue

        # Calculate properties
        qed, sa_score = calculate_properties(mol)

        # Save to SDF
        sdf_path = os.path.join(output_dir, f"{name}.sdf")

        writer = Chem.SDWriter(sdf_path)
        mol_3d.SetProp("_Name", name)
        mol_3d.SetProp("SMILES", smiles)
        mol_3d.SetProp("QED", f"{qed:.4f}")
        mol_3d.SetProp("SA_Score", f"{sa_score:.2f}")
        writer.write(mol_3d)
        writer.close()

        molecules.append({
            'filename': name,
            'smiles': smiles,
            'qed': round(qed, 4),
            'saScore': round(sa_score, 2),
            'sdfPath': sdf_path,
        })

    return molecules


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert SMILES CSV to 3D SDF')
    parser.add_argument('--input_csv', required=True, help='Input CSV with SMILES column')
    parser.add_argument('--output_dir', required=True, help='Output directory for SDF files')
    args = parser.parse_args()

    if not os.path.isfile(args.input_csv):
        print(f"ERROR: Input file not found: {args.input_csv}", file=sys.stderr)
        sys.exit(1)

    molecules = process_smiles_csv(args.input_csv, args.output_dir)

    print(f"\nGenerated {len(molecules)} 3D structures")

    # Output JSON (will be parsed by electron)
    print(json.dumps(molecules))


if __name__ == '__main__':
    main()
