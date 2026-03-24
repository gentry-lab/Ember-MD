#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
Convert structure CSV rows to 3D SDF files using RDKit ETKDG.

Usage:
    python smiles_to_3d.py --input_csv <path> --output_dir <path>

Input CSV format:
    Must have either a "smiles" column (case-insensitive) or a structure-file
    column such as "structure_file", "mol_file", "file", or "path".
    Optional "name" column for molecule names.

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


def load_structure_file(file_path: str) -> Any:
    """Load a structure file and ensure it has 3D coordinates."""
    lower = file_path.lower()

    if lower.endswith('.sdf.gz'):
        import gzip
        with gzip.open(file_path, 'rt') as f:
            supplier = Chem.ForwardSDMolSupplier(f, removeHs=False)
            mol = next((m for m in supplier if m is not None), None)
    elif lower.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(file_path, removeHs=False)
        mol = next((m for m in supplier if m is not None), None)
    elif lower.endswith('.mol2'):
        mol = Chem.MolFromMol2File(file_path, removeHs=False)
    elif lower.endswith('.mol'):
        mol = Chem.MolFromMolFile(file_path, removeHs=False)
    else:
        raise ValueError(f'Unsupported structure file: {file_path}')

    if mol is None:
        raise ValueError(f'Failed to parse structure file: {file_path}')

    has_3d = mol.GetNumConformers() > 0 and mol.GetConformer().Is3D()
    if has_3d:
        return mol

    return generate_3d_conformer(Chem.RemoveHs(mol))


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
    """Process a structure CSV and generate 3D SDF files."""
    molecules = []

    os.makedirs(output_dir, exist_ok=True)
    csv_dir = os.path.dirname(os.path.abspath(input_csv))

    # Read CSV and find columns
    with open(input_csv, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        # Find supported columns (case-insensitive)
        headers = reader.fieldnames or []
        smiles_col = None
        structure_col = None
        name_col = None

        for h in headers:
            h_lower = h.lower().strip()
            if h_lower == 'smiles':
                smiles_col = h
            elif h_lower in ('structure_file', 'mol_file', 'file', 'path', 'structure_path'):
                structure_col = h
            elif h_lower in ('name', 'id', 'molecule_name', 'mol_name', 'compound'):
                name_col = h

        if smiles_col is None and structure_col is None:
            print("ERROR: CSV must contain either 'smiles' or a structure-file column", file=sys.stderr)
            sys.exit(1)

        rows = list(reader)

    total = len(rows)
    print(f"Processing {total} molecules from {input_csv}")

    for i, row in enumerate(rows):
        smiles = row.get(smiles_col, '').strip() if smiles_col else ''
        structure_ref = row.get(structure_col, '').strip() if structure_col else ''

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

        try:
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print(f"  WARNING: Invalid SMILES: {smiles}", file=sys.stderr)
                    continue
                mol_3d = generate_3d_conformer(mol)
                if mol_3d is None:
                    print(f"  WARNING: Could not generate 3D for: {name}", file=sys.stderr)
                    continue
            elif structure_ref:
                structure_path = structure_ref
                if not os.path.isabs(structure_path):
                    structure_path = os.path.join(csv_dir, structure_path)
                if not os.path.isfile(structure_path):
                    print(f"  WARNING: Structure file not found: {structure_ref}", file=sys.stderr)
                    continue
                mol_3d = load_structure_file(structure_path)
                smiles = Chem.MolToSmiles(Chem.RemoveAllHs(mol_3d))
            else:
                print(f"  WARNING: Row {i+1} has neither SMILES nor structure file", file=sys.stderr)
                continue
        except Exception as exc:
            print(f"  WARNING: Could not process {name}: {exc}", file=sys.stderr)
            continue

        # Calculate properties
        qed, sa_score = calculate_properties(Chem.RemoveAllHs(mol_3d))

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
    parser = argparse.ArgumentParser(description='Convert structure CSV to 3D SDF')
    parser.add_argument('--input_csv', required=True, help='Input CSV with smiles or structure-file column')
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
