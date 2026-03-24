#!/usr/bin/env python
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
Generate results CSV from SDF files with molecular properties.
"""
import argparse
import os
import csv
from glob import glob

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, AllChem


def calculate_sa_score(mol):
    """Calculate synthetic accessibility score (1-10, lower is better)."""
    from utils import calculate_sa_score as _sa
    return _sa(mol, default=-1)


def check_lipinski(mol):
    """Check Lipinski's Rule of Five."""
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)

    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1

    return violations <= 1  # Pass if at most 1 violation


def check_veber(mol):
    """Check Veber's rules for oral bioavailability."""
    tpsa = Descriptors.TPSA(mol)
    rotatable = Lipinski.NumRotatableBonds(mol)

    return tpsa <= 140 and rotatable <= 10


def process_sdf(sdf_path):
    """Process a single SDF file and return molecular properties."""
    try:
        suppl = Chem.SDMolSupplier(sdf_path, sanitize=True)
        mol = next(iter(suppl))
        if mol is None:
            return None

        smiles = Chem.MolToSmiles(mol)
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        rotatable = Lipinski.NumRotatableBonds(mol)
        qed = QED.qed(mol)
        sa_score = calculate_sa_score(mol)
        lipinski_pass = check_lipinski(mol)
        veber_pass = check_veber(mol)

        return {
            'filename': os.path.basename(sdf_path),
            'smiles': smiles,
            'mw': round(mw, 2),
            'logp': round(logp, 2),
            'tpsa': round(tpsa, 2),
            'hbd': hbd,
            'hba': hba,
            'rotatable_bonds': rotatable,
            'qed': round(qed, 3),
            'sa_score': round(sa_score, 2) if sa_score > 0 else 'N/A',
            'lipinski': 'Pass' if lipinski_pass else 'Fail',
            'veber': 'Pass' if veber_pass else 'Fail',
        }
    except Exception as e:
        print(f"Error processing {sdf_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Generate results CSV from SDFs')
    parser.add_argument('--sdf_dir', required=True, help='Directory containing SDF files')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    args = parser.parse_args()

    sdf_files = sorted(glob(os.path.join(args.sdf_dir, '*.sdf')))

    if not sdf_files:
        print(f"No SDF files found in {args.sdf_dir}")
        return

    print(f"Processing {len(sdf_files)} SDF files...")

    results = []
    for sdf_path in sdf_files:
        result = process_sdf(sdf_path)
        if result:
            results.append(result)

    if not results:
        print("No valid molecules found")
        return

    # Write CSV
    fieldnames = ['filename', 'smiles', 'mw', 'logp', 'tpsa', 'hbd', 'hba',
                  'rotatable_bonds', 'qed', 'sa_score', 'lipinski', 'veber']

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Generated CSV with {len(results)} molecules: {args.output}")


if __name__ == '__main__':
    main()
