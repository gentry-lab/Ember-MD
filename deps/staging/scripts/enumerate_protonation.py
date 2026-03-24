#!/usr/bin/env python3
"""
Enumerate protonation/tautomeric states for ligands using Molscrub.

Molscrub (Forli Lab, same team as AutoDock Vina and Meeko) handles:
  - pH-dependent protonation via 28 curated pKa reactions
  - Tautomer enumeration (keto-enol, aromatic heteroatom, extended)
  - Ring conformation fixing (boat→chair, chair flips, amine N inversion)
  - Salt stripping (keeps largest fragment)
  - 3D conformer generation (ETKDGv3 + MMFF94s/UFF)

Usage:
    python enumerate_protonation.py \
        --ligand_list <json_file> \
        --output_dir <path> \
        --ph_min 6.4 \
        --ph_max 8.4

Output:
    JSON with { protonated_paths: [...], parent_mapping: {...} }
"""

import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Tuple

from utils import load_sdf

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("ERROR: RDKit not installed", file=sys.stderr)
    sys.exit(1)

try:
    from molscrub import Scrub
    HAS_MOLSCRUB = True
except ImportError:
    HAS_MOLSCRUB = False


def load_mol_from_sdf(sdf_path: str) -> Any:
    """Load first molecule from an SDF or gzipped SDF file."""
    try:
        return load_sdf(sdf_path)
    except Exception as e:
        print(f"Warning: Failed to read {sdf_path}: {e}", file=sys.stderr)
        return None


def process_ligand(sdf_path: str, output_dir: str, scrub: Any) -> List[Tuple[str, str]]:
    """Process a single ligand: enumerate variants and generate 3D structures.

    Returns list of (output_path, parent_name) tuples.
    """
    parent_name = Path(sdf_path).stem.replace('_docked', '').replace('.sdf', '')

    mol = load_mol_from_sdf(sdf_path)
    if mol is None:
        raise RuntimeError(f"Could not read ligand input: {sdf_path}")

    # Molscrub's protonation logic is more reliable on the heavy-atom graph than
    # on fully explicit-H 3D inputs produced by our import pipeline.
    mol = Chem.RemoveHs(mol)

    try:
        variants = list(scrub(mol))
    except Exception as e:
        raise RuntimeError(f"Molscrub failed for {parent_name}: {e}") from e

    results = []

    for variant_idx, variant_mol in enumerate(variants):
        if variant_mol is None:
            continue

        if len(variants) == 1:
            output_name = parent_name
        else:
            output_name = f"{parent_name}_prot_{variant_idx}"

        smiles = Chem.MolToSmiles(Chem.RemoveHs(variant_mol))
        variant_mol.SetProp("_Name", output_name)
        variant_mol.SetProp("SMILES", smiles)
        variant_mol.SetProp("parent_molecule", parent_name)
        variant_mol.SetProp("protonation_variant", str(variant_idx))

        output_path = os.path.join(output_dir, f"{output_name}.sdf")
        writer = Chem.SDWriter(output_path)
        writer.write(variant_mol)
        writer.close()

        results.append((output_path, parent_name))
        print(f"Generated: {output_name} from {parent_name}")

    if not results:
        raise RuntimeError(f"Molscrub produced no protonation states for {parent_name}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='Enumerate protonation states for ligands')
    parser.add_argument('--ligand_list', required=True, help='JSON file with list of SDF paths')
    parser.add_argument('--output_dir', required=True, help='Output directory for protonated SDFs')
    parser.add_argument('--ph_min', type=float, default=6.4, help='Minimum pH for protonation')
    parser.add_argument('--ph_max', type=float, default=8.4, help='Maximum pH for protonation')
    args = parser.parse_args()

    if not HAS_MOLSCRUB:
        print("ERROR: Molscrub is not installed in the active Python environment.", file=sys.stderr)
        print("Install with: pip install molscrub", file=sys.stderr)
        sys.exit(1)

    with open(args.ligand_list, 'r') as f:
        ligand_paths = json.load(f)

    if not ligand_paths:
        print("No ligands to process")
        print(json.dumps({
            "protonated_paths": [],
            "parent_mapping": {}
        }))
        return

    os.makedirs(args.output_dir, exist_ok=True)

    scrub = Scrub(ph_low=args.ph_min, ph_high=args.ph_max) if HAS_MOLSCRUB else None

    all_protonated_paths: List[str] = []
    parent_mapping: dict[str, str] = {}

    for i, sdf_path in enumerate(ligand_paths):
        print(f"Processing {i+1}/{len(ligand_paths)}: {os.path.basename(sdf_path)}")

        try:
            results = process_ligand(sdf_path, args.output_dir, scrub)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

        for output_path, parent_name in results:
            all_protonated_paths.append(output_path)
            variant_name = Path(output_path).stem
            parent_mapping[variant_name] = parent_name

    print(f"\nProtonation complete: {len(all_protonated_paths)} variants from {len(ligand_paths)} molecules")

    print(json.dumps({
        "protonated_paths": all_protonated_paths,
        "parent_mapping": parent_mapping
    }))


if __name__ == '__main__':
    main()
