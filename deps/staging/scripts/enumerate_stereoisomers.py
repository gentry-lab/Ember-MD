#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
Enumerate stereoisomers for ligands with unspecified stereocenters.

Uses RDKit EnumerateStereoisomers with onlyUnassigned=True so that molecules
with explicit chirality are left unchanged, and only undefined stereocenters
are expanded.

Usage:
    python enumerate_stereoisomers.py \
        --ligand_list <json_file> \
        --output_dir <path> \
        --max_stereoisomers 4

Output:
    JSON with { stereoisomer_paths: [...], parent_mapping: {...} }
"""

import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.EnumerateStereoisomers import (
        EnumerateStereoisomers,
        StereoEnumerationOptions,
    )
except ImportError:
    print("ERROR: RDKit not installed", file=sys.stderr)
    sys.exit(1)


def load_mol_from_sdf(sdf_path: str) -> Any:
    """Load first molecule from an SDF or gzipped SDF file."""
    try:
        if sdf_path.endswith('.gz'):
            with gzip.open(sdf_path, 'rb') as f:
                suppl = Chem.ForwardSDMolSupplier(f)
                return next(suppl, None)
        else:
            suppl = Chem.SDMolSupplier(sdf_path)
            return suppl[0] if len(suppl) > 0 else None
    except Exception as e:
        print(f"Warning: Failed to read {sdf_path}: {e}", file=sys.stderr)
        return None


def process_ligand(sdf_path: str, output_dir: str, max_stereoisomers: int) -> List[Tuple[str, str]]:
    """Enumerate stereoisomers for a single ligand.

    Returns list of (output_path, parent_name) tuples.
    If the molecule has no unspecified stereocenters, returns the original path unchanged.
    """
    parent_name = Path(sdf_path).stem

    mol = load_mol_from_sdf(sdf_path)
    if mol is None:
        print(f"Warning: Could not read {sdf_path}", file=sys.stderr)
        return [(sdf_path, parent_name)]

    # Strip 3D-inferred stereochemistry before enumerating.
    # When a conformer is generated from a flat SMILES, RDKit assigns chirality
    # from the 3D geometry — so re-reading the SDF makes every stereocenter look
    # "specified". Round-tripping through a flat SMILES restores the true state.
    flat_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=False)
    mol_flat = Chem.MolFromSmiles(flat_smiles)
    if mol_flat is None:
        print(f"Warning: Could not parse flat SMILES for {parent_name}, using original", file=sys.stderr)
        mol_flat = mol

    opts = StereoEnumerationOptions(
        unique=True,
        onlyUnassigned=True,
        maxIsomers=max_stereoisomers,
    )

    isomers = list(EnumerateStereoisomers(mol_flat, options=opts))

    if len(isomers) <= 1:
        # No unspecified centers — return original unchanged
        print(f"No unspecified stereocenters: {parent_name}")
        return [(sdf_path, parent_name)]

    results = []
    for idx, isomer in enumerate(isomers):
        output_name = f"{parent_name}_stereo_{idx}"
        smiles = Chem.MolToSmiles(isomer, isomericSmiles=True)

        # Embed 3D coordinates so downstream steps (Vina, conformer gen) get a valid SDF
        isomer_3d = Chem.AddHs(isomer)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(isomer_3d, params) != 0:
            params.useRandomCoords = True
            AllChem.EmbedMolecule(isomer_3d, params)
        try:
            AllChem.MMFFOptimizeMolecule(isomer_3d, maxIters=500)
        except Exception:
            pass
        isomer_3d = Chem.RemoveHs(isomer_3d)

        isomer_3d.SetProp("_Name", output_name)
        isomer_3d.SetProp("SMILES", smiles)
        isomer_3d.SetProp("parent_molecule", parent_name)
        isomer_3d.SetProp("stereoisomer_index", str(idx))

        output_path = os.path.join(output_dir, f"{output_name}.sdf")
        writer = Chem.SDWriter(output_path)
        writer.write(isomer_3d)
        writer.close()

        results.append((output_path, parent_name))
        print(f"Generated: {output_name} from {parent_name} ({smiles})")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='Enumerate stereoisomers for ligands')
    parser.add_argument('--ligand_list', required=True, help='JSON file with list of SDF paths')
    parser.add_argument('--output_dir', required=True, help='Output directory for stereoisomer SDFs')
    parser.add_argument('--max_stereoisomers', type=int, default=8,
                        help='Max stereoisomers per molecule (default: 8)')
    args = parser.parse_args()

    with open(args.ligand_list, 'r') as f:
        ligand_paths = json.load(f)

    if not ligand_paths:
        print("No ligands to process")
        print(json.dumps({"stereoisomer_paths": [], "parent_mapping": {}}))
        return

    os.makedirs(args.output_dir, exist_ok=True)

    all_paths: List[str] = []
    parent_mapping: dict[str, str] = {}

    for i, sdf_path in enumerate(ligand_paths):
        print(f"Processing {i+1}/{len(ligand_paths)}: {os.path.basename(sdf_path)}")
        results = process_ligand(sdf_path, args.output_dir, args.max_stereoisomers)
        for output_path, parent_name in results:
            all_paths.append(output_path)
            variant_name = Path(output_path).stem
            parent_mapping[variant_name] = parent_name

    expanded = len(all_paths) - len(ligand_paths)
    print(f"\nStereoisomer enumeration complete: {len(all_paths)} variants from {len(ligand_paths)} molecules (+{expanded} new)")

    print(json.dumps({
        "stereoisomer_paths": all_paths,
        "parent_mapping": parent_mapping,
    }))


if __name__ == '__main__':
    main()
