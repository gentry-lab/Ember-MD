#!/usr/bin/env python3
"""
Generate conformers for ligands using RDKit ETKDG.

This script takes a list of SDF files, generates multiple 3D conformers for each
using RDKit's ETKDG algorithm, filters by RMSD diversity and energy window,
and outputs them with a naming convention that tracks the parent molecule.

Usage:
    python generate_conformers.py \
        --ligand_list <json_file> \
        --output_dir <path> \
        --max_conformers 10 \
        --rmsd_cutoff 0.5 \
        --energy_window 10.0

Output:
    JSON with { conformer_paths: [...], parent_mapping: {...} }
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Chem.rdMolAlign import GetBestRMS
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("ERROR: RDKit not installed", file=sys.stderr)
    sys.exit(1)


def read_molecule_from_sdf(sdf_path: str) -> Any:
    """
    Read a molecule from an SDF file.
    Returns the first molecule in the file.
    """
    try:
        if sdf_path.endswith('.gz'):
            import gzip
            with gzip.open(sdf_path, 'rb') as f:
                suppl = Chem.ForwardSDMolSupplier(f)
                mol = next(suppl, None)
        else:
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
            mol = suppl[0] if len(suppl) > 0 else None

        return mol
    except Exception as e:
        print(f"Warning: Failed to read {sdf_path}: {e}", file=sys.stderr)
        return None


def generate_conformers_etkdg(mol: Any, max_conformers: int, rmsd_cutoff: float, energy_window: float) -> Tuple[Any, List[Tuple[int, float]]]:
    """
    Generate diverse conformers using ETKDG algorithm.

    Args:
        mol: RDKit molecule with 3D coordinates
        max_conformers: Maximum number of conformers to generate
        rmsd_cutoff: Minimum RMSD between conformers (diversity filter)
        energy_window: Maximum energy difference from lowest energy conformer (kcal/mol)

    Returns:
        List of (conformer_id, energy) tuples for selected conformers
    """
    # Make a copy and add hydrogens if needed
    mol = Chem.AddHs(mol)

    # Generate many conformers initially (3x the requested amount for filtering)
    num_to_generate = max_conformers * 3

    # ETKDG parameters
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 0  # Use all available
    params.pruneRmsThresh = rmsd_cutoff * 0.8  # Initial pruning
    params.maxIterations = 100

    # Generate conformers
    try:
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_to_generate, params=params)
    except Exception as e:
        print(f"Warning: Initial embedding failed, trying with random coords: {e}", file=sys.stderr)
        params.useRandomCoords = True
        try:
            conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_to_generate, params=params)
        except Exception as e2:
            print(f"Warning: Conformer generation failed: {e2}", file=sys.stderr)
            return mol, []

    if len(conf_ids) == 0:
        print("Warning: No conformers generated", file=sys.stderr)
        return mol, []

    # Optimize all conformers and calculate energies
    energies = []
    for conf_id in conf_ids:
        try:
            # Try MMFF first
            ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conf_id)
            if ff is not None:
                ff.Minimize(maxIts=200)
                energy = ff.CalcEnergy()
                energies.append((conf_id, energy))
            else:
                # Fall back to UFF
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                if ff is not None:
                    ff.Minimize(maxIts=200)
                    energy = ff.CalcEnergy()
                    energies.append((conf_id, energy))
        except Exception as e:
            # Skip conformers that fail optimization
            pass

    if not energies:
        # If no energies calculated, return all conformers without filtering
        return mol, [(conf_id, 0.0) for conf_id in conf_ids[:max_conformers]]

    # Sort by energy
    energies.sort(key=lambda x: x[1])

    # Filter by energy window
    min_energy = energies[0][1]
    energies = [(cid, e) for cid, e in energies if e - min_energy <= energy_window]

    # Diversity filtering by RMSD
    selected: List[Tuple[int, float]] = []
    for conf_id, energy in energies:
        if len(selected) >= max_conformers:
            break

        # Check RMSD to all selected conformers
        is_diverse = True
        for sel_id, _ in selected:
            try:
                rmsd = GetBestRMS(mol, mol, sel_id, conf_id)
                if rmsd < rmsd_cutoff:
                    is_diverse = False
                    break
            except Exception:
                pass  # If RMSD fails, assume diverse

        if is_diverse:
            selected.append((conf_id, energy))

    return mol, selected


def process_ligand(sdf_path: str, output_dir: str, max_conformers: int, rmsd_cutoff: float, energy_window: float) -> List[Tuple[str, str]]:
    """
    Process a single ligand: generate conformers and write to output.
    Returns list of (output_path, parent_name) tuples.
    """
    parent_name = Path(sdf_path).stem.replace('_docked', '').replace('.sdf', '')

    # Read input molecule
    mol = read_molecule_from_sdf(sdf_path)
    if mol is None:
        print(f"Warning: Could not read molecule from {sdf_path}", file=sys.stderr)
        return [(sdf_path, parent_name)]  # Return original

    # Get original SMILES for property storage
    try:
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception:
        smiles = ""

    # Get properties from original molecule
    original_props = {}
    for prop in mol.GetPropsAsDict():
        original_props[prop] = mol.GetProp(prop)

    # Generate conformers
    mol_with_confs, selected_conformers = generate_conformers_etkdg(
        mol, max_conformers, rmsd_cutoff, energy_window
    )

    if not selected_conformers:
        print(f"Warning: No conformers generated for {parent_name}, using original", file=sys.stderr)
        return [(sdf_path, parent_name)]

    results = []

    for conf_idx, (conf_id, energy) in enumerate(selected_conformers):
        # Create output name with conformer suffix
        if len(selected_conformers) == 1:
            # If only one conformer, keep original name
            output_name = parent_name
        else:
            output_name = f"{parent_name}_conf_{conf_idx}"

        # Create a new molecule with just this conformer
        conf_mol = Chem.Mol(mol_with_confs, confId=conf_id)

        # Set properties
        conf_mol.SetProp("_Name", output_name)
        if smiles:
            conf_mol.SetProp("SMILES", smiles)
        conf_mol.SetProp("parent_molecule", parent_name)
        conf_mol.SetProp("conformer_index", str(conf_idx))
        conf_mol.SetProp("conformer_energy", f"{energy:.2f}")

        # Copy original properties (CNNscore, etc.)
        for key, value in original_props.items():
            if key not in ['_Name', 'SMILES', 'parent_molecule', 'conformer_index', 'conformer_energy']:
                try:
                    conf_mol.SetProp(key, str(value))
                except Exception:
                    pass

        # Write to SDF
        output_path = os.path.join(output_dir, f"{output_name}.sdf")
        writer = Chem.SDWriter(output_path)
        writer.write(conf_mol)
        writer.close()

        results.append((output_path, parent_name))

    print(f"Generated: {len(results)} conformers from {parent_name}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate conformers for ligands using ETKDG')
    parser.add_argument('--ligand_list', required=True, help='JSON file with list of SDF paths')
    parser.add_argument('--output_dir', required=True, help='Output directory for conformer SDFs')
    parser.add_argument('--max_conformers', type=int, default=10, help='Maximum conformers per molecule')
    parser.add_argument('--rmsd_cutoff', type=float, default=0.5, help='RMSD cutoff for diversity (Angstroms)')
    parser.add_argument('--energy_window', type=float, default=10.0, help='Energy window for filtering (kcal/mol)')
    args = parser.parse_args()

    # Read ligand list
    with open(args.ligand_list, 'r') as f:
        ligand_paths = json.load(f)

    if not ligand_paths:
        print("No ligands to process")
        print(json.dumps({
            "conformer_paths": [],
            "parent_mapping": {}
        }))
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Conformer Generation (ETKDG) ===")
    print(f"Input molecules: {len(ligand_paths)}")
    print(f"Max conformers: {args.max_conformers}")
    print(f"RMSD cutoff: {args.rmsd_cutoff} A")
    print(f"Energy window: {args.energy_window} kcal/mol")
    print()

    # Process each ligand
    all_conformer_paths = []
    parent_mapping = {}

    for i, sdf_path in enumerate(ligand_paths):
        print(f"Processing {i+1}/{len(ligand_paths)}: {os.path.basename(sdf_path)}")

        results = process_ligand(
            sdf_path, args.output_dir,
            args.max_conformers, args.rmsd_cutoff, args.energy_window
        )

        for output_path, parent_name in results:
            all_conformer_paths.append(output_path)
            variant_name = Path(output_path).stem
            parent_mapping[variant_name] = parent_name

    # Output results
    print(f"\nConformer generation complete: {len(all_conformer_paths)} conformers from {len(ligand_paths)} molecules")

    # Print JSON result for parsing
    print(json.dumps({
        "conformer_paths": all_conformer_paths,
        "parent_mapping": parent_mapping
    }))


if __name__ == '__main__':
    main()
