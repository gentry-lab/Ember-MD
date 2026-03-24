#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
Unified receptor preparation for Dock and MD workflows.

Runs PDBFixer structural repair (missing residue/atom filling) followed
by PROPKA-guided protonation.  Both docking and MD consume the same
fully-prepared receptor_prepared.pdb.

Pipeline:
  1. CIF -> PDB conversion (if needed)
  2. Reduce side-chain flip optimization (if available)
  3. PROPKA shifted-residue detection
  4. PDBFixer: findMissingResidues -> findMissingAtoms -> addMissingAtoms
  5. Protonation via prepare_receptor_with_propka (shared core)
  6. Write receptor_prepared.pdb + receptor_prepared.prep.json

CLI:
  python prepare_receptor.py \\
    --input <raw.pdb|cif> \\
    --output_dir <dir> \\
    [--ph 7.4] \\
    [--pocket_ligand_sdf <path>]
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Set, Tuple

from utils import load_sdf
from receptor_protonation import (
    collect_propka_shifted_residues,
    identify_pocket_residue_keys_from_pdb,
    prepare_receptor_with_propka,
    run_reduce_if_available,
    POCKET_RESIDUE_CUTOFF_A,
)

try:
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer
except ImportError as e:
    print(f"ERROR:Missing dependency: {e}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_pdb_format(file_path: str) -> str:
    """Convert CIF to PDB if needed.  Returns path to a .pdb file."""
    if not file_path.lower().endswith('.cif'):
        return file_path

    pdb_path = file_path.rsplit('.', 1)[0] + '_converted.pdb'
    if os.path.exists(pdb_path):
        print(f'  Using cached CIF->PDB conversion: {os.path.basename(pdb_path)}', file=sys.stderr)
        return pdb_path

    print(f'  Converting CIF to PDB: {os.path.basename(file_path)}', file=sys.stderr)
    fixer = PDBFixer(filename=file_path)
    with open(pdb_path, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
    print(f'  Converted to: {os.path.basename(pdb_path)}', file=sys.stderr)
    return pdb_path


# ---------------------------------------------------------------------------
# Main preparation function
# ---------------------------------------------------------------------------

def prepare_receptor(
    input_path: str,
    output_dir: str,
    ph: float = 7.4,
    pocket_ligand_sdf: Optional[str] = None,
    pocket_residue_keys: Optional[Set[str]] = None,
    output_path: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Run the full receptor preparation pipeline.

    Args:
        input_path: Raw PDB or CIF file.
        output_dir: Directory for output files.
        ph: Protonation pH (default 7.4).
        pocket_ligand_sdf: Optional ligand SDF for pocket-aware protonation.
            Ignored if pocket_residue_keys is provided.
        pocket_residue_keys: Pre-computed pocket residue keys (e.g. from
            detect_pdb_ligands.py). Takes precedence over pocket_ligand_sdf.
        output_path: Explicit output PDB path. Defaults to
            {output_dir}/receptor_prepared.pdb.

    Returns (path_to_prepared_pdb, metadata_dict).
    """
    os.makedirs(output_dir, exist_ok=True)
    output_pdb = output_path or os.path.join(output_dir, 'receptor_prepared.pdb')

    print('PROGRESS:prepare_receptor:0', flush=True)

    # 1. CIF -> PDB conversion
    pdb_path = ensure_pdb_format(input_path)
    print('PROGRESS:prepare_receptor:5', flush=True)

    # 2. Reduce side-chain flips
    print('  Running Reduce for side-chain flips...', file=sys.stderr)
    reduced_path, reduce_report = run_reduce_if_available(pdb_path)
    print('PROGRESS:prepare_receptor:10', flush=True)

    # 3. PROPKA shifted-residue detection
    print('  Collecting PROPKA shifted residues...', file=sys.stderr)
    propka_report = collect_propka_shifted_residues(reduced_path, ph)
    print('PROGRESS:prepare_receptor:20', flush=True)

    # 4. Pocket residue detection (if ligand provided and keys not pre-computed)
    if pocket_residue_keys is not None:
        print(f'  Using {len(pocket_residue_keys)} pre-computed pocket residue keys', file=sys.stderr)
    elif pocket_ligand_sdf:
        try:
            mol = load_sdf(pocket_ligand_sdf, remove_hs=True)
            if mol is not None:
                conf = mol.GetConformer()
                ligand_coords = [
                    (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z)
                    for i in range(mol.GetNumAtoms())
                ]
                pocket_residue_keys = identify_pocket_residue_keys_from_pdb(
                    reduced_path, ligand_coords, POCKET_RESIDUE_CUTOFF_A,
                )
                print(f'  Identified {len(pocket_residue_keys)} pocket residues from ligand SDF', file=sys.stderr)
        except Exception as exc:
            print(f'  Warning: pocket detection from SDF failed: {exc}', file=sys.stderr)
    print('PROGRESS:prepare_receptor:25', flush=True)

    # 5. PDBFixer: load and fix structure
    print('  Loading receptor into PDBFixer...', file=sys.stderr)
    fixer = PDBFixer(filename=reduced_path)
    fixer.findMissingResidues()
    print('PROGRESS:prepare_receptor:35', flush=True)

    print('  Adding missing atoms...', file=sys.stderr)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    print('PROGRESS:prepare_receptor:60', flush=True)

    # 6. Delegate to prepare_receptor_with_propka for protonation + write
    print('  Protonating and writing receptor...', file=sys.stderr)
    extra_metadata = {
        'schema_version': 2,
        'input_path': input_path,
    }

    metadata = prepare_receptor_with_propka(
        pdb_path,
        output_pdb,
        ph,
        pocket_residue_keys=pocket_residue_keys,
        fixer=fixer,
        propka_report=propka_report,
        reduce_report=reduce_report,
        extra_metadata=extra_metadata,
    )

    print('PROGRESS:prepare_receptor:100', flush=True)
    print(f'  Receptor prepared: {metadata.get("prepared_atom_count", "?")} atoms', file=sys.stderr)

    # Clean up reduce temp file
    if reduced_path != pdb_path:
        try:
            os.remove(reduced_path)
        except OSError:
            pass

    return output_pdb, metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Unified receptor preparation for Dock and MD')
    parser.add_argument('--input', required=True, help='Input PDB or CIF file')
    parser.add_argument('--output_dir', required=True, help='Output directory for prepared receptor')
    parser.add_argument('--ph', type=float, default=7.4, help='Protonation pH (default: 7.4)')
    parser.add_argument('--pocket_ligand_sdf', help='Optional ligand SDF for pocket-aware protonation')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'ERROR:Input file not found: {args.input}', file=sys.stderr)
        sys.exit(1)

    try:
        output_pdb, metadata = prepare_receptor(
            args.input,
            args.output_dir,
            ph=args.ph,
            pocket_ligand_sdf=args.pocket_ligand_sdf,
        )
        print(json.dumps({
            'prepared_pdb': output_pdb,
            'metadata_path': metadata.get('metadata_path', ''),
        }, indent=2))
    except Exception as exc:
        print(f'ERROR:Receptor preparation failed: {exc}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
