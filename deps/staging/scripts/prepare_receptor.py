#!/usr/bin/env python3
"""
Unified receptor preparation for Dock and MD workflows.

Runs PDBFixer structural repair (chain break detection, missing residue/atom
filling) followed by PROPKA-guided protonation in a single step.  Both docking
and MD consume the same fully-prepared receptor_prepared.pdb.

Pipeline:
  1. CIF -> PDB conversion (if needed)
  2. Reduce side-chain flip optimization (if available)
  3. PROPKA shifted-residue detection
  4. PDBFixer: findMissingResidues -> chain break detection/splitting ->
     filter internal gaps -> findMissingAtoms -> addMissingAtoms
  5. Sanitize positions (fix PDBFixer nested-Quantity bug)
  6. Build protonation variant plan (pocket-filtered PROPKA overrides)
  7. Add hydrogens with explicit variants
  8. Write receptor_prepared.pdb + receptor_prepared.prep.json

CLI:
  python prepare_receptor.py \\
    --input <raw.pdb|cif> \\
    --output_dir <dir> \\
    [--ph 7.4] \\
    [--pocket_ligand_sdf <path>]
"""

import argparse
import builtins
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

from receptor_protonation import (
    _sanitize_positions,
    add_hydrogens_with_variants,
    build_variant_plan,
    collect_propka_shifted_residues,
    identify_pocket_residue_keys_from_pdb,
    run_reduce_if_available,
    write_prepared_receptor_pdb,
    write_receptor_prep_metadata,
    POCKET_RESIDUE_CUTOFF_A,
)

try:
    from openmm import Vec3
    from openmm.app import PDBFile
    from openmm.unit import nanometers
    from pdbfixer import PDBFixer
except ImportError as e:
    print(f"ERROR:Missing dependency: {e}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STANDARD_PROTEIN_RESIDUES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'HID', 'HIE', 'HIP', 'CYX', 'ASH', 'GLH', 'LYN', 'ACE', 'NME',
}


# ---------------------------------------------------------------------------
# Helpers (extracted from run_md_simulation.py)
# ---------------------------------------------------------------------------

def ensure_pdb_format(file_path: str) -> str:
    """Convert CIF to PDB if needed.  Returns path to a .pdb file.

    OpenMM's PDBFile class only reads PDB format, not mmCIF.
    Uses PDBFixer (which handles CIF) to convert, preserving all residues.
    """
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


def _safe_parse_residue_number(residue_id: str) -> Optional[int]:
    """Parse the integer component of a PDB residue id."""
    match = re.match(r'^\s*(-?\d+)', str(residue_id))
    return int(match.group(1)) if match else None


def _residue_label(residue: Any) -> str:
    insertion = residue.insertionCode.strip() if residue.insertionCode else ''
    return f'{residue.chain.id}:{residue.name}{residue.id}{insertion}'


def _find_atom_by_name(residue: Any, atom_name: str) -> Optional[Any]:
    for atom in residue.atoms():
        if atom.name == atom_name:
            return atom
    return None


def _detect_chain_breaks(topology: Any, positions: Any, threshold_nm: float = 0.2) -> List[Dict[str, Any]]:
    """Detect physical or numbering-based chain breaks in protein chains."""
    break_records: List[Dict[str, Any]] = []
    for chain_idx, chain in enumerate(topology.chains()):
        residues = list(chain.residues())
        for prev_index in range(len(residues) - 1):
            next_index = prev_index + 1
            prev_residue = residues[prev_index]
            next_residue = residues[next_index]
            if prev_residue.name not in STANDARD_PROTEIN_RESIDUES or next_residue.name not in STANDARD_PROTEIN_RESIDUES:
                continue
            reasons: List[str] = []
            missing_backbone_atoms: List[str] = []

            prev_c = _find_atom_by_name(prev_residue, 'C')
            next_n = _find_atom_by_name(next_residue, 'N')
            if prev_c is None:
                missing_backbone_atoms.append(f'{_residue_label(prev_residue)} missing C')
            if next_n is None:
                missing_backbone_atoms.append(f'{_residue_label(next_residue)} missing N')
            if missing_backbone_atoms:
                reasons.append('missing_backbone')

            prev_num = _safe_parse_residue_number(prev_residue.id)
            next_num = _safe_parse_residue_number(next_residue.id)
            residue_number_gap: Optional[int] = None
            if prev_num is not None and next_num is not None and next_num > prev_num + 1:
                residue_number_gap = next_num - prev_num - 1
                reasons.append('residue_number_gap')

            c_n_distance_a: Optional[float] = None
            if prev_c is not None and next_n is not None:
                c_pos = positions[prev_c.index].value_in_unit(nanometers)
                n_pos = positions[next_n.index].value_in_unit(nanometers)
                dist_nm = builtins.sum([(a - b) ** 2 for a, b in zip(c_pos, n_pos)]) ** 0.5
                c_n_distance_a = round(dist_nm * 10.0, 3)
                if dist_nm > threshold_nm:
                    reasons.append('distance')

            if reasons:
                break_records.append({
                    'original_chain_index': chain_idx,
                    'original_chain_id': chain.id,
                    'split_before_residue_index': next_index,
                    'previous_residue': _residue_label(prev_residue),
                    'next_residue': _residue_label(next_residue),
                    'reasons': reasons,
                    'missing_backbone_atoms': missing_backbone_atoms,
                    'residue_number_gap': residue_number_gap,
                    'c_n_distance_angstrom': c_n_distance_a,
                })

    return break_records


def _build_split_topology(
    topology: Any,
    positions: Any,
    break_records: List[Dict[str, Any]],
) -> Tuple[Any, Any, List[List[Dict[str, Any]]]]:
    """Split a topology in-memory at detected break points."""
    from openmm.app import Topology

    split_points_by_chain: Dict[int, Set[int]] = {}
    for record in break_records:
        split_points_by_chain.setdefault(record['original_chain_index'], set()).add(record['split_before_residue_index'])

    new_top = Topology()
    new_positions: List[Any] = []
    atom_map: Dict[Any, Any] = {}
    residue_to_chain_index: Dict[Any, int] = {}
    chain_segments: List[List[Dict[str, Any]]] = []
    next_chain_index = 0

    for chain_idx, chain in enumerate(topology.chains()):
        residues = list(chain.residues())
        split_points = sorted(split_points_by_chain.get(chain_idx, set()))
        segment_starts = [0] + split_points
        segment_ends = split_points + [len(residues)]
        segments: List[Dict[str, Any]] = []

        for frag_idx, (start, end) in enumerate(zip(segment_starts, segment_ends)):
            new_chain_id = chain.id if frag_idx == 0 else f'{chain.id}:{frag_idx+1}'
            new_chain = new_top.addChain(new_chain_id)
            segment = {
                'original_chain_index': chain_idx,
                'original_chain_id': chain.id,
                'new_chain_index': next_chain_index,
                'new_chain_id': new_chain_id,
                'start_residue_index': start,
                'end_residue_index': end,
                'length': end - start,
            }
            next_chain_index += 1
            segments.append(segment)

            for res_idx in range(start, end):
                residue = residues[res_idx]
                new_residue = new_top.addResidue(residue.name, new_chain, residue.id, residue.insertionCode)
                residue_to_chain_index[residue] = segment['new_chain_index']
                for atom in residue.atoms():
                    new_atom = new_top.addAtom(atom.name, atom.element, new_residue, atom.id)
                    atom_map[atom] = new_atom
                    new_positions.append(positions[atom.index])

        chain_segments.append(segments)

    for atom1, atom2 in topology.bonds():
        mapped1 = atom_map.get(atom1)
        mapped2 = atom_map.get(atom2)
        if mapped1 is None or mapped2 is None:
            continue
        if residue_to_chain_index.get(atom1.residue) != residue_to_chain_index.get(atom2.residue):
            continue
        new_top.addBond(mapped1, mapped2)

    if topology.getPeriodicBoxVectors() is not None:
        new_top.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())

    return new_top, new_positions, chain_segments


def _ensure_positive_unit_cell(topology: Any, positions: Any, padding_nm: float = 1.0) -> None:
    """Ensure the topology has a nonzero unit cell before PDBFixer operations."""
    dimensions = topology.getUnitCellDimensions()
    if dimensions is not None:
        dims_nm = dimensions.value_in_unit(nanometers)
        if all(value > 0 for value in dims_nm):
            return

    coords_nm = [pos.value_in_unit(nanometers) for pos in positions]
    if not coords_nm:
        return

    mins = [min(coord[i] for coord in coords_nm) for i in range(3)]
    maxs = [max(coord[i] for coord in coords_nm) for i in range(3)]
    spans = [max(maxs[i] - mins[i], 0.1) + padding_nm for i in range(3)]
    topology.setUnitCellDimensions(Vec3(*spans) * nanometers)


def _remap_missing_residues(
    missing_residues: Dict[Tuple[int, int], List[str]],
    chain_lengths: Dict[int, int],
    chain_segments: List[List[Dict[str, Any]]],
) -> Tuple[Dict[Tuple[int, int], List[str]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Keep only terminal missing residues and remap them to the split topology."""
    adjusted: Dict[Tuple[int, int], List[str]] = {}
    removed_internal: List[Dict[str, Any]] = []
    retained_terminal: List[Dict[str, Any]] = []

    for (chain_idx, insertion_idx), residue_names in missing_residues.items():
        entry = {
            'original_chain_index': chain_idx,
            'original_insertion_index': insertion_idx,
            'residues': list(residue_names),
        }
        chain_length = chain_lengths[chain_idx]
        if insertion_idx not in (0, chain_length):
            removed_internal.append(entry)
            continue

        if insertion_idx == 0:
            target_chain = chain_segments[chain_idx][0]
            new_key = (target_chain['new_chain_index'], 0)
        else:
            target_chain = chain_segments[chain_idx][-1]
            new_key = (target_chain['new_chain_index'], target_chain['length'])

        adjusted[new_key] = list(residue_names)
        retained_terminal.append({
            **entry,
            'new_chain_index': new_key[0],
            'new_insertion_index': new_key[1],
        })

    return adjusted, removed_internal, retained_terminal


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
    handle_chain_breaks: bool = True,
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
            import rdkit.Chem as Chem
            if pocket_ligand_sdf.endswith('.gz'):
                import gzip
                with gzip.open(pocket_ligand_sdf, 'rt') as f:
                    mol = Chem.MolFromMolBlock(f.read(), removeHs=True)
            else:
                supplier = Chem.SDMolSupplier(pocket_ligand_sdf, removeHs=True)
                mol = supplier[0]
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

    # 5. PDBFixer: load and find missing residues
    print('  Loading receptor into PDBFixer...', file=sys.stderr)
    fixer = PDBFixer(filename=reduced_path)
    original_chain_lengths = {
        chain.index: len(list(chain.residues()))
        for chain in fixer.topology.chains()
    }
    fixer.findMissingResidues()
    original_missing_residues = {
        key: list(value)
        for key, value in fixer.missingResidues.items()
    }
    print('PROGRESS:prepare_receptor:35', flush=True)

    # 6. Chain break detection -> split topology -> remap missing residues
    #    (needed for MD to prevent PDBFixer from modelling long internal loops;
    #     skipped for docking where the Sage force field rejects split-chain terminals)
    break_records: List[Dict[str, Any]] = []
    removed_internal: List[Dict[str, Any]] = []
    retained_terminal: List[Dict[str, Any]] = []

    if handle_chain_breaks:
        print('  Detecting chain breaks...', file=sys.stderr)
        break_records = _detect_chain_breaks(fixer.topology, fixer.positions)
        split_topology, split_positions, chain_segments = _build_split_topology(
            fixer.topology, fixer.positions, break_records,
        )
        adjusted_missing, removed_internal, retained_terminal = _remap_missing_residues(
            original_missing_residues, original_chain_lengths, chain_segments,
        )

        fixer.topology = split_topology
        fixer.positions = split_positions
        fixer.missingResidues = adjusted_missing

        if break_records:
            print(f'  Detected {len(break_records)} internal chain break(s)', file=sys.stderr)
        if removed_internal:
            print(f'  Removing {len(removed_internal)} internal gap(s) from missingResidues', file=sys.stderr)
        if retained_terminal:
            print(f'  Retaining {len(retained_terminal)} terminal missing-residue segment(s)', file=sys.stderr)

        _ensure_positive_unit_cell(fixer.topology, fixer.positions)

    print('PROGRESS:prepare_receptor:50', flush=True)

    # 7. Find and add missing atoms
    print('  Adding missing atoms...', file=sys.stderr)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    print('PROGRESS:prepare_receptor:60', flush=True)

    # 8. Sanitize positions (fix PDBFixer nested-Quantity bug that causes
    #    AssertionError in addHydrogens)
    positions = _sanitize_positions(fixer.positions)

    # 9. Build variant plan and protonate
    print('  Building protonation variant plan...', file=sys.stderr)
    variant_plan = build_variant_plan(
        fixer.topology,
        positions,
        ph,
        pocket_residue_keys=pocket_residue_keys,
        shifted_residues=propka_report.get('shifted_residues', []),
    )
    print('PROGRESS:prepare_receptor:70', flush=True)

    print('  Adding hydrogens...', file=sys.stderr)
    protonated_topology, protonated_positions, actual_variants = add_hydrogens_with_variants(
        fixer.topology,
        positions,
        ph,
        variant_plan['variants'],
    )
    print('PROGRESS:prepare_receptor:85', flush=True)

    # 10. Write output
    print('  Writing prepared receptor...', file=sys.stderr)
    write_prepared_receptor_pdb(
        protonated_topology,
        protonated_positions,
        output_pdb,
        variant_plan['resolved_variants'],
    )

    metadata: Dict[str, Any] = {
        'schema_version': 2,
        'prepared_receptor_pdb': output_pdb,
        'input_path': input_path,
        'receptor_protonation_ph': ph,
        'pocket_filtered': pocket_residue_keys is not None,
        'pocket_cutoff_angstrom': POCKET_RESIDUE_CUTOFF_A,
        'pocket_residue_keys': sorted(pocket_residue_keys) if pocket_residue_keys else [],
        'reduce_available': reduce_report['reduce_available'],
        'reduce_applied': reduce_report['reduce_applied'],
        'propka_available': propka_report.get('propka_available', False),
        'propka_error': propka_report.get('propka_error'),
        'propka_shifted_residues': propka_report.get('shifted_residues', []),
        'applied_overrides': variant_plan['applied_overrides'],
        'ignored_shifted_residues': variant_plan['ignored_shifted_residues'],
        'resolved_variants': variant_plan['resolved_variants'],
        'actual_variants': actual_variants,
        'disulfide_residue_keys': variant_plan['disulfide_residue_keys'],
        'detected_chain_breaks': break_records,
        'removed_internal_missing_residues': removed_internal,
        'retained_terminal_missing_residues': retained_terminal,
        'prepared_chain_count': builtins.sum(1 for _ in protonated_topology.chains()),
        'prepared_residue_count': builtins.sum(1 for _ in protonated_topology.residues()),
        'prepared_atom_count': protonated_topology.getNumAtoms(),
    }
    metadata['metadata_path'] = write_receptor_prep_metadata(output_pdb, metadata)
    print('PROGRESS:prepare_receptor:100', flush=True)
    print(f'  Receptor prepared: {protonated_topology.getNumAtoms()} atoms', file=sys.stderr)

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
