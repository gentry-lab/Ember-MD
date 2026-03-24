#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""Direct regression runner for MD receptor preparation/preflight."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

from rdkit import Chem
from rdkit.Chem import AllChem

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
FIXTURE_DIR = SCRIPT_DIR / 'md_receptor_fixtures'

sys.path.insert(0, str(SCRIPTS_ROOT))

import run_md_simulation as md  # noqa: E402


FIXTURES: Dict[str, Dict[str, Any]] = {
    'healthy_contiguous.pdb': {
        'expect': 'pass',
        'min_breaks': 0,
        'min_removed_internal': 0,
        'min_retained_terminal': 0,
    },
    'internal_break_no_seqres.pdb': {
        'expect': 'fail',
        'min_breaks': 1,
        'min_removed_internal': 0,
        'min_retained_terminal': 0,
    },
    'internal_gap_with_seqres.pdb': {
        'expect': 'fail',
        'min_breaks': 1,
        'min_removed_internal': 1,
        'min_retained_terminal': 0,
    },
    'terminal_truncation_with_seqres.pdb': {
        'expect': 'pass',
        'min_breaks': 0,
        'min_removed_internal': 0,
        'min_retained_terminal': 1,
    },
    'missing_backbone_adjacent_break.pdb': {
        'expect': 'fail',
        'min_breaks': 1,
        'min_removed_internal': 0,
        'min_retained_terminal': 0,
    },
}


REQUIRED_DEBUG_BUNDLE_FILES = (
    'input_receptor.pdb',
    'prepared_receptor.pdb',
    'build_report.json',
    'template_failure.txt',
)


def _write_smoke_ligand(sdf_path: Path) -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles('CCO'))
    if mol is None:
        raise RuntimeError('Failed to create smoke-test ligand')
    status = AllChem.EmbedMolecule(mol, randomSeed=42)
    if status != 0:
        raise RuntimeError('Failed to embed smoke-test ligand')
    AllChem.UFFOptimizeMolecule(mol)
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()


def _assert_report_counts(pdb_path: Path, report: Dict[str, Any], expected: Dict[str, Any]) -> None:
    if len(report['detected_chain_breaks']) < expected['min_breaks']:
        raise AssertionError(
            f'{pdb_path.name}: expected at least {expected["min_breaks"]} detected break(s), '
            f'got {len(report["detected_chain_breaks"])}'
        )
    if len(report['removed_internal_missing_residues']) < expected['min_removed_internal']:
        raise AssertionError(
            f'{pdb_path.name}: expected at least {expected["min_removed_internal"]} removed internal gap(s), '
            f'got {len(report["removed_internal_missing_residues"])}'
        )
    if len(report['retained_terminal_missing_residues']) < expected['min_retained_terminal']:
        raise AssertionError(
            f'{pdb_path.name}: expected at least {expected["min_retained_terminal"]} retained terminal gap(s), '
            f'got {len(report["retained_terminal_missing_residues"])}'
        )


def _assert_fixture_passes(pdb_path: Path, expected: Dict[str, Any]) -> None:
    topology, positions, report = md._prepare_receptor_topology(str(pdb_path))
    md._preflight_receptor_topology(topology, md.PRESETS['ff19sb-opc'])
    _assert_report_counts(pdb_path, report, expected)


def _assert_fixture_fails_with_debug_bundle(pdb_path: Path, expected: Dict[str, Any]) -> None:
    topology, positions, report = md._prepare_receptor_topology(str(pdb_path))
    _assert_report_counts(pdb_path, report, expected)

    with tempfile.TemporaryDirectory(prefix='md_preflight_fail_') as tmpdir:
        tmpdir_path = Path(tmpdir)
        ligand_sdf = tmpdir_path / 'smoke_ligand.sdf'
        _write_smoke_ligand(ligand_sdf)

        try:
            md.build_system(
                str(pdb_path),
                str(ligand_sdf),
                str(tmpdir_path),
                force_field_preset='ff19sb-opc',
            )
        except ValueError as exc:
            message = str(exc)
            if 'Receptor topology preflight failed:' not in message:
                raise AssertionError(f'{pdb_path.name}: unexpected failure message: {message}') from exc
        else:
            raise AssertionError(f'{pdb_path.name}: expected receptor preflight failure, but build_system succeeded')

        debug_dir = tmpdir_path / 'receptor_debug'
        missing = [
            file_name
            for file_name in REQUIRED_DEBUG_BUNDLE_FILES
            if not (debug_dir / file_name).exists()
        ]
        if missing:
            raise AssertionError(f'{pdb_path.name}: missing debug bundle files: {missing}')

        report_data = json.loads((debug_dir / 'build_report.json').read_text())
        if report_data.get('error_class') != 'receptor_topology':
            raise AssertionError(f'{pdb_path.name}: expected error_class=receptor_topology, got {report_data.get("error_class")!r}')
        if report_data.get('preflight_passed') is not False:
            raise AssertionError(f'{pdb_path.name}: expected preflight_passed=false in debug bundle')
        if 'first_unmatched_residue' not in report_data:
            raise AssertionError(f'{pdb_path.name}: missing first_unmatched_residue in debug bundle')


def _run_build_smoke(healthy_pdb: Path) -> None:
    with tempfile.TemporaryDirectory(prefix='md_build_smoke_') as tmpdir:
        tmpdir_path = Path(tmpdir)
        ligand_sdf = tmpdir_path / 'smoke_ligand.sdf'
        _write_smoke_ligand(ligand_sdf)

        md.build_system(
            str(healthy_pdb),
            str(ligand_sdf),
            str(tmpdir_path),
            force_field_preset='ff19sb-opc',
        )
        if not (tmpdir_path / 'system.pdb').exists():
            raise AssertionError('Protein-ligand smoke build did not create system.pdb')


def _run_ligand_only_smoke() -> None:
    with tempfile.TemporaryDirectory(prefix='md_ligand_only_smoke_') as tmpdir:
        tmpdir_path = Path(tmpdir)
        ligand_sdf = tmpdir_path / 'ligand_only_smoke.sdf'
        _write_smoke_ligand(ligand_sdf)

        md.build_ligand_only_system(
            str(ligand_sdf),
            str(tmpdir_path),
            force_field_preset='ff19sb-opc',
        )
        if not (tmpdir_path / 'system.pdb').exists():
            raise AssertionError('Ligand-only smoke build did not create system.pdb')


def main() -> int:
    for fixture_name, expected in FIXTURES.items():
        pdb_path = FIXTURE_DIR / fixture_name
        if not pdb_path.exists():
            raise FileNotFoundError(f'Missing fixture: {pdb_path}')

        if expected['expect'] == 'pass':
            _assert_fixture_passes(pdb_path, expected)
            print(f'PASS {fixture_name}')
        elif expected['expect'] == 'fail':
            _assert_fixture_fails_with_debug_bundle(pdb_path, expected)
            print(f'PASS {fixture_name} (expected preflight failure)')
        else:
            raise AssertionError(f'Unsupported expectation type: {expected["expect"]}')

    healthy_fixture = FIXTURE_DIR / 'healthy_contiguous.pdb'
    _run_build_smoke(healthy_fixture)
    print('PASS protein-ligand smoke build')

    _run_ligand_only_smoke()
    print('PASS ligand-only smoke build')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
