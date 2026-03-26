#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""Regression test for residue-template remapping during chain-break fallback."""

from __future__ import annotations

import sys
from pathlib import Path

from openmm.app import Topology

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPTS_ROOT))

import run_md_simulation as md  # noqa: E402


class _FakeForceField:
    def __init__(self) -> None:
        self.calls = 0

    def createSystem(self, topology, *args, **kwargs):
        self.calls += 1
        residue_templates = kwargs.get('residueTemplates', {})
        residues = list(topology.residues())
        first_residue = residues[0]
        second_residue = residues[1]

        if self.calls == 1:
            if kwargs.get('ignoreExternalBonds'):
                raise AssertionError('Initial createSystem call should not ignore external bonds')
            raise Exception(
                'No template found for residue 1 (ALA). '
                'The set of atoms matches NSER, but the bonds are different.'
            )

        if self.calls == 2:
            if not kwargs.get('ignoreExternalBonds'):
                raise AssertionError('Retry call should enable ignoreExternalBonds')
            if residue_templates:
                raise AssertionError('First retry should not have residue overrides yet')
            raise Exception(
                'No template found for residue 1 (ALA). '
                'The set of atoms matches NSER, but the bonds are different.'
            )

        if self.calls == 3:
            if residue_templates.get(first_residue) != 'NSER':
                raise AssertionError('Expected first residue to be remapped to NSER')
            if second_residue in residue_templates:
                raise AssertionError('Second residue must not receive the first residue template override')
            return 'ok'

        raise AssertionError(f'Unexpected createSystem call count: {self.calls}')


def _build_mock_topology() -> Topology:
    topology = Topology()
    chain = topology.addChain('A')
    topology.addResidue('ALA', chain, id='175')
    topology.addResidue('UNK', chain, id='176')
    return topology


def main() -> int:
    ff = _FakeForceField()
    md._patch_forcefield_for_chain_breaks(ff)
    result = ff.createSystem(_build_mock_topology())
    if result != 'ok':
        raise AssertionError(f'Expected patched createSystem to return ok, got {result!r}')
    print('PASS residue remap uses OpenMM 1-based residue numbering')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
