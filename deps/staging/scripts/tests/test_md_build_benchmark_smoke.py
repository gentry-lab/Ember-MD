#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""Smoke test for MD system build, benchmark, and production startup."""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

from openmm import LangevinMiddleIntegrator, Platform
from openmm.app import Simulation
from openmm.unit import kelvin, picosecond, picoseconds
from rdkit import Chem
from rdkit.Chem import AllChem

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
FIXTURE_DIR = SCRIPT_DIR / 'md_receptor_fixtures'

sys.path.insert(0, str(SCRIPTS_ROOT))

import run_md_simulation as md  # noqa: E402


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


def _create_production_simulation(system, modeller):
    integrator = LangevinMiddleIntegrator(50 * kelvin, 1 / picosecond, 0.002 * picoseconds)

    try:
        platform = Platform.getPlatformByName('CUDA')
        properties = {'CudaPrecision': 'mixed'}
        simulation = Simulation(modeller.topology, system, integrator, platform, properties)
        platform_name = 'CUDA'
    except Exception:
        try:
            platform = Platform.getPlatformByName('OpenCL')
            simulation = Simulation(modeller.topology, system, integrator, platform)
            platform_name = 'OpenCL'
        except Exception:
            try:
                platform = Platform.getPlatformByName('Metal')
                simulation = Simulation(modeller.topology, system, integrator, platform)
                platform_name = 'Metal'
            except Exception:
                platform = Platform.getPlatformByName('CPU')
                simulation = Simulation(modeller.topology, system, integrator, platform)
                platform_name = 'CPU'

    simulation.context.setPositions(modeller.positions)
    energy = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        md.kilocalories_per_mole
    )
    if not math.isfinite(energy):
        raise AssertionError(f'Production startup energy must be finite, got {energy!r}')
    return platform_name, energy


def main() -> int:
    receptor_pdb = FIXTURE_DIR / 'healthy_contiguous_prepared.pdb'
    if not receptor_pdb.exists():
        raise FileNotFoundError(f'Missing fixture: {receptor_pdb}')

    with tempfile.TemporaryDirectory(prefix='md_build_benchmark_smoke_') as tmpdir:
        tmpdir_path = Path(tmpdir)
        ligand_sdf = tmpdir_path / 'smoke_ligand.sdf'
        _write_smoke_ligand(ligand_sdf)

        system, modeller, ff, job_name = md.build_system(
            str(receptor_pdb),
            str(ligand_sdf),
            str(tmpdir_path),
            force_field_preset='ff19sb-opc',
            padding_nm=2.0,
        )
        system_pdbs = sorted(tmpdir_path.glob('*system.pdb'))
        if len(system_pdbs) != 1:
            raise AssertionError(f'Expected exactly one system.pdb output, found {len(system_pdbs)}')
        system_pdb = system_pdbs[0]

        ns_per_day = md.run_benchmark(system, modeller, str(tmpdir_path))
        if not (ns_per_day > 0):
            raise AssertionError(f'Benchmark must report positive ns/day, got {ns_per_day!r}')

        platform_name, energy = _create_production_simulation(system, modeller)
        print(f'PASS build_system created {system_pdb.name}')
        print(f'PASS benchmark ns/day = {ns_per_day:.1f}')
        print(f'PASS production startup on {platform_name} with finite PE = {energy:.1f} kcal/mol')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
