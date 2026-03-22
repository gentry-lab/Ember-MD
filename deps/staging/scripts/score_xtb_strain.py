#!/usr/bin/env python3
"""
GFN2-xTB energy utility for Ember.

Computes single-point energies, geometry optimizations, and ligand strain
energies using the GFN2-xTB semiempirical method with ALPB solvation.

Modes:
  single_point  — compute energy at fixed geometry
  optimize      — geometry optimization, write optimized SDF
  strain        — pose strain = E(pose) - E(free minimum)

Output lines (parsed by electron/main.ts):
  XTB_SP_ENERGY:<hartree>
  XTB_OPT_ENERGY:<hartree>
  XTB_STRAIN:<kcal/mol>
  XTB_OPTIMIZED_SDF:<path>
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

HARTREE_TO_KCAL = 627.5095


# ---------------------------------------------------------------------------
# Molecule I/O helpers
# ---------------------------------------------------------------------------

def _mol_to_xyz(sdf_path: str, output_xyz: str) -> int:
    """Convert an SDF file to XYZ format using RDKit. Returns atom count."""
    from rdkit import Chem

    mol = _load_sdf(sdf_path)
    if mol is None:
        raise RuntimeError(f"Failed to load molecule from {sdf_path}")

    mol = Chem.AddHs(mol, addCoords=True)
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()

    with open(output_xyz, 'w') as f:
        f.write(f"{n_atoms}\n")
        f.write(f"from {Path(sdf_path).name}\n")
        for i in range(n_atoms):
            pos = conf.GetAtomPosition(i)
            symbol = mol.GetAtomWithIdx(i).GetSymbol()
            f.write(f"{symbol:2s}  {pos.x:14.8f}  {pos.y:14.8f}  {pos.z:14.8f}\n")

    return n_atoms


def _load_sdf(path: str):
    """Load the first molecule from an SDF (or .sdf.gz) file."""
    from utils import load_sdf
    return load_sdf(path)


def _xyz_to_sdf(xyz_path: str, template_sdf: str, output_sdf: str) -> None:
    """Write an SDF using bond orders from template_sdf and coordinates from xyz_path."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    template = _load_sdf(template_sdf)
    if template is None:
        raise RuntimeError(f"Failed to load template molecule from {template_sdf}")

    # Parse XYZ coordinates
    coords = []
    with open(xyz_path) as f:
        n_atoms = int(f.readline().strip())
        f.readline()  # comment
        for _ in range(n_atoms):
            parts = f.readline().split()
            coords.append((float(parts[1]), float(parts[2]), float(parts[3])))

    # The template may have fewer atoms (no H) or same count.
    # Add Hs to template to match xTB output which always has explicit H.
    tmol = Chem.AddHs(template, addCoords=True)

    if tmol.GetNumAtoms() != n_atoms:
        # Fallback: just write XYZ as-is without bond orders
        print(f"  Warning: atom count mismatch (template={tmol.GetNumAtoms()}, "
              f"xtb={n_atoms}), writing XYZ-based SDF", file=sys.stderr)
        # Use RDKit's xyz reader
        raw = Chem.MolFromXYZFile(xyz_path)
        if raw is not None:
            writer = Chem.SDWriter(output_sdf)
            writer.write(raw)
            writer.close()
        return

    # Overlay xTB-optimized coordinates onto template
    conf = tmol.GetConformer()
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (x, y, z))

    writer = Chem.SDWriter(output_sdf)
    writer.write(tmol)
    writer.close()


# ---------------------------------------------------------------------------
# xTB runner
# ---------------------------------------------------------------------------

def _resolve_xtb_env(xtb_binary: str) -> dict:
    """Build environment for xTB subprocess."""
    env = os.environ.copy()
    xtb_bin_dir = str(Path(xtb_binary).parent)
    xtb_root = str(Path(xtb_bin_dir).parent)

    # XTBPATH must point to the parameter files
    xtb_share = os.path.join(xtb_root, 'share', 'xtb')
    if os.path.isdir(xtb_share):
        env['XTBPATH'] = xtb_share

    # Sensible threading defaults
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('OMP_STACKSIZE', '1G')
    env.setdefault('MKL_NUM_THREADS', '1')

    return env


def _parse_total_energy(stdout: str) -> Optional[float]:
    """Extract TOTAL ENERGY (Hartree) from xTB stdout."""
    # Pattern: | TOTAL ENERGY              -42.567890123 Eh   |
    match = re.search(r'TOTAL ENERGY\s+([-\d.]+)\s+Eh', stdout)
    if match:
        return float(match.group(1))
    return None


def run_xtb(
    xtb_binary: str,
    xyz_path: str,
    mode: str = 'sp',
    solvent: str = 'water',
    work_dir: Optional[str] = None,
) -> Tuple[float, Optional[str]]:
    """Run xTB and return (energy_hartree, optimized_xyz_path_or_none).

    Args:
        xtb_binary: Path to xtb executable.
        xyz_path: Input structure in XYZ format.
        mode: 'sp' for single-point, 'opt' for optimization.
        solvent: ALPB solvent name (e.g. 'water') or empty to skip.
        work_dir: Working directory for xTB output files.

    Returns:
        Tuple of (energy in Hartree, path to xtbopt.xyz if optimization).
    """
    cmd = [xtb_binary, xyz_path, '--gfn2']

    if mode == 'opt':
        cmd.append('--opt')
    else:
        cmd.append('--sp')

    if solvent:
        cmd.extend(['--alpb', solvent])

    env = _resolve_xtb_env(xtb_binary)
    cwd = work_dir or os.path.dirname(xyz_path)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
        timeout=300,
    )

    if result.returncode != 0:
        # Print stderr for debugging but don't fail on warnings
        print(f"  xTB stderr: {result.stderr[:500]}", file=sys.stderr)
        if 'convergence' not in result.stderr.lower():
            raise RuntimeError(
                f"xTB failed (exit {result.returncode}): {result.stderr[:300]}"
            )

    energy = _parse_total_energy(result.stdout)
    if energy is None:
        raise RuntimeError(f"Could not parse energy from xTB output:\n{result.stdout[:500]}")

    opt_xyz = None
    if mode == 'opt':
        candidate = os.path.join(cwd, 'xtbopt.xyz')
        if os.path.exists(candidate):
            opt_xyz = candidate

    return energy, opt_xyz


# ---------------------------------------------------------------------------
# High-level modes
# ---------------------------------------------------------------------------

def single_point(xtb_binary: str, ligand_sdf: str, solvent: str = 'water') -> float:
    """Compute GFN2-xTB single-point energy. Returns energy in Hartree."""
    with tempfile.TemporaryDirectory(prefix='xtb_sp_') as tmpdir:
        xyz = os.path.join(tmpdir, 'input.xyz')
        _mol_to_xyz(ligand_sdf, xyz)
        energy, _ = run_xtb(xtb_binary, xyz, mode='sp', solvent=solvent, work_dir=tmpdir)
        return energy


def optimize(
    xtb_binary: str,
    ligand_sdf: str,
    output_sdf: Optional[str] = None,
    solvent: str = 'water',
) -> Tuple[float, str]:
    """Optimize geometry with GFN2-xTB. Returns (energy_hartree, output_sdf_path)."""
    with tempfile.TemporaryDirectory(prefix='xtb_opt_') as tmpdir:
        xyz = os.path.join(tmpdir, 'input.xyz')
        _mol_to_xyz(ligand_sdf, xyz)
        energy, opt_xyz = run_xtb(xtb_binary, xyz, mode='opt', solvent=solvent, work_dir=tmpdir)

        if opt_xyz is None:
            raise RuntimeError("xTB optimization did not produce xtbopt.xyz")

        if output_sdf is None:
            output_sdf = ligand_sdf.replace('.sdf', '_xtbopt.sdf')

        _xyz_to_sdf(opt_xyz, ligand_sdf, output_sdf)
        return energy, output_sdf


def strain(
    xtb_binary: str,
    pose_sdf: str,
    reference_sdf: Optional[str] = None,
    solvent: str = 'water',
) -> Tuple[float, float, float]:
    """Compute ligand strain energy.

    If reference_sdf is provided, uses it as the pre-optimized free ligand.
    Otherwise optimizes the pose geometry to find the free minimum.

    Returns:
        (strain_kcal, pose_energy_hartree, reference_energy_hartree)
    """
    # Pose single-point
    e_pose = single_point(xtb_binary, pose_sdf, solvent)

    # Reference: either provided or optimize
    if reference_sdf:
        e_ref = single_point(xtb_binary, reference_sdf, solvent)
    else:
        e_ref, _ = optimize(xtb_binary, pose_sdf, solvent=solvent)

    strain_kcal = (e_pose - e_ref) * HARTREE_TO_KCAL
    return strain_kcal, e_pose, e_ref


def batch_strain(
    xtb_binary: str,
    ligand_paths: list,
    reference_sdf: Optional[str] = None,
    solvent: str = 'water',
) -> dict:
    """Compute strain energy for multiple ligands in a single Python invocation.

    Each unique molecule (by canonical SMILES) gets its own optimized
    free-minimum reference.  Strain = E(pose) - E(free_min) per molecule.

    Returns dict mapping basename → strain_kcal.
    """
    from rdkit import Chem

    ref_cache: dict = {}  # canonical SMILES -> e_ref (hartree)

    results = {}
    for i, lig_path in enumerate(ligand_paths):
        name = os.path.basename(lig_path).replace('_docked.sdf.gz', '').replace('_docked.sdf', '').replace('.sdf.gz', '').replace('.sdf', '')
        try:
            mol = _load_sdf(lig_path)
            if mol is None:
                print(f"Warning: Could not load {name}, skipping", file=sys.stderr)
                continue
            smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))

            if smiles not in ref_cache:
                e_ref, _ = optimize(xtb_binary, lig_path, solvent=solvent)
                ref_cache[smiles] = e_ref
                print(f"  Reference for {smiles[:60]}: {e_ref:.6f} Eh", file=sys.stderr)

            e_pose = single_point(xtb_binary, lig_path, solvent)
            strain_kcal = (e_pose - ref_cache[smiles]) * HARTREE_TO_KCAL
            results[f"{name}_0"] = round(strain_kcal, 1)
            print(f"PROGRESS:batch_strain:{int(100 * (i + 1) / len(ligand_paths))}", flush=True)
        except Exception as e:
            print(f"Warning: xTB failed for {name}: {e}", file=sys.stderr)

    return results


def batch_energy(
    xtb_binary: str,
    ligand_paths: list,
    solvent: str = 'water',
) -> dict:
    """Compute relative xTB single-point energies grouped by compound.

    No optimization — just single-point each pose.  Energies are relative
    to the lowest-energy pose of the same compound (by canonical SMILES).
    Useful for pose ranking within docking results.

    Returns dict mapping basename → relative_energy_kcal.
    """
    from rdkit import Chem

    # Phase 1: single-point all poses, group by SMILES
    by_smiles: dict = {}  # SMILES -> [(name, energy_hartree)]
    name_map: dict = {}   # name -> SMILES

    for i, lig_path in enumerate(ligand_paths):
        name = os.path.basename(lig_path).replace('_docked.sdf.gz', '').replace('_docked.sdf', '').replace('.sdf.gz', '').replace('.sdf', '')
        try:
            mol = _load_sdf(lig_path)
            if mol is None:
                print(f"Warning: Could not load {name}, skipping", file=sys.stderr)
                continue
            smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
            e = single_point(xtb_binary, lig_path, solvent)
            by_smiles.setdefault(smiles, []).append((name, e))
            name_map[name] = smiles
            print(f"PROGRESS:batch_energy:{int(100 * (i + 1) / len(ligand_paths))}", flush=True)
        except Exception as exc:
            print(f"Warning: xTB failed for {name}: {exc}", file=sys.stderr)

    # Phase 2: compute relative energies per compound
    results = {}
    for smiles, entries in by_smiles.items():
        min_e = min(e for _, e in entries)
        for name, e in entries:
            rel_kcal = (e - min_e) * HARTREE_TO_KCAL
            results[f"{name}_0"] = round(rel_kcal, 1)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='GFN2-xTB energy utility for Ember'
    )
    parser.add_argument('--ligand', help='Input ligand SDF file (single modes)')
    parser.add_argument('--xtb_binary', required=True, help='Path to xtb executable')
    parser.add_argument(
        '--mode',
        choices=['single_point', 'optimize', 'strain', 'batch_strain', 'batch_energy'],
        default='single_point',
        help='Calculation mode',
    )
    parser.add_argument(
        '--reference_sdf',
        help='Pre-optimized reference SDF for strain mode (skip optimization)',
    )
    parser.add_argument(
        '--output_sdf',
        help='Output SDF path for optimize mode',
    )
    parser.add_argument(
        '--solvent',
        default='water',
        help='ALPB solvent (default: water, empty string for gas phase)',
    )
    parser.add_argument(
        '--ligand_dir',
        help='Directory of SDF files for batch_strain mode',
    )
    parser.add_argument(
        '--output_json',
        help='Output JSON path for batch_strain mode',
    )

    args = parser.parse_args()

    if not os.path.isfile(args.xtb_binary):
        print(f"Error: xTB binary not found at {args.xtb_binary}", file=sys.stderr)
        sys.exit(1)

    if args.mode == 'batch_strain':
        import json
        if not args.ligand_dir or not os.path.isdir(args.ligand_dir):
            print(f"Error: --ligand_dir required for batch_strain mode", file=sys.stderr)
            sys.exit(1)
        sdfs = sorted([
            os.path.join(args.ligand_dir, f)
            for f in os.listdir(args.ligand_dir)
            if f.endswith('.sdf') or f.endswith('.sdf.gz')
        ])
        if not sdfs:
            print("Error: No SDF files found in ligand_dir", file=sys.stderr)
            sys.exit(1)
        results = batch_strain(args.xtb_binary, sdfs, args.reference_sdf, args.solvent)
        output_path = args.output_json or os.path.join(args.ligand_dir, 'xtb_strain.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"BATCH_STRAIN_JSON:{output_path}")
        print(f"Scored {len(results)} ligands", file=sys.stderr)

    elif args.mode == 'batch_energy':
        import json
        if not args.ligand_dir or not os.path.isdir(args.ligand_dir):
            print(f"Error: --ligand_dir required for batch_energy mode", file=sys.stderr)
            sys.exit(1)
        sdfs = sorted([
            os.path.join(args.ligand_dir, f)
            for f in os.listdir(args.ligand_dir)
            if f.endswith('.sdf') or f.endswith('.sdf.gz')
        ])
        if not sdfs:
            print("Error: No SDF files found in ligand_dir", file=sys.stderr)
            sys.exit(1)
        results = batch_energy(args.xtb_binary, sdfs, args.solvent)
        output_path = args.output_json or os.path.join(args.ligand_dir, 'xtb_energy.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"BATCH_ENERGY_JSON:{output_path}")
        print(f"Scored {len(results)} poses", file=sys.stderr)

    elif args.mode == 'single_point':
        if not args.ligand or not os.path.isfile(args.ligand):
            print(f"Error: --ligand required", file=sys.stderr)
            sys.exit(1)
        energy = single_point(args.xtb_binary, args.ligand, args.solvent)
        print(f"XTB_SP_ENERGY:{energy:.10f}")

    elif args.mode == 'optimize':
        if not args.ligand or not os.path.isfile(args.ligand):
            print(f"Error: --ligand required", file=sys.stderr)
            sys.exit(1)
        energy, out_sdf = optimize(
            args.xtb_binary, args.ligand, args.output_sdf, args.solvent
        )
        print(f"XTB_OPT_ENERGY:{energy:.10f}")
        print(f"XTB_OPTIMIZED_SDF:{out_sdf}")

    elif args.mode == 'strain':
        if not args.ligand or not os.path.isfile(args.ligand):
            print(f"Error: --ligand required", file=sys.stderr)
            sys.exit(1)
        strain_kcal, e_pose, e_ref = strain(
            args.xtb_binary, args.ligand, args.reference_sdf, args.solvent
        )
        print(f"XTB_STRAIN:{strain_kcal:.4f}")
        print(f"XTB_SP_ENERGY:{e_pose:.10f}")
        print(f"XTB_OPT_ENERGY:{e_ref:.10f}")


if __name__ == '__main__':
    main()
