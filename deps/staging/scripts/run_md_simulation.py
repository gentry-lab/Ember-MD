#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
OpenMM MD simulation for FragGen GUI.
Builds system from GNINA docking output and runs simulation.

Force field presets:
  - ff14sb-tip3p: ff14SB + TIP3P (classic validated combination)
  - ff19sb-opc: ff19SB + OPC (modern, higher accuracy, default)
  - ff19sb-opc3: ff19SB + OPC3 (fast modern, nearly OPC accuracy)

AMBER-style equilibration protocol with positional restraints:
1. Restrained minimization (heavy atoms, 10 kcal/mol/A²)
2. Unrestrained minimization
3. NVT heating 10K→100K + NPT heating 100K→300K (backbone restraints, ~70ps)
4. NPT equilibration at 300K with backbone restraints (~50ps)
5. Gradual restraint release in NPT (~100ps)
6. Unrestrained NPT equilibration (~50ps)
7. Production with HMR (4fs timestep)

Key insight: AMBER recommends switching to NPT around 100K, not 300K.
Running NVT too long can create vacuum bubbles. Adding barostat abruptly
at 300K causes pressure instability.
"""

import argparse
import builtins
import functools
import json
import os
import re
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from prepare_receptor import ensure_pdb_format, prepare_receptor
from utils import load_sdf

try:
    from openmm import *
    from openmm.app import *
    from openmm.unit import *
    import rdkit.Chem as Chem
    from openff.toolkit import Molecule
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator, SystemGenerator
    import openmmforcefields
    from pdbfixer import PDBFixer
except ImportError as e:
    print(f"ERROR:Missing dependency: {e}", file=sys.stderr)
    print("Please install: conda install -c conda-forge openmm openmmforcefields openff-toolkit pdbfixer", file=sys.stderr)
    sys.exit(1)

# Load plugins (needed for Metal/HIP platform and OpenCL on bundled installs)
_default_plugins = Platform.getDefaultPluginsDirectory()
_bundled_plugins = os.path.join(os.path.dirname(os.path.realpath(sys.executable)), '..', 'lib', 'plugins')
for _pdir in [_default_plugins, _bundled_plugins]:
    _pdir = os.path.normpath(_pdir)
    if os.path.isdir(_pdir):
        try:
            Platform.loadPluginsFromDirectory(_pdir)
        except Exception:
            pass


# Force field presets
PRESETS = {
    'ff14sb-tip3p':    {'protein_ff': ['amber14-all.xml'],              'water_ff': ['amber14/tip3p.xml'],         'water_model': 'tip3p',   'water_label': 'TIP3P'},
    'ff19sb-opc':      {'protein_ff': ['amber/protein.ff19SB.xml'],     'water_ff': ['amber/opc_standard.xml'],    'water_model': 'tip4pew', 'water_label': 'OPC'},
    'ff19sb-opc3':     {'protein_ff': ['amber/protein.ff19SB.xml'],     'water_ff': ['amber/opc3_standard.xml'],   'water_model': 'tip3p',   'water_label': 'OPC3'},
}

# Backward compat mapping for old preset names
_PRESET_ALIASES = {'fast': 'ff14sb-tip3p', 'accurate': 'ff19sb-opc'}

# Force constant unit
KCAL_MOL_A2 = kilocalories_per_mole / angstroms**2

STANDARD_PROTEIN_RESIDUES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'HID', 'HIE', 'HIP', 'CYX', 'ASH', 'GLH', 'LYN', 'ACE', 'NME',
}

SOLVENT_AND_ION_RESIDUES = {
    'HOH', 'WAT', 'TIP3', 'TIP4', 'OPC', 'SPC', 'SOL',
    'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'NA+', 'CL-',
}


def _optimize_box(modeller, padding_nm=1.0):
    """Choose the smallest periodic box for the solute.

    OpenMM's addSolvent(padding=...) uses a bounding SPHERE for all box shapes:
      width = 2*sphere_radius + padding
    This wastes water for non-spherical proteins because the sphere must enclose
    the longest diagonal.

    Instead, we compare two strategies and pick the smaller:
      1. Dodecahedron with sphere-based sizing (OpenMM default) — best for
         globular proteins where sphere_radius ≈ max_extent/2.
      2. Orthorhombic with per-axis sizing (Maestro's approach) — best for
         elongated proteins where per-axis padding avoids sphere overhead.

    For strategy 2, principal-axis rotation minimizes the AABB volume.
    Returns (kwargs_for_addSolvent, description_string).
    """
    import numpy as np

    positions = modeller.positions
    coords = np.array(positions.value_in_unit(nanometers))  # Nx3, nm

    # --- Strategy 1: dodecahedron (sphere-based) ---
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    sphere_radius = np.linalg.norm(centered, axis=1).max()
    dodec_width = 2 * sphere_radius + padding_nm
    dodec_vol = dodec_width**3 * (2**0.5 / 2)  # nm³

    # --- Strategy 2: orthorhombic with PCA rotation ---
    # Inertia tensor for PCA (unit masses)
    c = centered
    Ixx = np.sum(c[:, 1]**2 + c[:, 2]**2)
    Iyy = np.sum(c[:, 0]**2 + c[:, 2]**2)
    Izz = np.sum(c[:, 0]**2 + c[:, 1]**2)
    Ixy = -np.sum(c[:, 0] * c[:, 1])
    Ixz = -np.sum(c[:, 0] * c[:, 2])
    Iyz = -np.sum(c[:, 1] * c[:, 2])
    inertia = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    _, evecs = np.linalg.eigh(inertia)
    if np.linalg.det(evecs) < 0:
        evecs[:, 0] *= -1

    rotated = centered @ evecs
    rot_extent = rotated.max(axis=0) - rotated.min(axis=0)  # nm
    ortho_box = rot_extent + 2 * padding_nm  # per-axis padding
    ortho_vol = ortho_box[0] * ortho_box[1] * ortho_box[2]  # nm³

    extent_A = rot_extent * 10
    sphere_r_A = sphere_radius * 10

    if dodec_vol <= ortho_vol:
        # Use dodecahedron — center solute and let OpenMM size via sphere
        modeller.positions = [Vec3(*r) for r in centered] * nanometers
        desc = (f'dodecahedron (sphere r={sphere_r_A:.0f} A, '
                f'extent {extent_A[0]:.0f}x{extent_A[1]:.0f}x{extent_A[2]:.0f})')
        kwargs = dict(boxShape='dodecahedron', padding=padding_nm * nanometers)
    else:
        # Use orthorhombic — rotate + per-axis box size
        modeller.positions = [Vec3(*r) for r in rotated] * nanometers
        box_A = ortho_box * 10
        desc = (f'orthorhombic {box_A[0]:.0f}x{box_A[1]:.0f}x{box_A[2]:.0f} A '
                f'({(1 - ortho_vol / dodec_vol) * 100:.0f}% smaller than dodecahedron)')
        kwargs = dict(boxSize=Vec3(*ortho_box) * nanometers)

    vol_A3 = min(dodec_vol, ortho_vol) * 1000
    print(f'  Box optimization: {desc}, est. {vol_A3:.0f} A^3', file=sys.stderr)
    return kwargs


class ReceptorPreparationError(RuntimeError):
    """Raised when receptor preparation or validation fails."""

    def __init__(
        self,
        message: str,
        report: Dict[str, Any],
        topology: Optional[Any] = None,
        positions: Optional[Any] = None,
    ) -> None:
        super().__init__(message)
        self.report = report
        self.topology = topology
        self.positions = positions



def _prepare_receptor_topology(receptor_pdb: str) -> Tuple[Any, Any, Dict[str, Any]]:
    """Prepare a receptor topology for MD and return a structured report.

    Checks for an existing receptor_prepared.pdb alongside the input.
    If not found, runs the unified prepare_receptor pipeline.
    """
    output_dir = os.path.dirname(receptor_pdb) or '.'
    prepared_pdb_path = os.path.join(output_dir, 'receptor_prepared.pdb')
    prepared_json_path = os.path.join(output_dir, 'receptor_prepared.prep.json')

    report: Dict[str, Any] = {
        'input_receptor_pdb': receptor_pdb,
        'preflight_passed': False,
    }

    try:
        if os.path.exists(prepared_pdb_path) and os.path.exists(prepared_json_path):
            print('  Using existing prepared receptor', file=sys.stderr)
            pdb = PDBFile(prepared_pdb_path)
            with open(prepared_json_path) as f:
                report.update(json.load(f))
            return pdb.topology, pdb.positions, report

        _, prep_report = prepare_receptor(receptor_pdb, output_dir=output_dir)
        report.update(prep_report)
        pdb = PDBFile(prepared_pdb_path)
        return pdb.topology, pdb.positions, report
    except Exception as exc:
        raise ReceptorPreparationError(
            f'Receptor preparation failed: {exc}',
            report,
        ) from exc


def _extract_template_failure_details(error_text: str) -> Optional[Dict[str, Any]]:
    """Extract the first unmatched residue from an OpenMM template failure."""
    match = re.search(r'No template found for residue (\d+) \(([^)]+)\)\.\s*(.*)', error_text, re.S)
    if not match:
        return None
    return {
        'residue_index_1based': int(match.group(1)),
        'residue_name': match.group(2),
        'details': match.group(3).strip(),
    }


def _write_receptor_debug_bundle(
    output_dir: str,
    input_receptor_pdb: str,
    report: Dict[str, Any],
    error: Exception,
    prepared_topology: Optional[Any] = None,
    prepared_positions: Optional[Any] = None,
) -> None:
    """Write receptor-preflight debug artifacts for postmortem analysis."""
    debug_dir = os.path.join(output_dir, 'receptor_debug')
    os.makedirs(debug_dir, exist_ok=True)

    if os.path.exists(input_receptor_pdb):
        shutil.copyfile(input_receptor_pdb, os.path.join(debug_dir, 'input_receptor.pdb'))

    if prepared_topology is not None and prepared_positions is not None:
        with open(os.path.join(debug_dir, 'prepared_receptor.pdb'), 'w') as f:
            PDBFile.writeFile(prepared_topology, prepared_positions, f)

    with open(os.path.join(debug_dir, 'template_failure.txt'), 'w') as f:
        f.write(str(error))
        f.write('\n')

    with open(os.path.join(debug_dir, 'build_report.json'), 'w') as f:
        json.dump(report, f, indent=2, sort_keys=True)


def _raise_receptor_topology_preflight_failure(
    output_dir: str,
    receptor_pdb: str,
    report: Dict[str, Any],
    error: Exception,
    prepared_topology: Optional[Any] = None,
    prepared_positions: Optional[Any] = None,
) -> None:
    """Persist receptor-preflight diagnostics and raise a classified build error."""
    report['error_class'] = 'receptor_topology'
    report['preflight_passed'] = False

    template_details = _extract_template_failure_details(str(error))
    if template_details is not None:
        report['first_unmatched_residue'] = template_details

    _write_receptor_debug_bundle(
        output_dir,
        receptor_pdb,
        report,
        error,
        prepared_topology,
        prepared_positions,
    )
    raise ValueError(f'Receptor topology preflight failed: {error}') from error


def _preflight_receptor_topology(prepared_topology: Any, prepared_positions: Any, preset: Dict[str, Any]) -> None:
    """Quick validation that the prepared protein topology is broadly compatible.

    Non-fatal: chain breaks, missing virtual sites, and variant mismatches
    are handled by the full build_system pipeline downstream.
    """
    ff = ForceField(*preset['protein_ff'], *preset['water_ff'])
    modeller = Modeller(prepared_topology, prepared_positions)
    try:
        modeller.addExtraParticles(ff)
        ff.createSystem(
            modeller.topology,
            nonbondedMethod=NoCutoff,
            constraints=HBonds,
        )
    except Exception as exc:
        print(f'  Preflight warning (non-fatal): {exc}', file=sys.stderr)


def _patch_forcefield_for_chain_breaks(ff: Any) -> None:
    """Monkey-patch ForceField.createSystem to auto-fallback with ignoreExternalBonds.

    Chain-break receptors have terminal residues with bond mismatches.
    This patches createSystem so that addSolvent, addExtraParticles fallbacks,
    and direct createSystem calls all tolerate these mismatches.

    Also handles residue template mismatches (e.g. ALA atoms matching NOR template
    but with different bonds) by iteratively parsing errors and mapping residues
    to the closest matching template until all mismatches are resolved.
    """
    import re
    original_createSystem = ff.createSystem
    MAX_RETRIES = 50  # Safety cap — one retry per mismatched residue

    @functools.wraps(original_createSystem)
    def _patched_createSystem(topology, *create_args, **create_kwargs):
        kwargs = dict(create_kwargs)
        # Build base residue template overrides
        rt = dict(kwargs.pop('residueTemplates', {}))
        # CYS/CYX resolution by atom content
        for res in topology.residues():
            if res.name in ('CYS', 'CYX', 'CYM'):
                atom_names = {a.name for a in res.atoms()}
                rt[res] = 'CYS' if 'HG' in atom_names else 'CYX'

        # First attempt without ignoreExternalBonds
        try:
            return original_createSystem(topology, *create_args, **kwargs)
        except Exception:
            pass

        # Retry loop: each iteration fixes one mismatched residue
        kwargs['ignoreExternalBonds'] = True
        kwargs['residueTemplates'] = rt
        for attempt in range(MAX_RETRIES):
            try:
                return original_createSystem(topology, *create_args, **kwargs)
            except Exception as exc:
                err_msg = str(exc)
                match = re.search(
                    r'residue (\d+) \((\w+)\).*atoms matches (\w+)',
                    err_msg
                )
                if not match:
                    raise  # Not a template mismatch — can't fix
                res_index = int(match.group(1))
                res_name = match.group(2)
                matched_template = match.group(3)
                # Find and remap the residue
                remapped = False
                for res in topology.residues():
                    if res.index == res_index:
                        rt[res] = matched_template
                        print(f'  Remapping residue {res_index} ({res_name}) → {matched_template}', file=sys.stderr)
                        remapped = True
                        break
                if not remapped:
                    raise  # Couldn't find the residue
        raise RuntimeError(f'Too many residue template mismatches (>{MAX_RETRIES})')

    ff.createSystem = _patched_createSystem


def _estimate_am1bcc_time(n_atoms: int) -> str:
    """Estimate AM1-BCC charge computation time from atom count.

    Based on empirical profiling of AmberTools sqm:
      12 atoms → 0.3s, 21 → 4s, 33 → 10s, 52 → 395s, 59 → 264s
    Scaling is roughly O(N^3.5) but varies with molecular topology.
    Returns a human-readable string.
    """
    if n_atoms <= 20:
        return '< 10s'
    elif n_atoms <= 30:
        return '~10-30s'
    elif n_atoms <= 40:
        return '~30s-2min'
    elif n_atoms <= 50:
        return '~1-4min'
    elif n_atoms <= 65:
        return '~2-6min'
    else:
        return '~5min+'


def _build_mixed_system_generator(preset: Dict[str, Any], ligand: Any, include_protein: bool = True) -> Any:
    """Create a SystemGenerator for protein/water + OpenFF ligand systems."""
    forcefields = [*preset['water_ff']]
    if include_protein:
        forcefields = [*preset['protein_ff'], *forcefields]
    return SystemGenerator(
        forcefields=forcefields,
        small_molecule_forcefield='openff-2.3.0',
        molecules=[ligand],
        nonperiodic_forcefield_kwargs={
            'nonbondedMethod': NoCutoff,
        },
        periodic_forcefield_kwargs={
            'nonbondedMethod': PME,
            'nonbondedCutoff': 1.0*nanometers,
            'ewaldErrorTolerance': 0.0005,
            'constraints': HBonds,
            'hydrogenMass': 1.5*amu,
        },
    )


def get_backbone_atoms(topology: Any) -> List[int]:
    """Get indices of backbone atoms (N, CA, C, O) for restraints."""
    backbone_names = {'N', 'CA', 'C', 'O'}
    indices = []
    for atom in topology.atoms():
        if atom.name in backbone_names and atom.residue.name not in ('HOH', 'WAT', 'TIP3', 'TIP4', 'OPC'):
            indices.append(atom.index)
    return indices


def get_heavy_atoms(topology: Any) -> List[int]:
    """Get indices of all non-hydrogen, non-water atoms for restraints.

    This includes protein backbone+sidechain and ligand heavy atoms.
    """
    water_residues = {'HOH', 'WAT', 'TIP3', 'TIP4', 'OPC', 'NA', 'CL', 'Na+', 'Cl-'}
    indices = []
    for atom in topology.atoms():
        # Skip water and ions
        if atom.residue.name in water_residues:
            continue
        # Skip hydrogens
        if atom.element.symbol == 'H':
            continue
        indices.append(atom.index)
    return indices


def add_heavy_atom_restraints(system: Any, positions: Any, topology: Any, force_constant: float) -> Tuple[Any, List[int]]:
    """Add harmonic restraints to all heavy atoms (protein + ligand).

    Returns the restraint force and list of restrained atom indices.
    """
    heavy_indices = get_heavy_atoms(topology)

    # Create custom external force for position restraints
    restraint = CustomExternalForce('k_heavy*periodicdistance(x,y,z,x0,y0,z0)^2')
    restraint.addGlobalParameter('k_heavy', force_constant * KCAL_MOL_A2)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')

    for idx in heavy_indices:
        pos = positions[idx]
        # Handle both Vec3 and numpy array formats
        if hasattr(pos, 'x'):
            restraint.addParticle(idx, [pos.x, pos.y, pos.z])
        else:
            # Numpy array or list format
            restraint.addParticle(idx, [pos[0], pos[1], pos[2]])

    system.addForce(restraint)
    return restraint, heavy_indices


def add_backbone_restraints(system: Any, positions: Any, topology: Any, force_constant: float) -> Tuple[Any, List[int]]:
    """Add harmonic restraints to backbone atoms.

    Returns the restraint force and list of restrained atom indices.
    """
    backbone_indices = get_backbone_atoms(topology)

    # Create custom external force for position restraints
    restraint = CustomExternalForce('k*periodicdistance(x,y,z,x0,y0,z0)^2')
    restraint.addGlobalParameter('k', force_constant * KCAL_MOL_A2)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')

    for idx in backbone_indices:
        pos = positions[idx]
        # Handle both Vec3 and numpy array formats
        if hasattr(pos, 'x'):
            restraint.addParticle(idx, [pos.x, pos.y, pos.z])
        else:
            restraint.addParticle(idx, [pos[0], pos[1], pos[2]])

    system.addForce(restraint)
    return restraint, backbone_indices


def _prefixed(job_name: str, base_name: str) -> str:
    """Build filename with optional job_name prefix."""
    return f'{job_name}_{base_name}' if job_name else base_name


def build_ligand_only_system(ligand_sdf: str, output_dir: str, force_field_preset: str = 'ff19sb-opc',
                             temperature_k: float = 300, salt_concentration_m: float = 0.15, padding_nm: float = 1.0,
                             project_name: Optional[str] = None) -> Tuple[Any, Any, Any, str]:
    """Build solvated ligand-only system (no protein).

    For studying small molecule dynamics in solution.
    Uses OpenFF Sage 2.3.0 for ligand parameterization.

    Returns system, modeller, force field, and job_name.
    """
    print('PROGRESS:building:0', flush=True)
    t_start = time.time()

    # New layout: no prefix on filenames (self-contained job directories)
    # Legacy: project_name prefix when explicitly passed
    job_name = project_name if project_name else ''

    # 1. Load ligand
    print(f'[{time.time()-t_start:.1f}s] Loading ligand SDF...', file=sys.stderr)
    mol = load_sdf(ligand_sdf)
    if mol is None:
        raise ValueError(f"Failed to load ligand from {ligand_sdf}")

    ligand = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
    n_atoms = mol.GetNumAtoms()
    print('PROGRESS:building:20', flush=True)

    # 2. Setup force field (water model + OpenFF Sage 2.3.0 for ligand)
    preset = PRESETS[force_field_preset]
    water_model = preset['water_model']
    water_label = preset['water_label']
    print(f'[{time.time()-t_start:.1f}s] Setting up force fields ({water_label} + OpenFF Sage 2.3.0)...', file=sys.stderr)
    system_generator = _build_mixed_system_generator(preset, ligand, include_protein=False)
    ff = system_generator.forcefield
    print(f'[{time.time()-t_start:.1f}s] Ligand FF: OpenFF Sage 2.3.0', file=sys.stderr)
    print('PROGRESS:building:40', flush=True)

    # 3. Create modeller with just the ligand
    print(f'[{time.time()-t_start:.1f}s] Creating ligand system...', file=sys.stderr)
    lig_top = ligand.to_topology().to_openmm()
    lig_pos = ligand.conformers[0].to_openmm()
    modeller = Modeller(lig_top, lig_pos)
    print('PROGRESS:building:50', flush=True)

    # 4. Pre-compute SMIRNOFF template (AM1-BCC charges via sqm).
    # sqm scales ~O(N^3.5) with atom count: ~4s at 20 atoms, ~3min at 50+ atoms.
    # This is done explicitly here so progress is visible (otherwise it's hidden
    # inside addSolvent's internal createSystem call).
    estimate = _estimate_am1bcc_time(n_atoms)
    print(f'[{time.time()-t_start:.1f}s] Computing AM1-BCC partial charges ({n_atoms} atoms, est. {estimate})...', file=sys.stderr)
    print(f'PROGRESS:parameterizing:0:{n_atoms}', flush=True)
    _temp_sys = system_generator.create_system(lig_top, molecules=[ligand])
    del _temp_sys
    print(f'[{time.time()-t_start:.1f}s] AM1-BCC charges computed', file=sys.stderr)
    print('PROGRESS:parameterizing:100', flush=True)

    # 5. Optimize box shape and size, then add solvent
    box_kwargs = _optimize_box(modeller, padding_nm)
    print(f'[{time.time()-t_start:.1f}s] Adding solvent ({water_label} water, {salt_concentration_m*1000:.0f} mM NaCl)...', file=sys.stderr)
    modeller.addSolvent(
        ff,
        model=water_model,
        ionicStrength=salt_concentration_m*molar,
        positiveIon='Na+',
        negativeIon='Cl-',
        neutralize=True,
        **box_kwargs
    )
    print(f'[{time.time()-t_start:.1f}s] Solvation complete', file=sys.stderr)
    print('PROGRESS:building:80', flush=True)

    # Save solvated system
    print(f'[{time.time()-t_start:.1f}s] Saving system PDB...', file=sys.stderr)
    system_pdb = os.path.join(output_dir, _prefixed(job_name, 'system.pdb'))
    with open(system_pdb, 'w') as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)

    # Calculate system info
    box_vectors = modeller.topology.getPeriodicBoxVectors()
    a = box_vectors[0]
    b = box_vectors[1]
    c = box_vectors[2]
    cross = Vec3(
        b[1]*c[2] - b[2]*c[1],
        b[2]*c[0] - b[0]*c[2],
        b[0]*c[1] - b[1]*c[0]
    )
    volume_nm3 = abs(a[0]*cross[0] + a[1]*cross[1] + a[2]*cross[2]) / (nanometers**3)
    volume_A3 = volume_nm3 * 1000
    atom_count = modeller.topology.getNumAtoms()

    print(f'SYSTEM_INFO:{atom_count}:{volume_A3:.0f}', flush=True)
    print('PROGRESS:building:90', flush=True)

    # 5. Create system with HMR
    print(f'[{time.time()-t_start:.1f}s] Creating OpenMM system with HMR...', file=sys.stderr)
    system = system_generator.create_system(modeller.topology, molecules=[ligand])
    print(f'[{time.time()-t_start:.1f}s] System created', file=sys.stderr)

    print('PROGRESS:building:100', flush=True)
    print(f'System built: {atom_count} atoms, {volume_A3:.0f} A^3 (ligand-only)', file=sys.stderr)

    return system, modeller, ff, job_name


def build_apo_system(receptor_pdb: str, output_dir: str, force_field_preset: str = 'ff19sb-opc',
                     temperature_k: float = 300, salt_concentration_m: float = 0.15, padding_nm: float = 1.0,
                     project_name: Optional[str] = None) -> Tuple[Any, Any, Any, str]:
    """Build solvated apo (protein-only) system — no ligand.

    Returns system, modeller, force field, and job_name.
    """
    print('PROGRESS:building:0', flush=True)

    receptor_pdb = ensure_pdb_format(receptor_pdb)
    job_name = project_name if project_name else ''

    # 1. Load and prepare receptor
    print('Loading and fixing receptor PDB...', file=sys.stderr)
    prepared_topology, prepared_positions, prep_report = _prepare_receptor_topology(receptor_pdb)

    preset = PRESETS[force_field_preset]
    print('  Running protein-only topology preflight...', file=sys.stderr)
    _preflight_receptor_topology(prepared_topology, prepared_positions, preset)
    print('PROGRESS:building:30', flush=True)

    modeller = Modeller(prepared_topology, prepared_positions)

    # 2. Setup force field (protein + water, no ligand FF needed)
    water_model = preset['water_model']
    water_label = preset['water_label']
    print(f'Setting up force fields ({", ".join(preset["protein_ff"])} + {water_label})...', file=sys.stderr)
    ff = ForceField(*preset['protein_ff'], *preset['water_ff'])
    _patch_forcefield_for_chain_breaks(ff)
    print('PROGRESS:building:50', flush=True)

    # 3. Add extra particles (OPC virtual sites), align, and solvate
    try:
        modeller.addExtraParticles(ff)
    except Exception:
        print('  Warning: addExtraParticles failed — skipping', file=sys.stderr)

    box_kwargs = _optimize_box(modeller, padding_nm)
    print(f'Adding solvent ({water_label} water, {salt_concentration_m*1000:.0f} mM NaCl)...', file=sys.stderr)
    modeller.addSolvent(
        ff,
        model=water_model,
        ionicStrength=salt_concentration_m*molar,
        positiveIon='Na+',
        negativeIon='Cl-',
        neutralize=True,
        **box_kwargs
    )
    print('PROGRESS:building:80', flush=True)

    # 4. Save solvated system
    system_pdb = os.path.join(output_dir, _prefixed(job_name, 'system.pdb'))
    with open(system_pdb, 'w') as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)

    box_vectors = modeller.topology.getPeriodicBoxVectors()
    a, b, c = box_vectors[0], box_vectors[1], box_vectors[2]
    cross = Vec3(b[1]*c[2] - b[2]*c[1], b[2]*c[0] - b[0]*c[2], b[0]*c[1] - b[1]*c[0])
    volume_nm3 = abs(a[0]*cross[0] + a[1]*cross[1] + a[2]*cross[2]) / (nanometers**3)
    volume_A3 = volume_nm3 * 1000
    atom_count = modeller.topology.getNumAtoms()

    print(f'SYSTEM_INFO:{atom_count}:{volume_A3:.0f}', flush=True)
    print('PROGRESS:building:90', flush=True)

    # 5. Create system with HMR
    print('Creating OpenMM system with HMR...', file=sys.stderr)
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0*nanometers,
        ewaldErrorTolerance=0.0005,
        constraints=HBonds,
        hydrogenMass=1.5*amu
    )

    print('PROGRESS:building:100', flush=True)
    print(f'System built: {atom_count} atoms, {volume_A3:.0f} A^3 (apo)', file=sys.stderr)

    return system, modeller, ff, job_name


def build_system(receptor_pdb: str, ligand_sdf: str, output_dir: str, force_field_preset: str = 'ff19sb-opc',
                 temperature_k: float = 300, salt_concentration_m: float = 0.15, padding_nm: float = 1.0,
                 project_name: Optional[str] = None) -> Tuple[Any, Any, Any, str]:
    """Build solvated protein-ligand system.

    Force field presets:
      - 'fast': ff14SB + TIP3P (classic, well-tested)
      - 'accurate': ff19SB + OPC (modern, higher accuracy for protein-ligand)

    Returns system, modeller, force field, and job_name.
    """
    print('PROGRESS:building:0', flush=True)

    # Convert CIF to PDB if needed (OpenMM PDBFile only reads PDB format)
    receptor_pdb = ensure_pdb_format(receptor_pdb)

    # Extract job name from output directory for self-contained filenames
    # New layout: no prefix on filenames (self-contained job directories)
    # Legacy: project_name prefix when explicitly passed
    job_name = project_name if project_name else ''

    # 1. Load receptor PDB and capture ligand coordinates BEFORE stripping
    print('Loading and fixing receptor PDB...', file=sys.stderr)

    # First pass: find heterogens in the original PDB to capture ligand coordinates
    # This ensures we preserve the X-ray pose even after PDBFixer strips heterogens
    import numpy as np
    original_pdb = PDBFile(receptor_pdb)
    hetatm_positions = {}  # resname -> list of (atom_name, [x, y, z])
    solvent_ions = set(SOLVENT_AND_ION_RESIDUES) | {'SO4', 'PO4', 'GOL', 'EDO', 'ACT', 'DMS'}
    for atom in original_pdb.topology.atoms():
        rname = atom.residue.name
        if rname not in STANDARD_PROTEIN_RESIDUES and rname not in solvent_ions:
            if rname not in hetatm_positions:
                hetatm_positions[rname] = []
            pos = original_pdb.positions[atom.index].value_in_unit(angstroms)
            hetatm_positions[rname].append((atom.name, pos))

    if hetatm_positions:
        het_names = list(hetatm_positions.keys())
        het_counts = {k: len(v) for k, v in hetatm_positions.items()}
        print(f'  Found heterogens in PDB: {het_counts}', file=sys.stderr)

    preset = PRESETS[force_field_preset]
    prepared_topology: Optional[Any] = None
    prepared_positions: Optional[Any] = None
    prep_report: Dict[str, Any] = {'input_receptor_pdb': receptor_pdb, 'preflight_passed': False}
    try:
        prepared_topology, prepared_positions, prep_report = _prepare_receptor_topology(receptor_pdb)
        print('  Running protein-only topology preflight...', file=sys.stderr)
        _preflight_receptor_topology(prepared_topology, prepared_positions, preset)
        prep_report['preflight_passed'] = True
        print('  Protein-only topology preflight passed', file=sys.stderr)
    except Exception as exc:
        if isinstance(exc, ReceptorPreparationError):
            _raise_receptor_topology_preflight_failure(
                output_dir,
                receptor_pdb,
                exc.report,
                exc,
                exc.topology,
                exc.positions,
            )
        _raise_receptor_topology_preflight_failure(
            output_dir,
            receptor_pdb,
            prep_report,
            exc,
            prepared_topology,
            prepared_positions,
        )

    modeller = Modeller(prepared_topology, prepared_positions)
    print('PROGRESS:building:20', flush=True)

    # 2. Load ligand from SDF for parameterization, but use PDB coordinates for positioning
    print('Loading ligand SDF...', file=sys.stderr)

    mol = load_sdf(ligand_sdf)
    if mol is None:
        raise ValueError(f"Failed to load ligand from {ligand_sdf}")

    # Create OpenFF molecule for parameterization
    ligand = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
    n_atoms = mol.GetNumAtoms()
    print('PROGRESS:building:30', flush=True)

    # Check if the SDF ligand coordinates are aligned with the protein
    # by comparing to the heterogens we captured from the PDB
    sdf_conf = mol.GetConformer()
    sdf_com = np.mean([[sdf_conf.GetAtomPosition(i).x,
                        sdf_conf.GetAtomPosition(i).y,
                        sdf_conf.GetAtomPosition(i).z]
                       for i in range(mol.GetNumAtoms())], axis=0)

    # Protein COM from modeller
    prot_positions = modeller.positions
    prot_com = np.mean([prot_positions[i].value_in_unit(angstroms)
                        for i in range(min(len(prot_positions), modeller.topology.getNumAtoms()))],
                       axis=0)

    sdf_prot_dist = np.linalg.norm(sdf_com - prot_com)
    print(f'  SDF ligand COM distance to protein: {sdf_prot_dist:.1f} A', file=sys.stderr)

    # If SDF coordinates are far from protein, use PDB coordinates instead
    # This handles the X-ray → MD case where SDF has arbitrary coordinates
    use_pdb_coords = False
    if sdf_prot_dist > 15.0 and hetatm_positions:
        # Find the largest heterogen (likely the ligand)
        largest_het = max(hetatm_positions.keys(), key=lambda k: len(hetatm_positions[k]))
        het_atoms = hetatm_positions[largest_het]
        het_com = np.mean([pos for _, pos in het_atoms], axis=0)
        het_prot_dist = np.linalg.norm(het_com - prot_com)

        if het_prot_dist < sdf_prot_dist:
            print(f'  PDB heterogen "{largest_het}" is closer to protein ({het_prot_dist:.1f} A) — using PDB coordinates', file=sys.stderr)

            # Align SDF molecule to PDB heterogen coordinates using heavy atom matching
            # Strategy: compute translation from SDF COM to PDB COM, then apply
            # This is a simple rigid-body translation (assumes same atom ordering)
            het_heavy = [(name, pos) for name, pos in het_atoms
                         if not name.startswith('H') and name != 'EP']
            sdf_heavy_pos = []
            for i in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(i)
                if atom.GetAtomicNum() > 1:  # non-hydrogen
                    p = sdf_conf.GetAtomPosition(i)
                    sdf_heavy_pos.append([p.x, p.y, p.z])

            if len(het_heavy) > 0 and len(sdf_heavy_pos) > 0:
                pdb_heavy_com = np.mean([pos for _, pos in het_heavy], axis=0)
                sdf_heavy_com = np.mean(sdf_heavy_pos, axis=0)
                translation = pdb_heavy_com - sdf_heavy_com

                # Apply translation to all SDF atoms
                from rdkit.Geometry import Point3D
                for i in range(mol.GetNumAtoms()):
                    p = sdf_conf.GetAtomPosition(i)
                    sdf_conf.SetAtomPosition(i, Point3D(
                        p.x + translation[0],
                        p.y + translation[1],
                        p.z + translation[2]
                    ))
                use_pdb_coords = True
                print(f'  Applied translation: [{translation[0]:.1f}, {translation[1]:.1f}, {translation[2]:.1f}] A', file=sys.stderr)

                # Recreate OpenFF molecule with corrected coordinates
                ligand = Molecule.from_rdkit(mol, allow_undefined_stereo=True)

                # Verify
                new_com = np.mean([[sdf_conf.GetAtomPosition(i).x,
                                    sdf_conf.GetAtomPosition(i).y,
                                    sdf_conf.GetAtomPosition(i).z]
                                   for i in range(mol.GetNumAtoms())], axis=0)
                print(f'  Ligand COM after correction: distance to protein = {np.linalg.norm(new_com - prot_com):.1f} A', file=sys.stderr)

    # 3. Setup force field with ligand parameters (OpenFF Sage 2.3.0)
    water_model = preset['water_model']
    water_label = preset['water_label']
    print(f'Setting up force fields ({", ".join(preset["protein_ff"])} + {water_label} + OpenFF Sage 2.3.0)...', file=sys.stderr)
    system_generator = _build_mixed_system_generator(preset, ligand)
    ff = system_generator.forcefield
    # Patch createSystem to tolerate chain-break bond mismatches (e.g. terminal GLU)
    _patch_forcefield_for_chain_breaks(ff)
    print('Ligand FF: OpenFF Sage 2.3.0', file=sys.stderr)
    print('PROGRESS:building:40', flush=True)

    # 4. Add ligand to modeller (protein already has hydrogens from PDBFixer)
    print('Adding ligand to system...', file=sys.stderr)
    lig_top = ligand.to_topology().to_openmm()
    lig_pos = ligand.conformers[0].to_openmm()

    # Pre-compute AM1-BCC charges explicitly (otherwise hidden inside addSolvent)
    estimate = _estimate_am1bcc_time(n_atoms)
    print(f'Computing AM1-BCC partial charges ({n_atoms} atoms, est. {estimate})...', file=sys.stderr)
    print(f'PROGRESS:parameterizing:0:{n_atoms}', flush=True)
    _temp_sys = system_generator.create_system(lig_top, molecules=[ligand])
    del _temp_sys
    print('PROGRESS:parameterizing:100', flush=True)

    modeller.add(lig_top, lig_pos)
    # 4-site water models (OPC) require virtual particles on crystallographic
    # waters that came from receptor preparation.  Must be called before
    # addSolvent which tries to match all residues to templates.
    try:
        modeller.addExtraParticles(ff)
    except Exception:
        # Chain-break residues may not match templates for addExtraParticles;
        # skip — addSolvent will add virtual sites to new waters.
        print('  Warning: addExtraParticles failed (chain break residues) — skipping', file=sys.stderr)
    print('PROGRESS:building:50', flush=True)

    box_kwargs = _optimize_box(modeller, padding_nm)
    print(f'Adding solvent ({water_label} water, {salt_concentration_m*1000:.0f} mM NaCl)...', file=sys.stderr)
    modeller.addSolvent(
        ff,
        model=water_model,
        ionicStrength=salt_concentration_m*molar,
        positiveIon='Na+',
        negativeIon='Cl-',
        neutralize=True,
        **box_kwargs
    )
    print('PROGRESS:building:80', flush=True)

    # Save solvated system with job name for self-contained filename
    system_pdb = os.path.join(output_dir, _prefixed(job_name, 'system.pdb'))
    with open(system_pdb, 'w') as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)

    # Calculate system info
    box_vectors = modeller.topology.getPeriodicBoxVectors()
    # For rhombic dodecahedron, volume calculation is more complex
    # Use the volume from the box vectors
    a = box_vectors[0]
    b = box_vectors[1]
    c = box_vectors[2]
    # Volume = a . (b x c)
    cross = Vec3(
        b[1]*c[2] - b[2]*c[1],
        b[2]*c[0] - b[0]*c[2],
        b[0]*c[1] - b[1]*c[0]
    )
    volume_nm3 = abs(a[0]*cross[0] + a[1]*cross[1] + a[2]*cross[2]) / (nanometers**3)
    volume_A3 = volume_nm3 * 1000  # nm^3 -> A^3
    atom_count = modeller.topology.getNumAtoms()

    print(f'SYSTEM_INFO:{atom_count}:{volume_A3:.0f}', flush=True)
    print('PROGRESS:building:90', flush=True)

    # 6. Create system WITH HMR for 4fs production timestep
    # HMR (Hydrogen Mass Repartitioning) allows 4fs timestep for faster production
    # Equilibration uses 2fs for stability, production uses 4fs
    print('Creating OpenMM system with HMR...', file=sys.stderr)
    system = system_generator.create_system(modeller.topology, molecules=[ligand])

    print('PROGRESS:building:100', flush=True)
    print(f'System built: {atom_count} atoms, {volume_A3:.0f} A^3', file=sys.stderr)

    return system, modeller, ff, job_name


def check_energy(simulation: Any, stage_name: str) -> float:
    """Check for NaN or extreme energies."""
    state = simulation.context.getState(getEnergy=True)
    pe = state.getPotentialEnergy().value_in_unit(kilocalories_per_mole)
    if pe != pe:  # NaN check
        raise ValueError(f"NaN potential energy detected at {stage_name}")
    if abs(pe) > 1e10:
        raise ValueError(f"Extreme potential energy ({pe:.2e}) at {stage_name}")
    return pe


def _find_ligand_indices(topology: Any) -> Tuple[List[int], Set[str]]:
    """Find ligand heavy atom indices in an OpenMM topology.

    Identifies non-protein, non-solvent, non-ion heavy atoms.
    """
    STANDARD_PROTEIN = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        # Protonation variants (AMBER)
        'HID', 'HIE', 'HIP', 'CYX', 'ASH', 'GLH', 'LYN', 'ACE', 'NME',
    }
    SOLVENT_IONS = {
        'HOH', 'WAT', 'TIP3', 'TIP4', 'OPC', 'SPC', 'SOL',
        'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'NA+', 'CL-',
    }

    # Identify protein residues: must be a standard amino acid name AND have backbone atoms
    protein_residue_indices = set()
    for residue in topology.residues():
        if residue.name in STANDARD_PROTEIN:
            protein_residue_indices.add(residue.index)

    # Ligand = heavy atoms in residues that aren't protein or solvent/ions
    lig_indices = []
    lig_resnames = set()
    for atom in topology.atoms():
        res = atom.residue
        if res.index in protein_residue_indices:
            continue
        if res.name in SOLVENT_IONS:
            continue
        # Heavy atoms only (skip H, and virtual sites like EP)
        if atom.element is None:
            continue
        mass = atom.element.mass
        if hasattr(mass, 'value_in_unit'):
            mass = mass.value_in_unit(amu)
        if mass < 1.1:
            continue
        lig_indices.append(atom.index)
        lig_resnames.add(res.name)

    return lig_indices, lig_resnames


def log_stage_diagnostics(simulation: Any, stage_name: str, topology: Any, initial_ligand_com: Any = None) -> Optional[Any]:
    """Log detailed diagnostics at the end of each equilibration stage.

    Reports: PE, KE, temperature, restraint parameters, box volume,
    ligand COM position and displacement from initial.
    """
    import numpy as np

    state = simulation.context.getState(
        getPositions=True, getEnergy=True, getVelocities=True
    )
    pe = state.getPotentialEnergy().value_in_unit(kilocalories_per_mole)
    ke = state.getKineticEnergy().value_in_unit(kilocalories_per_mole)
    positions = state.getPositions(asNumpy=True).value_in_unit(angstroms)
    box = state.getPeriodicBoxVectors()
    vol = box[0][0] * box[1][1] * box[2][2]
    vol_A3 = vol.value_in_unit(angstroms**3)

    # Temperature: use OpenMM's built-in DOF count (accounts for constraints)
    n_dof = simulation.system.getNumParticles() * 3
    # Subtract constraints
    n_dof -= simulation.system.getNumConstraints()
    # Subtract COM motion (3 DOF)
    n_dof -= 3
    n_dof = max(n_dof, 1)
    kB_kcal = 0.001987204  # kcal/mol/K
    temp = 2.0 * ke / (n_dof * kB_kcal)

    # Restraint parameters (convert back from internal units to kcal/mol/A²)
    # Convert internal units back to kcal/mol/A²
    # OpenMM stores in kJ/mol/nm². 1 kcal/mol/A² = 418.4 kJ/mol/nm²
    KCAL_MOL_A2_INTERNAL = (1.0 * KCAL_MOL_A2).value_in_unit(kilojoules_per_mole / nanometers**2)
    k_heavy_kcal = None
    k_bb_kcal = None
    try:
        k_heavy_raw = simulation.context.getParameter('k_heavy')
        k_heavy_kcal = k_heavy_raw / KCAL_MOL_A2_INTERNAL
    except Exception:
        pass
    try:
        k_bb_raw = simulation.context.getParameter('k')
        k_bb_kcal = k_bb_raw / KCAL_MOL_A2_INTERNAL
    except Exception:
        pass

    # Ligand COM and displacement
    lig_com = None
    lig_displacement = None
    lig_info = ""
    try:
        lig_indices, lig_resnames = _find_ligand_indices(topology)
        if lig_indices:
            lig_positions = positions[lig_indices]
            lig_com = np.mean(lig_positions, axis=0)
            lig_info = f" ({len(lig_indices)} atoms, resnames: {','.join(sorted(lig_resnames))})"
            if initial_ligand_com is not None:
                lig_displacement = np.linalg.norm(lig_com - initial_ligand_com)
    except Exception:
        pass

    # Format output
    print(f'\n  === DIAGNOSTICS: {stage_name} ===', file=sys.stderr)
    print(f'    PE: {pe:.1f} kcal/mol  |  KE: {ke:.1f} kcal/mol  |  T: {temp:.1f} K', file=sys.stderr)
    print(f'    Box volume: {vol_A3:.0f} A³', file=sys.stderr)
    if k_heavy_kcal is not None:
        print(f'    k_heavy: {k_heavy_kcal:.2f} kcal/mol/A²', file=sys.stderr)
    if k_bb_kcal is not None:
        print(f'    k_backbone: {k_bb_kcal:.2f} kcal/mol/A²', file=sys.stderr)
    if lig_com is not None:
        print(f'    Ligand COM: [{lig_com[0]:.1f}, {lig_com[1]:.1f}, {lig_com[2]:.1f}] A{lig_info}', file=sys.stderr)
    else:
        print(f'    Ligand: not detected in topology', file=sys.stderr)
    if lig_displacement is not None:
        print(f'    Ligand displacement from start: {lig_displacement:.2f} A', file=sys.stderr)
    print(f'  ===', file=sys.stderr)

    return lig_com


def run_equilibration(simulation: Any, modeller: Any, output_dir: str, job_name: str, target_temp: int = 300, platform_name: str = 'CPU', seed: int = 0) -> Tuple[Any, str]:
    """Run AMBER-style equilibration with positional restraints.

    Protocol (Salomon-Ferrer et al. 2013, Case et al. AMBER Manual):
    1. Restrained minimization (heavy atoms, 10 kcal/mol/A²) — 10000 steps
    2. Graduated minimization (10 → 5 → 2 kcal/mol/A², never fully unrestrained)
    3. NVT heating 5K→100K (heavy atom + backbone restraints)
       NPT heating 100K→300K (heavy 2 + backbone 10 kcal/mol/A²)
    4. NPT equilibration at 300K (heavy 2 + backbone 10)
    5. Gradual restraint release in NPT (both backbone and heavy atoms)
    6. Unrestrained NPT equilibration (50ps)

    Total: ~360 ps
    """
    topology = modeller.topology
    positions = modeller.positions
    system = simulation.system
    integrator = simulation.integrator

    # === Stage 1: Restrained minimization (heavy atoms) ===
    print('PROGRESS:min_restrained:0', flush=True)
    print('Stage 1: Restrained minimization (heavy atoms, 10 kcal/mol/A²)...', file=sys.stderr)

    # Add heavy atom restraints to prevent solute distortion during solvent relaxation
    heavy_restraint, heavy_indices = add_heavy_atom_restraints(
        system, positions, topology, 10.0
    )
    simulation.context.reinitialize(preserveState=True)
    simulation.context.setPositions(positions)

    simulation.minimizeEnergy(maxIterations=10000)
    pe = check_energy(simulation, "after restrained minimization")
    print(f'  Restrained min: {pe:.1f} kcal/mol ({len(heavy_indices)} restrained atoms)', file=sys.stderr)
    initial_lig_com = log_stage_diagnostics(simulation, "Stage 1: Restrained Min", topology)
    print('PROGRESS:min_restrained:100', flush=True)

    # === Stage 2: Reduced-restraint minimization ===
    print('PROGRESS:min_unrestrained:0', flush=True)
    print('Stage 2: Reduced-restraint minimization...', file=sys.stderr)

    # Gradually reduce heavy atom restraints — but NEVER fully release during minimization
    # Ligand should stay near X-ray pose; only release at 300K during Stage 5
    simulation.context.setParameter('k_heavy', 5.0 * KCAL_MOL_A2)
    simulation.minimizeEnergy(maxIterations=10000)
    pe = check_energy(simulation, "after 5 kcal/mol/A² minimization")
    print(f'  Reduced restraint min (5 kcal/mol/A²): {pe:.1f} kcal/mol', file=sys.stderr)

    simulation.context.setParameter('k_heavy', 2.0 * KCAL_MOL_A2)
    simulation.minimizeEnergy(maxIterations=10000)
    pe = check_energy(simulation, "after 2 kcal/mol/A² minimization")
    print(f'  Reduced restraint min (2 kcal/mol/A²): {pe:.1f} kcal/mol', file=sys.stderr)
    log_stage_diagnostics(simulation, "Stage 2: Reduced-Restraint Min", topology, initial_lig_com)
    print('PROGRESS:min_unrestrained:100', flush=True)

    # === Stage 3: Heating with heavy atom + backbone restraints ===
    print('PROGRESS:heating:0', flush=True)
    print('Stage 3: Heating with restraints...', file=sys.stderr)

    # Get minimized positions for restraint references
    min_positions = simulation.context.getState(getPositions=True).getPositions()

    # Update heavy atom restraint reference positions to minimized coordinates
    # (Original references were pre-minimization — re-enabling would yank atoms back)
    for param_idx, atom_idx in enumerate(heavy_indices):
        pos = min_positions[atom_idx]
        heavy_restraint.setParticleParameters(param_idx, atom_idx, [pos.x, pos.y, pos.z])
    heavy_restraint.updateParametersInContext(simulation.context)

    # Re-enable heavy atom restraints during heating (prevents ligand explosion)
    simulation.context.setParameter('k_heavy', 10.0 * KCAL_MOL_A2)

    # Also add backbone restraints using minimized positions as reference
    bb_restraint, bb_indices = add_backbone_restraints(
        system, min_positions, topology, 10.0
    )
    simulation.context.reinitialize(preserveState=True)
    print(f'  Heavy atom restraints: 10 kcal/mol/A² ({len(heavy_indices)} atoms)', file=sys.stderr)
    print(f'  Backbone restraints: 10 kcal/mol/A² ({len(bb_indices)} atoms)', file=sys.stderr)

    # NVT heating: 5K -> 100K with heavy atom restraints
    simulation.context.setVelocitiesToTemperature(5 * kelvin, seed)
    integrator.setRandomNumberSeed(seed)
    integrator.setTemperature(5 * kelvin)

    nvt_temps = [5, 10, 20, 40, 60, 80, 100]
    nvt_steps_per_temp = 5000  # 10ps each = 70ps total

    for i, temp in enumerate(nvt_temps):
        integrator.setTemperature(temp * kelvin)
        simulation.step(nvt_steps_per_temp)
        pe = check_energy(simulation, f"NVT heating at {temp}K")
        print(f'  NVT {temp}K: PE = {pe:.1f} kcal/mol', file=sys.stderr)
        print(f'PROGRESS:heating:{int((i+1)*30/len(nvt_temps))}', flush=True)

    log_stage_diagnostics(simulation, "Stage 3a: NVT Heating Complete (100K)", topology, initial_lig_com)

    # Reduce heavy atom restraints before NPT (keep backbone)
    print('  Reducing heavy atom restraints to 2 kcal/mol/A²...', file=sys.stderr)
    simulation.context.setParameter('k_heavy', 2.0 * KCAL_MOL_A2)
    simulation.step(5000)  # 10ps at reduced restraints

    # Add barostat at 100K (AMBER-recommended: early NPT switch)
    print('  Adding barostat at 100K (AMBER-style)...', file=sys.stderr)
    barostat = MonteCarloBarostat(1*bar, 100*kelvin, 25)
    system.addForce(barostat)
    simulation.context.reinitialize(preserveState=True)

    # NPT heating: 100K -> 300K — keep heavy atom restraints at 2 kcal/mol/A²
    # (released later in Stage 5 alongside backbone restraints)
    npt_temps = [100, 125, 150, 175, 200, 225, 250, 275, 300]
    npt_steps_per_temp = 5000  # 10ps each = 90ps total

    for i, temp in enumerate(npt_temps):
        integrator.setTemperature(temp * kelvin)
        simulation.context.setParameter(MonteCarloBarostat.Temperature(), temp * kelvin)
        simulation.step(npt_steps_per_temp)
        pe = check_energy(simulation, f"NPT heating at {temp}K")
        print(f'  NPT {temp}K: PE = {pe:.1f} kcal/mol', file=sys.stderr)
        print(f'PROGRESS:heating:{30 + int((i+1)*70/len(npt_temps))}', flush=True)

    print('PROGRESS:heating:100', flush=True)
    log_stage_diagnostics(simulation, "Stage 3b: NPT Heating Complete (300K)", topology, initial_lig_com)

    # === Stage 4: NPT equilibration at 300K with backbone restraints (50ps) ===
    print('PROGRESS:npt_restrained:0', flush=True)
    print('Stage 4: NPT equilibration at 300K with backbone restraints (50ps)...', file=sys.stderr)

    npt_equil_steps = 25000  # 50ps at 2fs
    chunk_size = npt_equil_steps // 5

    for i in range(5):
        simulation.step(chunk_size)
        pe = check_energy(simulation, f"NPT restrained equilibration {(i+1)*20}%")
        print(f'  NPT restrained equil {(i+1)*20}%: PE = {pe:.1f} kcal/mol', file=sys.stderr)
        print(f'PROGRESS:npt_restrained:{(i+1)*20}', flush=True)

    print('PROGRESS:npt_restrained:100', flush=True)
    log_stage_diagnostics(simulation, "Stage 4: NPT Restrained Equil Complete", topology, initial_lig_com)

    # === Stage 5: Gradual restraint release (100ps) ===
    print('PROGRESS:release:0', flush=True)
    print('Stage 5: Gradual restraint release (backbone + heavy atoms)...', file=sys.stderr)

    # Reduce backbone restraints: 10 -> 5 -> 2 -> 0.5 -> 0 kcal/mol/A²
    # Reduce heavy atom restraints: 2 -> 1 -> 0.5 -> 0.1 -> 0 kcal/mol/A²
    # 25ps per step = 100ps total
    bb_release = [5.0, 2.0, 0.5, 0.0]
    heavy_release = [1.0, 0.5, 0.1, 0.0]
    release_steps = 12500  # 25ps at 2fs per step

    for i in range(len(bb_release)):
        simulation.context.setParameter('k', bb_release[i] * KCAL_MOL_A2)
        simulation.context.setParameter('k_heavy', heavy_release[i] * KCAL_MOL_A2)
        simulation.step(release_steps)
        pe = check_energy(simulation, f"restraint release bb={bb_release[i]}, heavy={heavy_release[i]}")
        print(f'  Release bb={bb_release[i]:.1f}, heavy={heavy_release[i]:.1f}: PE = {pe:.1f} kcal/mol', file=sys.stderr)
        print(f'PROGRESS:release:{(i+1)*25}', flush=True)

    print('PROGRESS:release:100', flush=True)
    log_stage_diagnostics(simulation, "Stage 5: Restraint Release Complete", topology, initial_lig_com)

    # === Stage 6: Unrestrained NPT equilibration (50ps) ===
    print('PROGRESS:equilibration:0', flush=True)
    print('Stage 6: Unrestrained NPT equilibration at 300K (50ps)...', file=sys.stderr)

    unrestrained_steps = 25000  # 50ps at 2fs
    chunk_size = unrestrained_steps // 5

    for i in range(5):
        simulation.step(chunk_size)
        pe = check_energy(simulation, f"unrestrained equilibration {(i+1)*20}%")
        print(f'  Unrestrained equil {(i+1)*20}%: PE = {pe:.1f} kcal/mol', file=sys.stderr)
        print(f'PROGRESS:equilibration:{(i+1)*20}', flush=True)

    print('PROGRESS:equilibration:100', flush=True)

    # Save equilibrated state (positions, velocities, box vectors)
    state = simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
    equilibrated_pdb = os.path.join(output_dir, _prefixed(job_name, 'equilibrated.pdb'))
    with open(equilibrated_pdb, 'w') as f:
        PDBFile.writeFile(topology, state.getPositions(), f)

    pe = state.getPotentialEnergy().value_in_unit(kilocalories_per_mole)
    box_vectors = state.getPeriodicBoxVectors()
    import numpy as np
    box_a = np.array([box_vectors[0].x, box_vectors[0].y, box_vectors[0].z]) * 10  # nm → Å
    box_b = np.array([box_vectors[1].x, box_vectors[1].y, box_vectors[1].z]) * 10
    box_c = np.array([box_vectors[2].x, box_vectors[2].y, box_vectors[2].z]) * 10
    volume_A3 = abs(np.dot(box_a, np.cross(box_b, box_c)))
    total_mass = builtins.sum([a.element.mass.value_in_unit(amu) for a in topology.atoms() if a.element is not None])
    density_g_cm3 = (total_mass * 1.66054e-24) / (volume_A3 * 1e-24)
    print(f'Equilibration complete: PE = {pe:.1f} kcal/mol, density = {density_g_cm3:.4f} g/cm³, volume = {volume_A3:.0f} Å³', file=sys.stderr)
    log_stage_diagnostics(simulation, "Stage 6: Unrestrained Equil Complete (FINAL)", topology, initial_lig_com)
    return state, platform_name


def run_production(system: Any, modeller: Any, equilibrated_state: Any, output_dir: str, job_name: str, production_ns: float, platform_name: str, temperature_k: float = 300, seed: int = 0) -> str:
    """Run production MD and save trajectory.

    Creates a new simulation with 4fs timestep (HMR enabled in system).
    Saves trajectory every 10 ps (2500 steps at 4fs).
    Saves energy every 2 ps (500 steps).

    """
    print('PROGRESS:production:0', flush=True)
    print(f'Running production MD ({production_ns} ns) with 4fs timestep (HMR)...', file=sys.stderr)

    # Create new integrator with 4fs timestep for production (HMR enabled in system)
    integrator = LangevinMiddleIntegrator(temperature_k*kelvin, 1/picosecond, 0.004*picoseconds)
    if seed > 0:
        integrator.setRandomNumberSeed(seed)

    # Create new simulation with 4fs integrator
    # Platform dispatch: uses whichever platform was selected during equilibration
    if platform_name == 'CUDA':
        platform = Platform.getPlatformByName('CUDA')
        properties = {'CudaPrecision': 'mixed'}
        simulation = Simulation(modeller.topology, system, integrator, platform, properties)
    elif platform_name == 'Metal':
        platform = Platform.getPlatformByName('Metal')
        simulation = Simulation(modeller.topology, system, integrator, platform)
    elif platform_name == 'OpenCL':
        platform = Platform.getPlatformByName('OpenCL')
        simulation = Simulation(modeller.topology, system, integrator, platform)
    else:
        platform = Platform.getPlatformByName('CPU')
        simulation = Simulation(modeller.topology, system, integrator, platform)

    # Restore equilibrated state (positions + velocities)
    simulation.context.setPositions(equilibrated_state.getPositions())
    simulation.context.setVelocities(equilibrated_state.getVelocities())
    simulation.context.setPeriodicBoxVectors(*equilibrated_state.getPeriodicBoxVectors())

    # Ensure all equilibration restraint forces are OFF for production
    # (new Context resets global parameters to their initial values)
    simulation.context.setParameter('k_heavy', 0)
    simulation.context.setParameter('k', 0)

    # Setup trajectory reporter (save every 10 ps = 2500 steps at 4fs)
    # Use job name for self-contained filenames
    dcd_file = os.path.join(output_dir, _prefixed(job_name, 'trajectory.dcd'))
    simulation.reporters.append(DCDReporter(dcd_file, 2500))
    simulation.reporters.append(StateDataReporter(
        os.path.join(output_dir, _prefixed(job_name, 'energy.csv')), 500,
        step=True, time=True, potentialEnergy=True,
        kineticEnergy=True, temperature=True, volume=True
    ))

    # Checkpoint every 0.5 ns (125000 steps at 4fs) for crash recovery
    checkpoint_file = os.path.join(output_dir, _prefixed(job_name, 'checkpoint.chk'))
    simulation.reporters.append(CheckpointReporter(checkpoint_file, 125000))

    # Run production with 4fs timestep: 250000 steps/ns
    # Report every 0.1 ns (25000 steps at 4fs)
    total_steps = int(production_ns * 250000)
    steps_per_report = 25000  # 0.1 ns at 4fs

    steps_completed = 0
    while steps_completed < total_steps:
        steps_to_run = min(steps_per_report, total_steps - steps_completed)

        simulation.step(steps_to_run)
        steps_completed += steps_to_run

        # Report progress as ns completed (format: current_ns/total_ns)
        ns_completed = steps_completed / 250000
        print(f'PROGRESS:production:{ns_completed:.1f}/{production_ns:.1f}', flush=True)

    print(f'PROGRESS:production:{production_ns:.1f}/{production_ns:.1f}', flush=True)

    # Save final frame
    state = simulation.context.getState(getPositions=True)
    final_pdb = os.path.join(output_dir, _prefixed(job_name, 'final.pdb'))
    with open(final_pdb, 'w') as f:
        PDBFile.writeFile(modeller.topology, state.getPositions(), f)

    print('Production complete.', file=sys.stderr)
    return dcd_file


def run_benchmark(system: Any, modeller: Any, output_dir: str, temperature_k: float = 300) -> float:
    """Run short benchmark to estimate ns/day.

    Runs ~2500 steps (10 ps at 4fs with HMR) after brief warmup.
    """
    print('PROGRESS:benchmark:0', flush=True)
    print('Running performance benchmark (4fs timestep with HMR)...', file=sys.stderr)

    # Create simulation for benchmark (4fs timestep, HMR enabled in system)
    integrator = LangevinMiddleIntegrator(temperature_k*kelvin, 1/picosecond, 0.004*picoseconds)

    # Platform cascade: CUDA → OpenCL (cl2Metal) → Metal → CPU
    try:
        platform = Platform.getPlatformByName('CUDA')
        properties = {'CudaPrecision': 'mixed'}
        simulation = Simulation(modeller.topology, system, integrator, platform, properties)
        print('Using CUDA platform', file=sys.stderr)
    except Exception:
        try:
            platform = Platform.getPlatformByName('OpenCL')
            simulation = Simulation(modeller.topology, system, integrator, platform)
            print('Using OpenCL platform (Apple GPU via cl2Metal)', file=sys.stderr)
        except Exception:
            try:
                platform = Platform.getPlatformByName('Metal')
                simulation = Simulation(modeller.topology, system, integrator, platform)
                print('Using Metal platform (Apple GPU)', file=sys.stderr)
            except Exception:
                platform = Platform.getPlatformByName('CPU')
                simulation = Simulation(modeller.topology, system, integrator, platform)
                print('Using CPU platform', file=sys.stderr)

    simulation.context.setPositions(modeller.positions)

    # Quick minimization
    simulation.minimizeEnergy(maxIterations=500)

    # Warm up
    simulation.step(250)
    print('PROGRESS:benchmark:50', flush=True)

    # Timed benchmark
    benchmark_steps = 2500  # 10 ps at 4fs
    start_time = time.time()
    simulation.step(benchmark_steps)
    elapsed = time.time() - start_time

    # Calculate ns/day
    ns_per_step = 0.004 / 1000  # 4 fs = 0.000004 ns
    ns_simulated = benchmark_steps * ns_per_step
    seconds_per_day = 86400
    ns_per_day = (ns_simulated / elapsed) * seconds_per_day

    print('PROGRESS:benchmark:100', flush=True)
    print(f'BENCHMARK:{ns_per_day:.1f}', flush=True)
    print(f'Performance: {ns_per_day:.1f} ns/day', file=sys.stderr)

    return ns_per_day


def main() -> None:
    parser = argparse.ArgumentParser(description='OpenMM MD simulation for FragGen')
    parser.add_argument('--receptor', help='Receptor PDB file (not required for ligand-only mode)')
    parser.add_argument('--ligand', help='Ligand SDF file (not required for apo mode)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--production_ns', type=float, default=10, help='Production duration in ns')
    parser.add_argument('--force_field_preset',
                        choices=list(PRESETS.keys()) + ['fast', 'accurate'],
                        default='ff19sb-opc',
                        help='Force field preset (default: ff19sb-opc)')
    parser.add_argument('--ligand_only', action='store_true',
                        help='Ligand-only mode (no protein, small molecule in solvent)')
    parser.add_argument('--apo', action='store_true',
                        help='Apo mode (protein-only, no ligand)')
    parser.add_argument('--benchmark_only', action='store_true', help='Only run benchmark')
    parser.add_argument('--temperature', type=float, default=300, help='Production temperature in K (default: 300)')
    parser.add_argument('--salt_concentration', type=float, default=0.15, help='Salt concentration in M (default: 0.15)')
    parser.add_argument('--padding', type=float, default=1.0, help='Box padding in nm (default: 1.0)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for velocity generation and Langevin noise (0=auto from clock)')
    parser.add_argument('--project_name', default=None,
                        help='Project name prefix for output files (default: no prefix)')
    args = parser.parse_args()

    # Map legacy preset names
    args.force_field_preset = _PRESET_ALIASES.get(args.force_field_preset, args.force_field_preset)

    if not args.ligand_only and not args.apo and not args.receptor:
        parser.error('--receptor is required unless --ligand_only or --apo is specified')
    if not args.apo and not args.ligand:
        parser.error('--ligand is required unless --apo is specified')

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve random seed (0 = auto from clock, >0 = explicit for reproducibility)
    import random as _random
    if args.seed == 0:
        args.seed = _random.randint(1, 2**31 - 1)
    print(f'Random seed: {args.seed}', file=sys.stderr)

    # Save seed so the run can be reproduced
    seed_path = os.path.join(args.output_dir, 'seed.txt')
    with open(seed_path, 'w') as f:
        f.write(str(args.seed))

    # Set up log file — tee stderr to a log file for debugging
    log_path = os.path.join(args.output_dir, 'simulation.log')
    log_file = open(log_path, 'w')

    class TeeStderr:
        """Write to both stderr and log file."""
        def __init__(self, original: Any, log: Any) -> None:
            self.original = original
            self.log = log
        def write(self, msg: str) -> None:
            self.original.write(msg)
            self.log.write(msg)
            self.log.flush()
        def flush(self) -> None:
            self.original.flush()
            self.log.flush()

    sys.stderr = TeeStderr(sys.stderr, log_file)

    if args.apo:
        print(f'Building apo system (preset: {args.force_field_preset}, {args.temperature}K, {args.salt_concentration*1000:.0f}mM)...', file=sys.stderr)
        system, modeller, ff, job_name = build_apo_system(
            args.receptor, args.output_dir, args.force_field_preset,
            args.temperature, args.salt_concentration, args.padding,
            project_name=args.project_name
        )
    elif args.ligand_only:
        print(f'Building ligand-only system (preset: {args.force_field_preset}, {args.temperature}K, {args.salt_concentration*1000:.0f}mM)...', file=sys.stderr)
        system, modeller, ff, job_name = build_ligand_only_system(
            args.ligand, args.output_dir, args.force_field_preset,
            args.temperature, args.salt_concentration, args.padding,
            project_name=args.project_name
        )
    else:
        print(f'Building system (preset: {args.force_field_preset}, {args.temperature}K, {args.salt_concentration*1000:.0f}mM)...', file=sys.stderr)
        system, modeller, ff, job_name = build_system(
            args.receptor, args.ligand, args.output_dir, args.force_field_preset,
            args.temperature, args.salt_concentration, args.padding,
            project_name=args.project_name
        )

    if args.benchmark_only:
        ns_day = run_benchmark(system, modeller, args.output_dir, args.temperature)
        print(f'SUCCESS:benchmark:{ns_day:.1f}', flush=True)
        return

    # Create simulation with 2fs timestep (standard for HBonds constraints)
    print('Creating simulation...', file=sys.stderr)
    integrator = LangevinMiddleIntegrator(50*kelvin, 1/picosecond, 0.002*picoseconds)  # 2fs

    # List available platforms
    print(f'Available platforms: {[Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]}', file=sys.stderr)

    # Platform cascade: CUDA → OpenCL (cl2Metal) → Metal → CPU
    platform_name = 'CPU'
    try:
        platform = Platform.getPlatformByName('CUDA')
        properties = {'CudaPrecision': 'mixed'}
        simulation = Simulation(modeller.topology, system, integrator, platform, properties)
        platform_name = 'CUDA'
        try:
            device_name = platform.getPropertyValue(simulation.context, 'DeviceName')
            print(f'Using CUDA platform: {device_name}', file=sys.stderr)
        except Exception:
            print('Using CUDA platform', file=sys.stderr)
    except Exception as cuda_err:
        print(f'CUDA not available: {cuda_err}', file=sys.stderr)
        try:
            platform = Platform.getPlatformByName('OpenCL')
            simulation = Simulation(modeller.topology, system, integrator, platform)
            platform_name = 'OpenCL'
            try:
                device_name = platform.getPropertyValue(simulation.context, 'DeviceName')
                print(f'Using OpenCL platform (single precision): {device_name}', file=sys.stderr)
            except Exception:
                print('Using OpenCL platform (single precision)', file=sys.stderr)
        except Exception as ocl_err:
            print(f'OpenCL not available: {ocl_err}', file=sys.stderr)
            try:
                platform = Platform.getPlatformByName('Metal')
                simulation = Simulation(modeller.topology, system, integrator, platform)
                platform_name = 'Metal'
                print('Using Metal platform (Apple GPU)', file=sys.stderr)
            except Exception as metal_err:
                print(f'Metal not available: {metal_err}', file=sys.stderr)
                platform = Platform.getPlatformByName('CPU')
                simulation = Simulation(modeller.topology, system, integrator, platform)
                print('Using CPU platform', file=sys.stderr)

    simulation.context.setPositions(modeller.positions)

    print('Running equilibration (~170 ps, AMBER-style protocol)...', file=sys.stderr)
    equilibrated_state, actual_platform = run_equilibration(simulation, modeller, args.output_dir, job_name, target_temp=args.temperature, platform_name=platform_name, seed=args.seed)

    print(f'Running production ({args.production_ns} ns) on {actual_platform}...', file=sys.stderr)
    trajectory = run_production(system, modeller, equilibrated_state, args.output_dir, job_name, args.production_ns, actual_platform, args.temperature, seed=args.seed)

    print(f'SUCCESS:{trajectory}', flush=True)


if __name__ == '__main__':
    main()
