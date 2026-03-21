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
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolTransforms
    from rdkit.Chem.rdMolAlign import GetBestRMS
    from rdkit.Geometry import Point3D
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("ERROR: RDKit not installed", file=sys.stderr)
    sys.exit(1)

try:
    import openmm
    from openmm import app as omm_app
    from openmm import unit as omm_unit
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False


class GBSAMinimizer:
    """
    OpenMM minimizer with OpenFF Sage 2.3.0 + OBC2 implicit solvent.
    Parameterizes molecule once via OpenFF toolkit, adds GBSA-OBC2
    solvation force, then provides fast minimize() calls by reusing
    the OpenMM Context.

    Falls back to MMFF94s vacuum if OpenFF/OpenMM unavailable.
    """

    # mbondi3 Born radii (nm) and OBC2 screening factors by element
    _RADII_NM: dict = {
        'H': 0.12, 'C': 0.17, 'N': 0.155, 'O': 0.15, 'F': 0.15,
        'S': 0.18, 'P': 0.185, 'Cl': 0.17, 'Br': 0.185, 'I': 0.198,
    }
    _SCREEN: dict = {
        'H': 0.85, 'C': 0.72, 'N': 0.79, 'O': 0.85, 'F': 0.88,
        'S': 0.96, 'P': 0.86, 'Cl': 0.80, 'Br': 0.80, 'I': 0.80,
    }

    def __init__(self, rdmol: Any, conf_id: int) -> None:
        self.ready = False
        self.n_atoms = rdmol.GetNumAtoms()

        if not HAS_OPENMM:
            print("  OpenMM not available, using MMFF94s vacuum", file=sys.stderr)
            return

        try:
            self._parameterize(rdmol)
        except Exception as e:
            print(f"  Warning: GBSA setup failed ({e}), using MMFF94s vacuum",
                  file=sys.stderr)

    def _parameterize(self, rdmol: Any) -> None:
        from openff.toolkit import Molecule as OFFMolecule
        from openff.toolkit import ForceField as OFFForceField

        # Convert RDKit mol to OpenFF
        off_mol = OFFMolecule.from_rdkit(rdmol, allow_undefined_stereo=True)
        off_mol.assign_partial_charges('gasteiger')

        # Sage 2.3.0 parameterization (fall back to 2.0.0)
        sage_version = None
        for ver in ['openff-2.3.0.offxml', 'openff-2.0.0.offxml']:
            try:
                sage = OFFForceField(ver)
                sage_version = ver
                break
            except Exception:
                continue
        if sage is None:
            raise RuntimeError("No OpenFF Sage force field available")

        system = sage.create_openmm_system(
            off_mol.to_topology(), charge_from_molecules=[off_mol]
        )

        # Verify particle count
        if system.getNumParticles() != self.n_atoms:
            raise RuntimeError(
                f"Atom count mismatch: RDKit={self.n_atoms}, "
                f"OpenMM={system.getNumParticles()}"
            )

        # Extract charges from NonbondedForce
        nb_force = None
        for f in system.getForces():
            if isinstance(f, openmm.NonbondedForce):
                nb_force = f
                break
        if nb_force is None:
            raise RuntimeError("No NonbondedForce in OpenFF system")

        # Add OBC2 implicit solvent force
        gbsa = openmm.GBSAOBCForce()
        gbsa.setSolventDielectric(78.5)
        gbsa.setSoluteDielectric(1.0)
        gbsa.setNonbondedMethod(openmm.GBSAOBCForce.NoCutoff)

        for i in range(self.n_atoms):
            charge, sigma, epsilon = nb_force.getParticleParameters(i)
            q = charge.value_in_unit(omm_unit.elementary_charge)
            element = rdmol.GetAtomWithIdx(i).GetSymbol()
            radius = self._RADII_NM.get(element, 0.15)
            screen = self._SCREEN.get(element, 0.80)
            # mbondi3: H bonded to N gets larger radius
            if element == 'H':
                for nbr in rdmol.GetAtomWithIdx(i).GetNeighbors():
                    if nbr.GetSymbol() == 'N':
                        radius = 0.13
                        break
            gbsa.addParticle(q, radius, screen)

        system.addForce(gbsa)

        # Create context on CPU (fast for small molecules, avoids Metal kernel compilation)
        integrator = openmm.VerletIntegrator(0.001 * omm_unit.picoseconds)
        platform = openmm.Platform.getPlatformByName('CPU')
        self.context = openmm.Context(system, integrator, platform)
        self.ready = True

        label = sage_version.replace('.offxml', '').replace('openff-', 'Sage ')
        print(f"  {label} + OBC2 implicit solvent ready", flush=True)

    def minimize(self, rdmol: Any, conf_id: int, max_iters: int = 5000) -> Optional[float]:
        """Minimize conformer with implicit solvent. Returns energy in kcal/mol.
        Uses 5000 max iterations and tight gradient convergence (0.01 kJ/mol/nm ≈
        0.001 kcal/mol/Å) to match MacroModel PRCG defaults."""
        if not self.ready:
            return _minimize_mmff94s(rdmol, conf_id, max_iters)

        try:
            conf = rdmol.GetConformer(conf_id)

            # Set positions (Å → nm)
            positions = []
            for i in range(self.n_atoms):
                pos = conf.GetAtomPosition(i)
                positions.append(
                    openmm.Vec3(pos.x * 0.1, pos.y * 0.1, pos.z * 0.1)
                )
            self.context.setPositions(positions)

            # Minimize with tight convergence (0.01 kJ/mol/nm ≈ 0.001 kcal/mol/Å)
            openmm.LocalEnergyMinimizer.minimize(
                self.context, tolerance=0.01, maxIterations=max_iters
            )

            # Get energy and minimized positions
            state = self.context.getState(getEnergy=True, getPositions=True)
            energy = state.getPotentialEnergy().value_in_unit(
                omm_unit.kilocalories_per_mole
            )

            # Copy minimized positions back (nm → Å)
            min_pos = state.getPositions()
            for i in range(self.n_atoms):
                p = min_pos[i].value_in_unit(omm_unit.angstrom)
                conf.SetAtomPosition(i, Point3D(float(p[0]), float(p[1]), float(p[2])))

            return energy
        except Exception as e:
            print(f"Warning: GBSA minimize failed: {e}", file=sys.stderr)
            return _minimize_mmff94s(rdmol, conf_id, max_iters)

    def cleanup(self) -> None:
        """Clean up resources."""
        self.context = None


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


def get_rotatable_bonds_mcmm(
    mol: Any, sample_amides: bool
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Get rotatable bonds for MCMM perturbation.
    Returns (exocyclic_bonds, ring_bonds) as lists of (atom_i, atom_j) pairs.
    Exocyclic: single bonds between non-terminal atoms, not in rings.
    Ring: endocyclic bonds in small rings (3-7 members) for pucker sampling.
    If sample_amides is True, amide bonds are included (snapped to cis/trans).
    """
    # All exocyclic single bonds between non-terminal atoms
    rot_pattern = Chem.MolFromSmarts('[!D1]-&!@[!D1]')
    rot_matches = mol.GetSubstructMatches(rot_pattern) if rot_pattern else []

    # Amide bond pattern
    amide_pattern = Chem.MolFromSmarts('[NX3]-[CX3]=[OX1]')
    amide_matches = mol.GetSubstructMatches(amide_pattern) if amide_pattern else []
    amide_bonds = set()
    for match in amide_matches:
        # The N-C bond is atoms 0 and 1
        amide_bonds.add((min(match[0], match[1]), max(match[0], match[1])))

    exocyclic: List[Tuple[int, int]] = []
    for i, j in rot_matches:
        key = (min(i, j), max(i, j))
        is_amide = key in amide_bonds
        if is_amide and not sample_amides:
            continue
        exocyclic.append((i, j))

    # Ring bonds: find endocyclic single bonds in small rings (3-7 members)
    # for ring pucker / envelope sampling
    ring_bonds: List[Tuple[int, int]] = []
    seen_ring_bonds: set = set()
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        ring_size = len(ring)
        if ring_size < 3 or ring_size > 7:
            continue
        # Pick one bond per ring for pucker perturbation
        # Choose the bond opposite the most substituted atom for best pucker effect
        for idx in range(ring_size):
            ai = ring[idx]
            aj = ring[(idx + 1) % ring_size]
            bond = mol.GetBondBetweenAtoms(ai, aj)
            if bond is None:
                continue
            # Only single bonds (skip aromatic/double)
            if bond.GetBondTypeAsDouble() != 1.0 or bond.GetIsAromatic():
                continue
            key = (min(ai, aj), max(ai, aj))
            if key not in seen_ring_bonds:
                seen_ring_bonds.add(key)
                ring_bonds.append((ai, aj))

    return exocyclic, ring_bonds


def get_dihedral_atoms(
    mol: Any, bond_i: int, bond_j: int
) -> Optional[Tuple[int, int, int, int]]:
    """
    Find a full dihedral quartet (a, i, j, b) for a rotatable bond i-j.
    Picks the heaviest neighbor of each terminal atom.
    """
    atom_i = mol.GetAtomWithIdx(bond_i)
    atom_j = mol.GetAtomWithIdx(bond_j)

    # Find neighbor of i that is not j
    nbrs_i = [n.GetIdx() for n in atom_i.GetNeighbors() if n.GetIdx() != bond_j]
    if not nbrs_i:
        return None
    # Pick heaviest neighbor
    a = max(nbrs_i, key=lambda idx: mol.GetAtomWithIdx(idx).GetAtomicNum())

    # Find neighbor of j that is not i
    nbrs_j = [n.GetIdx() for n in atom_j.GetNeighbors() if n.GetIdx() != bond_i]
    if not nbrs_j:
        return None
    b = max(nbrs_j, key=lambda idx: mol.GetAtomWithIdx(idx).GetAtomicNum())

    return (a, bond_i, bond_j, b)


def _reembed_ring_pucker(
    mol: Any, conf_id: int, rng: random.Random
) -> Optional[int]:
    """
    Sample a new ring pucker by re-embedding the molecule with a different
    random seed, then copying exocyclic atom positions from the parent.

    RDKit's distance geometry naturally explores ring conformations
    (chair/boat/twist/envelope) during embedding — this is far more
    reliable than coordinate perturbation, which the minimizer undoes.

    Returns new conformer ID, or None on failure.
    """
    seed = rng.randint(1, 999999)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.maxIterations = 200
    params.clearConfs = False  # Keep existing conformers
    params.useRandomCoords = rng.random() < 0.3  # 30% chance of random coords for diversity

    new_cid = AllChem.EmbedMolecule(mol, params)
    if new_cid < 0:
        # Fallback with random coords
        params.useRandomCoords = True
        new_cid = AllChem.EmbedMolecule(mol, params)
        if new_cid < 0:
            return None

    return new_cid


def _minimize_mmff94s(mol: Any, conf_id: int, max_iters: int = 5000) -> Optional[float]:
    """Minimize a conformer with MMFF94s, fallback to UFF. Returns energy or None."""
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
    if props is not None:
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        if ff is not None:
            ff.Minimize(maxIts=max_iters)
            return ff.CalcEnergy()
    # Fallback to UFF
    ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
    if ff is not None:
        ff.Minimize(maxIts=max_iters)
        return ff.CalcEnergy()
    return None


def generate_conformers_mcmm(
    mol: Any,
    max_conformers: int,
    rmsd_cutoff: float,
    energy_window: float,
    mcmm_steps: int,
    temperature: float,
    sample_amides: bool,
) -> Tuple[Any, List[Tuple[int, float]]]:
    """
    Monte Carlo Multiple Minimum conformer search.

    Iteratively perturbs torsion angles, minimizes with GBn2 implicit
    solvent (GAFF2 + AM1-BCC via antechamber/tleap/OpenMM), and
    accepts/rejects based on Metropolis criterion + RMSD dedup.
    Falls back to MMFF94s vacuum if OpenMM/AmberTools unavailable.

    Returns (mol_with_conformers, [(conf_id, energy), ...])
    """
    rng = random.Random(42)
    R = 1.987204e-3  # Gas constant in kcal/(mol·K)
    RT = R * temperature

    mol = Chem.AddHs(mol)

    # Get rotatable bonds (exocyclic + ring)
    exocyclic_bonds, ring_bonds = get_rotatable_bonds_mcmm(mol, sample_amides)

    # Build amide bond set for special handling
    amide_pattern = Chem.MolFromSmarts('[NX3]-[CX3]=[OX1]')
    amide_matches = mol.GetSubstructMatches(amide_pattern) if amide_pattern else []
    amide_bond_set = set()
    for match in amide_matches:
        amide_bond_set.add((min(match[0], match[1]), max(match[0], match[1])))

    # Build dihedral quartets for exocyclic/amide bonds
    dihedrals: List[Tuple[Tuple[int, int, int, int], str]] = []
    for bi, bj in exocyclic_bonds:
        quartet = get_dihedral_atoms(mol, bi, bj)
        if quartet is not None:
            key = (min(bi, bj), max(bi, bj))
            btype = 'amide' if key in amide_bond_set else 'exocyclic'
            dihedrals.append((quartet, btype))

    # Collect ring atom lists for pucker perturbation (no dihedral setting —
    # RDKit forbids SetDihedralDeg on ring bonds, so we use coordinate perturbation)
    ring_info = mol.GetRingInfo()
    rings: List[List[int]] = []
    for ring in ring_info.AtomRings():
        ring_size = len(ring)
        if 3 <= ring_size <= 7:
            # Only non-aromatic rings (aromatic rings are planar, no puckering)
            has_non_aromatic_bond = False
            for idx in range(ring_size):
                bond = mol.GetBondBetweenAtoms(ring[idx], ring[(idx + 1) % ring_size])
                if bond and not bond.GetIsAromatic():
                    has_non_aromatic_bond = True
                    break
            if has_non_aromatic_bond:
                rings.append(list(ring))

    # Step 1: Embed starting structure
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    try:
        cid = AllChem.EmbedMolecule(mol, params)
    except Exception:
        params.useRandomCoords = True
        try:
            cid = AllChem.EmbedMolecule(mol, params)
        except Exception:
            return mol, []

    if cid < 0:
        return mol, []

    # Quick MMFF94s pre-minimize for clean geometry before parameterization
    _minimize_mmff94s(mol, cid, max_iters=200)

    # Initialize GBn2 minimizer (parameterizes once with antechamber/tleap/OpenMM)
    minimizer = GBSAMinimizer(mol, cid)

    if not dihedrals and not rings:
        print("Warning: No rotatable bonds found, using single conformer", file=sys.stderr)
        energy = minimizer.minimize(mol, cid)
        minimizer.cleanup()
        if energy is not None:
            return mol, [(cid, energy)]
        return mol, [(cid, 0.0)]

    # Minimize starting structure with GBn2
    e0 = minimizer.minimize(mol, cid)
    if e0 is None:
        minimizer.cleanup()
        return mol, [(cid, 0.0)]

    # Pool: list of (conf_id, energy)
    pool: List[Tuple[int, float]] = [(cid, e0)]
    best_energy = e0

    # Log torsion details
    n_exo = sum(1 for _, bt in dihedrals if bt == 'exocyclic')
    n_amide = sum(1 for _, bt in dihedrals if bt == 'amide')
    total_dof = len(dihedrals) + len(rings)
    print(f"  MCMM: {total_dof} DOF ({n_exo} exocyclic, {len(rings)} ring pucker, {n_amide} amide), "
          f"{mcmm_steps} steps, T={temperature}K", flush=True)
    for di, (quartet, btype) in enumerate(dihedrals):
        labels = []
        for idx in quartet:
            a = mol.GetAtomWithIdx(idx)
            labels.append(f"{a.GetSymbol()}{idx}")
        print(f"    torsion {di}: {'-'.join(labels)} [{btype}]", flush=True)
    for ri, ring in enumerate(rings):
        atoms = [f"{mol.GetAtomWithIdx(idx).GetSymbol()}{idx}" for idx in ring]
        print(f"    ring {ri}: {'-'.join(atoms)} [{len(ring)}-membered]", flush=True)

    # Rejection counters for diagnostics
    n_energy_reject = 0
    n_metropolis_reject = 0
    n_rmsd_reject = 0
    n_accepted = 0
    n_replaced = 0

    print(f"  Starting energy: {e0:.2f} kcal/mol", flush=True)

    for step in range(mcmm_steps):
        # Pick random parent from pool
        parent_idx = rng.randrange(len(pool))
        parent_cid, parent_energy = pool[parent_idx]

        # Copy parent conformer
        new_cid = mol.AddConformer(mol.GetConformer(parent_cid), assignId=True)
        conf = mol.GetConformer(new_cid)

        # Decide whether to perturb ring puckers this step (30% chance if rings exist)
        perturb_rings = len(rings) > 0 and rng.random() < 0.3

        if perturb_rings:
            # Re-embed to sample a new ring pucker, then apply torsion perturbations
            reembed_cid = _reembed_ring_pucker(mol, parent_cid, rng)
            if reembed_cid is not None and reembed_cid != new_cid:
                # Copy re-embedded coordinates into our new conformer
                reembed_conf = mol.GetConformer(reembed_cid)
                for ai in range(mol.GetNumAtoms()):
                    pos = reembed_conf.GetAtomPosition(ai)
                    conf.SetAtomPosition(ai, pos)
                mol.RemoveConformer(reembed_cid)

        # Perturb 1-5 exocyclic/amide torsions on top
        if dihedrals:
            max_dof = min(5, len(dihedrals))
            n_perturb = rng.randint(1, max_dof)
            chosen = rng.sample(range(len(dihedrals)), min(n_perturb, len(dihedrals)))

            for di in chosen:
                quartet, btype = dihedrals[di]
                if btype == 'amide':
                    angle = rng.choice([0.0, 180.0])
                else:
                    angle = rng.uniform(-180.0, 180.0)
                rdMolTransforms.SetDihedralDeg(conf, *quartet, angle)

        # Minimize
        new_energy = minimizer.minimize(mol, new_cid)
        rejected = False

        if new_energy is None:
            mol.RemoveConformer(new_cid)
            rejected = True
        elif new_energy > best_energy + energy_window:
            n_energy_reject += 1
            mol.RemoveConformer(new_cid)
            rejected = True
        else:
            # Metropolis criterion
            dE = new_energy - parent_energy
            if dE > 0:
                p_accept = math.exp(-dE / RT)
                if rng.random() > p_accept:
                    n_metropolis_reject += 1
                    mol.RemoveConformer(new_cid)
                    rejected = True

        if not rejected:
            # RMSD dedup against pool
            dominated_idx = None
            is_duplicate = False
            for pi, (pool_cid, pool_energy) in enumerate(pool):
                try:
                    rmsd = GetBestRMS(mol, mol, pool_cid, new_cid)
                except Exception:
                    continue
                if rmsd < rmsd_cutoff:
                    if new_energy < pool_energy:
                        dominated_idx = pi
                    else:
                        is_duplicate = True
                    break

            if is_duplicate:
                n_rmsd_reject += 1
                mol.RemoveConformer(new_cid)
            elif dominated_idx is not None:
                old_cid = pool[dominated_idx][0]
                mol.RemoveConformer(old_cid)
                pool[dominated_idx] = (new_cid, new_energy)
                n_replaced += 1
            else:
                pool.append((new_cid, new_energy))
                n_accepted += 1

            if new_energy < best_energy:
                best_energy = new_energy
                # Prune pool: remove stale conformers outside energy window
                pruned = []
                for pc, pe in pool:
                    if pe <= best_energy + energy_window:
                        pruned.append((pc, pe))
                    else:
                        mol.RemoveConformer(pc)
                if len(pruned) < len(pool):
                    pool = pruned

        # Progress logging every 20 steps (always runs, never skipped by continue)
        if (step + 1) % 20 == 0:
            print(f"  MCMM step {step+1}/{mcmm_steps}: pool={len(pool)}, best={best_energy:.1f} kcal/mol "
                  f"[+{n_accepted} new, {n_replaced} replaced, "
                  f"rej: {n_energy_reject} energy, {n_metropolis_reject} metropolis, {n_rmsd_reject} rmsd]", flush=True)

    # Sort pool by energy, return top max_conformers
    pool.sort(key=lambda x: x[1])
    selected = pool[:max_conformers]

    # Final summary with per-conformer details
    print(f"  MCMM complete: {len(selected)} conformers (from pool of {len(pool)})", flush=True)
    print(f"  Rejection breakdown: {n_energy_reject} energy window, "
          f"{n_metropolis_reject} Metropolis, {n_rmsd_reject} RMSD duplicate", flush=True)
    print(f"  Accepted: {n_accepted} new + {n_replaced} replacements", flush=True)
    for i, (cid, energy) in enumerate(selected):
        # Compute pairwise RMSD to first conformer for reference
        if i == 0:
            print(f"    conf {i}: E={energy:.2f} kcal/mol (global min)", flush=True)
        else:
            try:
                rmsd_to_best = GetBestRMS(mol, mol, selected[0][0], cid)
                print(f"    conf {i}: E={energy:.2f} kcal/mol, RMSD-to-best={rmsd_to_best:.2f} A", flush=True)
            except Exception:
                print(f"    conf {i}: E={energy:.2f} kcal/mol", flush=True)

    minimizer.cleanup()
    return mol, selected


def xtb_rerank_conformers(
    mol: Any,
    selected_conformers: List[Tuple[int, float]],
    xtb_binary: str,
    energy_window: float,
) -> List[Tuple[int, float]]:
    """Re-rank conformers by GFN2-xTB single-point energy with ALPB water.

    Replaces force-field energies with xTB energies and re-filters by energy window.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from score_xtb_strain import single_point, HARTREE_TO_KCAL

    print(f"  xTB reranking {len(selected_conformers)} conformers...", file=sys.stderr)

    # Write all conformers to temp SDFs first
    conf_files = []
    tmpdir = tempfile.mkdtemp(prefix='xtb_rerank_')
    for conf_id, ff_energy in selected_conformers:
        tmp_path = os.path.join(tmpdir, f'conf_{conf_id}.sdf')
        conf_mol = Chem.Mol(mol, confId=conf_id)
        writer = Chem.SDWriter(tmp_path)
        writer.write(conf_mol)
        writer.close()
        conf_files.append((conf_id, ff_energy, tmp_path))

    # Run xTB single-points in parallel (xTB is single-threaded with OMP_NUM_THREADS=1)
    def score_one(item: tuple) -> tuple:
        cid, ff_e, sdf_path = item
        try:
            e_hartree = single_point(xtb_binary, sdf_path, solvent='water')
            return (cid, e_hartree * HARTREE_TO_KCAL)
        except Exception as e:
            print(f"  Warning: xTB failed for conformer {cid}: {e}", file=sys.stderr)
            return (cid, ff_e)

    xtb_energies = []
    n_workers = min(4, len(conf_files))
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(score_one, item): item for item in conf_files}
        for future in as_completed(futures):
            xtb_energies.append(future.result())

    # Cleanup temp files
    shutil.rmtree(tmpdir, ignore_errors=True)

    if not xtb_energies:
        return selected_conformers

    # Re-sort by xTB energy and apply energy window
    xtb_energies.sort(key=lambda x: x[1])
    min_energy = xtb_energies[0][1]
    filtered = [(cid, e) for cid, e in xtb_energies if e - min_energy <= energy_window]

    print(f"  xTB reranking: {len(selected_conformers)} → {len(filtered)} conformers "
          f"(window={energy_window} kcal/mol)", file=sys.stderr)

    return filtered


def generate_conformers_crest(
    mol: Any,
    max_conformers: int,
    rmsd_cutoff: float,
    energy_window: float,
    crest_binary: str,
    xtb_binary: str,
    threads: int = 4,
) -> Tuple[Any, List[Tuple[int, float]]]:
    """Generate conformers using CREST (GFN2-xTB metadynamics).

    CREST performs exhaustive conformer searching at the semiempirical QM level,
    providing more diverse and better-ranked conformers than force-field methods.
    """
    with tempfile.TemporaryDirectory(prefix='crest_') as tmpdir:
        # Write input XYZ
        mol_h = Chem.AddHs(mol, addCoords=True)
        if mol_h.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol_h)

        input_xyz = os.path.join(tmpdir, 'input.xyz')
        conf = mol_h.GetConformer()
        n_atoms = mol_h.GetNumAtoms()
        with open(input_xyz, 'w') as f:
            f.write(f"{n_atoms}\n")
            f.write("input for CREST\n")
            for i in range(n_atoms):
                pos = conf.GetAtomPosition(i)
                sym = mol_h.GetAtomWithIdx(i).GetSymbol()
                f.write(f"{sym}  {pos.x:.8f}  {pos.y:.8f}  {pos.z:.8f}\n")

        # Resolve xTB environment
        xtb_root = str(Path(xtb_binary).parent.parent)
        xtb_share = os.path.join(xtb_root, 'share', 'xtb')
        env = os.environ.copy()
        if os.path.isdir(xtb_share):
            env['XTBPATH'] = xtb_share
        env['OMP_NUM_THREADS'] = str(threads)
        env['OMP_STACKSIZE'] = '1G'
        # CREST needs xTB on PATH
        env['PATH'] = str(Path(xtb_binary).parent) + ':' + env.get('PATH', '')

        # Run CREST
        cmd = [
            crest_binary, input_xyz,
            '--gfn2',
            '--alpb', 'water',
            '--ewin', str(energy_window),
            '--rthr', str(rmsd_cutoff),
            '--T', str(threads),
        ]

        print(f"  Running CREST: {' '.join(cmd[:6])}...", file=sys.stderr)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=tmpdir,
            env=env,
            timeout=600,
        )

        if result.returncode != 0:
            print(f"  CREST stderr: {result.stderr[:500]}", file=sys.stderr)
            raise RuntimeError(f"CREST failed (exit {result.returncode})")

        # Parse crest_conformers.xyz
        crest_output = os.path.join(tmpdir, 'crest_conformers.xyz')
        if not os.path.exists(crest_output):
            raise RuntimeError("CREST did not produce crest_conformers.xyz")

        # Parse multi-structure XYZ file
        conformer_data = []
        with open(crest_output) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                try:
                    n = int(line.strip())
                except ValueError:
                    continue
                comment = f.readline().strip()
                # Energy is typically in the comment line
                energy = 0.0
                for token in comment.split():
                    try:
                        energy = float(token)
                        break
                    except ValueError:
                        continue

                coords = []
                symbols = []
                for _ in range(n):
                    parts = f.readline().split()
                    symbols.append(parts[0])
                    coords.append((float(parts[1]), float(parts[2]), float(parts[3])))

                conformer_data.append((energy, symbols, coords))

        if not conformer_data:
            raise RuntimeError("No conformers parsed from CREST output")

        print(f"  CREST found {len(conformer_data)} conformers", file=sys.stderr)

        # Convert to RDKit mol with multiple conformers
        # Use the input mol as template for bond orders
        template = Chem.AddHs(mol, addCoords=True)

        # Limit to max_conformers
        conformer_data = conformer_data[:max_conformers]

        # Build mol with conformers
        result_mol = Chem.RWMol(template)
        result_mol.RemoveAllConformers()

        selected = []
        min_energy = conformer_data[0][0] if conformer_data else 0.0

        for idx, (energy, symbols, coords) in enumerate(conformer_data):
            # Energy window filter (CREST energies are in Hartree)
            e_kcal = (energy - min_energy) * 627.5095
            if e_kcal > energy_window:
                continue

            if len(coords) != result_mol.GetNumAtoms():
                # Atom count mismatch — skip
                continue

            new_conf = Chem.Conformer(result_mol.GetNumAtoms())
            for i, (x, y, z) in enumerate(coords):
                new_conf.SetAtomPosition(i, Point3D(x, y, z))
            conf_id = result_mol.AddConformer(new_conf, assignId=True)
            selected.append((conf_id, e_kcal))

        if not selected:
            raise RuntimeError("No conformers passed energy window filter")

        print(f"  Selected {len(selected)} conformers within {energy_window} kcal/mol window",
              file=sys.stderr)

        return result_mol.GetMol(), selected


def process_ligand(sdf_path: str, output_dir: str, max_conformers: int, rmsd_cutoff: float, energy_window: float, method: str = 'etkdg', mcmm_steps: int = 100, mcmm_temperature: float = 298.0, sample_amides: bool = True, xtb_rerank: bool = False, xtb_binary: str = '', crest_binary: str = '') -> List[Tuple[str, str]]:
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

    # Generate conformers using selected method
    if method == 'crest' and crest_binary and xtb_binary:
        mol_with_confs, selected_conformers = generate_conformers_crest(
            mol, max_conformers, rmsd_cutoff, energy_window,
            crest_binary, xtb_binary
        )
    elif method == 'mcmm':
        mol_with_confs, selected_conformers = generate_conformers_mcmm(
            mol, max_conformers, rmsd_cutoff, energy_window,
            mcmm_steps, mcmm_temperature, sample_amides
        )
    else:
        mol_with_confs, selected_conformers = generate_conformers_etkdg(
            mol, max_conformers, rmsd_cutoff, energy_window
        )

    # Optional xTB reranking (for ETKDG/MCMM — CREST already uses xTB)
    if xtb_rerank and xtb_binary and method != 'crest' and selected_conformers:
        selected_conformers = xtb_rerank_conformers(
            mol_with_confs, selected_conformers, xtb_binary, energy_window
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
    parser = argparse.ArgumentParser(description='Generate conformers for ligands using ETKDG or MCMM')
    parser.add_argument('--ligand_list', required=True, help='JSON file with list of SDF paths')
    parser.add_argument('--output_dir', required=True, help='Output directory for conformer SDFs')
    parser.add_argument('--max_conformers', type=int, default=10, help='Maximum conformers per molecule')
    parser.add_argument('--rmsd_cutoff', type=float, default=0.5, help='RMSD cutoff for diversity (Angstroms)')
    parser.add_argument('--energy_window', type=float, default=10.0, help='Energy window for filtering (kcal/mol)')
    parser.add_argument('--method', choices=['etkdg', 'mcmm', 'crest'], default='etkdg', help='Conformer generation method')
    parser.add_argument('--mcmm_steps', type=int, default=100, help='MCMM search steps (MCMM only)')
    parser.add_argument('--mcmm_temperature', type=float, default=298.0, help='MCMM temperature in K (MCMM only)')
    parser.add_argument('--sample_amides', action='store_true', help='Sample amide cis/trans rotations (MCMM only)')
    parser.add_argument('--xtb_rerank', action='store_true', help='Re-rank conformers by GFN2-xTB energy')
    parser.add_argument('--xtb_binary', type=str, default='', help='Path to xtb executable')
    parser.add_argument('--crest_binary', type=str, default='', help='Path to crest executable')
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

    method_label = args.method.upper()
    print(f"=== Conformer Generation ({method_label}) ===")
    print(f"Input molecules: {len(ligand_paths)}")
    print(f"Max conformers: {args.max_conformers}")
    print(f"RMSD cutoff: {args.rmsd_cutoff} A")
    print(f"Energy window: {args.energy_window} kcal/mol")
    if args.method == 'mcmm':
        print(f"MCMM steps: {args.mcmm_steps}")
        print(f"Temperature: {args.mcmm_temperature} K")
        print(f"Sample amides: {args.sample_amides}")
    if args.method == 'crest':
        print(f"CREST binary: {args.crest_binary}")
        print(f"xTB binary: {args.xtb_binary}")
    if args.xtb_rerank:
        print(f"xTB reranking: enabled")
    print()

    # Process each ligand
    all_conformer_paths = []
    parent_mapping = {}

    for i, sdf_path in enumerate(ligand_paths):
        print(f"Processing {i+1}/{len(ligand_paths)}: {os.path.basename(sdf_path)}")

        results = process_ligand(
            sdf_path, args.output_dir,
            args.max_conformers, args.rmsd_cutoff, args.energy_window,
            method=args.method,
            mcmm_steps=args.mcmm_steps,
            mcmm_temperature=args.mcmm_temperature,
            sample_amides=args.sample_amides,
            xtb_rerank=args.xtb_rerank,
            xtb_binary=args.xtb_binary,
            crest_binary=args.crest_binary,
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
