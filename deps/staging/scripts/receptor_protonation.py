#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
Shared receptor protonation helpers for Dock and MD.

Provides:
  - pocket-residue detection from a crystallographic ligand
  - reduce side-chain flip optimization
  - PROPKA-driven shifted-residue detection
  - explicit OpenMM Modeller hydrogen-variant selection
  - sidecar metadata persistence for reuse in downstream MD prep
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from openmm.app import ForceField, Modeller, PDBFile
except ImportError as exc:
    raise RuntimeError(f"OpenMM is required for receptor protonation helpers: {exc}") from exc

try:
    from pdbfixer import PDBFixer
except ImportError as exc:
    raise RuntimeError(f"PDBFixer is required for receptor protonation helpers: {exc}") from exc

try:
    from propka.run import single as propka_single
    HAS_PROPKA = True
except ImportError:
    HAS_PROPKA = False


POCKET_RESIDUE_CUTOFF_A = 6.0
DISULFIDE_SG_CUTOFF_A = 2.3
SUPPORTED_FAMILIES = {"ASP", "GLU", "HIS", "LYS", "CYS", "TYR"}
STANDARD_PKA = {
    "ASP": 3.8,
    "GLU": 4.5,
    "HIS": 6.5,
    "LYS": 10.5,
    "CYS": 8.3,
    "TYR": 10.1,
}
PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "ASH", "CYS", "CYX", "GLN", "GLU", "GLH", "GLY", "HIS", "HID", "HIE",
    "HIP", "HIN", "ILE", "LEU", "LYS", "LYN", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "ACE", "NME",
}
FAMILY_BY_RESNAME = {
    "ASP": "ASP", "ASH": "ASP",
    "GLU": "GLU", "GLH": "GLU",
    "HIS": "HIS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HIN": "HIS",
    "LYS": "LYS", "LYN": "LYS",
    "CYS": "CYS", "CYX": "CYS",
    "TYR": "TYR",
}


def metadata_path_for_receptor(receptor_pdb_path: str) -> str:
    base, _ext = os.path.splitext(receptor_pdb_path)
    return f"{base}.prep.json"


def load_receptor_prep_metadata(receptor_pdb_path: str) -> Optional[Dict[str, Any]]:
    metadata_path = metadata_path_for_receptor(receptor_pdb_path)
    if not os.path.exists(metadata_path):
        return None
    try:
        with open(metadata_path, "r") as handle:
            return json.load(handle)
    except Exception as exc:
        print(f"Warning: Failed to read receptor prep metadata {metadata_path}: {exc}", file=sys.stderr)
        return None


def _parse_residue_number(residue_id: str) -> str:
    match = re.match(r"^\s*(-?\d+)", str(residue_id).strip())
    return match.group(1) if match else str(residue_id).strip()


def residue_key(chain_id: str, residue_number: str, insertion_code: str = "") -> str:
    chain = (chain_id or "_").strip() or "_"
    resnum = _parse_residue_number(residue_number)
    insertion = (insertion_code or "").strip()
    return f"{chain}:{resnum}:{insertion}"


def topology_residue_key(residue: Any) -> str:
    return residue_key(residue.chain.id, residue.id, getattr(residue, "insertionCode", ""))


def residue_family(residue_name: str) -> str:
    return FAMILY_BY_RESNAME.get((residue_name or "").strip().upper(), (residue_name or "").strip().upper())


def build_protein_forcefield() -> Any:
    try:
        return ForceField(
            "amber/protein.ff19SB.xml",
            "amber/tip3p_standard.xml",
            "amber/tip3p_HFE_multivalent.xml",
        )
    except Exception:
        return ForceField("amber/protein.ff19SB.xml", "amber/tip3p_standard.xml")


def _add_gbsa_obc2_protein(system: Any, topology: Any) -> None:
    """Add GBSA-OBC2 implicit solvent for protein-only systems.

    Inline version of utils.add_gbsa_obc2_force (canonical source) specialised
    for receptor prep — no RDKit mol needed, uses topology bonds for H-N lookup.
    """
    import openmm
    import openmm.unit as unit

    RADII_NM = {
        'H': 0.12, 'C': 0.17, 'N': 0.155, 'O': 0.15, 'F': 0.15,
        'S': 0.18, 'P': 0.185, 'Cl': 0.17, 'Br': 0.185, 'I': 0.198,
        'Na': 0.102, 'K': 0.138, 'Mg': 0.072, 'Ca': 0.10, 'Zn': 0.074,
        'Fe': 0.064, 'Mn': 0.067,
    }
    SCREEN = {
        'H': 0.85, 'C': 0.72, 'N': 0.79, 'O': 0.85, 'F': 0.88,
        'S': 0.96, 'P': 0.86, 'Cl': 0.80, 'Br': 0.80, 'I': 0.80,
        'Na': 0.80, 'K': 0.80, 'Mg': 0.80, 'Ca': 0.80, 'Zn': 0.80,
        'Fe': 0.80, 'Mn': 0.80,
    }

    nb_force = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nb_force = force
            break
    if nb_force is None:
        return

    h_bonded_to_n: set = set()
    for a1, a2 in topology.bonds():
        sym1 = a1.element.symbol if a1.element else ''
        sym2 = a2.element.symbol if a2.element else ''
        if sym1 == 'H' and sym2 == 'N':
            h_bonded_to_n.add(a1.index)
        elif sym2 == 'H' and sym1 == 'N':
            h_bonded_to_n.add(a2.index)

    gbsa = openmm.GBSAOBCForce()
    gbsa.setSolventDielectric(78.5)
    gbsa.setSoluteDielectric(1.0)
    gbsa.setNonbondedMethod(openmm.GBSAOBCForce.NoCutoff)

    for idx, atom in enumerate(topology.atoms()):
        charge, _sigma, _epsilon = nb_force.getParticleParameters(idx)
        q = charge.value_in_unit(unit.elementary_charge)
        symbol = atom.element.symbol if atom.element else 'C'
        radius = RADII_NM.get(symbol, 0.15)
        screen = SCREEN.get(symbol, 0.80)
        if symbol == 'H' and idx in h_bonded_to_n:
            radius = 0.13  # mbondi3 adjustment
        gbsa.addParticle(q, radius, screen)

    system.addForce(gbsa)


def _minimize_hydrogens(
    topology: Any,
    positions: Any,
    force_constant_kcal: float = 50.0,
    max_iterations: int = 500,
    tolerance: float = 1.0,
) -> Tuple[Any, Dict[str, Any]]:
    """Minimize hydrogen positions while restraining all heavy atoms.

    Uses ff19SB + OBC2 implicit solvent (vacuum fallback) with Cartesian
    heavy-atom restraints.  Returns (minimized_positions, report_dict).
    """
    import time
    import openmm
    from openmm import app as omm_app
    import openmm.unit as unit

    try:
        t0 = time.monotonic()

        # --- Build force field system (with CYS/CYX fallback) ---
        ff = build_protein_forcefield()
        try:
            system = ff.createSystem(
                topology,
                nonbondedMethod=omm_app.NoCutoff,
                constraints=None,
                rigidWater=False,
            )
        except Exception:
            cys_templates = {}
            for res in topology.residues():
                if res.name in ('CYS', 'CYX', 'CYM'):
                    atom_names = {a.name for a in res.atoms()}
                    cys_templates[res] = 'CYS' if 'HG' in atom_names else 'CYX'
            system = ff.createSystem(
                topology,
                nonbondedMethod=omm_app.NoCutoff,
                constraints=None,
                rigidWater=False,
                ignoreExternalBonds=True,
                residueTemplates=cys_templates,
            )

        # --- OBC2 implicit solvent (skip for very large systems) ---
        used_gbsa = False
        n_atoms = topology.getNumAtoms()
        if n_atoms <= 15000:
            try:
                _add_gbsa_obc2_protein(system, topology)
                used_gbsa = True
            except Exception as exc:
                print(f"  Warning: OBC2 setup failed ({exc}), minimizing in vacuum",
                      file=sys.stderr)
        else:
            print(f"  Skipping OBC2 for large system ({n_atoms} atoms), using vacuum",
                  file=sys.stderr)

        # --- Heavy-atom Cartesian restraints ---
        k_kjmol_nm2 = force_constant_kcal * 4.184 * 100.0
        restraint = openmm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        restraint.addGlobalParameter("k", k_kjmol_nm2)
        restraint.addPerParticleParameter("x0")
        restraint.addPerParticleParameter("y0")
        restraint.addPerParticleParameter("z0")
        for idx, atom in enumerate(topology.atoms()):
            if atom.element is not None and atom.element.symbol != 'H':
                pos = positions[idx]
                restraint.addParticle(idx, [pos[0], pos[1], pos[2]])
        system.addForce(restraint)

        # --- Minimise ---
        integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        platform = openmm.Platform.getPlatformByName("CPU")
        context = openmm.Context(system, integrator, platform)
        context.setPositions(positions)

        energy_before = (context.getState(getEnergy=True)
                         .getPotentialEnergy()
                         .value_in_unit(unit.kilojoules_per_mole))

        openmm.LocalEnergyMinimizer.minimize(
            context, tolerance=tolerance, maxIterations=max_iterations,
        )

        state_after = context.getState(getEnergy=True, getPositions=True)
        energy_after = state_after.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        minimized_positions = state_after.getPositions()

        elapsed = time.monotonic() - t0
        return minimized_positions, {
            "applied": True,
            "energy_before_kjmol": round(energy_before, 1),
            "energy_after_kjmol": round(energy_after, 1),
            "energy_reduction_kjmol": round(energy_before - energy_after, 1),
            "used_gbsa": used_gbsa,
            "wall_time_s": round(elapsed, 2),
            "max_iterations": max_iterations,
            "force_constant_kcal_mol_a2": force_constant_kcal,
        }

    except Exception as exc:
        print(f"  Warning: hydrogen minimization failed ({exc}), using template positions",
              file=sys.stderr)
        return positions, {"applied": False, "error": str(exc)}


def _evaluate_variant_energy(
    topology: Any,
    positions: Any,
    protonation_ph: float,
    variants: Sequence[Optional[str]],
    max_iterations: int = 50,
) -> Optional[float]:
    """Add hydrogens with the given variant list and return minimized energy (kJ/mol).

    Returns None if hydrogen addition or minimization fails.
    """
    try:
        topo, pos, _ = add_hydrogens_with_variants(
            topology, positions, protonation_ph, variants,
        )
        _, report = _minimize_hydrogens(topo, pos, max_iterations=max_iterations)
        if report.get("applied"):
            return report["energy_after_kjmol"]
        return None
    except Exception:
        return None


def score_histidine_variants(
    topology: Any,
    positions: Any,
    protonation_ph: float,
    variant_plan: Dict[str, Any],
    energy_threshold_kjmol: float = 4.0,
) -> Dict[str, Any]:
    """Score HID vs HIE for each neutral histidine using short energy minimization.

    For each neutral HIS in the variant plan, builds the full system twice (once
    with HID, once with HIE), runs a 50-iteration L-BFGS minimization, and picks
    the lower-energy tautomer if the difference exceeds the threshold.

    Returns a dict with 'variants_changed', 'updated_plan', and 'his_scoring_report'.
    """
    import time

    resolved = variant_plan["resolved_variants"]
    variants = list(variant_plan["variants"])
    residues = list(topology.residues())

    # Find neutral HIS residues (HID or HIE) and their indices
    neutral_his: List[Tuple[str, str, int]] = []  # (residue_key, current_variant, residue_index)
    for res in residues:
        family = residue_family(res.name)
        if family != "HIS":
            continue
        key = topology_residue_key(res)
        current = resolved.get(key)
        if current in ("HID", "HIE"):
            neutral_his.append((key, current, res.index))

    report: Dict[str, Any] = {
        "scored_count": len(neutral_his),
        "changes_made": 0,
        "wall_time_s": 0.0,
        "per_residue": [],
    }

    if not neutral_his:
        return {"variants_changed": False, "updated_plan": variant_plan, "his_scoring_report": report}

    t0 = time.monotonic()
    changes_made = 0

    for key, current_variant, res_idx in neutral_his:
        # Build variant lists for HID and HIE
        variants_hid = list(variants)
        variants_hid[res_idx] = "HID"
        variants_hie = list(variants)
        variants_hie[res_idx] = "HIE"

        energy_hid = _evaluate_variant_energy(topology, positions, protonation_ph, variants_hid)
        energy_hie = _evaluate_variant_energy(topology, positions, protonation_ph, variants_hie)

        entry: Dict[str, Any] = {
            "residue_key": key,
            "geometric_pick": current_variant,
            "hid_energy_kjmol": round(energy_hid, 1) if energy_hid is not None else None,
            "hie_energy_kjmol": round(energy_hie, 1) if energy_hie is not None else None,
        }

        # Compare energies and decide
        if energy_hid is not None and energy_hie is not None:
            delta = energy_hie - energy_hid  # positive means HID is lower
            entry["delta_kjmol"] = round(delta, 1)
            if delta > energy_threshold_kjmol:
                best = "HID"
            elif delta < -energy_threshold_kjmol:
                best = "HIE"
            else:
                best = current_variant  # keep geometric pick within threshold

            entry["final_pick"] = best
            entry["changed"] = best != current_variant

            if best != current_variant:
                variants[res_idx] = best
                resolved[key] = best
                changes_made += 1
                print(f"  HIS {key}: {current_variant} → {best} (delta={delta:.1f} kJ/mol)",
                      file=sys.stderr)
        else:
            entry["delta_kjmol"] = None
            entry["final_pick"] = current_variant
            entry["changed"] = False

        report["per_residue"].append(entry)

    report["changes_made"] = changes_made
    report["wall_time_s"] = round(time.monotonic() - t0, 2)

    updated_plan = dict(variant_plan)
    updated_plan["variants"] = variants
    updated_plan["resolved_variants"] = resolved

    return {
        "variants_changed": changes_made > 0,
        "updated_plan": updated_plan,
        "his_scoring_report": report,
    }


def identify_pocket_residue_keys_from_pdb(
    pdb_path: str,
    ligand_coords: Sequence[Tuple[float, float, float]],
    cutoff_a: float = POCKET_RESIDUE_CUTOFF_A,
) -> Set[str]:
    if not ligand_coords:
        return set()

    cutoff_sq = cutoff_a * cutoff_a
    pocket_keys: Set[str] = set()
    with open(pdb_path, "r") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            resname = line[17:20].strip().upper()
            if residue_family(resname) not in PROTEIN_RESIDUES and resname not in PROTEIN_RESIDUES:
                continue
            chain = line[21].strip() or "_"
            resnum = line[22:26].strip()
            insertion = line[26].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            for lx, ly, lz in ligand_coords:
                dx = x - lx
                dy = y - ly
                dz = z - lz
                if dx * dx + dy * dy + dz * dz <= cutoff_sq:
                    pocket_keys.add(residue_key(chain, resnum, insertion))
                    break
    return pocket_keys


def run_reduce_if_available(input_pdb: str) -> Tuple[str, Dict[str, Any]]:
    result = {
        "reduce_available": False,
        "reduce_applied": False,
        "reduce_output_path": input_pdb,
    }
    reduce_bin = shutil.which("reduce")
    if not reduce_bin:
        return input_pdb, result

    result["reduce_available"] = True
    fd, reduced_path = tempfile.mkstemp(prefix="reduce_", suffix=".pdb")
    os.close(fd)
    try:
        proc = subprocess.run(
            [reduce_bin, "-FLIP", "-Quiet", input_pdb],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode == 0 and proc.stdout:
            with open(reduced_path, "w") as handle:
                handle.write(proc.stdout)
            result["reduce_applied"] = True
            result["reduce_output_path"] = reduced_path
            return reduced_path, result
    except Exception as exc:
        print(f"Warning: reduce failed: {exc}", file=sys.stderr)

    try:
        os.remove(reduced_path)
    except OSError:
        pass
    return input_pdb, result


def _default_state_for_family(family: str, protonation_ph: float) -> str:
    if family == "HIS":
        return "protonated" if protonation_ph <= STANDARD_PKA["HIS"] else "neutral"
    if family in {"ASP", "GLU", "CYS", "TYR"}:
        return "protonated" if protonation_ph <= STANDARD_PKA[family] else "deprotonated"
    if family == "LYS":
        return "protonated" if protonation_ph <= STANDARD_PKA["LYS"] else "deprotonated"
    return "unknown"


def collect_propka_shifted_residues(input_pdb: str, protonation_ph: float) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "propka_available": HAS_PROPKA,
        "shifted_residues": [],
        "propka_input_pdb": input_pdb,
    }
    if not HAS_PROPKA:
        return report

    try:
        mol = propka_single(input_pdb)
    except Exception as exc:
        report["propka_available"] = False
        report["propka_error"] = str(exc)
        return report

    for conformation_name in mol.conformations:
        conformation = mol.conformations[conformation_name]
        for group in conformation.get_titratable_groups():
            family = residue_family(group.residue_type)
            if family not in SUPPORTED_FAMILIES:
                continue
            if group.pka_value is None:
                continue
            default_state = _default_state_for_family(family, protonation_ph)
            if family == "HIS":
                propka_state = "protonated" if group.pka_value > protonation_ph else "neutral"
            else:
                propka_state = "protonated" if group.pka_value > protonation_ph else "deprotonated"
            if propka_state == default_state:
                continue
            # chain_id/res_num live on group.atom, not the group itself
            g_chain = group.atom.chain_id
            g_resnum = group.atom.res_num
            report["shifted_residues"].append({
                "residue_key": residue_key(g_chain, str(g_resnum)),
                "chain_id": g_chain,
                "residue_number": str(g_resnum),
                "residue_name": family,
                "pka": float(group.pka_value),
                "default_state": default_state,
                "propka_state": propka_state,
            })
        break
    return report


def detect_disulfide_keys(topology: Any, positions: Any) -> Set[str]:
    sulfur_atoms: List[Tuple[str, Any]] = []
    for residue in topology.residues():
        if residue_family(residue.name) != "CYS":
            continue
        for atom in residue.atoms():
            if atom.name == "SG":
                sulfur_atoms.append((topology_residue_key(residue), atom))
                break

    disulfide_keys: Set[str] = set()
    for i, (key_i, atom_i) in enumerate(sulfur_atoms):
        pos_i = positions[atom_i.index]
        for key_j, atom_j in sulfur_atoms[i + 1:]:
            pos_j = positions[atom_j.index]
            dx = pos_i.x - pos_j.x
            dy = pos_i.y - pos_j.y
            dz = pos_i.z - pos_j.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz) * 10.0
            if dist <= DISULFIDE_SG_CUTOFF_A:
                disulfide_keys.add(key_i)
                disulfide_keys.add(key_j)
    return disulfide_keys


def _nearest_acceptor_distance(atom_name: str, residue: Any, topology: Any, positions: Any) -> float:
    atom_obj = None
    for atom in residue.atoms():
        if atom.name == atom_name:
            atom_obj = atom
            break
    if atom_obj is None:
        return float("inf")
    pos = positions[atom_obj.index]
    best = float("inf")
    for atom in topology.atoms():
        if atom.residue == residue:
            continue
        if atom.element is None or atom.element.symbol not in {"N", "O"}:
            continue
        other = positions[atom.index]
        dx = pos.x - other.x
        dy = pos.y - other.y
        dz = pos.z - other.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz) * 10.0
        if dist < best:
            best = dist
    return best


def choose_neutral_histidine_variant(residue: Any, topology: Any, positions: Any) -> str:
    nd1_best = _nearest_acceptor_distance("ND1", residue, topology, positions)
    ne2_best = _nearest_acceptor_distance("NE2", residue, topology, positions)
    return "HID" if nd1_best <= ne2_best else "HIE"


def default_variant_for_residue(
    residue: Any,
    topology: Any,
    positions: Any,
    protonation_ph: float,
    disulfide_keys: Set[str],
) -> Optional[str]:
    family = residue_family(residue.name)
    key = topology_residue_key(residue)
    if family == "ASP":
        return "ASH" if protonation_ph <= STANDARD_PKA["ASP"] else "ASP"
    if family == "GLU":
        return "GLH" if protonation_ph <= STANDARD_PKA["GLU"] else "GLU"
    if family == "LYS":
        return "LYS" if protonation_ph <= STANDARD_PKA["LYS"] else "LYN"
    if family == "HIS":
        if protonation_ph <= STANDARD_PKA["HIS"]:
            return "HIP"
        return choose_neutral_histidine_variant(residue, topology, positions)
    if family == "CYS":
        return "CYX" if key in disulfide_keys else "CYS"
    if family == "TYR":
        # Neutral tyrosine is handled by Modeller's default logic. Unlike
        # residues such as LYS/CYS/HIS, there is no explicit "TYR" hydrogen
        # variant token we can pass in the variants list.
        return None
    return None


def desired_variant_from_propka(
    residue: Any,
    topology: Any,
    positions: Any,
    protonation_ph: float,
    shifted_record: Dict[str, Any],
    disulfide_keys: Set[str],
) -> Tuple[Optional[str], Optional[str]]:
    family = residue_family(residue.name)
    key = topology_residue_key(residue)
    pka = shifted_record["pka"]

    if family == "ASP":
        return ("ASH" if pka > protonation_ph else "ASP"), None
    if family == "GLU":
        return ("GLH" if pka > protonation_ph else "GLU"), None
    if family == "LYS":
        return ("LYS" if pka > protonation_ph else "LYN"), None
    if family == "HIS":
        if pka > protonation_ph:
            return "HIP", None
        return choose_neutral_histidine_variant(residue, topology, positions), None
    if family == "CYS":
        if pka > protonation_ph:
            return "CYS", None
        if key in disulfide_keys:
            return "CYX", None
        return None, "cys_thiolate_requires_disulfide"
    if family == "TYR":
        return None, "tyr_deprotonation_not_supported"
    return None, "unsupported_residue_family"


def build_variant_plan(
    topology: Any,
    positions: Any,
    protonation_ph: float,
    pocket_residue_keys: Optional[Set[str]] = None,
    shifted_residues: Optional[Sequence[Dict[str, Any]]] = None,
    resolved_variant_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    residues = list(topology.residues())
    variants: List[Optional[str]] = [None] * len(residues)
    disulfide_keys = detect_disulfide_keys(topology, positions)
    shifted_by_key = {entry["residue_key"]: entry for entry in (shifted_residues or [])}

    resolved_variants: Dict[str, str] = {}
    applied_overrides: List[Dict[str, Any]] = []
    ignored_shifted: List[Dict[str, Any]] = []
    pocket_keys = pocket_residue_keys or set()

    for residue in residues:
        family = residue_family(residue.name)
        if family not in SUPPORTED_FAMILIES:
            continue

        key = topology_residue_key(residue)
        default_variant = default_variant_for_residue(residue, topology, positions, protonation_ph, disulfide_keys)
        selected_variant = default_variant
        override_reason: Optional[str] = None
        shifted_record = shifted_by_key.get(key)

        if resolved_variant_overrides and key in resolved_variant_overrides:
            selected_variant = resolved_variant_overrides[key]
            override_reason = "metadata_reuse"
        elif shifted_record is not None:
            if pocket_keys and key not in pocket_keys:
                ignored = dict(shifted_record)
                ignored["reason"] = "outside_pocket"
                ignored_shifted.append(ignored)
            else:
                desired_variant, ignore_reason = desired_variant_from_propka(
                    residue,
                    topology,
                    positions,
                    protonation_ph,
                    shifted_record,
                    disulfide_keys,
                )
                if desired_variant is None:
                    ignored = dict(shifted_record)
                    ignored["reason"] = ignore_reason or "unsupported_shifted_residue"
                    ignored_shifted.append(ignored)
                else:
                    selected_variant = desired_variant
                    override_reason = "propka_shifted"

        if selected_variant is None:
            continue

        resolved_variants[key] = selected_variant
        variants[residue.index] = selected_variant

        if override_reason is not None and selected_variant != default_variant:
            applied_entry = {
                "residue_key": key,
                "chain_id": residue.chain.id,
                "residue_number": _parse_residue_number(residue.id),
                "residue_name": family,
                "selected_variant": selected_variant,
                "default_variant": default_variant,
                "reason": override_reason,
            }
            if shifted_record is not None:
                applied_entry["pka"] = shifted_record["pka"]
            applied_overrides.append(applied_entry)

    return {
        "variants": variants,
        "resolved_variants": resolved_variants,
        "applied_overrides": applied_overrides,
        "ignored_shifted_residues": ignored_shifted,
        "disulfide_residue_keys": sorted(disulfide_keys),
    }


def _sanitize_positions(positions: Any) -> Any:
    """Flatten positions to avoid nested Quantity objects that crash OpenMM's addHydrogens.

    PDBFixer.addMissingAtoms() can produce positions where individual elements are
    Quantity objects instead of Vec3, causing an AssertionError in Quantity.__getitem__.
    """
    from openmm import unit as u
    try:
        # If positions is already a flat Quantity(list-of-Vec3, unit), this is a no-op
        test = positions[0]
        if hasattr(test, 'unit'):
            # Nested Quantity — flatten by extracting value_in_unit
            flat = [pos.value_in_unit(u.nanometers) for pos in positions]
            return u.Quantity(flat, u.nanometers)
    except Exception:
        pass
    return positions


def add_hydrogens_with_variants(
    topology: Any,
    positions: Any,
    protonation_ph: float,
    variants: Sequence[Optional[str]],
) -> Tuple[Any, Any, List[Optional[str]]]:
    positions = _sanitize_positions(positions)
    modeller = Modeller(topology, positions)
    try:
        actual_variants = modeller.addHydrogens(
            forcefield=build_protein_forcefield(),
            pH=protonation_ph,
            variants=list(variants),
        )
    except Exception as exc:
        print(
            f"Warning: forcefield-guided hydrogen placement failed ({exc}); retrying without forcefield templates",
            file=sys.stderr,
        )
        try:
            modeller = Modeller(topology, positions)
            actual_variants = modeller.addHydrogens(
                pH=protonation_ph,
                variants=list(variants),
            )
        except Exception as exc2:
            print(
                f"Warning: variant-guided hydrogen placement also failed ({exc2}); retrying with default protonation",
                file=sys.stderr,
            )
            modeller = Modeller(topology, positions)
            actual_variants = modeller.addHydrogens(pH=protonation_ph)
    return modeller.topology, modeller.positions, actual_variants


def _rename_residue_variants_in_pdb(output_path: str, resolved_variants: Dict[str, str]) -> None:
    if not resolved_variants:
        return

    rewritten: List[str] = []
    with open(output_path, "r") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM", "ANISOU", "TER")) and len(line) >= 27:
                chain = line[21].strip() or "_"
                resnum = line[22:26].strip()
                insertion = line[26].strip()
                key = residue_key(chain, resnum, insertion)
                variant = resolved_variants.get(key)
                if variant is not None and len(variant) == 3:
                    line = f"{line[:17]}{variant:>3}{line[20:]}"
            rewritten.append(line)

    with open(output_path, "w") as handle:
        handle.writelines(rewritten)


def write_prepared_receptor_pdb(
    topology: Any,
    positions: Any,
    output_path: str,
    resolved_variants: Dict[str, str],
) -> None:
    with open(output_path, "w") as handle:
        PDBFile.writeFile(topology, positions, handle, keepIds=True)
    _rename_residue_variants_in_pdb(output_path, resolved_variants)


def write_receptor_prep_metadata(output_path: str, metadata: Dict[str, Any]) -> str:
    metadata_path = metadata_path_for_receptor(output_path)
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata_path


def prepare_receptor_with_propka(
    input_pdb: str,
    output_path: str,
    protonation_ph: float,
    pocket_residue_keys: Optional[Set[str]] = None,
    *,
    fixer: Optional[Any] = None,
    propka_report: Optional[Dict[str, Any]] = None,
    reduce_report: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    keep_terminal_missing: bool = False,
    minimize_hydrogens: bool = True,
) -> Dict[str, Any]:
    """Unified receptor preparation: PDBFixer + PROPKA-guided protonation.

    When called with no keyword args (docking path), runs the full pipeline
    from scratch.  When called with fixer= (MD path via prepare_receptor.py),
    the caller has already run PDBFixer with chain-break handling.
    """
    reduced_path: Optional[str] = None

    if fixer is None:
        # Self-contained path: Reduce -> PROPKA -> PDBFixer from scratch
        reduced_path, reduce_report = run_reduce_if_available(input_pdb)
        if propka_report is None:
            propka_report = collect_propka_shifted_residues(reduced_path, protonation_ph)

        fixer = PDBFixer(filename=reduced_path)
        fixer.findMissingResidues()
        # Remove terminal missing residues — disordered tails absent from
        # crystal structures produce unreliable modelled geometry.  Only
        # fill internal gaps (missing loops).
        if not keep_terminal_missing:
            chains = list(fixer.topology.chains())
            terminal_keys = [
                key for key in fixer.missingResidues
                if key[1] == 0 or key[1] == len(list(chains[key[0]].residues()))
            ]
            for key in terminal_keys:
                del fixer.missingResidues[key]
            if terminal_keys:
                print(f"  Skipping {len(terminal_keys)} terminal disordered region(s)", file=sys.stderr)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
    else:
        # External fixer path: caller already ran PDBFixer + chain breaks.
        # Sanitize positions here — chain break splitting can produce nested
        # Quantity objects that crash addHydrogens.
        fixer.positions = _sanitize_positions(fixer.positions)
        if reduce_report is None:
            reduce_report = {"reduce_available": False, "reduce_applied": False}
        if propka_report is None:
            propka_report = {"propka_available": False, "shifted_residues": []}

    variant_plan = build_variant_plan(
        fixer.topology,
        fixer.positions,
        protonation_ph,
        pocket_residue_keys=pocket_residue_keys,
        shifted_residues=propka_report.get("shifted_residues", []),
    )

    # HIS tautomer energy scoring — evaluate HID vs HIE for each neutral histidine
    his_scoring_report: Dict[str, Any] = {"scored_count": 0}
    if minimize_hydrogens:
        try:
            scoring_result = score_histidine_variants(
                fixer.topology, fixer.positions, protonation_ph, variant_plan,
            )
            his_scoring_report = scoring_result.get("his_scoring_report", his_scoring_report)
            if scoring_result.get("variants_changed"):
                variant_plan = scoring_result["updated_plan"]
                print(f"  HIS scoring: {his_scoring_report['changes_made']} tautomer(s) swapped",
                      file=sys.stderr)
            elif his_scoring_report["scored_count"] > 0:
                print(f"  HIS scoring: {his_scoring_report['scored_count']} evaluated, geometric picks confirmed",
                      file=sys.stderr)
        except Exception as exc:
            print(f"  Warning: HIS tautomer scoring failed ({exc}), using geometric picks",
                  file=sys.stderr)

    protonated_topology, protonated_positions, actual_variants = add_hydrogens_with_variants(
        fixer.topology,
        fixer.positions,
        protonation_ph,
        variant_plan["variants"],
    )

    # Hydrogen minimization — relaxes template-placed H with heavy atoms frozen.
    h_min_report: Dict[str, Any] = {"applied": False}
    if minimize_hydrogens:
        print("  Minimizing hydrogen positions (ff19SB + OBC2)...", file=sys.stderr)
        protonated_positions, h_min_report = _minimize_hydrogens(
            protonated_topology, protonated_positions,
        )
        if h_min_report.get("applied"):
            reduction = h_min_report.get("energy_reduction_kjmol", 0)
            wall = h_min_report.get("wall_time_s", 0)
            print(f"  H-min: {reduction:.0f} kJ/mol reduction in {wall:.1f}s",
                  file=sys.stderr)

    write_prepared_receptor_pdb(
        protonated_topology,
        protonated_positions,
        output_path,
        variant_plan["resolved_variants"],
    )

    metadata: Dict[str, Any] = {
        "schema_version": 1,
        "prepared_receptor_pdb": output_path,
        "receptor_protonation_ph": protonation_ph,
        "pocket_filtered": pocket_residue_keys is not None,
        "pocket_cutoff_angstrom": POCKET_RESIDUE_CUTOFF_A,
        "pocket_residue_keys": sorted(pocket_residue_keys or []),
        "reduce_available": reduce_report["reduce_available"],
        "reduce_applied": reduce_report["reduce_applied"],
        "propka_available": propka_report.get("propka_available", False),
        "propka_error": propka_report.get("propka_error"),
        "propka_shifted_residues": propka_report.get("shifted_residues", []),
        "applied_overrides": variant_plan["applied_overrides"],
        "ignored_shifted_residues": variant_plan["ignored_shifted_residues"],
        "resolved_variants": variant_plan["resolved_variants"],
        "actual_variants": actual_variants,
        "disulfide_residue_keys": variant_plan["disulfide_residue_keys"],
        "prepared_chain_count": sum(1 for _ in protonated_topology.chains()),
        "prepared_residue_count": sum(1 for _ in protonated_topology.residues()),
        "prepared_atom_count": protonated_topology.getNumAtoms(),
        "hydrogen_minimization": h_min_report,
        "his_tautomer_scoring": his_scoring_report,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata["metadata_path"] = write_receptor_prep_metadata(output_path, metadata)

    # Clean up reduce temp file (self-contained path only)
    if reduced_path is not None and reduced_path != input_pdb:
        try:
            os.remove(reduced_path)
        except OSError:
            pass

    return metadata


