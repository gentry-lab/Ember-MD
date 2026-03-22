#!/usr/bin/env python3
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
            report["shifted_residues"].append({
                "residue_key": residue_key(group.chain_id, str(group.res_num)),
                "chain_id": group.chain_id,
                "residue_number": str(group.res_num),
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
    protonated_topology, protonated_positions, actual_variants = add_hydrogens_with_variants(
        fixer.topology,
        fixer.positions,
        protonation_ph,
        variant_plan["variants"],
    )
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


def protonate_existing_prepared_receptor(
    topology: Any,
    positions: Any,
    protonation_ph: float,
    prep_metadata: Optional[Dict[str, Any]],
) -> Tuple[Any, Any, Dict[str, Any]]:
    resolved_variant_overrides = None
    if prep_metadata is not None:
        resolved_variant_overrides = prep_metadata.get("resolved_variants")

    variant_plan = build_variant_plan(
        topology,
        positions,
        protonation_ph,
        resolved_variant_overrides=resolved_variant_overrides,
    )
    protonated_topology, protonated_positions, actual_variants = add_hydrogens_with_variants(
        topology,
        positions,
        protonation_ph,
        variant_plan["variants"],
    )
    metadata = {
        "receptor_protonation_ph": protonation_ph,
        "propka_available": bool(prep_metadata and prep_metadata.get("propka_available")),
        "applied_overrides": variant_plan["applied_overrides"],
        "ignored_shifted_residues": variant_plan["ignored_shifted_residues"],
        "resolved_variants": variant_plan["resolved_variants"],
        "actual_variants": actual_variants,
    }
    return protonated_topology, protonated_positions, metadata
