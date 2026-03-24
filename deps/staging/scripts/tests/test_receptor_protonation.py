#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""Targeted regression checks for pocket-focused receptor protonation helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import openmm
from openmm.app import Topology, element

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPTS_ROOT))

import receptor_protonation as rp  # noqa: E402


def _build_mock_topology():
    topology = Topology()
    chain = topology.addChain("A")
    positions = []

    def add_atom(residue, atom_name, atom_element, xyz_angstrom):
        topology.addAtom(atom_name, atom_element, residue)
        positions.append(openmm.Vec3(
            xyz_angstrom[0] / 10.0,
            xyz_angstrom[1] / 10.0,
            xyz_angstrom[2] / 10.0,
        ))

    asp = topology.addResidue("ASP", chain, id="10")
    add_atom(asp, "CA", element.carbon, (0.0, 0.0, 0.0))
    add_atom(asp, "OD1", element.oxygen, (1.2, 0.0, 0.0))

    glu = topology.addResidue("GLU", chain, id="11")
    add_atom(glu, "CA", element.carbon, (4.0, 0.0, 0.0))
    add_atom(glu, "OE1", element.oxygen, (5.2, 0.0, 0.0))

    lys = topology.addResidue("LYS", chain, id="12")
    add_atom(lys, "CA", element.carbon, (8.0, 0.0, 0.0))
    add_atom(lys, "NZ", element.nitrogen, (9.3, 0.0, 0.0))

    his = topology.addResidue("HIS", chain, id="13")
    add_atom(his, "ND1", element.nitrogen, (12.0, 0.0, 0.0))
    add_atom(his, "NE2", element.nitrogen, (13.0, 0.0, 0.0))

    asn = topology.addResidue("ASN", chain, id="14")
    add_atom(asn, "OD1", element.oxygen, (13.2, 0.0, 0.0))

    tyr = topology.addResidue("TYR", chain, id="15")
    add_atom(tyr, "OH", element.oxygen, (16.0, 0.0, 0.0))

    cys_disulfide_a = topology.addResidue("CYS", chain, id="16")
    add_atom(cys_disulfide_a, "SG", element.sulfur, (20.0, 0.0, 0.0))

    cys_disulfide_b = topology.addResidue("CYS", chain, id="17")
    add_atom(cys_disulfide_b, "SG", element.sulfur, (22.1, 0.0, 0.0))

    cys_isolated = topology.addResidue("CYS", chain, id="18")
    add_atom(cys_isolated, "SG", element.sulfur, (30.0, 0.0, 0.0))

    return topology, positions, {
        "asp": asp,
        "glu": glu,
        "lys": lys,
        "his": his,
        "tyr": tyr,
        "cys_disulfide_a": cys_disulfide_a,
        "cys_disulfide_b": cys_disulfide_b,
        "cys_isolated": cys_isolated,
    }


def _assert_equal(actual, expected, label):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def _test_histidine_geometry_choice():
    topology, positions, residues = _build_mock_topology()
    variant = rp.choose_neutral_histidine_variant(residues["his"], topology, positions)
    _assert_equal(variant, "HIE", "neutral histidine tautomer")


def _test_variant_plan_applies_pocket_overrides():
    topology, positions, residues = _build_mock_topology()
    shifted_residues = [
        {
            "residue_key": rp.topology_residue_key(residues["asp"]),
            "chain_id": "A",
            "residue_number": "10",
            "residue_name": "ASP",
            "pka": 8.0,
            "default_state": "deprotonated",
            "propka_state": "protonated",
        },
        {
            "residue_key": rp.topology_residue_key(residues["glu"]),
            "chain_id": "A",
            "residue_number": "11",
            "residue_name": "GLU",
            "pka": 8.0,
            "default_state": "deprotonated",
            "propka_state": "protonated",
        },
        {
            "residue_key": rp.topology_residue_key(residues["lys"]),
            "chain_id": "A",
            "residue_number": "12",
            "residue_name": "LYS",
            "pka": 7.0,
            "default_state": "protonated",
            "propka_state": "deprotonated",
        },
        {
            "residue_key": rp.topology_residue_key(residues["his"]),
            "chain_id": "A",
            "residue_number": "13",
            "residue_name": "HIS",
            "pka": 8.0,
            "default_state": "neutral",
            "propka_state": "protonated",
        },
    ]
    pocket_keys = {entry["residue_key"] for entry in shifted_residues}
    plan = rp.build_variant_plan(
        topology,
        positions,
        protonation_ph=7.4,
        pocket_residue_keys=pocket_keys,
        shifted_residues=shifted_residues,
    )

    _assert_equal(plan["resolved_variants"][rp.topology_residue_key(residues["asp"])], "ASH", "ASP override")
    _assert_equal(plan["resolved_variants"][rp.topology_residue_key(residues["glu"])], "GLH", "GLU override")
    _assert_equal(plan["resolved_variants"][rp.topology_residue_key(residues["lys"])], "LYN", "LYS override")
    _assert_equal(plan["resolved_variants"][rp.topology_residue_key(residues["his"])], "HIP", "HIS override")
    _assert_equal(len(plan["applied_overrides"]), 4, "override count")


def _test_variant_plan_ignores_outside_pocket_and_unsupported():
    topology, positions, residues = _build_mock_topology()
    shifted_residues = [
        {
            "residue_key": rp.topology_residue_key(residues["glu"]),
            "chain_id": "A",
            "residue_number": "11",
            "residue_name": "GLU",
            "pka": 8.0,
            "default_state": "deprotonated",
            "propka_state": "protonated",
        },
        {
            "residue_key": rp.topology_residue_key(residues["tyr"]),
            "chain_id": "A",
            "residue_number": "15",
            "residue_name": "TYR",
            "pka": 7.0,
            "default_state": "protonated",
            "propka_state": "deprotonated",
        },
        {
            "residue_key": rp.topology_residue_key(residues["cys_isolated"]),
            "chain_id": "A",
            "residue_number": "18",
            "residue_name": "CYS",
            "pka": 7.0,
            "default_state": "protonated",
            "propka_state": "deprotonated",
        },
    ]
    plan = rp.build_variant_plan(
        topology,
        positions,
        protonation_ph=7.4,
        pocket_residue_keys={
            rp.topology_residue_key(residues["tyr"]),
            rp.topology_residue_key(residues["cys_isolated"]),
        },
        shifted_residues=shifted_residues,
    )

    ignored_reasons = {
        entry["residue_key"]: entry["reason"]
        for entry in plan["ignored_shifted_residues"]
    }
    _assert_equal(ignored_reasons[rp.topology_residue_key(residues["glu"])], "outside_pocket", "outside pocket reason")
    _assert_equal(
        ignored_reasons[rp.topology_residue_key(residues["tyr"])],
        "tyr_deprotonation_not_supported",
        "unsupported TYR reason",
    )
    _assert_equal(
        ignored_reasons[rp.topology_residue_key(residues["cys_isolated"])],
        "cys_thiolate_requires_disulfide",
        "unsupported CYS reason",
    )


def _test_disulfide_defaults():
    topology, positions, residues = _build_mock_topology()
    plan = rp.build_variant_plan(topology, positions, protonation_ph=7.4)
    _assert_equal(
        plan["resolved_variants"][rp.topology_residue_key(residues["cys_disulfide_a"])],
        "CYX",
        "first disulfide cysteine",
    )
    _assert_equal(
        plan["resolved_variants"][rp.topology_residue_key(residues["cys_disulfide_b"])],
        "CYX",
        "second disulfide cysteine",
    )
    _assert_equal(
        plan["resolved_variants"][rp.topology_residue_key(residues["cys_isolated"])],
        "CYS",
        "isolated cysteine default",
    )


def _test_identify_pocket_residue_keys_from_pdb():
    pdb_text = """\
ATOM      1  CA  ASP A  10       0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CA  GLU A  11      20.000   0.000   0.000  1.00 20.00           C
END
"""
    with tempfile.TemporaryDirectory(prefix="pocket_keys_") as tmpdir:
        pdb_path = Path(tmpdir) / "mini.pdb"
        pdb_path.write_text(pdb_text)
        pocket_keys = rp.identify_pocket_residue_keys_from_pdb(
            str(pdb_path),
            ligand_coords=[(1.0, 0.0, 0.0)],
            cutoff_a=6.0,
        )
    _assert_equal(pocket_keys, {"A:10:"}, "pocket residue detection")


def main() -> int:
    _test_histidine_geometry_choice()
    print("PASS neutral histidine tautomer selection")

    _test_variant_plan_applies_pocket_overrides()
    print("PASS pocket override mapping")

    _test_variant_plan_ignores_outside_pocket_and_unsupported()
    print("PASS ignored shifted residue handling")

    _test_disulfide_defaults()
    print("PASS disulfide default mapping")

    _test_identify_pocket_residue_keys_from_pdb()
    print("PASS pocket residue detection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
