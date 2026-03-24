#!/usr/bin/env python3
"""
Compare Ember's QupKake runtimes against an upstream-head source snapshot.

The report is intended to answer one question quickly: is the divergence caused
by the source patches, the Python/xTB runtime, or the packaged bundle layout?
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = REPO_ROOT / "deps" / "staging" / "scripts" / "predict_ligand_pka.py"
VENDOR_QUPKAKE = REPO_ROOT / "vendor" / "QupKake"
DEFAULT_USER_LIGAND = (
    Path.home()
    / "Ember"
    / "golden-quartz-spider"
    / "docking"
    / "Vina_VU9"
    / "inputs"
    / "ligands"
    / "Indole(N-H)_7,31-Piperidine_3-Oxy_4-Aryl.sdf"
)

CONTROL_SMILES = [
    ("acetic_acid", "CC(=O)O"),
    ("pyridine", "c1ccncc1"),
    ("piperidine", "C1CCNCC1"),
    ("indole", "c1ccc2[nH]ccc2c1"),
]


def load_wrapper_module() -> Any:
    spec = importlib.util.spec_from_file_location("ember_predict_ligand_pka", WRAPPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load QupKake wrapper from {WRAPPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WRAPPER = load_wrapper_module()


def existing_path(*paths: Path) -> str | None:
    for path in paths:
        if path.exists():
            return str(path.resolve())
    return None


def create_clean_qupkake_head_snapshot(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "-C", str(VENDOR_QUPKAKE), "archive", "--format=tar", "HEAD"],
        capture_output=True,
        check=True,
    )
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        try:
            tar.extractall(destination, filter="data")
        except TypeError:
            tar.extractall(destination)
    return destination


def write_smiles_sdf(name: str, smiles: str, output_path: Path) -> None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"Failed to build control molecule {name} from SMILES {smiles}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE) != 0:
        raise RuntimeError(f"Failed to embed control molecule {name}")
    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    mol.SetProp("_Name", name)
    writer = Chem.SDWriter(str(output_path))
    writer.write(mol)
    writer.close()


def build_control_panel(root: Path, user_ligand: str | None) -> list[dict[str, str]]:
    root.mkdir(parents=True, exist_ok=True)
    controls: list[dict[str, str]] = []
    for name, smiles in CONTROL_SMILES:
        path = root / f"{name}.sdf"
        write_smiles_sdf(name, smiles, path)
        controls.append({"name": name, "path": str(path), "kind": "smiles_control"})

    if user_ligand:
        ligand_path = Path(user_ligand).expanduser()
        if ligand_path.exists():
            controls.append(
                {
                    "name": "user_ligand",
                    "path": str(ligand_path.resolve()),
                    "kind": "user_ligand",
                }
            )
    return controls


def env_overrides(spec: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    env["QUPKAKE_PYTHON"] = spec["pythonPath"]
    env["QUPKAKE_XTBPATH"] = spec["xtbPath"]
    if spec.get("qupkakeRoot"):
        env["QUPKAKE_ROOT"] = spec["qupkakeRoot"]
    return env


def run_wrapper_check(spec: dict[str, Any]) -> dict[str, Any]:
    env = env_overrides(spec)
    proc = subprocess.run(
        [sys.executable, str(WRAPPER_PATH), "--check"],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    parsed: dict[str, Any] | None = None
    parse_error: str | None = None
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if stdout:
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError as exc:
            parse_error = str(exc)

    return {
        "returnCode": proc.returncode,
        "rawStdout": stdout,
        "rawStderr": stderr,
        "parsed": parsed,
        "parseError": parse_error,
    }


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for case in cases:
        summary[case["controlName"]] = {
            "entryCount": case["entryCount"],
            "failureStage": case.get("failureStage"),
            "types": [entry.get("type") for entry in case.get("entries", [])],
            "atomIndices": [entry.get("atomIndices") for entry in case.get("entries", [])],
        }
    return summary


def build_runtime_specs(clean_root: str) -> list[dict[str, Any]]:
    bundle_python = existing_path(
        REPO_ROOT / "bundle-mac" / "extra-resources" / "qupkake-python" / "bin" / "python3.9",
        REPO_ROOT / "bundle-mac" / "extra-resources" / "qupkake-python" / "bin" / "python",
    )
    upstream_python = existing_path(
        Path.home() / "miniconda3" / "envs" / "qupkake" / "bin" / "python",
        Path.home() / "anaconda3" / "envs" / "qupkake" / "bin" / "python",
        Path.home() / "miniforge3" / "envs" / "qupkake" / "bin" / "python",
        Path.home() / "mambaforge" / "envs" / "qupkake" / "bin" / "python",
    )
    xtb_641 = existing_path(
        REPO_ROOT / "vendor" / "xtb-6.4.1" / "install-openblas" / "bin" / "xtb",
        REPO_ROOT / "vendor" / "xtb-6.4.1" / "install" / "bin" / "xtb",
    )
    xtb_671 = existing_path(
        REPO_ROOT / "vendor" / "xtb-env" / "bin" / "xtb",
        REPO_ROOT / "bundle-mac" / "extra-resources" / "qupkake-xtb" / "bin" / "xtb",
        REPO_ROOT / "bundle-mac" / "extra-resources" / "qupkake-python" / "bin" / "xtb",
    )
    bundle_root = existing_path(REPO_ROOT / "bundle-mac" / "extra-resources" / "qupkake-fork")
    local_root = str(VENDOR_QUPKAKE.resolve())

    specs: list[dict[str, Any]] = []

    def add_spec(
        spec_id: str,
        label: str,
        python_path: str | None,
        xtb_path: str | None,
        qupkake_root: str | None,
        source_variant: str,
        runtime_variant: str,
    ) -> None:
        if not python_path or not xtb_path or not qupkake_root:
            return
        specs.append(
            {
                "id": spec_id,
                "label": label,
                "pythonPath": python_path,
                "xtbPath": xtb_path,
                "qupkakeRoot": qupkake_root,
                "sourceVariant": source_variant,
                "runtimeVariant": runtime_variant,
            }
        )

    add_spec(
        "upstream_like_head_xtb641",
        "Upstream head source + upstream qupkake env + xTB 6.4.1",
        upstream_python,
        xtb_641,
        clean_root,
        "upstream_head",
        "upstream_like",
    )
    add_spec(
        "ember_upstream_head_xtb671",
        "Upstream head source + Ember mac runtime + xTB 6.7.1",
        bundle_python,
        xtb_671,
        clean_root,
        "upstream_head",
        "ember_mac",
    )
    add_spec(
        "ember_local_patched_xtb671",
        "Local patched source + Ember mac runtime + xTB 6.7.1",
        bundle_python,
        xtb_671,
        local_root,
        "local_patched",
        "ember_mac",
    )
    add_spec(
        "ember_bundle_snapshot_xtb671",
        "Bundled source snapshot + Ember mac runtime + xTB 6.7.1",
        bundle_python,
        xtb_671,
        bundle_root,
        "bundle_snapshot",
        "ember_mac",
    )
    add_spec(
        "ember_upstream_head_xtb641",
        "Upstream head source + Ember mac runtime + xTB 6.4.1",
        bundle_python,
        xtb_641,
        clean_root,
        "upstream_head",
        "ember_mac",
    )
    add_spec(
        "ember_local_patched_xtb641",
        "Local patched source + Ember mac runtime + xTB 6.4.1",
        bundle_python,
        xtb_641,
        local_root,
        "local_patched",
        "ember_mac",
    )
    return specs


def analyze_runtime(spec: dict[str, Any], controls: list[dict[str, str]]) -> dict[str, Any]:
    runtime = {
        "pythonPath": spec["pythonPath"],
        "xtbPath": spec["xtbPath"],
        "qupkakeRoot": spec["qupkakeRoot"],
    }
    fingerprint = WRAPPER.build_runtime_fingerprint(runtime)
    capability = run_wrapper_check(spec)
    cases: list[dict[str, Any]] = []

    for control in controls:
        result = WRAPPER.run_qupkake_cli_with_runtime(control["path"], runtime)
        result["controlName"] = control["name"]
        result["controlKind"] = control["kind"]
        cases.append(result)

    return {
        "id": spec["id"],
        "label": spec["label"],
        "sourceVariant": spec["sourceVariant"],
        "runtimeVariant": spec["runtimeVariant"],
        "runtimeFingerprint": fingerprint,
        "capabilityCheck": capability,
        "cases": cases,
        "summary": summarize_cases(cases),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare QupKake runtimes and source variants")
    parser.add_argument("--ligand", default=str(DEFAULT_USER_LIGAND), help="Optional ligand SDF to include in the control panel")
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument(
        "--runtime-id",
        action="append",
        default=[],
        help="Limit execution to one or more runtime ids from the matrix output",
    )
    parser.add_argument(
        "--control",
        action="append",
        default=[],
        help="Limit execution to one or more control names (acetic_acid, pyridine, piperidine, indole, user_ligand)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with tempfile.TemporaryDirectory(prefix="qupkake_matrix_") as temp_root:
        temp_path = Path(temp_root)
        clean_root = create_clean_qupkake_head_snapshot(temp_path / "qupkake_head")
        controls = build_control_panel(temp_path / "controls", args.ligand)
        if args.control:
            requested_controls = set(args.control)
            controls = [control for control in controls if control["name"] in requested_controls]
        specs = build_runtime_specs(str(clean_root))
        if args.runtime_id:
            requested_runtime_ids = set(args.runtime_id)
            specs = [spec for spec in specs if spec["id"] in requested_runtime_ids]

        report = {
            "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
            "wrapperPath": str(WRAPPER_PATH),
            "vendorQupkake": str(VENDOR_QUPKAKE),
            "controls": controls,
            "runtimes": [analyze_runtime(spec, controls) for spec in specs],
        }

    serialized = json.dumps(report, indent=2 if args.pretty or args.output else None)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
