#!/usr/bin/env python3
"""
Thin Ember wrapper around a forked or external QupKake installation.

The wrapper prefers a repo-local or bundled QupKake fork and falls back to an
installed package in the selected Python runtime. It supports a cheap
availability check and a single-molecule prediction path that normalizes
QupKake's SDF output into stable JSON.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem


ENV_NAMES = ("qupkake",)
CONDA_DIRS = ("miniconda3", "anaconda3", "miniforge3", "mambaforge")
VALIDATION_SMILES = "CC(=O)O"


def load_molecule(path: str) -> Any:
    lower = path.lower()
    if lower.endswith(".sdf.gz"):
        with gzip.open(path, "rb") as handle:
            supplier = Chem.ForwardSDMolSupplier(handle, removeHs=False)
            return next(supplier, None)
    if lower.endswith(".sdf"):
        supplier = Chem.SDMolSupplier(path, removeHs=False)
        return supplier[0] if len(supplier) > 0 else None
    if lower.endswith(".mol"):
        return Chem.MolFromMolFile(path, removeHs=False)
    if lower.endswith(".mol2"):
        return Chem.MolFromMol2File(path, removeHs=False)
    return None


def ensure_3d_molecule(mol: Any) -> Any:
    if mol is None:
        return None
    mol = Chem.Mol(mol)
    if mol.GetNumConformers() > 0 and mol.GetConformer().Is3D():
        return mol

    mol = Chem.AddHs(mol, addCoords=True)
    if AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE) != 0:
        raise RuntimeError("Failed to generate 3D coordinates for the ligand.")
    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    return mol


def normalize_input_for_qupkake(input_path: str, root: str) -> tuple[str, Any]:
    lower = input_path.lower()
    if lower.endswith(".sdf"):
        mol = load_molecule(input_path)
        if mol is None:
            raise RuntimeError(f"Failed to load ligand from {input_path}")
        return input_path, mol

    mol = load_molecule(input_path)
    if mol is None:
        raise RuntimeError(
            "QupKake requires a readable molecule input. Supported viewer inputs are .sdf, .sdf.gz, .mol, and .mol2."
        )

    mol = ensure_3d_molecule(mol)
    if not mol.HasProp("_Name") or not mol.GetProp("_Name").strip():
        mol.SetProp("_Name", Path(input_path).stem)

    normalized_path = os.path.join(root, "normalized_input.sdf")
    writer = Chem.SDWriter(normalized_path)
    writer.write(mol)
    writer.close()
    return normalized_path, mol


def candidate_python_paths() -> list[str]:
    candidates: list[str] = []
    env_python = os.environ.get("QUPKAKE_PYTHON")
    if env_python and os.path.exists(env_python):
        candidates.append(env_python)

    script_path = Path(__file__).resolve()
    resources_root = script_path.parent.parent
    bundled_python = resources_root / "qupkake-python" / "bin" / "python"
    if bundled_python.exists():
        candidates.append(str(bundled_python))

    repo_root = script_path.parents[3] if len(script_path.parents) >= 4 else None
    if repo_root:
        dev_bundled_python = repo_root / "bundle-mac" / "extra-resources" / "qupkake-python" / "bin" / "python"
        if dev_bundled_python.exists():
            candidates.append(str(dev_bundled_python))

    home = Path.home()
    for env_name in ENV_NAMES:
        for conda_dir in CONDA_DIRS:
            candidate = home / conda_dir / "envs" / env_name / "bin" / "python"
            if candidate.exists():
                candidates.append(str(candidate))

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(Path(candidate).resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def candidate_qupkake_roots() -> list[str]:
    candidates: list[str] = []
    env_root = os.environ.get("QUPKAKE_ROOT")
    if env_root and os.path.exists(env_root):
        candidates.append(env_root)

    script_path = Path(__file__).resolve()
    resources_root = script_path.parent.parent
    bundled_root = resources_root / "qupkake-fork"
    if bundled_root.exists():
        candidates.append(str(bundled_root))

    repo_root = script_path.parents[3] if len(script_path.parents) >= 4 else None
    if repo_root:
        dev_bundled_root = repo_root / "bundle-mac" / "extra-resources" / "qupkake-fork"
        if dev_bundled_root.exists():
            candidates.append(str(dev_bundled_root))
        repo_vendor_root = repo_root / "vendor" / "QupKake"
        if repo_vendor_root.exists():
            candidates.append(str(repo_vendor_root))

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(Path(candidate).resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def candidate_xtb_paths(python_path: str) -> list[str]:
    candidates: list[str] = []

    for env_var in ("QUPKAKE_XTBPATH", "XTBPATH"):
        env_xtb = os.environ.get(env_var)
        if env_xtb and os.path.exists(env_xtb):
            candidates.append(env_xtb)

    script_path = Path(__file__).resolve()
    resources_root = script_path.parent.parent
    bundled_xtb = resources_root / "qupkake-xtb" / "bin" / "xtb"
    if bundled_xtb.exists():
        candidates.append(str(bundled_xtb))

    env_bin = Path(python_path).resolve().parent
    env_xtb = env_bin / "xtb"
    if env_xtb.exists():
        candidates.append(str(env_xtb))

    repo_root = script_path.parents[3] if len(script_path.parents) >= 4 else None
    if repo_root:
        dev_bundled_xtb = repo_root / "bundle-mac" / "extra-resources" / "qupkake-xtb" / "bin" / "xtb"
        if dev_bundled_xtb.exists():
            candidates.append(str(dev_bundled_xtb))
        for repo_xtb in (
            repo_root / "vendor" / "xtb-env" / "bin" / "xtb",
            repo_root / "vendor" / "xtb-6.4.1" / "install" / "bin" / "xtb",
            repo_root / "vendor" / "xtb-6.4.1" / "install-openblas" / "bin" / "xtb",
        ):
            if repo_xtb.exists():
                candidates.append(str(repo_xtb))

    which_xtb = shutil.which("xtb", path=f"{env_bin}:{os.environ.get('PATH', '')}")
    if which_xtb:
        candidates.append(which_xtb)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(Path(candidate).resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def build_external_env(python_path: str, xtb_path: str, qupkake_root: str | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env_bin = str(Path(python_path).resolve().parent)
    xtb_bin = str(Path(xtb_path).resolve().parent)
    xtb_root = str(Path(xtb_bin).parent)
    env["PATH"] = f"{xtb_bin}:{env_bin}:{env.get('PATH', '')}"
    env["XTBPATH"] = xtb_path
    env["QUPKAKE_XTBPATH"] = xtb_path
    xtb_lib = Path(xtb_root) / "lib"
    if xtb_lib.exists():
        env["DYLD_LIBRARY_PATH"] = f"{xtb_lib}:{env.get('DYLD_LIBRARY_PATH', '')}"
        env["LD_LIBRARY_PATH"] = f"{xtb_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    if qupkake_root:
        env["QUPKAKE_ROOT"] = qupkake_root
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{qupkake_root}:{existing_pythonpath}" if existing_pythonpath else qupkake_root
    return env


def run_external(
    python_path: str,
    args: list[str],
    env: dict[str, str],
    cwd: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [python_path, *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        timeout=180,
    )


def resolve_runtime() -> dict[str, Any]:
    candidates = candidate_python_paths()
    if not candidates:
        return {
            "available": False,
            "validated": False,
            "message": "Dedicated QupKake Python not found. Build or bundle the qupkake runtime first.",
        }

    qupkake_roots = candidate_qupkake_roots()
    last_error = "QupKake is unavailable."
    missing_xtb = False
    for python_path in candidates:
        xtb_paths = candidate_xtb_paths(python_path)
        if not xtb_paths:
            missing_xtb = True
            last_error = "Dedicated xTB executable not found. Bundle or build qupkake-xtb and point QUPKAKE_XTBPATH at it."
            continue

        for xtb_path in xtb_paths:
            for qupkake_root in [*qupkake_roots, None]:
                env = build_external_env(python_path, xtb_path, qupkake_root)

                import_proc = run_external(python_path, ["-c", "import qupkake; print('OK')"], env)
                if import_proc.returncode != 0:
                    last_error = import_proc.stderr.strip() or import_proc.stdout.strip() or "Failed to import qupkake."
                    continue

                cli_proc = run_external(python_path, ["-m", "qupkake.cli", "--version"], env)
                if cli_proc.returncode != 0:
                    last_error = cli_proc.stderr.strip() or cli_proc.stdout.strip() or "QupKake CLI is not runnable."
                    continue

                return {
                    "available": True,
                    "validated": False,
                    "pythonPath": python_path,
                    "xtbPath": xtb_path,
                    "qupkakeRoot": qupkake_root,
                }

    if missing_xtb and "xTB executable" in last_error:
        return {"available": False, "validated": False, "message": last_error}
    return {"available": False, "validated": False, "message": last_error}


def create_validation_ligand(path: str) -> None:
    mol = Chem.MolFromSmiles(VALIDATION_SMILES)
    if mol is None:
        raise RuntimeError("Failed to build the QupKake validation molecule.")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE) != 0:
        raise RuntimeError("Failed to embed the QupKake validation molecule.")
    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    mol.SetProp("_Name", "acetic_acid_validation")
    writer = Chem.SDWriter(path)
    writer.write(mol)
    writer.close()


def find_validation_ligand() -> str | None:
    explicit = os.environ.get("QUPKAKE_VALIDATE_LIGAND")
    if explicit and os.path.exists(explicit):
        return explicit

    ember_root = Path.home() / "Ember"
    if not ember_root.exists():
        return None

    for project_dir in sorted(p for p in ember_root.iterdir() if p.is_dir()):
        docking_root = project_dir / "docking"
        if not docking_root.exists():
            continue
        for docking_dir in sorted(p for p in docking_root.iterdir() if p.is_dir()):
            prep_dir = docking_dir / "prep"
            if not prep_dir.exists():
                continue
            for candidate in sorted(prep_dir.iterdir()):
                if candidate.suffix.lower() in {".sdf", ".mol", ".mol2"}:
                    return str(candidate)
                if candidate.name.lower().endswith(".sdf.gz"):
                    return str(candidate)
    return None


def normalize_smiles(mol: Any) -> str | None:
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)))
    except Exception:
        try:
            return Chem.MolToSmiles(Chem.Mol(mol))
        except Exception:
            return None


def molecule_name(mol: Any, fallback_path: str) -> str:
    if mol is not None and mol.HasProp("_Name"):
        name = mol.GetProp("_Name").strip()
        if name:
            return name
    filename = os.path.basename(fallback_path)
    if filename.endswith(".sdf.gz"):
        filename = filename[:-7]
    else:
        filename = os.path.splitext(filename)[0]
    return filename or "ligand"


def parse_predictions(output_path: str) -> list[dict[str, Any]]:
    supplier = Chem.SDMolSupplier(output_path, removeHs=False)
    entries: list[dict[str, Any]] = []

    for index, mol in enumerate(supplier):
        if mol is None:
            continue
        if not mol.HasProp("pka"):
            continue

        entry: dict[str, Any] = {
            "label": f"Site {index + 1}",
            "pka": float(mol.GetProp("pka")),
        }

        if mol.HasProp("pka_type"):
            pka_type = mol.GetProp("pka_type").strip().lower()
            if pka_type in {"acidic", "basic"}:
                entry["type"] = pka_type

        if mol.HasProp("idx"):
            try:
                entry["atomIndices"] = [int(mol.GetProp("idx"))]
            except ValueError:
                pass

        entries.append(entry)

    return entries


def run_prediction_with_runtime(ligand_path: str, runtime: dict[str, Any]) -> dict[str, Any]:
    python_path = runtime["pythonPath"]
    xtb_path = runtime["xtbPath"]
    env = build_external_env(python_path, xtb_path, runtime.get("qupkakeRoot"))

    with tempfile.TemporaryDirectory(prefix="ember_qupkake_") as root:
        normalized_input_path, input_mol = normalize_input_for_qupkake(ligand_path, root)
        name = molecule_name(input_mol, ligand_path)
        smiles = normalize_smiles(input_mol)
        output_name = "ember_qupkake_output.sdf"
        started = time.perf_counter()
        proc = run_external(
            python_path,
            ["-m", "qupkake.cli", "file", normalized_input_path, "-r", root, "-o", output_name],
            env,
        )
        runtime_ms = round((time.perf_counter() - started) * 1000, 1)

        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "QupKake prediction failed")

        output_path = os.path.join(root, "output", output_name)
        if not os.path.exists(output_path):
            if (
                "No protonation/deprotonation sites were found." in proc.stdout
                or "No valid QupKake features were generated" in proc.stdout
                or "Output file will not be created." in proc.stdout
            ):
                return {
                    "name": name,
                    "smiles": smiles,
                    "method": "qupkake",
                    "methodLabel": "QupKake",
                    "runtimeMs": runtime_ms,
                    "entries": [],
                }
            raise RuntimeError(proc.stdout.strip() or "QupKake did not produce an output SDF")

        return {
            "name": name,
            "smiles": smiles,
            "method": "qupkake",
            "methodLabel": "QupKake",
            "runtimeMs": runtime_ms,
            "entries": parse_predictions(output_path),
        }


def check_installation() -> dict[str, Any]:
    runtime = resolve_runtime()
    if not runtime.get("available"):
        return runtime

    validation_ligand = find_validation_ligand()
    runtime["validationLigand"] = validation_ligand

    with tempfile.TemporaryDirectory(prefix="ember_qupkake_check_") as root:
        validation_input = os.path.join(root, "validation_input.sdf")
        create_validation_ligand(validation_input)

        try:
            validation_result = run_prediction_with_runtime(validation_input, runtime)
            if validation_result["entries"]:
                runtime["validated"] = True
            else:
                runtime["warning"] = (
                    "QupKake launched, but the bundled validation molecule returned no micro-pKa entries on this machine."
                )
        except Exception as exc:
            runtime["warning"] = f"QupKake launched, but validation failed: {exc}"

    if validation_ligand and runtime.get("available") and runtime.get("validated"):
        try:
            real_result = run_prediction_with_runtime(validation_ligand, runtime)
            if not real_result["entries"] and not runtime.get("warning"):
                runtime["warning"] = (
                    f"Validation ligand {os.path.basename(validation_ligand)} produced no micro-pKa entries."
                )
        except Exception as exc:
            if not runtime.get("warning"):
                runtime["warning"] = (
                    f"Validation ligand {os.path.basename(validation_ligand)} failed: {exc}"
                )

    return runtime


def predict(ligand_path: str) -> dict[str, Any]:
    status = check_installation()
    if not status.get("available"):
        raise RuntimeError(status.get("message") or "QupKake is unavailable")
    return run_prediction_with_runtime(ligand_path, status)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict ligand micro-pKa with external QupKake")
    parser.add_argument("--check", action="store_true", help="Check whether external QupKake is available")
    parser.add_argument("--ligand", help="Input ligand file for a single prediction")
    args = parser.parse_args()

    try:
        if args.check:
            print(json.dumps(check_installation()))
            return
        if not args.ligand:
            raise RuntimeError("--ligand is required unless --check is used")
        print(json.dumps(predict(args.ligand)))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
