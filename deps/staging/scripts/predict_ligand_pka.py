#!/usr/bin/env python3
"""
Thin Ember wrapper around the authoritative QupKake runtime.

The wrapper resolves a single QupKake source tree, Python runtime, and xTB
binary. It supports a cheap
availability check and a single-molecule prediction path that normalizes
QupKake's SDF output into stable JSON.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem


VALIDATION_SMILES = "CC(=O)O"
RUNTIME_PROBE_CODE = """
import json
import os
import re
import subprocess
import sys

payload = {
    "pythonVersion": sys.version.split()[0],
    "moduleVersions": {},
    "moduleErrors": {},
}

for label, module_name in [
    ("rdkit", "rdkit"),
    ("torch", "torch"),
    ("pytorch_lightning", "pytorch_lightning"),
    ("torch_geometric", "torch_geometric"),
    ("qupkake", "qupkake"),
    ("qupkake_cli", "qupkake.cli"),
]:
    try:
        module = __import__(module_name, fromlist=["*"])
        payload["moduleVersions"][label] = getattr(module, "__version__", None)
    except Exception as exc:
        payload["moduleErrors"][label] = f"{type(exc).__name__}: {exc}"

xtb_version = None
xtb_path = os.environ.get("QUPKAKE_XTBPATH") or os.environ.get("XTBPATH")
if xtb_path:
    try:
        xtb_proc = subprocess.run([xtb_path, "--version"], capture_output=True, text=True, timeout=30)
        xtb_output = "\\n".join(
            chunk for chunk in [xtb_proc.stdout.strip(), xtb_proc.stderr.strip()] if chunk
        )
        match = re.search(r"xtb version\\s+([^\\s)]+)", xtb_output, flags=re.IGNORECASE)
        if match:
            xtb_version = match.group(1)
    except Exception as exc:
        payload["moduleErrors"]["xtb_probe"] = f"{type(exc).__name__}: {exc}"

payload["xtbVersion"] = xtb_version
try:
    from qupkake.xtbp import resolve_fukui_compatibility_mode, resolve_fukui_compatibility_source

    payload["fukuiCompatibilityMode"] = resolve_fukui_compatibility_mode(xtb_version)
    payload["fukuiCompatibilitySource"] = resolve_fukui_compatibility_source(
        xtb_version, payload["fukuiCompatibilityMode"]
    )
except Exception as exc:
    payload["moduleErrors"]["qupkake_xtbp"] = f"{type(exc).__name__}: {exc}"

payload["ok"] = len(payload["moduleErrors"]) == 0
print(json.dumps(payload))
""".strip()
VALIDATION_PANEL_CODE = """
import json
from qupkake.contracts import to_json_data
from qupkake.validation import run_validation_panel

print(json.dumps(to_json_data(run_validation_panel(mp=False))))
""".strip()


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


def resolve_qupkake_python_path() -> str | None:
    env_python = os.environ.get("QUPKAKE_PYTHON")
    if env_python and os.path.exists(env_python):
        return str(Path(env_python).resolve())

    script_path = Path(__file__).resolve()
    resources_root = script_path.parent.parent
    bundled_python = resources_root / "qupkake-python" / "bin" / "python"
    if bundled_python.exists():
        return str(bundled_python.resolve())

    canonical_local = Path.home() / "miniconda3" / "envs" / "qupkake" / "bin" / "python3.9"
    if canonical_local.exists():
        return str(canonical_local.resolve())

    return None


def resolve_qupkake_root() -> str | None:
    env_root = os.environ.get("QUPKAKE_ROOT")
    if env_root and os.path.exists(env_root):
        return str(Path(env_root).resolve())

    script_path = Path(__file__).resolve()
    resources_root = script_path.parent.parent
    bundled_root = resources_root / "qupkake-fork"
    if bundled_root.exists():
        return str(bundled_root.resolve())

    repo_root = script_path.parents[3] if len(script_path.parents) >= 4 else None
    if repo_root:
        repo_vendor_root = repo_root / "vendor" / "QupKake"
        if repo_vendor_root.exists():
            return str(repo_vendor_root.resolve())

    return None


def resolve_xtb_path() -> str | None:
    for env_var in ("QUPKAKE_XTBPATH", "XTBPATH"):
        env_xtb = os.environ.get(env_var)
        if env_xtb and os.path.exists(env_xtb):
            return str(Path(env_xtb).resolve())

    script_path = Path(__file__).resolve()
    resources_root = script_path.parent.parent
    bundled_xtb = resources_root / "qupkake-xtb" / "bin" / "xtb"
    if bundled_xtb.exists():
        return str(bundled_xtb.resolve())

    repo_root = script_path.parents[3] if len(script_path.parents) >= 4 else None
    if repo_root:
        repo_xtb = repo_root / "vendor" / "xtb-env" / "bin" / "xtb"
        if repo_xtb.exists():
            return str(repo_xtb.resolve())

    return None


def build_external_env(python_path: str, xtb_path: str, qupkake_root: str | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env_bin = str(Path(python_path).resolve().parent)
    xtb_bin = str(Path(xtb_path).resolve().parent)
    xtb_root = str(Path(xtb_bin).parent)
    env["PATH"] = f"{xtb_bin}:{env_bin}:{env.get('PATH', '')}"
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


def trimmed_output(text: str) -> str | None:
    stripped = text.strip()
    return stripped or None


def resolve_runtime() -> dict[str, Any]:
    python_path = resolve_qupkake_python_path()
    if not python_path:
        return {
            "available": False,
            "validated": False,
            "failureStage": "runtime_setup",
            "message": "Dedicated QupKake Python not found. Build or bundle the qupkake runtime first.",
        }

    xtb_path = resolve_xtb_path()
    if not xtb_path:
        return {
            "available": False,
            "validated": False,
            "failureStage": "runtime_setup",
            "message": "Dedicated xTB executable not found. Build or bundle qupkake-xtb and point QUPKAKE_XTBPATH at it.",
        }

    qupkake_root = resolve_qupkake_root()
    if not qupkake_root:
        return {
            "available": False,
            "validated": False,
            "failureStage": "runtime_setup",
            "message": "QupKake source root not found. Provide QUPKAKE_ROOT or restore the authoritative fork.",
        }

    runtime = {
        "available": True,
        "validated": False,
        "pythonPath": python_path,
        "xtbPath": xtb_path,
        "qupkakeRoot": qupkake_root,
    }
    runtime_probe = run_runtime_probe(runtime)
    if not runtime_probe.get("ok"):
        return {
            "available": False,
            "validated": False,
            "failureStage": "runtime_setup",
            "message": runtime_probe.get("message") or "QupKake runtime preflight failed.",
            "runtimeProbe": runtime_probe,
            "pythonPath": python_path,
            "xtbPath": xtb_path,
            "qupkakeRoot": qupkake_root,
        }

    runtime["runtimeProbe"] = runtime_probe
    return runtime


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


def probe_python_runtime(python_path: str, env: dict[str, str]) -> dict[str, Any]:
    proc = run_external(
        python_path,
        ["-c", RUNTIME_PROBE_CODE],
        env,
    )
    if proc.returncode != 0:
        return {"pythonVersion": None}
    try:
        payload = json.loads(proc.stdout.strip())
        module_versions = payload.get("moduleVersions", {}) if isinstance(payload, dict) else {}
        module_errors = payload.get("moduleErrors", {}) if isinstance(payload, dict) else {}
        return {
            "pythonVersion": payload.get("pythonVersion"),
            "rdkitVersion": module_versions.get("rdkit"),
            "qupkakeVersion": module_versions.get("qupkake"),
            "runtimeProbeOk": payload.get("ok"),
            "runtimeProbeModuleErrors": module_errors or None,
            "fukuiCompatibilityMode": payload.get("fukuiCompatibilityMode"),
            "fukuiCompatibilitySource": payload.get("fukuiCompatibilitySource"),
        }
    except json.JSONDecodeError:
        return {"pythonVersion": None}


def parse_xtb_version(output: str) -> str | None:
    match = re.search(r"xtb version\s+([^\s)]+)", output, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def probe_xtb_runtime(xtb_path: str, env: dict[str, str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [xtb_path, "--version"],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
    except Exception as exc:
        return {"xtbVersion": None, "xtbError": str(exc)}

    combined_output = "\n".join(
        chunk for chunk in [proc.stdout.strip(), proc.stderr.strip()] if chunk
    )
    return {
        "xtbVersion": parse_xtb_version(combined_output),
        "xtbRawVersion": trimmed_output(combined_output),
    }


def build_runtime_fingerprint(runtime: dict[str, Any]) -> dict[str, Any]:
    fingerprint = {
        "pythonPath": runtime.get("pythonPath"),
        "xtbPath": runtime.get("xtbPath"),
        "qupkakeRoot": runtime.get("qupkakeRoot"),
    }
    python_path = runtime.get("pythonPath")
    xtb_path = runtime.get("xtbPath")
    if not python_path or not xtb_path:
        return fingerprint

    env = build_external_env(python_path, xtb_path, runtime.get("qupkakeRoot"))
    fingerprint.update(probe_python_runtime(python_path, env))
    fingerprint.update(probe_xtb_runtime(xtb_path, env))
    return fingerprint


def process_signal_name(returncode: int | None) -> str | None:
    if returncode is None or returncode >= 0:
        return None
    return f"signal_{abs(returncode)}"


def runtime_setup_message(proc: subprocess.CompletedProcess[str], fallback: str) -> str:
    stderr = trimmed_output(proc.stderr or "")
    stdout = trimmed_output(proc.stdout or "")
    return stderr or stdout or fallback


def run_runtime_probe(runtime: dict[str, Any]) -> dict[str, Any]:
    python_path = runtime["pythonPath"]
    xtb_path = runtime["xtbPath"]
    env = build_external_env(python_path, xtb_path, runtime.get("qupkakeRoot"))
    payload, proc = run_json_probe(python_path, env, RUNTIME_PROBE_CODE)
    xtb_probe = probe_xtb_runtime(xtb_path, env)

    result: dict[str, Any] = {
        "ok": False,
        "returnCode": proc.returncode,
        "signal": process_signal_name(proc.returncode),
        "rawStdout": trimmed_output(proc.stdout or ""),
        "rawStderr": trimmed_output(proc.stderr or ""),
        "xtbVersion": xtb_probe.get("xtbVersion"),
        "xtbRawVersion": xtb_probe.get("xtbRawVersion"),
    }
    if payload is None:
        result["failureStage"] = "runtime_setup"
        result["message"] = runtime_setup_message(proc, "Failed to execute the QupKake runtime preflight.")
        return result

    result["pythonVersion"] = payload.get("pythonVersion")
    result["moduleVersions"] = payload.get("moduleVersions", {})
    result["moduleErrors"] = payload.get("moduleErrors", {})
    result["fukuiCompatibilityMode"] = payload.get("fukuiCompatibilityMode")
    result["fukuiCompatibilitySource"] = payload.get("fukuiCompatibilitySource")
    result["ok"] = bool(payload.get("ok"))

    if result["moduleErrors"]:
        result["failureStage"] = "runtime_setup"
        result["message"] = "; ".join(
            f"{name}: {message}" for name, message in result["moduleErrors"].items()
        )
        return result

    if not xtb_probe.get("xtbVersion"):
        result["failureStage"] = "runtime_setup"
        result["message"] = xtb_probe.get("xtbError") or "xTB did not report a version during runtime preflight."
        return result

    return result


def run_json_probe(
    python_path: str,
    env: dict[str, str],
    code: str,
) -> tuple[dict[str, Any] | None, subprocess.CompletedProcess[str]]:
    proc = run_external(python_path, ["-c", code], env)
    if proc.returncode != 0:
        return None, proc
    try:
        return json.loads(proc.stdout.strip()), proc
    except json.JSONDecodeError:
        return None, proc


def classify_failure_stage(proc: subprocess.CompletedProcess[str], output_exists: bool) -> str | None:
    stdout = proc.stdout or ""
    if proc.returncode != 0:
        return "prediction_failed"
    if output_exists:
        return None
    if "No valid QupKake features were generated" in stdout:
        return "feature_generation"
    if "No protonation/deprotonation sites were found." in stdout:
        return "no_sites"
    return "prediction_failed"


def run_validation_with_runtime(runtime: dict[str, Any]) -> tuple[dict[str, Any] | None, subprocess.CompletedProcess[str]]:
    python_path = runtime["pythonPath"]
    xtb_path = runtime["xtbPath"]
    env = build_external_env(python_path, xtb_path, runtime.get("qupkakeRoot"))
    return run_json_probe(python_path, env, VALIDATION_PANEL_CODE)


def warning_from_validation_report(report: dict[str, Any]) -> str:
    summary = report.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary
    overall_failure = report.get("overallFailure")
    if isinstance(overall_failure, dict):
        message = overall_failure.get("message")
        if isinstance(message, str) and message.strip():
            return message
    return "QupKake validation failed the required acid/base control panel."


def normalize_validation_report(
    report: dict[str, Any], runtime_fingerprint: dict[str, Any]
) -> dict[str, Any]:
    normalized_controls: list[dict[str, Any]] = []
    for control in report.get("controls", []) if isinstance(report.get("controls"), list) else []:
        if not isinstance(control, dict):
            continue
        failure = control.get("failure") if isinstance(control.get("failure"), dict) else {}
        normalized_controls.append(
            {
                "name": control.get("name"),
                "required": bool(control.get("required")),
                "smiles": control.get("smiles"),
                "expectedTypeCounts": control.get("expected_type_counts", {}),
                "forbiddenTypes": control.get("forbidden_types", []),
                "observedTypeCounts": control.get("observed_type_counts", {}),
                "entries": control.get("entries", []),
                "entryCount": control.get("entry_count", 0),
                "passed": bool(control.get("passed")),
                "reasons": control.get("reasons", []),
                "failureStage": failure.get("stage"),
            }
        )

    overall_failure = report.get("overall_failure") if isinstance(report.get("overall_failure"), dict) else {}
    return {
        "controls": normalized_controls,
        "requiredControlNames": report.get("required_control_names", []),
        "informationalControlNames": report.get("informational_control_names", []),
        "validated": bool(report.get("validated")),
        "summary": report.get("summary"),
        "overallFailureStage": overall_failure.get("stage"),
        "runtimeFingerprint": runtime_fingerprint,
    }


def legacy_validation_case_from_report(report: dict[str, Any]) -> dict[str, Any] | None:
    controls = report.get("controls")
    if not isinstance(controls, list) or not controls:
        return None
    preferred = None
    for control in controls:
        if isinstance(control, dict) and control.get("required") and not control.get("passed"):
            preferred = control
            break
    if preferred is None:
        for control in controls:
            if isinstance(control, dict) and control.get("required"):
                preferred = control
                break
    if preferred is None or not isinstance(preferred, dict):
        return None

    return {
        "name": preferred.get("name"),
        "smiles": preferred.get("smiles"),
        "outputCreated": preferred.get("entryCount", 0) > 0,
        "entryCount": preferred.get("entryCount", 0),
        "entries": preferred.get("entries", []),
        "failureStage": preferred.get("failureStage"),
        "rawStdout": None,
        "rawStderr": None,
    }


def run_qupkake_cli_with_runtime(ligand_path: str, runtime: dict[str, Any]) -> dict[str, Any]:
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
        output_path = os.path.join(root, "output", output_name)
        output_exists = os.path.exists(output_path)
        failure_stage = classify_failure_stage(proc, output_exists)
        entries = parse_predictions(output_path) if output_exists else []

        return {
            "name": name,
            "inputPath": ligand_path,
            "smiles": smiles,
            "method": "qupkake",
            "methodLabel": "QupKake",
            "runtimeMs": runtime_ms,
            "entries": entries,
            "entryCount": len(entries),
            "outputCreated": output_exists,
            "outputPath": output_path if output_exists else None,
            "returnCode": proc.returncode,
            "rawStdout": trimmed_output(proc.stdout),
            "rawStderr": trimmed_output(proc.stderr),
            "failureStage": failure_stage,
        }


def run_prediction_with_runtime(ligand_path: str, runtime: dict[str, Any]) -> dict[str, Any]:
    cli_result = run_qupkake_cli_with_runtime(ligand_path, runtime)

    if cli_result["returnCode"] != 0:
        raise RuntimeError(
            cli_result.get("rawStderr")
            or cli_result.get("rawStdout")
            or "QupKake prediction failed"
        )

    if not cli_result["outputCreated"]:
        if cli_result["failureStage"] in {"no_sites", "feature_generation"}:
            return {
                "name": cli_result["name"],
                "smiles": cli_result["smiles"],
                "method": "qupkake",
                "methodLabel": "QupKake",
                "runtimeMs": cli_result["runtimeMs"],
                "entries": [],
            }
        raise RuntimeError(cli_result.get("rawStdout") or "QupKake did not produce an output SDF")

    return {
        "name": cli_result["name"],
        "smiles": cli_result["smiles"],
        "method": "qupkake",
        "methodLabel": "QupKake",
        "runtimeMs": cli_result["runtimeMs"],
        "entries": cli_result["entries"],
    }


def warning_from_validation_case(case: dict[str, Any]) -> str:
    failure_stage = case.get("failureStage")
    if failure_stage == "feature_generation":
        return "QupKake launched, but the validation molecule produced no usable features on this machine."
    if failure_stage == "no_sites":
        return "QupKake launched, but the validation molecule produced no predicted micro-pKa sites on this machine."
    details = case.get("rawStderr") or case.get("rawStdout")
    if details:
        return f"QupKake launched, but validation failed: {details}"
    return "QupKake launched, but validation failed during prediction."


def check_installation() -> dict[str, Any]:
    runtime = resolve_runtime()
    if not runtime.get("available"):
        return runtime

    runtime_probe = runtime.get("runtimeProbe")
    if not isinstance(runtime_probe, dict):
        runtime_probe = run_runtime_probe(runtime)
        runtime["runtimeProbe"] = runtime_probe
    if not runtime_probe.get("ok"):
        runtime["available"] = False
        runtime["validated"] = False
        runtime["failureStage"] = "runtime_setup"
        runtime["message"] = runtime_probe.get("message") or "QupKake runtime preflight failed."
        runtime["warning"] = runtime["message"]
        return runtime

    runtime["runtimeFingerprint"] = build_runtime_fingerprint(runtime)

    validation_ligand = find_validation_ligand()
    runtime["validationLigand"] = validation_ligand

    validation_report, validation_proc = run_validation_with_runtime(runtime)
    if validation_report is None:
        runtime["validated"] = False
        runtime["failureStage"] = "prediction_failed"
        runtime["warning"] = (
            trimmed_output(validation_proc.stderr)
            or trimmed_output(validation_proc.stdout)
            or "Failed to run the QupKake validation panel."
        )
        runtime["validationReport"] = {
            "controls": [],
            "requiredControlNames": ["acetic_acid_validation", "pyridine_validation"],
            "informationalControlNames": ["piperidine_validation", "indole_validation"],
            "validated": False,
            "summary": runtime["warning"],
            "overallFailureStage": runtime["failureStage"],
        }
        return runtime

    runtime["validationReport"] = normalize_validation_report(
        validation_report,
        runtime["runtimeFingerprint"],
    )
    runtime["validationCase"] = legacy_validation_case_from_report(runtime["validationReport"])
    runtime["validated"] = bool(runtime["validationReport"].get("validated"))
    if not runtime["validated"]:
        runtime["failureStage"] = runtime["validationReport"].get("overallFailureStage")
        if not runtime.get("failureStage") and runtime.get("validationCase"):
            runtime["failureStage"] = runtime["validationCase"].get("failureStage")
        runtime["warning"] = warning_from_validation_report(runtime["validationReport"])

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
    if not status.get("validated"):
        raise RuntimeError(status.get("warning") or "QupKake runtime failed validation and is blocked.")
    return run_prediction_with_runtime(ligand_path, status)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict ligand micro-pKa with external QupKake")
    parser.add_argument("--check", action="store_true", help="Check whether external QupKake is available")
    parser.add_argument("--probe-runtime", action="store_true", help="Run only the QupKake runtime preflight probe")
    parser.add_argument("--ligand", help="Input ligand file for a single prediction")
    args = parser.parse_args()

    try:
        if args.check:
            print(json.dumps(check_installation()))
            return
        if args.probe_runtime:
            runtime = resolve_runtime()
            if not runtime.get("available"):
                print(json.dumps(runtime))
                return
            print(json.dumps(run_runtime_probe(runtime)))
            return
        if not args.ligand:
            raise RuntimeError("--ligand is required unless --check is used")
        print(json.dumps(predict(args.ligand)))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
