#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "deps" / "staging" / "scripts" / "predict_ligand_pka.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Ember's QupKake validation report")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for the wrapper")
    parser.add_argument("--require-valid", action="store_true", help="Fail if the runtime does not validate")
    args = parser.parse_args()

    proc = subprocess.run(
        [args.python, str(WRAPPER), "--check"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or proc.stdout.strip() or "Wrapper check failed")

    payload = json.loads(proc.stdout.strip())
    report = payload.get("validationReport")
    if not isinstance(report, dict):
        raise SystemExit("Missing validationReport in QupKake capability output")

    controls = report.get("controls")
    if not isinstance(controls, list) or not controls:
        raise SystemExit("validationReport.controls was missing or empty")

    control_names = {control.get("name") for control in controls if isinstance(control, dict)}
    required = {"acetic_acid_validation", "pyridine_validation"}
    if not required.issubset(control_names):
        raise SystemExit(f"Missing required validation controls: {sorted(required - control_names)}")

    if args.require_valid and not payload.get("validated"):
        raise SystemExit(payload.get("warning") or "QupKake runtime did not validate")

    print(
        json.dumps(
            {
                "validated": bool(payload.get("validated")),
                "failureStage": payload.get("failureStage"),
                "requiredControls": sorted(required),
                "observedControls": sorted(control_names),
            }
        )
    )


if __name__ == "__main__":
    main()
