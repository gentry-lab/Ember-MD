#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem


def load_molecule(name: str, input_value: str, kind: str) -> Chem.Mol:
    if kind == "smiles":
        mol = Chem.MolFromSmiles(input_value)
        if mol is None:
            raise RuntimeError(f"Failed to parse SMILES for {name}: {input_value}")
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE) != 0:
            raise RuntimeError(f"Failed to embed {name}")
        AllChem.MMFFOptimizeMolecule(mol)
        mol.SetProp("_Name", name)
        return mol

    supplier = Chem.SDMolSupplier(input_value, removeHs=False)
    mol = supplier[0] if supplier and len(supplier) else None
    if mol is None:
        raise RuntimeError(f"Failed to load SDF for {name}: {input_value}")
    if not mol.HasProp("_Name"):
        mol.SetProp("_Name", name)
    return mol


def xtb_version(xtb_path: str) -> dict[str, Any]:
    proc = subprocess.run(
        [xtb_path, "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    text = (proc.stdout or proc.stderr).strip()
    version = None
    for line in text.splitlines():
        if "xtb version" in line:
            parts = line.split()
            if len(parts) >= 4:
                version = parts[3]
            break
    return {
        "path": xtb_path,
        "version": version,
        "raw": text,
        "returnCode": proc.returncode,
    }


def summarize_xtb_attributes(attrs: dict[str, Any]) -> dict[str, Any]:
    atomprop = attrs.get("atomprop", {})
    bondprop = attrs.get("bondprop", {})
    fukui = atomprop.get("fukui")
    return {
        "keys": sorted(attrs.keys()),
        "metadata": attrs.get("metadata"),
        "charge": attrs.get("charge"),
        "totalenergy": attrs.get("totalenergy"),
        "natom": attrs.get("natom"),
        "atompropKeys": sorted(atomprop.keys()),
        "bondpropKeys": sorted(bondprop.keys()),
        "qCount": len(atomprop.get("q", [])),
        "convcnCount": len(atomprop.get("convcn", [])),
        "alphaCount": len(atomprop.get("alpha", [])),
        "wboRows": len(bondprop.get("wbo", [])),
        "fukuiCounts": [len(x) for x in fukui] if isinstance(fukui, list) else None,
        "qHead": atomprop.get("q", [])[:5],
        "convcnHead": atomprop.get("convcn", [])[:5],
        "alphaHead": atomprop.get("alpha", [])[:5],
        "fukuiHead": [x[:5] for x in fukui] if isinstance(fukui, list) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe QupKake xTB feature generation")
    parser.add_argument("--qupkake-root", required=True, help="Path to qupkake source tree")
    parser.add_argument("--xtb-path", required=True, help="Path to xtb executable")
    parser.add_argument("--name", action="append", dest="names", default=[])
    parser.add_argument("--smiles", action="append", dest="smiles_inputs", default=[])
    parser.add_argument("--sdf", action="append", dest="sdf_inputs", default=[])
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.qupkake_root).resolve()))
    os.environ["QUPKAKE_XTBPATH"] = str(Path(args.xtb_path).resolve())
    os.environ["XTBPATH"] = str(Path(args.xtb_path).resolve())

    import torch
    from qupkake.featurizer import Featurizer
    from qupkake.predict import load_models
    from qupkake.xtbp import RunXTB, XTBP

    prot_model, deprot_model, _ = load_models()
    prot_model.eval()
    deprot_model.eval()

    entries: list[tuple[str, str, str]] = []
    for i, smiles in enumerate(args.smiles_inputs):
        name = args.names[i] if i < len(args.names) else f"smiles_{i+1}"
        entries.append((name, smiles, "smiles"))
    sdf_offset = len(entries)
    for i, sdf in enumerate(args.sdf_inputs):
        name_idx = sdf_offset + i
        name = args.names[name_idx] if name_idx < len(args.names) else Path(sdf).stem
        entries.append((name, sdf, "sdf"))

    results: list[dict[str, Any]] = []
    for name, input_value, kind in entries:
        item: dict[str, Any] = {"name": name, "kind": kind, "input": input_value}
        try:
            mol = load_molecule(name, input_value, kind)
            item["smiles"] = Chem.MolToSmiles(Chem.RemoveHs(mol))
            item["atomCount"] = mol.GetNumAtoms()

            opt_out = RunXTB(mol, "--opt --alpb water --lmo -P 1")()
            opt_attrs = XTBP(opt_out)()
            item["opt"] = summarize_xtb_attributes(opt_attrs)

            fukui_out = RunXTB(mol, "--vfukui")()
            fukui_attrs = XTBP(fukui_out)()
            item["vfukui"] = summarize_xtb_attributes(fukui_attrs)

            combined = dict(opt_attrs)
            combined.setdefault("atomprop", {})
            combined["atomprop"] = dict(combined.get("atomprop", {}))
            combined["atomprop"]["fukui"] = fukui_attrs.get("atomprop", {}).get("fukui")
            item["combined"] = summarize_xtb_attributes(combined)

            try:
                graph = Featurizer(mol=mol, name=name, num_processes=1).data
                atom_rows = []
                with torch.no_grad():
                    prot_logits = prot_model(
                        x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr
                    ).reshape(-1)
                    deprot_logits = deprot_model(
                        x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr
                    ).reshape(-1)
                    prot_probs = torch.sigmoid(prot_logits)
                    deprot_probs = torch.sigmoid(deprot_logits)
                for atom in mol.GetAtoms():
                    idx = atom.GetIdx()
                    atom_rows.append(
                        {
                            "idx": idx,
                            "symbol": atom.GetSymbol(),
                            "protLogit": round(float(prot_logits[idx]), 6),
                            "protProb": round(float(prot_probs[idx]), 6),
                            "deprotLogit": round(float(deprot_logits[idx]), 6),
                            "deprotProb": round(float(deprot_probs[idx]), 6),
                        }
                    )
                item["featurizer"] = {
                    "ok": True,
                    "nodeShape": list(graph.x.shape),
                    "edgeShape": list(graph.edge_attr.shape),
                    "globalShape": list(graph.global_attr.shape),
                    "energy": graph.energy.tolist() if hasattr(graph, "energy") else None,
                    "nodeHead": graph.x[: min(3, graph.x.shape[0])].tolist(),
                    "globalHead": graph.global_attr.tolist(),
                    "siteModel": {
                        "protPositiveAtoms": [i for i, value in enumerate(prot_logits.tolist()) if value > 0],
                        "deprotPositiveAtoms": [i for i, value in enumerate(deprot_logits.tolist()) if value > 0],
                        "atomScores": atom_rows,
                    },
                }
            except Exception as exc:
                item["featurizer"] = {"ok": False, "error": repr(exc)}
        except Exception as exc:
            item["error"] = repr(exc)
        results.append(item)

    payload = {
        "runtime": {
            "python": sys.executable,
            "qupkakeRoot": str(Path(args.qupkake_root).resolve()),
            "xtb": xtb_version(str(Path(args.xtb_path).resolve())),
        },
        "results": results,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
