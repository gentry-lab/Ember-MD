#!/usr/bin/env python3
"""
Align small molecules by MCS or shared rigid substructures.

Usage:
  align_molecules.py --mode mcs --ref REF.sdf --mobile MOBILE.sdf --out ALIGNED.sdf
  align_molecules.py --mode scaffolds --ref REF.sdf --mobile MOBILE.sdf
  align_molecules.py --mode align_scaffold --ref REF.sdf --mobile MOBILE.sdf --scaffold-index 0 --out ALIGNED.sdf
"""
import argparse
import json
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, Draw


def load_molecule(path: str) -> Chem.Mol:
    """Load a molecule from SDF, return the first conformer."""
    suppl = Chem.SDMolSupplier(path, removeHs=False)
    mol = next(suppl, None)
    if mol is None:
        raise ValueError(f"Could not read molecule from {path}")
    return mol


def mcs_align(ref_mol: Chem.Mol, mobile_mol: Chem.Mol) -> Chem.Mol:
    """Align mobile to ref by Maximum Common Substructure."""
    ref_noH = Chem.RemoveHs(ref_mol)
    mob_noH = Chem.RemoveHs(mobile_mol)

    mcs = rdFMCS.FindMCS(
        [ref_noH, mob_noH],
        threshold=1.0,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=10,
    )
    if mcs.numAtoms < 2:
        raise ValueError("No meaningful common substructure found (< 2 atoms)")

    pattern = Chem.MolFromSmarts(mcs.smartsString)
    ref_match = ref_noH.GetSubstructMatch(pattern)
    mob_match = mob_noH.GetSubstructMatch(pattern)

    if not ref_match or not mob_match:
        raise ValueError("Substructure match failed after MCS detection")

    atom_map = list(zip(mob_match, ref_match))

    # Work on heavy-atom molecule for alignment, then propagate to full mol
    rms = AllChem.AlignMol(mob_noH, ref_noH, atomMap=atom_map)

    # Copy aligned coords back to the mobile (with H) molecule
    conf_aligned = mob_noH.GetConformer()
    conf_mobile = mobile_mol.GetConformer()
    heavy_to_full = {i: i for i in range(mob_noH.GetNumAtoms())}
    # Map heavy atom indices in mob_noH back to mobile_mol
    mob_noH_to_full = mobile_mol.GetSubstructMatch(mob_noH)
    if mob_noH_to_full:
        for noH_idx, full_idx in enumerate(mob_noH_to_full):
            pos = conf_aligned.GetAtomPosition(noH_idx)
            conf_mobile.SetAtomPosition(full_idx, pos)

    return mobile_mol


def detect_shared_ring_scaffolds(ref_mol: Chem.Mol, mobile_mol: Chem.Mol) -> list:
    """
    Find shared rigid ring substructures between two molecules.
    Returns list of { label, smartsPattern, refAtomIndices, mobileAtomIndices }.
    """
    ref_noH = Chem.RemoveHs(ref_mol)
    mob_noH = Chem.RemoveHs(mobile_mol)

    # Get ring systems from each molecule
    ref_ring_info = ref_noH.GetRingInfo()
    mob_ring_info = mob_noH.GetRingInfo()

    scaffolds = []
    seen_smarts = set()

    # For each ring in the reference, check if mobile has a matching ring
    for ring_atoms in ref_ring_info.AtomRings():
        # Build a SMARTS pattern for this ring
        env = Chem.MolFragmentToSmiles(ref_noH, ring_atoms, canonical=True)
        ring_mol = Chem.MolFromSmiles(env)
        if ring_mol is None:
            continue
        smarts = Chem.MolToSmarts(ring_mol)
        if smarts in seen_smarts:
            continue

        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue

        ref_match = ref_noH.GetSubstructMatch(pattern)
        mob_match = mob_noH.GetSubstructMatch(pattern)

        if ref_match and mob_match:
            seen_smarts.add(smarts)
            # Try to give a human-readable label
            label = _ring_label(ring_mol, env)
            scaffolds.append({
                "label": label,
                "smartsPattern": smarts,
                "refAtomIndices": list(ref_match),
                "mobileAtomIndices": list(mob_match),
            })

    return scaffolds


def _ring_label(ring_mol: Chem.Mol, smiles: str) -> str:
    """Generate a human-readable label for a ring system."""
    ring_names = {
        "c1ccccc1": "Phenyl",
        "c1ccncc1": "Pyridine",
        "c1ccoc1": "Furan",
        "c1ccsc1": "Thiophene",
        "c1cc[nH]c1": "Pyrrole",
        "c1ccc2[nH]ccc2c1": "Indole",
        "c1ccc2ncccc2c1": "Quinoline",
        "c1ccc2ccccc2c1": "Naphthalene",
        "c1cnc2ccccc2n1": "Quinazoline",
        "c1ccc2[nH]ncc2c1": "Indazole",
        "c1cnc[nH]1": "Imidazole",
        "c1ccnc(=O)[nH]1": "Pyrimidone",
    }
    canonical = Chem.MolToSmiles(ring_mol, canonical=True)
    for pattern, name in ring_names.items():
        if canonical == pattern:
            return name
    n_atoms = ring_mol.GetNumAtoms()
    n_hetero = sum(1 for a in ring_mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
    if n_hetero == 0:
        return f"{n_atoms}-ring"
    return f"{n_atoms}-ring ({n_hetero} het)"


def scaffold_align(ref_mol: Chem.Mol, mobile_mol: Chem.Mol, scaffold_index: int) -> Chem.Mol:
    """Align mobile to ref by the specified scaffold index."""
    scaffolds = detect_shared_ring_scaffolds(ref_mol, mobile_mol)
    if scaffold_index >= len(scaffolds):
        raise ValueError(f"Scaffold index {scaffold_index} out of range (found {len(scaffolds)})")

    scaffold = scaffolds[scaffold_index]
    atom_map = list(zip(scaffold["mobileAtomIndices"], scaffold["refAtomIndices"]))

    ref_noH = Chem.RemoveHs(ref_mol)
    mob_noH = Chem.RemoveHs(mobile_mol)
    AllChem.AlignMol(mob_noH, ref_noH, atomMap=atom_map)

    # Copy aligned coords back
    conf_aligned = mob_noH.GetConformer()
    conf_mobile = mobile_mol.GetConformer()
    mob_noH_to_full = mobile_mol.GetSubstructMatch(mob_noH)
    if mob_noH_to_full:
        for noH_idx, full_idx in enumerate(mob_noH_to_full):
            pos = conf_aligned.GetAtomPosition(noH_idx)
            conf_mobile.SetAtomPosition(full_idx, pos)

    return mobile_mol


def write_sdf(mol: Chem.Mol, path: str):
    """Write a single molecule to SDF."""
    writer = Chem.SDWriter(path)
    writer.write(mol)
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Align molecules by MCS or shared scaffolds")
    parser.add_argument("--mode", required=True, choices=["mcs", "scaffolds", "align_scaffold"])
    parser.add_argument("--ref", required=True, help="Reference molecule SDF")
    parser.add_argument("--mobile", required=True, help="Mobile molecule SDF")
    parser.add_argument("--out", help="Output aligned SDF (required for mcs/align_scaffold)")
    parser.add_argument("--scaffold-index", type=int, default=0, help="Scaffold index for align_scaffold mode")
    args = parser.parse_args()

    ref_mol = load_molecule(args.ref)
    mobile_mol = load_molecule(args.mobile)

    if args.mode == "mcs":
        aligned = mcs_align(ref_mol, mobile_mol)
        if not args.out:
            print("ERROR: --out required for mcs mode", file=sys.stderr)
            sys.exit(1)
        write_sdf(aligned, args.out)
        print(json.dumps({"status": "ok", "output": args.out}))

    elif args.mode == "scaffolds":
        scaffolds = detect_shared_ring_scaffolds(ref_mol, mobile_mol)
        print(json.dumps({"status": "ok", "scaffolds": scaffolds}))

    elif args.mode == "align_scaffold":
        aligned = scaffold_align(ref_mol, mobile_mol, args.scaffold_index)
        if not args.out:
            print("ERROR: --out required for align_scaffold mode", file=sys.stderr)
            sys.exit(1)
        write_sdf(aligned, args.out)
        print(json.dumps({"status": "ok", "output": args.out, "scaffoldIndex": args.scaffold_index}))


if __name__ == "__main__":
    main()
