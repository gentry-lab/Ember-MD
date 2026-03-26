#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
Convert a single SMILES string or MOL/SDF file to a 3D SDF file + 2D PNG thumbnail.

Used for single-molecule docking input in the GNINA pipeline.
"""

import argparse
import base64
import io
import json
import math
import os
import statistics
import sys
from typing import Any, Optional, Tuple

try:
    import rdkit.Chem as Chem
    from rdkit.Chem import AllChem, Draw, Descriptors
    from rdkit.Geometry import Point3D
except ImportError:
    print("ERROR:Missing dependency: rdkit", file=sys.stderr)
    sys.exit(1)


def median_bond_length(mol: Any) -> Optional[float]:
    """Return the median bond length in angstroms for the first conformer."""
    if mol.GetNumConformers() == 0 or mol.GetNumBonds() == 0:
        return None

    conf = mol.GetConformer()
    lengths = []
    for bond in mol.GetBonds():
        begin = conf.GetAtomPosition(bond.GetBeginAtomIdx())
        end = conf.GetAtomPosition(bond.GetEndAtomIdx())
        dx = begin.x - end.x
        dy = begin.y - end.y
        dz = begin.z - end.z
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length > 1e-6:
            lengths.append(length)

    if not lengths:
        return None
    return float(statistics.median(lengths))


def normalize_bond_lengths(mol: Any, target_length: float = 1.45) -> Any:
    """Rescale suspiciously stretched or shrunk coordinates to normal bond lengths."""
    if mol.GetNumConformers() == 0:
        return mol

    median_length = median_bond_length(mol)
    if median_length is None:
        return mol

    # Preserve ordinary molecular coordinates, but rescue 2D editor-style scales.
    if 0.9 <= median_length <= 1.9:
        return mol

    scale = target_length / median_length
    conf = mol.GetConformer()
    for atom_idx in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_idx)
        conf.SetAtomPosition(
            atom_idx,
            Point3D(pos.x * scale, pos.y * scale, pos.z * scale),
        )
    return mol


def smiles_to_mol(smiles: str) -> Tuple[Any, Optional[str]]:
    """Parse SMILES string and return RDKit molecule with 3D coords."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES string"

    mol = Chem.AddHs(mol)

    # Generate 3D conformer via ETKDG
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        # Fallback with less strict parameters
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            return None, "Failed to generate 3D coordinates"

    # Energy minimize
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
        except Exception:
            pass  # Keep unoptimized conformer

    return mol, None


def load_mol_file(mol_path: str) -> Tuple[Any, Optional[str]]:
    """Load a MOL or SDF file and return RDKit molecule with 3D coords."""
    ext = os.path.splitext(mol_path)[1].lower()

    if ext == '.sdf':
        supplier = Chem.SDMolSupplier(mol_path, removeHs=False)
        mol = next(iter(supplier), None)
    elif ext in ('.mol', '.mol2'):
        mol = Chem.MolFromMolFile(mol_path, removeHs=False)
    else:
        return None, f"Unsupported file format: {ext}"

    if mol is None:
        return None, f"Failed to parse molecule from {mol_path}"

    # If no 3D coordinates, generate them
    conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None
    if conf is None or conf.Is3D() is False:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        result = AllChem.EmbedMolecule(mol, params)
        if result == -1:
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result == -1:
                return None, "Failed to generate 3D coordinates"
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
        except Exception:
            pass
    else:
        mol = normalize_bond_lengths(mol)

    return mol, None


def generate_thumbnail(mol: Any, pixels_per_angstrom: int = 32, min_size: int = 150, max_size: int = 600) -> Tuple[str, int, int]:
    """Generate a 2D PNG thumbnail scaled to molecule size.

    Uses MolDraw2DCairo with fixed bond length so the image grows
    proportionally with the molecule.
    """
    from rdkit.Chem.Draw import rdMolDraw2D

    mol_2d = Chem.RWMol(mol)
    try:
        Chem.RemoveAllHs(mol_2d)
    except Exception:
        pass

    AllChem.Compute2DCoords(mol_2d)

    # Compute bounding box of 2D coordinates
    conf = mol_2d.GetConformer()
    xs = [conf.GetAtomPosition(i).x for i in range(mol_2d.GetNumAtoms())]
    ys = [conf.GetAtomPosition(i).y for i in range(mol_2d.GetNumAtoms())]
    span_x = max(xs) - min(xs) if xs else 1.0
    span_y = max(ys) - min(ys) if ys else 1.0

    padding = 3.0
    w = int((span_x + padding) * pixels_per_angstrom)
    h = int((span_y + padding) * pixels_per_angstrom)

    w = max(min_size, min(max_size, w))
    h = max(min_size, min(max_size, h))

    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.backgroundColour = (1, 1, 1, 1)
    opts.fixedBondLength = pixels_per_angstrom * 1.5
    drawer.DrawMolecule(mol_2d)
    drawer.FinishDrawing()

    png_data = drawer.GetDrawingText()
    return base64.b64encode(png_data).decode('utf-8'), w, h


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert SMILES/MOL to 3D SDF + thumbnail')
    parser.add_argument('--smiles', help='SMILES string to convert')
    parser.add_argument('--mol_file', help='MOL/SDF file to load')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--name', default='single_mol', help='Molecule name')
    args = parser.parse_args()

    if not args.smiles and not args.mol_file:
        print("ERROR: Either --smiles or --mol_file is required", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load or create molecule
    if args.smiles:
        mol, error = smiles_to_mol(args.smiles)
        if error:
            print(json.dumps({"error": error}))
            sys.exit(1)
        smiles = args.smiles
    else:
        mol, error = load_mol_file(args.mol_file)
        if error:
            print(json.dumps({"error": error}))
            sys.exit(1)
        smiles = Chem.MolToSmiles(Chem.RemoveAllHs(mol))

    # Calculate properties
    mol_no_h = Chem.RemoveAllHs(mol)
    qed = Descriptors.qed(mol_no_h)
    mw = Descriptors.MolWt(mol_no_h)

    # Write 3D SDF
    name = args.name.replace(' ', '_')
    sdf_path = os.path.join(args.output_dir, f'{name}.sdf')
    writer = Chem.SDWriter(sdf_path)
    mol.SetProp("_Name", name)
    mol.SetProp("SMILES", smiles)
    writer.write(mol)
    writer.close()

    # Generate 2D thumbnail
    thumbnail, thumb_w, thumb_h = generate_thumbnail(mol)

    # Output result as JSON
    result = {
        "sdfPath": sdf_path,
        "smiles": smiles,
        "name": name,
        "qed": round(qed, 3),
        "mw": round(mw, 1),
        "thumbnail": thumbnail,
        "thumbnailWidth": thumb_w,
        "thumbnailHeight": thumb_h,
    }
    print(json.dumps(result))


if __name__ == '__main__':
    main()
