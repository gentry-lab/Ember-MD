#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
Extract a ligand from an X-ray PDB structure and convert to SDF for MD simulation.

Handles the bond order problem: PDB HETATM records don't store bond orders,
but OpenFF/SMIRNOFF requires correct chemistry for parameterization.

Approach:
1. Extract ligand HETATM records from PDB
2. Try OpenBabel PDB→SDF conversion (handles most common ligands)
3. If SMILES is provided, use it as a template to assign bond orders
4. Generate 2D thumbnail for visual verification
"""

import argparse
import base64
import io
import json
import os
import sys
import tempfile
from typing import Any, Optional, Tuple


from utils import convert_cif_to_pdb

try:
    import rdkit.Chem as Chem
    from rdkit.Chem import AllChem, Draw, Descriptors
except ImportError:
    print("ERROR:Missing dependency: rdkit", file=sys.stderr)
    sys.exit(1)


def extract_ligand_pdb(pdb_path: str, ligand_id: str, output_pdb: str) -> None:
    """Extract HETATM records for a specific ligand to a PDB file."""
    resname, chain, resnum = ligand_id.rsplit('_', 2)

    with open(pdb_path, 'r') as f_in, open(output_pdb, 'w') as f_out:
        for line in f_in:
            if line.startswith('HETATM'):
                line_resname = line[17:20].strip()
                line_chain = line[21].strip() or '_'
                line_resnum = line[22:26].strip()
                if line_resname == resname and line_chain == chain and line_resnum == resnum:
                    f_out.write(line)
        f_out.write("END\n")


def pdb_to_sdf_rdkit(pdb_path: str) -> Tuple[Any, Optional[str]]:
    """Try to convert ligand PDB to RDKit mol using RDKit's PDB reader."""
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
    if mol is None:
        return None, "RDKit failed to parse PDB"

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        # Try without sanitization — bond orders may be wrong
        return None, f"Sanitization failed: {e}"

    return mol, None


def pdb_to_sdf_obabel(pdb_path: str, sdf_path: str) -> Tuple[Any, Optional[str]]:
    """Try to convert ligand PDB to SDF using OpenBabel."""
    import subprocess, shutil

    # Find obabel: check alongside the running Python interpreter first,
    # then fall back to PATH (handles Electron spawning without conda activation)
    python_bin_dir = os.path.dirname(os.path.realpath(sys.executable))
    obabel_bin = os.path.join(python_bin_dir, 'obabel')
    if not os.path.isfile(obabel_bin):
        obabel_bin = shutil.which('obabel') or 'obabel'

    print(f"  obabel path: {obabel_bin} (exists: {os.path.isfile(obabel_bin)})", file=sys.stderr)

    # Set BABEL_DATADIR and BABEL_LIBDIR for bundled OpenBabel
    env = os.environ.copy()
    conda_prefix = os.path.dirname(python_bin_dir)
    babel_data = os.path.join(conda_prefix, 'share', 'openbabel')
    if os.path.isdir(babel_data):
        for d in os.listdir(babel_data):
            full = os.path.join(babel_data, d)
            if os.path.isdir(full):
                env['BABEL_DATADIR'] = full
                break
    babel_lib = os.path.join(conda_prefix, 'lib', 'openbabel')
    if os.path.isdir(babel_lib):
        for d in os.listdir(babel_lib):
            full = os.path.join(babel_lib, d)
            if os.path.isdir(full):
                env['BABEL_LIBDIR'] = full
                break
    # Also set DYLD_LIBRARY_PATH so obabel finds its shared libs
    lib_dir = os.path.join(conda_prefix, 'lib')
    env['DYLD_LIBRARY_PATH'] = lib_dir + ':' + env.get('DYLD_LIBRARY_PATH', '')

    try:
        result = subprocess.run(
            [obabel_bin, pdb_path, '-O', sdf_path],
            capture_output=True, text=True, timeout=30,
            env=env
        )
        if result.returncode == 0 and os.path.exists(sdf_path):
            mol = Chem.SDMolSupplier(sdf_path, removeHs=False)
            m = next(iter(mol), None)
            if m is not None:
                return m, None
        return None, f"OpenBabel conversion failed: {result.stderr[:200]}"
    except FileNotFoundError:
        return None, "OpenBabel (obabel) not found"
    except Exception as e:
        return None, str(e)


def assign_bond_orders_from_smiles(pdb_path: str, smiles: str) -> Tuple[Any, Optional[str]]:
    """Load PDB coordinates and assign bond orders from a SMILES template."""
    # Load the PDB (with potentially wrong bond orders)
    pdb_mol = Chem.MolFromPDBFile(pdb_path, removeHs=True, sanitize=False)
    if pdb_mol is None:
        return None, "Failed to load PDB"

    # Parse the SMILES template
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        return None, "Invalid SMILES string"

    try:
        # Assign bond orders from the template to the PDB coordinates
        mol = AllChem.AssignBondOrdersFromTemplate(template, pdb_mol)
        mol = Chem.AddHs(mol, addCoords=True)
        return mol, None
    except Exception as e:
        return None, f"Bond order assignment failed: {e}"


def generate_thumbnail(mol: Any, pixels_per_angstrom: int = 32, min_size: int = 150, max_size: int = 600) -> Tuple[str, int, int]:
    """Generate a 2D PNG thumbnail scaled to molecule size.

    Uses MolDraw2DCairo with fixed bond length so the image grows
    proportionally with the molecule. Small fragments get small images;
    large macrocycles get larger ones.
    """
    from rdkit.Chem.Draw import rdMolDraw2D

    mol_2d = Chem.RWMol(mol)
    try:
        mol_2d = Chem.RemoveAllHs(mol_2d)
    except Exception:
        pass

    AllChem.Compute2DCoords(mol_2d)

    # Compute bounding box of 2D coordinates
    conf = mol_2d.GetConformer()
    xs = [conf.GetAtomPosition(i).x for i in range(mol_2d.GetNumAtoms())]
    ys = [conf.GetAtomPosition(i).y for i in range(mol_2d.GetNumAtoms())]
    span_x = max(xs) - min(xs) if xs else 1.0
    span_y = max(ys) - min(ys) if ys else 1.0

    # Add padding (roughly 2 bond lengths on each side)
    padding = 3.0  # angstroms
    w = int((span_x + padding) * pixels_per_angstrom)
    h = int((span_y + padding) * pixels_per_angstrom)

    # Clamp to min/max
    w = max(min_size, min(max_size, w))
    h = max(min_size, min(max_size, h))

    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.backgroundColour = (1, 1, 1, 1)
    opts.fixedBondLength = pixels_per_angstrom * 1.5  # 1.5 A avg bond length
    drawer.DrawMolecule(mol_2d)
    drawer.FinishDrawing()

    png_data = drawer.GetDrawingText()
    return base64.b64encode(png_data).decode('utf-8'), w, h


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract X-ray ligand and convert to SDF')
    parser.add_argument('--pdb', required=True, help='Input PDB file')
    parser.add_argument('--ligand_id', required=True, help='Ligand ID (e.g., STI_A_501)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--smiles', help='SMILES template for bond order assignment (optional)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-convert CIF to PDB
    if args.pdb.lower().endswith('.cif'):
        args.pdb = convert_cif_to_pdb(args.pdb)

    # Step 1: Extract ligand PDB
    ligand_pdb = os.path.join(args.output_dir, f'{args.ligand_id}_extracted.pdb')
    extract_ligand_pdb(args.pdb, args.ligand_id, ligand_pdb)
    print(f"Extracted ligand to {ligand_pdb}", file=sys.stderr)

    # Step 2: Convert to molecule with correct bond orders
    mol = None
    method = None

    if args.smiles:
        # User provided SMILES — use as bond order template (most reliable)
        mol, error = assign_bond_orders_from_smiles(ligand_pdb, args.smiles)
        if mol:
            method = 'smiles_template'
        else:
            print(f"SMILES template failed: {error}, trying fallbacks...", file=sys.stderr)

    if mol is None:
        # Try OpenBabel first (better at bond order perception from 3D coords)
        temp_sdf = os.path.join(args.output_dir, f'{args.ligand_id}_obabel.sdf')
        mol, error = pdb_to_sdf_obabel(ligand_pdb, temp_sdf)
        if mol:
            method = 'openbabel'
        else:
            print(f"OpenBabel failed: {error}", file=sys.stderr)

    if mol is None:
        # Try RDKit PDB reader as last resort
        mol, error = pdb_to_sdf_rdkit(ligand_pdb)
        if mol:
            method = 'rdkit_pdb'
        else:
            print(f"RDKit PDB reader also failed: {error}", file=sys.stderr)

    if mol is None:
        print(json.dumps({
            "error": "Could not determine bond orders. Please provide a SMILES string.",
            "needsSmiles": True,
            "ligandPdb": ligand_pdb
        }))
        sys.exit(0)  # Not an error — just needs user input

    # Step 3: Write SDF
    sdf_path = os.path.join(args.output_dir, f'{args.ligand_id}.sdf')
    writer = Chem.SDWriter(sdf_path)
    mol.SetProp("_Name", args.ligand_id)
    smiles = Chem.MolToSmiles(Chem.RemoveAllHs(mol))
    mol.SetProp("SMILES", smiles)
    writer.write(mol)
    writer.close()

    # Step 4: Calculate properties
    mol_no_h = Chem.RemoveAllHs(mol)
    qed = Descriptors.qed(mol_no_h)
    mw = Descriptors.MolWt(mol_no_h)

    # Step 5: Generate thumbnail (scaled to molecule size)
    thumbnail, thumb_w, thumb_h = generate_thumbnail(mol)

    result = {
        "sdfPath": sdf_path,
        "ligandPdb": ligand_pdb,
        "smiles": smiles,
        "name": args.ligand_id,
        "qed": round(qed, 3),
        "mw": round(mw, 1),
        "thumbnail": thumbnail,
        "thumbnailWidth": thumb_w,
        "thumbnailHeight": thumb_h,
        "method": method,
    }
    print(json.dumps(result))


if __name__ == '__main__':
    main()
