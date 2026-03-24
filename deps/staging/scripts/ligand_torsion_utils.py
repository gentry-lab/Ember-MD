#!/usr/bin/env python3
"""
Shared ligand torsion and 2D depiction utilities for MD analysis scripts.

This module is the canonical source of truth for:
  - rotatable bond detection
  - stable torsion descriptors
  - 2D depiction geometry
  - ligand reconstruction from centroid heavy-atom coordinates
"""

from __future__ import annotations

import base64
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils import load_sdf, select_ligand_atoms


ROTATABLE_SMARTS = '[!$([NH]!@C(=O))&!D1]-&!@[!$([NH]!@C(=O))&!D1]'


def _as_float_triplets(coords: Iterable[Sequence[float]]) -> List[Tuple[float, float, float]]:
    return [(float(x), float(y), float(z)) for x, y, z in coords]


def load_canonical_ligand_mol(ligand_sdf_path: str) -> Any:
    """Load the canonical ligand template and strip hydrogens for stable indexing."""
    from rdkit import Chem

    mol = load_sdf(ligand_sdf_path, remove_hs=False)
    if mol is None:
        raise ValueError(f'Failed to load ligand template: {ligand_sdf_path}')

    heavy = Chem.RemoveHs(mol, sanitize=False)
    if heavy is None:
        raise ValueError(f'Failed to remove hydrogens from ligand template: {ligand_sdf_path}')

    try:
        Chem.SanitizeMol(heavy)
    except Exception:
        pass
    return heavy


def _get_atom_label(atom: Any) -> str:
    pdb_info = atom.GetPDBResidueInfo()
    if pdb_info is not None:
        name = pdb_info.GetName().strip()
        if name:
            return name
    return f'{atom.GetSymbol()}{atom.GetIdx() + 1}'


def _pick_dihedral_neighbor(mol: Any, atom_idx: int, exclude_idx: int) -> Optional[int]:
    atom = mol.GetAtomWithIdx(atom_idx)
    neighbors = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != exclude_idx and nbr.GetAtomicNum() > 1
    ]
    if not neighbors:
        return None
    return max(neighbors, key=lambda idx: mol.GetAtomWithIdx(idx).GetAtomicNum())


def make_bond_id(atom_i: int, atom_j: int) -> str:
    lo = min(atom_i, atom_j)
    hi = max(atom_i, atom_j)
    return f'bond_{lo}_{hi}'


def build_torsion_descriptors(ligand_mol: Any) -> List[Dict[str, Any]]:
    """Build stable torsion descriptors from the canonical heavy-atom ligand mol."""
    from rdkit import Chem

    pattern = Chem.MolFromSmarts(ROTATABLE_SMARTS)
    matches = ligand_mol.GetSubstructMatches(pattern)
    seen_bonds = set()
    descriptors: List[Dict[str, Any]] = []

    for match in matches:
        atom_i, atom_j = int(match[0]), int(match[1])
        bond_key = tuple(sorted((atom_i, atom_j)))
        if bond_key in seen_bonds:
            continue
        seen_bonds.add(bond_key)

        atom_a = _pick_dihedral_neighbor(ligand_mol, atom_i, atom_j)
        atom_d = _pick_dihedral_neighbor(ligand_mol, atom_j, atom_i)
        if atom_a is None or atom_d is None:
            continue

        bond = ligand_mol.GetBondBetweenAtoms(atom_i, atom_j)
        if bond is None:
            continue

        quartet = [atom_a, atom_i, atom_j, atom_d]
        atom_names = [_get_atom_label(ligand_mol.GetAtomWithIdx(idx)) for idx in quartet]
        bond_id = make_bond_id(atom_i, atom_j)

        descriptors.append({
            'torsionId': f'tor_{len(descriptors) + 1}',
            'bondId': bond_id,
            'bondIndex': int(bond.GetIdx()),
            'centralBondAtomIndices': [atom_i, atom_j],
            'quartetAtomIndices': quartet,
            'atomNames': atom_names,
            'label': '-'.join(atom_names),
        })

    return descriptors


def build_depiction_payload(ligand_mol: Any) -> Dict[str, Any]:
    """Build lightweight 2D depiction geometry for frontend SVG rendering."""
    from rdkit import Chem
    from rdkit.Chem import rdDepictor

    mol = Chem.Mol(ligand_mol)
    rdDepictor.Compute2DCoords(mol)

    conf = mol.GetConformer()
    atoms = []
    xs: List[float] = []
    ys: List[float] = []

    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        x = float(pos.x)
        y = float(-pos.y)
        xs.append(x)
        ys.append(y)
        atoms.append({
            'atomIndex': int(atom.GetIdx()),
            'symbol': atom.GetSymbol(),
            'label': _get_atom_label(atom),
            'x': x,
            'y': y,
            'formalCharge': int(atom.GetFormalCharge()),
            'isAromatic': bool(atom.GetIsAromatic()),
            'showLabel': atom.GetSymbol() != 'C' or atom.GetFormalCharge() != 0,
        })

    bonds = []
    for bond in mol.GetBonds():
        begin_idx = int(bond.GetBeginAtomIdx())
        end_idx = int(bond.GetEndAtomIdx())
        begin_pos = conf.GetAtomPosition(begin_idx)
        end_pos = conf.GetAtomPosition(end_idx)
        bonds.append({
            'bondId': make_bond_id(begin_idx, end_idx),
            'bondIndex': int(bond.GetIdx()),
            'beginAtomIndex': begin_idx,
            'endAtomIndex': end_idx,
            'order': int(round(float(bond.GetBondTypeAsDouble()))),
            'isAromatic': bool(bond.GetIsAromatic()),
            'x1': float(begin_pos.x),
            'y1': float(-begin_pos.y),
            'x2': float(end_pos.x),
            'y2': float(-end_pos.y),
        })

    min_x = min(xs) if xs else 0.0
    max_x = max(xs) if xs else 1.0
    min_y = min(ys) if ys else 0.0
    max_y = max(ys) if ys else 1.0

    return {
        'atoms': atoms,
        'bonds': bonds,
        'bounds': {
            'minX': min_x,
            'maxX': max_x,
            'minY': min_y,
            'maxY': max_y,
        },
    }


def render_png_data_url(ligand_mol: Any, size: int = 300) -> Optional[str]:
    """Render a PNG data URL from the same canonical depiction mol."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
        from rdkit.Chem import rdDepictor
    except ImportError:
        return None

    mol = Chem.Mol(ligand_mol)
    rdDepictor.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    opts = drawer.drawOptions()
    opts.padding = 0.08
    opts.useBWAtomPalette()
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()
    b64_data = base64.b64encode(png_bytes).decode('utf-8')
    return f'data:image/png;base64,{b64_data}'


def build_torsion_identity_bundle(ligand_sdf_path: str) -> Dict[str, Any]:
    """Build the canonical torsion descriptor + depiction payload from a ligand SDF."""
    ligand_mol = load_canonical_ligand_mol(ligand_sdf_path)
    return {
        'ligandMol': ligand_mol,
        'depiction': build_depiction_payload(ligand_mol),
        'torsions': build_torsion_descriptors(ligand_mol),
    }


def validate_heavy_atom_count(ligand_mol: Any, observed_count: int) -> None:
    expected = int(ligand_mol.GetNumAtoms())
    if expected != int(observed_count):
        raise ValueError(
            f'Ligand heavy atom count mismatch: expected {expected}, observed {observed_count}'
        )


def compute_dihedral_angles_for_positions(
    positions: Sequence[Sequence[float]],
    torsion_descriptors: Sequence[Dict[str, Any]],
) -> List[float]:
    """Compute torsion values for one conformer given heavy-atom positions."""
    import numpy as np
    from MDAnalysis.lib.distances import calc_dihedrals

    coords = np.asarray(positions, dtype=np.float32)
    values: List[float] = []
    for descriptor in torsion_descriptors:
        atom_a, atom_b, atom_c, atom_d = descriptor['quartetAtomIndices']
        angle_rad = calc_dihedrals(
            coords[atom_a].reshape(1, 3),
            coords[atom_b].reshape(1, 3),
            coords[atom_c].reshape(1, 3),
            coords[atom_d].reshape(1, 3),
        )
        values.append(float(np.rad2deg(angle_rad[0])))
    return values


def compute_dihedral_angles_for_mol(
    ligand_mol: Any,
    torsion_descriptors: Sequence[Dict[str, Any]],
) -> List[float]:
    conf = ligand_mol.GetConformer()
    coords = [
        (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z)
        for i in range(ligand_mol.GetNumAtoms())
    ]
    return compute_dihedral_angles_for_positions(coords, torsion_descriptors)


def rebuild_ligand_from_heavy_coords(
    input_ligand_sdf: str,
    heavy_coords: Iterable[Sequence[float]],
    *,
    include_hs: bool,
) -> Any:
    """Rebuild a ligand conformer from canonical heavy-atom coordinates."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D

    coords = _as_float_triplets(heavy_coords)
    base_no_h = load_canonical_ligand_mol(input_ligand_sdf)
    if base_no_h.GetNumAtoms() != len(coords):
        raise ValueError(
            f'Heavy atom count mismatch for ligand rebuild: expected {base_no_h.GetNumAtoms()}, got {len(coords)}'
        )

    positioned_no_h = Chem.Mol(base_no_h)
    conf = positioned_no_h.GetConformer()
    for atom_idx, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(atom_idx, Point3D(x, y, z))

    if not include_hs:
        return positioned_no_h

    positioned_h = Chem.AddHs(positioned_no_h, addCoords=True)
    try:
        AllChem.ConstrainedEmbed(positioned_h, positioned_no_h, randomseed=42)
    except Exception:
        pass
    return positioned_h


def extract_ligand_heavy_coords_from_centroid_pdb(centroid_pdb: str) -> List[Tuple[float, float, float]]:
    import MDAnalysis as mda

    universe = mda.Universe(centroid_pdb)
    ligand = select_ligand_atoms(universe)
    if len(ligand) == 0:
        raise ValueError(f'No ligand atoms found in centroid PDB: {centroid_pdb}')
    return _as_float_triplets(ligand.positions)


def rebuild_ligand_from_centroid_pdb(
    centroid_pdb: str,
    input_ligand_sdf: str,
    *,
    include_hs: bool,
) -> Any:
    coords = extract_ligand_heavy_coords_from_centroid_pdb(centroid_pdb)
    return rebuild_ligand_from_heavy_coords(input_ligand_sdf, coords, include_hs=include_hs)
