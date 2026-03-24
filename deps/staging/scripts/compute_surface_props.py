#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
Compute per-atom surface coloring properties for the currently loaded molecule.

The viewer uses NGL's real molecular surface geometry and colors that surface from
these per-atom fields. This script therefore focuses on current-structure,
atom-resolved properties rather than residue-level proxies.

Supported inputs:
- protein / complex structures: PDB, CIF
- ligands / small molecules: SDF, SDF.GZ, MOL, MOL2, PDB

Outputs:
- hydrophobic: atom-based lipophilic potential, normalized to [-1, 1]
- electrostatic: Coulombic potential from partial charges, normalized to [-1, 1]
"""

import argparse
import gzip
import io
import json
import math
import sys
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings('ignore')

KYTE_DOOLITTLE = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5, 'MET': 1.9, 'ALA': 1.8,
    'GLY': -0.4, 'THR': -0.7, 'SER': -0.8, 'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6,
    'HIS': -3.2, 'HSD': -3.2, 'HSE': -3.2, 'HSP': -3.2, 'MSE': 1.9,
    'GLU': -3.5, 'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5,
}

STANDARD_AA = set(KYTE_DOOLITTLE)
HALOGENS = {'F', 'CL', 'BR', 'I'}
METALS = {
    'LI', 'NA', 'K', 'RB', 'CS',
    'MG', 'CA', 'SR', 'BA',
    'ZN', 'FE', 'MN', 'CU', 'CO', 'NI',
}

BACKBONE_CHARGES = {
    'N': -0.4157, 'H': 0.2719, 'CA': 0.0337, 'HA': 0.0823,
    'C': 0.5973, 'O': -0.5679, 'OXT': -0.5679,
}
SIDECHAIN_CHARGES = {
    'ARG': {'CZ': 0.8281, 'NH1': -0.8693, 'NH2': -0.8693, 'NE': -0.5295, 'HE': 0.3456,
            'HH11': 0.4494, 'HH12': 0.4494, 'HH21': 0.4494, 'HH22': 0.4494},
    'LYS': {'NZ': -0.3854, 'HZ1': 0.3400, 'HZ2': 0.3400, 'HZ3': 0.3400, 'CE': -0.0143},
    'ASP': {'CG': 0.7994, 'OD1': -0.8014, 'OD2': -0.8014},
    'GLU': {'CD': 0.8054, 'OE1': -0.8188, 'OE2': -0.8188},
    'HIS': {'ND1': -0.3811, 'HD1': 0.3649, 'NE2': -0.5727, 'CE1': 0.2057, 'CD2': -0.2207},
    'HSD': {'ND1': -0.3811, 'HD1': 0.3649, 'NE2': -0.5727, 'CE1': 0.2057, 'CD2': -0.2207},
    'HSE': {'ND1': -0.3811, 'HD1': 0.3649, 'NE2': -0.5727, 'CE1': 0.2057, 'CD2': -0.2207},
    'HSP': {'ND1': -0.3811, 'HD1': 0.3649, 'NE2': -0.5727, 'CE1': 0.2057, 'CD2': -0.2207},
    'SER': {'OG': -0.6546, 'HG': 0.4275},
    'THR': {'OG1': -0.6761, 'HG1': 0.4102},
    'TYR': {'OH': -0.5579, 'HH': 0.3992},
    'CYS': {'SG': -0.3119, 'HG': 0.1933},
    'ASN': {'OD1': -0.5931, 'ND2': -0.9191, 'HD21': 0.4196, 'HD22': 0.4196},
    'GLN': {'OE1': -0.6086, 'NE2': -0.9407, 'HE21': 0.4251, 'HE22': 0.4251},
    'TRP': {'NE1': -0.3418, 'HE1': 0.3412},
}


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute per-atom surface properties')
    parser.add_argument('--pdb_path', required=True, help='Input structure or ligand file')
    parser.add_argument('--output_path', required=True, help='Output JSON path')
    args = parser.parse_args()

    try:
        import numpy as np

        coords, atom_names, residue_names, elements, charges, lipophilicity = load_atom_fields(args.pdb_path)
        atom_count = len(coords)
        if atom_count == 0:
            raise ValueError('No atoms found')

        hydrophobic = compute_smoothed_field(coords, lipophilicity, sigma=2.4, cutoff=7.0)
        electrostatic = compute_coulombic_potential(coords, charges, cutoff=12.0)

        result = {
            'atomCount': atom_count,
            'hydrophobic': normalize_signed(hydrophobic).round(4).astype(float).tolist(),
            'electrostatic': normalize_signed(electrostatic).round(4).astype(float).tolist(),
        }

        with open(args.output_path, 'w', encoding='utf-8') as handle:
            json.dump(result, handle)
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({'error': str(exc)}))
        sys.exit(1)


def load_atom_fields(file_path: str) -> Tuple[Any, List[str], List[str], List[str], Any, Any]:
    file_path_lower = file_path.lower()

    if file_path_lower.endswith(('.sdf', '.sdf.gz', '.mol', '.mol2')):
        return load_small_molecule(file_path)

    if file_path_lower.endswith('.pdb'):
        return load_pdb_like_structure(file_path, use_cif=False)

    if file_path_lower.endswith('.cif'):
        return load_pdb_like_structure(file_path, use_cif=True)

    raise ValueError(f'Unsupported surface-property input: {file_path}')


def load_small_molecule(file_path: str) -> Tuple[Any, List[str], List[str], List[str], Any, Any]:
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    mol = None
    lower = file_path.lower()

    if lower.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(file_path, removeHs=False, sanitize=True)
        mol = next((entry for entry in supplier if entry is not None), None)
    elif lower.endswith('.sdf.gz'):
        with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as handle:
            block = handle.read()
        supplier = Chem.ForwardSDMolSupplier(io.BytesIO(block.encode('utf-8')), removeHs=False, sanitize=True)
        mol = next((entry for entry in supplier if entry is not None), None)
    elif lower.endswith('.mol'):
        mol = Chem.MolFromMolFile(file_path, removeHs=False, sanitize=True)
    elif lower.endswith('.mol2'):
        mol = Chem.MolFromMol2File(file_path, removeHs=False, sanitize=True)

    if mol is None:
        raise ValueError(f'Failed to parse small-molecule file: {file_path}')

    if mol.GetNumConformers() == 0:
        raise ValueError(f'No 3D conformer found in: {file_path}')

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    conf = mol.GetConformer()
    coords = np.array(conf.GetPositions(), dtype=np.float32)
    atom_names: List[str] = []
    residue_names: List[str] = []
    elements: List[str] = []

    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        atom_names.append(info.GetName().strip() if info and info.GetName() else atom.GetSymbol())
        residue_names.append(info.GetResidueName().strip() if info and info.GetResidueName() else 'LIG')
        elements.append(atom.GetSymbol().upper())

    charges = np.array(compute_rdkit_charges(mol), dtype=np.float32)

    try:
        crippen = rdMolDescriptors._CalcCrippenContribs(mol)
        lipophilicity = np.array([float(logp) for logp, _mr in crippen], dtype=np.float32)
    except Exception:
        lipophilicity = np.array([
            generic_lipophilicity(elements[i], charges[i], atom_names[i], residue_names[i])
            for i in range(mol.GetNumAtoms())
        ], dtype=np.float32)

    lipophilicity -= np.minimum(np.abs(charges), 1.5) * 0.35
    return coords, atom_names, residue_names, elements, charges, lipophilicity


def load_pdb_like_structure(file_path: str, use_cif: bool) -> Tuple[Any, List[str], List[str], List[str], Any, Any]:
    import numpy as np
    from openmm.app import PDBFile, PDBxFile

    structure = PDBxFile(file_path) if use_cif else PDBFile(file_path)
    positions = structure.getPositions(asNumpy=True)
    coords = np.array(positions.value_in_unit(positions.unit) * 10.0, dtype=np.float32)

    atom_names: List[str] = []
    residue_names: List[str] = []
    elements: List[str] = []
    atom_keys: List[Tuple[str, str, str, str]] = []

    for atom in structure.topology.atoms():
        atom_names.append(atom.name)
        residue_names.append(atom.residue.name)
        elements.append((atom.element.symbol if atom.element else atom.name[:1]).upper())
        chain_id = getattr(atom.residue.chain, 'id', '')
        residue_id = str(atom.residue.id)
        atom_keys.append((chain_id, residue_id, atom.residue.name, atom.name))

    charges = get_pdb_like_charges(file_path, structure, atom_names, residue_names, elements, atom_keys)
    lipophilicity = np.array([
        protein_lipophilicity(atom_names[i], residue_names[i], elements[i], charges[i])
        for i in range(len(atom_names))
    ], dtype=np.float32)
    return coords, atom_names, residue_names, elements, charges, lipophilicity


def compute_rdkit_charges(mol: Any) -> List[float]:
    from rdkit.Chem import AllChem

    try:
        AllChem.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            raw = atom.GetProp('_GasteigerCharge') if atom.HasProp('_GasteigerCharge') else '0.0'
            try:
                charge = float(raw)
            except Exception:
                charge = 0.0
            if not math.isfinite(charge):
                charge = 0.0
            charges.append(charge)
        return charges
    except Exception:
        charges = []
        for atom in mol.GetAtoms():
            charge = float(atom.GetFormalCharge())
            if atom.GetSymbol().upper() == 'N' and charge == 0.0:
                charge = -0.15
            elif atom.GetSymbol().upper() == 'O' and charge == 0.0:
                charge = -0.25
            charges.append(charge)
        return charges


def get_pdb_like_charges(
    file_path: str,
    structure: Any,
    atom_names: Sequence[str],
    residue_names: Sequence[str],
    elements: Sequence[str],
    atom_keys: Sequence[Tuple[str, str, str, str]],
) -> Any:
    import numpy as np

    charges = np.full(len(atom_names), np.nan, dtype=np.float32)

    raw = charges_via_openmm_raw(structure)
    if raw is not None and len(raw) == len(atom_names):
        charges[:] = raw
    else:
        fixed = charges_via_pdbfixer(file_path, atom_keys)
        if fixed is not None:
            fixed_mask = ~np.isnan(fixed)
            charges[fixed_mask] = fixed[fixed_mask]

    lookup = charges_via_lookup(atom_names, residue_names, elements)
    generic = generic_element_charges(atom_names, residue_names, elements)

    missing = np.isnan(charges)
    if np.any(missing):
        charges[missing] = lookup[missing]
        still_missing = np.isnan(charges)
        if np.any(still_missing):
            charges[still_missing] = generic[still_missing]

    # Fill remaining standard-residue zeros from generic atom heuristics.
    zero_mask = np.abs(charges) < 1e-6
    charges[zero_mask] = generic[zero_mask]
    return charges.astype(np.float32)


def charges_via_openmm_raw(structure: Any) -> Optional[Any]:
    import numpy as np
    try:
        from openmm import NonbondedForce, unit
        from openmm.app import ForceField, NoCutoff

        ff = ForceField('amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml')
        system = ff.createSystem(structure.topology, nonbondedMethod=NoCutoff)

        charges = np.zeros(sum(1 for _ in structure.topology.atoms()), dtype=np.float32)
        for force in system.getForces():
            if isinstance(force, NonbondedForce):
                for idx in range(force.getNumParticles()):
                    charges[idx] = force.getParticleParameters(idx)[0].value_in_unit(unit.elementary_charge)
                return charges
    except Exception:
        return None
    return None


def charges_via_pdbfixer(file_path: str, atom_keys: Sequence[Tuple[str, str, str, str]]) -> Optional[Any]:
    import numpy as np
    try:
        from pdbfixer import PDBFixer
        from openmm import NonbondedForce, unit
        from openmm.app import ForceField, NoCutoff

        fixer = PDBFixer(filename=file_path)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.4)

        ff = ForceField('amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml')
        system = ff.createSystem(fixer.topology, nonbondedMethod=NoCutoff)

        fixed_charges = []
        for force in system.getForces():
            if isinstance(force, NonbondedForce):
                for idx in range(force.getNumParticles()):
                    fixed_charges.append(force.getParticleParameters(idx)[0].value_in_unit(unit.elementary_charge))
                break
        else:
            return None

        charge_by_key: Dict[Tuple[str, str, str, str], List[float]] = {}
        for atom, charge in zip(fixer.topology.atoms(), fixed_charges):
            chain_id = getattr(atom.residue.chain, 'id', '')
            residue_id = str(atom.residue.id)
            key = (chain_id, residue_id, atom.residue.name, atom.name)
            charge_by_key.setdefault(key, []).append(float(charge))

        mapped = np.full(len(atom_keys), np.nan, dtype=np.float32)
        seen_counts: Dict[Tuple[str, str, str, str], int] = {}
        for idx, key in enumerate(atom_keys):
            values = charge_by_key.get(key)
            if not values:
                continue
            use_idx = seen_counts.get(key, 0)
            if use_idx < len(values):
                mapped[idx] = values[use_idx]
                seen_counts[key] = use_idx + 1
        return mapped
    except Exception:
        return None


def charges_via_lookup(atom_names: Sequence[str], residue_names: Sequence[str], elements: Sequence[str]) -> Any:
    import numpy as np

    charges = np.zeros(len(atom_names), dtype=np.float32)
    for idx, (atom_name, residue_name, element) in enumerate(zip(atom_names, residue_names, elements)):
        if atom_name in BACKBONE_CHARGES:
            charges[idx] = BACKBONE_CHARGES[atom_name]
            continue

        residue_table = SIDECHAIN_CHARGES.get(residue_name)
        if residue_table and atom_name in residue_table:
            charges[idx] = residue_table[atom_name]
            continue

        if element in METALS:
            charges[idx] = 2.0 if element in {'MG', 'CA', 'ZN', 'FE', 'MN'} else 1.0
        elif residue_name in {'NA', 'K'}:
            charges[idx] = 1.0
        elif residue_name in {'CL'}:
            charges[idx] = -1.0
    return charges


def generic_element_charges(atom_names: Sequence[str], residue_names: Sequence[str], elements: Sequence[str]) -> Any:
    import numpy as np

    charges = np.zeros(len(atom_names), dtype=np.float32)
    for idx, (atom_name, residue_name, element) in enumerate(zip(atom_names, residue_names, elements)):
        if element == 'O':
            charges[idx] = -0.35
        elif element == 'N':
            charges[idx] = -0.20
        elif element == 'S':
            charges[idx] = -0.10
        elif element == 'P':
            charges[idx] = 0.40
        elif element in HALOGENS:
            charges[idx] = -0.05
        elif element in METALS:
            charges[idx] = 2.0 if element in {'MG', 'CA', 'ZN', 'FE', 'MN'} else 1.0

        if residue_name in {'LYS', 'ARG'} and atom_name.startswith(('N', 'NZ', 'NH', 'NE')):
            charges[idx] = max(charges[idx], 0.35)
        elif residue_name in {'ASP', 'GLU'} and atom_name.startswith(('O', 'OD', 'OE')):
            charges[idx] = min(charges[idx], -0.45)
        elif residue_name in {'NA', 'K'}:
            charges[idx] = 1.0
        elif residue_name in {'CL'}:
            charges[idx] = -1.0
    return charges


def protein_lipophilicity(atom_name: str, residue_name: str, element: str, charge: float) -> float:
    residue_score = max(-1.0, min(1.0, KYTE_DOOLITTLE.get(residue_name, 0.0) / 4.5))

    if residue_name in STANDARD_AA:
        if atom_name in {'N', 'H', 'HN', 'H1', 'H2', 'H3', 'C', 'O', 'OXT'}:
            base = -0.55 if element in {'N', 'O'} else -0.10
        elif atom_name in {'CA', 'HA', 'HA2', 'HA3'}:
            base = 0.05
        elif element == 'C':
            base = 0.30 + 0.70 * residue_score
        elif element == 'S':
            base = 0.20 + 0.45 * residue_score
        elif element in {'N', 'O', 'P'}:
            base = -0.70
        elif element in HALOGENS:
            base = 0.55
        else:
            base = 0.0
    else:
        base = generic_lipophilicity(element, charge, atom_name, residue_name)

    base -= min(abs(charge), 1.5) * 0.45
    return float(max(-2.0, min(2.0, base)))


def generic_lipophilicity(element: str, charge: float, atom_name: str, residue_name: str) -> float:
    if element == 'C':
        base = 0.65
    elif element == 'S':
        base = 0.45
    elif element in HALOGENS:
        base = 0.55
    elif element in {'N', 'O', 'P'}:
        base = -0.65
    elif element == 'H':
        base = 0.0
    elif element in METALS:
        base = -0.80
    else:
        base = 0.0

    if residue_name in {'NA', 'K', 'CL'}:
        base = -1.0
    if atom_name.startswith(('NZ', 'NH', 'NE', 'OD', 'OE')):
        base -= 0.20

    base -= min(abs(charge), 1.5) * 0.45
    return float(max(-2.0, min(2.0, base)))


def compute_smoothed_field(coords: Any, atom_values: Any, sigma: float, cutoff: float) -> Any:
    import numpy as np
    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    neighborhoods = tree.query_ball_point(coords, cutoff)
    denom_scale = 2.0 * sigma * sigma
    field = np.zeros(len(coords), dtype=np.float32)

    for idx, neighbors in enumerate(neighborhoods):
        if not neighbors:
            field[idx] = atom_values[idx]
            continue
        points = coords[np.array(neighbors, dtype=np.intp)]
        deltas = points - coords[idx]
        distances_sq = np.sum(deltas * deltas, axis=1)
        weights = np.exp(-distances_sq / denom_scale)
        weight_sum = float(weights.sum())
        if weight_sum > 1e-8:
            field[idx] = float(np.dot(weights, atom_values[np.array(neighbors, dtype=np.intp)]) / weight_sum)
        else:
            field[idx] = atom_values[idx]
    return field


def compute_coulombic_potential(coords: Any, charges: Any, cutoff: float) -> Any:
    import numpy as np
    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    neighborhoods = tree.query_ball_point(coords, cutoff)
    potential = np.zeros(len(coords), dtype=np.float32)

    for idx, neighbors in enumerate(neighborhoods):
        if not neighbors:
            continue
        nbr_idx = np.array(neighbors, dtype=np.intp)
        nbr_idx = nbr_idx[nbr_idx != idx]
        if len(nbr_idx) == 0:
            continue
        deltas = coords[nbr_idx] - coords[idx]
        distances = np.sqrt(np.sum(deltas * deltas, axis=1))
        distances = np.maximum(distances, 0.6)
        potential[idx] = float(np.sum(charges[nbr_idx] / (4.0 * distances)))
    return potential


def normalize_signed(values: Any) -> Any:
    import numpy as np

    if len(values) == 0:
        return values

    v = np.array(values, dtype=np.float32)
    lo = float(np.percentile(v, 5))
    hi = float(np.percentile(v, 95))

    if hi - lo < 1e-8:
        max_abs = float(np.max(np.abs(v)))
        if max_abs < 1e-8:
            return np.zeros_like(v)
        return np.clip(v / max_abs, -1.0, 1.0)

    v = np.clip(v, lo, hi)
    return ((v - lo) / (hi - lo) * 2.0 - 1.0).astype(np.float32)


if __name__ == '__main__':
    main()
