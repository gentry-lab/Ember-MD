#!/usr/bin/env python3
"""
Compute per-atom surface properties (hydrophobic, electrostatic) for a protein PDB.

Hydrophobic: Gaussian-smoothed Kyte-Doolittle per-atom values.
Electrostatic: Coulombic potential from AMBER partial charges with distance-dependent
dielectric (epsilon = 4*r), computed at each atom position. Uses PDBFixer to clean
the structure so OpenMM charge assignment succeeds on real-world PDBs.
"""

import argparse
import json
import sys
import warnings
from typing import Any, List, Optional

warnings.filterwarnings('ignore')

KYTE_DOOLITTLE = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5, 'MET': 1.9, 'ALA': 1.8,
    'GLY': -0.4, 'THR': -0.7, 'SER': -0.8, 'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6,
    'HIS': -3.2, 'HSD': -3.2, 'HSE': -3.2, 'HSP': -3.2, 'MSE': 1.9,
    'GLU': -3.5, 'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5,
}

# Per-atom partial charges for standard amino acid atoms (AMBER ff14SB subset)
# Only used as last-resort fallback if PDBFixer + OpenMM both fail
BACKBONE_CHARGES = {'N': -0.4157, 'H': 0.2719, 'CA': 0.0337, 'HA': 0.0823,
                    'C': 0.5973, 'O': -0.5679}
SIDECHAIN_CHARGES = {
    'ARG': {'CZ': 0.8281, 'NH1': -0.8693, 'NH2': -0.8693, 'NE': -0.5295, 'HE': 0.3456,
            'HH11': 0.4494, 'HH12': 0.4494, 'HH21': 0.4494, 'HH22': 0.4494},
    'LYS': {'NZ': -0.3854, 'HZ1': 0.3400, 'HZ2': 0.3400, 'HZ3': 0.3400, 'CE': -0.0143},
    'ASP': {'CG': 0.7994, 'OD1': -0.8014, 'OD2': -0.8014},
    'GLU': {'CD': 0.8054, 'OE1': -0.8188, 'OE2': -0.8188},
    'HIS': {'ND1': -0.3811, 'HD1': 0.3649, 'NE2': -0.5727, 'CE1': 0.2057, 'CD2': -0.2207},
    'SER': {'OG': -0.6546, 'HG': 0.4275},
    'THR': {'OG1': -0.6761, 'HG1': 0.4102},
    'TYR': {'OH': -0.5579, 'HH': 0.3992},
    'CYS': {'SG': -0.3119, 'HG': 0.1933},
    'ASN': {'OD1': -0.5931, 'ND2': -0.9191, 'HD21': 0.4196, 'HD22': 0.4196},
    'GLN': {'OE1': -0.6086, 'NE2': -0.9407, 'HE21': 0.4251, 'HE22': 0.4251},
    'TRP': {'NE1': -0.3418, 'HE1': 0.3412},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compute per-atom surface properties for a protein PDB'
    )
    parser.add_argument('--pdb_path', required=True, help='Input PDB file')
    parser.add_argument('--output_path', required=True, help='Output JSON file path')
    args = parser.parse_args()

    try:
        import numpy as np
        from scipy.spatial import cKDTree
    except ImportError as e:
        print(json.dumps({"error": f"Missing required package: {e}"}))
        sys.exit(1)

    try:
        # ============================================================
        # Step 1: Parse PDB and extract atom coordinates
        # ============================================================
        print("PROGRESS:parsing_pdb", file=sys.stderr)

        from openmm.app import PDBFile
        pdb = PDBFile(args.pdb_path)
        positions = pdb.getPositions(asNumpy=True)
        atom_coords = (positions.value_in_unit(positions.unit) * 10.0).astype(np.float32)
        n_atoms = len(atom_coords)

        # Build per-atom info
        atom_names = []
        residue_names = []
        ca_indices = []
        ca_residues = []

        for atom in pdb.topology.atoms():
            atom_names.append(atom.name)
            residue_names.append(atom.residue.name)
            if atom.name == 'CA':
                ca_indices.append(atom.index)
                ca_residues.append(atom.residue.name)

        ca_indices = np.array(ca_indices, dtype=np.intp)
        ca_coords = atom_coords[ca_indices]
        print(f"  Atoms: {n_atoms}, CA atoms: {len(ca_indices)}", file=sys.stderr)

        if len(ca_indices) == 0:
            print(json.dumps({"error": "No CA atoms found — not a protein structure"}))
            sys.exit(1)

        # ============================================================
        # Step 2: Hydrophobic (Kyte-Doolittle, Gaussian-smoothed)
        # ============================================================
        print("PROGRESS:computing_hydrophobic", file=sys.stderr)

        kd_ca = np.array([KYTE_DOOLITTLE.get(r, 0.0) for r in ca_residues], dtype=np.float32)
        tree_ca = cKDTree(ca_coords)
        sigma_h = 5.0
        cutoff_h = 12.0
        two_sigma_sq_h = 2.0 * sigma_h * sigma_h

        hydrophobic = np.zeros(n_atoms, dtype=np.float32)
        nbrs_h = tree_ca.query_ball_point(atom_coords, cutoff_h)
        for i in range(n_atoms):
            idx = nbrs_h[i]
            if not idx:
                continue
            idx = np.array(idx, dtype=np.intp)
            d2 = np.sum((ca_coords[idx] - atom_coords[i]) ** 2, axis=1)
            w = np.exp(-d2 / two_sigma_sq_h)
            ws = w.sum()
            if ws > 1e-12:
                hydrophobic[i] = np.dot(w, kd_ca[idx]) / ws

        # Normalize to [-1, 1]
        hmin, hmax = hydrophobic.min(), hydrophobic.max()
        if hmax - hmin > 1e-12:
            hydrophobic = (hydrophobic - hmin) / (hmax - hmin) * 2.0 - 1.0

        # ============================================================
        # Step 3: Electrostatic (Coulombic with distance-dependent dielectric)
        # ============================================================
        print("PROGRESS:computing_electrostatic", file=sys.stderr)

        charges = _get_charges(args.pdb_path, pdb, atom_coords, atom_names, residue_names, n_atoms)

        # Coulombic potential with distance-dependent dielectric: phi_i = sum q_j / (4 * r_ij)
        # The factor 4*r is the Skolnick distance-dependent dielectric (epsilon = 4r)
        # This gives smooth gradients, not sharp per-residue blocks
        print("  Computing Coulombic potential (epsilon=4r)...", file=sys.stderr)
        tree_all = cKDTree(atom_coords)
        cutoff_e = 12.0
        electrostatic = np.zeros(n_atoms, dtype=np.float32)
        nbrs_e = tree_all.query_ball_point(atom_coords, cutoff_e)

        for i in range(n_atoms):
            idx = nbrs_e[i]
            if not idx:
                continue
            idx = np.array(idx, dtype=np.intp)
            mask = idx != i
            idx = idx[mask]
            if len(idx) == 0:
                continue
            d = np.sqrt(np.sum((atom_coords[idx] - atom_coords[i]) ** 2, axis=1))
            d = np.maximum(d, 0.5)  # floor to avoid singularity
            # phi = sum(q / (4*r)) — distance-dependent dielectric
            electrostatic[i] = np.sum(charges[idx] / (4.0 * d))

        # Normalize with percentile clamping to handle outliers
        p5, p95 = np.percentile(electrostatic, 5), np.percentile(electrostatic, 95)
        if p95 - p5 > 1e-12:
            electrostatic = np.clip(electrostatic, p5, p95)
            electrostatic = (electrostatic - p5) / (p95 - p5) * 2.0 - 1.0
        else:
            electrostatic = np.zeros(n_atoms, dtype=np.float32)

        # ============================================================
        # Step 4: Write output
        # ============================================================
        print("PROGRESS:writing_output", file=sys.stderr)

        result = {
            "atomCount": n_atoms,
            "hydrophobic": [round(float(v), 4) for v in hydrophobic],
            "electrostatic": [round(float(v), 4) for v in electrostatic],
        }

        with open(args.output_path, 'w') as f:
            json.dump(result, f)

        print(f"  Written to: {args.output_path}", file=sys.stderr)
        print("PROGRESS:done", file=sys.stderr)

    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


def _get_charges(pdb_path: str, pdb_obj: Any, atom_coords: Any, atom_names: List[str], residue_names: List[str], n_atoms: int) -> Any:
    """Get per-atom partial charges. Tries three methods in order:
    1. PDBFixer → OpenMM ff14SB (best — works on most real PDBs)
    2. Raw PDB → OpenMM ff14SB (fails if PDB has missing atoms/residues)
    3. Per-atom AMBER charge lookup table (last resort)
    """
    import numpy as np

    # Method 1: PDBFixer + OpenMM
    charges = _charges_via_pdbfixer(pdb_path, n_atoms)
    if charges is not None:
        print("  Charges: PDBFixer + OpenMM ff14SB", file=sys.stderr)
        return charges

    # Method 2: Raw PDB + OpenMM (no fixer)
    charges = _charges_via_openmm_raw(pdb_obj, n_atoms)
    if charges is not None:
        print("  Charges: Raw OpenMM ff14SB", file=sys.stderr)
        return charges

    # Method 3: Per-atom lookup table
    print("  Charges: AMBER atom-level lookup (fallback)", file=sys.stderr)
    return _charges_via_lookup(atom_names, residue_names, n_atoms)


def _charges_via_pdbfixer(pdb_path: str, n_atoms: int) -> Optional[Any]:
    """Use PDBFixer to clean PDB, then assign charges via OpenMM."""
    import numpy as np
    try:
        from pdbfixer import PDBFixer
        from openmm.app import ForceField, NoCutoff
        from openmm import NonbondedForce, unit

        fixer = PDBFixer(filename=pdb_path)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.4)

        ff = ForceField('amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml')
        system = ff.createSystem(fixer.topology, nonbondedMethod=NoCutoff)

        # PDBFixer may add atoms — extract charges for original atom count
        fixed_n = sum(1 for _ in fixer.topology.atoms())
        charges = np.zeros(fixed_n, dtype=np.float32)

        for force in system.getForces():
            if isinstance(force, NonbondedForce):
                for i in range(force.getNumParticles()):
                    charges[i] = force.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                break
        else:
            return None

        # If fixer added atoms, truncate to original count
        # The original atoms come first in PDBFixer output
        return charges[:n_atoms] if len(charges) >= n_atoms else None

    except Exception as e:
        print(f"  PDBFixer method failed: {e}", file=sys.stderr)
        return None


def _charges_via_openmm_raw(pdb_obj: Any, n_atoms: int) -> Optional[Any]:
    """Try raw PDB with OpenMM (no PDBFixer cleanup)."""
    import numpy as np
    try:
        from openmm.app import ForceField, NoCutoff
        from openmm import NonbondedForce, unit

        ff = ForceField('amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml')
        system = ff.createSystem(pdb_obj.topology, nonbondedMethod=NoCutoff)

        charges = np.zeros(n_atoms, dtype=np.float32)
        for force in system.getForces():
            if isinstance(force, NonbondedForce):
                for i in range(min(force.getNumParticles(), n_atoms)):
                    charges[i] = force.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                return charges

        return None
    except Exception as e:
        print(f"  Raw OpenMM method failed: {e}", file=sys.stderr)
        return None


def _charges_via_lookup(atom_names: List[str], residue_names: List[str], n_atoms: int) -> Any:
    """Last resort: use hardcoded AMBER partial charge tables for key atoms."""
    import numpy as np
    charges = np.zeros(n_atoms, dtype=np.float32)
    for i in range(n_atoms):
        aname = atom_names[i]
        rname = residue_names[i]
        # Check backbone
        if aname in BACKBONE_CHARGES:
            charges[i] = BACKBONE_CHARGES[aname]
        # Check sidechain
        elif rname in SIDECHAIN_CHARGES and aname in SIDECHAIN_CHARGES[rname]:
            charges[i] = SIDECHAIN_CHARGES[rname][aname]
        # Default: 0 (nonpolar carbon, etc.)
    return charges


if __name__ == '__main__':
    main()
