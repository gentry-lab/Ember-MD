#!/usr/bin/env python3
"""
Single-ligand AutoDock Vina docking via Python API.

Cross-platform replacement for GNINA. Uses Meeko for PDBQT preparation
and export (preserves SMILES + atom mapping in PDBQT REMARK lines, so
bond orders survive the round-trip without coordinate-based heuristics).
Output format matches the GNINA pipeline so the Electron frontend can
parse it unchanged.

Usage:
    python run_vina_docking.py \
        --receptor receptor.pdb \
        --ligand ligand.sdf \
        --reference reference_ligand.sdf \
        --output_dir /path/to/output \
        [--exhaustiveness 8] [--num_poses 9] [--autobox_add 4]
"""

import argparse
import gzip
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, List, Tuple

try:
    from vina import Vina
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, QED
    from meeko import (MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule,
                       RDKitMolCreate, Polymer, ResidueChemTemplates)
except ImportError as e:
    print(f"ERROR:Missing dependency: {e}", file=sys.stderr)
    print("Please install: conda install -c conda-forge vina rdkit meeko", file=sys.stderr)
    sys.exit(1)


def sdf_to_pdbqt_string(sdf_path: str, mol: Any = None) -> str:
    """Convert SDF to PDBQT string using Meeko.

    Meeko embeds the original SMILES and atom index mapping in REMARK lines,
    allowing correct bond order reconstruction on export. Accepts an optional
    pre-loaded RDKit mol to avoid re-reading the file.
    """
    if mol is None:
        if sdf_path.endswith('.gz'):
            with gzip.open(sdf_path, 'rt') as f:
                mol = Chem.MolFromMolBlock(f.read(), removeHs=False)
        else:
            supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
            mol = supplier[0]

    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Failed to read molecule from {sdf_path}")

    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)
    pdbqt_string, is_ok, err_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
    if not is_ok:
        raise ValueError(f"Meeko PDBQT preparation failed: {err_msg}")

    return pdbqt_string


def _extract_metal_pdbqt_lines(pdb_path: str) -> str:
    """Extract metal ions from PDB and format as PDBQT ATOM lines.

    Vina supports AD4 atom types for: Zn, Mg, Mn, Ca, Fe.
    Returns PDBQT-formatted ATOM lines for any metals found.
    """
    # Map PDB residue names to AD4 atom types and formal charges
    METAL_AD4 = {
        'ZN': ('Zn', 2.0), 'ZN2': ('Zn', 2.0),
        'MG': ('Mg', 2.0), 'MG2': ('Mg', 2.0),
        'MN': ('Mn', 2.0), 'MN2': ('Mn', 2.0),
        'CA': ('Ca', 2.0), 'CA2': ('Ca', 2.0),
        'FE': ('Fe', 2.0), 'FE2': ('Fe', 2.0), 'FE3': ('Fe', 3.0),
    }

    lines = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('HETATM'):
                resname = line[17:20].strip().upper()
                if resname in METAL_AD4:
                    ad4_type, charge = METAL_AD4[resname]
                    # Reformat as PDBQT: keep coords, set charge and AD4 type
                    atom_line = line[:70].ljust(70)
                    atom_line += f"{charge:>6.3f} {ad4_type:<2s}"
                    lines.append(atom_line.rstrip() + '\n')
    return ''.join(lines)


def pdb_to_pdbqt_string(pdb_path: str) -> str:
    """Convert receptor PDB to rigid PDBQT string using Meeko's Polymer.

    Meeko's Polymer parser handles standard amino acids, nucleotides, water, and
    common cofactors via ResidueChemTemplates. Non-standard residues are skipped
    with allow_bad_res=True (same behavior as obabel -xr ignoring HETATM).
    Ions and other residues that Meeko recognizes but cannot assign AD4 atom
    types to (atom_type=None) are marked is_ignore so the writer skips them.
    Metal ions are re-injected after Meeko processing with correct AD4 types
    (Vina natively supports Zn, Mg, Mn, Ca, Fe).
    Returns a PDBQT string (no ROOT/BRANCH torsion tree — rigid receptor).
    """
    import warnings

    with open(pdb_path) as f:
        pdb_string = f.read()

    templates = ResidueChemTemplates.create_from_defaults()
    mk_prep = MoleculePreparation()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        polymer = Polymer.from_pdb_string(pdb_string, templates, mk_prep,
                                          allow_bad_res=True)

    # Skip atoms without AD4 atom types (Meeko can't type metals/unknowns)
    for _res_id, monomer in polymer.get_valid_monomers().items():
        for atom in monomer.molsetup.atoms:
            if atom.atom_type is None:
                atom.is_ignore = True

    rigid_pdbqt, _flex_dict = PDBQTWriterLegacy.write_from_polymer(polymer)

    if not rigid_pdbqt or 'ATOM' not in rigid_pdbqt:
        raise ValueError(f"Meeko Polymer produced empty PDBQT for {pdb_path}")

    # Inject metal ions with correct AD4 types (Meeko can't type them)
    metal_lines = _extract_metal_pdbqt_lines(pdb_path)
    if metal_lines:
        # Insert before END record
        if rigid_pdbqt.rstrip().endswith('END'):
            rigid_pdbqt = rigid_pdbqt.rstrip()[:-3] + metal_lines + 'END\n'
        else:
            rigid_pdbqt += metal_lines

    return rigid_pdbqt


def get_box_from_reference(reference_path: str, autobox_add: float = 4.0) -> Tuple[List[float], List[float]]:
    """Compute docking box center and size from a reference ligand."""
    if reference_path.endswith('.gz'):
        with gzip.open(reference_path, 'rt') as f:
            sdf_content = f.read()
        mol = Chem.MolFromMolBlock(sdf_content, removeHs=False)
    elif reference_path.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(reference_path, removeHs=False)
        mol = supplier[0]
    elif reference_path.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(reference_path, removeHs=False)
    else:
        raise ValueError(f"Unsupported reference format: {reference_path}")

    if mol is None:
        raise ValueError(f"Failed to read reference ligand from {reference_path}")

    conf = mol.GetConformer()
    coords = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]

    import numpy as np
    coords = np.array(coords)
    center = coords.mean(axis=0)
    span = coords.max(axis=0) - coords.min(axis=0)
    size = span + 2 * autobox_add  # Add padding on each side

    return center.tolist(), size.tolist()


def pdbqt_poses_to_sdf(pdbqt_path: str, output_sdf_path: str, scores: List[float]) -> None:
    """Convert Vina output PDBQT poses back to SDF with score properties.

    Uses Meeko to reconstruct RDKit molecules from the docked PDBQT. Meeko
    reads the SMILES + atom index mapping it embedded in REMARK lines during
    ligand preparation, so bond orders and implicit hydrogens are always correct.
    """
    pdbqt_mol = PDBQTMolecule.from_file(pdbqt_path, skip_typing=True)
    rdkit_results = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)

    writer = Chem.SDWriter(output_sdf_path)

    for i, score in enumerate(scores):
        if i >= len(rdkit_results):
            break

        rd_mol = rdkit_results[i]
        if rd_mol is None:
            print(f"WARNING: Pose {i} failed Meeko export, skipping", file=sys.stderr)
            continue

        rd_mol.SetProp("minimizedAffinity", f"{score:.2f}")
        rd_mol.SetProp("pose_index", str(i))
        writer.write(rd_mol)

    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Dock single ligand with AutoDock Vina')
    parser.add_argument('--receptor', required=True, help='Path to receptor PDB file')
    parser.add_argument('--ligand', required=True, help='Path to single ligand SDF file')
    parser.add_argument('--reference', required=True, help='Path to reference ligand for autobox')
    parser.add_argument('--output_dir', required=True, help='Output directory for docked pose')
    parser.add_argument('--exhaustiveness', type=int, default=8, help='Search exhaustiveness')
    parser.add_argument('--num_poses', type=int, default=9, help='Number of poses to generate')
    parser.add_argument('--autobox_add', type=float, default=4.0, help='Autobox margin in Angstroms')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (0=random)')
    parser.add_argument('--cpu', type=int, default=0, help='Number of CPUs (0=auto)')
    parser.add_argument('--core_constrain', action='store_true', help='MCS core-constrained alignment')
    parser.add_argument('--reference_sdf', default=None, help='Reference SDF for MCS alignment')
    parser.add_argument('--project_name', default=None, help='Project name prefix for output files')
    args = parser.parse_args()

    # Validate inputs
    for path, label in [(args.receptor, 'Receptor'), (args.ligand, 'Ligand'), (args.reference, 'Reference')]:
        if not os.path.exists(path):
            print(f'ERROR: {label} file not found: {path}', file=sys.stderr)
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    raw_name = Path(args.ligand).stem
    name = f'{args.project_name}_{raw_name}' if args.project_name else raw_name
    t_start = time.time()

    with tempfile.TemporaryDirectory(prefix='vina_') as tmp_dir:
        # 1. Convert receptor PDB → PDBQT via Meeko Polymer
        print(f'Preparing receptor...', file=sys.stderr)
        receptor_pdbqt_str = pdb_to_pdbqt_string(args.receptor)
        receptor_pdbqt = os.path.join(tmp_dir, 'receptor.pdbqt')
        with open(receptor_pdbqt, 'w') as f:
            f.write(receptor_pdbqt_str)

        # 2. Convert ligand SDF → PDBQT string via Meeko
        print(f'Preparing ligand {name}...', file=sys.stderr)
        ligand_pdbqt_string = sdf_to_pdbqt_string(args.ligand)

        # 3. Compute box from reference ligand
        center, size = get_box_from_reference(args.reference, args.autobox_add)
        print(f'Box center: [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}], '
              f'size: [{size[0]:.1f}, {size[1]:.1f}, {size[2]:.1f}]', file=sys.stderr)

        # 4. Run Vina
        v = Vina(sf_name='vina', cpu=args.cpu if args.cpu > 0 else 0,
                 seed=args.seed if args.seed > 0 else 0)
        v.set_receptor(receptor_pdbqt)
        v.set_ligand_from_string(ligand_pdbqt_string)
        v.compute_vina_maps(center=center, box_size=size)

        print(f'Docking (exhaustiveness={args.exhaustiveness})...', file=sys.stderr)
        v.dock(exhaustiveness=args.exhaustiveness, n_poses=args.num_poses)

        # 5. Get scores
        energies = v.energies()  # shape: (n_poses, 5) — [total, inter, intra, torsion, intra_best]
        scores = [e[0] for e in energies]  # total affinity (kcal/mol)

        # 6. Write output PDBQT
        output_pdbqt = os.path.join(tmp_dir, f'{name}_docked.pdbqt')
        v.write_poses(output_pdbqt, n_poses=args.num_poses, overwrite=True)

        # 7. Convert docked PDBQT → SDF with scores (Meeko reads its own REMARK lines)
        out_sdf = os.path.join(args.output_dir, f'{name}_docked.sdf.gz')
        temp_sdf = os.path.join(tmp_dir, f'{name}_docked.sdf')

        pdbqt_poses_to_sdf(output_pdbqt, temp_sdf, scores)

        # MCS core-constrained alignment
        if args.core_constrain and args.reference_sdf:
            try:
                from rdkit.Chem import rdFMCS
                ref_path = args.reference_sdf
                if ref_path.endswith('.gz'):
                    with gzip.open(ref_path, 'rt') as f:
                        ref_mol = Chem.MolFromMolBlock(f.read(), removeHs=False)
                elif ref_path.endswith('.sdf'):
                    supplier = Chem.SDMolSupplier(ref_path, removeHs=False)
                    ref_mol = supplier[0]
                elif ref_path.endswith('.pdb'):
                    ref_mol = Chem.MolFromPDBFile(ref_path, removeHs=False)
                else:
                    ref_mol = None

                if ref_mol is not None:
                    # Read poses and align to reference core
                    aligned_poses = []
                    supplier = Chem.SDMolSupplier(temp_sdf, removeHs=False)
                    for pose_mol in supplier:
                        if pose_mol is None:
                            continue
                        try:
                            mcs = rdFMCS.FindMCS(
                                [ref_mol, pose_mol],
                                ringMatchesRingOnly=True,
                                timeout=10
                            )
                            if mcs.numAtoms >= 3:
                                core_smarts = Chem.MolFromSmarts(mcs.smartsString)
                                ref_match = ref_mol.GetSubstructMatch(core_smarts)
                                pose_match = pose_mol.GetSubstructMatch(core_smarts)
                                if ref_match and pose_match:
                                    atom_map = list(zip(pose_match, ref_match))
                                    rmsd = AllChem.AlignMol(pose_mol, ref_mol, atomMap=atom_map)
                                    pose_mol.SetProp("coreRMSD", f"{rmsd:.3f}")
                        except Exception as e:
                            print(f'  MCS alignment warning: {e}', file=sys.stderr)
                        aligned_poses.append(pose_mol)

                    # Rewrite SDF with aligned poses
                    if aligned_poses:
                        writer = Chem.SDWriter(temp_sdf)
                        for mol in aligned_poses:
                            writer.write(mol)
                        writer.close()
                        print(f'  Core-constrained alignment: {len(aligned_poses)} poses aligned', file=sys.stderr)
                else:
                    print(f'  Warning: Could not read reference for MCS alignment', file=sys.stderr)
            except ImportError:
                print(f'  Warning: rdFMCS not available, skipping core alignment', file=sys.stderr)

        # Gzip the output SDF
        with open(temp_sdf, 'rb') as f_in:
            with gzip.open(out_sdf, 'wb') as f_out:
                f_out.write(f_in.read())

    elapsed = time.time() - t_start
    print(f'Docking complete in {elapsed:.1f}s: {len(scores)} poses', file=sys.stderr)
    print(f'  Best affinity: {scores[0]:.2f} kcal/mol', file=sys.stderr)
    for i, s in enumerate(scores):
        print(f'  Pose {i+1}: {s:.2f} kcal/mol', file=sys.stderr)

    # Output format parsed by Node.js (same as GNINA)
    print(f'SUCCESS:{name}:{out_sdf}')


if __name__ == '__main__':
    main()
