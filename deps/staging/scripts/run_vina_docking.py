#!/usr/bin/env python3
"""
Single-ligand AutoDock Vina docking via Python API.

Cross-platform replacement for GNINA. Uses Meeko for PDBQT preparation
and the vina Python package for docking. Output format matches the
GNINA pipeline so the Electron frontend can parse it unchanged.

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

try:
    from vina import Vina
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, QED
    from openbabel import openbabel
except ImportError as e:
    print(f"ERROR:Missing dependency: {e}", file=sys.stderr)
    print("Please install: conda install -c conda-forge vina rdkit openbabel", file=sys.stderr)
    sys.exit(1)


def sdf_to_pdbqt(sdf_path, pdbqt_path):
    """Convert SDF to PDBQT using OpenBabel (handles bond orders, charges, torsions)."""
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("sdf", "pdbqt")
    # Add hydrogens and compute Gasteiger charges
    obConversion.AddOption("h", openbabel.OBConversion.GENOPTIONS)

    mol = openbabel.OBMol()
    if sdf_path.endswith('.gz'):
        with gzip.open(sdf_path, 'rt') as f:
            sdf_content = f.read()
        obConversion.ReadString(mol, sdf_content)
    else:
        obConversion.ReadFile(mol, sdf_path)

    if mol.NumAtoms() == 0:
        raise ValueError(f"Failed to read molecule from {sdf_path}")

    obConversion.WriteFile(mol, pdbqt_path)
    return pdbqt_path


def pdb_to_pdbqt(pdb_path, pdbqt_path):
    """Convert receptor PDB to rigid PDBQT using obabel CLI.

    The Python API's -r flag doesn't produce rigid PDBQT (still writes ROOT/BRANCH
    torsion tree), causing Vina to reject the file. The obabel CLI with -xr correctly
    outputs flat rigid PDBQT with no torsion tree.
    """
    import shutil
    import subprocess

    obabel_bin = shutil.which('obabel')
    if not obabel_bin:
        # Fallback: look in the conda env bin directory
        env_bin = os.path.dirname(sys.executable)
        obabel_bin = os.path.join(env_bin, 'obabel')
        if not os.path.exists(obabel_bin):
            raise RuntimeError("obabel not found in PATH or conda env")

    result = subprocess.run(
        [obabel_bin, pdb_path, '-O', pdbqt_path, '-xr'],
        capture_output=True, text=True, timeout=120
    )

    if not os.path.exists(pdbqt_path) or os.path.getsize(pdbqt_path) == 0:
        raise ValueError(
            f"Failed to convert receptor to PDBQT: {result.stderr[:300]}"
        )

    return pdbqt_path


def get_box_from_reference(reference_path, autobox_add=4.0):
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


def pdbqt_poses_to_sdf(pdbqt_path, output_sdf_path, original_mol, scores):
    """Convert Vina output PDBQT poses back to SDF with score properties."""
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdbqt", "sdf")

    # Read all poses from PDBQT
    mol = openbabel.OBMol()
    poses = []
    obConversion.ReadFile(mol, pdbqt_path)
    while mol.NumAtoms() > 0:
        poses.append(mol)
        mol = openbabel.OBMol()
        if not obConversion.Read(mol):
            break

    # Write poses as SDF with score properties
    writer = Chem.SDWriter(output_sdf_path)

    for i, score in enumerate(scores):
        # Use RDKit to create a molecule with the docked coordinates
        if i < len(poses):
            ob_mol = poses[i]
            # Convert via SDF intermediate
            sdf_conv = openbabel.OBConversion()
            sdf_conv.SetOutFormat("sdf")
            sdf_block = sdf_conv.WriteString(ob_mol)
            rd_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False, sanitize=False)
            if rd_mol is not None:
                try:
                    Chem.SanitizeMol(rd_mol)
                except Exception:
                    pass
                rd_mol.SetProp("minimizedAffinity", f"{score:.2f}")
                rd_mol.SetProp("pose_index", str(i))
                writer.write(rd_mol)

    writer.close()


def main():
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
    args = parser.parse_args()

    # Validate inputs
    for path, label in [(args.receptor, 'Receptor'), (args.ligand, 'Ligand'), (args.reference, 'Reference')]:
        if not os.path.exists(path):
            print(f'ERROR: {label} file not found: {path}', file=sys.stderr)
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    name = Path(args.ligand).stem
    t_start = time.time()

    with tempfile.TemporaryDirectory(prefix='vina_') as tmp_dir:
        # 1. Convert receptor PDB → PDBQT
        print(f'Preparing receptor...', file=sys.stderr)
        receptor_pdbqt = os.path.join(tmp_dir, 'receptor.pdbqt')
        pdb_to_pdbqt(args.receptor, receptor_pdbqt)

        # 2. Convert ligand SDF → PDBQT
        print(f'Preparing ligand {name}...', file=sys.stderr)
        ligand_pdbqt = os.path.join(tmp_dir, 'ligand.pdbqt')
        sdf_to_pdbqt(args.ligand, ligand_pdbqt)

        # 3. Compute box from reference ligand
        center, size = get_box_from_reference(args.reference, args.autobox_add)
        print(f'Box center: [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}], '
              f'size: [{size[0]:.1f}, {size[1]:.1f}, {size[2]:.1f}]', file=sys.stderr)

        # 4. Run Vina
        v = Vina(sf_name='vina', cpu=args.cpu if args.cpu > 0 else 0,
                 seed=args.seed if args.seed > 0 else 0)
        v.set_receptor(receptor_pdbqt)
        v.set_ligand_from_file(ligand_pdbqt)
        v.compute_vina_maps(center=center, box_size=size)

        print(f'Docking (exhaustiveness={args.exhaustiveness})...', file=sys.stderr)
        v.dock(exhaustiveness=args.exhaustiveness, n_poses=args.num_poses)

        # 5. Get scores
        energies = v.energies()  # shape: (n_poses, 5) — [total, inter, intra, torsion, intra_best]
        scores = [e[0] for e in energies]  # total affinity (kcal/mol)

        # 6. Write output PDBQT
        output_pdbqt = os.path.join(tmp_dir, f'{name}_docked.pdbqt')
        v.write_poses(output_pdbqt, n_poses=args.num_poses, overwrite=True)

        # 7. Convert docked PDBQT → SDF with scores
        out_sdf = os.path.join(args.output_dir, f'{name}_docked.sdf.gz')
        temp_sdf = os.path.join(tmp_dir, f'{name}_docked.sdf')

        # Read original mol for property transfer
        if args.ligand.endswith('.gz'):
            with gzip.open(args.ligand, 'rt') as f:
                original_mol = Chem.MolFromMolBlock(f.read(), removeHs=False)
        else:
            supplier = Chem.SDMolSupplier(args.ligand, removeHs=False)
            original_mol = supplier[0]

        pdbqt_poses_to_sdf(output_pdbqt, temp_sdf, original_mol, scores)

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
