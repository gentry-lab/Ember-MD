#!/usr/bin/env python
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
Score docked poses using CORDIAL (COnvolutional Representation of Distance-dependent
Interactions with Attention Learning).

Takes docking output directory and scores all docked poses.

Usage:
    python score_cordial.py --dock_dir /path/to/dock_output --cordial_root /path/to/CORDIAL --output scores.csv

Author: FragGen Team
"""

import argparse
import os
import sys
import json
import gzip
import tempfile
import csv
from pathlib import Path

# CORDIAL imports can pull in multiple OpenMP-linked libraries in the same
# Python process on macOS/conda. Keep this subprocess single-threaded and allow
# duplicate libomp initialization so rescoring can proceed.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

def setup_cordial_path(cordial_root):
    """Add CORDIAL to Python path.

    Also remove the script's own directory so that scripts/utils.py
    (a plain module) does not shadow CORDIAL's utils/ package.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p) != script_dir]
    if cordial_root not in sys.path:
        sys.path.insert(0, cordial_root)


def prepare_receptor_for_scoring(receptor_pdb: Path, temp_dir: str) -> Path:
    """Write a CORDIAL/RDKit-friendly receptor copy for feature generation."""
    sanitized_path = Path(temp_dir) / f'{receptor_pdb.stem}_cordial.pdb'
    with receptor_pdb.open('r') as src, sanitized_path.open('w') as dst:
        for line in src:
            # PDBFixer can add OXT to truncated crystal fragments. RDKit-based
            # parsers in Meeko/CORDIAL may then infer an impossible CA-OXT bond.
            # OXT is not needed for docking rescoring features.
            if line.startswith(('ATOM', 'HETATM')) and line[12:16].strip() == 'OXT':
                continue
            dst.write(line)
    return sanitized_path


def find_receptor_for_dock_dir(dock_dir: Path) -> Path:
    """Find the receptor PDB used for docking/refinement."""
    receptor_candidates = [
        dock_dir / 'prep' / 'complex' / 'receptor_refined.pdb',
        dock_dir / 'prep' / 'canonical_receptor.pdb',
        dock_dir / 'inputs' / 'receptor.pdb',
        dock_dir / 'receptor_prepared.pdb',
    ]
    receptor_candidates.extend(dock_dir.glob('*_receptor_prepared.pdb'))
    receptor_candidates.extend(dock_dir.glob('*_receptor*.pdb'))
    receptor_pdb = next((candidate for candidate in receptor_candidates if candidate.exists()), None)
    if receptor_pdb is None:
        raise FileNotFoundError(f"No receptor PDB found in {dock_dir}")
    return receptor_pdb


def extract_pose_entries_from_dock_dir(dock_dir: Path, temp_dir: str):
    """Extract per-pose SDFs from a docking directory."""
    from rdkit import Chem

    receptor_pdb = find_receptor_for_dock_dir(dock_dir)

    new_poses_dir = dock_dir / 'results' / 'poses'
    legacy_poses_dir = dock_dir / 'poses'
    if new_poses_dir.is_dir():
        search_dir = new_poses_dir
    elif legacy_poses_dir.is_dir():
        search_dir = legacy_poses_dir
    else:
        search_dir = dock_dir

    docked_sdfs = list(search_dir.glob('*_docked.sdf.gz')) + list(search_dir.glob('*_docked.sdf'))
    if not docked_sdfs and search_dir != dock_dir:
        docked_sdfs = list(dock_dir.glob('*_docked.sdf.gz')) + list(dock_dir.glob('*_docked.sdf'))
    docked_sdfs = [s for s in docked_sdfs if '_all_docked' not in s.name]
    if not docked_sdfs:
        raise FileNotFoundError(f"No docked SDF files found in {dock_dir}")

    print(f"Found receptor: {receptor_pdb}")
    print(f"Found {len(docked_sdfs)} docked SDF files")
    print("Extracting poses from docked SDFs...")

    pose_entries = []
    for sdf_path in docked_sdfs:
        sdf_name = sdf_path.name.replace('_docked.sdf.gz', '').replace('_docked.sdf', '')

        if str(sdf_path).endswith('.gz'):
            with gzip.open(sdf_path, 'rt') as f:
                sdf_content = f.read()
            temp_sdf = os.path.join(temp_dir, f'{sdf_name}_temp.sdf')
            with open(temp_sdf, 'w') as f:
                f.write(sdf_content)
            supplier = Chem.SDMolSupplier(temp_sdf, removeHs=False, sanitize=False)
        else:
            supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)

        for pose_idx, mol in enumerate(supplier):
            if mol is None:
                continue

            try:
                Chem.SanitizeMol(mol)
            except Exception:
                try:
                    Chem.SanitizeMol(
                        mol,
                        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
                    )
                except Exception:
                    print(f"  Warning: Could not sanitize pose {pose_idx} from {sdf_name}, skipping")
                    continue

            pose_sdf_path = os.path.join(temp_dir, f'{sdf_name}_pose{pose_idx}.sdf')
            writer = Chem.SDWriter(pose_sdf_path)
            writer.write(mol)
            writer.close()

            pose_entries.append({
                'source_sdf': str(sdf_path),
                'source_name': sdf_name,
                'pose_index': pose_idx,
                'pose_sdf': pose_sdf_path,
                'receptor_pdb': str(receptor_pdb),
            })

    return pose_entries


def load_pair_entries_from_csv(pair_csv: Path):
    """Load explicit ligand/receptor pairs from CSV."""
    required_columns = {'source_name', 'ligand_sdf', 'receptor_pdb', 'pose_index'}
    with pair_csv.open('r', newline='') as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            missing = sorted(required_columns.difference(reader.fieldnames or []))
            raise ValueError(f"pair_csv missing required columns: {', '.join(missing)}")

        pose_entries = []
        for row in reader:
            ligand_sdf = (row.get('ligand_sdf') or '').strip()
            receptor_pdb = (row.get('receptor_pdb') or '').strip()
            source_name = (row.get('source_name') or '').strip()
            pose_index = int((row.get('pose_index') or '0').strip() or '0')
            if not ligand_sdf or not receptor_pdb or not source_name:
                continue
            pose_entries.append({
                'source_sdf': ligand_sdf,
                'source_name': source_name,
                'pose_index': pose_index,
                'pose_sdf': ligand_sdf,
                'receptor_pdb': receptor_pdb,
            })

    return pose_entries


def build_pair_file(pose_entries, temp_dir: str):
    """Write the ligand;protein pair file used by CORDIAL."""
    pair_file_path = os.path.join(temp_dir, 'pairs.csv')
    receptor_cache = {}
    pose_metadata = []

    with open(pair_file_path, 'w') as pair_file:
        for entry in pose_entries:
            pose_sdf = entry['pose_sdf']
            receptor_source = Path(entry['receptor_pdb'])
            sanitized_receptor = receptor_cache.get(str(receptor_source))
            if sanitized_receptor is None:
                sanitized_receptor = str(prepare_receptor_for_scoring(receptor_source, temp_dir))
                receptor_cache[str(receptor_source)] = sanitized_receptor

            pair_file.write(f"{pose_sdf};{sanitized_receptor}\n")
            pose_metadata.append({
                'source_sdf': entry['source_sdf'],
                'source_name': entry['source_name'],
                'pose_index': entry['pose_index'],
                'pose_sdf': pose_sdf,
            })

    return pair_file_path, pose_metadata

def main():
    parser = argparse.ArgumentParser(description='Score docked poses with CORDIAL')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--dock_dir', help='Docking output directory')
    input_group.add_argument('--pair_csv', help='CSV of explicit ligand/receptor pairs')
    parser.add_argument('--cordial_root', required=True, help='Path to CORDIAL repository')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    args = parser.parse_args()

    # Setup CORDIAL imports
    setup_cordial_path(args.cordial_root)

    # Now import CORDIAL modules
    import torch
    import numpy as np
    from rdkit import Chem
    from scipy.special import expit

    from modules.datasets.interaction_graph_dataset_legacy import InteractionGraphDatasetLegacy
    from modules.architectures.model_initializer import ModelInitializer
    from utils.arg_parser_utils import MasterArgumentParser

    temp_dir = tempfile.mkdtemp(prefix='cordial_')
    if args.dock_dir:
        try:
            pose_entries = extract_pose_entries_from_dock_dir(Path(args.dock_dir), temp_dir)
        except Exception as err:
            print(f"Error: {err}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            pose_entries = load_pair_entries_from_csv(Path(args.pair_csv))
        except Exception as err:
            print(f"Error: {err}", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded {len(pose_entries)} explicit receptor/ligand pairs")

    pose_count = len(pose_entries)
    if pose_count == 0:
        print("Error: No valid poses found", file=sys.stderr)
        sys.exit(1)

    pair_file_path, pose_metadata = build_pair_file(pose_entries, temp_dir)
    print(f"Prepared {pose_count} poses for CORDIAL")

    # Setup CORDIAL model paths
    model_path = os.path.join(args.cordial_root, 'weights',
                              'full.cordial.v2b.conv1d-k7c4-k3c1-nomix.attn-row_ah2-col_ah1-ff4-2x.mlp-256-256-mishx2.1-9-1.bcel-lte.model')
    norm_path = os.path.join(args.cordial_root, 'resources', 'normalization', 'full.train.norm.pkl')

    if not os.path.exists(model_path):
        print(f"Error: CORDIAL model not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create dataset
    print("Building interaction graphs and computing features...")
    print("(This may take a while for large numbers of poses)")

    dataset = InteractionGraphDatasetLegacy(
        ligand_protein_pair_file=pair_file_path,
        inference=True,
        load_normalization_data_pkl=norm_path,
        search_method='cdist',
        distance_cutoff=16.0,
        step_size=0.25,
        num_distance_bins=64,
        reduce_interaction_graph=False,
        device=device,
        use_gpu=device.type == 'cuda',
        precompute_features=True,
        cache_dir=os.path.join(temp_dir, 'cache')
    )

    print(f"Dataset created with {len(dataset)} valid samples")

    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.gpu_collate_fn
    )

    # Initialize model using the argument parser defaults
    args_parser = MasterArgumentParser()
    model_args = args_parser.parse_args([
        '--inference',
        '--load_model', model_path,
        '--model_type', 'cordial',
        '--num_result_classes', '8',
        '--device', 'cpu' if device.type == 'cpu' else '0'
    ])

    model_initializer = ModelInitializer(args=model_args, dataset=dataset, model_str='cordial')
    model = model_initializer.model

    # Load weights
    print(f"Loading model weights from {model_path}")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    # Handle DDP-trained models
    if all(key.startswith('module.') for key in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    # Run inference
    print("Running CORDIAL inference...")
    all_predictions = []
    all_indices = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            features = batch['features'].to(device)
            original_indices = batch['original_index']

            with torch.amp.autocast(device.type if device.type == 'cuda' else 'cpu'):
                predictions, _, _, _, _, _ = model({'features': features})

            # Convert logits to probabilities (float32 — autocast may produce bfloat16 on CPU)
            probs = expit(predictions.float().cpu().numpy())

            all_predictions.extend(probs)
            all_indices.extend(original_indices)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * args.batch_size} / {len(dataset)} poses")

    print(f"Inference complete. Scored {len(all_predictions)} poses.")

    # Map predictions back to original poses
    # The dataset's valid_indices maps sequential indices to original indices
    index_to_prediction = {}
    for seq_idx, (orig_idx, pred) in enumerate(zip(all_indices, all_predictions)):
        index_to_prediction[orig_idx] = pred

    # Compute summary scores from 8-class ordinal cumulative probabilities.
    # CORDIAL outputs P(pKd >= k) for k=1..8 (binary cross-entropy, ordinal "lte").
    # For ordinal cumulative probs, E[Y] = sum(P(Y >= k)), NOT sum(k * P(Y >= k)).
    # High affinity probability: P(pKd >= 6)
    def compute_scores(probs):
        """Compute summary scores from 8-class ordinal cumulative probabilities."""
        probs = np.array(probs)
        # Expected pKd: survival function identity E[Y] = sum(P(Y >= k))
        expected_pkd = float(np.sum(probs))
        # Probability of high affinity (pKd >= 6, which is index 5)
        p_high_affinity = probs[5] if len(probs) > 5 else 0.0
        # Probability of very high affinity (pKd >= 7)
        p_very_high = probs[6] if len(probs) > 6 else 0.0
        return expected_pkd, p_high_affinity, p_very_high, probs

    # Write output CSV
    print(f"Writing results to {args.output}")

    with open(args.output, 'w', newline='') as csvfile:
        fieldnames = [
            'source_sdf', 'source_name', 'pose_index',
            'cordial_expected_pkd', 'cordial_p_high_affinity', 'cordial_p_very_high',
            'cordial_p_pkd1', 'cordial_p_pkd2', 'cordial_p_pkd3', 'cordial_p_pkd4',
            'cordial_p_pkd5', 'cordial_p_pkd6', 'cordial_p_pkd7', 'cordial_p_pkd8'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for orig_idx, meta in enumerate(pose_metadata):
            if orig_idx in index_to_prediction:
                probs = index_to_prediction[orig_idx]
                expected_pkd, p_high, p_very_high, all_probs = compute_scores(probs)

                row = {
                    'source_sdf': meta['source_sdf'],
                    'source_name': meta['source_name'],
                    'pose_index': meta['pose_index'],
                    'cordial_expected_pkd': f'{expected_pkd:.4f}',
                    'cordial_p_high_affinity': f'{p_high:.4f}',
                    'cordial_p_very_high': f'{p_very_high:.4f}',
                }
                for i, p in enumerate(all_probs):
                    row[f'cordial_p_pkd{i+1}'] = f'{p:.4f}'

                writer.writerow(row)
            else:
                # Pose was skipped due to featurization error
                row = {
                    'source_sdf': meta['source_sdf'],
                    'source_name': meta['source_name'],
                    'pose_index': meta['pose_index'],
                    'cordial_expected_pkd': 'N/A',
                    'cordial_p_high_affinity': 'N/A',
                    'cordial_p_very_high': 'N/A',
                }
                for i in range(8):
                    row[f'cordial_p_pkd{i+1}'] = 'N/A'
                writer.writerow(row)

    # Cleanup temp directory
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    print("CORDIAL scoring complete!")

    # Also output JSON for easy parsing by Electron
    json_output = args.output.replace('.csv', '.json')
    results = []
    for orig_idx, meta in enumerate(pose_metadata):
        if orig_idx in index_to_prediction:
            probs = index_to_prediction[orig_idx]
            expected_pkd, p_high, p_very_high, all_probs = compute_scores(probs)
            results.append({
                'source_sdf': meta['source_sdf'],
                'source_name': meta['source_name'],
                'pose_index': meta['pose_index'],
                'cordial_expected_pkd': float(expected_pkd),
                'cordial_p_high_affinity': float(p_high),
                'cordial_p_very_high': float(p_very_high),
                'cordial_probabilities': [float(p) for p in all_probs]
            })

    with open(json_output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"JSON results written to {json_output}")

if __name__ == '__main__':
    main()
