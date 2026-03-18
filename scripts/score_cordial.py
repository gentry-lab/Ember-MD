#!/usr/bin/env python
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

def setup_cordial_path(cordial_root):
    """Add CORDIAL to Python path."""
    if cordial_root not in sys.path:
        sys.path.insert(0, cordial_root)

def main():
    parser = argparse.ArgumentParser(description='Score docked poses with CORDIAL')
    parser.add_argument('--dock_dir', required=True, help='Docking output directory')
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

    dock_dir = Path(args.dock_dir)

    # Find receptor PDB — Vina names it {project}_receptor_prepared.pdb
    receptor_candidates = list(dock_dir.glob('*_receptor_prepared.pdb')) + list(dock_dir.glob('*_receptor*.pdb'))
    if not receptor_candidates:
        # Legacy fallback
        receptor_pdb = dock_dir / 'receptor_prepared.pdb'
        if not receptor_pdb.exists():
            print(f"Error: No receptor PDB found in {dock_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        receptor_pdb = receptor_candidates[0]

    # Find all docked SDF files — check poses/ subfolder first, then top-level (legacy)
    poses_dir = dock_dir / 'poses'
    search_dir = poses_dir if poses_dir.is_dir() else dock_dir
    docked_sdfs = list(search_dir.glob('*_docked.sdf.gz')) + list(search_dir.glob('*_docked.sdf'))
    if not docked_sdfs:
        # Fallback: also check top-level if poses/ was empty
        if search_dir != dock_dir:
            docked_sdfs = list(dock_dir.glob('*_docked.sdf.gz')) + list(dock_dir.glob('*_docked.sdf'))
    # Filter out the pooled _all_docked.sdf file (that's for portability, not scoring)
    docked_sdfs = [s for s in docked_sdfs if '_all_docked' not in s.name]
    if not docked_sdfs:
        print(f"Error: No docked SDF files found in {dock_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found receptor: {receptor_pdb}")
    print(f"Found {len(docked_sdfs)} docked SDF files")

    # Extract all poses to temporary SDF files and create pair file
    temp_dir = tempfile.mkdtemp(prefix='cordial_')
    pair_file_path = os.path.join(temp_dir, 'pairs.csv')
    pose_metadata = []  # Track which pose came from which file

    print("Extracting poses from docked SDFs...")
    pose_count = 0

    with open(pair_file_path, 'w') as pair_file:
        for sdf_path in docked_sdfs:
            sdf_name = sdf_path.name.replace('_docked.sdf.gz', '').replace('_docked.sdf', '')

            # Handle gzipped files
            if str(sdf_path).endswith('.gz'):
                with gzip.open(sdf_path, 'rt') as f:
                    sdf_content = f.read()
                # Write to temp uncompressed file for RDKit
                temp_sdf = os.path.join(temp_dir, f'{sdf_name}_temp.sdf')
                with open(temp_sdf, 'w') as f:
                    f.write(sdf_content)
                supplier = Chem.SDMolSupplier(temp_sdf, removeHs=False, sanitize=False)
            else:
                supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)

            # Extract each pose as a separate SDF
            for pose_idx, mol in enumerate(supplier):
                if mol is None:
                    continue

                # Try to sanitize
                try:
                    Chem.SanitizeMol(mol)
                except:
                    try:
                        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    except:
                        print(f"  Warning: Could not sanitize pose {pose_idx} from {sdf_name}, skipping")
                        continue

                # Write individual pose SDF
                pose_sdf_path = os.path.join(temp_dir, f'{sdf_name}_pose{pose_idx}.sdf')
                writer = Chem.SDWriter(pose_sdf_path)
                writer.write(mol)
                writer.close()

                # Add to pair file (ligand;protein format)
                pair_file.write(f"{pose_sdf_path};{receptor_pdb}\n")

                # Track metadata
                pose_metadata.append({
                    'source_sdf': str(sdf_path),
                    'source_name': sdf_name,
                    'pose_index': pose_idx,
                    'pose_sdf': pose_sdf_path
                })
                pose_count += 1

    print(f"Extracted {pose_count} poses")

    if pose_count == 0:
        print("Error: No valid poses found", file=sys.stderr)
        sys.exit(1)

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

            # Convert logits to probabilities
            probs = expit(predictions.cpu().numpy())

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

    # Compute summary scores from 8-class ordinal probabilities
    # Expected pKd: weighted sum of thresholds
    # High affinity probability: P(pKd >= 6)
    def compute_scores(probs):
        """Compute summary scores from 8-class ordinal probabilities."""
        probs = np.array(probs)
        # Expected pKd (weighted sum, thresholds 1-8)
        expected_pkd = sum((i + 1) * probs[i] for i in range(8))
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
