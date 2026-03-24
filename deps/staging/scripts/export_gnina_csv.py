#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
Export GNINA docking results to CSV for FragGen GUI.
Includes CORDIAL rescoring results if available.
"""

import argparse
import csv
import gzip
import json
import os
import sys
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def parse_sdf_file(sdf_path):
    """Parse a docked SDF file and extract results."""
    results = []
    # Extract ligand name from filename, handling both .sdf and .sdf.gz
    filename = os.path.basename(sdf_path)
    ligand_name = filename.replace('_docked.sdf.gz', '').replace('_docked.sdf', '')

    if not HAS_RDKIT:
        print(f"WARNING: RDKit not available, SMILES will be empty", file=sys.stderr)
        return parse_sdf_fallback(sdf_path, ligand_name)

    # Handle gzipped files
    if sdf_path.endswith('.gz'):
        with gzip.open(sdf_path, 'rb') as f:
            suppl = Chem.ForwardSDMolSupplier(f)
            for pose_idx, mol in enumerate(suppl):
                if mol is None:
                    continue
                results.append(extract_mol_data(mol, ligand_name, pose_idx, sdf_path))
    else:
        suppl = Chem.SDMolSupplier(sdf_path)
        for pose_idx, mol in enumerate(suppl):
            if mol is None:
                continue
            results.append(extract_mol_data(mol, ligand_name, pose_idx, sdf_path))

    return results


def extract_mol_data(mol, ligand_name, pose_idx, sdf_path):
    """Extract data from RDKit molecule."""
    # Get SMILES
    try:
        smiles = Chem.MolToSmiles(mol)
    except Exception:
        smiles = ''

    # Get QED
    try:
        qed = Descriptors.qed(mol)
    except Exception:
        qed = 0.0

    # Get scores from properties
    cnn_score = 0.0
    cnn_affinity = 0.0
    vina_affinity = 0.0

    for key in ['CNNscore', 'CNN_VS', 'cnn_score']:
        if mol.HasProp(key):
            try:
                cnn_score = float(mol.GetProp(key))
                break
            except (ValueError, KeyError):
                pass

    for key in ['CNNaffinity', 'CNN_affinity', 'cnn_affinity']:
        if mol.HasProp(key):
            try:
                cnn_affinity = float(mol.GetProp(key))
                break
            except (ValueError, KeyError):
                pass

    for key in ['minimizedAffinity', 'vina_affinity', 'minimized_affinity']:
        if mol.HasProp(key):
            try:
                vina_affinity = float(mol.GetProp(key))
                break
            except (ValueError, KeyError):
                pass

    return {
        'ligand_name': ligand_name,
        'pose': pose_idx + 1,
        'smiles': smiles,
        'cnn_score': cnn_score,
        'cnn_affinity': cnn_affinity,
        'vina_affinity': vina_affinity,
        'qed': qed,
        'sdf_file': os.path.basename(sdf_path),
    }


def parse_sdf_fallback(sdf_path, ligand_name):
    """Fallback parser without RDKit."""
    results = []
    current_props = {}
    pose_idx = 0

    def open_file(path):
        if path.endswith('.gz'):
            return gzip.open(path, 'rt')
        return open(path, 'r')

    with open_file(sdf_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>') and '<' in line:
                prop_name = line.split('<')[1].split('>')[0]
                value_line = next(f, '').strip()
                try:
                    current_props[prop_name] = float(value_line)
                except ValueError:
                    current_props[prop_name] = value_line
            elif line == '$$$$':
                results.append({
                    'ligand_name': ligand_name,
                    'pose': pose_idx + 1,
                    'smiles': '',
                    'cnn_score': current_props.get('CNNscore', current_props.get('CNN_VS', 0.0)),
                    'cnn_affinity': current_props.get('CNNaffinity', 0.0),
                    'vina_affinity': current_props.get('minimizedAffinity', 0.0),
                    'qed': 0.0,
                    'sdf_file': os.path.basename(sdf_path),
                })
                current_props = {}
                pose_idx += 1

    return results


def main():
    parser = argparse.ArgumentParser(description='Export GNINA results to CSV')
    parser.add_argument('--output_dir', required=True, help='Directory containing docked SDF files')
    parser.add_argument('--csv_output', required=True, help='Output CSV file path')
    parser.add_argument('--best_only', action='store_true', help='Only include best pose per ligand')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print(f'ERROR: Directory not found: {args.output_dir}', file=sys.stderr)
        sys.exit(1)

    # Find docked SDF files
    sdf_files = []
    for f in os.listdir(args.output_dir):
        if f.endswith('_docked.sdf') or f.endswith('_docked.sdf.gz'):
            sdf_files.append(os.path.join(args.output_dir, f))

    if not sdf_files:
        print('ERROR: No docked SDF files found', file=sys.stderr)
        sys.exit(1)

    # Parse all files
    all_results = []
    for sdf_path in sorted(sdf_files):
        try:
            results = parse_sdf_file(sdf_path)
            all_results.extend(results)
        except Exception as e:
            print(f'WARNING: Failed to parse {sdf_path}: {e}', file=sys.stderr)

    # Check for CORDIAL scores
    cordial_scores_path = os.path.join(args.output_dir, 'cordial_scores.json')
    cordial_scores = {}
    if os.path.exists(cordial_scores_path):
        try:
            with open(cordial_scores_path, 'r') as f:
                cordial_data = json.load(f)
            # Create lookup map: source_name_poseIndex -> scores
            for score in cordial_data:
                key = f"{score['source_name']}_{score['pose_index']}"
                cordial_scores[key] = {
                    'cordial_expected_pkd': score.get('cordial_expected_pkd', None),
                    'cordial_p_high_affinity': score.get('cordial_p_high_affinity', None),
                    'cordial_p_very_high': score.get('cordial_p_very_high', None),
                }
            print(f'Loaded {len(cordial_scores)} CORDIAL scores')
        except Exception as e:
            print(f'WARNING: Failed to load CORDIAL scores: {e}', file=sys.stderr)

    # Merge CORDIAL scores into results
    has_cordial = len(cordial_scores) > 0
    for r in all_results:
        key = f"{r['ligand_name']}_{r['pose'] - 1}"  # pose is 1-indexed in results
        if key in cordial_scores:
            r.update(cordial_scores[key])
        elif has_cordial:
            r['cordial_expected_pkd'] = None
            r['cordial_p_high_affinity'] = None
            r['cordial_p_very_high'] = None

    # Sort by CNN Affinity descending (higher = better binding), then CNN Score as tiebreaker
    all_results.sort(key=lambda x: (x['cnn_affinity'], x['cnn_score']), reverse=True)

    # Filter to best pose per ligand if requested
    if args.best_only:
        seen = set()
        filtered = []
        for r in all_results:
            if r['ligand_name'] not in seen:
                seen.add(r['ligand_name'])
                filtered.append(r)
        all_results = filtered

    # Add rank column
    for i, r in enumerate(all_results, 1):
        r['rank'] = i

    # Write CSV - column order: rank, name, pose, smiles, cnn_score, cnn_affinity, [CORDIAL], vina_affinity, qed, sdf_file
    if has_cordial:
        fieldnames = ['rank', 'ligand_name', 'pose', 'smiles', 'cnn_score', 'cnn_affinity',
                      'cordial_expected_pkd', 'cordial_p_high_affinity', 'cordial_p_very_high',
                      'vina_affinity', 'qed', 'sdf_file']
    else:
        fieldnames = ['rank', 'ligand_name', 'pose', 'smiles', 'cnn_score', 'cnn_affinity',
                      'vina_affinity', 'qed', 'sdf_file']

    with open(args.csv_output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)

    cordial_msg = ' (including CORDIAL scores)' if has_cordial else ''
    print(f'Exported {len(all_results)} results to {args.csv_output}{cordial_msg}')


if __name__ == '__main__':
    main()
