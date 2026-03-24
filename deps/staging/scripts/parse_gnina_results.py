#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
Parse GNINA docking results for FragGen GUI.

This script parses the output SDF files from GNINA docking and extracts
the scoring information (CNNscore, CNNaffinity, minimizedAffinity).
"""

import argparse
import json
import os
import sys
import gzip
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def parse_sdf_properties(mol):
    """Extract GNINA scoring properties from an RDKit molecule."""
    props = {}

    # Try to get CNNscore (also called CNN_VS in some versions)
    for key in ['CNNscore', 'CNN_VS', 'cnn_score']:
        if mol.HasProp(key):
            try:
                props['cnnScore'] = float(mol.GetProp(key))
                break
            except ValueError:
                pass

    # Try to get CNNaffinity
    for key in ['CNNaffinity', 'CNN_affinity', 'cnn_affinity']:
        if mol.HasProp(key):
            try:
                props['cnnAffinity'] = float(mol.GetProp(key))
                break
            except ValueError:
                pass

    # Try to get Vina affinity (minimizedAffinity)
    for key in ['minimizedAffinity', 'vina_affinity', 'minimized_affinity']:
        if mol.HasProp(key):
            try:
                props['vinaAffinity'] = float(mol.GetProp(key))
                break
            except ValueError:
                pass

    # Try to get parent_molecule (for protonation tracking)
    for key in ['parent_molecule', 'parentMolecule']:
        if mol.HasProp(key):
            props['parentMolecule'] = mol.GetProp(key)
            break

    # Try to get protonation_variant
    for key in ['protonation_variant', 'protonationVariant']:
        if mol.HasProp(key):
            try:
                props['protonationVariant'] = int(mol.GetProp(key))
            except ValueError:
                pass
            break

    # Try to get conformer_index
    for key in ['conformer_index', 'conformerIndex']:
        if mol.HasProp(key):
            try:
                props['conformerIndex'] = int(mol.GetProp(key))
            except ValueError:
                pass
            break

    return props


def parse_parent_from_name(ligand_name):
    """
    Parse parent molecule, protonation variant, and conformer index from ligand name.
    Format: mol_123_prot_2_conf_1 -> parent=mol_123, prot_variant=2, conf_index=1
    Order: conformer suffix parsed first, then protonation suffix
    """
    parent = ligand_name
    conformer_index = None
    protonation_variant = None

    # Parse conformer suffix first (innermost)
    if '_conf_' in parent:
        parts = parent.rsplit('_conf_', 1)
        parent = parts[0]
        try:
            conformer_index = int(parts[1])
        except ValueError:
            pass

    # Then parse protonation suffix
    if '_prot_' in parent:
        parts = parent.rsplit('_prot_', 1)
        parent = parts[0]
        try:
            protonation_variant = int(parts[1])
        except ValueError:
            pass

    return parent, protonation_variant, conformer_index


def parse_sdf_file(sdf_path):
    """Parse a single SDF file (potentially gzipped) and extract results."""
    results = []
    # Handle both .sdf and .sdf.gz files
    filename = Path(sdf_path).name
    ligand_name = filename.replace('_docked.sdf.gz', '').replace('_docked.sdf', '')

    # Parse parent molecule from ligand name (fallback)
    parent_from_name, variant_from_name, conformer_from_name = parse_parent_from_name(ligand_name)

    # Handle gzipped files
    if sdf_path.endswith('.gz'):
        with gzip.open(sdf_path, 'rb') as f:
            suppl = Chem.ForwardSDMolSupplier(f)
            for pose_idx, mol in enumerate(suppl):
                if mol is None:
                    continue

                props = parse_sdf_properties(mol)

                # Get SMILES
                try:
                    smiles = Chem.MolToSmiles(mol)
                except Exception:
                    smiles = ''

                # Calculate QED if not present
                try:
                    qed = Descriptors.qed(mol)
                except Exception:
                    qed = 0.5

                # Determine parent molecule, protonation variant, and conformer index
                # Priority: 1) SDF properties, 2) parsed from name, 3) ligand name itself
                parent_molecule = props.get('parentMolecule', parent_from_name)
                protonation_variant = props.get('protonationVariant', variant_from_name)
                conformer_index = props.get('conformerIndex', conformer_from_name)

                result = {
                    'ligandName': ligand_name,
                    'smiles': smiles,
                    'qed': qed,
                    'cnnScore': props.get('cnnScore', 0.0),
                    'cnnAffinity': props.get('cnnAffinity', 0.0),
                    'vinaAffinity': props.get('vinaAffinity', 0.0),
                    'poseIndex': pose_idx,
                    'outputSdf': sdf_path,
                    'parentMolecule': parent_molecule,
                    'protonationVariant': protonation_variant,
                    'conformerIndex': conformer_index,
                }
                results.append(result)
    else:
        suppl = Chem.SDMolSupplier(sdf_path)
        for pose_idx, mol in enumerate(suppl):
            if mol is None:
                continue

            props = parse_sdf_properties(mol)

            # Get SMILES
            try:
                smiles = Chem.MolToSmiles(mol)
            except Exception:
                smiles = ''

            # Calculate QED if not present
            try:
                qed = Descriptors.qed(mol)
            except Exception:
                qed = 0.5

            # Determine parent molecule, protonation variant, and conformer index
            parent_molecule = props.get('parentMolecule', parent_from_name)
            protonation_variant = props.get('protonationVariant', variant_from_name)
            conformer_index = props.get('conformerIndex', conformer_from_name)

            result = {
                'ligandName': ligand_name,
                'smiles': smiles,
                'qed': qed,
                'cnnScore': props.get('cnnScore', 0.0),
                'cnnAffinity': props.get('cnnAffinity', 0.0),
                'vinaAffinity': props.get('vinaAffinity', 0.0),
                'poseIndex': pose_idx,
                'outputSdf': sdf_path,
                'parentMolecule': parent_molecule,
                'protonationVariant': protonation_variant,
                'conformerIndex': conformer_index,
            }
            results.append(result)

    return results


def parse_sdf_file_fallback(sdf_path):
    """Fallback parser when RDKit is not available."""
    results = []
    # Handle both .sdf and .sdf.gz files
    filename = Path(sdf_path).name
    ligand_name = filename.replace('_docked.sdf.gz', '').replace('_docked.sdf', '')

    # Parse parent molecule from ligand name (fallback)
    parent_from_name, variant_from_name, conformer_from_name = parse_parent_from_name(ligand_name)

    # Simple SDF property parser
    current_props = {}
    pose_idx = 0

    def open_file(path):
        if path.endswith('.gz'):
            return gzip.open(path, 'rt')
        return open(path, 'r')

    with open_file(sdf_path) as f:
        for line in f:
            line = line.strip()

            # Property line format: >  <PropertyName>
            if line.startswith('>') and '<' in line and '>' in line:
                prop_name = line.split('<')[1].split('>')[0]
                # Next line should be the value
                value_line = next(f, '').strip()
                try:
                    current_props[prop_name] = float(value_line)
                except ValueError:
                    current_props[prop_name] = value_line

            # End of molecule marker
            elif line == '$$$$':
                # Get parent molecule from props or name
                parent_molecule = current_props.get('parent_molecule', parent_from_name)
                prot_variant = current_props.get('protonation_variant', variant_from_name)
                if isinstance(prot_variant, str):
                    try:
                        prot_variant = int(prot_variant)
                    except ValueError:
                        prot_variant = None

                # Get conformer index from props or name
                conf_index = current_props.get('conformer_index', conformer_from_name)
                if isinstance(conf_index, str):
                    try:
                        conf_index = int(conf_index)
                    except ValueError:
                        conf_index = None

                result = {
                    'ligandName': ligand_name,
                    'smiles': '',
                    'qed': 0.5,
                    'cnnScore': current_props.get('CNNscore', current_props.get('CNN_VS', 0.0)),
                    'cnnAffinity': current_props.get('CNNaffinity', current_props.get('CNN_affinity', 0.0)),
                    'vinaAffinity': current_props.get('minimizedAffinity', current_props.get('vina_affinity', 0.0)),
                    'poseIndex': pose_idx,
                    'outputSdf': sdf_path,
                    'parentMolecule': parent_molecule,
                    'protonationVariant': prot_variant,
                    'conformerIndex': conf_index,
                }
                results.append(result)
                current_props = {}
                pose_idx += 1

    return results


def main():
    parser = argparse.ArgumentParser(description='Parse GNINA docking results')
    parser.add_argument('--output_dir', required=True, help='Directory containing docked SDF files')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print(f'ERROR: Output directory not found: {args.output_dir}', file=sys.stderr)
        sys.exit(1)

    # Find all docked SDF files
    sdf_files = []
    for f in os.listdir(args.output_dir):
        if f.endswith('_docked.sdf') or f.endswith('_docked.sdf.gz'):
            sdf_files.append(os.path.join(args.output_dir, f))

    if not sdf_files:
        print('[]')  # Return empty array
        return

    all_results = []

    for sdf_path in sorted(sdf_files):
        try:
            if HAS_RDKIT:
                results = parse_sdf_file(sdf_path)
            else:
                results = parse_sdf_file_fallback(sdf_path)
            all_results.extend(results)
        except Exception as e:
            print(f'WARNING: Failed to parse {sdf_path}: {e}', file=sys.stderr)

    # Sort by CNN score (descending)
    all_results.sort(key=lambda x: x.get('cnnScore', 0), reverse=True)

    # Output as JSON
    print(json.dumps(all_results))


if __name__ == '__main__':
    main()
