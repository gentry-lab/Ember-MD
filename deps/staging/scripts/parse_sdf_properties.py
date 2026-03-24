#!/usr/bin/env python
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
Parse SDF file to extract molecular properties, GNINA scores, and generate thumbnail.
Outputs JSON to stdout with all extracted data.

Handles both .sdf and .sdf.gz files.
"""
import argparse
import base64
import gzip
import io
import json
import sys
import tempfile
import os

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, QED


def parse_sdf_properties(sdf_path: str, generate_thumbnail: bool = True, thumbnail_size: int = 300) -> dict:
    """
    Parse an SDF file and extract all properties.

    Args:
        sdf_path: Path to .sdf or .sdf.gz file
        generate_thumbnail: Whether to generate 2D thumbnail
        thumbnail_size: Size of thumbnail in pixels

    Returns:
        Dictionary with extracted properties
    """
    result = {
        'success': False,
        'error': None,
        'smiles': None,
        'cnnScore': 0.0,
        'cnnAffinity': 0.0,
        'vinaAffinity': None,
        'vinaScoreOnlyAffinity': None,
        'refinementEnergy': None,
        'isReferencePose': False,
        'qed': 0.0,
        'mw': 0.0,
        'logp': 0.0,
        'thumbnail': None,
    }

    try:
        # Handle gzipped files
        if sdf_path.endswith('.gz'):
            with gzip.open(sdf_path, 'rt') as f:
                sdf_content = f.read()
            # Write to temp file for RDKit
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as tmp:
                tmp.write(sdf_content)
                tmp_path = tmp.name
            try:
                suppl = Chem.SDMolSupplier(tmp_path, sanitize=True)
                mol = next(iter(suppl), None)
            finally:
                os.unlink(tmp_path)
        else:
            suppl = Chem.SDMolSupplier(sdf_path, sanitize=True)
            mol = next(iter(suppl), None)

        if mol is None:
            result['error'] = 'Failed to parse molecule from SDF'
            return result

        # Extract SMILES
        result['smiles'] = Chem.MolToSmiles(mol)

        # Calculate molecular properties
        result['qed'] = round(QED.qed(mol), 4)
        result['mw'] = round(Descriptors.MolWt(mol), 2)
        result['logp'] = round(Descriptors.MolLogP(mol), 2)

        # Extract GNINA scores from SDF properties
        props = mol.GetPropsAsDict()

        # CNNscore (0-1, higher is better)
        if 'CNNscore' in props:
            result['cnnScore'] = float(props['CNNscore'])

        # CNNaffinity (predicted binding affinity)
        # GNINA outputs positive values where higher = better binding
        # We negate for display consistency with Vina (more negative = better)
        if 'CNNaffinity' in props:
            result['cnnAffinity'] = -abs(float(props['CNNaffinity']))

        # Vina/minimized affinity (kcal/mol, more negative is better)
        for key in ['minimizedAffinity', 'vina_affinity', 'minimized_affinity']:
            if key in props:
                result['vinaAffinity'] = float(props[key])
                break

        if 'vinaScoreOnlyAffinity' in props:
            result['vinaScoreOnlyAffinity'] = float(props['vinaScoreOnlyAffinity'])

        if 'refinement_energy' in props:
            result['refinementEnergy'] = float(props['refinement_energy'])

        if 'isReferencePose' in props:
            value = str(props['isReferencePose']).strip().lower()
            result['isReferencePose'] = value in ('1', 'true', 'yes')

        # Generate 2D thumbnail if requested
        if generate_thumbnail:
            try:
                AllChem.Compute2DCoords(mol)
                img = Draw.MolToImage(mol, size=(thumbnail_size, thumbnail_size))
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                buffer.seek(0)
                b64_data = base64.b64encode(buffer.read()).decode('utf-8')
                result['thumbnail'] = f"data:image/png;base64,{b64_data}"
            except Exception as e:
                # Thumbnail generation failed, but continue with other data
                result['thumbnail'] = None

        result['success'] = True

    except Exception as e:
        result['error'] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description='Parse SDF file for molecular properties')
    parser.add_argument('--sdf_file', required=True, help='Input SDF file (.sdf or .sdf.gz)')
    parser.add_argument('--no_thumbnail', action='store_true', help='Skip thumbnail generation')
    parser.add_argument('--thumbnail_size', type=int, default=300, help='Thumbnail size in pixels')
    args = parser.parse_args()

    result = parse_sdf_properties(
        args.sdf_file,
        generate_thumbnail=not args.no_thumbnail,
        thumbnail_size=args.thumbnail_size
    )

    print(json.dumps(result))

    if not result['success']:
        sys.exit(1)


if __name__ == '__main__':
    main()
