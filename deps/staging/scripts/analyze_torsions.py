#!/usr/bin/env python3
"""
Compute ligand rotatable bond torsion profiles for an MD trajectory.

Outputs:
  torsions_results.json  — typed torsion bundle for the frontend
  torsions_data.csv      — per-frame dihedral angles (degrees)
  torsions.pdf           — grid of polar histogram plots
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Any, Dict, List, Optional

from ligand_torsion_utils import (
    build_torsion_identity_bundle,
    compute_dihedral_angles_for_positions,
    load_canonical_ligand_mol,
    validate_heavy_atom_count,
)
from utils import apply_pbc_transforms, select_ligand_atoms

warnings.filterwarnings('ignore')


def circular_mean(angles_deg: Any) -> float:
    import numpy as np

    rads = np.deg2rad(angles_deg)
    mean_rad = np.arctan2(np.mean(np.sin(rads)), np.mean(np.cos(rads)))
    return float(np.rad2deg(mean_rad))


def circular_std(angles_deg: Any) -> float:
    import numpy as np

    rads = np.deg2rad(angles_deg)
    R = np.sqrt(np.mean(np.cos(rads)) ** 2 + np.mean(np.sin(rads)) ** 2)
    R = min(R, 1.0)
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(R)))) if R > 0 else 180.0


def infer_ligand_sdf(topology_path: str, output_dir: str) -> Optional[str]:
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(output_dir)), 'inputs', 'ligand.sdf'),
        os.path.join(os.path.dirname(os.path.dirname(topology_path)), 'inputs', 'ligand.sdf'),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def write_empty_results(output_dir: str, n_frames: int, ligand_sdf_path: Optional[str] = None) -> None:
    results = {
        'type': 'torsions',
        'pdfPath': None,
        'csvPath': None,
        'ligandPresent': False,
        'ligandSdfPath': ligand_sdf_path,
        'nFrames': n_frames,
        'nSampledFrames': 0,
        'stride': 1,
        'sampledFrameIndices': [],
        'nRotatableBonds': 0,
        'depiction': None,
        'data': {
            'torsions': [],
        },
    }
    results_path = os.path.join(output_dir, 'torsions_results.json')
    with open(results_path, 'w') as fh:
        json.dump(results, fh, indent=2)

    csv_path = os.path.join(output_dir, 'torsions_data.csv')
    with open(csv_path, 'w') as fh:
        fh.write('frame\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute ligand rotatable bond torsion profiles')
    parser.add_argument('--topology', required=True, help='Topology file (PDB)')
    parser.add_argument('--trajectory', required=True, help='Trajectory file (DCD)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--ligand_selection', default=None, help='Custom ligand selection string')
    parser.add_argument('--ligand_sdf', default=None, help='Canonical ligand template SDF')
    args = parser.parse_args()

    print('PROGRESS:torsions:0')

    try:
        import MDAnalysis as mda
        import matplotlib
        import numpy as np
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as exc:
        print(f'Error: Missing required package: {exc}', file=sys.stderr)
        sys.exit(1)

    print(f'Loading trajectory: {args.trajectory}')
    print(f'Topology: {args.topology}')

    universe = mda.Universe(args.topology, args.trajectory)
    n_frames = len(universe.trajectory)
    print(f'Frames: {n_frames}')

    os.makedirs(args.output_dir, exist_ok=True)

    protein = universe.select_atoms('protein')
    ligand = select_ligand_atoms(universe, args.ligand_selection)
    apply_pbc_transforms(universe, protein if len(protein) > 0 else None, ligand if len(ligand) > 0 else None)

    if len(ligand) == 0:
        print('No ligand atoms found. Writing empty results.')
        write_empty_results(args.output_dir, n_frames, args.ligand_sdf)
        print('PROGRESS:torsions:100')
        print('Done!')
        return

    print(f'Ligand heavy atoms: {len(ligand)}')
    print('PROGRESS:torsions:5')

    ligand_sdf_path = args.ligand_sdf or infer_ligand_sdf(args.topology, args.output_dir)
    if not ligand_sdf_path or not os.path.exists(ligand_sdf_path):
        print('Warning: Ligand template SDF not found. Writing empty results.', file=sys.stderr)
        write_empty_results(args.output_dir, n_frames, ligand_sdf_path)
        print('PROGRESS:torsions:100')
        print('Done!')
        return

    try:
        identity = build_torsion_identity_bundle(ligand_sdf_path)
        canonical_ligand = identity['ligandMol']
        torsion_descriptors = identity['torsions']
        depiction = identity['depiction']
        validate_heavy_atom_count(canonical_ligand, len(ligand))
    except Exception as exc:
        print(f'Warning: Could not build canonical torsion identity: {exc}', file=sys.stderr)
        write_empty_results(args.output_dir, n_frames, ligand_sdf_path)
        print('PROGRESS:torsions:100')
        print('Done!')
        return

    if len(torsion_descriptors) == 0:
        print('No rotatable bonds detected in ligand. Writing empty results.')
        write_empty_results(args.output_dir, n_frames, ligand_sdf_path)
        print('PROGRESS:torsions:100')
        print('Done!')
        return

    print(f'Rotatable bonds detected: {len(torsion_descriptors)}')
    print('PROGRESS:torsions:15')

    stride = 1
    if n_frames > 1000:
        stride = max(1, n_frames // 500)
        print(f'Auto-stride: analyzing every {stride}th frame (~{n_frames // stride} frames)')

    sampled_frame_indices = list(range(0, n_frames, stride))
    n_sampled = len(sampled_frame_indices)
    all_angles = np.zeros((len(torsion_descriptors), n_sampled), dtype=np.float64)

    print('Calculating dihedral angles...')
    for sample_idx, frame_idx in enumerate(sampled_frame_indices):
        universe.trajectory[frame_idx]
        values = compute_dihedral_angles_for_positions(ligand.positions, torsion_descriptors)
        for torsion_idx, angle in enumerate(values):
            all_angles[torsion_idx, sample_idx] = angle

        if sample_idx % max(1, n_sampled // 20) == 0:
            pct = 15 + int(65 * sample_idx / max(1, n_sampled))
            print(f'PROGRESS:torsions:{pct}')

    print('PROGRESS:torsions:80')

    torsion_rows: List[Dict[str, Any]] = []
    for torsion_idx, descriptor in enumerate(torsion_descriptors):
        angles = all_angles[torsion_idx]
        torsion_rows.append({
            **descriptor,
            'circularMean': round(circular_mean(angles), 2),
            'circularStd': round(circular_std(angles), 2),
            'min': round(float(np.min(angles)), 2),
            'max': round(float(np.max(angles)), 2),
            'median': round(float(np.median(angles)), 2),
            'nFrames': int(n_sampled),
            'trajectoryAngles': [round(float(v), 2) for v in angles.tolist()],
            'clusterValues': [],
        })

    csv_path = os.path.join(args.output_dir, 'torsions_data.csv')
    with open(csv_path, 'w') as fh:
        header = ['frame'] + [row['torsionId'] for row in torsion_rows]
        fh.write(','.join(header) + '\n')
        for sample_idx, frame_idx in enumerate(sampled_frame_indices):
            values = [str(frame_idx)] + [f'{all_angles[i, sample_idx]:.2f}' for i in range(len(torsion_rows))]
            fh.write(','.join(values) + '\n')
    print(f'Data saved to: {csv_path}')
    print('PROGRESS:torsions:85')

    pdf_path = os.path.join(args.output_dir, 'torsions.pdf')
    if len(torsion_rows) > 0:
        print('Generating torsion plots...')
        torsions_per_page = 12
        n_pages = max(1, (len(torsion_rows) + torsions_per_page - 1) // torsions_per_page)

        with PdfPages(pdf_path) as pdf:
            for page_idx in range(n_pages):
                start = page_idx * torsions_per_page
                end = min(start + torsions_per_page, len(torsion_rows))
                page_rows = torsion_rows[start:end]
                cols = min(4, len(page_rows))
                rows = max(1, (len(page_rows) + cols - 1) // cols)
                fig = plt.figure(figsize=(4 * cols, 4 * rows))
                fig.suptitle('Ligand Rotatable Bond Torsion Profiles', fontsize=14, y=0.99)

                for local_idx, row in enumerate(page_rows):
                    values = np.asarray(row['trajectoryAngles'], dtype=np.float64)
                    ax = fig.add_subplot(rows, cols, local_idx + 1, projection='polar')
                    n_bins = 36
                    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
                    values_rad = np.deg2rad(values)
                    counts, _ = np.histogram(values_rad, bins=bin_edges)
                    max_count = counts.max() if counts.max() > 0 else 1
                    width = 2 * np.pi / n_bins
                    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
                    bars = ax.bar(
                        centers,
                        counts,
                        width=width,
                        bottom=0,
                        color='#2563eb',
                        edgecolor='#1d4ed8',
                        linewidth=0.5,
                    )
                    for bar, alpha in zip(bars, 0.3 + 0.7 * (counts / max_count)):
                        bar.set_alpha(float(alpha))

                    ax.set_theta_zero_location('N')
                    ax.set_theta_direction(-1)
                    ax.set_xticks(np.deg2rad([0, 60, 120, 180, 240, 300]))
                    ax.set_xticklabels(['0', '60', '120', '180', '-120', '-60'], fontsize=7)
                    ax.tick_params(axis='y', labelsize=6)
                    ax.set_title(row['label'], fontsize=8, pad=12)
                    ax.axvline(np.deg2rad(row['circularMean']), color='#dc2626', linewidth=1.5, linestyle='--', alpha=0.8)

                plt.tight_layout(rect=[0, 0, 1, 0.97])
                pdf.savefig(fig, dpi=150)
                plt.close(fig)

        print(f'Plot saved to: {pdf_path}')
    else:
        pdf_path = None

    print('PROGRESS:torsions:95')

    results = {
        'type': 'torsions',
        'pdfPath': pdf_path if pdf_path and os.path.exists(pdf_path) else None,
        'csvPath': csv_path if os.path.exists(csv_path) else None,
        'ligandPresent': True,
        'ligandSdfPath': ligand_sdf_path,
        'nFrames': int(n_frames),
        'nSampledFrames': int(n_sampled),
        'stride': int(stride),
        'sampledFrameIndices': sampled_frame_indices,
        'nRotatableBonds': len(torsion_rows),
        'depiction': depiction,
        'data': {
            'torsions': torsion_rows,
        },
    }

    results_path = os.path.join(args.output_dir, 'torsions_results.json')
    with open(results_path, 'w') as fh:
        json.dump(results, fh, indent=2)

    print(f'Results saved to: {results_path}')
    print('PROGRESS:torsions:100')
    print('Done!')


if __name__ == '__main__':
    main()
