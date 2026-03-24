#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
Single-ligand GNINA docking for Node.js-managed parallelism.

This script docks a single ligand with GNINA. Parallelism is controlled
by the Node.js/Electron layer to avoid Open Babel initialization race
conditions that occur with Python's ProcessPoolExecutor.
"""

import argparse
import subprocess
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Dock single ligand with GNINA')
    parser.add_argument('--gnina', required=True, help='Path to GNINA executable')
    parser.add_argument('--receptor', required=True, help='Path to receptor PDB file')
    parser.add_argument('--ligand', required=True, help='Path to single ligand SDF file')
    parser.add_argument('--reference', required=True, help='Path to reference ligand for autobox')
    parser.add_argument('--output_dir', required=True, help='Output directory for docked pose')
    parser.add_argument('--exhaustiveness', type=int, default=8, help='Search exhaustiveness')
    parser.add_argument('--num_poses', type=int, default=9, help='Number of poses to generate')
    parser.add_argument('--autobox_add', type=float, default=4.0, help='Autobox margin in Angstroms')
    parser.add_argument('--no_gpu', action='store_true', help='Disable GPU, use CPU for CNN scoring')
    parser.add_argument('--minimize', action='store_true', help='Post-docking MMFF energy minimization')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility (0=random)')
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.gnina):
        print(f'ERROR: GNINA executable not found: {args.gnina}', file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.receptor):
        print(f'ERROR: Receptor file not found: {args.receptor}', file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.ligand):
        print(f'ERROR: Ligand file not found: {args.ligand}', file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.reference):
        print(f'ERROR: Reference ligand not found: {args.reference}', file=sys.stderr)
        sys.exit(1)

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    name = Path(args.ligand).stem
    out_file = os.path.join(args.output_dir, f'{name}_docked.sdf.gz')

    cmd = [
        args.gnina,
        '-r', args.receptor,
        '-l', args.ligand,
        '--autobox_ligand', args.reference,
        '--autobox_add', str(args.autobox_add),
        '-o', out_file,
        '--exhaustiveness', str(args.exhaustiveness),
        '--num_modes', str(args.num_poses),
        '--cpu', '1',  # Limit CPU usage per GNINA process
    ]

    # Post-docking MMFF energy minimization (improves pose quality)
    if args.minimize:
        cmd.append('--minimize')

    # Random seed for reproducibility
    if args.seed > 0:
        cmd.extend(['--seed', str(args.seed)])

    # Disable GPU to avoid VRAM exhaustion with parallel workers
    if args.no_gpu:
        cmd.append('--no_gpu')

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per molecule
        )

        if result.returncode == 0:
            # Output format parsed by Node.js
            print(f'SUCCESS:{name}:{out_file}')
            sys.exit(0)
        else:
            error_msg = result.stderr[:200] if result.stderr else 'Unknown error'
            print(f'FAILED:{name}:{error_msg}', file=sys.stderr)
            sys.exit(1)

    except subprocess.TimeoutExpired:
        print(f'FAILED:{name}:TIMEOUT (exceeded 5 minutes)', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'FAILED:{name}:{str(e)}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
