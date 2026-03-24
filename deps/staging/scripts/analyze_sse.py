#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
Compute secondary structure elements (SSE) over MD trajectory using mdtraj DSSP.

Generates per-residue SSE fractions (helix/strand/coil), a stacked bar chart,
a timeline heatmap, and raw DSSP assignment data — similar to the SSE timeline
in Schrödinger Maestro.
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute secondary structure over trajectory (DSSP)')
    parser.add_argument('--topology', required=True, help='Topology file (PDB)')
    parser.add_argument('--trajectory', required=True, help='Trajectory file (DCD)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    args = parser.parse_args()

    print(f"PROGRESS:sse:0", flush=True)

    try:
        import mdtraj
        import numpy as np
    except ImportError as e:
        print(f"Error: Missing required package: {e}", file=sys.stderr)
        print("Install with: pip install mdtraj", file=sys.stderr)
        sys.exit(1)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
    except ImportError:
        print("Warning: matplotlib not available, skipping plot generation", file=sys.stderr)
        plt = None

    print(f"Loading trajectory: {args.trajectory}")
    print(f"Topology: {args.topology}")

    # Load trajectory
    try:
        traj = mdtraj.load(args.trajectory, top=args.topology)
    except Exception as e:
        print(f"Error: Could not load trajectory: {e}", file=sys.stderr)
        sys.exit(1)

    n_frames_total = traj.n_frames
    print(f"Frames: {n_frames_total}")

    print(f"PROGRESS:sse:10", flush=True)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Identify protein residues
    protein_residues = [r for r in traj.topology.residues if r.is_protein]
    if len(protein_residues) == 0:
        print("No protein residues found, skipping SSE analysis")
        results = {
            'type': 'sse',
            'perResiduePlotPath': None,
            'timelinePlotPath': None,
            'csvPath': None,
            'data': {
                'residues': [],
                'nFrames': n_frames_total,
                'nResidues': 0,
            },
        }
        results_path = os.path.join(args.output_dir, 'sse_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"PROGRESS:sse:100", flush=True)
        print("Done!")
        sys.exit(0)

    print(f"Protein residues: {len(protein_residues)}")

    # Auto-stride for large trajectories (>1000 frames → ~500 frames)
    stride = 1
    if n_frames_total > 1000:
        stride = max(1, n_frames_total // 500)
        print(f"Large trajectory detected, using stride={stride} ({n_frames_total // stride} frames)")
        traj = traj[::stride]

    n_frames = traj.n_frames
    print(f"Frames for analysis: {n_frames}")

    print(f"PROGRESS:sse:20", flush=True)

    # Compute DSSP — returns (n_frames, n_residues) array of codes: 'H', 'E', 'C', 'NA'
    print("Computing DSSP secondary structure assignments...")
    try:
        dssp = mdtraj.compute_dssp(traj)
    except Exception as e:
        print(f"Error: DSSP computation failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"PROGRESS:sse:50", flush=True)

    # dssp shape: (n_frames, n_residues) where n_residues = all residues in topology
    n_residues_all = dssp.shape[1]

    # Build residue info and identify protein residue indices
    residue_names = []
    residue_nums = []
    protein_indices = []
    for i, res in enumerate(traj.topology.residues):
        if res.is_protein:
            protein_indices.append(i)
            residue_names.append(res.name)
            residue_nums.append(res.resSeq)

    n_protein = len(protein_indices)
    print(f"Protein residues for DSSP: {n_protein}")

    # Extract only protein residue columns
    dssp_protein = dssp[:, protein_indices]

    print(f"PROGRESS:sse:60", flush=True)

    # Compute per-residue fractions
    print("Computing per-residue SSE fractions...")
    helix_frac = np.zeros(n_protein)
    strand_frac = np.zeros(n_protein)
    coil_frac = np.zeros(n_protein)

    for j in range(n_protein):
        col = dssp_protein[:, j]
        n_valid = np.sum(col != 'NA')
        if n_valid > 0:
            helix_frac[j] = np.sum(col == 'H') / n_valid
            strand_frac[j] = np.sum(col == 'E') / n_valid
            coil_frac[j] = np.sum(col == 'C') / n_valid

    print(f"PROGRESS:sse:70", flush=True)

    # Save CSV — raw DSSP assignments per frame per residue
    print("Saving CSV data...")
    csv_path = os.path.join(args.output_dir, 'sse_data.csv')
    with open(csv_path, 'w') as f:
        header = ['frame'] + [f'{residue_names[j]}{residue_nums[j]}' for j in range(n_protein)]
        f.write(','.join(header) + '\n')
        for i in range(n_frames):
            frame_idx = i * stride if stride > 1 else i
            row = [str(frame_idx)]
            for j in range(n_protein):
                row.append(dssp_protein[i, j])
            f.write(','.join(row) + '\n')
    print(f"Data saved to: {csv_path}")

    print(f"PROGRESS:sse:80", flush=True)

    # Build per-residue results for JSON
    residue_data = []
    for j in range(n_protein):
        residue_data.append({
            'residueName': residue_names[j],
            'residueNum': int(residue_nums[j]),
            'helixFraction': round(float(helix_frac[j]), 4),
            'strandFraction': round(float(strand_frac[j]), 4),
            'coilFraction': round(float(coil_frac[j]), 4),
        })

    # Overall stats
    mean_helix = float(np.mean(helix_frac))
    mean_strand = float(np.mean(strand_frac))
    mean_coil = float(np.mean(coil_frac))

    print(f"  Mean helix: {mean_helix:.1%}")
    print(f"  Mean strand: {mean_strand:.1%}")
    print(f"  Mean coil: {mean_coil:.1%}")

    # Colors: Helix=#ef4444 (red), Strand=#2563eb (blue), Coil=#d1d5db (gray)
    COLOR_HELIX = '#ef4444'
    COLOR_STRAND = '#2563eb'
    COLOR_COIL = '#d1d5db'

    # Generate per-residue stacked bar chart (PDF)
    per_residue_plot_path = os.path.join(args.output_dir, 'sse_per_residue.pdf')
    if plt is not None and n_protein > 0:
        print("Generating per-residue stacked bar chart...")
        fig, ax = plt.subplots(figsize=(max(10, n_protein * 0.08), 5))

        x = np.arange(n_protein)
        ax.bar(x, helix_frac, color=COLOR_HELIX, label='Helix', width=1.0, edgecolor='none')
        ax.bar(x, strand_frac, bottom=helix_frac, color=COLOR_STRAND, label='Strand', width=1.0, edgecolor='none')
        ax.bar(x, coil_frac, bottom=helix_frac + strand_frac, color=COLOR_COIL, label='Coil', width=1.0, edgecolor='none')

        ax.set_xlabel('Residue Index')
        ax.set_ylabel('Fraction')
        ax.set_title('Secondary Structure per Residue')
        ax.set_xlim(-0.5, n_protein - 0.5)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right')

        # Add tick labels at intervals for readability
        if n_protein > 50:
            tick_interval = max(1, n_protein // 20)
            tick_positions = list(range(0, n_protein, tick_interval))
            tick_labels = [f'{residue_names[i]}{residue_nums[i]}' for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([f'{residue_names[i]}{residue_nums[i]}' for i in range(n_protein)],
                               rotation=90, fontsize=7)

        plt.tight_layout()
        plt.savefig(per_residue_plot_path, format='pdf', dpi=150)
        plt.close()
        print(f"Per-residue plot saved to: {per_residue_plot_path}")
    else:
        per_residue_plot_path = None

    print(f"PROGRESS:sse:90", flush=True)

    # Generate timeline heatmap (PDF)
    # Map: H→0, E→1, C→2, NA→3
    timeline_plot_path = os.path.join(args.output_dir, 'sse_timeline.pdf')
    if plt is not None and n_protein > 0 and n_frames > 0:
        print("Generating SSE timeline heatmap...")

        code_map = {'H': 0, 'E': 1, 'C': 2, 'NA': 3}
        timeline = np.full((n_frames, n_protein), 3, dtype=int)
        for i in range(n_frames):
            for j in range(n_protein):
                timeline[i, j] = code_map.get(dssp_protein[i, j], 3)

        # Discrete colormap: Helix=red, Strand=blue, Coil=gray, NA=white
        cmap = ListedColormap([COLOR_HELIX, COLOR_STRAND, COLOR_COIL, '#ffffff'])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        norm = BoundaryNorm(bounds, cmap.N)

        # Time axis in ns (assume frames are evenly spaced)
        # mdtraj stores time in ps
        time_ps = np.array([traj.time[i] for i in range(n_frames)])
        time_ns = time_ps / 1000.0

        fig, ax = plt.subplots(figsize=(max(10, n_frames * 0.015), max(6, n_protein * 0.04)))

        # Transpose so residues are on Y axis and time on X axis
        im = ax.imshow(timeline.T, aspect='auto', origin='lower',
                       cmap=cmap, norm=norm, interpolation='nearest',
                       extent=[time_ns[0], time_ns[-1], 0, n_protein])

        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Residue')
        ax.set_title('Secondary Structure Timeline')

        # Y-axis residue labels at intervals
        if n_protein > 50:
            tick_interval = max(1, n_protein // 20)
            tick_positions = list(range(0, n_protein, tick_interval))
            tick_labels = [f'{residue_names[i]}{residue_nums[i]}' for i in tick_positions]
            ax.set_yticks([p + 0.5 for p in tick_positions])
            ax.set_yticklabels(tick_labels, fontsize=7)
        else:
            ax.set_yticks([j + 0.5 for j in range(n_protein)])
            ax.set_yticklabels([f'{residue_names[j]}{residue_nums[j]}' for j in range(n_protein)],
                               fontsize=7)

        # Legend using patches
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLOR_HELIX, label='Helix'),
            Patch(facecolor=COLOR_STRAND, label='Strand'),
            Patch(facecolor=COLOR_COIL, label='Coil'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        plt.tight_layout()
        plt.savefig(timeline_plot_path, format='pdf', dpi=150)
        plt.close()
        print(f"Timeline plot saved to: {timeline_plot_path}")
    else:
        timeline_plot_path = None

    # Save results JSON
    print("Saving results JSON...")
    results = {
        'type': 'sse',
        'perResiduePlotPath': per_residue_plot_path if per_residue_plot_path and os.path.exists(per_residue_plot_path) else None,
        'timelinePlotPath': timeline_plot_path if timeline_plot_path and os.path.exists(timeline_plot_path) else None,
        'csvPath': csv_path if os.path.exists(csv_path) else None,
        'nFrames': n_frames_total,
        'stride': stride,
        'nFramesAnalyzed': n_frames,
        'data': {
            'residues': residue_data,
            'nResidues': n_protein,
            'stats': {
                'meanHelixFraction': round(mean_helix, 4),
                'meanStrandFraction': round(mean_strand, 4),
                'meanCoilFraction': round(mean_coil, 4),
            },
        },
    }

    results_path = os.path.join(args.output_dir, 'sse_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"PROGRESS:sse:100", flush=True)
    print("Done!")


if __name__ == '__main__':
    main()
