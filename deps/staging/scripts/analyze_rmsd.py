#!/usr/bin/env python3
"""
Calculate RMSD over MD trajectory for protein backbone and ligand.

Generates RMSD timeseries plot and data file.
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')


def main() -> None:
    parser = argparse.ArgumentParser(description='Calculate RMSD from trajectory')
    parser.add_argument('--topology', required=True, help='Topology file (PDB)')
    parser.add_argument('--trajectory', required=True, help='Trajectory file (DCD)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--ligand_selection', default=None, help='Custom ligand selection string')
    parser.add_argument('--output_format', default='png', choices=['png', 'pdf'], help='Plot output format')
    args = parser.parse_args()

    try:
        import MDAnalysis as mda
        from MDAnalysis import transformations as trans
        from MDAnalysis.analysis.rms import RMSD as RMSD_Analysis
        from MDAnalysis.analysis.rms import rmsd
        import numpy as np
    except ImportError as e:
        print(f"Error: Missing required package: {e}", file=sys.stderr)
        print("Install with: pip install MDAnalysis", file=sys.stderr)
        sys.exit(1)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plot generation", file=sys.stderr)
        plt = None

    print(f"Loading trajectory: {args.trajectory}")
    print(f"Topology: {args.topology}")

    # Load universe
    u = mda.Universe(args.topology, args.trajectory)
    n_frames = len(u.trajectory)
    print(f"Frames: {n_frames}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Select protein and backbone atoms
    # IMPORTANT: Must restrict to 'protein' to avoid including water oxygens (TIP3P uses atom name 'O')
    protein = u.select_atoms('protein')
    backbone = u.select_atoms('protein and (backbone or name CA C N O)')
    if len(backbone) == 0:
        print("Warning: No backbone atoms found, trying protein selection", file=sys.stderr)
        backbone = u.select_atoms('protein')
    print(f"Backbone atoms: {len(backbone)}")

    # Select ligand and apply PBC transforms
    from utils import select_ligand_atoms, apply_pbc_transforms
    ligand = select_ligand_atoms(u, args.ligand_selection)
    apply_pbc_transforms(u, protein, ligand)

    # Use MDAnalysis RMSD analysis class which properly handles alignment
    # This performs superposition (translation + rotation) to reference frame
    print("Calculating backbone RMSD (with alignment)...")

    # Create reference universe from same topology at frame 0
    # Also apply same transformations to reference
    ref = mda.Universe(args.topology, args.trajectory)
    try:
        ref_protein = ref.select_atoms('protein')
        ref_ligand_sele = 'not protein and not resname WAT HOH TIP3 TIP4 NA CL SOL and not element H'
        ref_ligand = ref.select_atoms(ref_ligand_sele)
        if len(ref_ligand) == 0:
            ref_ligand = ref.select_atoms('(resname LIG UNL UNK MOL) and not element H')
        if len(ref_protein) > 0:
            if len(ref_ligand) > 0:
                ref_complex = ref_protein + ref_ligand
            else:
                ref_complex = ref_protein
            ref_workflow = [
                trans.unwrap(ref_complex),
                trans.center_in_box(ref_protein, center='mass'),
                trans.wrap(ref_complex, compound='fragments'),
            ]
            ref.trajectory.add_transformations(*ref_workflow)
        elif len(ref_ligand) > 0:
            ref_workflow = [
                trans.unwrap(ref_ligand),
                trans.center_in_box(ref_ligand, center='mass'),
                trans.wrap(ref_ligand, compound='fragments'),
            ]
            ref.trajectory.add_transformations(*ref_workflow)
    except Exception:
        pass  # If transformations fail on ref, continue anyway
    ref.trajectory[0]

    # RMSD analysis with proper superposition
    rmsd_protein = None
    time_ns = None
    if len(protein) > 0:
        # IMPORTANT: Must restrict to 'protein' to avoid including water oxygens
        R = RMSD_Analysis(u, ref,
                          select='protein and (backbone or name CA C N O)',
                          ref_frame=0)
        R.run(verbose=True)

        # R.results.rmsd has columns: [frame, time, rmsd]
        time_ns = R.results.rmsd[:, 1] / 1000.0  # Convert ps to ns
        rmsd_protein = R.results.rmsd[:, 2]

        print(f"  Protein RMSD: {rmsd_protein.mean():.2f} ± {rmsd_protein.std():.2f} Å")
    else:
        print("  No protein atoms found, skipping protein RMSD")

    # Calculate ligand RMSD
    print("Calculating ligand RMSD...")
    rmsd_ligand = None
    if len(ligand) > 0:
        print(f"  Ligand atoms: {len(ligand)}")

        # Build ligand selection string for RMSD analysis
        lig_resnames = set(ligand.resnames)
        lig_sele_str = ' or '.join([f'resname {rn}' for rn in lig_resnames])
        lig_sele_str = f'({lig_sele_str}) and not element H'

        if len(protein) > 0:
            # Superpose on backbone, measure ligand RMSD
            R_lig = RMSD_Analysis(u, ref,
                                  select='protein and (backbone or name CA C N O)',
                                  groupselections=[lig_sele_str],
                                  ref_frame=0)
            R_lig.run(verbose=True)
            rmsd_ligand = R_lig.results.rmsd[:, 3]
            if time_ns is None:
                time_ns = R_lig.results.rmsd[:, 1] / 1000.0
        else:
            # Ligand-only: measure ligand RMSD directly (self-aligned)
            R_lig = RMSD_Analysis(u, ref,
                                  select=lig_sele_str,
                                  ref_frame=0)
            R_lig.run(verbose=True)
            rmsd_ligand = R_lig.results.rmsd[:, 2]
            if time_ns is None:
                time_ns = R_lig.results.rmsd[:, 1] / 1000.0

        print(f"  Ligand RMSD: {rmsd_ligand.mean():.2f} ± {rmsd_ligand.std():.2f} Å")
    else:
        print("  No ligand found, skipping ligand RMSD")

    # Save data as CSV
    print("Saving CSV data...")
    csv_path = os.path.join(args.output_dir, 'rmsd_data.csv')
    if time_ns is None:
        print("  No RMSD data to save, skipping CSV")
    else:
        with open(csv_path, 'w') as f:
            has_protein = rmsd_protein is not None
            has_ligand = rmsd_ligand is not None
            # Header
            cols = ['time_ns']
            if has_protein:
                cols.append('rmsd_protein')
            if has_ligand:
                cols.append('rmsd_ligand')
            f.write(','.join(cols) + '\n')
            # Data
            for i in range(len(time_ns)):
                vals = [f'{time_ns[i]:.4f}']
                if has_protein:
                    vals.append(f'{rmsd_protein[i]:.4f}')
                if has_ligand:
                    vals.append(f'{rmsd_ligand[i]:.4f}')
                f.write(','.join(vals) + '\n')
        print(f"Data saved to: {csv_path}")

    # Generate plot
    print("Generating plot...")
    fmt = args.output_format
    plot_path = os.path.join(args.output_dir, f'rmsd.{fmt}')
    # Also keep legacy path for backward compat
    plot_path_legacy = os.path.join(args.output_dir, 'rmsd_plot.png')
    if plt is not None and time_ns is not None and len(time_ns) > 0:
        has_both = rmsd_protein is not None and rmsd_ligand is not None

        if has_both:
            # Dual Y-axis: protein on left, ligand on right
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax2 = ax1.twinx()

            l1, = ax1.plot(time_ns, rmsd_protein, label='Protein Backbone', color='#2563eb', linewidth=0.8)
            l2, = ax2.plot(time_ns, rmsd_ligand, label='Ligand', color='#dc2626', linewidth=0.8)

            ax1.set_xlabel('Time (ns)')
            ax1.set_ylabel('Protein Cα RMSD (Å)', color='#2563eb')
            ax2.set_ylabel('Ligand RMSD (Å)', color='#dc2626')
            ax1.tick_params(axis='y', labelcolor='#2563eb')
            ax2.tick_params(axis='y', labelcolor='#dc2626')
            ax1.set_title('RMSD over Time')
            ax1.grid(True, alpha=0.3)

            lines = [l1, l2]
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')

            stats_text = (f'Protein: {np.mean(rmsd_protein):.2f} ± {np.std(rmsd_protein):.2f} Å\n'
                          f'Ligand: {np.mean(rmsd_ligand):.2f} ± {np.std(rmsd_ligand):.2f} Å')
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            stats_parts = []
            if rmsd_protein is not None:
                ax.plot(time_ns, rmsd_protein, label='Protein Backbone', color='#2563eb', linewidth=0.8)
                stats_parts.append(f'Protein: {np.mean(rmsd_protein):.2f} ± {np.std(rmsd_protein):.2f} Å')
            if rmsd_ligand is not None:
                ax.plot(time_ns, rmsd_ligand, label='Ligand', color='#dc2626', linewidth=0.8)
                stats_parts.append(f'Ligand: {np.mean(rmsd_ligand):.2f} ± {np.std(rmsd_ligand):.2f} Å')

            ax.set_xlabel('Time (ns)')
            ax.set_ylabel('RMSD (Å)')
            ax.set_title('RMSD over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)

            if stats_parts:
                stats_text = '\n'.join(stats_parts)
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        if fmt == 'pdf':
            plt.savefig(plot_path_legacy, dpi=150)
        plt.close()
        print(f"Plot saved to: {plot_path}")

    # Save results JSON (stats only - full data is in CSV)
    print("Saving results JSON...")
    n_frames = len(time_ns) if time_ns is not None else 0
    results = {
        'type': 'rmsd',
        'plotPath': plot_path if os.path.exists(plot_path) else None,
        'csvPath': csv_path if os.path.exists(csv_path) else None,
        'nFrames': n_frames,
        'data': {
            'stats': {
                'proteinMean': float(np.mean(rmsd_protein)) if rmsd_protein is not None and len(rmsd_protein) > 0 else None,
                'proteinStd': float(np.std(rmsd_protein)) if rmsd_protein is not None and len(rmsd_protein) > 0 else None,
                'proteinMin': float(np.min(rmsd_protein)) if rmsd_protein is not None and len(rmsd_protein) > 0 else None,
                'proteinMax': float(np.max(rmsd_protein)) if rmsd_protein is not None and len(rmsd_protein) > 0 else None,
                'ligandMean': float(np.mean(rmsd_ligand)) if rmsd_ligand is not None else None,
                'ligandStd': float(np.std(rmsd_ligand)) if rmsd_ligand is not None else None,
                'ligandMin': float(np.min(rmsd_ligand)) if rmsd_ligand is not None else None,
                'ligandMax': float(np.max(rmsd_ligand)) if rmsd_ligand is not None else None,
            }
        }
    }

    results_path = os.path.join(args.output_dir, 'rmsd_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("Done!")


if __name__ == '__main__':
    main()
