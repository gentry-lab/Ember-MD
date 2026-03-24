#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
Analyze hydrogen bonds between protein and ligand over MD trajectory.

Calculates H-bond occupancy (% of frames with each H-bond) and generates
a summary table and plot.
"""

import argparse
import json
import os
import sys
import warnings
from typing import Any

warnings.filterwarnings('ignore')


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze protein-ligand H-bonds from trajectory')
    parser.add_argument('--topology', required=True, help='Topology file (PDB)')
    parser.add_argument('--trajectory', required=True, help='Trajectory file (DCD)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--ligand_selection', default=None, help='Custom ligand selection string')
    parser.add_argument('--distance', type=float, default=3.0, help='H-bond distance cutoff (Å)')
    parser.add_argument('--angle', type=float, default=150.0, help='H-bond angle cutoff (degrees)')
    args = parser.parse_args()

    try:
        import MDAnalysis as mda
        from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
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

    # Define selections
    protein_sel = 'protein'
    protein = u.select_atoms(protein_sel)
    if len(protein) == 0:
        print("No protein atoms found, skipping H-bond analysis")
        results = {
            'type': 'hbonds',
            'plotPath': None,
            'csvPath': None,
            'data': {
                'hbonds': [],
                'totalUnique': 0,
                'nFrames': n_frames,
            },
        }
        results_path = os.path.join(args.output_dir, 'hbonds_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        sys.exit(0)

    if args.ligand_selection:
        ligand_sel = args.ligand_selection
    else:
        # Try common ligand selections
        ligand_sel = 'not protein and not resname WAT HOH TIP3 TIP4 NA CL SOL'
        ligand = u.select_atoms(ligand_sel)
        if len(ligand) == 0:
            ligand_sel = 'resname LIG UNL MOL'

    ligand = u.select_atoms(ligand_sel)
    if len(ligand) == 0:
        print("Warning: No ligand found, analyzing protein-protein H-bonds", file=sys.stderr)
        ligand_sel = protein_sel

    print(f"Protein selection: '{protein_sel}'")
    print(f"Ligand selection: '{ligand_sel}'")

    # Run H-bond analysis
    print("Analyzing hydrogen bonds...")
    print(f"  Distance cutoff: {args.distance} Å")
    print(f"  Angle cutoff: {args.angle}°")

    # Analyze protein as donor, ligand as acceptor
    hbonds_prot_lig = HydrogenBondAnalysis(
        universe=u,
        donors_sel=f"({protein_sel}) and (name N or name O*)",
        acceptors_sel=f"({ligand_sel}) and (name N* or name O* or name S*)",
        d_a_cutoff=args.distance,
        d_h_a_angle_cutoff=args.angle,
    )

    # Analyze ligand as donor, protein as acceptor
    hbonds_lig_prot = HydrogenBondAnalysis(
        universe=u,
        donors_sel=f"({ligand_sel}) and (name N or name O*)",
        acceptors_sel=f"({protein_sel}) and (name N* or name O* or name S*)",
        d_a_cutoff=args.distance,
        d_h_a_angle_cutoff=args.angle,
    )

    print("  Analyzing protein→ligand H-bonds...")
    hbonds_prot_lig.run()

    print("  Analyzing ligand→protein H-bonds...")
    hbonds_lig_prot.run()

    # Combine results
    hbonds_all = {}

    def process_hbonds(hbond_analysis: Any, direction: str) -> None:
        if len(hbond_analysis.results.hbonds) == 0:
            return

        for hb in hbond_analysis.results.hbonds:
            frame, donor_idx, hydrogen_idx, acceptor_idx, distance, angle = hb

            # Get atom info
            donor = u.atoms[int(donor_idx)]
            acceptor = u.atoms[int(acceptor_idx)]

            donor_str = f"{donor.resname}{donor.resnum}:{donor.name}"
            acceptor_str = f"{acceptor.resname}{acceptor.resnum}:{acceptor.name}"
            key = f"{donor_str}→{acceptor_str}"

            if key not in hbonds_all:
                hbonds_all[key] = {
                    'donor': donor_str,
                    'acceptor': acceptor_str,
                    'direction': direction,
                    'frames': [],
                    'distances': [],
                    'angles': [],
                }

            hbonds_all[key]['frames'].append(int(frame))
            hbonds_all[key]['distances'].append(float(distance))
            hbonds_all[key]['angles'].append(float(angle))

    process_hbonds(hbonds_prot_lig, 'protein→ligand')
    process_hbonds(hbonds_lig_prot, 'ligand→protein')

    # Calculate occupancy
    hbond_list = []
    for key, data in hbonds_all.items():
        occupancy = 100.0 * len(data['frames']) / n_frames
        hbond_list.append({
            'donor': data['donor'],
            'acceptor': data['acceptor'],
            'direction': data['direction'],
            'occupancy': occupancy,
            'meanDistance': float(np.mean(data['distances'])),
            'meanAngle': float(np.mean(data['angles'])),
            'frameCount': len(data['frames']),
        })

    # Sort by occupancy
    hbond_list.sort(key=lambda x: x['occupancy'], reverse=True)

    print(f"\nFound {len(hbond_list)} unique H-bonds")
    if hbond_list:
        print("Top H-bonds by occupancy:")
        for hb in hbond_list[:10]:
            print(f"  {hb['donor']} → {hb['acceptor']}: {hb['occupancy']:.1f}%")

    # Save data as CSV
    csv_path = os.path.join(args.output_dir, 'hbonds_data.csv')
    with open(csv_path, 'w') as f:
        f.write('donor,acceptor,direction,occupancy,mean_distance,mean_angle,frame_count\n')
        for hb in hbond_list:
            f.write(f"{hb['donor']},{hb['acceptor']},{hb['direction']},"
                    f"{hb['occupancy']:.2f},{hb['meanDistance']:.2f},{hb['meanAngle']:.1f},"
                    f"{hb['frameCount']}\n")
    print(f"\nData saved to: {csv_path}")

    # Generate plot
    plot_path = os.path.join(args.output_dir, 'hbonds_plot.png')
    if plt is not None and len(hbond_list) > 0:
        # Show top 20 H-bonds
        top_hbonds = hbond_list[:20]

        fig, ax = plt.subplots(figsize=(10, max(6, len(top_hbonds) * 0.3)))

        labels = [f"{hb['donor']} → {hb['acceptor']}" for hb in top_hbonds]
        occupancies = [hb['occupancy'] for hb in top_hbonds]
        colors = ['#2563eb' if hb['direction'] == 'protein→ligand' else '#dc2626' for hb in top_hbonds]

        y_pos = np.arange(len(labels))
        ax.barh(y_pos, occupancies, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Occupancy (%)')
        ax.set_title('Protein-Ligand Hydrogen Bonds')
        ax.set_xlim(0, 100)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2563eb', alpha=0.8, label='Protein → Ligand'),
            Patch(facecolor='#dc2626', alpha=0.8, label='Ligand → Protein'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Plot saved to: {plot_path}")

    # Save results JSON
    results = {
        'type': 'hbonds',
        'plotPath': plot_path if os.path.exists(plot_path) else None,
        'csvPath': csv_path if os.path.exists(csv_path) else None,
        'data': {
            'hbonds': hbond_list,
            'totalUnique': len(hbond_list),
            'nFrames': n_frames,
            'distanceCutoff': args.distance,
            'angleCutoff': args.angle,
        }
    }

    results_path = os.path.join(args.output_dir, 'hbonds_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("Done!")


if __name__ == '__main__':
    main()
