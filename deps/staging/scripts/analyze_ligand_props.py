#!/usr/bin/env python3
"""
Compute ligand property timeseries over an MD trajectory.

Calculates per-frame radius of gyration, SASA, polar surface area,
molecular surface area, intramolecular H-bonds, and (optionally) ligand RMSD.
Generates a 6-panel PDF figure and CSV/JSON output files.
"""

import argparse
import json
import os
import sys
import warnings
from typing import Any, Dict, Optional

warnings.filterwarnings('ignore')


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute ligand property timeseries from trajectory')
    parser.add_argument('--topology', required=True, help='Topology file (PDB)')
    parser.add_argument('--trajectory', required=True, help='Trajectory file (DCD)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--ligand_selection', default=None, help='Custom ligand selection string')
    parser.add_argument('--rmsd_csv', default=None, help='Path to existing rmsd_data.csv (reads rmsd_ligand column)')
    args = parser.parse_args()

    try:
        import MDAnalysis as mda
        from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
        import numpy as np
    except ImportError as e:
        print(f"Error: Missing required package: {e}", file=sys.stderr)
        print("Install with: pip install MDAnalysis numpy", file=sys.stderr)
        sys.exit(1)

    try:
        import mdtraj
    except ImportError:
        print("Warning: mdtraj not available, SASA/PSA/MolSA will be skipped", file=sys.stderr)
        mdtraj = None

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print("Warning: matplotlib not available, skipping plot generation", file=sys.stderr)
        plt = None

    print(f"Loading trajectory: {args.trajectory}")
    print(f"Topology: {args.topology}")
    print("PROGRESS:ligand_props:0")

    # Load universe
    u = mda.Universe(args.topology, args.trajectory)
    n_frames = len(u.trajectory)
    print(f"Frames: {n_frames}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Select ligand
    if args.ligand_selection:
        ligand_sele = args.ligand_selection
    else:
        ligand_sele = 'not protein and not resname WAT HOH TIP3 TIP4 NA CL SOL and not element H'

    ligand = u.select_atoms(ligand_sele)
    if len(ligand) == 0:
        ligand = u.select_atoms('(resname LIG UNL UNK MOL) and not element H')

    if len(ligand) == 0:
        print("No ligand atoms found. Writing empty results and exiting.")
        empty_results = {
            'type': 'ligand_props',
            'plotPath': None,
            'csvPath': None,
            'nFrames': n_frames,
            'data': {'stats': {}, 'properties': []},
        }
        results_path = os.path.join(args.output_dir, 'ligand_props_results.json')
        with open(results_path, 'w') as f:
            json.dump(empty_results, f, indent=2)
        print("PROGRESS:ligand_props:100")
        print("Done!")
        sys.exit(0)

    print(f"Ligand atoms: {len(ligand)}")
    lig_indices = ligand.indices

    # Auto-stride for large trajectories
    stride = 1
    if n_frames > 1000:
        stride = max(1, n_frames // 500)
        print(f"Auto-striding: stride={stride} ({n_frames} frames -> ~{n_frames // stride} frames)")

    n_strided = len(range(0, n_frames, stride))

    # --- Compute per-frame properties ---

    # 1. Radius of gyration
    print("Calculating radius of gyration...")
    print("PROGRESS:ligand_props:10")
    rgyr_values = []
    time_ns = []
    for i, ts in enumerate(u.trajectory[::stride]):
        rgyr_values.append(ligand.radius_of_gyration())
        time_ns.append(ts.time / 1000.0)  # ps -> ns
    rgyr_values = np.array(rgyr_values)
    time_ns = np.array(time_ns)
    print(f"  rGyr: {rgyr_values.mean():.2f} +/- {rgyr_values.std():.2f} A")

    # 2. Intramolecular H-bonds
    print("Calculating intramolecular H-bonds...")
    print("PROGRESS:ligand_props:25")
    intra_hb_counts = np.zeros(n_strided)
    try:
        # Build selection strings for ligand-only donors/acceptors
        lig_resnames = set(ligand.resnames)
        lig_sele_str = ' or '.join([f'resname {rn}' for rn in lig_resnames])
        lig_donor_sel = f"({lig_sele_str}) and (name N* or name O*)"
        lig_acceptor_sel = f"({lig_sele_str}) and (name N* or name O* or name S*)"

        hba = HydrogenBondAnalysis(
            universe=u,
            donors_sel=lig_donor_sel,
            acceptors_sel=lig_acceptor_sel,
            d_a_cutoff=3.0,
            d_h_a_angle_cutoff=150.0,
        )
        hba.run(start=0, stop=n_frames, step=stride)

        if len(hba.results.hbonds) > 0:
            # Count H-bonds per frame
            frame_indices = hba.results.hbonds[:, 0].astype(int)
            # Map absolute frame indices to strided indices
            strided_frame_map = {f: i for i, f in enumerate(range(0, n_frames, stride))}
            for fi in frame_indices:
                si = strided_frame_map.get(fi, None)
                if si is not None:
                    intra_hb_counts[si] += 1

        print(f"  intraHB: {intra_hb_counts.mean():.2f} +/- {intra_hb_counts.std():.2f}")
    except Exception as e:
        print(f"  Warning: Intramolecular H-bond analysis failed: {e}", file=sys.stderr)
        intra_hb_counts = None

    # 3. SASA, PSA, MolSA via mdtraj
    print("Calculating surface areas (SASA, PSA, MolSA)...")
    print("PROGRESS:ligand_props:40")
    sasa_values = None
    psa_values = None
    molsa_values = None

    if mdtraj is not None:
        try:
            traj = mdtraj.load(args.trajectory, top=args.topology,
                               atom_indices=lig_indices, stride=stride)
            n_traj_frames = traj.n_frames

            # SASA: shrake_rupley with default probe radius (0.14 nm)
            sasa_per_atom = mdtraj.shrake_rupley(traj)  # (n_frames, n_atoms) in nm^2
            sasa_values = sasa_per_atom.sum(axis=1) * 100.0  # nm^2 -> A^2
            print(f"  SASA: {sasa_values.mean():.1f} +/- {sasa_values.std():.1f} A^2")

            # PSA: SASA of N, O, S atoms only
            # Get element info from the mdtraj topology
            polar_mask = np.array([
                atom.element.symbol in ('N', 'O', 'S')
                for atom in traj.topology.atoms
            ])
            if polar_mask.any():
                psa_values = sasa_per_atom[:, polar_mask].sum(axis=1) * 100.0  # nm^2 -> A^2
                print(f"  PSA: {psa_values.mean():.1f} +/- {psa_values.std():.1f} A^2")
            else:
                print("  PSA: No polar atoms (N/O/S) found in ligand")
                psa_values = np.zeros(n_traj_frames)

            # MolSA: van der Waals surface (probe_radius=0)
            molsa_per_atom = mdtraj.shrake_rupley(traj, probe_radius=0.0)  # nm^2
            molsa_values = molsa_per_atom.sum(axis=1) * 100.0  # nm^2 -> A^2
            print(f"  MolSA: {molsa_values.mean():.1f} +/- {molsa_values.std():.1f} A^2")

        except Exception as e:
            print(f"  Warning: Surface area calculation failed: {e}", file=sys.stderr)
    else:
        print("  Skipped (mdtraj not available)")

    print("PROGRESS:ligand_props:60")

    # 4. Ligand RMSD from CSV
    rmsd_values = None
    rmsd_time = None
    if args.rmsd_csv and os.path.exists(args.rmsd_csv):
        print(f"Reading ligand RMSD from: {args.rmsd_csv}")
        try:
            import csv
            with open(args.rmsd_csv, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if 'rmsd_ligand' in rows[0]:
                rmsd_time_raw = np.array([float(r['time_ns']) for r in rows])
                rmsd_raw = np.array([float(r['rmsd_ligand']) for r in rows])
                # Apply same stride if needed
                if stride > 1 and len(rmsd_raw) > n_strided:
                    rmsd_values = rmsd_raw[::stride][:n_strided]
                    rmsd_time = rmsd_time_raw[::stride][:n_strided]
                else:
                    rmsd_values = rmsd_raw[:n_strided]
                    rmsd_time = rmsd_time_raw[:n_strided]
                print(f"  Ligand RMSD: {rmsd_values.mean():.2f} +/- {rmsd_values.std():.2f} A")
            else:
                print("  Warning: rmsd_ligand column not found in CSV", file=sys.stderr)
        except Exception as e:
            print(f"  Warning: Could not read RMSD CSV: {e}", file=sys.stderr)
    else:
        print("  Ligand RMSD: not provided (use --rmsd_csv)")

    print("PROGRESS:ligand_props:70")

    # --- Save CSV ---
    print("Saving CSV data...")
    csv_path = os.path.join(args.output_dir, 'ligand_props_data.csv')
    with open(csv_path, 'w') as f:
        cols = ['time_ns', 'rGyr']
        if intra_hb_counts is not None:
            cols.append('intraHB')
        if sasa_values is not None:
            cols.append('SASA')
        if psa_values is not None:
            cols.append('PSA')
        if molsa_values is not None:
            cols.append('MolSA')
        if rmsd_values is not None:
            cols.append('rmsd_ligand')
        f.write(','.join(cols) + '\n')

        for i in range(n_strided):
            vals = [f'{time_ns[i]:.4f}', f'{rgyr_values[i]:.4f}']
            if intra_hb_counts is not None:
                vals.append(f'{int(intra_hb_counts[i])}')
            if sasa_values is not None and i < len(sasa_values):
                vals.append(f'{sasa_values[i]:.2f}')
            if psa_values is not None and i < len(psa_values):
                vals.append(f'{psa_values[i]:.2f}')
            if molsa_values is not None and i < len(molsa_values):
                vals.append(f'{molsa_values[i]:.2f}')
            if rmsd_values is not None and i < len(rmsd_values):
                vals.append(f'{rmsd_values[i]:.4f}')
            f.write(','.join(vals) + '\n')
    print(f"Data saved to: {csv_path}")

    print("PROGRESS:ligand_props:80")

    # --- Generate PDF figure (2x3 grid) ---
    print("Generating PDF plot...")
    plot_path = os.path.join(args.output_dir, 'ligand_props.pdf')

    if plt is not None:
        fig, axes = plt.subplots(2, 3, figsize=(12, 10))
        fig.suptitle('Ligand Properties over Time', fontsize=14, fontweight='bold')

        panel_data = [
            ('RMSD', rmsd_values, 'RMSD (A)', '#dc2626'),
            ('rGyr', rgyr_values, 'Radius of Gyration (A)', '#2563eb'),
            ('intraHB', intra_hb_counts, 'Intramolecular H-bonds', '#059669'),
            ('MolSA', molsa_values, 'Molecular Surface Area (A^2)', '#7c3aed'),
            ('SASA', sasa_values, 'SASA (A^2)', '#d97706'),
            ('PSA', psa_values, 'Polar Surface Area (A^2)', '#0891b2'),
        ]

        for idx, (label, data, ylabel, color) in enumerate(panel_data):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            if data is not None and len(data) > 0:
                t = time_ns[:len(data)]
                ax.plot(t, data, color=color, linewidth=0.8)
                ax.set_xlabel('Time (ns)')
                ax.set_ylabel(ylabel)
                ax.set_title(label)
                ax.grid(True, alpha=0.3)

                # Add mean +/- std annotation
                mean_val = np.mean(data)
                std_val = np.std(data)
                stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, fontsize=20,
                        ha='center', va='center', color='#9ca3af')
                ax.set_title(label)
                ax.set_xlabel('Time (ns)')
                ax.set_ylabel(ylabel)

        plt.tight_layout()
        with PdfPages(plot_path) as pdf:
            pdf.savefig(fig, dpi=150)
        plt.close()
        print(f"Plot saved to: {plot_path}")

    print("PROGRESS:ligand_props:90")

    # --- Build stats for JSON ---
    def prop_stats(name: str, data: Optional[Any]) -> Dict[str, Dict[str, Optional[float]]]:
        if data is None or len(data) == 0:
            return {name: {'mean': None, 'std': None, 'min': None, 'max': None}}
        return {name: {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
        }}

    stats = {}
    stats.update(prop_stats('rGyr', rgyr_values))
    stats.update(prop_stats('SASA', sasa_values))
    stats.update(prop_stats('PSA', psa_values))
    stats.update(prop_stats('MolSA', molsa_values))
    stats.update(prop_stats('intraHB', intra_hb_counts))
    stats.update(prop_stats('rmsd_ligand', rmsd_values))

    properties = ['rGyr', 'SASA', 'PSA', 'MolSA', 'intraHB', 'rmsd_ligand']

    results = {
        'type': 'ligand_props',
        'plotPath': plot_path if os.path.exists(plot_path) else None,
        'csvPath': csv_path if os.path.exists(csv_path) else None,
        'nFrames': n_strided,
        'stride': stride,
        'ligandAtoms': len(ligand),
        'data': {
            'stats': stats,
            'properties': properties,
        },
    }

    results_path = os.path.join(args.output_dir, 'ligand_props_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("PROGRESS:ligand_props:100")
    print("Done!")


if __name__ == '__main__':
    main()
