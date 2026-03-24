#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
Detect protein-ligand interaction contacts over MD trajectory.

Identifies 4 interaction types per frame per residue (similar to Schrodinger
Maestro Simulation Interactions Diagram):
  - H-bonds (D-H...A, 2.5 A D-A distance, 120 deg angle)
  - Hydrophobic contacts (non-polar atoms within 4.0 A)
  - Ionic interactions (charged sidechains within 3.7 A of charged ligand atoms)
  - Water bridges (water O within 3.5 A of both protein and ligand heavy atoms)

Generates per-residue occupancy bar chart, interaction timeline heatmap,
and CSV/JSON data files.
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')


def main() -> None:
    parser = argparse.ArgumentParser(description='Detect protein-ligand interaction contacts from trajectory')
    parser.add_argument('--topology', required=True, help='Topology file (PDB)')
    parser.add_argument('--trajectory', required=True, help='Trajectory file (DCD)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--ligand_selection', default=None, help='Custom ligand selection string')
    args = parser.parse_args()

    try:
        import MDAnalysis as mda
        from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
        from MDAnalysis.lib.distances import distance_array
        import numpy as np
    except ImportError as e:
        print(f"Error: Missing required package: {e}", file=sys.stderr)
        print("Install with: pip install MDAnalysis", file=sys.stderr)
        sys.exit(1)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("Warning: matplotlib not available, skipping plot generation", file=sys.stderr)
        plt = None

    print(f"Loading trajectory: {args.trajectory}")
    print(f"Topology: {args.topology}")

    # Load universe
    u = mda.Universe(args.topology, args.trajectory)
    n_frames_total = len(u.trajectory)
    print(f"Frames: {n_frames_total}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    print("PROGRESS:contacts:5")

    # Select protein
    protein = u.select_atoms('protein')
    if len(protein) == 0:
        print("No protein atoms found, cannot analyze contacts")
        results = {
            'type': 'contacts',
            'summaryPdfPath': None,
            'timelinePdfPath': None,
            'csvPath': None,
            'contactResidues': [],
            'data': {
                'residues': [],
                'nFrames': n_frames_total,
                'nFramesAnalyzed': 0,
            },
        }
        results_path = os.path.join(args.output_dir, 'contacts_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print("Done!")
        sys.exit(0)

    # Select ligand
    if args.ligand_selection:
        ligand_sele = args.ligand_selection
    else:
        ligand_sele = 'not protein and not resname WAT HOH TIP3 TIP4 NA CL SOL and not element H'

    ligand = u.select_atoms(ligand_sele)
    if len(ligand) == 0:
        ligand = u.select_atoms('(resname LIG UNL UNK MOL) and not element H')

    if len(ligand) == 0:
        print("No ligand atoms found, cannot analyze contacts")
        results = {
            'type': 'contacts',
            'summaryPdfPath': None,
            'timelinePdfPath': None,
            'csvPath': None,
            'contactResidues': [],
            'data': {
                'residues': [],
                'nFrames': n_frames_total,
                'nFramesAnalyzed': 0,
            },
        }
        results_path = os.path.join(args.output_dir, 'contacts_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print("Done!")
        sys.exit(0)

    print(f"Protein atoms: {len(protein)}")
    print(f"Ligand atoms: {len(ligand)}")

    # Auto-stride for large trajectories
    stride = 1
    if n_frames_total > 1000:
        stride = max(1, n_frames_total // 500)
        print(f"Large trajectory detected, using stride={stride} (~{n_frames_total // stride} frames)")

    # Build frame list
    frame_indices = list(range(0, n_frames_total, stride))
    n_frames = len(frame_indices)

    print("PROGRESS:contacts:10")

    # -------------------------------------------------------------------------
    # Precompute atom selections for each interaction type
    # -------------------------------------------------------------------------

    # Protein heavy atoms (for water bridges)
    protein_heavy = u.select_atoms('protein and not element H')

    # Hydrophobic: non-polar C/S atoms on protein side
    protein_nonpolar = u.select_atoms('protein and (element C or element S) and not backbone')

    # Hydrophobic: non-polar atoms on ligand side (C, S, Cl, Br, F)
    ligand_nonpolar = u.select_atoms(
        f'({ligand_sele if len(u.select_atoms(ligand_sele)) > 0 else "resname LIG UNL UNK MOL"}) '
        f'and (element C or element S or element Cl or element Br or element F)'
    )
    if len(ligand_nonpolar) == 0:
        # Fallback: use ligand carbons
        lig_resnames = set(ligand.resnames)
        lig_rn_sel = ' or '.join([f'resname {rn}' for rn in lig_resnames])
        ligand_nonpolar = u.select_atoms(f'({lig_rn_sel}) and element C')

    # Ionic: charged sidechain atoms on protein
    ionic_pos_sel = 'protein and ((resname LYS and name NZ) or (resname ARG and (name NH1 or name NH2 or name NE)))'
    ionic_neg_sel = 'protein and ((resname ASP and (name OD1 or name OD2)) or (resname GLU and (name OE1 or name OE2)))'
    ionic_his_sel = 'protein and resname HIS HID HIE HIP and (name ND1 or name NE2)'
    protein_charged = u.select_atoms(f'({ionic_pos_sel}) or ({ionic_neg_sel}) or ({ionic_his_sel})')

    # Ionic: charged atoms on ligand (N and O — heuristic for N+/O-)
    lig_resnames = set(ligand.resnames)
    lig_rn_sel = ' or '.join([f'resname {rn}' for rn in lig_resnames])
    ligand_charged = u.select_atoms(f'({lig_rn_sel}) and (element N or element O)')

    # Water: select water oxygens
    water = u.select_atoms('resname WAT HOH TIP3 TIP4 SOL and name O OW')

    # Ligand heavy atoms (for H-bonds and water bridges)
    ligand_heavy = u.select_atoms(f'({lig_rn_sel}) and not element H')

    # Build protein residue map: resnum -> resname
    protein_residues = {}
    for atom in protein.atoms:
        if atom.resnum not in protein_residues:
            protein_residues[atom.resnum] = atom.resname

    # -------------------------------------------------------------------------
    # Per-frame interaction detection
    # -------------------------------------------------------------------------

    # Data structure: {resnum: {interaction_type: set_of_frame_indices}}
    residue_interactions = {}

    def record(resnum: int, itype: str, frame_idx: int) -> None:
        if resnum not in residue_interactions:
            residue_interactions[resnum] = {
                'hbond': set(),
                'hydrophobic': set(),
                'ionic': set(),
                'waterbridge': set(),
            }
        residue_interactions[resnum][itype].add(frame_idx)

    # Cutoffs
    HBOND_DIST = 2.5     # D-A distance in Angstrom
    HBOND_ANGLE = 120.0  # D-H-A angle in degrees
    HYDRO_DIST = 4.0
    IONIC_DIST = 3.7
    WATER_DIST = 3.5

    print("Detecting interactions across frames...")

    for fi, frame_idx in enumerate(frame_indices):
        u.trajectory[frame_idx]

        # Progress: 10% to 80%
        if fi % max(1, n_frames // 20) == 0:
            pct = 10 + int(70 * fi / max(1, n_frames - 1))
            print(f"PROGRESS:contacts:{pct}")

        box = u.dimensions

        # --- Hydrophobic contacts ---
        if len(protein_nonpolar) > 0 and len(ligand_nonpolar) > 0:
            dists = distance_array(protein_nonpolar.positions, ligand_nonpolar.positions, box=box)
            prot_idx, _ = np.where(dists < HYDRO_DIST)
            for pi in np.unique(prot_idx):
                atom = protein_nonpolar[pi]
                record(atom.resnum, 'hydrophobic', fi)

        # --- Ionic contacts ---
        if len(protein_charged) > 0 and len(ligand_charged) > 0:
            dists = distance_array(protein_charged.positions, ligand_charged.positions, box=box)
            prot_idx, _ = np.where(dists < IONIC_DIST)
            for pi in np.unique(prot_idx):
                atom = protein_charged[pi]
                record(atom.resnum, 'ionic', fi)

        # --- Water bridges ---
        if len(water) > 0 and len(protein_heavy) > 0 and len(ligand_heavy) > 0:
            # Water O within WATER_DIST of ligand heavy atoms
            dists_water_lig = distance_array(water.positions, ligand_heavy.positions, box=box)
            water_near_lig_mask = np.any(dists_water_lig < WATER_DIST, axis=1)
            bridging_waters = water[water_near_lig_mask]

            if len(bridging_waters) > 0:
                # Which protein heavy atoms are near these bridging waters?
                dists_water_prot = distance_array(bridging_waters.positions, protein_heavy.positions, box=box)
                _, prot_idx = np.where(dists_water_prot < WATER_DIST)
                for pi in np.unique(prot_idx):
                    atom = protein_heavy[pi]
                    record(atom.resnum, 'waterbridge', fi)

        # --- H-bonds (distance-based fast check, then angle filter) ---
        # Protein donors (N, O with H) -> ligand acceptors (N, O, S)
        # Ligand donors -> protein acceptors
        # Use a simple geometric criterion: D-A < HBOND_DIST and D-H-A > HBOND_ANGLE
        # For speed, first filter by D-A distance, then check angles

        # Protein donor heavy atoms (N, O that could have H)
        prot_donors = u.select_atoms('protein and (name N or name O or name NE or name NE2 or name ND1 or name ND2 '
                                      'or name NZ or name NH1 or name NH2 or name NE1 or name OG or name OG1 '
                                      'or name OH or name OE1 or name NE2 or name OD1)')
        # Ligand acceptors (N, O, S)
        lig_acceptors = u.select_atoms(f'({lig_rn_sel}) and (element N or element O or element S)')

        if len(prot_donors) > 0 and len(lig_acceptors) > 0:
            dists = distance_array(prot_donors.positions, lig_acceptors.positions, box=box)
            prot_idx, lig_idx = np.where(dists < HBOND_DIST)

            for pi, li in zip(prot_idx, lig_idx):
                donor = prot_donors[pi]
                acceptor = lig_acceptors[li]

                # Find H atoms bonded to this donor
                try:
                    donor_hydrogens = u.select_atoms(
                        f'element H and (around 1.3 index {donor.index})'
                    )
                except Exception:
                    donor_hydrogens = mda.AtomGroup([], u)

                if len(donor_hydrogens) > 0:
                    # Check D-H-A angle for each H
                    d_pos = donor.position
                    a_pos = acceptor.position
                    for h_atom in donor_hydrogens:
                        h_pos = h_atom.position
                        # Vectors H->D and H->A
                        v_hd = d_pos - h_pos
                        v_ha = a_pos - h_pos
                        norm_hd = np.linalg.norm(v_hd)
                        norm_ha = np.linalg.norm(v_ha)
                        if norm_hd > 0 and norm_ha > 0:
                            cos_angle = np.dot(v_hd, v_ha) / (norm_hd * norm_ha)
                            cos_angle = np.clip(cos_angle, -1.0, 1.0)
                            angle = np.degrees(np.arccos(cos_angle))
                            if angle >= HBOND_ANGLE:
                                record(donor.resnum, 'hbond', fi)
                                break
                else:
                    # No hydrogen found — accept based on distance alone (common for PDB topologies)
                    record(donor.resnum, 'hbond', fi)

        # Ligand donors -> protein acceptors
        lig_donors = u.select_atoms(f'({lig_rn_sel}) and (element N or element O)')
        prot_acceptors = u.select_atoms('protein and (element N or element O or element S) and not backbone')

        if len(lig_donors) > 0 and len(prot_acceptors) > 0:
            dists = distance_array(lig_donors.positions, prot_acceptors.positions, box=box)
            lig_idx, prot_idx = np.where(dists < HBOND_DIST)

            for li, pi in zip(lig_idx, prot_idx):
                donor = lig_donors[li]
                acceptor = prot_acceptors[pi]

                try:
                    donor_hydrogens = u.select_atoms(
                        f'element H and (around 1.3 index {donor.index})'
                    )
                except Exception:
                    donor_hydrogens = mda.AtomGroup([], u)

                if len(donor_hydrogens) > 0:
                    d_pos = donor.position
                    a_pos = acceptor.position
                    for h_atom in donor_hydrogens:
                        h_pos = h_atom.position
                        v_hd = d_pos - h_pos
                        v_ha = a_pos - h_pos
                        norm_hd = np.linalg.norm(v_hd)
                        norm_ha = np.linalg.norm(v_ha)
                        if norm_hd > 0 and norm_ha > 0:
                            cos_angle = np.dot(v_hd, v_ha) / (norm_hd * norm_ha)
                            cos_angle = np.clip(cos_angle, -1.0, 1.0)
                            angle = np.degrees(np.arccos(cos_angle))
                            if angle >= HBOND_ANGLE:
                                record(acceptor.resnum, 'hbond', fi)
                                break
                else:
                    record(acceptor.resnum, 'hbond', fi)

    print("PROGRESS:contacts:80")

    # -------------------------------------------------------------------------
    # Compute per-residue occupancies
    # -------------------------------------------------------------------------

    interaction_types = ['hbond', 'hydrophobic', 'ionic', 'waterbridge']
    type_labels = {
        'hbond': 'H-bonds',
        'hydrophobic': 'Hydrophobic',
        'ionic': 'Ionic',
        'waterbridge': 'Water bridges',
    }
    type_colors = {
        'hbond': '#2563eb',
        'hydrophobic': '#22c55e',
        'ionic': '#ef4444',
        'waterbridge': '#a855f7',
    }

    residue_data = []
    for resnum in sorted(residue_interactions.keys()):
        idata = residue_interactions[resnum]
        resname = protein_residues.get(resnum, 'UNK')
        total_frames = set()
        occupancies = {}
        for itype in interaction_types:
            frames_set = idata.get(itype, set())
            occ = 100.0 * len(frames_set) / n_frames if n_frames > 0 else 0.0
            occupancies[itype] = occ
            total_frames |= frames_set

        total_occ = 100.0 * len(total_frames) / n_frames if n_frames > 0 else 0.0
        residue_data.append({
            'resnum': int(resnum),
            'resname': resname,
            'label': f'{resname}{resnum}',
            'hbond': occupancies['hbond'],
            'hydrophobic': occupancies['hydrophobic'],
            'ionic': occupancies['ionic'],
            'waterbridge': occupancies['waterbridge'],
            'total': total_occ,
        })

    # Sort by total occupancy descending
    residue_data.sort(key=lambda x: x['total'], reverse=True)

    contact_residues = [r['resnum'] for r in residue_data if r['total'] > 0]

    print(f"\nFound interactions at {len(residue_data)} residues")
    if residue_data:
        print("Top residues by total contact occupancy:")
        for r in residue_data[:10]:
            parts = []
            if r['hbond'] > 0:
                parts.append(f"Hbond:{r['hbond']:.0f}%")
            if r['hydrophobic'] > 0:
                parts.append(f"Hydro:{r['hydrophobic']:.0f}%")
            if r['ionic'] > 0:
                parts.append(f"Ionic:{r['ionic']:.0f}%")
            if r['waterbridge'] > 0:
                parts.append(f"Water:{r['waterbridge']:.0f}%")
            print(f"  {r['label']}: {', '.join(parts)}")

    print("PROGRESS:contacts:85")

    # -------------------------------------------------------------------------
    # Save CSV: per-frame per-residue interaction data
    # -------------------------------------------------------------------------

    csv_path = os.path.join(args.output_dir, 'contacts_data.csv')
    with open(csv_path, 'w') as f:
        f.write('frame,time_ns,resnum,resname,hbond,hydrophobic,ionic,waterbridge\n')
        for fi, frame_idx in enumerate(frame_indices):
            # Get time from trajectory
            u.trajectory[frame_idx]
            time_ns = u.trajectory.time / 1000.0  # ps to ns

            for resnum, idata in residue_interactions.items():
                has_any = False
                hb = 1 if fi in idata.get('hbond', set()) else 0
                hp = 1 if fi in idata.get('hydrophobic', set()) else 0
                io = 1 if fi in idata.get('ionic', set()) else 0
                wb = 1 if fi in idata.get('waterbridge', set()) else 0
                if hb or hp or io or wb:
                    resname = protein_residues.get(resnum, 'UNK')
                    f.write(f'{frame_idx},{time_ns:.4f},{resnum},{resname},{hb},{hp},{io},{wb}\n')

    print(f"Data saved to: {csv_path}")

    print("PROGRESS:contacts:90")

    # -------------------------------------------------------------------------
    # Generate summary PDF: stacked horizontal bar chart (top 30 residues)
    # -------------------------------------------------------------------------

    summary_pdf_path = os.path.join(args.output_dir, 'contacts_summary.pdf')
    timeline_pdf_path = os.path.join(args.output_dir, 'contacts_timeline.pdf')

    if plt is not None and len(residue_data) > 0:
        top_n = min(30, len(residue_data))
        top_residues = residue_data[:top_n]
        # Reverse so highest occupancy is at top
        top_residues = top_residues[::-1]

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))

        labels = [r['label'] for r in top_residues]
        y_pos = np.arange(len(labels))

        # Stacked horizontal bars
        left = np.zeros(len(labels))
        for itype in interaction_types:
            values = np.array([r[itype] for r in top_residues])
            ax.barh(y_pos, values, left=left, color=type_colors[itype],
                    label=type_labels[itype], alpha=0.85, height=0.7)
            left += values

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Occupancy (%)')
        ax.set_title('Protein-Ligand Interaction Contacts')
        ax.set_xlim(0, min(max(left) * 1.1, 400) if len(left) > 0 else 100)

        legend_elements = [
            Patch(facecolor=type_colors[it], alpha=0.85, label=type_labels[it])
            for it in interaction_types
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.2, axis='x')

        plt.tight_layout()
        plt.savefig(summary_pdf_path, format='pdf')
        plt.close()
        print(f"Summary plot saved to: {summary_pdf_path}")
    else:
        summary_pdf_path = None

    print("PROGRESS:contacts:95")

    # -------------------------------------------------------------------------
    # Generate timeline PDF: heatmap (residue vs frame)
    # -------------------------------------------------------------------------

    if plt is not None and len(residue_data) > 0:
        # Use top 30 residues for the timeline
        top_n = min(30, len(residue_data))
        top_for_timeline = residue_data[:top_n]

        # Build heatmap matrix: rows=residues, cols=frames
        # Value: 0=none, 1=hbond, 2=hydrophobic, 3=ionic, 4=waterbridge, 5+=multiple
        # For visualization, encode as sum of interaction types present
        heatmap = np.zeros((top_n, n_frames), dtype=np.float32)

        for ri, rdata in enumerate(top_for_timeline):
            resnum = rdata['resnum']
            if resnum in residue_interactions:
                idata = residue_interactions[resnum]
                for fi_idx in range(n_frames):
                    val = 0.0
                    if fi_idx in idata.get('hbond', set()):
                        val += 1.0
                    if fi_idx in idata.get('hydrophobic', set()):
                        val += 1.0
                    if fi_idx in idata.get('ionic', set()):
                        val += 1.0
                    if fi_idx in idata.get('waterbridge', set()):
                        val += 1.0
                    heatmap[ri, fi_idx] = val

        fig, ax = plt.subplots(figsize=(12, max(5, top_n * 0.3)))

        # Custom colormap: white -> light blue -> dark blue
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('contacts', ['#ffffff', '#93c5fd', '#2563eb', '#1e3a5f'], N=256)

        im = ax.imshow(heatmap, aspect='auto', cmap=cmap, interpolation='nearest',
                        vmin=0, vmax=4)

        ax.set_yticks(np.arange(top_n))
        ax.set_yticklabels([r['label'] for r in top_for_timeline], fontsize=7)
        ax.set_xlabel('Frame')
        ax.set_title('Interaction Timeline (darker = more interaction types)')

        # Reduce x-tick clutter
        n_xticks = min(10, n_frames)
        xtick_positions = np.linspace(0, n_frames - 1, n_xticks, dtype=int)
        xtick_labels = [str(frame_indices[i]) for i in xtick_positions]
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, fontsize=8)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Interaction types')
        cbar.set_ticks([0, 1, 2, 3, 4])

        plt.tight_layout()
        plt.savefig(timeline_pdf_path, format='pdf')
        plt.close()
        print(f"Timeline plot saved to: {timeline_pdf_path}")
    else:
        timeline_pdf_path = None

    print("PROGRESS:contacts:98")

    # -------------------------------------------------------------------------
    # Save results JSON
    # -------------------------------------------------------------------------

    results = {
        'type': 'contacts',
        'summaryPdfPath': summary_pdf_path if summary_pdf_path and os.path.exists(summary_pdf_path) else None,
        'timelinePdfPath': timeline_pdf_path if timeline_pdf_path and os.path.exists(timeline_pdf_path) else None,
        'csvPath': csv_path if os.path.exists(csv_path) else None,
        'contactResidues': contact_residues,
        'data': {
            'residues': residue_data,
            'nFrames': n_frames_total,
            'nFramesAnalyzed': n_frames,
            'stride': stride,
            'cutoffs': {
                'hbondDistance': HBOND_DIST,
                'hbondAngle': HBOND_ANGLE,
                'hydrophobicDistance': HYDRO_DIST,
                'ionicDistance': IONIC_DIST,
                'waterBridgeDistance': WATER_DIST,
            },
            'interactionTypes': {
                'hbond': {'label': 'H-bonds', 'color': type_colors['hbond']},
                'hydrophobic': {'label': 'Hydrophobic', 'color': type_colors['hydrophobic']},
                'ionic': {'label': 'Ionic', 'color': type_colors['ionic']},
                'waterbridge': {'label': 'Water bridges', 'color': type_colors['waterbridge']},
            },
        }
    }

    results_path = os.path.join(args.output_dir, 'contacts_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("PROGRESS:contacts:100")
    print("Done!")


if __name__ == '__main__':
    main()
