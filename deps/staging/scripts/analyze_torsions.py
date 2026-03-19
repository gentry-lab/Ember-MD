#!/usr/bin/env python3
"""
Compute rotatable bond dihedral profiles for a ligand over an MD trajectory.

Detects rotatable bonds via RDKit SMARTS, measures dihedral angles per frame
using MDAnalysis, and generates polar histogram (rose) plots similar to
Schrodinger Maestro's torsion profile page.

Outputs:
  torsions_results.json  — per-torsion circular statistics
  torsions_data.csv      — per-frame dihedral angles (degrees)
  torsions.pdf           — grid of polar rose plots, one per rotatable bond
"""

import argparse
import json
import os
import sys
import tempfile
import warnings
from typing import Any

warnings.filterwarnings('ignore')


def circular_mean(angles_deg: Any) -> float:
    """Compute circular mean of angles in degrees."""
    import numpy as np
    rads = np.deg2rad(angles_deg)
    mean_rad = np.arctan2(np.mean(np.sin(rads)), np.mean(np.cos(rads)))
    return float(np.rad2deg(mean_rad))


def circular_std(angles_deg: Any) -> float:
    """Compute circular standard deviation of angles in degrees."""
    import numpy as np
    rads = np.deg2rad(angles_deg)
    R = np.sqrt(np.mean(np.cos(rads))**2 + np.mean(np.sin(rads))**2)
    # Clamp R to [0, 1] for numerical safety
    R = min(R, 1.0)
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(R)))) if R > 0 else 180.0


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute ligand rotatable bond torsion profiles')
    parser.add_argument('--topology', required=True, help='Topology file (PDB)')
    parser.add_argument('--trajectory', required=True, help='Trajectory file (DCD)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--ligand_selection', default=None, help='Custom ligand selection string')
    args = parser.parse_args()

    print("PROGRESS:torsions:0")

    try:
        import MDAnalysis as mda
        from MDAnalysis.lib.distances import calc_dihedrals
        import numpy as np
    except ImportError as e:
        print(f"Error: Missing required package: {e}", file=sys.stderr)
        print("Install with: pip install MDAnalysis", file=sys.stderr)
        sys.exit(1)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print("Warning: matplotlib not available, skipping plot generation", file=sys.stderr)
        plt = None

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdMolDescriptors
    except ImportError:
        print("Error: RDKit not available, cannot detect rotatable bonds", file=sys.stderr)
        sys.exit(1)

    print(f"Loading trajectory: {args.trajectory}")
    print(f"Topology: {args.topology}")

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
        print("No ligand atoms found. Writing empty results.")
        _write_empty_results(args.output_dir, n_frames)
        print("PROGRESS:torsions:100")
        print("Done!")
        sys.exit(0)

    print(f"Ligand atoms: {len(ligand)}")

    print("PROGRESS:torsions:5")

    # ------------------------------------------------------------------
    # Extract ligand PDB block from first frame for RDKit parsing
    # ------------------------------------------------------------------
    u.trajectory[0]

    # Write ligand atoms to a temp PDB file for RDKit
    tmp_pdb = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w')
    try:
        ligand.write(tmp_pdb.name)
        tmp_pdb.close()

        mol = Chem.MolFromPDBFile(tmp_pdb.name, removeHs=False, sanitize=False)
        if mol is None:
            mol = Chem.MolFromPDBFile(tmp_pdb.name, removeHs=True, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass  # Continue with unsanitized mol — rotatable bond detection still works
    finally:
        os.unlink(tmp_pdb.name)

    if mol is None:
        print("Warning: RDKit could not parse ligand structure. Writing empty results.",
              file=sys.stderr)
        _write_empty_results(args.output_dir, n_frames)
        print("PROGRESS:torsions:100")
        print("Done!")
        sys.exit(0)

    print("PROGRESS:torsions:10")

    # ------------------------------------------------------------------
    # Detect rotatable bonds via SMARTS
    # ------------------------------------------------------------------
    rotatable_pattern = Chem.MolFromSmarts(
        '[!$([NH]!@C(=O))&!D1]-&!@[!$([NH]!@C(=O))&!D1]'
    )
    matches = mol.GetSubstructMatches(rotatable_pattern)

    if len(matches) == 0:
        print("No rotatable bonds detected in ligand. Writing empty results.")
        _write_empty_results(args.output_dir, n_frames)
        print("PROGRESS:torsions:100")
        print("Done!")
        sys.exit(0)

    print(f"Rotatable bonds detected: {len(matches)}")

    # ------------------------------------------------------------------
    # Map RDKit atom indices to MDAnalysis atom indices
    # ------------------------------------------------------------------
    # Strategy: match by coordinates from frame 0. RDKit mol atom order
    # corresponds to the PDB atom order written by MDAnalysis, which matches
    # the ligand AtomGroup order.
    # Build a coordinate-based mapping as a robust fallback.

    rdkit_to_mda = {}
    conf = mol.GetConformer()
    ligand_positions = ligand.positions  # (N, 3) from frame 0

    for rdkit_idx in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(rdkit_idx)
        rdkit_coord = np.array([pos.x, pos.y, pos.z])
        # Find closest MDAnalysis atom
        dists = np.linalg.norm(ligand_positions - rdkit_coord, axis=1)
        mda_local_idx = int(np.argmin(dists))
        if dists[mda_local_idx] < 1.0:  # Tolerance 1 A
            rdkit_to_mda[rdkit_idx] = mda_local_idx

    print(f"Atom mapping: {len(rdkit_to_mda)}/{mol.GetNumAtoms()} mapped")

    # ------------------------------------------------------------------
    # Build dihedral atom quartets for each rotatable bond
    # ------------------------------------------------------------------
    dihedrals = []  # List of dicts with mda indices and label

    for match in matches:
        rdkit_i, rdkit_j = match[0], match[1]

        # Find a neighbor of atom_i that is not atom_j
        atom_i = mol.GetAtomWithIdx(rdkit_i)
        neighbor_a = None
        for nbr in atom_i.GetNeighbors():
            if nbr.GetIdx() != rdkit_j:
                neighbor_a = nbr.GetIdx()
                break

        # Find a neighbor of atom_j that is not atom_i
        atom_j = mol.GetAtomWithIdx(rdkit_j)
        neighbor_d = None
        for nbr in atom_j.GetNeighbors():
            if nbr.GetIdx() != rdkit_i:
                neighbor_d = nbr.GetIdx()
                break

        if neighbor_a is None or neighbor_d is None:
            continue

        # Check all four atoms are mapped
        quartet_rdkit = [neighbor_a, rdkit_i, rdkit_j, neighbor_d]
        if not all(q in rdkit_to_mda for q in quartet_rdkit):
            continue

        quartet_mda = [rdkit_to_mda[q] for q in quartet_rdkit]

        # Build label from atom names
        atom_names = []
        for q in quartet_rdkit:
            rd_atom = mol.GetAtomWithIdx(q)
            pdb_info = rd_atom.GetPDBResidueInfo()
            if pdb_info is not None:
                atom_names.append(pdb_info.GetName().strip())
            else:
                atom_names.append(rd_atom.GetSymbol() + str(q))
        label = '-'.join(atom_names)

        dihedrals.append({
            'quartet_mda': quartet_mda,
            'label': label,
            'rdkit_bond': (rdkit_i, rdkit_j),
        })

    if len(dihedrals) == 0:
        print("Could not build dihedral quartets. Writing empty results.")
        _write_empty_results(args.output_dir, n_frames)
        print("PROGRESS:torsions:100")
        print("Done!")
        sys.exit(0)

    print(f"Dihedral quartets built: {len(dihedrals)}")

    print("PROGRESS:torsions:15")

    # ------------------------------------------------------------------
    # Auto-stride for large trajectories
    # ------------------------------------------------------------------
    stride = 1
    if n_frames > 1000:
        stride = max(1, n_frames // 500)
        print(f"Auto-stride: analyzing every {stride}th frame (~{n_frames // stride} frames)")

    frame_indices = list(range(0, n_frames, stride))
    n_sampled = len(frame_indices)

    # ------------------------------------------------------------------
    # Measure dihedrals per frame
    # ------------------------------------------------------------------
    print("Calculating dihedral angles...")

    # Pre-allocate: (n_dihedrals, n_sampled)
    all_angles = np.zeros((len(dihedrals), n_sampled), dtype=np.float64)

    for fi, frame_idx in enumerate(frame_indices):
        u.trajectory[frame_idx]
        positions = ligand.positions

        for di, dih in enumerate(dihedrals):
            q = dih['quartet_mda']
            a = positions[q[0]].reshape(1, 3).astype(np.float32)
            b = positions[q[1]].reshape(1, 3).astype(np.float32)
            c = positions[q[2]].reshape(1, 3).astype(np.float32)
            d = positions[q[3]].reshape(1, 3).astype(np.float32)
            angle_rad = calc_dihedrals(a, b, c, d)
            all_angles[di, fi] = np.rad2deg(angle_rad[0])

        # Progress: 15-80%
        if fi % max(1, n_sampled // 20) == 0:
            pct = 15 + int(65 * fi / n_sampled)
            print(f"PROGRESS:torsions:{pct}")

    print("PROGRESS:torsions:80")

    # ------------------------------------------------------------------
    # Compute per-torsion statistics
    # ------------------------------------------------------------------
    torsion_stats = []
    for di, dih in enumerate(dihedrals):
        angles = all_angles[di]
        cmean = circular_mean(angles)
        cstd = circular_std(angles)
        torsion_stats.append({
            'label': dih['label'],
            'bondAtoms': list(dih['rdkit_bond']),
            'circularMean': round(cmean, 2),
            'circularStd': round(cstd, 2),
            'min': round(float(np.min(angles)), 2),
            'max': round(float(np.max(angles)), 2),
            'median': round(float(np.median(angles)), 2),
            'nFrames': n_sampled,
        })

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(args.output_dir, 'torsions_data.csv')
    with open(csv_path, 'w') as f:
        header = ['frame'] + [d['label'] for d in dihedrals]
        f.write(','.join(header) + '\n')
        for fi, frame_idx in enumerate(frame_indices):
            vals = [str(frame_idx)]
            for di in range(len(dihedrals)):
                vals.append(f'{all_angles[di, fi]:.2f}')
            f.write(','.join(vals) + '\n')
    print(f"Data saved to: {csv_path}")

    print("PROGRESS:torsions:85")

    # ------------------------------------------------------------------
    # Generate PDF with polar histogram (rose) plots
    # ------------------------------------------------------------------
    pdf_path = os.path.join(args.output_dir, 'torsions.pdf')
    if plt is not None and len(dihedrals) > 0:
        print("Generating torsion plots...")
        n_torsions = len(dihedrals)
        torsions_per_page = 12
        n_pages = max(1, (n_torsions + torsions_per_page - 1) // torsions_per_page)

        with PdfPages(pdf_path) as pdf:
            for page in range(n_pages):
                start = page * torsions_per_page
                end = min(start + torsions_per_page, n_torsions)
                n_on_page = end - start

                # Grid layout: up to 4 columns
                cols = min(4, n_on_page)
                rows = max(1, (n_on_page + cols - 1) // cols)

                fig = plt.figure(figsize=(4 * cols, 4 * rows))
                fig.suptitle('Ligand Rotatable Bond Torsion Profiles', fontsize=14, y=0.99)

                for idx in range(n_on_page):
                    di = start + idx
                    angles = all_angles[di]

                    ax = fig.add_subplot(rows, cols, idx + 1, projection='polar')

                    # Polar histogram: 36 bins (10 degrees each)
                    n_bins = 36
                    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
                    angles_rad = np.deg2rad(angles)

                    counts, _ = np.histogram(angles_rad, bins=bin_edges)

                    # Normalize to density for coloring
                    max_count = counts.max() if counts.max() > 0 else 1

                    # Bar width = bin width
                    width = 2 * np.pi / n_bins
                    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

                    # Color intensity by density
                    base_color = np.array([37, 99, 235]) / 255.0  # #2563eb
                    alphas = 0.3 + 0.7 * (counts / max_count)

                    bars = ax.bar(centers, counts, width=width, bottom=0,
                                  color='#2563eb', edgecolor='#1d4ed8', linewidth=0.5)
                    for bar, alpha in zip(bars, alphas):
                        bar.set_alpha(float(alpha))

                    # Style the polar plot
                    ax.set_theta_zero_location('N')
                    ax.set_theta_direction(-1)
                    ax.set_xticks(np.deg2rad([0, 60, 120, 180, 240, 300]))
                    ax.set_xticklabels(['0', '60', '120', '180', '-120', '-60'], fontsize=7)
                    ax.tick_params(axis='y', labelsize=6)
                    ax.set_title(dihedrals[di]['label'], fontsize=8, pad=12)

                    # Add circular mean marker
                    cmean_rad = np.deg2rad(torsion_stats[di]['circularMean'])
                    ax.axvline(cmean_rad, color='#dc2626', linewidth=1.5, linestyle='--', alpha=0.8)

                plt.tight_layout(rect=[0, 0, 1, 0.97])
                pdf.savefig(fig, dpi=150)
                plt.close(fig)

        print(f"Plot saved to: {pdf_path}")
    else:
        pdf_path = None

    print("PROGRESS:torsions:95")

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    results = {
        'type': 'torsions',
        'pdfPath': pdf_path if pdf_path and os.path.exists(pdf_path) else None,
        'csvPath': csv_path if os.path.exists(csv_path) else None,
        'nFrames': n_frames,
        'nSampledFrames': n_sampled,
        'stride': stride,
        'nRotatableBonds': len(dihedrals),
        'data': {
            'torsions': torsion_stats,
        },
    }

    results_path = os.path.join(args.output_dir, 'torsions_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")
    print("PROGRESS:torsions:100")
    print("Done!")


def _write_empty_results(output_dir: str, n_frames: int) -> None:
    """Write empty results files when no torsions can be analyzed."""
    results = {
        'type': 'torsions',
        'pdfPath': None,
        'csvPath': None,
        'nFrames': n_frames,
        'nSampledFrames': 0,
        'stride': 1,
        'nRotatableBonds': 0,
        'data': {
            'torsions': [],
        },
    }
    results_path = os.path.join(output_dir, 'torsions_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(output_dir, 'torsions_data.csv')
    with open(csv_path, 'w') as f:
        f.write('frame\n')


if __name__ == '__main__':
    main()
