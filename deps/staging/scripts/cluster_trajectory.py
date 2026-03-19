#!/usr/bin/env python3
"""
Cluster MD trajectory using MDAnalysis and scikit-learn.

Performs RMSD-based clustering to identify representative conformations.
Outputs cluster assignments and centroid structures.
"""

import argparse
import json
import os
import sys
import warnings
from typing import Any

warnings.filterwarnings('ignore')


def main() -> None:
    parser = argparse.ArgumentParser(description='Cluster MD trajectory by RMSD')
    parser.add_argument('--topology', required=True, help='Topology file (PDB)')
    parser.add_argument('--trajectory', required=True, help='Trajectory file (DCD)')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters')
    parser.add_argument('--method', choices=['kmeans', 'dbscan', 'hierarchical'], default='kmeans',
                        help='Clustering method')
    parser.add_argument('--selection', choices=['ligand', 'backbone', 'all'], default='ligand',
                        help='Atoms to use for RMSD calculation')
    parser.add_argument('--strip_waters', action='store_true', help='Remove waters from output PDBs')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--stride', type=int, default=1, help='Frame stride for subsampling (e.g., 10 = every 10th frame)')
    parser.add_argument('--max_frames', type=int, default=1000, help='Maximum frames to use for clustering')
    args = parser.parse_args()

    try:
        import MDAnalysis as mda
        from MDAnalysis.analysis import align, rms
        import numpy as np
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    except ImportError as e:
        print(f"Error: Missing required package: {e}", file=sys.stderr)
        print("Install with: pip install MDAnalysis scikit-learn", file=sys.stderr)
        sys.exit(1)

    print(f"Loading trajectory: {args.trajectory}")
    print(f"Topology: {args.topology}")

    # Load universe
    u = mda.Universe(args.topology, args.trajectory)
    n_frames = len(u.trajectory)
    print(f"Frames: {n_frames}")

    # Build selection string based on selection type
    if args.selection == 'ligand':
        # Try common ligand selections
        sele_str = 'not protein and not resname WAT HOH TIP3 TIP4 NA CL SOL and not element H'
        sel = u.select_atoms(sele_str)
        if len(sel) == 0:
            # Fall back to resname LIG or UNL
            sele_str = '(resname LIG UNL MOL) and not element H'
            sel = u.select_atoms(sele_str)
        if len(sel) == 0:
            print("Warning: No ligand found, falling back to backbone", file=sys.stderr)
            args.selection = 'backbone'

    if args.selection == 'backbone':
        # IMPORTANT: Must restrict to 'protein' to avoid including water oxygens (TIP3P uses atom name 'O')
        sele_str = 'protein and (backbone or name CA C N O)'
        sel = u.select_atoms(sele_str)
    elif args.selection == 'all':
        sele_str = 'not (resname WAT HOH TIP3 TIP4 NA CL SOL) and not element H'
        sel = u.select_atoms(sele_str)

    if args.selection == 'ligand':
        pass  # Already set above

    if len(sel) == 0:
        print(f"Error: No atoms matched selection '{sele_str}' — cannot cluster", file=sys.stderr)
        sys.exit(1)

    print(f"Selection: '{sele_str}' ({len(sel)} atoms)")

    # Apply PBC transformations to handle periodic boundary wrapping
    from utils import select_ligand_atoms, apply_pbc_transforms
    protein = u.select_atoms('protein')
    ligand = select_ligand_atoms(u)
    apply_pbc_transforms(u, protein, ligand)

    # Calculate RMSD matrix
    print("Calculating RMSD matrix...")

    # Determine frame indices to use (subsampling for large trajectories)
    total_frames = len(u.trajectory)

    # Calculate effective stride to stay within max_frames
    effective_stride = max(args.stride, total_frames // args.max_frames) if args.max_frames > 0 else args.stride
    sampled_frame_indices = list(range(0, total_frames, effective_stride))

    if len(sampled_frame_indices) > args.max_frames and args.max_frames > 0:
        sampled_frame_indices = sampled_frame_indices[:args.max_frames]

    print(f"Using {len(sampled_frame_indices)} frames (stride={effective_stride}, total={total_frames})")

    # Store coordinates for selected frames
    coords = []
    for frame_idx in sampled_frame_indices:
        u.trajectory[frame_idx]
        coords.append(sel.positions.copy())
    coords = np.array(coords)

    # Calculate pairwise RMSD with proper alignment (superposition)
    n = len(coords)
    rmsd_matrix = np.zeros((n, n))

    def calc_rmsd_aligned(coords1: Any, coords2: Any) -> float:
        """Calculate RMSD with optimal superposition (Kabsch algorithm)."""
        # Center both coordinate sets
        c1 = coords1 - coords1.mean(axis=0)
        c2 = coords2 - coords2.mean(axis=0)

        # Compute optimal rotation using SVD
        H = c1.T @ c2
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Apply rotation and calculate RMSD
        c1_rotated = (R @ c1.T).T
        diff = c1_rotated - c2
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        return rmsd

    for i in range(n):
        for j in range(i + 1, n):
            # RMSD with proper alignment
            rmsd = calc_rmsd_aligned(coords[i], coords[j])
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd

        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"  Calculated {i + 1}/{n} rows")

    # Perform clustering
    print(f"Clustering with {args.method}...")

    if args.method == 'kmeans':
        # For KMeans, we use coordinates directly
        coords_flat = coords.reshape(n, -1)
        model = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(coords_flat)
    elif args.method == 'dbscan':
        # DBSCAN uses the distance matrix
        from sklearn.metrics import pairwise_distances
        # Auto-determine eps as median RMSD
        upper_tri = rmsd_matrix[np.triu_indices(n, k=1)]
        eps = np.median(upper_tri) if len(upper_tri) > 0 else 1.0
        print(f"  DBSCAN eps: {eps:.2f}")
        model = DBSCAN(eps=eps, min_samples=max(2, n // 20), metric='precomputed')
        labels = model.fit_predict(rmsd_matrix)
    elif args.method == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=args.n_clusters, metric='precomputed', linkage='average')
        labels = model.fit_predict(rmsd_matrix)

    # Analyze clusters
    unique_labels = sorted(set(labels))
    if -1 in unique_labels:
        unique_labels.remove(-1)  # Remove noise cluster from DBSCAN

    clusters = []
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nCluster analysis ({len(unique_labels)} clusters):")

    for cluster_id in unique_labels:
        cluster_member_indices = np.where(labels == cluster_id)[0]
        frame_count = len(cluster_member_indices)
        population = 100.0 * frame_count / n

        # Find centroid (frame with minimum average RMSD to all other frames in cluster)
        if frame_count == 1:
            centroid_idx_in_sample = cluster_member_indices[0]
        else:
            cluster_rmsds = rmsd_matrix[np.ix_(cluster_member_indices, cluster_member_indices)]
            avg_rmsds = cluster_rmsds.mean(axis=1)
            best_idx = np.argmin(avg_rmsds)
            centroid_idx_in_sample = cluster_member_indices[best_idx]

        # Map back to original trajectory frame index
        centroid_frame = sampled_frame_indices[centroid_idx_in_sample]

        print(f"  Cluster {cluster_id}: {frame_count} frames ({population:.1f}%), centroid: frame {centroid_frame}")

        # Export centroid PDB
        centroid_pdb_path = os.path.join(args.output_dir, f'cluster_{cluster_id}_centroid.pdb')

        u.trajectory[centroid_frame]

        if args.strip_waters:
            atoms_to_write = u.select_atoms('not (resname WAT HOH TIP3 TIP4 NA CL SOL)')
        else:
            atoms_to_write = u.atoms

        atoms_to_write.write(centroid_pdb_path)

        clusters.append({
            'clusterId': int(cluster_id),
            'frameCount': int(frame_count),
            'population': float(population),
            'centroidFrame': int(centroid_frame),
            'centroidPdbPath': centroid_pdb_path,
        })

    # Handle noise points from DBSCAN
    noise_frames = np.where(labels == -1)[0]
    if len(noise_frames) > 0:
        print(f"  Noise: {len(noise_frames)} frames ({100.0 * len(noise_frames) / n:.1f}%)")

    # Save results
    results = {
        'clusters': clusters,
        'frameAssignments': labels.tolist(),
        'sampledFrameIndices': sampled_frame_indices,  # Maps sample index to original frame
        'method': args.method,
        'selection': args.selection,
        'nFramesSampled': n,
        'nFramesTotal': total_frames,
        'stride': effective_stride,
    }

    results_path = os.path.join(args.output_dir, 'clustering_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Create pooled multi-model PDB from centroid PDBs
    centroid_files = sorted(
        [c for c in clusters if c.get('centroidPdbPath') and os.path.exists(c['centroidPdbPath'])],
        key=lambda c: c['clusterId']
    )
    if centroid_files:
        pooled_path = os.path.join(args.output_dir, 'all_clusters.pdb')
        with open(pooled_path, 'w') as pf:
            for c in centroid_files:
                pf.write(f"MODEL     {c['clusterId']}\n")
                with open(c['centroidPdbPath'], 'r') as cf:
                    for line in cf:
                        if line.startswith('END'):
                            continue
                        pf.write(line)
                pf.write("ENDMDL\n")
            pf.write("END\n")
        print(f"Pooled {len(centroid_files)} centroids into: {pooled_path}")

    print(f"\nResults saved to: {results_path}")
    print("Done!")


if __name__ == '__main__':
    main()
