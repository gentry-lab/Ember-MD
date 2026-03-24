#!/usr/bin/env python3
"""
Prepare MD cluster centroids for fixed-pose rescoring.

Reads clustering_results.json, splits each centroid PDB into receptor + ligand,
reconstructs ligand chemistry from the original input ligand template, and writes
cluster_scores.json scaffolding for downstream Vina/CORDIAL rescoring.

Designed to be called from electron/main.ts after cluster_trajectory.py.

Output lines parsed by main.ts:
  PROGRESS:scoring_split:<N>
  CLUSTER_SCORES_JSON:<path>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

from ligand_torsion_utils import rebuild_ligand_from_centroid_pdb


def split_centroid_pdb(
    centroid_pdb: str,
    input_ligand_sdf: str,
    receptor_out: str,
    ligand_out: str,
) -> bool:
    """Split a centroid PDB into receptor PDB + ligand SDF."""
    import MDAnalysis as mda
    from rdkit import Chem

    from utils import select_ligand_atoms

    u = mda.Universe(centroid_pdb)
    protein = u.select_atoms('protein')
    ligand = select_ligand_atoms(u)

    if len(ligand) == 0:
        print(f"  Warning: No ligand atoms found in {centroid_pdb}", file=sys.stderr)
        return False

    if len(protein) == 0:
        print(f"  Warning: No protein atoms found in {centroid_pdb}", file=sys.stderr)
        return False

    protein.write(receptor_out)
    try:
        template_h = rebuild_ligand_from_centroid_pdb(
            centroid_pdb,
            input_ligand_sdf,
            include_hs=True,
        )
    except Exception as exc:
        print(f"  Warning: Could not rebuild cluster ligand from template: {exc}", file=sys.stderr)
        print("  Warning: Falling back to PDB-derived ligand coordinates.", file=sys.stderr)
        lig_pdb = ligand_out.replace('.sdf', '_tmp.pdb')
        ligand.write(lig_pdb)
        raw = Chem.MolFromPDBFile(lig_pdb, removeHs=False, sanitize=False)
        if raw is not None:
            try:
                Chem.SanitizeMol(raw)
            except Exception:
                pass
            writer = Chem.SDWriter(ligand_out)
            writer.write(raw)
            writer.close()
            os.remove(lig_pdb)
            return True
        os.remove(lig_pdb)
        return False

    writer = Chem.SDWriter(ligand_out)
    writer.write(template_h)
    writer.close()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Prepare MD cluster centroids for fixed-pose rescoring'
    )
    parser.add_argument('--clustering_dir', required=True,
                        help='Directory containing clustering_results.json and centroid PDBs')
    parser.add_argument('--input_ligand_sdf', required=True,
                        help='Original input ligand SDF used as the ligand chemistry template')
    parser.add_argument('--output_dir',
                        help='Output directory (default: same as clustering_dir)')
    args = parser.parse_args()

    clustering_dir = args.clustering_dir
    output_dir = args.output_dir or clustering_dir

    results_json = os.path.join(clustering_dir, 'clustering_results.json')
    if not os.path.exists(results_json):
        print(f"Error: {results_json} not found", file=sys.stderr)
        sys.exit(1)

    with open(results_json, 'r') as fh:
        clustering = json.load(fh)

    clusters = clustering['clusters']
    n_clusters = len(clusters)
    print(f"Loaded {n_clusters} clusters from {results_json}", file=sys.stderr)

    os.makedirs(output_dir, exist_ok=True)

    prepared_count = 0
    prepared_clusters = []
    print("Splitting centroids into receptor + ligand...", file=sys.stderr)

    for i, cluster in enumerate(clusters):
        cluster_id = cluster['clusterId']
        centroid_pdb = cluster.get('centroidPdbPath', '')

        entry: Dict[str, Any] = {
            'clusterId': cluster_id,
            'frameCount': cluster['frameCount'],
            'population': cluster['population'],
            'centroidFrame': cluster['centroidFrame'],
            'centroidPdbPath': centroid_pdb,
        }

        pct = int(100 * (i + 1) / n_clusters) if n_clusters > 0 else 100
        print(f"PROGRESS:scoring_split:{pct}", flush=True)

        if not centroid_pdb or not os.path.exists(centroid_pdb):
            print(f"  Warning: Centroid PDB missing for cluster {cluster_id}", file=sys.stderr)
            prepared_clusters.append(entry)
            continue

        rec_out = os.path.join(output_dir, f'cluster_{cluster_id}_receptor.pdb')
        lig_out = os.path.join(output_dir, f'cluster_{cluster_id}_ligand.sdf')

        if split_centroid_pdb(centroid_pdb, args.input_ligand_sdf, rec_out, lig_out):
            entry['receptorPdbPath'] = rec_out
            entry['ligandSdfPath'] = lig_out
            prepared_count += 1
        else:
            print(f"  Skipping cluster {cluster_id} (split failed)", file=sys.stderr)

        prepared_clusters.append(entry)

    if prepared_count == 0:
        print("Error: No clusters could be prepared for rescoring", file=sys.stderr)
        sys.exit(1)

    output_json = os.path.join(output_dir, 'cluster_scores.json')
    with open(output_json, 'w') as fh:
        json.dump({'clusters': prepared_clusters}, fh, indent=2)

    print(f"Prepared {prepared_count}/{n_clusters} centroids", file=sys.stderr)
    print(f"CLUSTER_SCORES_JSON:{output_json}", flush=True)
    print(f"Scaffold written to {output_json}", file=sys.stderr)


if __name__ == '__main__':
    main()
