#!/usr/bin/env python3
"""Enrich torsions_results.json with cluster-centroid torsion values."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from ligand_torsion_utils import (
    build_torsion_identity_bundle,
    compute_dihedral_angles_for_mol,
    rebuild_ligand_from_centroid_pdb,
)
from utils import load_sdf


def load_cluster_ligand_mol(
    cluster_id: int,
    centroid_pdb_path: str,
    ligand_sdf_path: str,
    scored_clusters_dir: str | None,
) -> Any:
    if scored_clusters_dir:
        candidate = os.path.join(scored_clusters_dir, f'cluster_{cluster_id}_ligand.sdf')
        if os.path.exists(candidate):
            mol = load_sdf(candidate, remove_hs=True)
            if mol is not None:
                return mol

    return rebuild_ligand_from_centroid_pdb(
        centroid_pdb_path,
        ligand_sdf_path,
        include_hs=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Enrich torsion results with cluster centroid values')
    parser.add_argument('--torsions_json', required=True, help='Path to torsions_results.json')
    parser.add_argument('--clustering_dir', required=True, help='Directory containing clustering_results.json')
    parser.add_argument('--ligand_sdf', required=True, help='Canonical ligand template SDF')
    parser.add_argument('--scored_clusters_dir', default=None, help='Optional cluster ligand SDF directory')
    args = parser.parse_args()

    if not os.path.exists(args.torsions_json):
        print(f'Warning: torsion results JSON not found: {args.torsions_json}', file=sys.stderr)
        return

    clustering_path = os.path.join(args.clustering_dir, 'clustering_results.json')
    if not os.path.exists(clustering_path):
        print(f'Warning: clustering results not found: {clustering_path}', file=sys.stderr)
        return

    with open(args.torsions_json, 'r') as fh:
        bundle = json.load(fh)

    torsion_rows = bundle.get('data', {}).get('torsions', [])
    if not torsion_rows:
        return

    identity = build_torsion_identity_bundle(args.ligand_sdf)
    canonical_torsions: List[Dict[str, Any]] = identity['torsions']
    if [row.get('torsionId') for row in torsion_rows] != [row.get('torsionId') for row in canonical_torsions]:
        print('Warning: cluster torsion ids do not match trajectory torsion ids; skipping cluster enrichment', file=sys.stderr)
        return

    with open(clustering_path, 'r') as fh:
        clustering = json.load(fh)

    cluster_rows = []
    for cluster in clustering.get('clusters', []):
        cluster_id = cluster.get('clusterId')
        centroid_pdb_path = cluster.get('centroidPdbPath')
        if cluster_id is None or not centroid_pdb_path or not os.path.exists(centroid_pdb_path):
            continue

        try:
            ligand_mol = load_cluster_ligand_mol(
                int(cluster_id),
                centroid_pdb_path,
                args.ligand_sdf,
                args.scored_clusters_dir,
            )
            angles = compute_dihedral_angles_for_mol(ligand_mol, canonical_torsions)
        except Exception as exc:
            print(f'Warning: Failed cluster torsion analysis for cluster {cluster_id}: {exc}', file=sys.stderr)
            continue

        cluster_rows.append({
            'clusterId': int(cluster_id),
            'centroidFrame': int(cluster.get('centroidFrame', 0)),
            'population': float(cluster.get('population', 0.0)),
            'angles': [round(float(v), 2) for v in angles],
        })

    values_by_torsion = {row['torsionId']: [] for row in torsion_rows}
    for cluster_row in cluster_rows:
        for torsion_idx, torsion_row in enumerate(torsion_rows):
            values_by_torsion[torsion_row['torsionId']].append({
                'clusterId': cluster_row['clusterId'],
                'centroidFrame': cluster_row['centroidFrame'],
                'population': cluster_row['population'],
                'angle': cluster_row['angles'][torsion_idx],
            })

    for torsion_row in torsion_rows:
        torsion_row['clusterValues'] = values_by_torsion.get(torsion_row['torsionId'], [])

    with open(args.torsions_json, 'w') as fh:
        json.dump(bundle, fh, indent=2)

    print(f'Cluster torsions saved to: {args.torsions_json}')


if __name__ == '__main__':
    main()
