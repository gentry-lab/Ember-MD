#!/usr/bin/env python3
"""
Score MD cluster centroids with Vina rescore and xTB strain energy.

Reads clustering_results.json, splits each centroid PDB into receptor + ligand,
computes xTB strain energies and Vina score_only rescoring.

Designed to be called from electron/main.ts after cluster_trajectory.py.
CORDIAL scoring is handled separately by main.ts using the split files.

Output lines parsed by main.ts:
  PROGRESS:scoring_split:<N>
  PROGRESS:scoring_xtb:<N>
  PROGRESS:scoring_vina:<N>
  CLUSTER_SCORES_JSON:<path>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Centroid PDB splitting
# ---------------------------------------------------------------------------

def split_centroid_pdb(
    centroid_pdb: str,
    input_ligand_sdf: str,
    receptor_out: str,
    ligand_out: str,
) -> bool:
    """Split a centroid PDB into receptor PDB + ligand SDF.

    Uses MDAnalysis to separate protein from ligand atoms, and RDKit
    to reconstruct the ligand SDF with proper bond orders from the
    original input ligand template.

    Returns True on success.
    """
    import MDAnalysis as mda
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem

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

    # Reconstruct ligand SDF from template with centroid coordinates
    template_mol = _load_sdf(input_ligand_sdf)
    if template_mol is None:
        print(f"  Warning: Could not load template ligand from {input_ligand_sdf}",
              file=sys.stderr)
        return False

    # Add Hs to match what MDAnalysis sees
    template_h = Chem.AddHs(template_mol, addCoords=True)

    # Get heavy atom positions from the centroid
    ligand_heavy = u.select_atoms(
        f'(resname {" ".join(set(a.resname for a in ligand))}) and not element H'
    )
    centroid_heavy_coords = ligand_heavy.positions  # (N_heavy, 3)

    # Get heavy atoms from template
    template_heavy_indices = [
        i for i in range(template_h.GetNumAtoms())
        if template_h.GetAtomWithIdx(i).GetAtomicNum() > 1
    ]

    if len(template_heavy_indices) != len(centroid_heavy_coords):
        print(f"  Warning: Heavy atom count mismatch "
              f"(template={len(template_heavy_indices)}, centroid={len(centroid_heavy_coords)}). "
              f"Trying without H template.", file=sys.stderr)
        # Fallback: use template without explicit H
        template_noH = Chem.RemoveHs(template_mol)
        if template_noH.GetNumAtoms() == len(centroid_heavy_coords):
            conf = template_noH.GetConformer()
            for i in range(len(centroid_heavy_coords)):
                conf.SetAtomPosition(i, centroid_heavy_coords[i].tolist())
            # Re-add H with new coordinates
            template_h = Chem.AddHs(template_noH, addCoords=True)
            AllChem.ConstrainedEmbed(template_h, template_noH)
        else:
            print(f"  Warning: Cannot map centroid atoms to template. "
                  f"Writing PDB-derived coordinates.", file=sys.stderr)
            # Last resort: write ligand atoms as PDB, convert
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

    else:
        # Map centroid heavy atom coords onto template
        conf = template_h.GetConformer()
        for idx, template_idx in enumerate(template_heavy_indices):
            coord = centroid_heavy_coords[idx].tolist()
            conf.SetAtomPosition(template_idx, coord)

        # Recompute H positions based on new heavy atom positions
        try:
            AllChem.ConstrainedEmbed(
                template_h, template_h,
                randomseed=42,
            )
        except Exception:
            # ConstrainedEmbed can fail; the heavy atom coords are correct
            # and H positions from AddHs are reasonable defaults
            pass

    writer = Chem.SDWriter(ligand_out)
    writer.write(template_h)
    writer.close()
    return True


def _load_sdf(path: str):
    """Load the first molecule from an SDF (or .sdf.gz) file."""
    from score_xtb_strain import _load_sdf as _load
    return _load(path)


# ---------------------------------------------------------------------------
# xTB strain scoring
# ---------------------------------------------------------------------------

def compute_xtb_strain(
    xtb_binary: str,
    ligand_sdfs: List[str],
    input_ligand_sdf: str,
    output_dir: str,
) -> Dict[int, Dict[str, float]]:
    """Compute xTB strain energy for each cluster ligand.

    1. Optimize the free ligand once to get E_free_min
    2. Single-point each cluster's ligand to get E_pose
    3. Strain = E_pose - E_free_min (kcal/mol)

    Returns dict mapping cluster_id to {xtbStrainKcal, xtbPoseEnergy, xtbFreeMinEnergy}.
    """
    from score_xtb_strain import optimize, single_point, HARTREE_TO_KCAL

    results: Dict[int, Dict[str, float]] = {}

    # Step 1: optimize the free ligand
    opt_sdf = os.path.join(output_dir, 'free_ligand_optimized.sdf')
    print("  Optimizing free ligand with GFN2-xTB + ALPB water...", file=sys.stderr)
    try:
        e_free, opt_path = optimize(xtb_binary, input_ligand_sdf, opt_sdf, solvent='water')
        print(f"  Free ligand energy: {e_free:.6f} Eh ({e_free * HARTREE_TO_KCAL:.2f} kcal/mol)",
              file=sys.stderr)
    except Exception as e:
        print(f"  Error optimizing free ligand: {e}", file=sys.stderr)
        return results

    # Step 2: single-point each cluster ligand
    n = len(ligand_sdfs)
    for i, (cluster_id, sdf_path) in enumerate(ligand_sdfs):
        pct = int(100 * (i + 1) / n)
        print(f"PROGRESS:scoring_xtb:{pct}", flush=True)

        try:
            e_pose = single_point(xtb_binary, sdf_path, solvent='water')
            strain_kcal = (e_pose - e_free) * HARTREE_TO_KCAL
            results[cluster_id] = {
                'xtbStrainKcal': round(strain_kcal, 2),
                'xtbPoseEnergy': e_pose,
                'xtbFreeMinEnergy': e_free,
            }
            print(f"  Cluster {cluster_id}: strain = {strain_kcal:.2f} kcal/mol", file=sys.stderr)
        except Exception as e:
            print(f"  Error scoring cluster {cluster_id}: {e}", file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# Vina rescoring
# ---------------------------------------------------------------------------

def compute_vina_rescores(
    receptor_pdbs: List[tuple],
    ligand_sdfs: List[tuple],
    reference_ligand_sdf: str,
    autobox_add: float = 4.0,
) -> Dict[int, float]:
    """Rescore each cluster with Vina score_only.

    Returns dict mapping cluster_id to Vina affinity (kcal/mol).
    """
    try:
        from vina import Vina
    except ImportError:
        print("  Warning: Vina not available, skipping Vina rescoring", file=sys.stderr)
        return {}

    from rdkit import Chem

    results: Dict[int, float] = {}

    # Get reference ligand center and size for autobox
    ref_mol = _load_sdf(reference_ligand_sdf)
    if ref_mol is None:
        print("  Warning: Could not load reference ligand for autobox", file=sys.stderr)
        return results

    ref_conf = ref_mol.GetConformer()
    ref_coords = [ref_conf.GetAtomPosition(i) for i in range(ref_mol.GetNumAtoms())]
    center = [
        sum(c.x for c in ref_coords) / len(ref_coords),
        sum(c.y for c in ref_coords) / len(ref_coords),
        sum(c.z for c in ref_coords) / len(ref_coords),
    ]
    extent = [
        max(c.x for c in ref_coords) - min(c.x for c in ref_coords) + 2 * autobox_add,
        max(c.y for c in ref_coords) - min(c.y for c in ref_coords) + 2 * autobox_add,
        max(c.z for c in ref_coords) - min(c.z for c in ref_coords) + 2 * autobox_add,
    ]

    n = len(receptor_pdbs)
    for i in range(n):
        cluster_id_r, rec_pdb = receptor_pdbs[i]
        cluster_id_l, lig_sdf = ligand_sdfs[i]
        assert cluster_id_r == cluster_id_l

        pct = int(100 * (i + 1) / n)
        print(f"PROGRESS:scoring_vina:{pct}", flush=True)

        try:
            # Convert ligand SDF to PDBQT for Vina
            mol = _load_sdf(lig_sdf)
            if mol is None:
                continue

            # Use Meeko for PDBQT prep
            from meeko import MoleculePreparation, PDBQTWriterLegacy
            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol)
            pdbqt_str = PDBQTWriterLegacy.write_string(mol_setups[0])[0]

            # Write temp PDBQT
            lig_pdbqt = lig_sdf.replace('.sdf', '.pdbqt')
            with open(lig_pdbqt, 'w') as f:
                f.write(pdbqt_str)

            v = Vina(sf_name='vina', verbosity=0)
            v.set_receptor(rec_pdb)
            v.set_ligand_from_file(lig_pdbqt)
            v.compute_vina_maps(center=center, box_size=extent)
            score = v.score()
            affinity = float(score[0])
            results[cluster_id_r] = round(affinity, 2)
            print(f"  Cluster {cluster_id_r}: Vina = {affinity:.2f} kcal/mol", file=sys.stderr)

        except Exception as e:
            print(f"  Error scoring cluster {cluster_id_r} with Vina: {e}", file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Score MD cluster centroids with Vina and xTB'
    )
    parser.add_argument('--clustering_dir', required=True,
                        help='Directory containing clustering_results.json and centroid PDBs')
    parser.add_argument('--input_ligand_sdf', required=True,
                        help='Original input ligand SDF (for bond orders and free-ligand optimization)')
    parser.add_argument('--input_receptor_pdb',
                        help='Original input receptor PDB (optional, for Vina autobox fallback)')
    parser.add_argument('--xtb_binary',
                        help='Path to xtb executable (skip xTB if not provided)')
    parser.add_argument('--autobox_add', type=float, default=4.0,
                        help='Autobox padding in Angstroms (default: 4.0)')
    parser.add_argument('--skip_vina', action='store_true', help='Skip Vina rescoring')
    parser.add_argument('--skip_xtb', action='store_true', help='Skip xTB strain scoring')
    parser.add_argument('--output_dir',
                        help='Output directory (default: same as clustering_dir)')
    args = parser.parse_args()

    clustering_dir = args.clustering_dir
    output_dir = args.output_dir or clustering_dir

    # Load clustering results
    results_json = os.path.join(clustering_dir, 'clustering_results.json')
    if not os.path.exists(results_json):
        print(f"Error: {results_json} not found", file=sys.stderr)
        sys.exit(1)

    with open(results_json) as f:
        clustering = json.load(f)

    clusters = clustering['clusters']
    n_clusters = len(clusters)
    print(f"Loaded {n_clusters} clusters from {results_json}", file=sys.stderr)

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Split each centroid PDB into receptor + ligand
    print("Splitting centroids into receptor + ligand...", file=sys.stderr)
    receptor_pdbs: List[tuple] = []  # (cluster_id, path)
    ligand_sdfs: List[tuple] = []    # (cluster_id, path)

    for i, cluster in enumerate(clusters):
        cluster_id = cluster['clusterId']
        centroid_pdb = cluster.get('centroidPdbPath', '')

        if not centroid_pdb or not os.path.exists(centroid_pdb):
            print(f"  Warning: Centroid PDB missing for cluster {cluster_id}", file=sys.stderr)
            continue

        rec_out = os.path.join(output_dir, f'cluster_{cluster_id}_receptor.pdb')
        lig_out = os.path.join(output_dir, f'cluster_{cluster_id}_ligand.sdf')

        pct = int(100 * (i + 1) / n_clusters)
        print(f"PROGRESS:scoring_split:{pct}", flush=True)

        ok = split_centroid_pdb(centroid_pdb, args.input_ligand_sdf, rec_out, lig_out)
        if ok:
            receptor_pdbs.append((cluster_id, rec_out))
            ligand_sdfs.append((cluster_id, lig_out))
        else:
            print(f"  Skipping cluster {cluster_id} (split failed)", file=sys.stderr)

    if not ligand_sdfs:
        print("Error: No clusters could be split", file=sys.stderr)
        sys.exit(1)

    print(f"Split {len(ligand_sdfs)}/{n_clusters} centroids", file=sys.stderr)

    # Step 2: xTB strain energy
    xtb_results: Dict[int, Dict[str, float]] = {}
    if not args.skip_xtb and args.xtb_binary:
        print("Computing xTB strain energies...", file=sys.stderr)
        xtb_results = compute_xtb_strain(
            args.xtb_binary, ligand_sdfs, args.input_ligand_sdf, output_dir
        )
    else:
        print("Skipping xTB strain scoring", file=sys.stderr)

    # Step 3: Vina rescoring
    vina_results: Dict[int, float] = {}
    if not args.skip_vina:
        print("Computing Vina rescores...", file=sys.stderr)
        vina_results = compute_vina_rescores(
            receptor_pdbs, ligand_sdfs, args.input_ligand_sdf, args.autobox_add
        )
    else:
        print("Skipping Vina rescoring", file=sys.stderr)

    # Step 4: Merge and write results
    scored_clusters = []
    for cluster in clusters:
        cid = cluster['clusterId']
        entry: Dict[str, Any] = {
            'clusterId': cid,
            'frameCount': cluster['frameCount'],
            'population': cluster['population'],
            'centroidFrame': cluster['centroidFrame'],
            'centroidPdbPath': cluster.get('centroidPdbPath', ''),
        }

        # Add receptor/ligand paths if split succeeded
        rec_path = os.path.join(output_dir, f'cluster_{cid}_receptor.pdb')
        lig_path = os.path.join(output_dir, f'cluster_{cid}_ligand.sdf')
        if os.path.exists(rec_path):
            entry['receptorPdbPath'] = rec_path
        if os.path.exists(lig_path):
            entry['ligandSdfPath'] = lig_path

        # xTB scores
        if cid in xtb_results:
            entry.update(xtb_results[cid])

        # Vina scores
        if cid in vina_results:
            entry['vinaRescore'] = vina_results[cid]

        scored_clusters.append(entry)

    output_json = os.path.join(output_dir, 'cluster_scores.json')
    with open(output_json, 'w') as f:
        json.dump({'clusters': scored_clusters}, f, indent=2)

    print(f"CLUSTER_SCORES_JSON:{output_json}", flush=True)
    print(f"Scores written to {output_json}", file=sys.stderr)
    print("Done!", file=sys.stderr)


if __name__ == '__main__':
    main()
