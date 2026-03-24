#!/usr/bin/env python
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
Generate pocket surface PLY file for FragGen using MaSIF pipeline.

This script generates molecular surfaces with proper features:
- Electrostatic charges (from APBS)
- H-bond donor/acceptor features
- Hydrophobicity
- Surface normals
- Shape index (curvature)

Requires: Python 3.6 environment with pymesh2, and system tools (msms, apbs, pdb2pqr)

Usage:
    python generate_pocket_surface.py --pdb_file pocket.pdb --ligand_file ligand.sdf --output surface.ply
"""
import argparse
import os
import sys
import numpy as np
import tempfile
import shutil

# Check dependencies
try:
    import pymesh
    from Bio.PDB import PDBParser, PDBIO, Select, Selection, NeighborSearch
    from rdkit import Chem
    from scipy.spatial import distance, KDTree
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("This script requires Python 3.6 with pymesh2")
    print("Activate the surface_gen environment: conda activate surface_gen")
    sys.exit(1)

# Add MaSIF utilities to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils', 'masif'))

from compute_normal import compute_normal
from computeAPBS import computeAPBS
from computeCharges import computeCharges, assignChargesToNewMesh
from computeHydrophobicity import computeHydrophobicity
from computeMSMS import computeMSMS
from fixmesh import fix_mesh
from save_ply import save_ply

# Tool paths (Ubuntu system packages)
MSMS_BIN = "/usr/local/bin/msms"
APBS_BIN = "/usr/bin/apbs"
PDB2PQR_BIN = "/usr/bin/pdb2pqr"
MULTIVALUE_BIN = "/usr/lib/apbs/tools/bin/multivalue"


class SelectNeighbors(Select):
    """Select residues near the ligand."""
    def __init__(self, close_residues):
        self.close_residues = close_residues

    def accept_residue(self, residue):
        if residue in self.close_residues:
            # Check if residue has backbone atoms (is a real amino acid)
            atom_names = [a.get_name() for a in residue.get_unpacked_list()]
            if all(a in atom_names for a in ['N', 'CA', 'C', 'O']) or residue.resname == 'HOH':
                return True
        return False


def get_ligand_coords(ligand_file):
    """Get ligand atom coordinates from various file formats."""
    suffix = ligand_file.split('.')[-1].lower()

    if suffix == 'mol':
        mol = Chem.MolFromMolFile(ligand_file)
    elif suffix == 'mol2':
        mol = Chem.MolFromMol2File(ligand_file)
    elif suffix == 'sdf':
        suppl = Chem.SDMolSupplier(ligand_file, sanitize=False)
        mols = [m for m in suppl if m]
        if not mols:
            raise ValueError(f"No valid molecules in {ligand_file}")
        mol = mols[0]
    elif suffix == 'pdb':
        # For PDB ligands, use BioPython
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('ligand', ligand_file)
        atoms = list(structure.get_atoms())
        return np.array([a.get_coord() for a in atoms])
    else:
        raise ValueError(f"Unknown ligand format: {suffix}")

    if mol is None:
        raise ValueError(f"Failed to parse ligand file: {ligand_file}")

    return mol.GetConformers()[0].GetPositions()


def generate_surface(pdb_file, ligand_file, output_ply, dist_threshold=8.0, mesh_res=1.5):
    """
    Generate MaSIF-style molecular surface with full features.

    Args:
        pdb_file: Path to protein/pocket PDB file
        ligand_file: Path to ligand file (SDF, MOL, MOL2, or PDB)
        output_ply: Output PLY file path
        dist_threshold: Distance cutoff for pocket definition (default 8.0 A)
        mesh_res: Mesh resolution (default 1.5)
    """
    # Verify tools exist
    for tool, path in [("msms", MSMS_BIN), ("apbs", APBS_BIN), ("pdb2pqr", PDB2PQR_BIN)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{tool} not found at {path}. Install with: sudo apt install apbs pdb2pqr")

    # Create working directory
    workdir = tempfile.mkdtemp(prefix="surface_gen_")
    print(f"Working directory: {workdir}")

    try:
        # Get ligand coordinates
        print("Reading ligand coordinates...")
        lig_coords = get_ligand_coords(ligand_file)

        # Parse protein structure
        print("Parsing protein structure...")
        parser = PDBParser(QUIET=True)
        structures = parser.get_structure('protein', pdb_file)
        structure = structures[0]

        # Find residues near ligand
        atoms = Selection.unfold_entities(structure, 'A')
        ns = NeighborSearch(atoms)

        close_residues = []
        for coord in lig_coords:
            close_residues.extend(ns.search(coord, dist_threshold + 5, level='R'))
        close_residues = Selection.uniqueify(close_residues)

        # Save pocket PDB
        protname = os.path.basename(pdb_file).replace(".pdb", "")
        pocket_pdb = os.path.join(workdir, f"{protname}_pocket_{dist_threshold+5}.pdb")

        pdbio = PDBIO()
        pdbio.set_structure(structure)
        pdbio.save(pocket_pdb, SelectNeighbors(close_residues))

        print(f"Extracted pocket with {len(close_residues)} residues")

        # Find closest atom to ligand center (for MSMS cavity detection)
        structures = parser.get_structure('pocket', pocket_pdb)
        structure = structures[0]
        atoms = Selection.unfold_entities(structure, 'A')

        lig_center = lig_coords.mean(axis=0)
        dists = [distance.euclidean(lig_center, a.get_coord()) for a in atoms]
        atom_idx = np.argmin(dists)

        # Compute molecular surface with MSMS
        print("Computing molecular surface with MSMS...")
        vertices1, faces1, normals1, names1, areas1 = computeMSMS(
            pocket_pdb,
            protonate=True
        )

        print(f"Initial surface: {len(vertices1)} vertices, {len(faces1)} faces")

        # Filter to pocket region
        kdt = KDTree(lig_coords)
        d, _ = kdt.query(vertices1)
        iface_v = np.where(d <= dist_threshold)[0]
        faces_to_keep = [idx for idx, face in enumerate(faces1) if all(v in iface_v for v in face)]

        print(f"Filtered to {len(faces_to_keep)} faces near ligand")

        # Compute features on original mesh
        print("Computing H-bond features...")
        vertex_hbond = computeCharges(pdb_file.replace(".pdb", ""), vertices1, names1)

        print("Computing hydrophobicity...")
        vertex_hphob = computeHydrophobicity(names1)

        # Create and fix mesh
        print("Processing mesh with pymesh...")
        mesh = pymesh.form_mesh(vertices1, faces1)
        mesh = pymesh.submesh(mesh, faces_to_keep, 0)

        # Suppress pymesh output
        from IPython.utils import io
        with io.capture_output():
            regular_mesh = fix_mesh(mesh, mesh_res)

        regular_mesh, _ = pymesh.remove_degenerated_triangles(regular_mesh)

        print(f"Final mesh: {len(regular_mesh.vertices)} vertices, {len(regular_mesh.faces)} faces")

        # Compute normals (function expects (n,3) arrays and transposes internally)
        print("Computing surface normals...")
        vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)

        # Interpolate features to new mesh
        print("Interpolating features to regularized mesh...")
        interp_opts = {"feature_interpolation": True}
        vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hbond, interp_opts)
        vertex_hphob = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hphob, interp_opts)

        # Compute electrostatic potential with APBS
        # APBS function expects PDB to be in the working directory
        print("Computing electrostatic potential with APBS...")
        tmp_file_base = os.path.join(workdir, "apbs_output")
        vertex_charges = computeAPBS(
            regular_mesh.vertices,
            pocket_pdb,  # pocket_pdb is already in workdir
            tmp_file_base
        )

        # Compute curvature and shape index
        print("Computing curvature features...")
        regular_mesh.add_attribute("vertex_mean_curvature")
        H = regular_mesh.get_attribute("vertex_mean_curvature")
        regular_mesh.add_attribute("vertex_gaussian_curvature")
        K = regular_mesh.get_attribute("vertex_gaussian_curvature")

        elem = np.square(H) - K
        elem[elem < 0] = 1e-8  # Handle numerical issues
        k1 = H + np.sqrt(elem)
        k2 = H - np.sqrt(elem)

        # Shape index
        si = (k1 + k2) / (k1 - k2 + 1e-8)
        si = np.arctan(si) * (2 / np.pi)

        # Save PLY file
        print(f"Saving surface to {output_ply}...")
        os.makedirs(os.path.dirname(output_ply) or '.', exist_ok=True)

        save_ply(
            output_ply,
            regular_mesh.vertices,
            regular_mesh.faces,
            normals=vertex_normal,
            charges=vertex_charges,
            normalize_charges=True,
            hbond=vertex_hbond,
            hphob=vertex_hphob,
            si=si
        )

        print(f"Successfully generated surface with {len(regular_mesh.vertices)} vertices")
        print(f"Features: charge, hbond, hphob, normals, shape_index")

    finally:
        # Cleanup
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description='Generate MaSIF-style pocket surface for FragGen',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_pocket_surface.py --pdb_file pocket.pdb --ligand_file ligand.sdf --output surface.ply
    python generate_pocket_surface.py --pdb_file 1abc.pdb --ligand_file lig.mol2 --output out.ply --dist 10.0

Requirements:
    - Python 3.6 environment with pymesh2: conda activate surface_gen
    - System tools: sudo apt install apbs pdb2pqr
    - MSMS: download from https://ccsb.scripps.edu/msms/
        """
    )
    parser.add_argument('--pdb_file', required=True, help='Protein/pocket PDB file')
    parser.add_argument('--ligand_file', '--ligand_pdb', required=True, dest='ligand_file',
                        help='Ligand file (SDF, MOL, MOL2, or PDB)')
    parser.add_argument('--output', required=True, help='Output PLY file')
    parser.add_argument('--dist', type=float, default=8.0,
                        help='Distance threshold for pocket (default: 8.0 A)')
    parser.add_argument('--resolution', type=float, default=1.5,
                        help='Mesh resolution (default: 1.5)')

    args = parser.parse_args()

    if not os.path.exists(args.pdb_file):
        print(f"Error: PDB file not found: {args.pdb_file}")
        sys.exit(1)

    if not os.path.exists(args.ligand_file):
        print(f"Error: Ligand file not found: {args.ligand_file}")
        sys.exit(1)

    generate_surface(
        args.pdb_file,
        args.ligand_file,
        args.output,
        dist_threshold=args.dist,
        mesh_res=args.resolution
    )


if __name__ == '__main__':
    main()
