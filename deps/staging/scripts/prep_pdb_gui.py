#!/usr/bin/env python
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
Prep PDB for FragGen GUI
Extracts pocket and ligand from a protein-ligand complex PDB file.
"""
import argparse
import os
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select, NeighborSearch
from Bio.PDB.Polypeptide import is_aa
import numpy as np


class LigandSelect(Select):
    """Select only ligand atoms."""
    def __init__(self, ligand_residues):
        self.ligand_residues = set(ligand_residues)

    def accept_residue(self, residue):
        return residue in self.ligand_residues


class PocketSelect(Select):
    """Select pocket residues within radius of ligand."""
    def __init__(self, pocket_residues):
        self.pocket_residues = set(pocket_residues)

    def accept_residue(self, residue):
        return residue in self.pocket_residues


def get_ligand_residues(structure, ligand_name=None):
    """Find ligand residues (non-protein, non-water, non-ion)."""
    ligands = []
    excluded = {'HOH', 'WAT', 'NA', 'CL', 'MG', 'ZN', 'CA', 'FE', 'K', 'MN', 'CO', 'NI', 'CU'}

    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip()
                # Skip water, ions, and standard amino acids
                if resname in excluded:
                    continue
                if is_aa(residue, standard=True):
                    continue
                # If ligand name specified, filter by it
                if ligand_name and resname != ligand_name:
                    continue
                ligands.append(residue)

    return ligands


def get_pocket_residues(structure, ligand_residues, radius=10.0):
    """Get protein residues within radius of ligand atoms."""
    # Get all ligand atoms
    ligand_atoms = []
    for res in ligand_residues:
        ligand_atoms.extend(res.get_atoms())

    if not ligand_atoms:
        return []

    # Get ligand center
    ligand_coords = np.array([atom.get_coord() for atom in ligand_atoms])
    ligand_center = ligand_coords.mean(axis=0)

    # Get all protein atoms
    protein_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=True):
                    protein_atoms.extend(residue.get_atoms())

    if not protein_atoms:
        return []

    # Find residues within radius
    ns = NeighborSearch(protein_atoms)
    nearby_atoms = ns.search(ligand_center, radius, level='R')

    # Also include residues close to any ligand atom
    pocket_residues = set(nearby_atoms)
    for atom in ligand_atoms:
        nearby = ns.search(atom.get_coord(), radius, level='R')
        pocket_residues.update(nearby)

    return list(pocket_residues)


def main():
    parser = argparse.ArgumentParser(description='Prepare PDB for FragGen')
    parser.add_argument('--input_pdb', required=True, help='Input PDB file')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--ligand_name', default=None, help='Ligand residue name (auto-detect if not specified)')
    parser.add_argument('--pocket_radius', type=float, default=10.0, help='Pocket radius in Angstroms')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse structure (CIF or PDB)
    if args.input_pdb.lower().endswith('.cif'):
        parser_pdb = MMCIFParser(QUIET=True)
    else:
        parser_pdb = PDBParser(QUIET=True)
    structure = parser_pdb.get_structure('complex', args.input_pdb)

    # Find ligand
    ligand_residues = get_ligand_residues(structure, args.ligand_name)
    if not ligand_residues:
        print(f"ERROR: No ligand found in {args.input_pdb}")
        exit(1)

    print(f"Found {len(ligand_residues)} ligand residue(s): {[r.get_resname() for r in ligand_residues]}")

    # Get pocket
    pocket_residues = get_pocket_residues(structure, ligand_residues, args.pocket_radius)
    if not pocket_residues:
        print(f"ERROR: No pocket residues found within {args.pocket_radius}A of ligand")
        exit(1)

    print(f"Found {len(pocket_residues)} pocket residues within {args.pocket_radius}A")

    # Save ligand
    io = PDBIO()
    io.set_structure(structure)
    ligand_path = os.path.join(args.output_dir, 'ligand.pdb')
    io.save(ligand_path, LigandSelect(ligand_residues))
    print(f"Saved ligand to {ligand_path}")

    # Save pocket
    pocket_path = os.path.join(args.output_dir, 'pocket.pdb')
    io.save(pocket_path, PocketSelect(pocket_residues))
    print(f"Saved pocket to {pocket_path}")


if __name__ == '__main__':
    main()
