#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
Prepare a chemically consistent docking complex from a prepared receptor and X-ray ligand.

Flow:
  1. Protonate the X-ray ligand using the same protonation machinery as docking.
  2. Transfer exact crystal heavy-atom coordinates onto the chosen protonated ligand.
  3. Optimize hydrogen positions in the prepared receptor pocket with all heavy atoms fixed.
  4. Write a canonical refined receptor, prepared reference ligand, and a manifest.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Tuple

from utils import add_gbsa_obc2_force, load_sdf

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdFMCS
    from rdkit.Geometry import Point3D
except ImportError:
    print("ERROR: RDKit not installed", file=sys.stderr)
    sys.exit(1)

try:
    import openmm
    from openmm import app as omm_app
    from openmm import unit as omm_unit
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False

from enumerate_protonation import HAS_MOLSCRUB, process_ligand
from receptor_protonation import load_receptor_prep_metadata

if HAS_MOLSCRUB:
    from enumerate_protonation import Scrub


WATER_RESNAMES = {"HOH", "WAT", "H2O", "TIP3", "TIP4", "OPC"}


def load_first_sdf(sdf_path: str) -> Any:
    mol = load_sdf(sdf_path)
    if mol is None:
        raise RuntimeError(f"Failed to read molecule from {sdf_path}")
    return mol


def heavy_atom_smiles(mol: Any) -> str:
    return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)


def same_heavy_graph(mol_a: Any, mol_b: Any) -> bool:
    no_h_a = Chem.RemoveHs(mol_a)
    no_h_b = Chem.RemoveHs(mol_b)
    if no_h_a.GetNumAtoms() != no_h_b.GetNumAtoms():
        return False
    mcs = rdFMCS.FindMCS([no_h_a, no_h_b], completeRingsOnly=True, ringMatchesRingOnly=True)
    return mcs.numAtoms == no_h_a.GetNumAtoms() == no_h_b.GetNumAtoms()


def choose_reference_variant(
    raw_xray_path: str,
    protonation_dir: str,
    ph_min: float,
    ph_max: float,
    protonate_reference: bool,
) -> Tuple[str, Any, dict]:
    raw_mol = load_first_sdf(raw_xray_path)
    raw_smiles = heavy_atom_smiles(raw_mol)

    if not protonate_reference:
        reference_mol = Chem.AddHs(Chem.Mol(raw_mol), addCoords=True)
        metadata = {
            "selected_protonated_path": raw_xray_path,
            "candidate_count": 1,
            "candidate_paths": [raw_xray_path],
            "formal_charge": Chem.GetFormalCharge(reference_mol),
            "raw_smiles": raw_smiles,
            "selected_smiles": heavy_atom_smiles(reference_mol),
            "protonation_method": "disabled-addhs",
            "ph_min": ph_min,
            "ph_max": ph_max,
            "protonation_enabled": False,
        }
        return raw_xray_path, reference_mol, metadata

    os.makedirs(protonation_dir, exist_ok=True)
    scrub = Scrub(ph_low=ph_min, ph_high=ph_max) if HAS_MOLSCRUB else None

    protonated_results = process_ligand(raw_xray_path, protonation_dir, scrub)
    if not protonated_results:
        raise RuntimeError("No protonated variants were generated for the X-ray ligand")

    selected_path = None
    selected_mol = None
    candidate_paths = []
    for candidate_path, _parent_name in protonated_results:
        candidate_paths.append(candidate_path)
        candidate_mol = load_first_sdf(candidate_path)
        if same_heavy_graph(raw_mol, candidate_mol):
            selected_path = candidate_path
            selected_mol = candidate_mol
            break

    if selected_path is None or selected_mol is None:
        raise RuntimeError("No protonated X-ray variant matched the raw ligand identity")

    metadata = {
        "selected_protonated_path": selected_path,
        "candidate_count": len(candidate_paths),
        "candidate_paths": candidate_paths,
        "formal_charge": Chem.GetFormalCharge(selected_mol),
        "raw_smiles": raw_smiles,
        "selected_smiles": heavy_atom_smiles(selected_mol),
        "protonation_method": "molscrub" if HAS_MOLSCRUB else "fallback-original",
        "ph_min": ph_min,
        "ph_max": ph_max,
        "protonation_enabled": True,
    }
    return selected_path, selected_mol, metadata


def map_exact_crystal_coordinates(xray_mol: Any, protonated_mol: Any) -> Tuple[Any, list[Point3D], list[int]]:
    xray_no_h = Chem.RemoveHs(xray_mol)
    prot_no_h = Chem.RemoveHs(protonated_mol)
    mcs = rdFMCS.FindMCS([xray_no_h, prot_no_h], completeRingsOnly=True, ringMatchesRingOnly=True)
    if mcs.numAtoms == 0:
        raise RuntimeError("No MCS found between crystal and protonated ligand")

    core = Chem.MolFromSmarts(mcs.smartsString)
    xray_match = xray_no_h.GetSubstructMatch(core)
    prot_match = prot_no_h.GetSubstructMatch(core)
    if not xray_match or not prot_match:
        raise RuntimeError("MCS substructure match failed between crystal and protonated ligand")

    heavy_to_orig = []
    for idx, atom in enumerate(protonated_mol.GetAtoms()):
        if atom.GetAtomicNum() != 1:
            heavy_to_orig.append(idx)

    xray_conf = xray_mol.GetConformer()
    coord_map = {}
    original_positions: list[Point3D] = []
    heavy_indices: list[int] = []
    for prot_no_h_idx, xray_no_h_idx in zip(prot_match, xray_match):
        prot_orig_idx = heavy_to_orig[prot_no_h_idx]
        xray_pos = xray_conf.GetAtomPosition(xray_no_h_idx)
        coord_map[prot_orig_idx] = xray_pos
        original_positions.append(Point3D(xray_pos.x, xray_pos.y, xray_pos.z))
        heavy_indices.append(prot_orig_idx)

    embedded = Chem.RWMol(protonated_mol)
    embedded.RemoveAllConformers()
    status = AllChem.EmbedMolecule(
        embedded,
        coordMap=coord_map,
        randomSeed=42,
        useRandomCoords=True,
        enforceChirality=True,
    )
    if status < 0:
        conf = Chem.Conformer(protonated_mol.GetNumAtoms())
        for idx, pos in coord_map.items():
            conf.SetAtomPosition(idx, pos)
        for idx, atom in enumerate(embedded.GetAtoms()):
            if atom.GetAtomicNum() == 1 and atom.GetNeighbors():
                parent_idx = atom.GetNeighbors()[0].GetIdx()
                parent_pos = conf.GetAtomPosition(parent_idx)
                conf.SetAtomPosition(idx, Point3D(parent_pos.x + 0.7, parent_pos.y + 0.4, parent_pos.z + 0.3))
        embedded.AddConformer(conf, assignId=True)
    else:
        conf = embedded.GetConformer()
        for idx, pos in coord_map.items():
            conf.SetAtomPosition(idx, pos)

    return embedded, original_positions, heavy_indices


def build_forcefield() -> Any:
    ff_files = ["amber/protein.ff19SB.xml", "amber/tip3p_standard.xml"]
    optional_sets = [["amber/tip3p_HFE_multivalent.xml"], []]
    for extra in optional_sets:
        try:
            return omm_app.ForceField(*(ff_files + extra))
        except Exception:
            continue
    return omm_app.ForceField(*ff_files)


def create_complex_system(
    receptor_pdb_path: str,
    ligand_mol: Any,
    charge_method: str = "am1bcc",
    restrain_ligand_heavy: bool = True,
) -> Tuple[Any, Any, int, int]:
    try:
        from openff.toolkit import Molecule as OFFMolecule
        from openmm.app import Modeller, PDBFile
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator
    except ImportError as exc:
        raise RuntimeError(f"OpenFF/OpenMM not available: {exc}") from exc

    receptor_pdb = PDBFile(receptor_pdb_path)
    receptor_topology = receptor_pdb.topology
    receptor_positions = receptor_pdb.positions

    off_mol = OFFMolecule.from_rdkit(ligand_mol, allow_undefined_stereo=True)
    if charge_method == "am1bcc":
        try:
            from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
            nagl = NAGLToolkitWrapper()
            off_mol.assign_partial_charges(
                "openff-gnn-am1bcc-0.1.0-rc.2.pt",
                toolkit_registry=nagl,
            )
        except Exception:
            off_mol.assign_partial_charges("gasteiger")
    else:
        off_mol.assign_partial_charges("gasteiger")

    smirnoff = SMIRNOFFTemplateGenerator(molecules=[off_mol], forcefield="openff-2.3.0")
    forcefield = build_forcefield()
    forcefield.registerTemplateGenerator(smirnoff.generator)

    modeller = Modeller(receptor_topology, receptor_positions)
    n_receptor_atoms = modeller.topology.getNumAtoms()

    lig_top = off_mol.to_topology().to_openmm()
    lig_positions = []
    conf = ligand_mol.GetConformer()
    for i in range(ligand_mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        lig_positions.append(openmm.Vec3(pos.x * 0.1, pos.y * 0.1, pos.z * 0.1) * omm_unit.nanometers)
    modeller.add(lig_top, lig_positions)

    try:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=omm_app.NoCutoff,
            constraints=None,
            rigidWater=False,
        )
    except Exception:
        # Chain breaks cause bond mismatches; fall back to ignoring external
        # bonds and resolving CYS/CYX ambiguity by actual atom content.
        cys_templates = {}
        for res in modeller.topology.residues():
            if res.name in ('CYS', 'CYX', 'CYM'):
                atom_names = {a.name for a in res.atoms()}
                cys_templates[res] = 'CYS' if 'HG' in atom_names else 'CYX'
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=omm_app.NoCutoff,
            constraints=None,
            rigidWater=False,
            ignoreExternalBonds=True,
            residueTemplates=cys_templates,
        )

    add_gbsa_obc2_force(system, modeller.topology)

    restraint = openmm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k", 25000.0 * omm_unit.kilojoules_per_mole / omm_unit.nanometers ** 2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    positions = modeller.getPositions()
    for idx, atom in enumerate(modeller.topology.atoms()):
        is_heavy = atom.element is not None and atom.element.symbol != "H"
        is_receptor = idx < n_receptor_atoms
        is_ligand = idx >= n_receptor_atoms
        if is_heavy and (is_receptor or (is_ligand and restrain_ligand_heavy)):
            pos = positions[idx]
            restraint.addParticle(idx, [pos[0], pos[1], pos[2]])
    system.addForce(restraint)

    integrator = openmm.VerletIntegrator(0.001 * omm_unit.picoseconds)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CPU"))
    context.setPositions(positions)
    return context, receptor_topology, n_receptor_atoms, ligand_mol.GetNumAtoms()


def restore_heavy_atoms(
    receptor_topology: Any,
    receptor_positions: Any,
    ligand_mol: Any,
    minimized_positions: Any,
    n_receptor_atoms: int,
    ligand_heavy_indices: list[int],
    ligand_heavy_positions: list[Point3D],
) -> Tuple[list[Any], Any]:
    restored = list(minimized_positions)
    receptor_conf_nm = []
    for idx, atom in enumerate(receptor_topology.atoms()):
        if atom.element is not None and atom.element.symbol != "H":
            restored[idx] = receptor_positions[idx]

        receptor_pos_nm = restored[idx].value_in_unit(omm_unit.nanometer)
        receptor_conf_nm.append(
            openmm.Vec3(
                float(receptor_pos_nm[0]),
                float(receptor_pos_nm[1]),
                float(receptor_pos_nm[2]),
            )
        )

    refined_mol = Chem.RWMol(ligand_mol)
    conf = refined_mol.GetConformer()
    for atom_idx in range(ligand_mol.GetNumAtoms()):
        pos = restored[n_receptor_atoms + atom_idx].value_in_unit(omm_unit.angstrom)
        conf.SetAtomPosition(atom_idx, Point3D(float(pos[0]), float(pos[1]), float(pos[2])))

    for atom_idx, original_pos in zip(ligand_heavy_indices, ligand_heavy_positions):
        conf.SetAtomPosition(atom_idx, original_pos)

    return receptor_conf_nm * omm_unit.nanometer, refined_mol


def heavy_atom_rmsd(mol_a: Any, mol_b: Any) -> float:
    coords_a = []
    coords_b = []
    conf_a = mol_a.GetConformer()
    conf_b = mol_b.GetConformer()
    for idx, atom in enumerate(mol_a.GetAtoms()):
        if atom.GetAtomicNum() == 1:
            continue
        pos_a = conf_a.GetAtomPosition(idx)
        pos_b = conf_b.GetAtomPosition(idx)
        coords_a.append((pos_a.x, pos_a.y, pos_a.z))
        coords_b.append((pos_b.x, pos_b.y, pos_b.z))
    if not coords_a:
        return 0.0
    sq = 0.0
    for (ax, ay, az), (bx, by, bz) in zip(coords_a, coords_b):
        dx = ax - bx
        dy = ay - by
        dz = az - bz
        sq += dx * dx + dy * dy + dz * dz
    return (sq / len(coords_a)) ** 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare docking complex from receptor + X-ray ligand")
    parser.add_argument("--receptor_pdb", required=True, help="Prepared receptor PDB path")
    parser.add_argument("--xray_ligand_sdf", required=True, help="Raw extracted X-ray ligand SDF")
    parser.add_argument("--output_dir", required=True, help="Output directory for prepared complex assets")
    parser.add_argument("--charge_method", choices=["gasteiger", "am1bcc"], default="am1bcc")
    parser.add_argument("--ph_min", type=float, default=6.4)
    parser.add_argument("--ph_max", type=float, default=8.4)
    parser.add_argument("--skip_reference_protonation", action="store_true")
    parser.add_argument("--max_iterations", type=int, default=5000)
    args = parser.parse_args()

    if not HAS_OPENMM:
        print("ERROR: OpenMM not installed", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    protonation_dir = os.path.join(args.output_dir, "protonated_reference")
    manifest_path = os.path.join(args.output_dir, "prepared_complex_manifest.json")
    receptor_out = os.path.join(args.output_dir, "receptor_refined.pdb")
    ligand_out = os.path.join(args.output_dir, "reference_ligand_prepared.sdf")
    receptor_prep_metadata = load_receptor_prep_metadata(args.receptor_pdb)

    t0 = time.time()
    xray_raw_mol = load_first_sdf(args.xray_ligand_sdf)
    selected_prot_path, protonated_mol, proto_meta = choose_reference_variant(
        args.xray_ligand_sdf,
        protonation_dir,
        args.ph_min,
        args.ph_max,
        protonate_reference=not args.skip_reference_protonation,
    )
    positioned_mol, original_heavy_positions, heavy_indices = map_exact_crystal_coordinates(xray_raw_mol, protonated_mol)

    print("=== Prepare Docking Complex ===", flush=True)
    print(f"Receptor: {os.path.basename(args.receptor_pdb)}", flush=True)
    print(f"Reference ligand: {os.path.basename(args.xray_ligand_sdf)}", flush=True)
    print(f"Reference protonation: {'enabled' if proto_meta['protonation_enabled'] else 'disabled'}", flush=True)
    print(f"Selected reference input: {os.path.basename(selected_prot_path)}", flush=True)
    print(f"Force field: Sage 2.3.0 + OBC2 implicit solvent", flush=True)
    print(flush=True)

    context, receptor_topology, n_receptor_atoms, n_ligand_atoms = create_complex_system(
        args.receptor_pdb,
        positioned_mol,
        args.charge_method,
        restrain_ligand_heavy=True,
    )

    state = context.getState(getPositions=True)
    positions = list(state.getPositions().value_in_unit(omm_unit.nanometers))
    conf = positioned_mol.GetConformer()
    for idx in range(n_ligand_atoms):
        pos = conf.GetAtomPosition(idx)
        positions[n_receptor_atoms + idx] = openmm.Vec3(pos.x * 0.1, pos.y * 0.1, pos.z * 0.1)
    context.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(context, tolerance=0.01, maxIterations=args.max_iterations)

    state = context.getState(getPositions=True, getEnergy=True)
    minimized_positions = state.getPositions()
    energy = state.getPotentialEnergy().value_in_unit(omm_unit.kilocalories_per_mole)

    from openmm.app import PDBFile
    receptor_pdb = PDBFile(args.receptor_pdb)
    receptor_conf, refined_ligand = restore_heavy_atoms(
        receptor_topology,
        receptor_pdb.positions,
        positioned_mol,
        minimized_positions,
        n_receptor_atoms,
        heavy_indices,
        original_heavy_positions,
    )

    with open(receptor_out, "w") as handle:
        PDBFile.writeFile(receptor_topology, receptor_conf, handle, keepIds=True)

    refined_ligand.SetProp("_Name", "xray_reference")
    refined_ligand.SetProp("isReferencePose", "1")
    refined_ligand.SetProp("referenceSource", "prepared_complex")
    refined_ligand.SetProp("referenceFormalCharge", str(proto_meta["formal_charge"]))
    refined_ligand.SetProp("refinement_energy", f"{energy:.2f}")
    refined_ligand.SetProp("SMILES", proto_meta["selected_smiles"])
    writer = Chem.SDWriter(ligand_out)
    writer.write(refined_ligand)
    writer.close()

    rmsd = heavy_atom_rmsd(positioned_mol, refined_ligand)
    manifest = {
        "schema_version": 2,
        "prepared_at_epoch_s": time.time(),
        "prepared_receptor_pdb": receptor_out,
        "prepared_reference_ligand_sdf": ligand_out,
        "raw_reference_ligand_sdf": args.xray_ligand_sdf,
        "selected_protonated_reference_sdf": selected_prot_path,
        "reference_protonation_enabled": proto_meta["protonation_enabled"],
        "charge_method": args.charge_method,
        "reference_formal_charge": proto_meta["formal_charge"],
        "reference_raw_smiles": proto_meta["raw_smiles"],
        "reference_prepared_smiles": proto_meta["selected_smiles"],
        "protonation_method": proto_meta["protonation_method"],
        "protonation_ph_min": proto_meta["ph_min"],
        "protonation_ph_max": proto_meta["ph_max"],
        "protonation_candidate_count": proto_meta["candidate_count"],
        "protonation_candidate_paths": proto_meta["candidate_paths"],
        "xray_heavy_atom_rmsd": rmsd,
        "refinement_energy": energy,
        "retained_water_present": any(res.name in WATER_RESNAMES for res in receptor_topology.residues()),
        "receptor_prep_metadata_path": receptor_prep_metadata.get("metadata_path") if receptor_prep_metadata else None,
        "receptor_protonation_ph": receptor_prep_metadata.get("receptor_protonation_ph") if receptor_prep_metadata else None,
        "receptor_propka_available": receptor_prep_metadata.get("propka_available") if receptor_prep_metadata else False,
        "receptor_applied_overrides": receptor_prep_metadata.get("applied_overrides", []) if receptor_prep_metadata else [],
        "receptor_ignored_shifted_residues": receptor_prep_metadata.get("ignored_shifted_residues", []) if receptor_prep_metadata else [],
        "receptor_resolved_variants": receptor_prep_metadata.get("resolved_variants", {}) if receptor_prep_metadata else {},
        "receptor_pocket_filtered": receptor_prep_metadata.get("pocket_filtered") if receptor_prep_metadata else False,
        "receptor_pocket_residue_keys": receptor_prep_metadata.get("pocket_residue_keys", []) if receptor_prep_metadata else [],
    }
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Prepared complex complete in {time.time() - t0:.1f}s", flush=True)
    print(f"  Heavy-atom RMSD: {rmsd:.4f} A", flush=True)
    print(f"  Reference formal charge: {proto_meta['formal_charge']:+d}", flush=True)
    print(json.dumps({
        "manifest_path": manifest_path,
        "prepared_receptor_pdb": receptor_out,
        "prepared_reference_ligand_sdf": ligand_out,
        "xray_heavy_atom_rmsd": rmsd,
        "refinement_energy": energy,
    }))


if __name__ == "__main__":
    main()
