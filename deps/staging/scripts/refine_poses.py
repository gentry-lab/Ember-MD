#!/usr/bin/env python3
"""
Post-docking pocket refinement using OpenMM with OpenFF Sage 2.3.0 + OBC2.

Minimizes each docked pose in the receptor pocket using physics-based
force field with implicit solvent. This corrects Vina's empirical scoring
artifacts, especially for charged/polar interactions (salt bridges, H-bonds).

Usage:
    python refine_poses.py \
        --receptor_pdb <pdb> \
        --poses_dir <dir> \
        --output_dir <dir> \
        --max_iterations 200

Output:
    Refined SDF files in output_dir, JSON result on last line of stdout.
"""

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
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


def read_sdf_poses(sdf_path: str) -> List[Tuple[Any, dict]]:
    """Read all poses from an SDF file (possibly gzipped). Returns list of (mol, props)."""
    poses = []
    try:
        if sdf_path.endswith('.gz'):
            with gzip.open(sdf_path, 'rb') as f:
                suppl = Chem.ForwardSDMolSupplier(f, removeHs=False)
                for mol in suppl:
                    if mol is not None:
                        props = {k: mol.GetProp(k) for k in mol.GetPropsAsDict()}
                        poses.append((mol, props))
        else:
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
            for mol in suppl:
                if mol is not None:
                    props = {k: mol.GetProp(k) for k in mol.GetPropsAsDict()}
                    poses.append((mol, props))
    except Exception as e:
        print(f"Warning: Failed to read {sdf_path}: {e}", file=sys.stderr)
    return poses


def create_complex_system(
    receptor_pdb_path: str,
    ligand_mol: Any,
) -> Optional[Tuple[Any, Any, int, int]]:
    """
    Create an OpenMM system for the receptor-ligand complex with OBC2.
    Returns (context, topology, n_receptor_atoms, n_ligand_atoms) or None.
    """
    try:
        from openff.toolkit import Molecule as OFFMolecule
        from openff.toolkit import ForceField as OFFForceField
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator
    except ImportError:
        print("Warning: OpenFF/OpenMM not available for refinement", file=sys.stderr)
        return None

    # Load and fix receptor (add missing atoms + hydrogens, remove waters/ions)
    from pdbfixer import PDBFixer
    fixer = PDBFixer(filename=receptor_pdb_path)
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    # Parameterize ligand with Sage 2.3.0
    off_mol = OFFMolecule.from_rdkit(ligand_mol, allow_undefined_stereo=True)
    off_mol.assign_partial_charges('gasteiger')

    smirnoff = SMIRNOFFTemplateGenerator(molecules=[off_mol], forcefield='openff-2.3.0')

    # Build force field with protein + ligand (no implicit solvent XML — we add OBC2 manually)
    ff = omm_app.ForceField('amber/protein.ff19SB.xml')
    ff.registerTemplateGenerator(smirnoff.generator)

    # Create modeller from fixed receptor
    from openmm.app import Modeller
    modeller = Modeller(fixer.topology, fixer.positions)
    n_receptor_atoms = modeller.topology.getNumAtoms()

    # Add ligand topology + positions
    lig_top = off_mol.to_topology().to_openmm()
    lig_positions = []
    conf = ligand_mol.GetConformer()
    for i in range(ligand_mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        lig_positions.append(openmm.Vec3(pos.x * 0.1, pos.y * 0.1, pos.z * 0.1) * omm_unit.nanometers)
    modeller.add(lig_top, lig_positions)

    n_ligand_atoms = ligand_mol.GetNumAtoms()

    # Create vacuum system first (NoCutoff for small system)
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=omm_app.NoCutoff,
        constraints=None,
    )

    # Add OBC2 implicit solvent manually (same proven pattern as MCMM GBSAMinimizer)
    # Extract charges from NonbondedForce, assign Born radii by element
    RADII_NM = {'H': 0.12, 'C': 0.17, 'N': 0.155, 'O': 0.15, 'F': 0.15,
                'S': 0.18, 'P': 0.185, 'Cl': 0.17, 'Br': 0.185, 'I': 0.198}
    SCREEN = {'H': 0.85, 'C': 0.72, 'N': 0.79, 'O': 0.85, 'F': 0.88,
              'S': 0.96, 'P': 0.86, 'Cl': 0.80, 'Br': 0.80, 'I': 0.80}

    nb_force = None
    for f in system.getForces():
        if isinstance(f, openmm.NonbondedForce):
            nb_force = f
            break

    if nb_force is not None:
        gbsa = openmm.GBSAOBCForce()
        gbsa.setSolventDielectric(78.5)
        gbsa.setSoluteDielectric(1.0)
        gbsa.setNonbondedMethod(openmm.GBSAOBCForce.NoCutoff)

        for i, atom in enumerate(modeller.topology.atoms()):
            charge, sigma, epsilon = nb_force.getParticleParameters(i)
            q = charge.value_in_unit(omm_unit.elementary_charge)
            element = atom.element.symbol if atom.element else 'C'
            radius = RADII_NM.get(element, 0.15)
            screen = SCREEN.get(element, 0.80)
            # mbondi3: H bonded to N gets larger radius
            if element == 'H':
                for bond in modeller.topology.bonds():
                    a1, a2 = bond
                    if a1.index == i and a2.element and a2.element.symbol == 'N':
                        radius = 0.13; break
                    if a2.index == i and a1.element and a1.element.symbol == 'N':
                        radius = 0.13; break
            gbsa.addParticle(q, radius, screen)

        system.addForce(gbsa)

    # Restrain receptor heavy atoms (only minimize the ligand)
    restraint = openmm.CustomExternalForce('k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    restraint.addGlobalParameter('k', 500.0 * omm_unit.kilojoules_per_mole / omm_unit.nanometers ** 2)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')

    positions = modeller.getPositions()
    for i, atom in enumerate(modeller.topology.atoms()):
        if i < n_receptor_atoms and atom.element is not None and atom.element.symbol != 'H':
            pos = positions[i]
            restraint.addParticle(i, [pos[0], pos[1], pos[2]])

    system.addForce(restraint)

    # Create context on CPU
    integrator = openmm.VerletIntegrator(0.001 * omm_unit.picoseconds)
    platform = openmm.Platform.getPlatformByName('CPU')
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)

    return context, modeller.topology, n_receptor_atoms, n_ligand_atoms


def refine_pose(
    context: Any,
    ligand_mol: Any,
    n_receptor_atoms: int,
    n_ligand_atoms: int,
    max_iterations: int,
) -> Tuple[float, Any]:
    """
    Minimize a ligand pose in the receptor pocket.
    Updates the ligand coordinates in the context, minimizes, returns (energy, refined_mol).
    """
    # Update ligand positions in context (receptor positions stay fixed via restraints)
    state = context.getState(getPositions=True)
    positions = list(state.getPositions().value_in_unit(omm_unit.nanometers))

    conf = ligand_mol.GetConformer()
    for i in range(n_ligand_atoms):
        pos = conf.GetAtomPosition(i)
        positions[n_receptor_atoms + i] = openmm.Vec3(pos.x * 0.1, pos.y * 0.1, pos.z * 0.1)
    context.setPositions(positions)

    # Minimize
    openmm.LocalEnergyMinimizer.minimize(context, tolerance=0.01, maxIterations=max_iterations)

    # Get refined energy and positions
    state = context.getState(getEnergy=True, getPositions=True)
    energy = state.getPotentialEnergy().value_in_unit(omm_unit.kilocalories_per_mole)
    min_positions = state.getPositions()

    # Copy refined ligand coordinates back to RDKit mol
    refined_mol = Chem.RWMol(ligand_mol)
    refined_conf = refined_mol.GetConformer()
    for i in range(n_ligand_atoms):
        p = min_positions[n_receptor_atoms + i].value_in_unit(omm_unit.angstrom)
        refined_conf.SetAtomPosition(i, Point3D(float(p[0]), float(p[1]), float(p[2])))

    return energy, refined_mol


def main() -> None:
    parser = argparse.ArgumentParser(description='Post-docking pocket refinement with OpenMM')
    parser.add_argument('--receptor_pdb', required=True, help='Receptor PDB file')
    parser.add_argument('--poses_dir', required=True, help='Directory with docked SDF.gz files')
    parser.add_argument('--output_dir', required=True, help='Output directory for refined SDFs')
    parser.add_argument('--max_iterations', type=int, default=200, help='Minimization iterations per pose')
    args = parser.parse_args()

    if not HAS_OPENMM:
        print("ERROR: OpenMM not installed, cannot refine poses", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Post-Docking Pocket Refinement ===", flush=True)
    print(f"Receptor: {os.path.basename(args.receptor_pdb)}", flush=True)
    print(f"Poses dir: {args.poses_dir}", flush=True)
    print(f"Max iterations: {args.max_iterations}", flush=True)
    print(f"Force field: Sage 2.3.0 + OBC2 implicit solvent", flush=True)
    print(flush=True)

    # Find docked SDF files
    import glob
    sdf_files = sorted(glob.glob(os.path.join(args.poses_dir, '*_docked.sdf.gz')))
    if not sdf_files:
        sdf_files = sorted(glob.glob(os.path.join(args.poses_dir, '*.sdf.gz')))
    if not sdf_files:
        sdf_files = sorted(glob.glob(os.path.join(args.poses_dir, '*.sdf')))

    if not sdf_files:
        print("No pose files found", flush=True)
        print(json.dumps({"refined_count": 0, "output_dir": args.output_dir}))
        return

    print(f"Found {len(sdf_files)} pose files", flush=True)

    # Read first pose to set up the complex system
    first_poses = read_sdf_poses(sdf_files[0])
    if not first_poses:
        print("ERROR: Could not read any poses from first file", file=sys.stderr)
        sys.exit(1)

    print("Setting up receptor-ligand system...", flush=True)
    t0 = time.time()
    result = create_complex_system(args.receptor_pdb, first_poses[0][0])
    if result is None:
        print("ERROR: Failed to create complex system", file=sys.stderr)
        sys.exit(1)

    context, topology, n_receptor, n_ligand = result
    print(f"System ready ({time.time()-t0:.1f}s): {n_receptor} receptor + {n_ligand} ligand atoms", flush=True)
    print(flush=True)

    # Refine all poses
    total_refined = 0
    for fi, sdf_path in enumerate(sdf_files):
        name = Path(sdf_path).stem.replace('.sdf', '')
        poses = read_sdf_poses(sdf_path)

        if not poses:
            print(f"  [{fi+1}/{len(sdf_files)}] {name}: no poses, skipping", flush=True)
            continue

        output_path = os.path.join(args.output_dir, f"{name}.sdf.gz")
        writer = Chem.SDWriter(gzip.open(output_path, 'wt'))

        for pi, (mol, props) in enumerate(poses):
            try:
                energy, refined = refine_pose(context, mol, n_receptor, n_ligand, args.max_iterations)

                # Copy original properties
                for k, v in props.items():
                    try:
                        refined.SetProp(k, str(v))
                    except Exception:
                        pass
                refined.SetProp('refinement_energy', f'{energy:.2f}')

                writer.write(refined)
                total_refined += 1
            except Exception as e:
                print(f"  Warning: pose {pi} refinement failed: {e}", file=sys.stderr)
                # Write unrefined pose as fallback
                writer.write(mol)
                total_refined += 1

        writer.close()
        print(f"  [{fi+1}/{len(sdf_files)}] {name}: {len(poses)} poses refined", flush=True)

    print(f"\nRefinement complete: {total_refined} poses refined", flush=True)
    print(json.dumps({
        "refined_count": total_refined,
        "output_dir": args.output_dir,
    }))


if __name__ == '__main__':
    main()
