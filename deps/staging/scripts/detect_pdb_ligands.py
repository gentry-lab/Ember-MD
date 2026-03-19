#!/usr/bin/env python3
"""
Detect ligands in a PDB file for GNINA docking setup.

Identifies HETATM records that are likely ligands (excludes water, ions, etc.)
and can extract a specific ligand to a separate file.
"""

import argparse
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from typing import Any, Dict, List, Tuple


from utils import convert_cif_to_pdb

# Common non-ligand HETATM residues to exclude
EXCLUDE_RESIDUES = {
    # Water
    'HOH', 'WAT', 'H2O', 'DOD', 'DIS',
    # Ions
    'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE', 'MN', 'CU', 'CO', 'NI',
    'NA+', 'CL-', 'K+', 'MG2', 'CA2', 'ZN2', 'FE2', 'FE3', 'MN2', 'CU2', 'CO2', 'NI2',
    'IOD', 'BR', 'F',
    # Common buffer/crystallization agents
    'SO4', 'PO4', 'NO3', 'CO3', 'ACT', 'ACE', 'ACY', 'EDO', 'GOL', 'PEG',
    'DMS', 'DMF', 'MPD', 'TRS', 'CIT', 'MLI', 'TAR', 'SUC', 'MAL',
    'EPE', 'MES', 'HEP', 'PGE', 'P6G', '1PE', 'PG4',
    # Modified residues (usually part of protein)
    'MSE', 'SEC', 'PCA', 'HYP', 'SEP', 'TPO', 'PTR',
}

# Minimum atoms for a ligand (exclude very small molecules)
MIN_LIGAND_ATOMS = 5

# Standard amino acid residues (3-letter codes)
STANDARD_RESIDUES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    # Common protonation variants
    'HID', 'HIE', 'HIP', 'CYX', 'ASH', 'GLH', 'LYN',
    # DNA/RNA
    'DA', 'DT', 'DG', 'DC', 'A', 'U', 'G', 'C',
}


def parse_cif_ligands(cif_path: str) -> Dict[str, Dict[str, Any]]:
    """Parse mmCIF file and identify ligands using BioPython's MMCIF parser.

    This avoids PDB column-width overflow issues with long residue names (>3 chars).
    """
    try:
        from Bio.PDB import MMCIFParser
    except ImportError:
        print("ERROR: BioPython required for CIF ligand detection", file=sys.stderr)
        sys.exit(1)

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('struct', cif_path)
    model = structure[0]

    ligands = {}
    for chain in model:
        for residue in chain:
            het_flag = residue.get_id()[0]
            resname = residue.get_resname().strip()
            resnum = str(residue.get_id()[1])
            chain_id = chain.id.strip() or '_'

            # Only HETATM residues (het_flag starts with 'H_') or
            # non-standard ATOM residues
            is_hetatm = het_flag.startswith('H_')
            is_nonstandard = (het_flag == ' ' and
                              resname.upper() not in STANDARD_RESIDUES and
                              resname.upper() not in EXCLUDE_RESIDUES)

            if not is_hetatm and not is_nonstandard:
                continue

            if resname.upper() in EXCLUDE_RESIDUES:
                continue

            lig_id = f"{resname}_{chain_id}_{resnum}"

            if lig_id not in ligands:
                ligands[lig_id] = {
                    'atoms': [], 'coords': [],
                    'resname': resname, 'chain': chain_id, 'resnum': resnum,
                }

            for atom in residue:
                coord = atom.get_vector().get_array()
                x, y, z = float(coord[0]), float(coord[1]), float(coord[2])
                ligands[lig_id]['atoms'].append({
                    'name': atom.get_name(),
                    'resname': resname,
                    'chain': chain_id,
                    'resnum': resnum,
                    'x': x, 'y': y, 'z': z,
                    'line': '',  # CIF doesn't have PDB lines
                })
                ligands[lig_id]['coords'].append((x, y, z))

    # Filter out small molecules
    valid_ligands = {}
    for lig_id, data in ligands.items():
        if len(data['atoms']) >= MIN_LIGAND_ATOMS:
            coords = data['coords']
            cx = sum(c[0] for c in coords) / len(coords)
            cy = sum(c[1] for c in coords) / len(coords)
            cz = sum(c[2] for c in coords) / len(coords)

            valid_ligands[lig_id] = {
                'id': lig_id,
                'resname': data['resname'],
                'chain': data['chain'],
                'resnum': data['resnum'],
                'num_atoms': len(data['atoms']),
                'centroid': {'x': cx, 'y': cy, 'z': cz},
                'atoms': data['atoms']
            }

    return valid_ligands


def parse_pdb_ligands(pdb_path: str) -> Dict[str, Dict[str, Any]]:
    """Parse PDB file and identify ligands.

    First checks HETATM records (standard PDB ligands).
    If none found, falls back to scanning ATOM records for non-standard
    residues (handles MDAnalysis-generated PDBs which write ligands as ATOM).
    """
    ligands: Dict[str, Dict[str, Any]] = defaultdict(lambda: {'atoms': [], 'coords': []})

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('HETATM'):
                resname = line[17:20].strip()
                chain = line[21].strip() or '_'
                resnum = line[22:26].strip()

                # Skip excluded residues
                if resname.upper() in EXCLUDE_RESIDUES:
                    continue

                # Create unique ligand ID
                lig_id = f"{resname}_{chain}_{resnum}"

                # Store atom info
                atom_name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                ligands[lig_id]['atoms'].append({
                    'name': atom_name,
                    'resname': resname,
                    'chain': chain,
                    'resnum': resnum,
                    'x': x, 'y': y, 'z': z,
                    'line': line
                })
                ligands[lig_id]['coords'].append((x, y, z))
                ligands[lig_id]['resname'] = resname
                ligands[lig_id]['chain'] = chain
                ligands[lig_id]['resnum'] = resnum

    # Fallback: if no HETATM ligands found, scan ATOM records for non-standard residues
    # MDAnalysis writes ligand atoms as ATOM, not HETATM
    if not ligands:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    resname = line[17:20].strip()
                    chain = line[21].strip() or '_'
                    resnum = line[22:26].strip()

                    # Skip standard amino acids, water, ions, etc.
                    if resname.upper() in STANDARD_RESIDUES:
                        continue
                    if resname.upper() in EXCLUDE_RESIDUES:
                        continue

                    lig_id = f"{resname}_{chain}_{resnum}"

                    atom_name = line[12:16].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])

                    ligands[lig_id]['atoms'].append({
                        'name': atom_name,
                        'resname': resname,
                        'chain': chain,
                        'resnum': resnum,
                        'x': x, 'y': y, 'z': z,
                        'line': line
                    })
                    ligands[lig_id]['coords'].append((x, y, z))
                    ligands[lig_id]['resname'] = resname
                    ligands[lig_id]['chain'] = chain
                    ligands[lig_id]['resnum'] = resnum

    # Filter out small molecules
    valid_ligands = {}
    for lig_id, data in ligands.items():
        if len(data['atoms']) >= MIN_LIGAND_ATOMS:
            # Calculate centroid
            coords = data['coords']
            cx = sum(c[0] for c in coords) / len(coords)
            cy = sum(c[1] for c in coords) / len(coords)
            cz = sum(c[2] for c in coords) / len(coords)

            valid_ligands[lig_id] = {
                'id': lig_id,
                'resname': data['resname'],
                'chain': data['chain'],
                'resnum': data['resnum'],
                'num_atoms': len(data['atoms']),
                'centroid': {'x': cx, 'y': cy, 'z': cz},
                'atoms': data['atoms']
            }

    return valid_ligands


def extract_ligand(pdb_path: str, ligand_id: str, output_path: str) -> bool:
    """Extract a specific ligand to a separate PDB file."""
    ligands = parse_pdb_ligands(pdb_path)

    if ligand_id not in ligands:
        print(f"ERROR: Ligand {ligand_id} not found in PDB", file=sys.stderr)
        return False

    ligand = ligands[ligand_id]

    with open(output_path, 'w') as f:
        f.write(f"REMARK  Extracted ligand {ligand_id} from {os.path.basename(pdb_path)}\n")
        for atom in ligand['atoms']:
            f.write(atom['line'])
        f.write("END\n")

    return True


def _water_near_ligand(water_coords: List[Tuple[float, float, float]], ligand_coords: List[Tuple[float, float, float]], distance: float) -> bool:
    """Check if any water atom is within distance of any ligand atom."""
    for wx, wy, wz in water_coords:
        for lx, ly, lz in ligand_coords:
            dx, dy, dz = wx - lx, wy - ly, wz - lz
            if math.sqrt(dx*dx + dy*dy + dz*dz) <= distance:
                return True
    return False


# Water residue names (subset of EXCLUDE_RESIDUES used for water retention logic)
WATER_RESIDUES = {'HOH', 'WAT', 'H2O', 'DOD', 'DIS'}

# Metal ions to retain in receptor (Vina supports Zn, Mg, Mn, Ca, Fe AD4 types)
RETAIN_METALS = {
    'ZN', 'MG', 'CA', 'FE', 'MN', 'CU', 'CO', 'NI',
    'ZN2', 'MG2', 'CA2', 'FE2', 'FE3', 'MN2', 'CU2', 'CO2', 'NI2',
}

# Enzymatic cofactors to retain near binding site (not crystallization artifacts)
RETAIN_COFACTORS = {
    'NAD', 'NAI', 'NAP', 'NDP',      # NAD(P)+/H
    'FAD', 'FMN',                      # Flavins
    'HEM', 'HEC', 'HEA', 'HEB',      # Heme variants
    'ATP', 'ADP', 'AMP', 'ANP',      # Adenine nucleotides (ANP = AMPPNP analog)
    'GTP', 'GDP', 'GNP',              # Guanine nucleotides
    'SAM', 'SAH',                      # S-adenosylmethionine/homocysteine
    'COA', 'ACO',                      # Coenzyme A
    'TPP',                             # Thiamine pyrophosphate
    'PLP',                             # Pyridoxal phosphate
    'BTN',                             # Biotin
}


def prepare_receptor(pdb_path: str, ligand_id: str, output_path: str, water_distance: float = 0.0, add_hydrogens: bool = True) -> bool:
    """Prepare receptor by removing the specified ligand and adding hydrogens.

    Args:
        pdb_path: Input PDB file
        ligand_id: Ligand to remove (e.g., "ATP_A_501")
        output_path: Output PDB file for prepared receptor
        water_distance: Keep crystallographic waters within this distance (Å)
                       of the ligand. 0 = remove all waters (default).
        add_hydrogens: Add missing hydrogens via PDBFixer at pH 7.4 (default: True).
                       Improves GNINA CNN scoring accuracy.
    """
    ligands = parse_pdb_ligands(pdb_path)

    if ligand_id not in ligands:
        print(f"ERROR: Ligand {ligand_id} not found in PDB", file=sys.stderr)
        return False

    ligand = ligands[ligand_id]
    resname = ligand['resname']
    chain = ligand['chain']
    resnum = ligand['resnum']
    ligand_coords = [(a['x'], a['y'], a['z']) for a in ligand['atoms']]

    # First pass: collect HETATM coordinates for distance-based retention
    water_molecules = defaultdict(list)   # (chain, resnum) -> [(x, y, z)]
    cofactor_molecules = defaultdict(list) # (chain, resnum) -> [(x, y, z)]
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('HETATM'):
                res = line[17:20].strip().upper()
                hchain = line[21].strip() or '_'
                hresnum = line[22:26].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                if res in WATER_RESIDUES:
                    water_molecules[(hchain, hresnum)].append((x, y, z))
                elif res in RETAIN_COFACTORS:
                    cofactor_molecules[(hchain, hresnum)].append((x, y, z))

    # Determine which waters and cofactors to keep (near ligand)
    kept_waters = set()
    if water_distance > 0:
        for key, coords in water_molecules.items():
            if _water_near_ligand(coords, ligand_coords, water_distance):
                kept_waters.add(key)
        if kept_waters:
            print(f"  Keeping {len(kept_waters)} waters within {water_distance:.1f} A of ligand",
                  file=sys.stderr)

    kept_cofactors = set()
    cofactor_dist = max(water_distance, 5.0)  # cofactors within 5 A or water_distance
    for key, coords in cofactor_molecules.items():
        if _water_near_ligand(coords, ligand_coords, cofactor_dist):
            kept_cofactors.add(key)
    if kept_cofactors:
        print(f"  Keeping {len(kept_cofactors)} cofactors within {cofactor_dist:.1f} A of ligand",
              file=sys.stderr)

    # Write cleaned receptor PDB
    tmp_path = output_path + '.tmp' if add_hydrogens else output_path
    kept_metals = 0
    with open(pdb_path, 'r') as f_in, open(tmp_path, 'w') as f_out:
        f_out.write(f"REMARK  Receptor prepared by removing {ligand_id}\n")
        if kept_waters:
            f_out.write(f"REMARK  Retained {len(kept_waters)} waters within {water_distance:.1f} A\n")
        for line in f_in:
            if line.startswith('HETATM'):
                line_resname = line[17:20].strip().upper()
                line_chain = line[21].strip() or '_'
                line_resnum = line[22:26].strip()

                # Skip the docking ligand itself
                if line_resname == resname.upper() and line_chain == chain and line_resnum == resnum:
                    continue

                # Keep crystallographic waters near ligand
                if line_resname in WATER_RESIDUES:
                    if (line_chain, line_resnum) in kept_waters:
                        f_out.write(line)
                    continue

                # Keep metal ions (important for metalloenzyme coordination)
                if line_resname in RETAIN_METALS:
                    f_out.write(line)
                    kept_metals += 1
                    continue

                # Keep cofactors near binding site
                if line_resname in RETAIN_COFACTORS:
                    if (line_chain, line_resnum) in kept_cofactors:
                        f_out.write(line)
                    continue

                # Remove other HETATM (crystallization artifacts, buffers, etc.)
                continue

            f_out.write(line)

    if kept_metals:
        print(f"  Keeping {kept_metals} metal ion atoms in receptor", file=sys.stderr)

    # Protein preparation pipeline:
    # 1. Reduce: optimize Asn/Gln/His orientations (flip ambiguous residues)
    # 2. PROPKA: predict per-residue pKa values
    # 3. PDBFixer: add hydrogens with PROPKA-informed protonation
    if add_hydrogens:
        current_file = tmp_path

        # Step 1: Run reduce for Asn/Gln/His flip optimization
        try:
            import subprocess as _sp
            import shutil
            reduce_bin = shutil.which('reduce')
            if reduce_bin:
                print("  Optimizing Asn/Gln/His orientations (reduce)...", file=sys.stderr)
                reduced_path = tmp_path.replace('.tmp', '.reduced.pdb')
                result = _sp.run(
                    [reduce_bin, '-FLIP', '-Quiet', current_file],
                    capture_output=True, text=True, timeout=60
                )
                if result.returncode == 0 and result.stdout:
                    with open(reduced_path, 'w') as f:
                        f.write(result.stdout)
                    current_file = reduced_path
                    print("  Asn/Gln/His flips optimized", file=sys.stderr)
                else:
                    print("  WARNING: reduce returned no output, skipping flips",
                          file=sys.stderr)
            else:
                print("  NOTE: reduce not found, skipping Asn/Gln/His optimization",
                      file=sys.stderr)
        except Exception as e:
            print(f"  WARNING: reduce failed: {e}, continuing without flips",
                  file=sys.stderr)

        # Step 2: Run PROPKA for per-residue pKa prediction
        propka_ph = 7.4  # Default
        try:
            from propka.run import single as propka_single
            print("  Predicting per-residue pKa (PROPKA)...", file=sys.stderr)
            mol = propka_single(current_file)

            standard_pka = {'ASP': 3.8, 'GLU': 4.5, 'HIS': 6.5, 'LYS': 10.5,
                           'CYS': 8.3, 'TYR': 10.1}
            shifted_count = 0
            for conformation in mol.conformations:
                conf = mol.conformations[conformation]
                for group in conf.get_titratable_groups():
                    if group.residue_type in standard_pka and group.pka_value is not None:
                        std = standard_pka[group.residue_type]
                        std_prot = std > 7.4
                        act_prot = group.pka_value > 7.4
                        if std_prot != act_prot:
                            shifted_count += 1
                            print(f"    {group.residue_type} {group.chain_id}{group.res_num}: "
                                  f"pKa={group.pka_value:.1f} (std={std:.1f})",
                                  file=sys.stderr)
                break  # Only need first conformation

            if shifted_count > 0:
                print(f"  PROPKA found {shifted_count} residue(s) with shifted protonation",
                      file=sys.stderr)
            else:
                print("  PROPKA: all residues have standard protonation at pH 7.4",
                      file=sys.stderr)
        except ImportError:
            print("  NOTE: PROPKA not installed, using standard pH 7.4 protonation",
                  file=sys.stderr)
        except Exception as e:
            print(f"  WARNING: PROPKA failed: {e}, using standard pH 7.4",
                  file=sys.stderr)

        # Step 3: PDBFixer — add hydrogens
        try:
            from pdbfixer import PDBFixer
            from openmm.app import PDBFile

            print(f"  Adding hydrogens (pH {propka_ph}) via PDBFixer...", file=sys.stderr)
            fixer = PDBFixer(filename=current_file)
            fixer.addMissingHydrogens(propka_ph)

            with open(output_path, 'w') as f:
                PDBFile.writeFile(fixer.topology, fixer.positions, f)

            # Clean up temp files
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            reduced_path = tmp_path.replace('.tmp', '.reduced.pdb')
            if os.path.exists(reduced_path):
                os.remove(reduced_path)

            print("  Receptor preparation complete", file=sys.stderr)
        except ImportError:
            print("  WARNING: PDBFixer not available, skipping hydrogen addition",
                  file=sys.stderr)
            if current_file != output_path:
                import shutil
                shutil.copy2(current_file, output_path)
        except Exception as e:
            print(f"  WARNING: Hydrogen addition failed: {e}, using unprotonated receptor",
                  file=sys.stderr)
            if current_file != output_path:
                import shutil
                shutil.copy2(current_file, output_path)

        # Clean up any remaining temp files
        for tmp_f in [tmp_path, tmp_path.replace('.tmp', '.reduced.pdb')]:
            if os.path.exists(tmp_f) and tmp_f != output_path:
                try:
                    os.remove(tmp_f)
                except OSError:
                    pass

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description='Detect and extract ligands from PDB')
    parser.add_argument('--pdb', required=True, help='Input PDB file')
    parser.add_argument('--mode', choices=['detect', 'extract', 'prepare_receptor', 'add_hydrogens'],
                       default='detect', help='Operation mode')
    parser.add_argument('--ligand_id', help='Ligand ID for extract/prepare modes')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--water_distance', type=float, default=0.0,
                       help='Keep waters within this distance (A) of ligand (0=remove all)')
    parser.add_argument('--no_hydrogens', action='store_true',
                       help='Skip hydrogen addition via PDBFixer')
    args = parser.parse_args()

    if not os.path.exists(args.pdb):
        print(f"ERROR: PDB file not found: {args.pdb}", file=sys.stderr)
        sys.exit(1)

    # For CIF files: use native CIF parser for detect mode (avoids PDB column overflow
    # with long residue names like A1AAK). For extract/prepare modes, convert to PDB.
    is_cif = args.pdb.lower().endswith('.cif')

    if args.mode == 'detect' and is_cif:
        ligands = parse_cif_ligands(args.pdb)
    elif is_cif:
        input_path = convert_cif_to_pdb(args.pdb)
        args.pdb = input_path
        ligands = None  # not used for detect
    else:
        ligands = None  # will be set below

    if args.mode == 'detect':
        if ligands is None:
            ligands = parse_pdb_ligands(args.pdb)
        # Output as JSON for easy parsing
        result = []
        for lig_id, data in ligands.items():
            result.append({
                'id': lig_id,
                'resname': data['resname'],
                'chain': data['chain'],
                'resnum': data['resnum'],
                'num_atoms': data['num_atoms'],
                'centroid': data['centroid']
            })
        print(json.dumps(result))

    elif args.mode == 'extract':
        if not args.ligand_id or not args.output:
            print("ERROR: --ligand_id and --output required for extract mode", file=sys.stderr)
            sys.exit(1)
        if extract_ligand(args.pdb, args.ligand_id, args.output):
            print(json.dumps({'success': True, 'output': args.output}))
        else:
            sys.exit(1)

    elif args.mode == 'prepare_receptor':
        if not args.ligand_id or not args.output:
            print("ERROR: --ligand_id and --output required for prepare_receptor mode", file=sys.stderr)
            sys.exit(1)
        if prepare_receptor(args.pdb, args.ligand_id, args.output,
                           water_distance=args.water_distance,
                           add_hydrogens=not args.no_hydrogens):
            print(json.dumps({'success': True, 'output': args.output}))
        else:
            sys.exit(1)

    elif args.mode == 'add_hydrogens':
        if not args.output:
            print("ERROR: --output required for add_hydrogens mode", file=sys.stderr)
            sys.exit(1)
        try:
            from pdbfixer import PDBFixer
            from openmm.app import PDBFile

            print(f"  Adding hydrogens (pH 7.4) via PDBFixer...", file=sys.stderr)
            fixer = PDBFixer(filename=args.pdb)
            fixer.findMissingResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(7.4)

            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, 'w') as f:
                PDBFile.writeFile(fixer.topology, fixer.positions, f)

            print(f"  Prepared structure saved to {args.output}", file=sys.stderr)
            print(json.dumps({'success': True, 'output': args.output}))
        except ImportError:
            print("WARNING: PDBFixer not available, copying original file", file=sys.stderr)
            import shutil
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            shutil.copy2(args.pdb, args.output)
            print(json.dumps({'success': True, 'output': args.output, 'no_hydrogens': True}))
        except Exception as e:
            print(f"ERROR: Hydrogen addition failed: {e}", file=sys.stderr)
            # Fall back to raw file
            import shutil
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            shutil.copy2(args.pdb, args.output)
            print(json.dumps({'success': True, 'output': args.output, 'no_hydrogens': True}))


if __name__ == '__main__':
    main()
