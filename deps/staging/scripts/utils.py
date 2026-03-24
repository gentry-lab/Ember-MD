# Copyright (c) 2026 Ember Contributors. MIT License.
"""
Shared utilities for Ember staging scripts.

Consolidates duplicated patterns: CIF conversion, ligand selection,
PBC transforms, SA score calculation, and output schemas.
"""

import os
import sys
from typing import Any, Dict, List, Optional, TypedDict


# ---------------------------------------------------------------------------
# Output schemas — match TypeScript interfaces in shared/types/
# Used for documentation; enforce with mypy, not at runtime.
# ---------------------------------------------------------------------------

class LigandCentroid(TypedDict):
    """3D centroid coordinates. Matches shared/types/dock.ts DetectedLigand.centroid."""
    x: float
    y: float
    z: float

class DetectedLigand(TypedDict, total=False):
    """Ligand detected in a PDB file. Matches shared/types/dock.ts DetectedLigand."""
    id: str            # e.g. "ATP_A_501"
    resname: str
    chain: str
    resnum: str        # String, not int (PDB insertion codes)
    num_atoms: int
    centroid: LigandCentroid

class DockMolecule(TypedDict):
    """Molecule loaded for docking. Matches shared/types/dock.ts DockMolecule."""
    filename: str
    smiles: str
    qed: float
    saScore: float
    sdfPath: str

class DockResult(TypedDict, total=False):
    """Single docking pose result. Matches shared/types/dock.ts DockResult."""
    ligandName: str
    smiles: str
    qed: float
    vinaAffinity: float
    poseIndex: int
    outputSdf: str
    parentMolecule: Optional[str]
    protonationVariant: Optional[int]
    conformerIndex: Optional[int]

class AnalysisResultMeta(TypedDict):
    """Common fields for analysis result JSON files."""
    type: str          # 'rmsd', 'rmsf', 'hbonds', 'contacts'
    plotPath: Optional[str]
    csvPath: Optional[str]

class BindingSiteHotspot(TypedDict):
    """Single hotspot from binding site map analysis."""
    type: str          # 'hydrophobic', 'hbond_donor', 'hbond_acceptor'
    position: List[float]
    direction: List[float]
    score: float

class SurfacePropsResult(TypedDict):
    """Per-atom surface properties. Matches shared/types/ipc.ts SurfacePropsResult."""
    atomCount: int
    hydrophobic: List[float]     # current-structure atom field, [-1, 1]
    electrostatic: List[float]   # current-structure electrostatic field, [-1, 1]


class SingleMoleculeResult(TypedDict, total=False):
    """Single molecule extraction/conversion result. Matches shared/types/dock.ts SingleMoleculeResult."""
    sdfPath: str
    smiles: str
    name: str
    qed: float
    mw: float
    thumbnail: str       # Base64 PNG
    method: str          # Extraction method (e.g., 'openbabel', 'biopython')


class ClusterResultData(TypedDict, total=False):
    """Single cluster result. Matches shared/types/ipc.ts ClusterResultData."""
    clusterId: int
    frameCount: int
    population: float          # Percentage
    centroidFrame: int
    centroidPdbPath: str


class ClusteringResult(TypedDict):
    """Full clustering output. Matches shared/types/ipc.ts ClusteringResult."""
    clusters: List[ClusterResultData]
    frameAssignments: List[int]
    outputDir: str


class BindingSiteMapResult(TypedDict):
    """Binding site map output. Matches shared/types/ipc.ts BindingSiteMapResult."""
    hydrophobicDx: str
    hbondDonorDx: str
    hbondAcceptorDx: str
    hotspots: List[BindingSiteHotspot]
    gridDimensions: List[int]
    ligandCom: List[float]


class FepSnapshotResult(TypedDict):
    """Single FEP snapshot result. Matches shared/types/ipc.ts FepSnapshotResult."""
    snapshotIndex: int
    frameIndex: int
    timeNs: float
    deltaG_complex: float
    deltaG_solvent: float
    deltaG_bind: float
    uncertainty: float


class ProtonationResult(TypedDict):
    """Protonation enumeration output. Parsed inline in main.ts."""
    protonated_paths: List[str]
    parent_mapping: Dict[str, str]


class ConformerResult(TypedDict):
    """Conformer generation output. Parsed inline in main.ts."""
    conformer_paths: List[str]
    parent_mapping: Dict[str, str]


# ---------------------------------------------------------------------------
# Grid / DX utilities — used by map_binding_site, analyze_gist, run_probe_md
# ---------------------------------------------------------------------------

def write_dx(filepath: str, data: Any, origin: List[float], spacing: float,
             shape: 'tuple[int, int, int]') -> None:
    """Write a 3D grid in OpenDX format."""
    import numpy as np
    nx, ny, nz = shape
    flat = data.flatten(order='C')
    n_full_rows = len(flat) // 3
    remainder = len(flat) % 3
    with open(filepath, 'w') as f:
        f.write(f'object 1 class gridpositions counts {nx} {ny} {nz}\n')
        f.write(f'origin {origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}\n')
        f.write(f'delta {spacing:.6f} 0.000000 0.000000\n')
        f.write(f'delta 0.000000 {spacing:.6f} 0.000000\n')
        f.write(f'delta 0.000000 0.000000 {spacing:.6f}\n')
        f.write(f'object 2 class gridconnections counts {nx} {ny} {nz}\n')
        f.write(f'object 3 class array type double rank 0 items {len(flat)} data follows\n')
        if n_full_rows > 0:
            rows = flat[:n_full_rows * 3].reshape(-1, 3)
            lines = '\n'.join(f'{r[0]:.6f} {r[1]:.6f} {r[2]:.6f}' for r in rows)
            f.write(lines)
            f.write('\n')
        if remainder > 0:
            f.write(' '.join(f'{v:.6f}' for v in flat[n_full_rows * 3:]))
            f.write('\n')
        f.write('attribute "dep" string "positions"\n')
        f.write('object "regular positions regular connections" class field\n')
        f.write('component "positions" value 1\n')
        f.write('component "connections" value 2\n')
        f.write('component "data" value 3\n')


def read_dx(filepath: str) -> 'tuple[Any, List[float], float, tuple[int, int, int]]':
    """Read an OpenDX file into numpy array + grid metadata.

    Returns (data_3d, origin, spacing, shape).
    """
    import numpy as np

    origin = [0.0, 0.0, 0.0]
    spacing = 0.0
    nx = ny = nz = 0
    values: List[float] = []

    with open(filepath) as f:
        in_data = False
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('object 1'):
                parts = line.split()
                nx, ny, nz = int(parts[-3]), int(parts[-2]), int(parts[-1])
            elif line.startswith('origin'):
                parts = line.split()
                origin = [float(parts[1]), float(parts[2]), float(parts[3])]
            elif line.startswith('delta') and spacing == 0:
                parts = line.split()
                spacing = float(parts[1])
            elif 'data follows' in line:
                in_data = True
                continue
            elif line.startswith('attribute') or line.startswith('object') or line.startswith('component'):
                in_data = False
                continue
            if in_data:
                values.extend(float(v) for v in line.split())

    data = np.array(values).reshape((nx, ny, nz))
    return data, origin, spacing, (nx, ny, nz)


def normalize_grid(arr: Any) -> Any:
    """Normalize array to [0, 1] by dividing by max."""
    vmax = arr.max()
    if vmax > 0:
        return arr / vmax
    return arr


def find_hotspots(grid_3d: Any, channel_name: str, grid_origin: List[float],
                  grid_spacing: float, ligand_com: Any) -> List[dict]:
    """Find hotspot clusters above 70th percentile.

    Returns list of dicts with type, position, direction, score (top 5 by score).
    """
    import numpy as np
    from scipy import ndimage

    nonzero = grid_3d[grid_3d > 0]
    if len(nonzero) == 0:
        return []
    threshold = np.percentile(nonzero, 70)
    binary = grid_3d > threshold
    labeled, num_features = ndimage.label(binary)
    if num_features == 0:
        return []

    results = []
    for label_id in range(1, num_features + 1):
        cluster_mask = labeled == label_id
        cluster_scores = grid_3d[cluster_mask]
        score = float(cluster_scores.mean())

        centroid_idx = ndimage.center_of_mass(grid_3d, labeled, label_id)
        pos = [
            grid_origin[0] + centroid_idx[0] * grid_spacing,
            grid_origin[1] + centroid_idx[1] * grid_spacing,
            grid_origin[2] + centroid_idx[2] * grid_spacing,
        ]
        direction = [pos[0] - ligand_com[0], pos[1] - ligand_com[1], pos[2] - ligand_com[2]]
        mag = np.sqrt(sum(d ** 2 for d in direction))
        if mag > 0:
            direction = [d / mag for d in direction]
        results.append({
            'type': channel_name,
            'position': [round(p, 3) for p in pos],
            'direction': [round(d, 3) for d in direction],
            'score': round(score, 4),
        })

    results.sort(key=lambda h: h['score'], reverse=True)
    return results[:5]


def find_ligand_com(pdb_path: str, ligand_resname: str, ligand_resnum: int) -> Any:
    """Find ligand center of mass from a PDB/CIF file using BioPython.

    Returns numpy array of shape (3,) with COM coordinates in Angstroms.
    Exits with error if ligand not found.
    """
    import numpy as np

    if pdb_path.lower().endswith('.cif'):
        from Bio.PDB import MMCIFParser
        parser = MMCIFParser(QUIET=True)
    else:
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_path)

    coords = []
    for chain in structure[0]:
        for residue in chain:
            resname = residue.get_resname().strip()
            resnum = residue.get_id()[1]
            if resname == ligand_resname and resnum == ligand_resnum:
                for atom in residue:
                    coords.append(atom.get_vector().get_array())

    if not coords:
        print(f"Error: Ligand {ligand_resname} {ligand_resnum} not found", file=sys.stderr)
        sys.exit(1)

    return np.array(coords).mean(axis=0)


def convert_cif_to_pdb(cif_path: str) -> str:
    """Convert mmCIF (.cif) file to PDB format.

    Prefers BioPython (preserves all HETATM records including ligands).
    Falls back to PDBFixer (may drop some non-standard residues).
    Returns path to converted PDB, or original path if not CIF.
    """
    if not cif_path.lower().endswith('.cif'):
        return cif_path

    pdb_path = cif_path.rsplit('.', 1)[0] + '.pdb'
    if os.path.exists(pdb_path):
        print(f'  Using cached CIF->PDB conversion: {os.path.basename(pdb_path)}', file=sys.stderr)
        return pdb_path

    # Prefer BioPython — preserves all residues including HETATM
    try:
        from Bio.PDB import MMCIFParser, PDBIO
        print(f"  Converting CIF to PDB (BioPython): {os.path.basename(cif_path)}", file=sys.stderr)
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('struct', cif_path)
        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_path)
        print(f"  Converted to: {os.path.basename(pdb_path)}", file=sys.stderr)
        return pdb_path
    except ImportError:
        pass
    except Exception as e:
        print(f"  WARNING: BioPython CIF conversion failed: {e}, trying PDBFixer", file=sys.stderr)

    # Fallback to PDBFixer
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except ImportError:
        print("ERROR: Neither BioPython nor PDBFixer available for CIF support", file=sys.stderr)
        sys.exit(1)

    print(f"  Converting CIF to PDB (PDBFixer): {os.path.basename(cif_path)}", file=sys.stderr)
    fixer = PDBFixer(filename=cif_path)
    with open(pdb_path, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
    print(f"  Converted to: {os.path.basename(pdb_path)}", file=sys.stderr)
    return pdb_path


def select_ligand_atoms(universe: Any, custom_selection: Optional[str] = None) -> Any:
    """Select ligand atoms from an MDAnalysis Universe.

    Two-tier fallback:
    1. Custom selection or broad non-protein/non-solvent selection
    2. Common ligand resnames (LIG, UNL, UNK, MOL)

    Returns MDAnalysis AtomGroup (may be empty).
    """
    if custom_selection:
        return universe.select_atoms(custom_selection)

    # Primary: everything that's not protein, water, or ions
    ligand = universe.select_atoms(
        'not protein and not resname WAT HOH TIP3 TIP4 NA CL SOL K MG and not element H'
    )
    if len(ligand) == 0:
        # Fallback: common ligand resnames
        ligand = universe.select_atoms('(resname LIG UNL UNK MOL) and not element H')

    return ligand


def apply_pbc_transforms(universe: Any, protein: Any = None, ligand: Any = None) -> None:
    """Apply PBC unwrapping and centering transformations.

    Applies: unwrap → center_in_box → wrap workflow.
    Handles protein-ligand, protein-only, and ligand-only systems.
    Silently continues if transforms fail (e.g., missing box dimensions).
    """
    try:
        from MDAnalysis import transformations as trans

        if protein is not None and len(protein) > 0:
            if ligand is not None and len(ligand) > 0:
                complex_group = protein + ligand
            else:
                complex_group = protein
            workflow = [
                trans.unwrap(complex_group),
                trans.center_in_box(protein, center='mass'),
                trans.wrap(complex_group, compound='fragments'),
            ]
            universe.trajectory.add_transformations(*workflow)
            print("Applied PBC unwrapping and centering transformations")
        elif ligand is not None and len(ligand) > 0:
            workflow = [
                trans.unwrap(ligand),
                trans.center_in_box(ligand, center='mass'),
                trans.wrap(ligand, compound='fragments'),
            ]
            universe.trajectory.add_transformations(*workflow)
            print("Applied PBC unwrapping and centering transformations (ligand-only)")
    except Exception as e:
        print(f"Warning: Could not apply PBC transformations: {e}", file=sys.stderr)


def load_sdf(path: str, remove_hs: bool = False) -> Any:
    """Load the first molecule from an SDF or SDF.gz file.

    Returns an RDKit Mol or None if loading fails.
    """
    import gzip
    from rdkit import Chem

    def _sanitize_with_kekulize_fallback(mol: Any) -> Any:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            try:
                Chem.SanitizeMol(
                    mol,
                    sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
                )
            except Exception:
                return None
        return mol

    lower = path.lower()
    if lower.endswith('.sdf.gz'):
        with gzip.open(path, 'rb') as fh:
            suppl = Chem.ForwardSDMolSupplier(fh, removeHs=False, sanitize=False)
            mol = next(suppl, None)
    else:
        suppl = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
        mol = suppl[0] if len(suppl) > 0 else None

    if mol is None:
        return None

    mol = _sanitize_with_kekulize_fallback(mol)
    if mol is None:
        return None

    if remove_hs:
        mol = Chem.RemoveHs(mol, sanitize=False)

    return mol


def add_gbsa_obc2_force(
    system: Any,
    topology: Any,
    *,
    rdmol: Any = None,
) -> None:
    """Add a GBSA-OBC2 implicit solvent force to an OpenMM system.

    Uses mbondi3 Born radii and OBC2 screening factors. Extracts charges
    from the existing NonbondedForce. Works for both ligand-only systems
    (pass rdmol for H-N radius lookup) and protein-ligand complexes
    (uses topology bonds when rdmol is None).

    Args:
        system: OpenMM System with a NonbondedForce already present.
        topology: OpenMM Topology for the system.
        rdmol: Optional RDKit Mol for H-N neighbor lookup (ligand-only).
               If None, uses topology bonds instead.
    """
    import openmm
    import openmm.unit as unit

    RADII_NM = {
        'H': 0.12, 'C': 0.17, 'N': 0.155, 'O': 0.15, 'F': 0.15,
        'S': 0.18, 'P': 0.185, 'Cl': 0.17, 'Br': 0.185, 'I': 0.198,
        'Na': 0.102, 'K': 0.138, 'Mg': 0.072, 'Ca': 0.10, 'Zn': 0.074,
        'Fe': 0.064, 'Mn': 0.067,
    }
    SCREEN = {
        'H': 0.85, 'C': 0.72, 'N': 0.79, 'O': 0.85, 'F': 0.88,
        'S': 0.96, 'P': 0.86, 'Cl': 0.80, 'Br': 0.80, 'I': 0.80,
        'Na': 0.80, 'K': 0.80, 'Mg': 0.80, 'Ca': 0.80, 'Zn': 0.80,
        'Fe': 0.80, 'Mn': 0.80,
    }

    nb_force = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nb_force = force
            break
    if nb_force is None:
        return

    # Build bond neighbor map from topology for H-N detection
    bonds = list(topology.bonds())
    h_bonded_to_n: set = set()
    for a1, a2 in bonds:
        sym1 = a1.element.symbol if a1.element else ''
        sym2 = a2.element.symbol if a2.element else ''
        if sym1 == 'H' and sym2 == 'N':
            h_bonded_to_n.add(a1.index)
        elif sym2 == 'H' and sym1 == 'N':
            h_bonded_to_n.add(a2.index)

    # If rdmol provided, also check RDKit neighbors (more reliable for ligand-only)
    if rdmol is not None:
        for i in range(rdmol.GetNumAtoms()):
            atom = rdmol.GetAtomWithIdx(i)
            if atom.GetSymbol() == 'H':
                for nbr in atom.GetNeighbors():
                    if nbr.GetSymbol() == 'N':
                        h_bonded_to_n.add(i)
                        break

    gbsa = openmm.GBSAOBCForce()
    gbsa.setSolventDielectric(78.5)
    gbsa.setSoluteDielectric(1.0)
    gbsa.setNonbondedMethod(openmm.GBSAOBCForce.NoCutoff)

    for idx, atom in enumerate(topology.atoms()):
        charge, _sigma, _epsilon = nb_force.getParticleParameters(idx)
        q = charge.value_in_unit(unit.elementary_charge)
        symbol = atom.element.symbol if atom.element else 'C'
        radius = RADII_NM.get(symbol, 0.15)
        screen = SCREEN.get(symbol, 0.80)
        if symbol == 'H' and idx in h_bonded_to_n:
            radius = 0.13
        gbsa.addParticle(q, radius, screen)

    system.addForce(gbsa)


def calculate_sa_score(mol: Any, default: float = 3.0) -> float:
    """Calculate synthetic accessibility score using RDKit's SA_Score contrib.

    Returns SA score (1-10, lower is more synthetically accessible),
    or default value if sascorer is unavailable.
    """
    try:
        from rdkit.Chem import RDConfig
        sa_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
        if sa_path not in sys.path:
            sys.path.append(sa_path)
        import sascorer
        return float(sascorer.calculateScore(mol))
    except Exception:
        return default
