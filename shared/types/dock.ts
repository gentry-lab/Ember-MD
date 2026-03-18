/**
 * Docking mode types (Vina + CORDIAL)
 */

// Ligand source types for multi-input support
export type LigandSource = 'sdf_directory' | 'smiles_csv' | 'single_molecule';

// Single molecule input result (from SMILES/MOL paste or file)
export interface SingleMoleculeResult {
  sdfPath: string;
  smiles: string;
  name: string;
  qed: number;
  mw: number;
  thumbnail: string;  // Base64 PNG
  method?: string;    // Extraction method (e.g., 'openbabel', 'biopython')
}

// Molecule data for docking
export interface DockMolecule {
  filename: string;
  smiles: string;
  qed: number;
  sdfPath: string;
}

// Vina docking configuration
export interface DockConfig {
  exhaustiveness: number;      // Default: 32 (Eberhardt 2021 recommendation), range 1-64
  numPoses: number;            // Default: 5, range 1-20
  autoboxAdd: number;          // Default: 4 Angstroms, range 2-8
  numCpus: number;             // Default: 0 (auto-detect), range 0 to CPU count
  seed: number;                // Random seed (0 = random, default: 0)
  coreConstrained: boolean;    // MCS alignment (default: true)
}

export const DEFAULT_DOCK_CONFIG: DockConfig = {
  exhaustiveness: 32,
  numPoses: 9,
  autoboxAdd: 4,
  numCpus: 0,
  seed: 0,
  coreConstrained: false,
};

// Protonation state enumeration configuration
export interface ProtonationConfig {
  enabled: boolean;
  phMin: number;      // Default: 6.4
  phMax: number;      // Default: 8.4
}

export const DEFAULT_PROTONATION_CONFIG: ProtonationConfig = {
  enabled: true,
  phMin: 6.4,
  phMax: 8.4,
};

// Conformer generation configuration
export type ConformerMethod = 'none' | 'etkdg';

export interface ConformerConfig {
  method: ConformerMethod;
  maxConformers: number;    // Default: 10
  rmsdCutoff: number;       // Default: 0.5 Å
  energyWindow: number;     // Default: 10.0 kcal/mol
}

export const DEFAULT_CONFORMER_CONFIG: ConformerConfig = {
  method: 'etkdg',
  maxConformers: 5,
  rmsdCutoff: 1.0,
  energyWindow: 5.0,
};

// CORDIAL rescoring configuration
export interface CordialConfig {
  enabled: boolean;
  batchSize: number;        // Default: 32
}

export const DEFAULT_CORDIAL_CONFIG: CordialConfig = {
  enabled: true,
  batchSize: 32,
};

// CORDIAL scoring result for a single pose
export interface CordialScore {
  sourceSdf: string;
  sourceName: string;
  poseIndex: number;
  expectedPkd: number;           // Weighted sum of 8 ordinal classes (0-8 scale)
  pHighAffinity: number;         // P(pKd >= 6)
  pVeryHighAffinity: number;     // P(pKd >= 7)
  probabilities: number[];       // All 8 class probabilities
}

// Docking result for a single pose
export interface DockResult {
  ligandName: string;
  smiles: string;
  qed: number;
  vinaAffinity: number;        // kcal/mol
  poseIndex: number;
  outputSdf: string;
  parentMolecule: string;
  protonationVariant: number | null;
  conformerIndex: number | null;
  cordialExpectedPkd?: number;
  cordialPHighAffinity?: number;
  cordialPVeryHighAffinity?: number;
  coreRmsd?: number;           // MCS core RMSD vs reference
}

// Detected ligand from PDB
export interface DetectedLigand {
  id: string;           // e.g., "ATP_A_501"
  resname: string;      // e.g., "ATP"
  chain: string;        // e.g., "A"
  resnum: string;       // e.g., "501" (string from PDB column)
  num_atoms: number;
  centroid?: { x: number; y: number; z: number };
}
