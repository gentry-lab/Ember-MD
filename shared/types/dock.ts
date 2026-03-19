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

// Stereoisomer enumeration configuration
export interface StereoisomerConfig {
  enabled: boolean;
  maxStereoisomers: number;   // Default: 8 (caps at 2^3 unspecified centers)
}

export const DEFAULT_STEREOISOMER_CONFIG: StereoisomerConfig = {
  enabled: false,
  maxStereoisomers: 8,
};

// Conformer generation configuration
export type ConformerMethod = 'none' | 'etkdg' | 'mcmm';

export interface ConformerConfig {
  method: ConformerMethod;
  maxConformers: number;    // Default: 5 (ETKDG), 50 (MCMM)
  rmsdCutoff: number;       // Default: 1.0 Å
  energyWindow: number;     // Default: 5.0 kcal/mol
  mcmmSteps: number;        // Default: 100 (MCMM only)
  mcmmTemperature: number;  // Default: 298 K (MCMM only)
  sampleAmides: boolean;    // Default: true (MCMM only)
}

export const DEFAULT_CONFORMER_CONFIG: ConformerConfig = {
  method: 'mcmm',
  maxConformers: 50,
  rmsdCutoff: 1.0,
  energyWindow: 5.0,
  mcmmSteps: 1000,
  mcmmTemperature: 298,
  sampleAmides: true,
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
