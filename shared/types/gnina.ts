// Copyright (c) 2026 Ember Contributors. MIT License.
/**
 * GNINA docking mode types
 */

// Ligand source types for multi-input support
export type LigandSource = 'fraggen' | 'sdf_directory' | 'smiles_csv' | 'single_molecule';

// Single molecule input result (from SMILES/MOL paste or file)
export interface SingleMoleculeResult {
  sdfPath: string;
  smiles: string;
  name: string;
  qed: number;
  mw: number;
  thumbnail: string;  // Base64 PNG
}

// Molecule data from FragGen CSV
export interface GninaMolecule {
  filename: string;
  smiles: string;
  qed: number;
  saScore: number;
  sdfPath: string;
}

// GNINA docking configuration
export interface GninaDockingConfig {
  exhaustiveness: number;      // Default: 8, range 1-32
  numPoses: number;            // Default: 9, range 1-20
  autoboxAdd: number;          // Default: 4 Angstroms, range 2-8
  numThreads: number;          // Default: 0 (auto-detect = use all cores), range 1 to CPU count
  minimize: boolean;           // Post-docking MMFF energy minimization (default: false)
  seed: number;                // Random seed (0 = random, default: 0)
  waterRetentionDistance: number;  // Keep waters within this distance of ligand (0 = remove all)
}

export const DEFAULT_GNINA_CONFIG: GninaDockingConfig = {
  exhaustiveness: 8,
  numPoses: 9,
  autoboxAdd: 4,
  numThreads: 0,  // 0 = auto-detect (use all cores)
  minimize: false,
  seed: 0,
  waterRetentionDistance: 0,  // Remove all waters by default
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
export interface GninaDockingResult {
  ligandName: string;
  smiles: string;
  qed: number;
  cnnScore: number;            // 0-1, higher is better
  cnnAffinity: number;         // kcal/mol, more negative is better
  vinaAffinity: number;        // kcal/mol
  poseIndex: number;
  outputSdf: string;
  parentMolecule: string;           // Original molecule name (for protonation variants)
  protonationVariant: number | null; // Variant index (null if not protonated)
  conformerIndex: number | null;     // Conformer index (null if not generated)
  // CORDIAL rescoring (optional, populated after rescoring step)
  cordialExpectedPkd?: number;       // Expected pKd (0-8 scale)
  cordialPHighAffinity?: number;     // P(pKd >= 6)
  cordialPVeryHighAffinity?: number; // P(pKd >= 7)
}

// CSV parsing result
export interface ParseCsvResult {
  molecules: GninaMolecule[];
  qedRange: { min: number; max: number };
  sdfDirectory: string;
}

// GNINA download progress
export interface GninaDownloadProgress {
  downloaded: number;
  total: number;
  percentage: number;
}

// Detected ligand from PDB
export interface DetectedLigand {
  id: string;           // e.g., "ATP_A_501"
  resname: string;      // e.g., "ATP"
  chain: string;        // e.g., "A"
  resnum: string;       // e.g., "501"
  num_atoms: number;
  centroid: { x: number; y: number; z: number };
}
