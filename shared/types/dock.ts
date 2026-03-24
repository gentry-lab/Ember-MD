// Copyright (c) 2026 Ember Contributors. MIT License.
/**
 * Docking mode types (Vina + CORDIAL)
 */

// Ligand source types for docking input
export type LigandSource = 'structure_files' | 'molecule_csv' | 'smiles_input';

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
}

export const DEFAULT_DOCK_CONFIG: DockConfig = {
  exhaustiveness: 32,
  numPoses: 9,
  autoboxAdd: 4,
  numCpus: 0,
  seed: 0,
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

export const DEFAULT_RECEPTOR_WATER_DISTANCE = 3.5;

// Crystallographic water retention configuration
export interface WaterRetentionConfig {
  enabled: boolean;
  distance: number;  // Angstroms from reference ligand
}

export const DEFAULT_WATER_RETENTION_CONFIG: WaterRetentionConfig = {
  enabled: true,
  distance: DEFAULT_RECEPTOR_WATER_DISTANCE,
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
export type ConformerMethod = 'none' | 'etkdg' | 'mcmm' | 'crest';

export interface ConformerConfig {
  method: ConformerMethod;
  maxConformers: number;    // Default: 5 (ETKDG), 50 (MCMM)
  rmsdCutoff: number;       // Default: 1.0 Å
  energyWindow: number;     // Default: 5.0 kcal/mol
  mcmmSteps: number;        // Default: 1000 (MCMM only)
  mcmmTemperature: number;  // Default: 298 K (MCMM only)
  sampleAmides: boolean;    // Default: true (MCMM only)
  xtbRerank: boolean;       // Default: true — re-rank ETKDG/MCMM conformers by GFN2-xTB energy
}

export const DEFAULT_CONFORMER_CONFIG: ConformerConfig = {
  method: 'mcmm',
  maxConformers: 50,
  rmsdCutoff: 1.0,
  energyWindow: 5.0,
  mcmmSteps: 1000,
  mcmmTemperature: 298,
  sampleAmides: true,
  xtbRerank: true,
};

// Post-dock pocket refinement configuration
export type ChargeMethod = 'gasteiger' | 'am1bcc';

export interface RefinementConfig {
  enabled: boolean;
  maxIterations: number;    // Default: 5000
  chargeMethod: ChargeMethod;  // Default: 'am1bcc' (NAGL neural net AM1-BCC)
}

export const DEFAULT_REFINEMENT_CONFIG: RefinementConfig = {
  enabled: true,
  maxIterations: 5000,
  chargeMethod: 'am1bcc',
};

// xTB configuration for docking
export interface XtbConfig {
  preOptimize: boolean;         // xTB geometry optimization before docking (default: true)
}

export const DEFAULT_XTB_CONFIG: XtbConfig = {
  preOptimize: true,
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
  expectedPkd: number;           // Sum of ordinal cumulative probabilities P(pKd >= k), k=1..8
  pHighAffinity: number;         // P(pKd >= 6)
  pVeryHighAffinity: number;     // P(pKd >= 7)
  probabilities: number[];       // All 8 class probabilities
}

export interface PreparedComplexManifest {
  schema_version: number;
  prepared_at_epoch_s: number;
  prepared_receptor_pdb: string;
  prepared_reference_ligand_sdf: string;
  raw_reference_ligand_sdf: string;
  selected_protonated_reference_sdf: string;
  reference_protonation_enabled: boolean;
  charge_method: ChargeMethod;
  reference_formal_charge: number;
  reference_raw_smiles: string;
  reference_prepared_smiles: string;
  protonation_method: string;
  protonation_ph_min: number;
  protonation_ph_max: number;
  protonation_candidate_count: number;
  protonation_candidate_paths: string[];
  xray_heavy_atom_rmsd: number;
  refinement_energy: number;
  retained_water_present: boolean;
  receptor_prep_metadata_path?: string | null;
  receptor_protonation_ph?: number | null;
  receptor_propka_available?: boolean;
  receptor_applied_overrides?: Array<{
    residue_key: string;
    chain_id: string;
    residue_number: string;
    residue_name: string;
    selected_variant: string;
    default_variant?: string | null;
    reason: string;
    pka?: number;
  }>;
  receptor_ignored_shifted_residues?: Array<{
    residue_key: string;
    chain_id: string;
    residue_number: string;
    residue_name: string;
    pka: number;
    default_state?: string;
    propka_state?: string;
    reason: string;
  }>;
  receptor_resolved_variants?: Record<string, string>;
  receptor_pocket_filtered?: boolean;
  receptor_pocket_residue_keys?: string[];
}

// Docking result for a single pose
export interface DockResult {
  ligandName: string;
  smiles: string;
  qed: number;
  vinaAffinity: number | null;        // Docked Vina affinity in kcal/mol
  vinaScoreOnlyAffinity?: number;     // Vina score_only affinity for reference poses
  poseIndex: number;
  outputSdf: string;
  parentMolecule: string;
  protonationVariant: number | null;
  conformerIndex: number | null;
  isReferencePose: boolean;
  refinementEnergy?: number;
  cordialExpectedPkd?: number;
  cordialPHighAffinity?: number;
  cordialPVeryHighAffinity?: number;
  xtbEnergyKcal?: number;      // Relative xTB energy per compound (kcal/mol)
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
