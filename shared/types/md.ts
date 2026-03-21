/**
 * Molecular Dynamics simulation types
 */

export type MDForceFieldPreset = 'ff14sb-tip3p' | 'ff19sb-opc' | 'ff19sb-opc3' | 'charmm36-mtip3p';

export interface MDConfig {
  // User-adjustable
  productionNs: number;           // Nanoseconds (default: 10, text input)
  forceFieldPreset: MDForceFieldPreset;
  compoundId: string;             // Optional compound identifier (e.g., "imatinib")
  temperatureK: number;           // Kelvin (default: 300)
  saltConcentrationM: number;     // Molar (default: 0.15 = 150 mM)
  paddingNm: number;              // nm (default: 1.2)
  restrainLigandNs: number;       // ns of ligand restraint in production (0 = off, default: 0)
  seed: number;                   // Random seed for velocities + Langevin noise (0 = auto)
}

export const DEFAULT_MD_CONFIG: MDConfig = {
  productionNs: 10,
  forceFieldPreset: 'ff19sb-opc',
  compoundId: '',
  temperatureK: 300,
  saltConcentrationM: 0.15,
  paddingNm: 1.2,
  restrainLigandNs: 0,
  seed: 0,
};

// Fixed parameters that don't change with preset
export const MD_COMMON_PARAMS = {
  temperature: 300,           // K
  saltConcentration: 0.15,    // M (150 mM)
  boxShape: 'dodecahedron',
  paddingNm: 1.2,             // nm
  timestepFs: 4,              // fs (HMR enabled for faster production)
  forceFieldLigand: 'OpenFF Sage 2.3.0',
  integrator: 'LangevinMiddle',
  equilibrationPs: 270,       // ~270 ps (AMBER-style with restraints: min, heat, equil, release)
} as const;

// Preset-specific force field parameters
export const MD_PRESET_PARAMS: Record<MDForceFieldPreset, {
  label: string;
  forceFieldProtein: string;
  forceFieldWater: string;
  description: string;
  folderSuffix: string;
  recommended?: boolean;
}> = {
  'ff19sb-opc': {
    label: 'ff19SB + OPC',
    forceFieldProtein: 'ff19SB (AMBER)',
    forceFieldWater: 'OPC (4-site)',
    description: 'Best accuracy — modern AMBER with 4-site water, gold standard for drug discovery',
    folderSuffix: 'ff19sb-OPC',
    recommended: true,
  },
  'ff14sb-tip3p': {
    label: 'ff14SB + TIP3P',
    forceFieldProtein: 'ff14SB (AMBER)',
    forceFieldWater: 'TIP3P',
    description: 'Classic AMBER, fast (3-site water)',
    folderSuffix: 'ff14sb-TIP3P',
  },
  'ff19sb-opc3': {
    label: 'ff19SB + OPC3',
    forceFieldProtein: 'ff19SB (AMBER)',
    forceFieldWater: 'OPC3',
    description: 'Fast modern AMBER (3-site, nearly OPC accuracy)',
    folderSuffix: 'ff19sb-OPC3',
  },
  'charmm36-mtip3p': {
    label: 'CHARMM36 + mTIP3P',
    forceFieldProtein: 'CHARMM36',
    forceFieldWater: 'mTIP3P',
    description: 'Mature all-atom force field — strong for membranes, nucleic acids, and general proteins',
    folderSuffix: 'charmm36-mTIP3P',
  },
};

// Legacy export for backwards compatibility
export const MD_FIXED_PARAMS = {
  ...MD_COMMON_PARAMS,
  forceFieldProtein: MD_PRESET_PARAMS['ff14sb-tip3p'].forceFieldProtein,
  forceFieldWater: MD_PRESET_PARAMS['ff14sb-tip3p'].forceFieldWater,
} as const;

export interface MDSystemInfo {
  atomCount: number;          // Total atoms in solvated system
  boxVolumeA3: number;        // Box volume in A^3
}

export interface MDBenchmarkResult {
  nsPerDay: number;           // Estimated throughput
  estimatedHours: number;     // For production_ns duration
  systemInfo: MDSystemInfo;
}

export type MDStage =
  | 'building'
  | 'parameterizing'
  | 'min_restrained'
  | 'min_unrestrained'
  | 'heating'
  | 'npt_restrained'
  | 'release'
  | 'equilibration'
  | 'production'
  | 'benchmark'
  | 'clustering'
  | 'scoring'
  | 'report';

export interface MDProgress {
  stage: MDStage;
  progress: number;           // 0-100 for current stage
  systemInfo?: MDSystemInfo;  // Populated after building
}

export interface MDResult {
  systemPdbPath: string;      // Solvated system (system.pdb)
  trajectoryPath: string;     // Full trajectory (trajectory.dcd)
  equilibratedPdbPath: string; // Post-equilibration frame
  finalPdbPath: string;       // Final frame (final.pdb)
  energyCsvPath: string;      // Energy timeseries (energy.csv)
}

// Output data for MD progress events
export interface MDOutputData {
  type: 'stdout' | 'stderr';
  data: string;
}

// Ligand loaded from docking output for MD
export interface MDLoadedLigand {
  name: string;           // e.g., "mol_002"
  sdfPath: string;        // Path to *_docked.sdf.gz
  smiles: string;         // SMILES string extracted from SDF
  vinaAffinity: number;   // Best pose Vina affinity (from SDF properties)
  qed: number;            // QED calculated from structure
  mw?: number;            // Molecular weight
  logp?: number;          // LogP
  cordialPHighAffinity?: number;  // CORDIAL P(pKd >= 6) if available
  cordialExpectedPkd?: number;    // CORDIAL expected pKd if available
  thumbnail?: string;     // Base64 encoded 2D thumbnail
}

// Docking output directory structure for MD loading
export interface MDDockOutput {
  receptorPdb: string;           // receptor_prepared.pdb
  ligands: MDLoadedLigand[];     // Sorted by vinaAffinity ascending (best first)
}
