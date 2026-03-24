// Copyright (c) 2026 Ember Contributors. MIT License.
/**
 * IPC channel types and payloads
 */

// === Configuration types ===

export interface SamplingThresholds {
  focalThreshold: number;
  posThreshold: number;
  elementThreshold: number;
}

export interface SamplingConfig {
  seed: number;
  numSamples: number;
  beamSize: number;
  maxSteps: number;
  threshold: SamplingThresholds;
  nextThreshold: SamplingThresholds;
  queueSameSmiTolerance: number;
}

// Default sampling config (matches FragGen defaults)
export const DEFAULT_SAMPLING_CONFIG: SamplingConfig = {
  seed: 2020,
  numSamples: 500,
  beamSize: 500,
  maxSteps: 50,
  threshold: {
    focalThreshold: 0.5,
    posThreshold: 0.2,
    elementThreshold: 0.2,
  },
  nextThreshold: {
    focalThreshold: 0.25,
    posThreshold: 0.1,
    elementThreshold: 0.2,
  },
  queueSameSmiTolerance: 2,
};

// === Request/Response types ===

export interface PrepPdbOptions {
  ligandName?: string;
  pocketRadius?: number;
}

export interface PrepPdbResult {
  pocketPdb: string;
  ligandPdb: string;
  outputDir: string;
}

export interface SurfaceResult {
  surfaceFile: string;
}

export type ModelVariant = 'dihedral' | 'cartesian' | 'geomopt';
export type Device = 'cpu' | 'mps' | 'cuda';
export type GenerationMode = 'denovo' | 'grow';

export interface AnchorValidationResult {
  valid: boolean;
  atomCount: number;
  has3DCoords: boolean;
  errorMessage?: string;
}

export interface GenerationOptions {
  surfacePly: string;
  pocketPdb: string;
  ligandPdb: string | null;
  outputDir: string;
  modelVariant: ModelVariant;
  device: Device;
  sampling: SamplingConfig;
  pocketRadius: number;
  // Anchor mode fields
  generationMode: GenerationMode;
  anchorSdfPath: string | null;  // Path to anchor SDF (null for denovo)
}

export interface GenerationResult {
  outputDir: string;
  sdfDir: string;
  paramsFile: string;
}

export interface FileInfo {
  exists: boolean;
  size?: number;
  modified?: Date;
}

export interface DockLigandPreoptResult {
  optimizedLigandPaths: string[];
  optimizedCount: number;
  failedCount: number;
}

// === Run parameters log ===

export interface RunParameters {
  timestamp: string;
  inputPdb: string;
  modelVariant: ModelVariant;
  device: Device;
  pocketRadius: number;
  sampling: SamplingConfig;
  outputDir: string;
  pocketPdb: string;
  ligandPdb: string | null;
  surfacePly: string;
}

// === Output stream types ===

export interface OutputData {
  type: 'stdout' | 'stderr';
  data: string;
}

// === Stats types ===

export interface GenerationStats {
  totalMoleculesGenerated: number;
  sessionsCount: number;
  lastGenerationDate: string | null;
}

// === IPC Channel names ===

export const IpcChannels = {
  // Invoke channels (renderer -> main -> renderer)
  SELECT_PDB_FILE: 'select-pdb-file',
  SELECT_STRUCTURE_FILES_MULTI: 'select-structure-files-multi',
  SELECT_OUTPUT_FOLDER: 'select-output-folder',
  PREP_PDB: 'prep-pdb',
  GENERATE_SURFACE: 'generate-surface',
  RUN_GENERATION: 'run-generation',
  GENERATE_RESULTS_CSV: 'generate-results-csv',
  GENERATE_THUMBNAIL: 'generate-thumbnail',
  SELECT_PDB_FILES_MULTI: 'select-pdb-files-multi',
  FILE_EXISTS: 'file-exists',
  GET_FILE_INFO: 'get-file-info',
  CREATE_DIRECTORY: 'create-directory',
  LIST_SDF_FILES: 'list-sdf-files',
  OPEN_FOLDER: 'open-folder',
  GET_AVAILABLE_DEVICES: 'get-available-devices',
  GET_STATS: 'get-stats',
  UPDATE_STATS: 'update-stats',
  CHECK_JOB_EXISTS: 'check-job-exists',
  VALIDATE_ANCHOR_SDF: 'validate-anchor-sdf',

  // Docking channels
  SELECT_CSV_FILE: 'select-csv-file',
  SELECT_SDF_FILE: 'select-sdf-file',
  RUN_VINA_DOCKING: 'dock:run-vina',
  CANCEL_VINA_DOCKING: 'dock:cancel',
  PARSE_DOCK_RESULTS: 'dock:parse-results',
  LIST_SDF_IN_DIRECTORY: 'list-sdf-in-directory',
  DETECT_PDB_LIGANDS: 'detect-pdb-ligands',
  EXTRACT_LIGAND: 'extract-ligand',
  PREPARE_RECEPTOR: 'prepare-receptor',
  PREPARE_DOCKING_COMPLEX: 'prepare-docking-complex',
  EXPORT_DOCK_CSV: 'dock:export-csv',
  EXPORT_COMPLEX_PDB: 'export-complex-pdb',
  GET_CPU_COUNT: 'get-cpu-count',
  // Multi-input ligand source channels
  SELECT_MOLECULE_FILES_MULTI: 'select-molecule-files-multi',
  IMPORT_MOLECULE_FILES: 'import-molecule-files',
  SELECT_FOLDER: 'select-folder',
  SCAN_SDF_DIRECTORY: 'scan-sdf-directory',
  PARSE_SMILES_CSV: 'parse-smiles-csv',
  CONVERT_SMILES_LIST: 'convert-smiles-list',
  CONVERT_SINGLE_MOLECULE: 'convert-single-molecule',
  EXTRACT_XRAY_LIGAND: 'extract-xray-ligand',
  ENUMERATE_PROTONATION: 'enumerate-protonation',
  ENUMERATE_STEREOISOMERS: 'enumerate-stereoisomers',
  GENERATE_CONFORMERS: 'generate-conformers',
  PREOPTIMIZE_DOCK_LIGANDS: 'dock:preoptimize-ligands',
  SCORE_DOCKING_XTB_ENERGY: 'dock:score-xtb-energy',
  // CORDIAL rescoring
  CHECK_CORDIAL_INSTALLED: 'check-cordial-installed',
  RUN_CORDIAL_SCORING: 'run-cordial-scoring',

  // Viewer channels
  PREPARE_FOR_VIEWING: 'prepare-for-viewing',
  SAVE_PDB_FILE: 'save-pdb-file',

  // File writing
  WRITE_TEXT_FILE: 'write-text-file',

  // MD simulation channels
  LOAD_DOCK_OUTPUT_FOR_MD: 'md:load-dock-output',
  RUN_MD_BENCHMARK: 'md:benchmark',
  CANCEL_MD_BENCHMARK: 'md:cancel-benchmark',
  RUN_MD_SIMULATION: 'md:simulate',
  CANCEL_MD_SIMULATION: 'md:cancel',
  PAUSE_MD_SIMULATION: 'md:pause',
  RESUME_MD_SIMULATION: 'md:resume',

  // Trajectory viewer channels
  SELECT_DCD_FILE: 'select-dcd-file',
  LIST_PDB_IN_DIRECTORY: 'list-pdb-in-directory',
  GET_TRAJECTORY_INFO: 'get-trajectory-info',
  GET_TRAJECTORY_FRAME: 'get-trajectory-frame',
  GET_TRAJECTORY_COORDS: 'get-trajectory-coords',
  CLUSTER_TRAJECTORY: 'cluster-trajectory',
  SCAN_CLUSTER_DIRECTORY: 'scan-cluster-directory',
  LOAD_ALIGNED_CLUSTERS: 'load-aligned-clusters',
  EXPORT_TRAJECTORY_FRAME: 'export-trajectory-frame',
  ANALYZE_TRAJECTORY: 'analyze-trajectory',
  GENERATE_MD_REPORT: 'generate-md-report',
  SCAN_XRAY_DIRECTORY: 'xray:scan-directory',
  RUN_XRAY_ANALYSIS: 'xray:run-analysis',
  SCORE_MD_CLUSTERS: 'md:score-clusters',
  LOAD_MD_TORSION_ANALYSIS: 'md:load-torsion-analysis',
  SCORE_COMPLEX: 'score-complex',
  MAP_BINDING_SITE: 'map-binding-site',
  COMPUTE_SURFACE_PROPS: 'compute-surface-props',

  // Pocket map channels
  COMPUTE_POCKET_MAP: 'compute-pocket-map',

  // FEP scoring channels
  RUN_FEP_SCORING: 'fep:run',
  CANCEL_FEP_SCORING: 'fep:cancel',

  // Project browser channels
  ENSURE_PROJECT: 'ensure-project',
  SCAN_PROJECTS: 'scan-projects',
  SCAN_RUN_FILES: 'scan-run-files',
  IMPORT_STRUCTURE: 'import-structure',
  FETCH_PDB: 'fetch-pdb',
  RENAME_PROJECT: 'rename-project',
  DELETE_PROJECT: 'delete-project',
  GET_PROJECT_FILE_COUNT: 'get-project-file-count',
  SCAN_PROJECT_ARTIFACTS: 'scan-project-artifacts',
  SELECT_EMBER_JOB_FOLDER: 'select-ember-job-folder',

  // Molecule alignment
  ALIGN_MOLECULES_MCS: 'align:mcs',
  ALIGN_DETECT_SCAFFOLDS: 'align:detect-scaffolds',
  ALIGN_BY_SCAFFOLD: 'align:by-scaffold',

  // Conformer generation (standalone)
  RUN_CONFORM_GENERATION: 'conform:generate',

  // Send channels (main -> renderer)
  PREP_OUTPUT: 'prep-output',
  SURFACE_OUTPUT: 'surface-output',
  GENERATION_OUTPUT: 'generation-output',
  DOCK_OUTPUT: 'dock:output',
  MD_OUTPUT: 'md:output',
  CONFORM_OUTPUT: 'conform:output',
  XRAY_OUTPUT: 'xray:output',
} as const;

export type IpcChannel = (typeof IpcChannels)[keyof typeof IpcChannels];

// === Trajectory types ===

export interface TrajectoryInfoResult {
  frameCount: number;
  timestepPs: number;      // Timestep in picoseconds between frames
  totalTimeNs: number;     // Total simulation time in nanoseconds
}

export interface ClusteringOptions {
  topologyPath: string;      // PDB file path
  trajectoryPath: string;    // DCD file path
  numClusters: number;
  method: 'kmeans' | 'dbscan' | 'hierarchical';
  rmsdSelection: 'ligand' | 'backbone' | 'all';  // Which atoms to use for RMSD
  stripWaters: boolean;
  outputDir: string;
}

export interface ClusterResultData {
  clusterId: number;
  frameCount: number;
  population: number;        // Percentage
  centroidFrame: number;
  centroidPdbPath?: string;  // Path to centroid PDB if saved
}

export interface ClusteringResult {
  clusters: ClusterResultData[];
  frameAssignments: number[];  // Cluster ID for each frame
  outputDir: string;
  requestedClusters?: number;
  actualClusters?: number;
}

// Scored cluster result from holo MD fixed-pose rescoring
export interface ScoredClusterResult {
  clusterId: number;
  frameCount: number;
  population: number;           // Percentage
  centroidFrame: number;
  centroidPdbPath: string;
  receptorPdbPath?: string;
  ligandSdfPath?: string;
  vinaRescore?: number;         // kcal/mol
  cordialExpectedPkd?: number;
  cordialPHighAffinity?: number;
  cordialPVeryHighAffinity?: number;
  xtbStrainKcal?: number;         // kcal/mol, GFN2-xTB relative energy
}

export interface ScoreMdClustersOptions {
  topologyPath: string;
  trajectoryPath: string;
  outputDir: string;
  inputLigandSdf: string;
  inputReceptorPdb?: string;
  numClusters: number;
  enableVina: boolean;
  enableCordial: boolean;
}

export interface ScoreMdClustersResult {
  clusters: ScoredClusterResult[];
  outputDir: string;
  clusteringResults: ClusteringResult;
}

export interface ExportFrameOptions {
  topologyPath: string;
  trajectoryPath: string;
  frameIndex: number;
  outputPath: string;
  stripWaters?: boolean;
}

export interface AnalysisOptions {
  topologyPath: string;
  trajectoryPath: string;
  analysisType: 'rmsd' | 'rmsf' | 'hbonds' | 'contacts';
  outputDir: string;
  ligandSelection?: string;  // Optional ligand selection for ligand-specific analyses
}

export interface RmsdAnalysisResult {
  stats: {
    proteinMean: number;
    proteinStd: number;
    ligandMean?: number;
    ligandStd?: number;
  };
}

export interface RmsfAnalysisResult {
  residueIndices: number[];
  rmsf: number[];
  stats: {
    mean: number;
    std: number;
    max: number;
  };
}

export interface HbondAnalysisResult {
  hbonds: Array<{
    donor: string;
    acceptor: string;
    occupancy: number;      // Percentage of frames with this H-bond
  }>;
  totalUnique: number;
  nFrames: number;
}

export interface AnalysisResult {
  type: 'rmsd' | 'rmsf' | 'hbonds' | 'contacts';
  plotPath: string;
  csvPath?: string;
  data: RmsdAnalysisResult | RmsfAnalysisResult | HbondAnalysisResult;
}

export interface XrayAnalysisResult {
  inputDir: string;
  outputDir: string;
  pdfPaths: string[];
}

export interface XrayDirectoryScanResult {
  pdbCount: number;
  mtzCount: number;
  pairedCount: number;
  unpairedPdbCount: number;
}

export interface MdReportOptions {
  topologyPath: string;
  trajectoryPath: string;
  outputDir: string;
  ligandSelection?: string;
  ligandSdf?: string;
  simInfo?: {
    jobName?: string;
    atoms?: string;
    waters?: string;
    temperature?: string;
    duration?: string;
    forceField?: string;
    platform?: string;
    performance?: string;
  };
}

export interface MdReportResult {
  reportPath: string;       // Path to generated full_report.pdf
  analysisDir: string;      // Directory containing analysis outputs
  sectionPdfs: string[];    // Paths to individual section PDFs
  clusteringResults?: ClusterResultData[];  // Clustering results if available
}

export interface MdLigandDepictionBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

export interface MdLigandDepictionAtom {
  atomIndex: number;
  symbol: string;
  label: string;
  x: number;
  y: number;
  formalCharge: number;
  isAromatic: boolean;
  showLabel: boolean;
}

export interface MdLigandDepictionBond {
  bondId: string;
  bondIndex: number;
  beginAtomIndex: number;
  endAtomIndex: number;
  order: number;
  isAromatic: boolean;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface MdLigandDepiction {
  atoms: MdLigandDepictionAtom[];
  bonds: MdLigandDepictionBond[];
  bounds: MdLigandDepictionBounds;
}

export interface MdTorsionClusterValue {
  clusterId: number;
  centroidFrame: number;
  population: number;
  angle: number;
}

export interface MdTorsionEntry {
  torsionId: string;
  bondId: string;
  bondIndex: number;
  centralBondAtomIndices: number[];
  quartetAtomIndices: number[];
  atomNames: string[];
  label: string;
  circularMean: number;
  circularStd: number;
  min: number;
  max: number;
  median: number;
  nFrames: number;
  trajectoryAngles: number[];
  clusterValues: MdTorsionClusterValue[];
}

export interface MdTorsionAnalysis {
  type: 'torsions';
  pdfPath: string | null;
  csvPath: string | null;
  ligandPresent: boolean;
  ligandSdfPath?: string | null;
  nFrames: number;
  nSampledFrames: number;
  stride: number;
  sampledFrameIndices: number[];
  nRotatableBonds: number;
  depiction: MdLigandDepiction | null;
  data: {
    torsions: MdTorsionEntry[];
  };
}

export interface LoadMdTorsionAnalysisOptions {
  analysisDir: string;
}

// Cluster directory scanning types
export interface ScannedCluster {
  clusterId: number;
  pdbPath: string;
  population: number;       // Percentage from clustering results
}

export interface ScanClusterDirectoryResult {
  clusters: ScannedCluster[];
  clusteringResultsPath?: string;  // Path to clustering_results.json if found
}

export interface LoadedClusterPdb {
  clusterId: number;
  pdbPath: string;          // Original PDB path
  alignedPath: string;      // Path to aligned PDB file
  population: number;
}

// === Project browser types ===

export interface ProjectRunInfo {
  folderName: string;
  path: string;
  lastModified: number;
  hasTrajectory: boolean;
  hasFinalPdb: boolean;
}

export interface ProjectInfo {
  name: string;
  path: string;
  runs: ProjectRunInfo[];
  lastModified: number;
}

export interface RunFilesResult {
  systemPdb: string | null;
  trajectory: string | null;
  finalPdb: string | null;
  equilibratedPdb: string | null;
  energyCsv: string | null;
}

// === Project job types (for job selector dropdown) ===

export interface ProjectJobPose {
  name: string;
  path: string;
  affinity?: number;
}

export interface ProjectJob {
  id: string;               // Unique key: "dock:Vina_HWF" or "sim:ff19sb-OPC_MD-300K-1ns"
  type: 'docking' | 'docking-pose' | 'simulation' | 'conformer' | 'map';
  folder: string;           // Run folder name (e.g., "Vina_HWF")
  label: string;            // Display name
  path: string;             // Root path of the job directory
  parentId?: string;
  parentLabel?: string;
  sortKey?: number;
  lastModified?: number;

  // Docking-specific
  receptorPdb?: string;
  poses?: ProjectJobPose[];
  ligandPath?: string;
  poseIndex?: number;
  preparedLigandPath?: string;
  referenceLigandPath?: string;
  holoPdb?: string;

  // Simulation-specific
  systemPdb?: string;
  trajectoryDcd?: string;
  finalPdb?: string;
  hasTrajectory?: boolean;
  clusterCount?: number;
  clusterDir?: string;
  clusteringResultsPath?: string;

  // Conformer-specific
  conformerPaths?: string[];
  conformerCount?: number;

  // Map-specific
  mapMethod?: PocketMapMethod;
  mapResultJson?: string;
  mapPdb?: string;
  mapTrajectoryDcd?: string;
  hotspotCount?: number;
}

// Keep legacy alias for backward compat during transition
export type ProjectArtifact = ProjectJob;
export type ProjectArtifactPose = ProjectJobPose;

// === Binding site interaction map types ===

export interface BindingSiteMapOptions {
  pdbPath: string;
  ligandResname: string;
  ligandResnum: number;
  outputDir: string;
  sourcePdbPath?: string;
  sourceTrajectoryPath?: string;
  boxPadding?: number;
  gridSpacing?: number;
}

export interface BindingSiteHotspot {
  type: 'hydrophobic' | 'hbond_donor' | 'hbond_acceptor';
  position: [number, number, number];
  direction: [number, number, number];
  score: number;
}

export interface BindingSiteMapResult {
  hydrophobicDx: string;
  hbondDonorDx: string;
  hbondAcceptorDx: string;
  hotspots: BindingSiteHotspot[];
  gridDimensions: [number, number, number];
  ligandCom: [number, number, number];
}

// === Pocket map types ===

export type PocketMapMethod = 'solvation';

export interface PocketMapOptions {
  method: PocketMapMethod;
  pdbPath: string;
  ligandResname: string;
  ligandResnum: number;
  outputDir: string;
  trajectoryPath?: string;  // For solvation with existing trajectory
  sourcePdbPath?: string;
  sourceTrajectoryPath?: string;
  boxPadding?: number;
  gridSpacing?: number;
}

// === Computed surface property types ===

export interface SurfacePropsResult {
  atomCount: number;
  hydrophobic: number[];    // current-structure atom field, normalized to [-1, 1]
  electrostatic: number[];  // current-structure electrostatic potential, normalized to [-1, 1]
  cachedPath: string;       // where the JSON was stored
}

// === FEP scoring types ===

export type FepSpeedPreset = 'fast' | 'accurate';

export interface FepScoringOptions {
  topologyPath: string;
  trajectoryPath: string;
  startNs: number;
  endNs: number;
  numSnapshots: number;        // 3, 5, or 7
  speedPreset: FepSpeedPreset;
  outputDir: string;
  forceFieldPreset: string;
  ligandSdf?: string;
}

export interface FepSnapshotResult {
  snapshotIndex: number;
  frameIndex: number;
  timeNs: number;
  deltaG_complex: number;
  deltaG_solvent: number;
  deltaG_bind: number;
  uncertainty: number;
}

export interface FepScoringResult {
  snapshots: FepSnapshotResult[];
  meanDeltaG: number;
  sem: number;
  outputDir: string;
}
