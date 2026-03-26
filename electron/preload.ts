// Copyright (c) 2026 Ember Contributors. MIT License.
import { contextBridge, ipcRenderer } from 'electron';

// webUtils.getPathForFile available in Electron 22.3+ but types not exported until Electron 28
const webUtils = (require('electron') as any).webUtils;

// Inline IPC channel names (can't require external modules in preload sandbox)
const IpcChannels = {
  SELECT_PDB_FILE: 'select-pdb-file',
  SELECT_PDB_FILES_MULTI: 'select-pdb-files-multi',
  SELECT_STRUCTURE_FILES_MULTI: 'select-structure-files-multi',
  SELECT_OUTPUT_FOLDER: 'select-output-folder',
  PREP_PDB: 'prep-pdb',
  GENERATE_SURFACE: 'generate-surface',
  RUN_GENERATION: 'run-generation',
  GENERATE_RESULTS_CSV: 'generate-results-csv',
  GENERATE_THUMBNAIL: 'generate-thumbnail',
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
  // Conformer generation (standalone)
  CONFORM_OUTPUT: 'conform:output',
  RUN_CONFORM_GENERATION: 'conform:generate',
  // Post-dock refinement
  REFINE_POSES: 'dock:refine-poses',
  // CORDIAL rescoring
  CHECK_CORDIAL_INSTALLED: 'check-cordial-installed',
  RUN_CORDIAL_SCORING: 'run-cordial-scoring',
  // Viewer channels
  PREPARE_FOR_VIEWING: 'prepare-for-viewing',
  SAVE_PDB_FILE: 'save-pdb-file',
  // File writing
  WRITE_TEXT_FILE: 'write-text-file',
  // MD channels
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
  SCAN_XRAY_DIRECTORY: 'xray:scan-directory',
  GET_TRAJECTORY_INFO: 'get-trajectory-info',
  GET_TRAJECTORY_FRAME: 'get-trajectory-frame',
  GET_TRAJECTORY_COORDS: 'get-trajectory-coords',
  CLUSTER_TRAJECTORY: 'cluster-trajectory',
  SCAN_CLUSTER_DIRECTORY: 'scan-cluster-directory',
  LOAD_ALIGNED_CLUSTERS: 'load-aligned-clusters',
  EXPORT_TRAJECTORY_FRAME: 'export-trajectory-frame',
  ANALYZE_TRAJECTORY: 'analyze-trajectory',
  GENERATE_MD_REPORT: 'generate-md-report',
  RUN_XRAY_ANALYSIS: 'xray:run-analysis',
  SCORE_MD_CLUSTERS: 'md:score-clusters',
  LOAD_MD_TORSION_ANALYSIS: 'md:load-torsion-analysis',
  SCORE_COMPLEX: 'score-complex',
  MAP_BINDING_SITE: 'map-binding-site',
  COMPUTE_POCKET_MAP: 'compute-pocket-map',
  COMPUTE_SURFACE_PROPS: 'compute-surface-props',
  // FEP scoring channels
  RUN_FEP_SCORING: 'fep:run',
  CANCEL_FEP_SCORING: 'fep:cancel',
  // Project browser channels
  ENSURE_PROJECT: 'ensure-project',
  SCAN_PROJECTS: 'scan-projects',
  SCAN_RUN_FILES: 'scan-run-files',
  IMPORT_STRUCTURE: 'import-structure',
  RENAME_PROJECT: 'rename-project',
  DELETE_PROJECT: 'delete-project',
  GET_PROJECT_FILE_COUNT: 'get-project-file-count',
  SCAN_PROJECT_ARTIFACTS: 'scan-project-artifacts',
  SELECT_EMBER_JOB_FOLDER: 'select-ember-job-folder',
  OPEN_PROJECT_FOLDER: 'open-project-folder',
  IMPORT_EXTERNAL_PROJECT: 'import-external-project',
  GET_HOME_DIR: 'get-home-dir',
  SET_HOME_DIR: 'set-home-dir',
  // PDB ID fetch
  FETCH_PDB: 'fetch-pdb',
  // Image reading channel
  READ_IMAGE_AS_DATA_URL: 'read-image-as-data-url',
  // Receptor prep cancellation
  CANCEL_PREP: 'cancel-prep',
  // Score tab channels
  SCORE_BATCH: 'score:batch',
  SCORE_TRAJECTORY: 'score:trajectory',
  CANCEL_SCORE_BATCH: 'score:cancel',
  EXPORT_SCORE_CSV: 'score:export-csv',
  // Molecule details
  GET_MOLECULE_DETAILS: 'get-molecule-details',
  // Settings: last seen version
  GET_LAST_SEEN_VERSION: 'get-last-seen-version',
  SET_LAST_SEEN_VERSION: 'set-last-seen-version',
  // Send channels
  PREP_OUTPUT: 'prep-output',
  SURFACE_OUTPUT: 'surface-output',
  GENERATION_OUTPUT: 'generation-output',
  DOCK_OUTPUT: 'dock:output',
  MD_OUTPUT: 'md:output',
  XRAY_OUTPUT: 'xray:output',
  SCORE_OUTPUT: 'score:output',
  PREP_PROGRESS: 'prep:progress',
} as const;

interface OutputData {
  type: 'stdout' | 'stderr';
  data: string;
}

interface PrepPdbOptions {
  ligandName?: string;
  pocketRadius?: number;
}

interface GenerationOptions {
  surfacePly: string;
  pocketPdb: string;
  ligandPdb: string | null;
  outputDir: string;
  modelVariant: string;
  device: string;
  sampling: unknown;
  pocketRadius: number;
  // Anchor mode fields
  generationMode: 'denovo' | 'grow';
  anchorSdfPath: string | null;
}

interface DockConfig {
  exhaustiveness: number;
  numPoses: number;
  autoboxAdd: number;
  numCpus: number;
  seed: number;
}

interface MDConfig {
  productionNs: number;
  forceFieldPreset: string;
}

interface MDBenchmarkResult {
  nsPerDay: number;
  systemInfo: {
    atomCount: number;
    boxVolumeA3: number;
  };
}

interface TrajectoryInfoResult {
  frameCount: number;
  timestepPs: number;
  totalTimeNs: number;
}

interface ClusteringOptions {
  topologyPath: string;
  trajectoryPath: string;
  numClusters: number;
  method: 'kmeans' | 'dbscan' | 'hierarchical';
  rmsdSelection: 'ligand' | 'backbone' | 'all';
  stripWaters: boolean;
  outputDir: string;
}

interface ExportFrameOptions {
  topologyPath: string;
  trajectoryPath: string;
  frameIndex: number;
  outputPath: string;
  stripWaters?: boolean;
}

interface AnalysisOptions {
  topologyPath: string;
  trajectoryPath: string;
  analysisType: 'rmsd' | 'rmsf' | 'hbonds' | 'contacts';
  outputDir: string;
  ligandSelection?: string;
}

interface MdReportOptions {
  topologyPath: string;
  trajectoryPath: string;
  outputDir: string;
  ligandSelection?: string;
  ligandSdf?: string;
  simInfo?: Record<string, string>;
}

interface LoadMdTorsionAnalysisOptions {
  analysisDir: string;
}

interface ScoreMdClustersOptions {
  topologyPath: string;
  trajectoryPath: string;
  outputDir: string;
  inputLigandSdf: string;
  inputReceptorPdb?: string;
  numClusters: number;
  enableVina: boolean;
  enableCordial: boolean;
}

interface BindingSiteMapOptions {
  pdbPath: string;
  ligandResname: string;
  ligandResnum: number;
  outputDir: string;
  sourcePdbPath?: string;
  sourceTrajectoryPath?: string;
  boxPadding?: number;
  gridSpacing?: number;
}

const electronAPI = {
  // File operations
  selectPdbFile: (defaultPath?: string) => ipcRenderer.invoke(IpcChannels.SELECT_PDB_FILE, defaultPath),
  selectPdbFilesMulti: () => ipcRenderer.invoke(IpcChannels.SELECT_PDB_FILES_MULTI),
  selectStructureFilesMulti: () => ipcRenderer.invoke(IpcChannels.SELECT_STRUCTURE_FILES_MULTI),
  selectOutputFolder: () => ipcRenderer.invoke(IpcChannels.SELECT_OUTPUT_FOLDER),
  fileExists: (path: string) => ipcRenderer.invoke(IpcChannels.FILE_EXISTS, path),
  getFileInfo: (path: string) => ipcRenderer.invoke(IpcChannels.GET_FILE_INFO, path),
  createDirectory: (dirPath: string) => ipcRenderer.invoke(IpcChannels.CREATE_DIRECTORY, dirPath),
  listSdfFiles: (dirPath: string) => ipcRenderer.invoke(IpcChannels.LIST_SDF_FILES, dirPath),
  listPdbInDirectory: (dirPath: string) => ipcRenderer.invoke(IpcChannels.LIST_PDB_IN_DIRECTORY, dirPath),
  scanXrayDirectory: (dirPath: string) => ipcRenderer.invoke(IpcChannels.SCAN_XRAY_DIRECTORY, dirPath),
  openFolder: (folderPath: string) => ipcRenderer.invoke(IpcChannels.OPEN_FOLDER, folderPath),

  // Preparation steps
  prepPdb: (pdbPath: string, outputDir: string, options?: PrepPdbOptions) =>
    ipcRenderer.invoke(IpcChannels.PREP_PDB, pdbPath, outputDir, options),

  generateSurface: (pocketPdb: string, ligandPdb: string, outputPly: string) =>
    ipcRenderer.invoke(IpcChannels.GENERATE_SURFACE, pocketPdb, ligandPdb, outputPly),

  // Generation
  runGeneration: (options: GenerationOptions) =>
    ipcRenderer.invoke(IpcChannels.RUN_GENERATION, options),

  // Results CSV
  generateResultsCsv: (sdfDir: string, outputCsv: string) =>
    ipcRenderer.invoke(IpcChannels.GENERATE_RESULTS_CSV, sdfDir, outputCsv),

  // Thumbnail
  generateThumbnail: (sdfPath: string) =>
    ipcRenderer.invoke(IpcChannels.GENERATE_THUMBNAIL, sdfPath),

  // Device info
  getAvailableDevices: () => ipcRenderer.invoke(IpcChannels.GET_AVAILABLE_DEVICES),
  getCpuCount: () => ipcRenderer.invoke(IpcChannels.GET_CPU_COUNT),

  // Stats
  getStats: () => ipcRenderer.invoke(IpcChannels.GET_STATS),
  updateStats: (moleculeCount: number) => ipcRenderer.invoke(IpcChannels.UPDATE_STATS, moleculeCount),

  // Job management
  checkJobExists: (outputFolder: string, jobName: string) =>
    ipcRenderer.invoke(IpcChannels.CHECK_JOB_EXISTS, outputFolder, jobName),

  // Anchor SDF validation for fragment growing mode
  validateAnchorSdf: (sdfPath: string) =>
    ipcRenderer.invoke(IpcChannels.VALIDATE_ANCHOR_SDF, sdfPath),

  // Event listeners (return cleanup functions)
  onPrepOutput: (callback: (data: OutputData) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, data: OutputData) => callback(data);
    ipcRenderer.on(IpcChannels.PREP_OUTPUT, listener);
    return () => ipcRenderer.removeListener(IpcChannels.PREP_OUTPUT, listener);
  },

  onSurfaceOutput: (callback: (data: OutputData) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, data: OutputData) => callback(data);
    ipcRenderer.on(IpcChannels.SURFACE_OUTPUT, listener);
    return () => ipcRenderer.removeListener(IpcChannels.SURFACE_OUTPUT, listener);
  },

  onGenerationOutput: (callback: (data: OutputData) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, data: OutputData) => callback(data);
    ipcRenderer.on(IpcChannels.GENERATION_OUTPUT, listener);
    return () => ipcRenderer.removeListener(IpcChannels.GENERATION_OUTPUT, listener);
  },

  // Docking operations
  selectCsvFile: () => ipcRenderer.invoke(IpcChannels.SELECT_CSV_FILE),
  selectSdfFile: () => ipcRenderer.invoke(IpcChannels.SELECT_SDF_FILE),

  savePdbFile: (content: string, defaultName?: string) =>
    ipcRenderer.invoke(IpcChannels.SAVE_PDB_FILE, content, defaultName),

  runVinaDocking: (
    receptorPdb: string,
    referenceLigand: string,
    ligandSdfPaths: string[],
    outputDir: string,
    config: DockConfig
  ) => ipcRenderer.invoke(
    IpcChannels.RUN_VINA_DOCKING,
    receptorPdb,
    referenceLigand,
    ligandSdfPaths,
    outputDir,
    config
  ),

  cancelVinaDocking: () => ipcRenderer.invoke(IpcChannels.CANCEL_VINA_DOCKING),

  parseDockResults: (outputDir: string) =>
    ipcRenderer.invoke(IpcChannels.PARSE_DOCK_RESULTS, outputDir),

  listSdfInDirectory: (dirPath: string) =>
    ipcRenderer.invoke(IpcChannels.LIST_SDF_IN_DIRECTORY, dirPath),

  detectPdbLigands: (pdbPath: string) =>
    ipcRenderer.invoke(IpcChannels.DETECT_PDB_LIGANDS, pdbPath),

  extractLigand: (pdbPath: string, ligandId: string, outputPath: string) =>
    ipcRenderer.invoke(IpcChannels.EXTRACT_LIGAND, pdbPath, ligandId, outputPath),

  prepareReceptor: (
    pdbPath: string,
    ligandId: string,
    outputPath: string,
    waterDistance: number = 0,
    protonationPh: number = 7.4
  ) => ipcRenderer.invoke(
    IpcChannels.PREPARE_RECEPTOR,
    pdbPath,
    ligandId,
    outputPath,
    waterDistance,
    protonationPh
  ),

  cancelPrep: () => ipcRenderer.invoke(IpcChannels.CANCEL_PREP),

  onPrepProgress: (callback: (message: string) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, message: string) => callback(message);
    ipcRenderer.on(IpcChannels.PREP_PROGRESS, listener);
    return () => ipcRenderer.removeListener(IpcChannels.PREP_PROGRESS, listener);
  },

  prepareDockingComplex: (
    receptorPdb: string,
    xrayLigandSdf: string,
    outputDir: string,
    chargeMethod: 'gasteiger' | 'am1bcc' = 'am1bcc',
    phMin: number = 6.4,
    phMax: number = 8.4,
    protonateReference: boolean = true
  ) => ipcRenderer.invoke(
    IpcChannels.PREPARE_DOCKING_COMPLEX,
    receptorPdb,
    xrayLigandSdf,
    outputDir,
    chargeMethod,
    phMin,
    phMax,
    protonateReference
  ),

  exportDockCsv: (outputDir: string, csvOutput: string, bestOnly: boolean) =>
    ipcRenderer.invoke(IpcChannels.EXPORT_DOCK_CSV, outputDir, csvOutput, bestOnly),

  exportComplexPdb: (receptorPdb: string, ligandSdf: string, poseIndex: number, outputPath: string) =>
    ipcRenderer.invoke(IpcChannels.EXPORT_COMPLEX_PDB, receptorPdb, ligandSdf, poseIndex, outputPath),

  // Multi-input ligand source operations
  selectMoleculeFilesMulti: () => ipcRenderer.invoke(IpcChannels.SELECT_MOLECULE_FILES_MULTI),
  importMoleculeFiles: (filePaths: string[], outputDir: string) =>
    ipcRenderer.invoke(IpcChannels.IMPORT_MOLECULE_FILES, filePaths, outputDir),
  selectFolder: () => ipcRenderer.invoke(IpcChannels.SELECT_FOLDER),

  scanSdfDirectory: (dirPath: string, outputDir: string) =>
    ipcRenderer.invoke(IpcChannels.SCAN_SDF_DIRECTORY, dirPath, outputDir),

  parseSmilesCsv: (csvPath: string, outputDir: string) =>
    ipcRenderer.invoke(IpcChannels.PARSE_SMILES_CSV, csvPath, outputDir),

  convertSmilesList: (smilesList: string[], outputDir: string) =>
    ipcRenderer.invoke(IpcChannels.CONVERT_SMILES_LIST, smilesList, outputDir),

  convertSingleMolecule: (input: string, outputDir: string, inputType: 'smiles' | 'mol_file') =>
    ipcRenderer.invoke(IpcChannels.CONVERT_SINGLE_MOLECULE, input, outputDir, inputType),

  extractXrayLigand: (pdbPath: string, ligandId: string, outputDir: string, smiles?: string) =>
    ipcRenderer.invoke(IpcChannels.EXTRACT_XRAY_LIGAND, pdbPath, ligandId, outputDir, smiles),

  enumerateProtonation: (
    ligandSdfPaths: string[],
    outputDir: string,
    phMin: number,
    phMax: number
  ) => ipcRenderer.invoke(
    IpcChannels.ENUMERATE_PROTONATION,
    ligandSdfPaths,
    outputDir,
    phMin,
    phMax
  ),

  enumerateStereoisomers: (
    ligandSdfPaths: string[],
    outputDir: string,
    maxStereoisomers: number
  ) => ipcRenderer.invoke(
    IpcChannels.ENUMERATE_STEREOISOMERS,
    ligandSdfPaths,
    outputDir,
    maxStereoisomers
  ),

  generateConformers: (
    ligandSdfPaths: string[],
    outputDir: string,
    maxConformers: number,
    rmsdCutoff: number,
    energyWindow: number,
    method?: string,
    mcmmOptions?: { steps: number; temperature: number; sampleAmides: boolean }
  ) => ipcRenderer.invoke(
    IpcChannels.GENERATE_CONFORMERS,
    ligandSdfPaths,
    outputDir,
    maxConformers,
    rmsdCutoff,
    energyWindow,
    method,
    mcmmOptions
  ),

  preOptimizeDockLigands: (
    ligandSdfPaths: string[],
    outputDir: string
  ) => ipcRenderer.invoke(
    IpcChannels.PREOPTIMIZE_DOCK_LIGANDS,
    ligandSdfPaths,
    outputDir
  ),

  scoreDockingXtbEnergy: (dockOutputDir: string) =>
    ipcRenderer.invoke(IpcChannels.SCORE_DOCKING_XTB_ENERGY, dockOutputDir),

  // Conformer generation (standalone)
  runConformGeneration: (
    ligandSdfPath: string,
    outputDir: string,
    maxConformers: number,
    rmsdCutoff: number,
    energyWindow: number,
    method: string,
    mcmmOptions?: { steps: number; temperature: number; sampleAmides: boolean; xtbRerank?: boolean }
  ) => ipcRenderer.invoke(
    IpcChannels.RUN_CONFORM_GENERATION,
    ligandSdfPath,
    outputDir,
    maxConformers,
    rmsdCutoff,
    energyWindow,
    method,
    mcmmOptions
  ),

  onConformOutput: (callback: (data: OutputData) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, data: OutputData) => callback(data);
    ipcRenderer.on(IpcChannels.CONFORM_OUTPUT, listener);
    return () => ipcRenderer.removeListener(IpcChannels.CONFORM_OUTPUT, listener);
  },

  // Post-dock pocket refinement
  refinePoses: (
    receptorPdb: string,
    posesDir: string,
    outputDir: string,
    maxIterations: number,
    chargeMethod?: string
  ) => ipcRenderer.invoke(
    IpcChannels.REFINE_POSES,
    receptorPdb,
    posesDir,
    outputDir,
    maxIterations,
    chargeMethod
  ),

  // Complex scoring (viewer)
  scoreComplex: (pdbPath: string, ligandSdfPath?: string) =>
    ipcRenderer.invoke(IpcChannels.SCORE_COMPLEX, pdbPath, ligandSdfPath),

  // CORDIAL rescoring
  checkCordialInstalled: () => ipcRenderer.invoke(IpcChannels.CHECK_CORDIAL_INSTALLED),

  runCordialScoring: (dockOutputDir: string, batchSize: number) =>
    ipcRenderer.invoke(IpcChannels.RUN_CORDIAL_SCORING, dockOutputDir, batchSize),

  // Dock event listener
  onDockOutput: (callback: (data: OutputData) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, data: OutputData) => callback(data);
    ipcRenderer.on(IpcChannels.DOCK_OUTPUT, listener);
    return () => ipcRenderer.removeListener(IpcChannels.DOCK_OUTPUT, listener);
  },

  // MD simulation operations
  loadDockOutputForMd: (dirPath: string) =>
    ipcRenderer.invoke(IpcChannels.LOAD_DOCK_OUTPUT_FOR_MD, dirPath),

  runMdBenchmark: (
    receptorPdb: string | null,
    ligandSdf: string,
    outputDir: string,
    forceFieldPreset: string = 'ff19sb-opc',
    ligandOnly: boolean = false
  ) => ipcRenderer.invoke(
    IpcChannels.RUN_MD_BENCHMARK,
    receptorPdb,
    ligandSdf,
    outputDir,
    forceFieldPreset,
    ligandOnly
  ),

  runMdSimulation: (
    receptorPdb: string | null,
    ligandSdf: string,
    outputDir: string,
    config: MDConfig,
    ligandOnly: boolean = false,
    apo: boolean = false
  ) => ipcRenderer.invoke(
    IpcChannels.RUN_MD_SIMULATION,
    receptorPdb,
    ligandSdf,
    outputDir,
    config,
    ligandOnly,
    apo
  ),

  cancelMdBenchmark: () => ipcRenderer.invoke(IpcChannels.CANCEL_MD_BENCHMARK),
  cancelMdSimulation: () => ipcRenderer.invoke(IpcChannels.CANCEL_MD_SIMULATION),
  pauseMdSimulation: () => ipcRenderer.invoke(IpcChannels.PAUSE_MD_SIMULATION),
  resumeMdSimulation: () => ipcRenderer.invoke(IpcChannels.RESUME_MD_SIMULATION),

  // MD event listener
  onMdOutput: (callback: (data: OutputData) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, data: OutputData) => callback(data);
    ipcRenderer.on(IpcChannels.MD_OUTPUT, listener);
    return () => ipcRenderer.removeListener(IpcChannels.MD_OUTPUT, listener);
  },

  onXrayOutput: (callback: (data: OutputData) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, data: OutputData) => callback(data);
    ipcRenderer.on(IpcChannels.XRAY_OUTPUT, listener);
    return () => ipcRenderer.removeListener(IpcChannels.XRAY_OUTPUT, listener);
  },

  // Trajectory viewer operations
  selectDcdFile: () => ipcRenderer.invoke(IpcChannels.SELECT_DCD_FILE),

  getTrajectoryInfo: (topologyPath: string, trajectoryPath: string) =>
    ipcRenderer.invoke(IpcChannels.GET_TRAJECTORY_INFO, topologyPath, trajectoryPath),

  getTrajectoryFrame: (topologyPath: string, trajectoryPath: string, frameIndex: number) =>
    ipcRenderer.invoke(IpcChannels.GET_TRAJECTORY_FRAME, topologyPath, trajectoryPath, frameIndex),

  getTrajectoryCoords: (topologyPath: string, trajectoryPath: string, frameIndex: number) =>
    ipcRenderer.invoke(IpcChannels.GET_TRAJECTORY_COORDS, topologyPath, trajectoryPath, frameIndex),

  clusterTrajectory: (options: ClusteringOptions) =>
    ipcRenderer.invoke(IpcChannels.CLUSTER_TRAJECTORY, options),

  scanClusterDirectory: (directoryPath: string) =>
    ipcRenderer.invoke(IpcChannels.SCAN_CLUSTER_DIRECTORY, directoryPath),

  loadAlignedClusters: (directoryPath: string, clusterIds: number[]) =>
    ipcRenderer.invoke(IpcChannels.LOAD_ALIGNED_CLUSTERS, directoryPath, clusterIds),

  exportTrajectoryFrame: (options: ExportFrameOptions) =>
    ipcRenderer.invoke(IpcChannels.EXPORT_TRAJECTORY_FRAME, options),

  analyzeTrajectory: (options: AnalysisOptions) =>
    ipcRenderer.invoke(IpcChannels.ANALYZE_TRAJECTORY, options),

  generateMdReport: (options: MdReportOptions) =>
    ipcRenderer.invoke(IpcChannels.GENERATE_MD_REPORT, options),

  runXrayAnalysis: (inputDir: string, outputDir: string) =>
    ipcRenderer.invoke(IpcChannels.RUN_XRAY_ANALYSIS, inputDir, outputDir),

  // Score tab operations
  scoreBatch: (request: { entries: Array<{ id: string; name: string; pdbPath: string; ligandId: string | null; isPrepared: boolean }>; outputDir: string }) =>
    ipcRenderer.invoke(IpcChannels.SCORE_BATCH, request),
  scoreTrajectory: (request: { trajectoryPath: string; topologyPath: string; ligandSdfPath: string; numClusters: number; outputDir: string }) =>
    ipcRenderer.invoke(IpcChannels.SCORE_TRAJECTORY, request),
  cancelScoreBatch: () =>
    ipcRenderer.invoke(IpcChannels.CANCEL_SCORE_BATCH),
  onScoreOutput: (callback: (data: OutputData) => void) => {
    const listener = (_event: any, data: OutputData) => callback(data);
    ipcRenderer.on(IpcChannels.SCORE_OUTPUT, listener);
    return () => { ipcRenderer.removeListener(IpcChannels.SCORE_OUTPUT, listener); };
  },
  exportScoreCsv: (entries: string, csvPath: string) =>
    ipcRenderer.invoke(IpcChannels.EXPORT_SCORE_CSV, entries, csvPath),

  getMoleculeDetails: (sdfPath: string, referenceSdfPath?: string) =>
    ipcRenderer.invoke(IpcChannels.GET_MOLECULE_DETAILS, sdfPath, referenceSdfPath),

  getLastSeenVersion: () =>
    ipcRenderer.invoke(IpcChannels.GET_LAST_SEEN_VERSION),
  setLastSeenVersion: (version: string) =>
    ipcRenderer.invoke(IpcChannels.SET_LAST_SEEN_VERSION, version),

  loadMdTorsionAnalysis: (options: LoadMdTorsionAnalysisOptions) =>
    ipcRenderer.invoke(IpcChannels.LOAD_MD_TORSION_ANALYSIS, options),

  scoreMdClusters: (options: ScoreMdClustersOptions) =>
    ipcRenderer.invoke(IpcChannels.SCORE_MD_CLUSTERS, options),

  mapBindingSite: (options: BindingSiteMapOptions) =>
    ipcRenderer.invoke(IpcChannels.MAP_BINDING_SITE, options),

  // SYNC WITH shared/types/ipc.ts PocketMapOptions
  computePocketMap: (options: {
    method: 'static' | 'solvation' | 'probe';
    pdbPath: string;
    ligandResname: string;
    ligandResnum: number;
    outputDir: string;
    trajectoryPath?: string;
    boxPadding?: number;
    gridSpacing?: number;
  }) => ipcRenderer.invoke(IpcChannels.COMPUTE_POCKET_MAP, options),

  computeSurfaceProps: (pdbPath: string, outputDir: string) =>
    ipcRenderer.invoke(IpcChannels.COMPUTE_SURFACE_PROPS, pdbPath, outputDir),

  // FEP scoring
  runFepScoring: (options: {
    topologyPath: string;
    trajectoryPath: string;
    startNs: number;
    endNs: number;
    numSnapshots: number;
    speedPreset: 'fast' | 'accurate';
    outputDir: string;
    forceFieldPreset: string;
    ligandSdf?: string;
  }) => ipcRenderer.invoke(IpcChannels.RUN_FEP_SCORING, options),

  cancelFepScoring: () => ipcRenderer.invoke(IpcChannels.CANCEL_FEP_SCORING),

  // Molecule alignment
  alignMoleculesMcs: (refSdf: string, mobileSdf: string, outPath: string) =>
    ipcRenderer.invoke('align:mcs', refSdf, mobileSdf, outPath),
  alignDetectScaffolds: (refSdf: string, mobileSdf: string) =>
    ipcRenderer.invoke('align:detect-scaffolds', refSdf, mobileSdf),
  alignByScaffold: (refSdf: string, mobileSdf: string, scaffoldIndex: number, outPath: string) =>
    ipcRenderer.invoke('align:by-scaffold', refSdf, mobileSdf, scaffoldIndex, outPath),

  // Fetch structure from RCSB PDB by ID
  fetchPdb: (pdbId: string, projectDir: string) =>
    ipcRenderer.invoke(IpcChannels.FETCH_PDB, pdbId, projectDir),

  // Image reading
  readImageAsDataUrl: (imagePath: string) =>
    ipcRenderer.invoke(IpcChannels.READ_IMAGE_AS_DATA_URL, imagePath),

  // JSON file reading (for CORDIAL scores etc)
  readJsonFile: (jsonPath: string) =>
    ipcRenderer.invoke('read-json-file', jsonPath),

  // Text file writing (for logs)
  writeTextFile: (filePath: string, content: string) =>
    ipcRenderer.invoke(IpcChannels.WRITE_TEXT_FILE, filePath, content),

  // Get default output directory (user's Desktop)
  getDefaultOutputDir: () =>
    ipcRenderer.invoke('get-default-output-dir'),

  // Project browser
  ensureProject: (projectName: string) =>
    ipcRenderer.invoke(IpcChannels.ENSURE_PROJECT, projectName),
  scanProjects: () =>
    ipcRenderer.invoke(IpcChannels.SCAN_PROJECTS),
  scanRunFiles: (runDir: string) =>
    ipcRenderer.invoke(IpcChannels.SCAN_RUN_FILES, runDir),
  importStructure: (sourcePath: string, projectDir: string) =>
    ipcRenderer.invoke(IpcChannels.IMPORT_STRUCTURE, sourcePath, projectDir),
  prepareForViewing: (rawPdbPath: string, preparedPath: string) =>
    ipcRenderer.invoke(IpcChannels.PREPARE_FOR_VIEWING, rawPdbPath, preparedPath),
  renameProject: (projectDir: string, newName: string) =>
    ipcRenderer.invoke(IpcChannels.RENAME_PROJECT, projectDir, newName),
  deleteProject: (projectDir: string) =>
    ipcRenderer.invoke(IpcChannels.DELETE_PROJECT, projectDir),
  getProjectFileCount: (projectDir: string) =>
    ipcRenderer.invoke(IpcChannels.GET_PROJECT_FILE_COUNT, projectDir),
  scanProjectArtifacts: (projectDir: string) =>
    ipcRenderer.invoke(IpcChannels.SCAN_PROJECT_ARTIFACTS, projectDir),
  selectEmberJobFolder: () =>
    ipcRenderer.invoke(IpcChannels.SELECT_EMBER_JOB_FOLDER),

  // Ligand preparation for viewing (sanitize + add hydrogens + bond orders)
  prepareLigandForViewing: (inputSdf: string, outputSdf: string) =>
    ipcRenderer.invoke('prepare-ligand-for-viewing', inputSdf, outputSdf),

  // Drag-and-drop file path resolution
  getPathForFile: (file: File) => webUtils.getPathForFile(file),

  // Project portability
  openProjectFolder: (projectDir: string) => ipcRenderer.invoke(IpcChannels.OPEN_PROJECT_FOLDER, projectDir),
  importExternalProject: () => ipcRenderer.invoke(IpcChannels.IMPORT_EXTERNAL_PROJECT),
  getHomeDir: () => ipcRenderer.invoke(IpcChannels.GET_HOME_DIR),
  setHomeDir: () => ipcRenderer.invoke(IpcChannels.SET_HOME_DIR),

  // App version from package.json (via Electron)
  getAppVersion: () => ipcRenderer.invoke('get-app-version'),
};

contextBridge.exposeInMainWorld('electronAPI', electronAPI);

// Expose test flag so renderer can conditionally expose NGL stage
if (process.env.NODE_ENV === 'test') {
  contextBridge.exposeInMainWorld('__EMBER_TEST__', true);
}
