/**
 * Type definitions for window.electronAPI
 */

import type { Result } from './result';
import type { AppError } from './errors';
import type {
  PrepPdbOptions,
  PrepPdbResult,
  SurfaceResult,
  GenerationOptions,
  GenerationResult,
  FileInfo,
  OutputData,
  GenerationStats,
} from './ipc';
import type {
  DockConfig,
  DockResult,
  DetectedLigand,
  DockMolecule,
  SingleMoleculeResult,
} from './dock';
import type {
  MDConfig,
  MDForceFieldPreset,
  MDBenchmarkResult,
  MDDockOutput,
} from './md';
import type {
  TrajectoryInfoResult,
  ClusteringOptions,
  ClusteringResult,
  ExportFrameOptions,
  AnalysisOptions,
  AnalysisResult,
  MdReportOptions,
  MdReportResult,
  ScanClusterDirectoryResult,
  LoadedClusterPdb,
  ProjectInfo,
  RunFilesResult,
  BindingSiteMapOptions,
  BindingSiteMapResult,
  SurfacePropsResult,
  FepScoringOptions,
  FepScoringResult,
  ProjectArtifact,
} from './ipc';

export interface ElectronAPI {
  // File operations
  selectPdbFile: (defaultPath?: string) => Promise<string | null>;
  selectPdbFilesMulti: () => Promise<string[]>;
  selectOutputFolder: () => Promise<string | null>;
  fileExists: (path: string) => Promise<boolean>;
  getFileInfo: (path: string) => Promise<FileInfo>;
  createDirectory: (dirPath: string) => Promise<Result<void, AppError>>;
  listSdfFiles: (dirPath: string) => Promise<string[]>;
  listPdbInDirectory: (dirPath: string) => Promise<string[]>;
  openFolder: (folderPath: string) => Promise<void>;

  // Preparation steps
  prepPdb: (
    pdbPath: string,
    outputDir: string,
    options?: PrepPdbOptions
  ) => Promise<Result<PrepPdbResult, AppError>>;

  generateSurface: (
    pocketPdb: string,
    ligandPdb: string,
    outputPly: string
  ) => Promise<Result<SurfaceResult, AppError>>;

  // Generation
  runGeneration: (
    options: GenerationOptions
  ) => Promise<Result<GenerationResult, AppError>>;

  // Results CSV
  generateResultsCsv: (
    sdfDir: string,
    outputCsv: string
  ) => Promise<Result<string, AppError>>;

  // Thumbnail
  generateThumbnail: (sdfPath: string) => Promise<string | null>;

  // Device info
  getAvailableDevices: () => Promise<string[]>;
  getCpuCount: () => Promise<number>;

  // Stats
  getStats: () => Promise<GenerationStats>;
  updateStats: (moleculeCount: number) => Promise<GenerationStats>;

  // Job management
  checkJobExists: (outputFolder: string, jobName: string) => Promise<boolean>;

  // Event listeners (return cleanup functions)
  onPrepOutput: (callback: (data: OutputData) => void) => () => void;
  onSurfaceOutput: (callback: (data: OutputData) => void) => () => void;
  onGenerationOutput: (callback: (data: OutputData) => void) => () => void;

  // Docking operations
  selectCsvFile: () => Promise<string | null>;
  selectSdfFile: () => Promise<string | null>;
  savePdbFile: (content: string, defaultName?: string) => Promise<string | null>;
  runVinaDocking: (
    receptorPdb: string,
    referenceLigand: string,
    ligandSdfPaths: string[],
    outputDir: string,
    config: DockConfig
  ) => Promise<Result<string, AppError>>;
  cancelVinaDocking: () => Promise<void>;
  parseDockResults: (outputDir: string) => Promise<Result<DockResult[], AppError>>;
  listSdfInDirectory: (dirPath: string) => Promise<string[]>;
  detectPdbLigands: (pdbPath: string) => Promise<Result<DetectedLigand[], AppError>>;
  extractLigand: (pdbPath: string, ligandId: string, outputPath: string) => Promise<Result<string, AppError>>;
  prepareReceptor: (pdbPath: string, ligandId: string, outputPath: string, waterDistance?: number) => Promise<Result<string, AppError>>;
  exportDockCsv: (outputDir: string, csvOutput: string, bestOnly: boolean) => Promise<Result<string, AppError>>;
  exportComplexPdb: (receptorPdb: string, ligandSdf: string, poseIndex: number, outputPath: string) => Promise<Result<string, AppError>>;

  // Multi-input ligand source operations
  selectFolder: () => Promise<string | null>;
  scanSdfDirectory: (dirPath: string, outputDir: string) => Promise<Result<DockMolecule[], AppError>>;
  parseSmilesCsv: (csvPath: string, outputDir: string) => Promise<Result<DockMolecule[], AppError>>;
  convertSingleMolecule: (input: string, outputDir: string, inputType: 'smiles' | 'mol_file') => Promise<Result<SingleMoleculeResult, AppError>>;
  extractXrayLigand: (pdbPath: string, ligandId: string, outputDir: string, smiles?: string) => Promise<Result<SingleMoleculeResult & { needsSmiles?: boolean; ligandPdb?: string }, AppError>>;
  enumerateProtonation: (
    ligandSdfPaths: string[],
    outputDir: string,
    phMin: number,
    phMax: number
  ) => Promise<Result<{ protonatedPaths: string[]; parentMapping: Record<string, string> }, AppError>>;
  generateConformers: (
    ligandSdfPaths: string[],
    outputDir: string,
    maxConformers: number,
    rmsdCutoff: number,
    energyWindow: number
  ) => Promise<Result<{ conformerPaths: string[]; parentMapping: Record<string, string> }, AppError>>;

  // CORDIAL rescoring
  checkCordialInstalled: () => Promise<boolean>;
  runCordialScoring: (
    dockOutputDir: string,
    batchSize: number
  ) => Promise<Result<{ scoresFile: string; count: number }, AppError>>;

  // Dock event listener
  onDockOutput: (callback: (data: OutputData) => void) => () => void;

  // MD simulation operations
  loadDockOutputForMd: (dirPath: string) => Promise<Result<MDDockOutput, AppError>>;
  runMdBenchmark: (
    receptorPdb: string | null,
    ligandSdf: string,
    outputDir: string,
    forceFieldPreset?: MDForceFieldPreset,
    ligandOnly?: boolean
  ) => Promise<Result<MDBenchmarkResult, AppError>>;
  runMdSimulation: (
    receptorPdb: string | null,
    ligandSdf: string,
    outputDir: string,
    config: MDConfig,
    ligandOnly?: boolean
  ) => Promise<Result<string, AppError>>;
  cancelMdBenchmark: () => Promise<void>;
  cancelMdSimulation: () => Promise<void>;
  pauseMdSimulation: () => Promise<void>;
  resumeMdSimulation: () => Promise<void>;

  // MD event listener
  onMdOutput: (callback: (data: OutputData) => void) => () => void;

  // Trajectory viewer operations
  selectDcdFile: () => Promise<string | null>;
  getTrajectoryInfo: (
    topologyPath: string,
    trajectoryPath: string
  ) => Promise<Result<TrajectoryInfoResult, AppError>>;
  getTrajectoryFrame: (
    topologyPath: string,
    trajectoryPath: string,
    frameIndex: number
  ) => Promise<Result<{ pdbString: string }, AppError>>;
  clusterTrajectory: (options: ClusteringOptions) => Promise<Result<ClusteringResult, AppError>>;
  scanClusterDirectory: (directoryPath: string) => Promise<Result<ScanClusterDirectoryResult, AppError>>;
  loadAlignedClusters: (directoryPath: string, clusterIds: number[]) => Promise<Result<{ clusters: LoadedClusterPdb[] }, AppError>>;
  exportTrajectoryFrame: (options: ExportFrameOptions) => Promise<Result<{ pdbPath: string }, AppError>>;
  analyzeTrajectory: (options: AnalysisOptions) => Promise<Result<AnalysisResult, AppError>>;
  generateMdReport: (options: MdReportOptions) => Promise<Result<MdReportResult, AppError>>;
  mapBindingSite: (options: BindingSiteMapOptions) => Promise<Result<BindingSiteMapResult, AppError>>;
  computeSurfaceProps: (pdbPath: string, outputDir: string) => Promise<Result<SurfacePropsResult, AppError>>;

  // FEP scoring
  runFepScoring: (options: FepScoringOptions) => Promise<Result<FepScoringResult, AppError>>;
  cancelFepScoring: () => Promise<void>;

  // Image reading
  readImageAsDataUrl: (imagePath: string) => Promise<string | null>;

  // JSON file reading
  readJsonFile: (jsonPath: string) => Promise<unknown | null>;

  // Text file writing (for logs)
  writeTextFile: (filePath: string, content: string) => Promise<Result<string, AppError>>;

  // Get default output directory
  getDefaultOutputDir: () => Promise<string>;

  // Project browser
  ensureProject: (projectName: string) => Promise<Result<string, AppError>>;
  scanProjects: () => Promise<ProjectInfo[]>;
  scanRunFiles: (runDir: string) => Promise<RunFilesResult>;
  importStructure: (sourcePath: string, projectDir: string) => Promise<Result<string, AppError>>;
  renameProject: (oldName: string, newName: string) => Promise<Result<void, AppError>>;
  deleteProject: (projectName: string) => Promise<Result<void, AppError>>;
  getProjectFileCount: (projectName: string) => Promise<{ fileCount: number; totalSizeMb: number }>;
  prepareForViewing: (rawPdbPath: string, preparedPath: string) => Promise<Result<string, AppError>>;
  scanProjectArtifacts: (projectName: string) => Promise<ProjectArtifact[]>;
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}
