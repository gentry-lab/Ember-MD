// Copyright (c) 2026 Ember Contributors. MIT License.
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
  DockLigandPreoptResult,
  OutputData,
  GenerationStats,
} from './ipc';
import type {
  DockConfig,
  DockResult,
  DetectedLigand,
  DockMolecule,
  SingleMoleculeResult,
  PreparedComplexManifest,
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
  XrayAnalysisResult,
  XrayDirectoryScanResult,
  MdReportOptions,
  MdReportResult,
  MdTorsionAnalysis,
  LoadMdTorsionAnalysisOptions,
  ScoreMdClustersOptions,
  ScoreMdClustersResult,
  ScanClusterDirectoryResult,
  LoadedClusterPdb,
  ProjectInfo,
  RunFilesResult,
  BindingSiteMapOptions,
  BindingSiteMapResult,
  PocketMapOptions,
  SurfacePropsResult,
  FepScoringOptions,
  FepScoringResult,
  ProjectJob,
} from './ipc';

export interface ElectronAPI {
  // File operations
  selectPdbFile: (defaultPath?: string) => Promise<string | null>;
  selectPdbFilesMulti: () => Promise<string[]>;
  selectStructureFilesMulti: () => Promise<string[]>;
  selectOutputFolder: () => Promise<string | null>;
  fileExists: (path: string) => Promise<boolean>;
  getFileInfo: (path: string) => Promise<FileInfo>;
  createDirectory: (dirPath: string) => Promise<Result<void, AppError>>;
  listSdfFiles: (dirPath: string) => Promise<string[]>;
  listPdbInDirectory: (dirPath: string) => Promise<string[]>;
  scanXrayDirectory: (dirPath: string) => Promise<Result<XrayDirectoryScanResult, AppError>>;
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
  detectPdbLigands: (pdbPath: string) => Promise<Result<{
    ligands: DetectedLigand[];
    structureInfo?: { totalAtoms: number; hydrogenCount: number; isPrepared: boolean };
  }, AppError>>;
  extractLigand: (pdbPath: string, ligandId: string, outputPath: string) => Promise<Result<string, AppError>>;
  prepareReceptor: (
    pdbPath: string,
    ligandId: string,
    outputPath: string,
    waterDistance?: number,
    protonationPh?: number
  ) => Promise<Result<string, AppError>>;
  prepareDockingComplex: (
    receptorPdb: string,
    xrayLigandSdf: string,
    outputDir: string,
    chargeMethod?: 'gasteiger' | 'am1bcc',
    phMin?: number,
    phMax?: number,
    protonateReference?: boolean
  ) => Promise<Result<{
    manifestPath: string;
    preparedReceptorPdb: string;
    preparedReferenceLigandSdf: string;
    manifest: PreparedComplexManifest;
  }, AppError>>;
  exportDockCsv: (outputDir: string, csvOutput: string, bestOnly: boolean) => Promise<Result<string, AppError>>;
  exportComplexPdb: (receptorPdb: string, ligandSdf: string, poseIndex: number, outputPath: string) => Promise<Result<string, AppError>>;

  // Multi-input ligand source operations
  selectMoleculeFilesMulti: () => Promise<string[]>;
  importMoleculeFiles: (filePaths: string[], outputDir: string) => Promise<Result<DockMolecule[], AppError>>;
  selectFolder: () => Promise<string | null>;
  scanSdfDirectory: (dirPath: string, outputDir: string) => Promise<Result<DockMolecule[], AppError>>;
  parseSmilesCsv: (csvPath: string, outputDir: string) => Promise<Result<DockMolecule[], AppError>>;
  convertSmilesList: (smilesList: string[], outputDir: string) => Promise<Result<DockMolecule[], AppError>>;
  convertSingleMolecule: (input: string, outputDir: string, inputType: 'smiles' | 'mol_file') => Promise<Result<SingleMoleculeResult, AppError>>;
  extractXrayLigand: (pdbPath: string, ligandId: string, outputDir: string, smiles?: string) => Promise<Result<SingleMoleculeResult & { needsSmiles?: boolean; ligandPdb?: string }, AppError>>;
  enumerateProtonation: (
    ligandSdfPaths: string[],
    outputDir: string,
    phMin: number,
    phMax: number
  ) => Promise<Result<{ protonatedPaths: string[]; parentMapping: Record<string, string> }, AppError>>;
  enumerateStereoisomers: (
    ligandSdfPaths: string[],
    outputDir: string,
    maxStereoisomers: number
  ) => Promise<Result<{ stereoisomerPaths: string[]; parentMapping: Record<string, string> }, AppError>>;
  generateConformers: (
    ligandSdfPaths: string[],
    outputDir: string,
    maxConformers: number,
    rmsdCutoff: number,
    energyWindow: number,
    method?: string,
    mcmmOptions?: { steps: number; temperature: number; sampleAmides: boolean }
  ) => Promise<Result<{ conformerPaths: string[]; parentMapping: Record<string, string>; conformerEnergies: Record<string, number> }, AppError>>;
  preOptimizeDockLigands: (
    ligandSdfPaths: string[],
    outputDir: string
  ) => Promise<Result<DockLigandPreoptResult, AppError>>;
  scoreDockingXtbEnergy: (dockOutputDir: string) => Promise<Result<{ count: number }, AppError>>;

  // Conformer generation (standalone)
  runConformGeneration: (
    ligandSdfPath: string,
    outputDir: string,
    maxConformers: number,
    rmsdCutoff: number,
    energyWindow: number,
    method: string,
    mcmmOptions?: { steps: number; temperature: number; sampleAmides: boolean; xtbRerank?: boolean }
  ) => Promise<Result<{ conformerPaths: string[]; parentMapping: Record<string, string>; conformerEnergies: Record<string, number> }, AppError>>;
  onConformOutput: (callback: (data: OutputData) => void) => () => void;

  // Post-dock pocket refinement
  refinePoses: (
    receptorPdb: string,
    posesDir: string,
    outputDir: string,
    maxIterations: number,
    chargeMethod?: string
  ) => Promise<Result<{ refinedCount: number; outputDir: string }, AppError>>;

  // Complex scoring (viewer)
  scoreComplex: (
    pdbPath: string,
    ligandSdfPath?: string
  ) => Promise<Result<{
    vinaRescore?: number;
    xtbStrainKcal?: number;
    cordialExpectedPkd?: number;
    cordialPHighAffinity?: number;
    cordialPVeryHighAffinity?: number;
  }, AppError>>;

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
    ligandOnly?: boolean,
    apo?: boolean
  ) => Promise<Result<string, AppError>>;
  cancelMdBenchmark: () => Promise<void>;
  cancelMdSimulation: () => Promise<void>;
  pauseMdSimulation: () => Promise<void>;
  resumeMdSimulation: () => Promise<void>;

  // MD event listener
  onMdOutput: (callback: (data: OutputData) => void) => () => void;
  onXrayOutput: (callback: (data: OutputData) => void) => () => void;

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
  getTrajectoryCoords: (
    topologyPath: string,
    trajectoryPath: string,
    frameIndex: number
  ) => Promise<Result<{ coordsBase64: string; atomCount: number }, AppError>>;
  clusterTrajectory: (options: ClusteringOptions) => Promise<Result<ClusteringResult, AppError>>;
  scanClusterDirectory: (directoryPath: string) => Promise<Result<ScanClusterDirectoryResult, AppError>>;
  loadAlignedClusters: (directoryPath: string, clusterIds: number[]) => Promise<Result<{ clusters: LoadedClusterPdb[] }, AppError>>;
  exportTrajectoryFrame: (options: ExportFrameOptions) => Promise<Result<{ pdbPath: string }, AppError>>;
  analyzeTrajectory: (options: AnalysisOptions) => Promise<Result<AnalysisResult, AppError>>;
  generateMdReport: (options: MdReportOptions) => Promise<Result<MdReportResult, AppError>>;
  runXrayAnalysis: (inputDir: string, outputDir: string) => Promise<Result<XrayAnalysisResult, AppError>>;
  loadMdTorsionAnalysis: (options: LoadMdTorsionAnalysisOptions) => Promise<Result<MdTorsionAnalysis | null, AppError>>;
  scoreMdClusters: (options: ScoreMdClustersOptions) => Promise<Result<ScoreMdClustersResult, AppError>>;
  mapBindingSite: (options: BindingSiteMapOptions) => Promise<Result<BindingSiteMapResult, AppError>>;
  computePocketMap: (options: PocketMapOptions) => Promise<Result<BindingSiteMapResult, AppError>>;
  computeSurfaceProps: (pdbPath: string, outputDir: string) => Promise<Result<SurfacePropsResult, AppError>>;

  // FEP scoring
  runFepScoring: (options: FepScoringOptions) => Promise<Result<FepScoringResult, AppError>>;
  cancelFepScoring: () => Promise<void>;

  // Molecule alignment
  alignMoleculesMcs: (refSdf: string, mobileSdf: string, outPath: string) => Promise<Result<{ output: string }, AppError>>;
  alignDetectScaffolds: (refSdf: string, mobileSdf: string) => Promise<Result<{ scaffolds: Array<{ label: string; refAtomIndices: number[]; mobileAtomIndices: number[] }> }, AppError>>;
  alignByScaffold: (refSdf: string, mobileSdf: string, scaffoldIndex: number, outPath: string) => Promise<Result<{ output: string; scaffoldIndex: number }, AppError>>;

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
  fetchPdb: (pdbId: string, projectDir: string) => Promise<Result<string, AppError>>;
  renameProject: (oldName: string, newName: string) => Promise<Result<void, AppError>>;
  deleteProject: (projectName: string) => Promise<Result<void, AppError>>;
  getProjectFileCount: (projectName: string) => Promise<{ fileCount: number; totalSizeMb: number }>;
  prepareForViewing: (rawPdbPath: string, preparedPath: string) => Promise<Result<string, AppError>>;
  scanProjectArtifacts: (projectName: string) => Promise<ProjectJob[]>;
  selectEmberJobFolder: () => Promise<ProjectJob | null>;
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}
