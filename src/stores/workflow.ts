// Copyright (c) 2026 Ember Contributors. MIT License.
import { createSignal, createRoot, createEffect, on } from 'solid-js';
import { createProjectTableSaver } from '../utils/projectTablePersistence';
import { generateJobName } from '../utils/jobName';
import {
  MDConfig,
  MDStage,
  MDSystemInfo,
  MDBenchmarkResult,
  MDResult,
  MDLoadedLigand,
  DEFAULT_MD_CONFIG,
} from '../../shared/types/md';
import {
  DockConfig,
  DockResult,
  DockMolecule,
  CordialConfig,
  ProtonationConfig,
  StereoisomerConfig,
  ConformerConfig,
  DEFAULT_DOCK_CONFIG,
  DEFAULT_CORDIAL_CONFIG,
  DEFAULT_PROTONATION_CONFIG,
  DEFAULT_STEREOISOMER_CONFIG,
  DEFAULT_CONFORMER_CONFIG,
  RefinementConfig,
  DEFAULT_REFINEMENT_CONFIG,
  XtbConfig,
  DEFAULT_XTB_CONFIG,
  WaterRetentionConfig,
  DEFAULT_WATER_RETENTION_CONFIG,
  LigandSource,
  DetectedLigand,
  DetectedLigand as DockDetectedLigand,
} from '../../shared/types/dock';

export type WorkflowMode = 'dock' | 'md' | 'score' | 'xray' | 'viewer' | 'map' | 'conform';
export type DockStep = 'dock-load' | 'dock-configure' | 'dock-progress' | 'dock-results';
export type MDStep = 'md-home' | 'md-load' | 'md-configure' | 'md-progress' | 'md-results';
export type ConformStep = 'conform-load' | 'conform-configure' | 'conform-progress' | 'conform-results';
export type ScoreStep = 'score-load' | 'score-progress' | 'score-results';
export type XrayStep = 'xray-load' | 'xray-progress' | 'xray-results';
export type MapStep = 'map-load' | 'map-configure' | 'map-progress' | 'map-results';

// Map mode types — mirrors PocketMapMethod in shared/types/ipc.ts
import type {
  ClusteringResult as IpcClusteringResult,
  PocketMapMethod,
  ScoredClusterResult,
  MdTorsionAnalysis,
  XrayAnalysisResult,
  XrayDirectoryScanResult,
  BatchScoreEntryResult,
} from '../../shared/types/ipc';
export type MapMethod = PocketMapMethod;

// Viewer state types
export type ProteinRepresentation = 'cartoon' | 'ribbon' | 'spacefill';
export type LigandRepresentation = 'ball+stick' | 'stick' | 'spacefill';
export type SurfaceColorScheme = 'uniform-grey' | 'hydrophobic' | 'electrostatic';
export type PlaybackSpeed = 0.25 | 0.5 | 1 | 2 | 4;
export type CenterTarget = 'ligand' | 'protein' | 'none';

export interface TrajectoryInfo {
  frameCount: number;
  timestepPs: number;
  totalTimeNs: number;
}

export interface ClusteringConfig {
  numClusters: number;
  method: 'kmeans' | 'dbscan' | 'hierarchical';
  rmsdSelection: 'ligand' | 'backbone' | 'all';
  stripWaters: boolean;
  saveCentroids: boolean;
}

export interface ClusterResult {
  clusterId: number;
  frameCount: number;
  population: number;
  centroidFrame: number;
  centroidPdbPath?: string;
}

export interface ClusteringResults {
  clusters: ClusterResult[];
  frameAssignments: number[];
  requestedClusters?: number;
  actualClusters?: number;
}

export interface LoadedCluster {
  clusterId: number;
  pdbPath: string;
  population: number;
  color: string;
  visible: boolean;
}

export const CLUSTER_COLORS = [
  '#2563eb', '#dc2626', '#16a34a', '#9333ea', '#ea580c',
  '#0891b2', '#c026d3', '#65a30d', '#0d9488', '#e11d48',
];

// Re-export DetectedLigand from dock.ts (single source of truth)
export type { DetectedLigand };

export interface ViewerQueueItem {
  pdbPath: string;
  ligandPath?: string;
  label: string;
  type?: 'protein' | 'ligand' | 'conformer';
}

export type ViewerLayerType = 'protein' | 'ligand' | 'trajectory';

export interface ViewerLayer {
  id: string;
  type: ViewerLayerType;
  label: string;
  filePath: string;
  visible: boolean;
  groupId?: string;
  poseIndex?: number;
  affinity?: number;
}

export interface ViewerLayerGroup {
  id: string;
  type: 'docking' | 'simulation';
  label: string;
  expanded: boolean;
  visible: boolean;
}

export type ViewerProjectJobType = 'import' | 'docking' | 'conformer' | 'simulation';
export type ViewerProjectRowKind =
  | 'apo'
  | 'holo'
  | 'ligand'
  | 'prepared-ligand'
  | 'pose'
  | 'input'
  | 'conformer'
  | 'initial-complex'
  | 'cluster';
export type ViewerProjectSortDirection = 'asc' | 'desc';

export interface ViewerProjectColumn {
  key: string;
  label: string;
  kind?: 'text' | 'number' | 'percent';
  priority: number;
  minPanelWidth?: number;
}

export interface ViewerProjectRow {
  id: string;
  familyId: string;
  label: string;
  rowKind: ViewerProjectRowKind;
  jobType: ViewerProjectJobType;
  item: ViewerQueueItem;
  loadKind: 'structure' | 'standalone-ligand' | 'queue';
  queueIndex?: number;
  metrics: Record<string, string | number | null | undefined>;
  trajectoryPath?: string | null;
  pocketLigandPath?: string | null;
  pocketSourcePdbPath?: string | null;
}

export interface ViewerProjectFamily {
  id: string;
  title: string;
  jobType: ViewerProjectJobType;
  collapsed: boolean;
  rowIds: string[];
  columns: ViewerProjectColumn[];
  sortKey?: string | null;
  sortDirection?: ViewerProjectSortDirection;
  trajectoryPath?: string | null;
}

export interface ViewerProjectTableState {
  families: ViewerProjectFamily[];
  rows: ViewerProjectRow[];
  activeRowId: string | null;
  selectedRowIds: string[];
}

export interface BindingSiteMapChannel {
  visible: boolean;
  isolevel: number;
  opacity: number;
}

export interface BindingSiteMapState {
  hydrophobic: BindingSiteMapChannel;
  hbondDonor: BindingSiteMapChannel;
  hbondAcceptor: BindingSiteMapChannel;
  hydrophobicDx: string;
  hbondDonorDx: string;
  hbondAcceptorDx: string;
  hotspots: Array<{ type: string; position: number[]; direction: number[]; score: number }>;
  method?: PocketMapMethod;
}

export interface MapResultState extends BindingSiteMapState {
  method: MapMethod;
  pdbPath: string;
  outputDir: string;
  trajectoryPath: string | null;
}

export interface ViewerState {
  sessionKey: number;
  pdbPath: string | null;
  ligandPath: string | null;
  pdbQueue: ViewerQueueItem[];
  pdbQueueIndex: number;
  detectedLigands: DetectedLigand[];
  selectedLigandId: string | null;
  proteinRep: ProteinRepresentation;
  proteinSurface: boolean;
  proteinSurfaceOpacity: number;
  surfaceColorScheme: SurfaceColorScheme;
  proteinCarbonColor: string;
  showPocketResidues: boolean;
  showPocketLabels: boolean;
  hideWaterIons: boolean;
  ligandVisible: boolean;
  ligandRep: LigandRepresentation;
  ligandSurface: boolean;
  ligandSurfaceOpacity: number;
  ligandCarbonColor: string;
  ligandPolarHOnly: boolean;
  showInteractions: boolean;
  trajectoryPath: string | null;
  trajectoryInfo: TrajectoryInfo | null;
  currentFrame: number;
  isPlaying: boolean;
  playbackSpeed: PlaybackSpeed;
  smoothing: number;
  loopPlayback: boolean;
  centerTarget: CenterTarget;
  clusteringConfig: ClusteringConfig;
  clusteringResults: ClusteringResults | null;
  isClustering: boolean;
  loadedClusters: LoadedCluster[];
  isLoadingClusters: boolean;
  bindingSiteMap: BindingSiteMapState | null;
  isComputingBindingSiteMap: boolean;
  layers: ViewerLayer[];
  layerGroups: ViewerLayerGroup[];
  selectedLayerId: string | null;
  projectTable: ViewerProjectTableState | null;
}

interface OpenViewerSessionOptions {
  pdbPath?: string | null;
  ligandPath?: string | null;
  trajectoryPath?: string | null;
  trajectoryInfo?: TrajectoryInfo | null;
  pdbQueue?: ViewerQueueItem[];
  pdbQueueIndex?: number;
  bindingSiteMap?: BindingSiteMapState | null;
  projectTable?: ViewerProjectTableState | null;
}

export interface MapState {
  step: MapStep;
  method: MapMethod;
  isComputing: boolean;
  progress: string;
  progressPct: number;
  error: string | null;
  showMdConfirm: boolean;
  estimatedTimeMin: number | null;
  pdbPath: string | null;
  detectedLigands: DetectedLigand[];
  selectedLigandId: string | null;
  isDetecting: boolean;
  result: MapResultState | null;
}

export interface ConformState {
  ligandSdfPath: string | null;
  ligandName: string | null;
  outputName: string;
  protonationConfig: ProtonationConfig;
  config: ConformerConfig;
  outputDir: string | null;
  conformerPaths: string[];
  conformerEnergies: Record<string, number>;
  isRunning: boolean;
}

export interface ScoreComplexEntry {
  id: string;
  pdbPath: string;
  name: string;
  detectedLigands: DetectedLigand[];
  selectedLigandId: string | null;
  isPrepared: boolean;
  preparedReceptorPath: string | null;
  extractedLigandSdfPath: string | null;
  vinaScore: number | null;
  cordialExpectedPkd: number | null;
  cordialPHighAffinity: number | null;
  qed: number | null;
  status: 'pending' | 'detecting' | 'preparing' | 'scoring' | 'done' | 'error';
  errorMessage: string | null;
}

export interface ScoreState {
  entries: ScoreComplexEntry[];
  outputDir: string | null;
  isRunning: boolean;
  cordialAvailable: boolean;
}

export interface XrayState {
  inputDir: string | null;
  scanResult: XrayDirectoryScanResult | null;
  outputDir: string | null;
  result: XrayAnalysisResult | null;
  isRunning: boolean;
}

export interface PdbFile {
  path: string;
  name: string;
}

export type MDInputMode = 'holo' | 'ligand_only' | 'apo';

export interface DockState {
  receptorPdbPath: string | null;
  receptorPrepared: string | null;
  referenceLigandId: string | null;
  referenceLigandPath: string | null;
  preparedLigandPath: string | null;
  detectedLigands: DockDetectedLigand[];
  ligandSource: LigandSource;
  ligandSdfPaths: string[];
  ligandMolecules: DockMolecule[];
  config: DockConfig;
  cordialConfig: CordialConfig;
  protonationConfig: ProtonationConfig;
  stereoisomerConfig: StereoisomerConfig;
  conformerConfig: ConformerConfig;
  refinementConfig: RefinementConfig;
  xtbConfig: XtbConfig;
  waterRetentionConfig: WaterRetentionConfig;
  cachedConformerPaths: string[];
  dockingOutputDir: string | null;
  totalLigands: number;
  completedLigands: number;
  results: DockResult[];
  cordialAvailable: boolean;
  cordialScored: boolean;
}

export interface MDState {
  inputMode: MDInputMode;
  dockOutputDir: string | null;
  loadedLigands: MDLoadedLigand[];
  selectedLigandIndex: number | null;
  receptorPdb: string | null;
  ligandSdf: string | null;
  ligandName: string | null;
  pdbPath: string | null;
  thumbnailDataUrl: string | null;
  singleMoleculeInput: string | null;
  singleMoleculeThumbnail: string | null;
  config: MDConfig;
  outputDir: string | null;
  result: MDResult | null;
  currentStage: MDStage | null;
  stageProgress: number;
  systemInfo: MDSystemInfo | null;
  benchmarkResult: MDBenchmarkResult | null;
  isBenchmarking: boolean;
  clusteringResults: ClusteringResults | null;
  clusterScores: ScoredClusterResult[];
  torsionAnalysis: MdTorsionAnalysis | null;
}

export interface WorkflowState {
  mode: WorkflowMode;
  projectReady: boolean;
  projectDir: string | null;
  dockStep: DockStep;
  mdStep: MDStep;
  conformStep: ConformStep;
  scoreStep: ScoreStep;
  xrayStep: XrayStep;
  pdbFile: PdbFile | null;
  customOutputDir: string | null;
  jobName: string;
  isRunning: boolean;
  runningMode: WorkflowMode | null;
  isPaused: boolean;
  currentPhase: 'idle' | 'generation' | 'complete' | 'error';
  logs: string;
  errorMessage: string | null;
  dock: DockState;
  md: MDState;
  viewer: ViewerState;
  map: MapState;
  conform: ConformState;
  score: ScoreState;
  xray: XrayState;
}

const defaultDockState: DockState = {
  receptorPdbPath: null,
  receptorPrepared: null,
  referenceLigandId: null,
  referenceLigandPath: null,
  preparedLigandPath: null,
  detectedLigands: [],
  ligandSource: 'structure_files',
  ligandSdfPaths: [],
  ligandMolecules: [],
  config: { ...DEFAULT_DOCK_CONFIG },
  cordialConfig: { ...DEFAULT_CORDIAL_CONFIG },
  protonationConfig: { ...DEFAULT_PROTONATION_CONFIG },
  stereoisomerConfig: { ...DEFAULT_STEREOISOMER_CONFIG },
  conformerConfig: { ...DEFAULT_CONFORMER_CONFIG },
  refinementConfig: { ...DEFAULT_REFINEMENT_CONFIG },
  xtbConfig: { ...DEFAULT_XTB_CONFIG },
  waterRetentionConfig: { ...DEFAULT_WATER_RETENTION_CONFIG },
  cachedConformerPaths: [],
  dockingOutputDir: null,
  totalLigands: 0,
  completedLigands: 0,
  results: [],
  cordialAvailable: false,
  cordialScored: false,
};

const defaultMDState: MDState = {
  inputMode: 'holo',
  dockOutputDir: null,
  loadedLigands: [],
  selectedLigandIndex: null,
  receptorPdb: null,
  ligandSdf: null,
  ligandName: null,
  pdbPath: null,
  thumbnailDataUrl: null,
  singleMoleculeInput: null,
  singleMoleculeThumbnail: null,
  config: { ...DEFAULT_MD_CONFIG },
  outputDir: null,
  result: null,
  currentStage: null,
  stageProgress: 0,
  systemInfo: null,
  benchmarkResult: null,
  isBenchmarking: false,
  clusteringResults: null,
  clusterScores: [],
  torsionAnalysis: null,
};

const defaultMapState: MapState = {
  step: 'map-load',
  method: 'solvation',
  isComputing: false,
  progress: '',
  progressPct: 0,
  error: null,
  showMdConfirm: false,
  estimatedTimeMin: null,
  pdbPath: null,
  detectedLigands: [],
  selectedLigandId: null,
  isDetecting: false,
  result: null,
};

const defaultConformState: ConformState = {
  ligandSdfPath: null,
  ligandName: null,
  outputName: '',
  protonationConfig: { ...DEFAULT_PROTONATION_CONFIG },
  config: { ...DEFAULT_CONFORMER_CONFIG },
  outputDir: null,
  conformerPaths: [],
  conformerEnergies: {},
  isRunning: false,
};

const defaultScoreState: ScoreState = {
  entries: [],
  outputDir: null,
  isRunning: false,
  cordialAvailable: false,
};

const defaultXrayState: XrayState = {
  inputDir: null,
  scanResult: null,
  outputDir: null,
  result: null,
  isRunning: false,
};

const defaultClusteringConfig: ClusteringConfig = {
  numClusters: 5,
  method: 'kmeans',
  rmsdSelection: 'ligand',
  stripWaters: true,
  saveCentroids: true,
};

const defaultViewerState: ViewerState = {
  sessionKey: 0,
  pdbPath: null,
  ligandPath: null,
  pdbQueue: [],
  pdbQueueIndex: 0,
  detectedLigands: [],
  selectedLigandId: null,
  proteinRep: 'cartoon',
  proteinSurface: false,
  proteinSurfaceOpacity: 0.9,
  surfaceColorScheme: 'uniform-grey',
  proteinCarbonColor: '#909090',
  showPocketResidues: true,
  showPocketLabels: false,
  hideWaterIons: true,
  ligandVisible: true,
  ligandRep: 'stick',
  ligandSurface: false,
  ligandSurfaceOpacity: 0.5,
  ligandCarbonColor: '#00ff00',
  ligandPolarHOnly: true,
  showInteractions: true,
  trajectoryPath: null,
  trajectoryInfo: null,
  currentFrame: 0,
  isPlaying: false,
  playbackSpeed: 1,
  smoothing: 1,
  loopPlayback: true,
  centerTarget: 'ligand',
  clusteringConfig: { ...defaultClusteringConfig },
  clusteringResults: null,
  isClustering: false,
  loadedClusters: [],
  isLoadingClusters: false,
  bindingSiteMap: null,
  isComputingBindingSiteMap: false,
  layers: [],
  layerGroups: [],
  selectedLayerId: null,
  projectTable: null,
};

function createWorkflowStore() {
  const initialJobName = generateJobName();

  const [state, setState] = createSignal<WorkflowState>({
    mode: 'md',
    projectReady: false,
    projectDir: null,
    dockStep: 'dock-load',
    mdStep: 'md-load',
    conformStep: 'conform-load',
    scoreStep: 'score-load',
    xrayStep: 'xray-load',
    pdbFile: null,
    customOutputDir: null,
    jobName: initialJobName,
    isRunning: false,
    runningMode: null,
    isPaused: false,
    currentPhase: 'idle',
    logs: '',
    errorMessage: null,
    dock: { ...defaultDockState },
    md: { ...defaultMDState },
    viewer: { ...defaultViewerState },
    map: { ...defaultMapState },
    conform: { ...defaultConformState },
    score: { ...defaultScoreState },
    xray: { ...defaultXrayState },
  });

  // Mode selection
  const setMode = (mode: WorkflowMode) => {
    console.log(`[Store] setMode: ${state().mode} → ${mode}`);
    setState((s) => ({ ...s, mode }));
  };
  const setProjectReady = (projectReady: boolean) => setState((s) => ({ ...s, projectReady }));
  const setProjectDir = (projectDir: string | null) => setState((s) => ({ ...s, projectDir }));
  const setDockStep = (dockStep: DockStep) => {
    console.log(`[Store] setDockStep: ${state().dockStep} → ${dockStep}`);
    setState((s) => ({ ...s, dockStep }));
  };
  const setMdStep = (mdStep: MDStep) => {
    console.log(`[Store] setMdStep: ${state().mdStep} → ${mdStep}`);
    setState((s) => ({ ...s, mdStep }));
  };

  const setConformStep = (conformStep: ConformStep) => {
    console.log(`[Store] setConformStep: ${state().conformStep} → ${conformStep}`);
    setState((s) => ({ ...s, conformStep }));
  };

  const setScoreStep = (scoreStep: ScoreStep) => {
    console.log(`[Store] setScoreStep: ${state().scoreStep} → ${scoreStep}`);
    setState((s) => ({ ...s, scoreStep }));
  };

  const setXrayStep = (xrayStep: XrayStep) => {
    console.log(`[Store] setXrayStep: ${state().xrayStep} → ${xrayStep}`);
    setState((s) => ({ ...s, xrayStep }));
  };

  const setPdbFile = (pdbFile: PdbFile | null) =>
    setState((s) => ({ ...s, pdbFile, errorMessage: null }));

  const setCustomOutputDir = (customOutputDir: string | null) =>
    setState((s) => ({ ...s, customOutputDir }));

  const setJobName = (jobName: string) =>
    setState((s) => ({ ...s, jobName }));

  const regenerateJobName = () =>
    setState((s) => ({ ...s, jobName: generateJobName() }));

  const setIsRunning = (isRunning: boolean) =>
    setState((s) => ({ ...s, isRunning, runningMode: isRunning ? s.mode : s.runningMode }));

  const setIsPaused = (isPaused: boolean) =>
    setState((s) => ({ ...s, isPaused }));

  const appendLog = (text: string) =>
    setState((s) => {
      const combined = s.logs + text;
      return { ...s, logs: combined.length > 50000 ? combined.slice(-50000) : combined };
    });

  const clearLogs = () => setState((s) => ({ ...s, logs: '' }));

  const setCurrentPhase = (currentPhase: WorkflowState['currentPhase']) =>
    setState((s) => ({ ...s, currentPhase }));

  const setError = (errorMessage: string | null) =>
    setState((s) => ({ ...s, errorMessage }));

  // Dock state setters
  const setDockReceptorPdbPath = (receptorPdbPath: string | null) =>
    setState((s) => ({ ...s, dock: { ...s.dock, receptorPdbPath } }));

  const setDockReceptorPrepared = (receptorPrepared: string | null) =>
    setState((s) => ({ ...s, dock: { ...s.dock, receptorPrepared } }));

  const setDockReferenceLigandId = (referenceLigandId: string | null) =>
    setState((s) => ({ ...s, dock: { ...s.dock, referenceLigandId } }));

  const setDockReferenceLigandPath = (referenceLigandPath: string | null) =>
    setState((s) => ({ ...s, dock: { ...s.dock, referenceLigandPath } }));

  const setDockPreparedLigandPath = (preparedLigandPath: string | null) =>
    setState((s) => ({ ...s, dock: { ...s.dock, preparedLigandPath } }));

  const setDockDetectedLigands = (detectedLigands: DockDetectedLigand[]) =>
    setState((s) => ({ ...s, dock: { ...s.dock, detectedLigands } }));

  const setDockLigandSource = (ligandSource: LigandSource) =>
    setState((s) => ({ ...s, dock: { ...s.dock, ligandSource } }));

  const setDockLigandSdfPaths = (ligandSdfPaths: string[]) =>
    setState((s) => ({ ...s, dock: { ...s.dock, ligandSdfPaths } }));

  const setDockLigandMolecules = (ligandMolecules: DockMolecule[]) =>
    setState((s) => ({ ...s, dock: { ...s.dock, ligandMolecules } }));

  const setDockConfig = (config: Partial<DockConfig>) =>
    setState((s) => ({ ...s, dock: { ...s.dock, config: { ...s.dock.config, ...config } } }));

  const setDockCordialConfig = (cordialConfig: Partial<CordialConfig>) =>
    setState((s) => ({ ...s, dock: { ...s.dock, cordialConfig: { ...s.dock.cordialConfig, ...cordialConfig } } }));

  const setDockProtonationConfig = (protonationConfig: Partial<ProtonationConfig>) =>
    setState((s) => ({ ...s, dock: { ...s.dock, protonationConfig: { ...s.dock.protonationConfig, ...protonationConfig } } }));

  const setDockStereoisomerConfig = (stereoisomerConfig: Partial<StereoisomerConfig>) =>
    setState((s) => ({ ...s, dock: { ...s.dock, stereoisomerConfig: { ...s.dock.stereoisomerConfig, ...stereoisomerConfig } } }));

  const setDockConformerConfig = (conformerConfig: Partial<ConformerConfig>) =>
    setState((s) => ({ ...s, dock: { ...s.dock, conformerConfig: { ...s.dock.conformerConfig, ...conformerConfig } } }));

  const setDockRefinementConfig = (refinementConfig: Partial<RefinementConfig>) =>
    setState((s) => ({ ...s, dock: { ...s.dock, refinementConfig: { ...s.dock.refinementConfig, ...refinementConfig } } }));

  const setDockXtbConfig = (xtbConfig: Partial<XtbConfig>) =>
    setState((s) => ({ ...s, dock: { ...s.dock, xtbConfig: { ...s.dock.xtbConfig, ...xtbConfig } } }));

  const setDockWaterRetentionConfig = (waterRetentionConfig: Partial<WaterRetentionConfig>) =>
    setState((s) => ({ ...s, dock: { ...s.dock, waterRetentionConfig: { ...s.dock.waterRetentionConfig, ...waterRetentionConfig } } }));

  const setDockCachedConformerPaths = (cachedConformerPaths: string[]) =>
    setState((s) => ({ ...s, dock: { ...s.dock, cachedConformerPaths } }));

  const setDockOutputDir = (dockingOutputDir: string | null) =>
    setState((s) => ({ ...s, dock: { ...s.dock, dockingOutputDir } }));

  const setDockTotalLigands = (totalLigands: number) =>
    setState((s) => ({ ...s, dock: { ...s.dock, totalLigands } }));

  const setDockCompletedLigands = (completedLigands: number) =>
    setState((s) => ({ ...s, dock: { ...s.dock, completedLigands } }));

  const setDockResults = (results: DockResult[]) =>
    setState((s) => ({ ...s, dock: { ...s.dock, results } }));

  const setDockCordialAvailable = (cordialAvailable: boolean) =>
    setState((s) => ({ ...s, dock: { ...s.dock, cordialAvailable } }));

  const setDockCordialScored = (cordialScored: boolean) =>
    setState((s) => ({ ...s, dock: { ...s.dock, cordialScored } }));

  const resetDock = () =>
    setState((s) => ({
      ...s,
      dockStep: 'dock-load' as DockStep,
      currentPhase: 'idle',
      logs: '',
      errorMessage: null,
      isRunning: false,
      runningMode: null,
      dock: { ...defaultDockState },
    }));

  // MD state setters
  const setMdInputMode = (inputMode: MDInputMode) =>
    setState((s) => ({ ...s, md: { ...s.md, inputMode } }));

  const setMdSingleMoleculeInput = (singleMoleculeInput: string | null) =>
    setState((s) => ({ ...s, md: { ...s.md, singleMoleculeInput } }));

  const setMdSingleMoleculeThumbnail = (singleMoleculeThumbnail: string | null) =>
    setState((s) => ({ ...s, md: { ...s.md, singleMoleculeThumbnail } }));

  const setMdDockOutputDir = (dockOutputDir: string | null) =>
    setState((s) => ({ ...s, md: { ...s.md, dockOutputDir } }));

  const setMdLoadedLigands = (loadedLigands: MDLoadedLigand[]) =>
    setState((s) => ({ ...s, md: { ...s.md, loadedLigands } }));

  const setMdSelectedLigandIndex = (selectedLigandIndex: number | null) =>
    setState((s) => ({ ...s, md: { ...s.md, selectedLigandIndex } }));

  const setMdReceptorPdb = (receptorPdb: string | null) =>
    setState((s) => ({ ...s, md: { ...s.md, receptorPdb } }));

  const setMdLigandSdf = (ligandSdf: string | null) =>
    setState((s) => ({ ...s, md: { ...s.md, ligandSdf } }));

  const setMdLigandName = (ligandName: string | null) =>
    setState((s) => ({ ...s, md: { ...s.md, ligandName } }));

  const setMdPdbPath = (pdbPath: string | null) =>
    setState((s) => ({ ...s, md: { ...s.md, pdbPath } }));

  const setMdThumbnailDataUrl = (thumbnailDataUrl: string | null) =>
    setState((s) => ({ ...s, md: { ...s.md, thumbnailDataUrl } }));

  const setMdConfig = (config: Partial<MDConfig>) =>
    setState((s) => ({ ...s, md: { ...s.md, config: { ...s.md.config, ...config } } }));

  const setMdOutputDir = (outputDir: string | null) =>
    setState((s) => ({ ...s, md: { ...s.md, outputDir } }));

  const setMdResult = (result: MDResult | null) =>
    setState((s) => ({ ...s, md: { ...s.md, result } }));

  const setMdCurrentStage = (currentStage: MDStage | null) =>
    setState((s) => ({ ...s, md: { ...s.md, currentStage } }));

  const setMdStageProgress = (stageProgress: number) =>
    setState((s) => ({ ...s, md: { ...s.md, stageProgress } }));

  const setMdSystemInfo = (systemInfo: MDSystemInfo | null) =>
    setState((s) => ({ ...s, md: { ...s.md, systemInfo } }));

  const setMdBenchmarkResult = (benchmarkResult: MDBenchmarkResult | null) =>
    setState((s) => ({ ...s, md: { ...s.md, benchmarkResult } }));

  const setMdIsBenchmarking = (isBenchmarking: boolean) =>
    setState((s) => ({ ...s, md: { ...s.md, isBenchmarking } }));

  const setMdClusteringResults = (clusteringResults: ClusteringResults | null) =>
    setState((s) => ({ ...s, md: { ...s.md, clusteringResults } }));

  const setMdClusterScores = (clusterScores: ScoredClusterResult[]) =>
    setState((s) => ({ ...s, md: { ...s.md, clusterScores } }));

  const setMdTorsionAnalysis = (torsionAnalysis: MdTorsionAnalysis | null) =>
    setState((s) => ({ ...s, md: { ...s.md, torsionAnalysis } }));

  const setMdClusteringResultsFromIpc = (clusteringResults: IpcClusteringResult | null) =>
    setMdClusteringResults(clusteringResults ? {
      clusters: clusteringResults.clusters,
      frameAssignments: clusteringResults.frameAssignments,
      requestedClusters: clusteringResults.requestedClusters,
      actualClusters: clusteringResults.actualClusters,
    } : null);

  // Viewer state setters
  const setViewerPdbPath = (pdbPath: string | null) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, pdbPath } }));

  const setViewerPdbQueue = (pdbQueue: ViewerQueueItem[]) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, pdbQueue, pdbQueueIndex: 0 } }));

  const setViewerPdbQueueIndex = (pdbQueueIndex: number) =>
    setState((s) => {
      const item = s.viewer.pdbQueue[pdbQueueIndex];
      const activeProjectRowId = s.viewer.projectTable?.rows.find((row) => row.queueIndex === pdbQueueIndex)?.id
        ?? s.viewer.projectTable?.activeRowId
        ?? null;
      return {
        ...s,
        viewer: {
          ...s.viewer,
          pdbQueueIndex,
          pdbPath: item?.pdbPath || s.viewer.pdbPath,
          ligandPath: item?.ligandPath ?? s.viewer.ligandPath,
          projectTable: s.viewer.projectTable
            ? { ...s.viewer.projectTable, activeRowId: activeProjectRowId }
            : null,
        },
      };
    });

  const setViewerLigandPath = (ligandPath: string | null) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, ligandPath } }));

  const setViewerDetectedLigands = (detectedLigands: DetectedLigand[]) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, detectedLigands } }));

  const setViewerSelectedLigandId = (selectedLigandId: string | null) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, selectedLigandId } }));

  const setViewerProteinRep = (proteinRep: ProteinRepresentation) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, proteinRep } }));

  const setViewerProteinSurface = (proteinSurface: boolean) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, proteinSurface } }));

  const setViewerProteinSurfaceOpacity = (proteinSurfaceOpacity: number) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, proteinSurfaceOpacity } }));

  const setViewerShowPocketResidues = (showPocketResidues: boolean) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, showPocketResidues } }));

  const setViewerShowPocketLabels = (showPocketLabels: boolean) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, showPocketLabels } }));

  const setViewerHideWaterIons = (hideWaterIons: boolean) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, hideWaterIons } }));

  const setViewerSurfaceColorScheme = (surfaceColorScheme: SurfaceColorScheme) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, surfaceColorScheme } }));

  const setViewerProteinCarbonColor = (proteinCarbonColor: string) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, proteinCarbonColor } }));

  const setViewerLigandCarbonColor = (ligandCarbonColor: string) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, ligandCarbonColor } }));

  const setViewerLigandRep = (ligandRep: LigandRepresentation) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, ligandRep } }));

  const setViewerLigandSurface = (ligandSurface: boolean) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, ligandSurface } }));

  const setViewerLigandSurfaceOpacity = (ligandSurfaceOpacity: number) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, ligandSurfaceOpacity } }));

  const setViewerShowInteractions = (showInteractions: boolean) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, showInteractions } }));

  const setViewerLigandVisible = (ligandVisible: boolean) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, ligandVisible } }));

  const setViewerLigandPolarHOnly = (ligandPolarHOnly: boolean) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, ligandPolarHOnly } }));

  const setViewerTrajectoryPath = (trajectoryPath: string | null) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, trajectoryPath } }));

  const setViewerTrajectoryInfo = (trajectoryInfo: TrajectoryInfo | null) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, trajectoryInfo } }));

  const setViewerCurrentFrame = (currentFrame: number) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, currentFrame } }));

  const setViewerIsPlaying = (isPlaying: boolean) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, isPlaying } }));

  const setViewerPlaybackSpeed = (playbackSpeed: PlaybackSpeed) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, playbackSpeed } }));

  const setViewerSmoothing = (smoothing: number) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, smoothing } }));

  const setViewerLoopPlayback = (loopPlayback: boolean) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, loopPlayback } }));

  const setViewerCenterTarget = (centerTarget: CenterTarget) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, centerTarget } }));

  const setViewerClusteringConfig = (config: Partial<ClusteringConfig>) =>
    setState((s) => ({
      ...s,
      viewer: { ...s.viewer, clusteringConfig: { ...s.viewer.clusteringConfig, ...config } },
    }));

  const setViewerClusteringResults = (clusteringResults: ClusteringResults | null) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, clusteringResults } }));

  const setViewerIsClustering = (isClustering: boolean) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, isClustering } }));

  const setViewerLoadedClusters = (loadedClusters: LoadedCluster[]) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, loadedClusters } }));

  const setViewerIsLoadingClusters = (isLoadingClusters: boolean) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, isLoadingClusters } }));

  const setViewerClusterVisibility = (clusterId: number, visible: boolean) =>
    setState((s) => ({
      ...s,
      viewer: {
        ...s.viewer,
        loadedClusters: s.viewer.loadedClusters.map((c) =>
          c.clusterId === clusterId ? { ...c, visible } : c
        ),
      },
    }));

  const setViewerBindingSiteMap = (bindingSiteMap: BindingSiteMapState | null) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, bindingSiteMap } }));

  const setViewerIsComputingBindingSiteMap = (isComputingBindingSiteMap: boolean) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, isComputingBindingSiteMap } }));

  const setViewerBindingSiteChannel = (
    channel: 'hydrophobic' | 'hbondDonor' | 'hbondAcceptor',
    updates: Partial<BindingSiteMapChannel>
  ) =>
    setState((s) => {
      if (!s.viewer.bindingSiteMap) return s;
      return {
        ...s,
        viewer: {
          ...s.viewer,
          bindingSiteMap: {
            ...s.viewer.bindingSiteMap,
            [channel]: { ...s.viewer.bindingSiteMap[channel], ...updates },
          },
        },
      };
    });

  // Layer state setters
  let layerIdSeq = 0;
  const nextLayerId = (): string => `layer-${layerIdSeq++}`;

  const addViewerLayer = (layer: ViewerLayer) =>
    setState((s) => ({
      ...s,
      viewer: { ...s.viewer, layers: [...s.viewer.layers, layer] },
    }));

  const removeViewerLayer = (id: string) =>
    setState((s) => ({
      ...s,
      viewer: {
        ...s.viewer,
        layers: s.viewer.layers.filter((l) => l.id !== id),
        selectedLayerId: s.viewer.selectedLayerId === id ? null : s.viewer.selectedLayerId,
      },
    }));

  const updateViewerLayer = (id: string, updates: Partial<ViewerLayer>) =>
    setState((s) => ({
      ...s,
      viewer: {
        ...s.viewer,
        layers: s.viewer.layers.map((l) => (l.id === id ? { ...l, ...updates } : l)),
      },
    }));

  const setViewerLayerSelected = (id: string | null) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, selectedLayerId: id } }));

  const setViewerProjectTable = (projectTable: ViewerProjectTableState | null) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, projectTable } }));

  const addViewerProjectFamily = (family: ViewerProjectFamily, rows: ViewerProjectRow[]) =>
    setState((s) => {
      const existing = s.viewer.projectTable;
      if (!existing) {
        return {
          ...s,
          viewer: {
            ...s.viewer,
            projectTable: { families: [family], rows, activeRowId: rows[0]?.id ?? null, selectedRowIds: rows[0]?.id ? [rows[0].id] : [] },
          },
        };
      }
      const existingFamily = existing.families.find((f) => f.id === family.id);
      if (existingFamily) {
        // Merge new rows into existing family (deduplicate by row id)
        const existingRowIds = new Set(existing.rows.filter((r) => r.familyId === family.id).map((r) => r.id));
        const newRows = rows.filter((r) => !existingRowIds.has(r.id));
        const mergedRowIds = [...existingFamily.rowIds, ...newRows.map((r) => r.id)];
        return {
          ...s,
          viewer: {
            ...s.viewer,
            projectTable: {
              families: existing.families.map((f) => f.id === family.id ? { ...f, rowIds: mergedRowIds } : f),
              rows: [...existing.rows, ...newRows],
              activeRowId: existing.activeRowId,
              selectedRowIds: existing.selectedRowIds || [],
            },
          },
        };
      }
      // New family
      return {
        ...s,
        viewer: {
          ...s.viewer,
          projectTable: {
            families: [...existing.families, family],
            rows: [...existing.rows, ...rows],
            activeRowId: existing.activeRowId,
            selectedRowIds: existing.selectedRowIds || [],
          },
        },
      };
    });

  const buildTableAfterRemoval = (
    s: WorkflowState,
    families: ViewerProjectFamily[],
    rows: ViewerProjectRow[],
  ): WorkflowState => {
    const existing = s.viewer.projectTable!;
    if (families.length === 0) {
      return { ...s, viewer: { ...s.viewer, projectTable: null } };
    }
    const activeRowId = rows.some((r) => r.id === existing.activeRowId)
      ? existing.activeRowId
      : rows[0]?.id ?? null;
    return {
      ...s,
      viewer: {
        ...s.viewer,
        projectTable: {
          families,
          rows,
          activeRowId,
          selectedRowIds: (existing.selectedRowIds || []).filter((id) => rows.some((r) => r.id === id)),
        },
      },
    };
  };

  const removeViewerProjectFamily = (familyId: string) =>
    setState((s) => {
      if (!s.viewer.projectTable) return s;
      return buildTableAfterRemoval(
        s,
        s.viewer.projectTable.families.filter((f) => f.id !== familyId),
        s.viewer.projectTable.rows.filter((r) => r.familyId !== familyId),
      );
    });

  const removeViewerProjectRow = (rowId: string) =>
    setState((s) => {
      if (!s.viewer.projectTable) return s;
      return buildTableAfterRemoval(
        s,
        s.viewer.projectTable.families
          .map((f) => ({ ...f, rowIds: f.rowIds.filter((id) => id !== rowId) }))
          .filter((f) => f.rowIds.length > 0),
        s.viewer.projectTable.rows.filter((r) => r.id !== rowId),
      );
    });

  const renameViewerProjectRow = (rowId: string, newLabel: string) =>
    setState((s) => {
      const existing = s.viewer.projectTable;
      if (!existing) return s;
      return {
        ...s,
        viewer: {
          ...s.viewer,
          projectTable: {
            ...existing,
            rows: existing.rows.map((r) =>
              r.id === rowId ? { ...r, label: newLabel } : r
            ),
          },
        },
      };
    });

  const setViewerProjectActiveRow = (id: string | null) =>
    setState((s) => ({
      ...s,
      viewer: {
        ...s.viewer,
        projectTable: s.viewer.projectTable
          ? { ...s.viewer.projectTable, activeRowId: id, selectedRowIds: id ? [id] : [] }
          : null,
      },
    }));

  const toggleViewerProjectRowSelection = (rowId: string) =>
    setState((s) => {
      const pt = s.viewer.projectTable;
      if (!pt) return s;
      const selected = pt.selectedRowIds || [];
      const isSelected = selected.includes(rowId);
      const nextSelected = isSelected
        ? selected.filter((id) => id !== rowId)
        : [...selected, rowId];
      const nextActive = nextSelected.length > 0
        ? (nextSelected.includes(pt.activeRowId ?? '') ? pt.activeRowId : nextSelected[0])
        : null;
      return {
        ...s,
        viewer: {
          ...s.viewer,
          projectTable: { ...pt, selectedRowIds: nextSelected, activeRowId: nextActive },
        },
      };
    });

  const toggleViewerProjectFamilyCollapsed = (familyId: string) =>
    setState((s) => ({
      ...s,
      viewer: {
        ...s.viewer,
        projectTable: s.viewer.projectTable
          ? {
              ...s.viewer.projectTable,
              families: s.viewer.projectTable.families.map((family) =>
                family.id === familyId ? { ...family, collapsed: !family.collapsed } : family
              ),
            }
          : null,
      },
    }));

  const setViewerProjectFamilySort = (
    familyId: string,
    sortKey: string | null,
    sortDirection: ViewerProjectSortDirection
  ) =>
    setState((s) => ({
      ...s,
      viewer: {
        ...s.viewer,
        projectTable: s.viewer.projectTable
          ? {
              ...s.viewer.projectTable,
              families: s.viewer.projectTable.families.map((family) =>
                family.id === familyId ? { ...family, sortKey, sortDirection } : family
              ),
            }
          : null,
      },
    }));

  const addViewerLayerGroup = (group: ViewerLayerGroup) =>
    setState((s) => ({
      ...s,
      viewer: { ...s.viewer, layerGroups: [...s.viewer.layerGroups, group] },
    }));

  const removeViewerLayerGroup = (id: string) =>
    setState((s) => ({
      ...s,
      viewer: {
        ...s.viewer,
        layerGroups: s.viewer.layerGroups.filter((g) => g.id !== id),
        layers: s.viewer.layers.filter((l) => l.groupId !== id),
      },
    }));

  const toggleViewerLayerGroupExpanded = (id: string) =>
    setState((s) => ({
      ...s,
      viewer: {
        ...s.viewer,
        layerGroups: s.viewer.layerGroups.map((g) =>
          g.id === id ? { ...g, expanded: !g.expanded } : g
        ),
      },
    }));

  const toggleViewerLayerGroupVisible = (id: string) =>
    setState((s) => {
      const group = s.viewer.layerGroups.find((g) => g.id === id);
      if (!group) return s;
      const newVisible = !group.visible;
      return {
        ...s,
        viewer: {
          ...s.viewer,
          layerGroups: s.viewer.layerGroups.map((g) =>
            g.id === id ? { ...g, visible: newVisible } : g
          ),
          layers: s.viewer.layers.map((l) =>
            l.groupId === id ? { ...l, visible: newVisible } : l
          ),
        },
      };
    });

  const clearViewerLayers = () =>
    setState((s) => ({
      ...s,
      viewer: {
        ...s.viewer,
        layers: [],
        layerGroups: [],
        selectedLayerId: null,
      },
    }));

  // Map state setters
  const setMapMethod = (method: MapMethod) =>
    setState((s) => ({ ...s, map: { ...s.map, method } }));

  const setMapStep = (step: MapStep) =>
    setState((s) => ({ ...s, map: { ...s.map, step } }));

  const setMapIsComputing = (isComputing: boolean) =>
    setState((s) => ({ ...s, map: { ...s.map, isComputing } }));

  const setMapProgress = (progress: string, progressPct?: number) =>
    setState((s) => ({
      ...s,
      map: { ...s.map, progress, ...(progressPct !== undefined ? { progressPct } : {}) },
    }));

  const setMapError = (error: string | null) =>
    setState((s) => ({ ...s, map: { ...s.map, error } }));

  const setMapShowMdConfirm = (showMdConfirm: boolean, estimatedTimeMin?: number | null) =>
    setState((s) => ({
      ...s,
      map: {
        ...s.map,
        showMdConfirm,
        ...(estimatedTimeMin !== undefined ? { estimatedTimeMin } : {}),
      },
    }));

  const setMapPdbPath = (pdbPath: string | null) =>
    setState((s) => ({ ...s, map: { ...s.map, pdbPath } }));

  const setMapDetectedLigands = (detectedLigands: DetectedLigand[]) =>
    setState((s) => ({ ...s, map: { ...s.map, detectedLigands } }));

  const setMapSelectedLigandId = (selectedLigandId: string | null) =>
    setState((s) => ({ ...s, map: { ...s.map, selectedLigandId } }));

  const setMapIsDetecting = (isDetecting: boolean) =>
    setState((s) => ({ ...s, map: { ...s.map, isDetecting } }));

  const setMapResult = (result: MapResultState | null) =>
    setState((s) => ({ ...s, map: { ...s.map, result } }));

  const resetMap = () =>
    setState((s) => ({ ...s, map: { ...defaultMapState } }));

  // Conform state setters
  const setConformLigandSdf = (ligandSdfPath: string | null) =>
    setState((s) => ({ ...s, conform: { ...s.conform, ligandSdfPath } }));

  const setConformLigandName = (ligandName: string | null) =>
    setState((s) => ({ ...s, conform: { ...s.conform, ligandName } }));

  const setConformOutputName = (outputName: string) =>
    setState((s) => ({ ...s, conform: { ...s.conform, outputName } }));

  const setConformProtonationConfig = (protonationConfig: Partial<ProtonationConfig>) =>
    setState((s) => ({ ...s, conform: { ...s.conform, protonationConfig: { ...s.conform.protonationConfig, ...protonationConfig } } }));

  const setConformConfig = (config: Partial<ConformerConfig>) =>
    setState((s) => ({ ...s, conform: { ...s.conform, config: { ...s.conform.config, ...config } } }));

  const setConformOutputDir = (outputDir: string | null) =>
    setState((s) => ({ ...s, conform: { ...s.conform, outputDir } }));

  const setConformPaths = (conformerPaths: string[]) =>
    setState((s) => ({ ...s, conform: { ...s.conform, conformerPaths } }));

  const setConformEnergies = (conformerEnergies: Record<string, number>) =>
    setState((s) => ({ ...s, conform: { ...s.conform, conformerEnergies } }));

  const setConformRunning = (isRunning: boolean) =>
    setState((s) => ({ ...s, conform: { ...s.conform, isRunning } }));

  const setScoreEntries = (entries: ScoreComplexEntry[]) =>
    setState((s) => ({ ...s, score: { ...s.score, entries } }));

  const addScoreEntries = (newEntries: ScoreComplexEntry[]) =>
    setState((s) => ({ ...s, score: { ...s.score, entries: [...s.score.entries, ...newEntries] } }));

  const removeScoreEntry = (id: string) =>
    setState((s) => ({ ...s, score: { ...s.score, entries: s.score.entries.filter((e) => e.id !== id) } }));

  const updateScoreEntry = (id: string, updates: Partial<ScoreComplexEntry>) =>
    setState((s) => ({
      ...s,
      score: {
        ...s.score,
        entries: s.score.entries.map((e) => (e.id === id ? { ...e, ...updates } : e)),
      },
    }));

  const setScoreOutputDir = (outputDir: string | null) =>
    setState((s) => ({ ...s, score: { ...s.score, outputDir } }));

  const setScoreRunning = (isRunning: boolean) =>
    setState((s) => ({ ...s, score: { ...s.score, isRunning } }));

  const setScoreCordialAvailable = (cordialAvailable: boolean) =>
    setState((s) => ({ ...s, score: { ...s.score, cordialAvailable } }));

  const setXrayInputDir = (inputDir: string | null) =>
    setState((s) => ({ ...s, xray: { ...s.xray, inputDir } }));

  const setXrayScanResult = (scanResult: XrayDirectoryScanResult | null) =>
    setState((s) => ({ ...s, xray: { ...s.xray, scanResult } }));

  const setXrayOutputDir = (outputDir: string | null) =>
    setState((s) => ({ ...s, xray: { ...s.xray, outputDir } }));

  const setXrayResult = (result: XrayAnalysisResult | null) =>
    setState((s) => ({ ...s, xray: { ...s.xray, result } }));

  const setXrayRunning = (isRunning: boolean) =>
    setState((s) => ({ ...s, xray: { ...s.xray, isRunning } }));

  const applyScoreBatchResults = (results: BatchScoreEntryResult[], cordialAvailable: boolean) => {
    let matchedById = 0;
    let matchedByPath = 0;
    const usedResultIds = new Set<string>();

    setState((s) => ({
      ...s,
      score: {
        ...s.score,
        cordialAvailable,
        entries: s.score.entries.map((entry) => {
          let r = results.find((res) => res.id === entry.id);
          if (r) {
            matchedById += 1;
            usedResultIds.add(r.id);
          } else {
            const pathMatches = results.filter(
              (res) =>
                !usedResultIds.has(res.id) &&
                res.pdbPath === entry.pdbPath &&
                res.ligandId === entry.selectedLigandId,
            );
            if (pathMatches.length === 1) {
              r = pathMatches[0];
              matchedByPath += 1;
              usedResultIds.add(r.id);
            }
          }
          if (!r) return entry;
          return {
            ...entry,
            status: r.status as ScoreComplexEntry['status'],
            errorMessage: r.errorMessage,
            preparedReceptorPath: r.preparedReceptorPath,
            extractedLigandSdfPath: r.extractedLigandSdfPath,
            vinaScore: r.vinaScore,
            cordialExpectedPkd: r.cordialExpectedPkd,
            cordialPHighAffinity: r.cordialPHighAffinity,
            qed: r.qed,
          };
        }),
      },
    }));

    return {
      matchedById,
      matchedByPath,
      unmatchedResults: results.length - usedResultIds.size,
      updatedEntries: usedResultIds.size,
    };
  };

  const clearViewerSession = () =>
    setState((s) => ({
      ...s,
      viewer: { ...defaultViewerState, sessionKey: s.viewer.sessionKey + 1 },
    }));

  const openViewerSession = (options: OpenViewerSessionOptions = {}) =>
    setState((s) => {
      const pdbQueue = options.pdbQueue ? [...options.pdbQueue] : [];
      const pdbQueueIndex = pdbQueue.length > 0
        ? Math.min(Math.max(options.pdbQueueIndex ?? 0, 0), pdbQueue.length - 1)
        : 0;
      const queueItem = pdbQueue[pdbQueueIndex];

      return {
        ...s,
        mode: 'viewer',
        viewer: {
          ...defaultViewerState,
          sessionKey: s.viewer.sessionKey + 1,
          pdbQueue,
          pdbQueueIndex,
          pdbPath: options.pdbPath ?? queueItem?.pdbPath ?? null,
          ligandPath: options.ligandPath ?? queueItem?.ligandPath ?? null,
          trajectoryPath: options.trajectoryPath ?? null,
          trajectoryInfo: options.trajectoryInfo ?? null,
          bindingSiteMap: options.bindingSiteMap ?? null,
          projectTable: options.projectTable ?? s.viewer.projectTable,
        },
      };
    });

  const resetConform = () =>
    setState((s) => ({
      ...s,
      conformStep: 'conform-load' as ConformStep,
      currentPhase: 'idle',
      logs: '',
      errorMessage: null,
      conform: { ...defaultConformState },
    }));

  const resetScore = () =>
    setState((s) => ({
      ...s,
      scoreStep: 'score-load' as ScoreStep,
      currentPhase: 'idle',
      logs: '',
      errorMessage: null,
      isRunning: false,
      runningMode: s.runningMode === 'score' ? null : s.runningMode,
      score: { ...defaultScoreState },
    }));

  const resetXray = () =>
    setState((s) => ({
      ...s,
      xrayStep: 'xray-load' as XrayStep,
      currentPhase: 'idle',
      logs: '',
      errorMessage: null,
      isRunning: false,
      runningMode: s.runningMode === 'xray' ? null : s.runningMode,
      xray: { ...defaultXrayState },
    }));

  const resetViewer = () => {
    console.log('[Store] resetViewer');
    setState((s) => ({
      ...s,
      viewer: { ...defaultViewerState, sessionKey: s.viewer.sessionKey },
    }));
  };

  const resetMd = () =>
    setState((s) => ({
      ...s,
      mdStep: 'md-home',
      currentPhase: 'idle',
      logs: '',
      errorMessage: null,
      isRunning: false,
      runningMode: null,
      md: { ...defaultMDState },
    }));

  const reset = () =>
    setState((s) => ({
      mode: s.mode,
      projectReady: s.projectReady,
      projectDir: s.projectDir,
      dockStep: 'dock-load' as DockStep,
      mdStep: 'md-home',
      conformStep: 'conform-load' as ConformStep,
      scoreStep: 'score-load' as ScoreStep,
      xrayStep: 'xray-load' as XrayStep,
      pdbFile: null,
      customOutputDir: null,
      jobName: '',
      isRunning: false,
      runningMode: null,
      isPaused: false,
      currentPhase: 'idle' as const,
      logs: '',
      errorMessage: null,
      dock: { ...defaultDockState },
      md: { ...defaultMDState },
      viewer: { ...defaultViewerState },
      map: { ...defaultMapState },
      conform: { ...defaultConformState },
      score: { ...defaultScoreState },
      xray: { ...defaultXrayState },
    }));

  // Auto-save project table to disk when it changes
  const tableSaver = createProjectTableSaver(
    () => state().projectDir,
    (path, content) => window.electronAPI.writeTextFile(path, content),
  );
  createEffect(on(
    () => state().viewer.projectTable,
    (table) => tableSaver(table),
    { defer: true },
  ));

  return {
    state,
    setMode,
    setProjectReady,
    setProjectDir,
    setDockStep,
    setMdStep,
    setConformStep,
    setScoreStep,
    setXrayStep,
    setPdbFile,
    setCustomOutputDir,
    setJobName,
    regenerateJobName,
    setIsRunning,
    setIsPaused,
    setCurrentPhase,
    appendLog,
    clearLogs,
    setError,
    // Dock state
    setDockReceptorPdbPath,
    setDockReceptorPrepared,
    setDockReferenceLigandId,
    setDockReferenceLigandPath,
    setDockPreparedLigandPath,
    setDockDetectedLigands,
    setDockLigandSource,
    setDockLigandSdfPaths,
    setDockLigandMolecules,
    setDockConfig,
    setDockCordialConfig,
    setDockProtonationConfig,
    setDockStereoisomerConfig,
    setDockConformerConfig,
    setDockRefinementConfig,
    setDockXtbConfig,
    setDockWaterRetentionConfig,
    setDockCachedConformerPaths,
    setDockOutputDir,
    setDockTotalLigands,
    setDockCompletedLigands,
    setDockResults,
    setDockCordialAvailable,
    setDockCordialScored,
    // MD state
    setMdInputMode,
    setMdSingleMoleculeInput,
    setMdSingleMoleculeThumbnail,
    setMdDockOutputDir,
    setMdLoadedLigands,
    setMdSelectedLigandIndex,
    setMdReceptorPdb,
    setMdLigandSdf,
    setMdLigandName,
    setMdPdbPath,
    setMdThumbnailDataUrl,
    setMdConfig,
    setMdOutputDir,
    setMdResult,
    setMdCurrentStage,
    setMdStageProgress,
    setMdSystemInfo,
    setMdBenchmarkResult,
    setMdIsBenchmarking,
    setMdClusteringResults,
    setMdClusteringResultsFromIpc,
    setMdClusterScores,
    setMdTorsionAnalysis,
    // Viewer state
    setViewerPdbPath,
    setViewerPdbQueue,
    setViewerPdbQueueIndex,
    setViewerLigandPath,
    setViewerDetectedLigands,
    setViewerSelectedLigandId,
    setViewerProteinRep,
    setViewerProteinSurface,
    setViewerProteinSurfaceOpacity,
    setViewerSurfaceColorScheme,
    setViewerProteinCarbonColor,
    setViewerShowPocketResidues,
    setViewerShowPocketLabels,
    setViewerHideWaterIons,
    setViewerLigandVisible,
    setViewerLigandRep,
    setViewerLigandSurface,
    setViewerLigandSurfaceOpacity,
    setViewerLigandCarbonColor,
    setViewerLigandPolarHOnly,
    setViewerShowInteractions,
    setViewerTrajectoryPath,
    setViewerTrajectoryInfo,
    setViewerCurrentFrame,
    setViewerIsPlaying,
    setViewerPlaybackSpeed,
    setViewerSmoothing,
    setViewerLoopPlayback,
    setViewerCenterTarget,
    setViewerClusteringConfig,
    setViewerClusteringResults,
    setViewerIsClustering,
    setViewerLoadedClusters,
    setViewerIsLoadingClusters,
    setViewerClusterVisibility,
    setViewerBindingSiteMap,
    setViewerIsComputingBindingSiteMap,
    setViewerBindingSiteChannel,
    // Layer state
    nextLayerId,
    addViewerLayer,
    removeViewerLayer,
    updateViewerLayer,
    setViewerLayerSelected,
    setViewerProjectTable,
    addViewerProjectFamily,
    removeViewerProjectFamily,
    removeViewerProjectRow,
    renameViewerProjectRow,
    setViewerProjectActiveRow,
    addViewerLayerGroup,
    removeViewerLayerGroup,
    toggleViewerLayerGroupExpanded,
    toggleViewerLayerGroupVisible,
    toggleViewerProjectRowSelection,
    toggleViewerProjectFamilyCollapsed,
    setViewerProjectFamilySort,
    clearViewerLayers,
    // Map state
    setMapMethod,
    setMapStep,
    setMapIsComputing,
    setMapProgress,
    setMapError,
    setMapShowMdConfirm,
    setMapPdbPath,
    setMapDetectedLigands,
    setMapSelectedLigandId,
    setMapIsDetecting,
    setMapResult,
    resetMap,
    // Conform state
    setConformLigandSdf,
    setConformLigandName,
    setConformOutputName,
    setConformProtonationConfig,
    setConformConfig,
    setConformOutputDir,
    setConformPaths,
    setConformEnergies,
    setConformRunning,
    // Score state
    setScoreEntries,
    addScoreEntries,
    removeScoreEntry,
    updateScoreEntry,
    setScoreOutputDir,
    setScoreRunning,
    setScoreCordialAvailable,
    applyScoreBatchResults,
    setXrayInputDir,
    setXrayScanResult,
    setXrayOutputDir,
    setXrayResult,
    setXrayRunning,
    resetConform,
    resetScore,
    resetXray,
    clearViewerSession,
    openViewerSession,
    // Utilities
    getBaseOutputDir: async () => {
      const custom = state().customOutputDir;
      if (custom) return custom;
      return window.electronAPI.getDefaultOutputDir();
    },
    // Resets
    reset,
    resetDock,
    resetMd,
    resetViewer,
  };
}

export const workflowStore = createRoot(createWorkflowStore);

// Expose store for E2E test assertions and state manipulation
if ((window as any).__EMBER_TEST__) {
  (window as any).__emberStore = workflowStore;
}
