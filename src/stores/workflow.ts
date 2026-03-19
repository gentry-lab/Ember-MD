import { createSignal, createRoot } from 'solid-js';
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
  LigandSource,
  DetectedLigand,
  DetectedLigand as DockDetectedLigand,
} from '../../shared/types/dock';

export type WorkflowMode = 'dock' | 'md' | 'score' | 'viewer' | 'map' | 'conform';
export type DockStep = 'dock-load' | 'dock-configure' | 'dock-progress' | 'dock-results';
export type MDStep = 'md-home' | 'md-load' | 'md-configure' | 'md-progress' | 'md-results';
export type ConformStep = 'conform-load' | 'conform-configure' | 'conform-progress' | 'conform-results';

// Map mode types — mirrors PocketMapMethod in shared/types/ipc.ts
import type { PocketMapMethod } from '../../shared/types/ipc';
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
  method?: 'static' | 'solvation' | 'probe';
}

export interface ViewerState {
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
}

export interface MapState {
  method: MapMethod;
  isComputing: boolean;
  progress: string;
  progressPct: number;
  error: string | null;
  showMdConfirm: boolean;
  estimatedTimeMin: number | null;
}

export interface ConformState {
  ligandSdfPath: string | null;
  ligandName: string | null;
  config: ConformerConfig;
  outputDir: string | null;
  conformerPaths: string[];
  isRunning: boolean;
}

export interface PdbFile {
  path: string;
  name: string;
}

export type MDInputMode = 'protein_ligand' | 'ligand_only';

export interface DockState {
  receptorPdbPath: string | null;
  receptorPrepared: string | null;
  referenceLigandId: string | null;
  referenceLigandPath: string | null;
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
}

export interface WorkflowState {
  mode: WorkflowMode;
  projectReady: boolean;
  dockStep: DockStep;
  mdStep: MDStep;
  conformStep: ConformStep;
  pdbFile: PdbFile | null;
  customOutputDir: string | null;
  jobName: string;
  isRunning: boolean;
  isPaused: boolean;
  currentPhase: 'idle' | 'generation' | 'complete' | 'error';
  logs: string;
  errorMessage: string | null;
  dock: DockState;
  md: MDState;
  viewer: ViewerState;
  map: MapState;
  conform: ConformState;
}

const defaultDockState: DockState = {
  receptorPdbPath: null,
  receptorPrepared: null,
  referenceLigandId: null,
  referenceLigandPath: null,
  detectedLigands: [],
  ligandSource: 'sdf_directory',
  ligandSdfPaths: [],
  ligandMolecules: [],
  config: { ...DEFAULT_DOCK_CONFIG },
  cordialConfig: { ...DEFAULT_CORDIAL_CONFIG },
  protonationConfig: { ...DEFAULT_PROTONATION_CONFIG },
  stereoisomerConfig: { ...DEFAULT_STEREOISOMER_CONFIG },
  conformerConfig: { ...DEFAULT_CONFORMER_CONFIG },
  refinementConfig: { ...DEFAULT_REFINEMENT_CONFIG },
  cachedConformerPaths: [],
  dockingOutputDir: null,
  totalLigands: 0,
  completedLigands: 0,
  results: [],
  cordialAvailable: false,
  cordialScored: false,
};

const defaultMDState: MDState = {
  inputMode: 'protein_ligand',
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
};

const defaultMapState: MapState = {
  method: 'static',
  isComputing: false,
  progress: '',
  progressPct: 0,
  error: null,
  showMdConfirm: false,
  estimatedTimeMin: null,
};

const defaultConformState: ConformState = {
  ligandSdfPath: null,
  ligandName: null,
  config: { ...DEFAULT_CONFORMER_CONFIG },
  outputDir: null,
  conformerPaths: [],
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
};

function createWorkflowStore() {
  const initialJobName = generateJobName();

  const [state, setState] = createSignal<WorkflowState>({
    mode: 'viewer',
    projectReady: false,
    dockStep: 'dock-load',
    mdStep: 'md-load',
    conformStep: 'conform-load',
    pdbFile: null,
    customOutputDir: null,
    jobName: initialJobName,
    isRunning: false,
    isPaused: false,
    currentPhase: 'idle',
    logs: '',
    errorMessage: null,
    dock: { ...defaultDockState },
    md: { ...defaultMDState },
    viewer: { ...defaultViewerState },
    map: { ...defaultMapState },
    conform: { ...defaultConformState },
  });

  // Mode selection
  const setMode = (mode: WorkflowMode) => {
    console.log(`[Store] setMode: ${state().mode} → ${mode}`);
    setState((s) => ({ ...s, mode }));
  };
  const setProjectReady = (projectReady: boolean) => setState((s) => ({ ...s, projectReady }));
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

  const setPdbFile = (pdbFile: PdbFile | null) =>
    setState((s) => ({ ...s, pdbFile, errorMessage: null }));

  const setCustomOutputDir = (customOutputDir: string | null) =>
    setState((s) => ({ ...s, customOutputDir }));

  const setJobName = (jobName: string) =>
    setState((s) => ({ ...s, jobName }));

  const regenerateJobName = () =>
    setState((s) => ({ ...s, jobName: generateJobName() }));

  const setIsRunning = (isRunning: boolean) =>
    setState((s) => ({ ...s, isRunning }));

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

  // Viewer state setters
  const setViewerPdbPath = (pdbPath: string | null) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, pdbPath } }));

  const setViewerPdbQueue = (pdbQueue: ViewerQueueItem[]) =>
    setState((s) => ({ ...s, viewer: { ...s.viewer, pdbQueue, pdbQueueIndex: 0 } }));

  const setViewerPdbQueueIndex = (pdbQueueIndex: number) =>
    setState((s) => {
      const item = s.viewer.pdbQueue[pdbQueueIndex];
      return {
        ...s,
        viewer: {
          ...s.viewer,
          pdbQueueIndex,
          pdbPath: item?.pdbPath || s.viewer.pdbPath,
          ligandPath: item?.ligandPath ?? s.viewer.ligandPath,
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

  const resetMap = () =>
    setState((s) => ({ ...s, map: { ...defaultMapState } }));

  // Conform state setters
  const setConformLigandSdf = (ligandSdfPath: string | null) =>
    setState((s) => ({ ...s, conform: { ...s.conform, ligandSdfPath } }));

  const setConformLigandName = (ligandName: string | null) =>
    setState((s) => ({ ...s, conform: { ...s.conform, ligandName } }));

  const setConformConfig = (config: Partial<ConformerConfig>) =>
    setState((s) => ({ ...s, conform: { ...s.conform, config: { ...s.conform.config, ...config } } }));

  const setConformOutputDir = (outputDir: string | null) =>
    setState((s) => ({ ...s, conform: { ...s.conform, outputDir } }));

  const setConformPaths = (conformerPaths: string[]) =>
    setState((s) => ({ ...s, conform: { ...s.conform, conformerPaths } }));

  const setConformRunning = (isRunning: boolean) =>
    setState((s) => ({ ...s, conform: { ...s.conform, isRunning } }));

  const resetConform = () =>
    setState((s) => ({
      ...s,
      conformStep: 'conform-load' as ConformStep,
      currentPhase: 'idle',
      logs: '',
      errorMessage: null,
      conform: { ...defaultConformState },
    }));

  const resetViewer = () => {
    console.log('[Store] resetViewer');
    setState((s) => ({
      ...s,
      viewer: { ...defaultViewerState },
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
      md: { ...defaultMDState },
    }));

  const reset = () =>
    setState((s) => ({
      mode: s.mode,
      projectReady: s.projectReady,
      dockStep: 'dock-load' as DockStep,
      mdStep: 'md-home',
      conformStep: 'conform-load' as ConformStep,
      pdbFile: null,
      customOutputDir: null,
      jobName: '',
      isRunning: false,
      isPaused: false,
      currentPhase: 'idle' as const,
      logs: '',
      errorMessage: null,
      dock: { ...defaultDockState },
      md: { ...defaultMDState },
      viewer: { ...defaultViewerState },
      map: { ...defaultMapState },
      conform: { ...defaultConformState },
    }));

  return {
    state,
    setMode,
    setProjectReady,
    setDockStep,
    setMdStep,
    setConformStep,
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
    addViewerLayerGroup,
    removeViewerLayerGroup,
    toggleViewerLayerGroupExpanded,
    toggleViewerLayerGroupVisible,
    clearViewerLayers,
    // Map state
    setMapMethod,
    setMapIsComputing,
    setMapProgress,
    setMapError,
    setMapShowMdConfirm,
    resetMap,
    // Conform state
    setConformLigandSdf,
    setConformLigandName,
    setConformConfig,
    setConformOutputDir,
    setConformPaths,
    setConformRunning,
    resetConform,
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
