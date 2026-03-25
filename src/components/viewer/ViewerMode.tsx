// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, onMount, onCleanup, createSignal, createEffect, on, Show, batch } from 'solid-js';
import * as NGL from 'ngl';
import type { Vector3 } from 'ngl';
import {
  workflowStore,
} from '../../stores/workflow';
import type {
  ProteinRepresentation,
  LigandRepresentation,
  SurfaceColorScheme,
  BindingSiteMapState,
  ViewerLayer,
  ViewerProjectRow,
  ViewerQueueItem,
} from '../../stores/workflow';
import TrajectoryControls from './TrajectoryControls';
import ClusteringModal from './ClusteringModal';
import AnalysisPanel from './AnalysisPanel';
import LayerPanel from './LayerPanel';
import ProjectTable from './ProjectTable';
import { projectPaths } from '../../utils/projectPaths';
import { loadProjectJob } from '../../utils/projectJobLoader';
import { getVisibleProjectRows } from '../../utils/projectTable';
import { buildImportFamily } from '../../utils/viewerQueue';
import { deserializeAndValidateProjectTable } from '../../utils/projectTablePersistence';
import { theme } from '../../utils/theme';
import type { AtomProxy, ResidueProxy, SelectionSchemeEntry, NglLoadOptions, BindingSiteResultsJson, PreparedPath } from '../../types/ngl';
import type { ProjectJob } from '../../../shared/types/ipc';
import {
  LIGAND_REP_MAP, NGL_LABEL_RADIUS_SIZE, INTERACTION_COLORS, INTERACTION_RADIUS,
  detectInteractions, collectAtoms,
  type DetectedInteraction, type SurfacePropsCacheEntry,
} from '../../utils/interactionDetection';

const ViewerMode: Component = () => {
  const {
    state,
    clearViewerSession,
    setViewerPdbPath,
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
    setViewerPdbQueue,
    setViewerPdbQueueIndex,
    setViewerProjectActiveRow,
    addViewerProjectFamily,
    removeViewerProjectFamily,
    removeViewerProjectRow,
    renameViewerProjectRow,
    setViewerProjectTable,
    toggleViewerProjectRowSelection,
    toggleViewerProjectFamilyCollapsed,
    setViewerProjectFamilySort,
    setViewerBindingSiteMap,
    setViewerIsComputingBindingSiteMap,
    nextLayerId,
    addViewerLayer,
    removeViewerLayer,
    updateViewerLayer,
    setViewerLayerSelected,
    addViewerLayerGroup,
    removeViewerLayerGroup,
    toggleViewerLayerGroupExpanded,
    toggleViewerLayerGroupVisible,
    setMode,
    setMdStep,
    setMdReceptorPdb,
    setMdLigandSdf,
    setMdLigandName,
    setMdPdbPath,
    setMdConfig,
    setDockReceptorPdbPath,
    setDockLigandSdfPaths,
    setDockStep,
  } = workflowStore;

  // eslint-disable-next-line no-unassigned-vars -- SolidJS ref pattern
  let containerRef: HTMLDivElement | undefined;
  let stage: NGL.Stage | null = null;
  let proteinComponent: NGL.Component | null = null;
  let ligandComponent: NGL.Component | null = null;
  let pocketReferenceLigandComponent: NGL.Component | null = null;
  let interactionShapeComponent: NGL.Component | null = null;
  let playbackTimer: ReturnType<typeof setTimeout> | null = null;
  let isFrameLoading = false;
  let pendingFrameIndex: number | null = null;
  let playbackGeneration = 0;
  let loadPdbInFlight: string | null = null;
  let autoLoadPath: string | null = null;
  let autoLoadTrajectoryPath: string | null = null;
  let lastQueueIndex = -1;
  let lastQueueLength = 0;
  let isFirstFrameLoad = true;
  let cachedProteinTopology: { atomCount: number; sessionKey: number } | null = null;
  let viewerCanvasShellRef: HTMLDivElement | undefined;
  let projectTablePanelRef: HTMLDivElement | undefined;
  let resizeObserver: ResizeObserver | null = null;
  let projectTableResizeAnchorRight = 0;
  const alignedComponents: Map<number, NGL.Component> = new Map();  // For multi-PDB alignment
  const volumeComponents: Map<string, NGL.Component> = new Map();  // For binding site isosurfaces
  const layerComponents: Map<string, NGL.Component> = new Map();   // Layer ID → NGL component

  const [isLoading, setIsLoading] = createSignal(false);
  const [loadingStatus, setLoadingStatus] = createSignal('');
  const [error, setError] = createSignal<string | null>(null);
  const [surfacePropsLoadingCount, setSurfacePropsLoadingCount] = createSignal(0);
  const [recentJobs, setRecentJobs] = createSignal<ProjectJob[]>([]);
  const [isLoadingRecentJobs, setIsLoadingRecentJobs] = createSignal(false);
  const [loadingRecentJobId, setLoadingRecentJobId] = createSignal<string | null>(null);
  const [smilesInput, setSmilesInput] = createSignal('');
  const [viewerPdbIdInput, setViewerPdbIdInput] = createSignal('');
  const [isLoadingSmiles, setIsLoadingSmiles] = createSignal(false);
  const [isFetchingViewerPdb, setIsFetchingViewerPdb] = createSignal(false);
  const [projectTableWidth, setProjectTableWidth] = createSignal(300);
  const [structureLoadTick, setStructureLoadTick] = createSignal(0);
  const surfacePropsCache = new Map<string, SurfacePropsCacheEntry>();
  const surfacePropsInflight = new Map<string, Promise<SurfacePropsCacheEntry | null>>();
  let lastViewerSessionKey = state().viewer.sessionKey;
  let recentJobsRequestId = 0;
  let isResizingProjectTable = false;
  let shouldFitProjectTableResize = false;

  const api = window.electronAPI;
  const surfacePropsLoading = () => surfacePropsLoadingCount() > 0;

  // Load persisted project table when entering viewer with a project
  createEffect(on(
    () => [state().projectDir, state().mode] as const,
    async ([projectDir, mode]) => {
      if (mode !== 'viewer' || !projectDir) return;
      if (state().viewer.projectTable) return;
      const raw = await api.readJsonFile(`${projectDir}/project-table.json`);
      if (!raw) return;
      const validated = await deserializeAndValidateProjectTable(raw, api.fileExists);
      if (validated && validated.families.length > 0) {
        setViewerProjectTable(validated);
      }
    },
  ));

  const sortedRecentJobs = (jobs: ProjectJob[]) => {
    return [...jobs].sort((a, b) => {
      const modifiedCmp = (b.lastModified ?? 0) - (a.lastModified ?? 0);
      return modifiedCmp !== 0 ? modifiedCmp : a.label.localeCompare(b.label);
    });
  };

  createEffect(() => {
    const ready = state().projectReady;
    const projectName = state().jobName;

    if (!ready || !projectName) {
      setRecentJobs([]);
      return;
    }

    const requestId = ++recentJobsRequestId;
    setIsLoadingRecentJobs(true);

    void (async () => {
      try {
        const jobs = await api.scanProjectArtifacts(projectName);
        if (requestId !== recentJobsRequestId) return;
        const loadableJobs = sortedRecentJobs(jobs.filter((job) => job.type !== 'docking-pose'));
        setRecentJobs(loadableJobs);
      } catch (err) {
        if (requestId !== recentJobsRequestId) return;
        console.error('[Viewer] Failed to scan project artifacts:', err);
        setRecentJobs([]);
      } finally {
        if (requestId === recentJobsRequestId) {
          setIsLoadingRecentJobs(false);
        }
      }
    })();
  });

  const clearViewerStage = () => {
    playbackGeneration++;
    isFrameLoading = false;
    pendingFrameIndex = null;
    if (playbackTimer) {
      clearTimeout(playbackTimer);
      playbackTimer = null;
    }

    loadPdbInFlight = null;
    autoLoadPath = null;
    autoLoadTrajectoryPath = null;
    lastQueueIndex = -1;
    lastQueueLength = 0;
    isFirstFrameLoad = true;

    if (stage) {
      stage.removeAllComponents();
    }

    volumeComponents.clear();
    proteinComponent = null;
    ligandComponent = null;
    pocketReferenceLigandComponent = null;
    interactionShapeComponent = null;
    cachedProteinTopology = null;
    alignedComponents.clear();
    layerComponents.clear();
    setStructureLoadTick((tick) => tick + 1);
    setIsAligned(false);
    surfacePropsCache.clear();
    surfacePropsInflight.clear();
    setSurfacePropsLoadingCount(0);
    setError(null);
  };

  const hasViewerSession = () =>
    !!state().viewer.pdbPath ||
    !!state().viewer.ligandPath ||
    !!state().viewer.trajectoryPath ||
    state().viewer.pdbQueue.length > 0 ||
    state().viewer.layers.length > 0 ||
    state().viewer.layerGroups.length > 0;

  const hasProjectTable = () => (state().viewer.projectTable?.families.length || 0) > 0;
  const activeProjectRow = () => {
    const table = state().viewer.projectTable;
    return table?.rows.find((row) => row.id === table.activeRowId) || null;
  };
  const visibleProjectRows = () => getVisibleProjectRows(state().viewer.projectTable, projectTableWidth());
  const activeProjectRowIndex = () => visibleProjectRows().findIndex((row) => row.id === activeProjectRow()?.id);

  const fitVisibleStructure = () => {
    if (!stage) return;

    if (proteinComponent && state().viewer.selectedLigandId) {
      const ligandSele = getLigandSelection();
      if (ligandSele) {
        (proteinComponent as NGL.StructureComponent).autoView(ligandSele);
        return;
      }
    }

    if (proteinComponent && !ligandComponent) {
      proteinComponent.autoView();
      return;
    }

    if (ligandComponent && !proteinComponent) {
      ligandComponent.autoView();
      return;
    }

    stage.autoView();
  };

  const clearPocketReferenceLigand = () => {
    if (pocketReferenceLigandComponent && stage) {
      stage.removeComponent(pocketReferenceLigandComponent);
    }
    pocketReferenceLigandComponent = null;
  };

  const isInteractiveTextTarget = (target: EventTarget | null): boolean => {
    if (!(target instanceof HTMLElement)) return false;
    const tagName = target.tagName.toLowerCase();
    return tagName === 'input'
      || tagName === 'textarea'
      || tagName === 'select'
      || target.isContentEditable;
  };

  const handleProjectTableResizeMove = (event: MouseEvent) => {
    if (!isResizingProjectTable) return;
    const nextWidth = Math.max(260, Math.min(640, projectTableResizeAnchorRight - event.clientX));
    setProjectTableWidth(nextWidth);
  };

  const handleProjectTableResizeEnd = () => {
    if (!isResizingProjectTable) return;
    isResizingProjectTable = false;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    if (shouldFitProjectTableResize) {
      shouldFitProjectTableResize = false;
      requestAnimationFrame(() => {
        stage?.handleResize();
        fitVisibleStructure();
      });
    }
  };

  const handleProjectTableResizeStart = (event: MouseEvent) => {
    event.preventDefault();
    isResizingProjectTable = true;
    shouldFitProjectTableResize = true;
    projectTableResizeAnchorRight = projectTablePanelRef?.getBoundingClientRect().right ?? window.innerWidth;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  };

  createEffect(() => {
    const sessionKey = state().viewer.sessionKey;
    if (sessionKey === lastViewerSessionKey) return;
    lastViewerSessionKey = sessionKey;
    clearViewerStage();
  });

  const navigateProjectTable = async (direction: -1 | 1) => {
    const rows = visibleProjectRows();
    if (rows.length === 0) return;
    const currentIndex = activeProjectRowIndex();
    const nextIndex = currentIndex >= 0
      ? Math.min(rows.length - 1, Math.max(0, currentIndex + direction))
      : 0;
    const nextRow = rows[nextIndex];
    if (!nextRow || nextRow.id === activeProjectRow()?.id) return;
    await handleProjectTableRowSelect(nextRow.id);
  };

  const handleViewerKeyDown = (event: KeyboardEvent) => {
    if (state().mode !== 'viewer' || !hasProjectTable() || isInteractiveTextTarget(event.target)) return;

    if (event.key === 'ArrowDown' || event.key === 'ArrowRight') {
      event.preventDefault();
      void navigateProjectTable(1);
      return;
    }

    if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') {
      event.preventDefault();
      void navigateProjectTable(-1);
    }
  };

  onMount(() => {
    window.addEventListener('mousemove', handleProjectTableResizeMove);
    window.addEventListener('mouseup', handleProjectTableResizeEnd);
    window.addEventListener('keydown', handleViewerKeyDown);

    if (typeof ResizeObserver !== 'undefined' && viewerCanvasShellRef) {
      resizeObserver = new ResizeObserver(() => {
        requestAnimationFrame(() => stage?.handleResize());
      });
      resizeObserver.observe(viewerCanvasShellRef);
    }
  });

  // Helper to get ligand selection string for detected ligand
  // Uses resname + resnum (not chain ID) because MDAnalysis may rewrite chain IDs
  // when generating trajectory frame PDBs
  const getLigandSelection = (): string | null => {
    const selectedId = state().viewer.selectedLigandId;
    if (!selectedId) return null;
    const ligand = state().viewer.detectedLigands.find((l) => l.id === selectedId);
    if (!ligand) return null;
    return `[${ligand.resname}] and ${ligand.resnum}`;
  };

  // Counter for unique scheme names + cleanup tracker
  let colorSchemeCounter = 0;
  const registeredSchemes: string[] = [];
  const trackScheme = (id: string) => { registeredSchemes.push(id); return id; };
  const clearOldSchemes = () => {
    while (registeredSchemes.length > 10) {
      const old = registeredSchemes.shift()!;
      try { NGL.ColormakerRegistry.removeScheme(old); } catch { /* best-effort cleanup */ }
    }
  };

  // 3-letter to 1-letter amino acid code mapping
  const aa3to1: Record<string, string> = {
    ALA: 'A', ARG: 'R', ASN: 'N', ASP: 'D', CYS: 'C',
    GLN: 'Q', GLU: 'E', GLY: 'G', HIS: 'H', ILE: 'I',
    LEU: 'L', LYS: 'K', MET: 'M', PHE: 'F', PRO: 'P',
    SER: 'S', THR: 'T', TRP: 'W', TYR: 'Y', VAL: 'V',
    // Non-standard
    MSE: 'M', HSD: 'H', HSE: 'H', HSP: 'H',
  };

  // Interpolate between two hex colors (0xRRGGBB) by t in [0,1]
  const lerpColor = (c1: number, c2: number, t: number): number => {
    const r = ((c1 >> 16) & 0xff) + t * (((c2 >> 16) & 0xff) - ((c1 >> 16) & 0xff));
    const g = ((c1 >> 8) & 0xff) + t * (((c2 >> 8) & 0xff) - ((c1 >> 8) & 0xff));
    const b = (c1 & 0xff) + t * ((c2 & 0xff) - (c1 & 0xff));
    return (Math.round(r) << 16) | (Math.round(g) << 8) | Math.round(b);
  };

  const getSurfacePropsOutputDir = (sourcePath: string) => {
    const sourceDir = sourcePath.substring(0, sourcePath.lastIndexOf('/'));
    return state().customOutputDir
      ? `${state().customOutputDir}/${state().jobName}/surfaces`
      : `${sourceDir}/surfaces`;
  };

  const ensureSurfacePropsLoaded = async (sourcePath: string): Promise<SurfacePropsCacheEntry | null> => {
    if (!sourcePath) return null;

    const cached = surfacePropsCache.get(sourcePath);
    if (cached) return cached;

    const inflight = surfacePropsInflight.get(sourcePath);
    if (inflight) return inflight;

    const viewerSessionKey = state().viewer.sessionKey;
    const loadPromise = (async () => {
      const result = await api.computeSurfaceProps(sourcePath, getSurfacePropsOutputDir(sourcePath));
      if (!result.ok) {
        console.error('[Viewer] Surface property computation failed:', sourcePath, result.error?.message);
        return null;
      }
      surfacePropsCache.set(sourcePath, result.value);
      if (state().viewer.sessionKey === viewerSessionKey) {
        updateAllStyles();
        if (ligandComponent) {
          updateExternalLigandStyle();
        }
      }
      return result.value;
    })().finally(() => {
      surfacePropsInflight.delete(sourcePath);
      setSurfacePropsLoadingCount(surfacePropsInflight.size);
    });

    surfacePropsInflight.set(sourcePath, loadPromise);
    setSurfacePropsLoadingCount(surfacePropsInflight.size);
    return loadPromise;
  };

  const getSurfacePropValues = (
    sourcePath: string | null,
    property: 'hydrophobic' | 'electrostatic',
    expectedAtomCount?: number,
  ): number[] | null => {
    if (!sourcePath) return null;
    const entry = surfacePropsCache.get(sourcePath);
    if (!entry) return null;
    if (expectedAtomCount != null && entry.atomCount !== expectedAtomCount) {
      console.warn(
        `[Viewer] Surface atom-count mismatch for ${sourcePath}: cached=${entry.atomCount} ngl=${expectedAtomCount}`,
      );
      return null;
    }
    return entry[property];
  };

  // Create a surface color scheme from per-atom values for the current structure.
  const createComputedScheme = (
    values: number[],
    lowColor: number,
    midColor: number,
    highColor: number
  ): string => {
    clearOldSchemes();
    const id = `computed-${Date.now()}-${colorSchemeCounter++}`;
    const FALLBACK = 0x888888;
    return trackScheme(NGL.ColormakerRegistry.addScheme(function (this: NGL.Colormaker) {
      this.atomColor = function (atom: AtomProxy) {
        const v = values[atom.index];
        if (v === undefined) return FALLBACK;
        // v is in [-1, 1] — map to color ramp
        const t = (v + 1) / 2; // [0, 1]
        if (t < 0.5) return lerpColor(lowColor, midColor, t * 2);
        return lerpColor(midColor, highColor, (t - 0.5) * 2);
      };
    }, id));
  };

  // Hydrophobic: teal (hydrophilic) → white → goldenrod (hydrophobic)
  const createHydrophobicScheme = (sourcePath: string | null, expectedAtomCount?: number): string | null => {
    const values = getSurfacePropValues(sourcePath, 'hydrophobic', expectedAtomCount);
    if (!values) return null;
    return createComputedScheme(values, 0x3BA8A0, 0xFAFAFA, 0xB8860B);
  };

  // Coulombic electrostatic: red (negative/anionic) → white → blue (positive/cationic)
  // APBS/Maestro convention: blue = positive potential, red = negative potential
  const createElectrostaticScheme = (sourcePath: string | null, expectedAtomCount?: number): string | null => {
    const values = getSurfacePropValues(sourcePath, 'electrostatic', expectedAtomCount);
    if (!values) return null;
    return createComputedScheme(values, 0xD32F2F, 0xFAFAFA, 0x2979FF);
  };

  const getComputedSurfaceColorScheme = (sourcePath: string | null, expectedAtomCount?: number): string | null => {
    switch (state().viewer.surfaceColorScheme) {
      case 'hydrophobic':
        return createHydrophobicScheme(sourcePath, expectedAtomCount);
      case 'electrostatic':
        return createElectrostaticScheme(sourcePath, expectedAtomCount);
      default:
        return null;
    }
  };

  const addColoredSurfaceRepresentation = (
    target: NGL.Component,
    options: {
      opacity: number;
      sourcePath: string | null;
      expectedAtomCount?: number;
      sele?: string;
      solidColor: number | string;
    },
  ) => {
    const { opacity, sourcePath, expectedAtomCount, sele, solidColor } = options;
    const scheme = state().viewer.surfaceColorScheme;
    const surfaceParams: Record<string, any> = { opacity, color: solidColor };
    if (sele) surfaceParams.sele = sele;

    if (scheme === 'uniform-grey') {
      target.addRepresentation('surface', surfaceParams);
      return;
    }

    const colorSchemeId = getComputedSurfaceColorScheme(sourcePath, expectedAtomCount);
    if (colorSchemeId) {
      target.addRepresentation('surface', {
        ...surfaceParams,
        color: colorSchemeId,
      });
      return;
    }

    if (sourcePath) {
      void ensureSurfacePropsLoaded(sourcePath);
    }

    target.addRepresentation('surface', {
      ...surfaceParams,
      color: 0x888888,
    });
  };

  // Create a selection-based color scheme with custom carbon color
  // Uses addSelectionScheme which is simpler and more reliable
  const createCarbonColorScheme = (carbonColorHex: string): string => {
    clearOldSchemes();
    const id = `cpk-${Date.now()}-${colorSchemeCounter++}`;

    // Selection scheme: list of [color, selection] pairs
    // Last entry is the fallback/default
    const schemeData: SelectionSchemeEntry[] = [
      [carbonColorHex, '_C', undefined],      // Carbon - custom color
      ['#3050F8', '_N', undefined],           // Nitrogen - blue
      ['#FF0D0D', '_O', undefined],           // Oxygen - red
      ['#FFFF30', '_S', undefined],           // Sulfur - yellow
      ['#FF8000', '_P', undefined],           // Phosphorus - orange
      ['#FFFFFF', '_H', undefined],           // Hydrogen - white
      ['#1FF01F', '_CL', undefined],          // Chlorine - green
      ['#90E050', '_F', undefined],           // Fluorine - light green
      ['#A62929', '_BR', undefined],          // Bromine - dark red
      ['#940094', '_I', undefined],           // Iodine - purple
      ['#808080', '*', undefined],            // Default - gray
    ];

    return trackScheme(NGL.ColormakerRegistry.addSelectionScheme(schemeData, id));
  };

  // Compute polar hydrogen atom indices for a structure within a selection
  const getPolarHydrogenIndices = (structure: NGL.Structure, selection: string): number[] => {
    const polarHAtoms: number[] = [];
    try {
      structure.eachAtom((atom: AtomProxy) => {
        if (atom.element === 'H') {
          let isPolar = false;
          atom.eachBondedAtom((bonded: AtomProxy) => {
            const el = bonded.element.toUpperCase();
            if (el === 'N' || el === 'O' || el === 'S') {
              isPolar = true;
            }
          });
          if (isPolar) {
            polarHAtoms.push(atom.index);
          }
        }
      }, new NGL.Selection(selection));
    } catch (err) {
      console.warn('Failed to compute polar H:', err);
    }
    return polarHAtoms;
  };

  const [stageReady, setStageReady] = createSignal(false);

  const resetViewerTestState = () => {
    if (!(window as any).__EMBER_TEST__) return;
    (window as any).__viewerTestState = {
      renderedFrameIndex: null,
      coordinateSignature: null,
    };
  };

  const updateViewerTestState = (
    frameIndex: number,
    structure: NGL.Structure | null | undefined,
    component?: NGL.Component | null,
  ) => {
    if (!(window as any).__EMBER_TEST__) return;
    let coordinateSignature: number[] | null = null;
    if (structure) {
      const center = structure.atomCenter();
      coordinateSignature = [
        Number(center.x.toFixed(4)),
        Number(center.y.toFixed(4)),
        Number(center.z.toFixed(4)),
      ];
    }
    if (component && typeof (component as any).getBox === 'function') {
      const box = (component as any).getBox();
      if (box) {
        coordinateSignature = [
          ...(coordinateSignature ?? []),
          Number(box.min.x.toFixed(4)),
          Number(box.min.y.toFixed(4)),
          Number(box.min.z.toFixed(4)),
        ];
      }
    }
    (window as any).__viewerTestState = {
      renderedFrameIndex: frameIndex,
      coordinateSignature,
    };
  };

  onMount(() => {
    if (containerRef) {
      stage = new NGL.Stage(containerRef, {
        backgroundColor: '#ffffff',
      });

      // Dampen shift+scroll clipping plane adjustment for smoother control
      const CLIP_DAMPING = 0.3;
      stage.mouseControls.remove('scroll-shift');
      stage.mouseControls.add('scroll-shift', (stg: any, delta: number) => {
        (NGL as any).MouseActions.focusScroll(stg, delta * CLIP_DAMPING);
      });
      stage.mouseControls.remove('scroll-shift-ctrl');
      stage.mouseControls.add('scroll-shift-ctrl', (stg: any, delta: number) => {
        (NGL as any).MouseActions.zoomFocusScroll(stg, delta * CLIP_DAMPING);
      });

      // Handle window resize
      const handleResize = () => {
        if (stage) {
          stage.handleResize();
        }
      };
      window.addEventListener('resize', handleResize);
      setStageReady(true);
      // Expose stage for E2E test assertions (NGL state queries, not screenshots)
      if ((window as any).__EMBER_TEST__) {
        (window as any).__nglStage = stage;
        resetViewerTestState();
      }
      console.log('[Viewer] NGL Stage created');

      onCleanup(() => {
        console.log('[Viewer] NGL Stage disposing');
        window.removeEventListener('resize', handleResize);
        if (stage) {
          stage.dispose();
          stage = null;
        }
        proteinComponent = null;
        ligandComponent = null;
        interactionShapeComponent = null;
        resetViewerTestState();
      });
    }
  });

  // When viewer mode becomes active again after being CSS-hidden, NGL needs a resize
  // to recalculate canvas dimensions (display:none → display:block changes layout)
  createEffect(() => {
    if (state().mode === 'viewer' && stage) {
      // Defer to next frame so the DOM has finished layout
      requestAnimationFrame(() => stage?.handleResize());
    }
  });

  // React to theme changes — update NGL canvas background
  createEffect(() => {
    if (!stage) return;
    const bg = theme() === 'business' ? '#1d2432' : '#ffffff';
    stage.setParameters({ backgroundColor: bg });
  });


  const updateAllStyles = () => updateProteinStyle();

  // Render detected interactions as dashed cylinders on the stage
  const renderInteractionShape = (interactions: DetectedInteraction[]) => {
    if (interactions.length > 0 && stage) {
      const shape = new NGL.Shape('interactions');
      const DASH = 0.25; // dash length in Å
      const GAP = 0.15;  // gap length in Å
      const STEP = DASH + GAP;
      for (const ix of interactions) {
        // Skip hydrophobic contacts — too noisy to display clearly
        if (ix.type === 'hydrophobic') continue;
        const color = INTERACTION_COLORS[ix.type];
        const dx = ix.to[0] - ix.from[0];
        const dy = ix.to[1] - ix.from[1];
        const dz = ix.to[2] - ix.from[2];
        const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (len < 1e-4) continue;
        const ux = dx / len, uy = dy / len, uz = dz / len;
        let t = 0;
        while (t < len) {
          const tEnd = Math.min(t + DASH, len);
          const p1: [number, number, number] = [
            ix.from[0] + ux * t, ix.from[1] + uy * t, ix.from[2] + uz * t,
          ];
          const p2: [number, number, number] = [
            ix.from[0] + ux * tEnd, ix.from[1] + uy * tEnd, ix.from[2] + uz * tEnd,
          ];
          shape.addCylinder(p1, p2, color, INTERACTION_RADIUS, ix.type);
          t += STEP;
        }
      }
      interactionShapeComponent = stage.addComponentFromObject(shape) as NGL.Component || null;
      if (interactionShapeComponent) interactionShapeComponent.addRepresentation('buffer', {});
    }
  };

  // Update protein representation when state changes
  const updateProteinStyle = () => {
    if (!proteinComponent) return;

    // Remove previous interaction shape (separate stage component)
    if (interactionShapeComponent && stage) {
      stage.removeComponent(interactionShapeComponent);
      interactionShapeComponent = null;
    }

    proteinComponent.removeAllRepresentations();

    const rep = state().viewer.proteinRep;
    const hideWI = state().viewer.hideWaterIons;

    // Base selection for protein (exclude water, ions, and ligand if present)
    const ligandSele = getLigandSelection();
    const proteinPath = state().viewer.pdbPath;
    const proteinBaseSele = ligandSele
      ? `protein and not water and not ion and not (${ligandSele})`
      : 'protein and not water and not ion';
    const structure = (proteinComponent as NGL.StructureComponent).structure;
    const componentAtomCount = structure?.atomCount;

    // Main protein representation
    if (rep === 'spacefill') {
      // For spacefill: use custom colormaker with carbon color
      const proteinColorSchemeId = createCarbonColorScheme(state().viewer.proteinCarbonColor);

      // Always show only polar hydrogens for protein spacefill
      let proteinFullSele = proteinBaseSele;
      if (structure) {
        const polarHIndices = getPolarHydrogenIndices(structure, proteinBaseSele);
        if (polarHIndices.length > 0) {
          proteinFullSele = `(${proteinBaseSele} and not _H) or @${polarHIndices.join(',')}`;
        } else {
          proteinFullSele = `${proteinBaseSele} and not _H`;
        }
      }

      proteinComponent.addRepresentation('spacefill', {
        sele: proteinFullSele,
        color: proteinColorSchemeId,  // Custom scheme uses "color:"
      });
    } else {
      // Cartoon/ribbon - uniform gray
      proteinComponent.addRepresentation(rep, {
        sele: proteinBaseSele,
        color: 0x909090,
      });
    }

    // Water molecules - show as small spheres unless hidden
    if (!hideWI) {
      proteinComponent.addRepresentation('ball+stick', {
        sele: 'water',
        colorScheme: 'element',
        scale: 0.3,
      });
      // Ions - show as spheres unless hidden
      proteinComponent.addRepresentation('spacefill', {
        sele: 'ion',
        colorScheme: 'element',
        scale: 0.5,
      });
    }

    // Protein surface
    if (state().viewer.proteinSurface) {
      addColoredSurfaceRepresentation(proteinComponent, {
        sele: 'protein',
        opacity: state().viewer.proteinSurfaceOpacity,
        sourcePath: proteinPath,
        expectedAtomCount: componentAtomCount,
        solidColor: 0x888888,
      });
    }

    // If there's an auto-detected ligand selected, handle ligand-related visualizations
    if (ligandSele) {
      // Only render ligand representation if visible
      if (state().viewer.ligandVisible) {
        const ligandRep = state().viewer.ligandRep;
        const ligandPolarH = state().viewer.ligandPolarHOnly;

        // Create colormaker with custom carbon color
        const ligandColorSchemeId = createCarbonColorScheme(state().viewer.ligandCarbonColor);

        // Determine which hydrogens to show
        let ligandFullSele = ligandSele;
        if (ligandPolarH) {
          const structure = (proteinComponent as NGL.StructureComponent).structure;
          if (structure) {
            const polarHIndices = getPolarHydrogenIndices(structure, ligandSele);
            if (polarHIndices.length > 0) {
              ligandFullSele = `(${ligandSele} and not _H) or @${polarHIndices.join(',')}`;
            } else {
              ligandFullSele = `${ligandSele} and not _H`;
            }
          }
        }

        // Single representation for entire ligand (preserves bonds)
        // Custom schemes use "color:", built-in schemes use "colorScheme:"
        proteinComponent.addRepresentation(LIGAND_REP_MAP[ligandRep], {
          sele: ligandFullSele,
          color: ligandColorSchemeId,
          multipleBond: 'symmetric',
        });

        // Ligand surface
        if (state().viewer.ligandSurface) {
          addColoredSurfaceRepresentation(proteinComponent, {
            sele: ligandSele,
            opacity: state().viewer.ligandSurfaceOpacity,
            sourcePath: proteinPath,
            expectedAtomCount: componentAtomCount,
            solidColor: ligandColorSchemeId,
          });
        }
      }

      // Pocket residues (protein sidechains within 5A of ligand) - show even if ligand hidden
      // NGL does NOT support "around" in selection strings - must use JavaScript API
      if (state().viewer.showPocketResidues) {
        const structure = (proteinComponent as NGL.StructureComponent).structure;
        if (structure) {
          try {
            // Create selection for ligand atoms
            const ligandSelection = new NGL.Selection(ligandSele);

            // Get all atoms within 5 angstroms of ligand
            const nearbyAtoms = structure.getAtomSetWithinSelection(ligandSelection, 5);

            // Expand to complete residues
            const nearbyResidues = structure.getAtomSetWithinGroup(nearbyAtoms);

            // Convert to selection string
            const nearbyResString = nearbyResidues.toSeleString();

            if (nearbyResString) {
              // Show sidechains of nearby residues, excluding the ligand itself
              let pocketSele = `sidechainAttached and (${nearbyResString}) and not (${ligandSele})`;

              // Always apply polar-H filter for pocket residues
              const pocketPolarH = getPolarHydrogenIndices(structure, pocketSele);
              if (pocketPolarH.length > 0) {
                pocketSele = `(${pocketSele} and not _H) or @${pocketPolarH.join(',')}`;
              } else {
                pocketSele = `${pocketSele} and not _H`;
              }

              proteinComponent.addRepresentation('licorice', {
                sele: pocketSele,
                colorScheme: 'element',
                multipleBond: false,
              });

              // Add pocket residue labels if enabled (single-letter codes, no water/ions)
              if (state().viewer.showPocketLabels) {
                // Build custom label text for each CA atom (excludes water/ions)
                const labelText: Record<number, string> = {};
                const labelSele = `(${nearbyResString}) and .CA and protein and not (${ligandSele})`;

                structure.eachAtom((atom: AtomProxy) => {
                  const resname = atom.resname?.toUpperCase() || '';
                  const resno = atom.resno || '';
                  const oneLetterCode = aa3to1[resname] || resname.charAt(0);
                  labelText[atom.index] = `${oneLetterCode}${resno}`;
                }, new NGL.Selection(labelSele));

                if (Object.keys(labelText).length > 0) {
                  proteinComponent.addRepresentation('label', {
                    sele: labelSele,
                    labelType: 'text',
                    labelText: labelText,
                    labelGrouping: 'residue',
                    color: 'white',
                    fontWeight: 'bold',
                    xOffset: 0,
                    yOffset: 0,
                    zOffset: 2.0,
                    fixedSize: true,
                    radiusType: 'size',
                    radiusSize: NGL_LABEL_RADIUS_SIZE,
                    showBackground: true,
                    backgroundColor: 'black',
                    backgroundOpacity: 0.7,
                    depthWrite: false, // Render on top of other objects
                  });
                }
              }
            }
          } catch (err) {
            console.warn('Failed to compute pocket residues:', err);
          }
        }
      }

      // Protein-ligand interactions (contacts)
      if (state().viewer.showInteractions) {
        const structure = (proteinComponent as NGL.StructureComponent).structure;
        if (structure) {
          try {
            const ligandSelection = new NGL.Selection(ligandSele);
            const nearbyAtoms = structure.getAtomSetWithinSelection(ligandSelection, 5);
            const nearbyResidues = structure.getAtomSetWithinGroup(nearbyAtoms);
            const nearbyResString = nearbyResidues.toSeleString();

            if (nearbyResString) {
              const pocketSele = `protein and (${nearbyResString})`;
              const pAtoms = collectAtoms(structure, pocketSele, true);
              const lAtoms = collectAtoms(structure, ligandSele, false);
              const ixs = detectInteractions(pAtoms, lAtoms);
              renderInteractionShape(ixs);
            }
          } catch (err) {
            console.warn('Failed to compute interactions:', err);
          }
        }
      }
    }

    // Fallback: if no ligand detected yet, render non-protein/non-water/non-ion atoms
    // This ensures ligand-only PDBs (e.g. cluster centroids) are always visible,
    // even before async detection completes. For protein PDBs this selection is
    // typically empty so it adds nothing visible.
    if (!ligandSele && !ligandComponent) {
      proteinComponent.addRepresentation('ball+stick', {
        sele: 'not protein and not water and not ion',
        colorScheme: 'element',
        multipleBond: 'symmetric',
      });
    }

    // Handle pocket residues and interactions for EXTERNAL ligands (loaded from SDF)
    // This is separate because external ligands are in a different NGL component
    const pocketLigandComponent = ligandComponent || pocketReferenceLigandComponent;
    if (!ligandSele && pocketLigandComponent && (state().viewer.showPocketResidues || state().viewer.showInteractions)) {
      const proteinStructure = (proteinComponent as NGL.StructureComponent).structure;
      const ligandStructure = (pocketLigandComponent as NGL.StructureComponent).structure;

      if (proteinStructure && ligandStructure) {
        try {
          // Get all ligand atom positions
          const ligandPositions: { x: number; y: number; z: number }[] = [];
          ligandStructure.eachAtom((atom: AtomProxy) => {
            ligandPositions.push({ x: atom.x, y: atom.y, z: atom.z });
          });
          console.log('[Viewer] Ligand positions:', ligandPositions.length);

          if (ligandPositions.length > 0) {
            // Find protein residues within 5Å of any ligand atom
            const nearbyResidueIndices = new Set<number>();
            const cutoffSq = 5 * 5; // 5 Angstroms squared

            proteinStructure.eachAtom((atom: AtomProxy) => {
              if (atom.residueIndex !== undefined) {
                for (const ligPos of ligandPositions) {
                  const dx = atom.x - ligPos.x;
                  const dy = atom.y - ligPos.y;
                  const dz = atom.z - ligPos.z;
                  const distSq = dx * dx + dy * dy + dz * dz;
                  if (distSq <= cutoffSq) {
                    nearbyResidueIndices.add(atom.residueIndex);
                    break; // Found a nearby ligand atom, no need to check more
                  }
                }
              }
            }, new NGL.Selection('protein'));

            console.log('[Viewer] Nearby residue indices:', nearbyResidueIndices.size);

            if (nearbyResidueIndices.size > 0) {
              // Build a proper selection by getting residue info
              const residueSelections: string[] = [];
              const seenResidues = new Set<string>();
              proteinStructure.eachResidue((residue: ResidueProxy) => {
                if (nearbyResidueIndices.has(residue.index)) {
                  const key = `${residue.resno}:${residue.chainname}`;
                  if (!seenResidues.has(key)) {
                    seenResidues.add(key);
                    residueSelections.push(`(${residue.resno}:${residue.chainname})`);
                  }
                }
              });

              console.log('[Viewer] Residue selections:', residueSelections.length, residueSelections.slice(0, 5));

              if (residueSelections.length > 0) {
                const pocketResiduesSele = residueSelections.join(' or ');

                // Show pocket residues
                if (state().viewer.showPocketResidues) {
                  let pocketSele = `sidechainAttached and (${pocketResiduesSele})`;

                  // Apply polar-H filter
                  const pocketPolarH = getPolarHydrogenIndices(proteinStructure, pocketSele);
                  if (pocketPolarH.length > 0) {
                    pocketSele = `(${pocketSele} and not _H) or @${pocketPolarH.join(',')}`;
                  } else {
                    pocketSele = `${pocketSele} and not _H`;
                  }

                  proteinComponent.addRepresentation('licorice', {
                    sele: pocketSele,
                    colorScheme: 'element',
                    multipleBond: false,
                  });

                  // Add pocket residue labels if enabled
                  if (state().viewer.showPocketLabels) {
                    const labelText: Record<number, string> = {};
                    const labelSele = `(${pocketResiduesSele}) and .CA and protein`;

                    proteinStructure.eachAtom((atom: AtomProxy) => {
                      const resname = atom.resname?.toUpperCase() || '';
                      const resno = atom.resno || '';
                      const oneLetterCode = aa3to1[resname] || resname.charAt(0);
                      labelText[atom.index] = `${oneLetterCode}${resno}`;
                    }, new NGL.Selection(labelSele));

                    if (Object.keys(labelText).length > 0) {
                      proteinComponent.addRepresentation('label', {
                        sele: labelSele,
                        labelType: 'text',
                        labelText: labelText,
                        labelGrouping: 'residue',
                        color: 'white',
                        fontWeight: 'bold',
                        xOffset: 0,
                        yOffset: 0,
                        zOffset: 2.0,
                        fixedSize: true,
                        radiusType: 'size',
                        radiusSize: NGL_LABEL_RADIUS_SIZE,
                        showBackground: true,
                        backgroundColor: 'black',
                        backgroundOpacity: 0.7,
                        depthWrite: false, // Render on top of other objects
                      });
                    }
                  }
                }

                // Cross-structure interactions with external ligand
                if (state().viewer.showInteractions && pocketLigandComponent && ligandStructure) {
                  try {
                    const pAtoms = collectAtoms(proteinStructure, pocketResiduesSele, true);
                    const lAtoms = collectAtoms(ligandStructure, '*', false);
                    const ixs = detectInteractions(pAtoms, lAtoms);
                    renderInteractionShape(ixs);
                  } catch (err) {
                    console.warn('Failed to compute interactions for external ligand:', err);
                  }
                }
              }
            }
          }
        } catch (err) {
          console.warn('Failed to compute pocket for external ligand:', err);
        }
      }
    }
  };

  // Update external ligand (SDF) style
  const updateExternalLigandStyle = (component?: NGL.Component | null) => {
    const target = component || ligandComponent;
    if (!target) return;

    target.removeAllRepresentations();

    const ligandRep = state().viewer.ligandRep;
    const ligandPolarH = state().viewer.ligandPolarHOnly;

    // Create colormaker with custom carbon color
    const ligandColorSchemeId = createCarbonColorScheme(state().viewer.ligandCarbonColor);

    // Determine selection based on polar-H setting
    let sele = '*';  // All atoms by default
    if (ligandPolarH) {
      // Show only polar hydrogens - compute using Structure API
      const structure = (target as NGL.StructureComponent).structure;
      if (structure) {
        const polarHIndices = getPolarHydrogenIndices(structure, '*');
        if (polarHIndices.length > 0) {
          sele = `not _H or @${polarHIndices.join(',')}`;
        } else {
          sele = 'not _H';
        }
      } else {
        sele = 'not _H';
      }
    }

    // Single representation (preserves bonds)
    // Custom schemes use "color:", built-in schemes use "colorScheme:"
    target.addRepresentation(LIGAND_REP_MAP[ligandRep], {
      sele: sele,
      color: ligandColorSchemeId,
      multipleBond: 'symmetric',
    });

    if (state().viewer.ligandSurface) {
      const structure = (target as NGL.StructureComponent).structure;
      addColoredSurfaceRepresentation(target, {
        opacity: state().viewer.ligandSurfaceOpacity,
        sourcePath: state().viewer.ligandPath,
        expectedAtomCount: structure?.atomCount,
        solidColor: ligandColorSchemeId,
      });
    }

  };

  // Load PDB file into viewer (used by both browse and auto-load)
  // preserveExternalLigand: if true, don't clear ligand state (for auto-load with external SDF)
  const handleLoadPdb = async (rawPdbPath: string, preserveExternalLigand: boolean = false) => {
    // Guard against re-entrant calls from reactive effects
    if (loadPdbInFlight === rawPdbPath) return;
    const viewerSessionKey = state().viewer.sessionKey;
    loadPdbInFlight = rawPdbPath;
    setIsLoading(true);
    setError(null);

    try {
      // Auto-prepare raw structures (adds hydrogens for polar-H display)
      const pdbPath = await prepareStructure(rawPdbPath);
      if (state().viewer.sessionKey !== viewerSessionKey) return;
      console.log(`[Viewer] Loading PDB: ${pdbPath} (preserveLigand=${preserveExternalLigand})`);
      surfacePropsCache.delete(pdbPath);

      // Clear binding site volumes from previous PDB
      clearBindingSiteVolumes();

      // Reset ligand state only if not preserving external ligand
      if (!preserveExternalLigand) {
        clearPocketReferenceLigand();
        setViewerLigandPath(null);
        setViewerDetectedLigands([]);
        setViewerSelectedLigandId(null);
      }

      // Always clear the component reference (will be reloaded if needed)
      if (ligandComponent && stage) {
        stage.removeComponent(ligandComponent);
      }
      ligandComponent = null;

      // Load PDB into NGL (disable default representations to control styling)
      if (stage) {
        // Remove only the protein component if preserving ligand, otherwise remove all
        if (proteinComponent) {
          stage.removeComponent(proteinComponent);
        }
        if (!preserveExternalLigand) {
          stage.removeAllComponents();
        }

        proteinComponent = await stageLoadProtein(pdbPath, { defaultRepresentation: false });
        if (state().viewer.sessionKey !== viewerSessionKey) {
          if (proteinComponent && stage) {
            stage.removeComponent(proteinComponent);
          }
          proteinComponent = null;
          return;
        }

        updateProteinStyle();
        setStructureLoadTick((tick) => tick + 1);

        // Cache topology for fast coordinate updates (MD clusters, same-protein docking)
        const loadedStructure = (proteinComponent as NGL.StructureComponent)?.structure;
        if (loadedStructure) {
          cachedProteinTopology = {
            atomCount: loadedStructure.atomCount,
            sessionKey: state().viewer.sessionKey,
          };
        }

        // Detect ligands in the PDB in the background (don't block the viewer)
        if (!preserveExternalLigand) {
          api.detectPdbLigands(pdbPath).then((result) => {
            if (state().viewer.sessionKey !== viewerSessionKey) return;
            // Only apply if this PDB is still the current one (user may have navigated away)
            const ligands = result.ok ? (Array.isArray(result.value) ? result.value : result.value.ligands) : [];
            if (state().viewer.pdbPath === pdbPath && ligands.length > 0) {
              setViewerDetectedLigands(ligands);
              setViewerSelectedLigandId(ligands[0].id);
              updateAllStyles();
              // Center on ligand after detection
              if (proteinComponent) {
                const ligandSele = getLigandSelection();
                if (ligandSele) {
                  (proteinComponent as NGL.StructureComponent).autoView(ligandSele);
                } else {
                  proteinComponent.autoView();
                }
              }
              // Auto-load existing binding site map if available
              tryAutoLoadBindingSiteMap(pdbPath);
            } else {
              // No ligand found — center on whole structure
              if (proteinComponent) proteinComponent.autoView();
            }
          });
        } else {
          // External ligand mode — just center
          stage.autoView();
        }
      }

      if (state().viewer.sessionKey === viewerSessionKey) {
        setViewerPdbPath(pdbPath);
      }
    } catch (err) {
      console.error('[Viewer] Failed to load PDB:', err);
      setError(`Failed to load PDB: ${(err as Error).message}`);
      // Clear the pdb path to prevent auto-load from retrying infinitely
      if (state().viewer.sessionKey === viewerSessionKey) {
        setViewerPdbPath(null);
      }
    } finally {
      loadPdbInFlight = null;
      setIsLoading(false);
    }
  };

  // Import a PDB/CIF into the current project's raw/ directory
  // Loads the raw file directly — NGL handles bond orders natively (CIF bond tables / PDB CCD lookup)
  // and detects interactions via distance-based heuristics without needing explicit H atoms.
  const importToProject = async (sourcePath: string): Promise<string> => {
    const defaultDir = await api.getDefaultOutputDir();
    const baseOutputDir = state().customOutputDir || defaultDir;
    const paths = projectPaths(baseOutputDir, state().jobName);
    const result = await api.importStructure(sourcePath, paths.root);
    if (result.ok) return result.value;
    return sourcePath;
  };

  const normalizeLigandPathForViewer = async (filePath: string): Promise<string> => {
    if (!/\.mol2?$/i.test(filePath)) {
      return filePath;
    }

    const defaultDir = await api.getDefaultOutputDir();
    const baseOutputDir = state().customOutputDir || defaultDir;
    const safeName = (filePath.split('/').pop() || 'ligand')
      .replace(/\.[^.]+$/, '')
      .replace(/[^A-Za-z0-9._-]+/g, '_');
    const uniqueSuffix = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const outputDir = `${baseOutputDir}/${state().jobName}/structures/viewer-normalized/${safeName}-${uniqueSuffix}`;

    const result = await api.convertSingleMolecule(filePath, outputDir, 'mol_file');
    if (!result.ok) {
      throw new Error(result.error?.message || `Failed to normalize ligand for viewing: ${filePath}`);
    }

    return result.value.sdfPath;
  };

  const getViewerSupportDir = async (name: string): Promise<string> => {
    if (state().projectDir) {
      return `${state().projectDir}/structures/${name}`;
    }

    const defaultDir = await api.getDefaultOutputDir();
    const baseOutputDir = state().customOutputDir || defaultDir;
    return `${projectPaths(baseOutputDir, state().jobName).structures}/${name}`;
  };

  const resolvePocketReferenceLigandPath = async (row: ViewerProjectRow): Promise<string | null> => {
    if (row.pocketLigandPath) return row.pocketLigandPath;
    if (!row.pocketSourcePdbPath) return null;

    const detected = await api.detectPdbLigands(row.pocketSourcePdbPath);
    if (!detected.ok) return null;
    const ligands = Array.isArray(detected.value) ? detected.value : detected.value.ligands;
    const ligandId = ligands[0]?.id;
    if (!ligandId) return null;

    const outputDir = await getViewerSupportDir('viewer-pocket-reference');
    const extracted = await api.extractXrayLigand(row.pocketSourcePdbPath, ligandId, outputDir);
    return extracted.ok ? extracted.value.sdfPath : null;
  };

  const loadPocketReferenceLigand = async (row: ViewerProjectRow) => {
    const pocketLigandPath = await resolvePocketReferenceLigandPath(row);
    if (!pocketLigandPath) {
      clearPocketReferenceLigand();
      updateProteinStyle();
      return;
    }
    await handleLoadExternalLigand(pocketLigandPath, {
      target: 'pocket',
      assignViewerLigandPath: false,
      focus: false,
    });
  };

  // Load external ligand from path (supports .sdf, .sdf.gz, .mol, and .mol2)
  const handleLoadExternalLigand = async (
    filePath: string,
    options: {
      target?: 'visible' | 'pocket';
      assignViewerLigandPath?: boolean;
      focus?: boolean;
    } = {},
  ) => {
    const target = options.target ?? 'visible';
    const assignViewerLigandPath = options.assignViewerLigandPath ?? target === 'visible';
    const focus = options.focus ?? target === 'visible';

    console.log('[Viewer] Loading external ligand:', filePath);
    const viewerSessionKey = state().viewer.sessionKey;
    setIsLoading(true);
    setError(null);
    surfacePropsCache.delete(filePath);

    try {
      const normalizedLigandPath = await normalizeLigandPathForViewer(filePath);
      surfacePropsCache.delete(normalizedLigandPath);

      // Load SDF into NGL
      if (stage) {
        if (target === 'visible') {
          clearPocketReferenceLigand();
        }

        const existingComponent = target === 'visible' ? ligandComponent : pocketReferenceLigandComponent;
        if (existingComponent) {
          stage.removeComponent(existingComponent);
        }

        // NGL auto-detects .sdf.gz (format=SDF, compression=gzip) — don't override ext
        // Use firstModelOnly to load only the selected pose (not all conformers)
        const loadOptions: NglLoadOptions = {
          firstModelOnly: true,
        };

        console.log('[Viewer] Loading ligand file:', normalizedLigandPath, 'options:', loadOptions);
        const loadedComponent = await stage.loadFile(normalizedLigandPath, loadOptions) as NGL.Component || null;
        if (state().viewer.sessionKey !== viewerSessionKey) {
          if (loadedComponent && stage) {
            stage.removeComponent(loadedComponent);
          }
          if (target === 'visible') {
            ligandComponent = null;
          } else {
            pocketReferenceLigandComponent = null;
          }
          return;
        }

        if (target === 'visible') {
          ligandComponent = loadedComponent;
        } else {
          pocketReferenceLigandComponent = loadedComponent;
          pocketReferenceLigandComponent?.setVisibility(false);
        }

        if (state().viewer.sessionKey !== viewerSessionKey) return;

        if (target === 'visible') {
          updateExternalLigandStyle();
        }

        if (proteinComponent) {
          console.log('[Viewer] Updating protein style after ligand load');
          updateProteinStyle();
        }

        if (focus && ligandComponent) {
          ligandComponent.autoView();
        } else if (focus) {
          stage.autoView();
        }
      }

      if (assignViewerLigandPath && state().viewer.sessionKey === viewerSessionKey) {
        setViewerLigandPath(normalizedLigandPath);
      }
    } catch (err) {
      console.error('[Viewer] Failed to load ligand:', err);
      setError(`Failed to load ligand: ${(err as Error).message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLoadStandaloneLigand = async (filePath: string) => {
    if (!stage) return;
    clearBindingSiteVolumes();
    clearPocketReferenceLigand();
    if (interactionShapeComponent) {
      stage.removeComponent(interactionShapeComponent);
      interactionShapeComponent = null;
    }
    if (isAligned()) clearAlignment();
    setViewerDetectedLigands([]);
    setViewerSelectedLigandId(null);

    if (proteinComponent) {
      stage.removeComponent(proteinComponent);
      proteinComponent = null;
    }

    if (ligandComponent) {
      stage.removeComponent(ligandComponent);
      ligandComponent = null;
    }

    await handleLoadExternalLigand(filePath, {
      target: 'visible',
      assignViewerLigandPath: true,
      focus: true,
    });

    if (state().viewer.pdbPath !== state().viewer.ligandPath) {
      setViewerPdbPath(state().viewer.ligandPath);
    }
  };

  /**
   * Fast path: update protein coordinates in-place without destroying/recreating
   * the NGL component or its representations. Used for MD cluster centroids and
   * other same-topology navigation where only atom positions change.
   *
   * Returns true if update succeeded, false if caller should fall back to full reload.
   */
  const updateProteinCoordinatesInPlace = async (rawPdbPath: string): Promise<boolean> => {
    if (!stage || !proteinComponent || !cachedProteinTopology) return false;
    if (cachedProteinTopology.sessionKey !== state().viewer.sessionKey) return false;

    const existingStructure = (proteinComponent as NGL.StructureComponent)?.structure;
    if (!existingStructure) return false;

    try {
      const pdbPath = await prepareStructure(rawPdbPath);

      // Load new PDB structure-only (no representations — cheap PDB parse)
      const tempComp = await stage.loadFile(pdbPath, {
        defaultRepresentation: false,
      }) as NGL.Component | null;
      if (!tempComp) return false;

      const newStructure = (tempComp as NGL.StructureComponent)?.structure;
      const atomMatch = newStructure && newStructure.atomCount === existingStructure.atomCount;

      if (atomMatch) {
        // Extract xyz coordinates from the new structure's atom store
        const n = newStructure.atomCount;
        const coords = new Float32Array(n * 3);
        const store = newStructure.atomStore as { x: Float32Array; y: Float32Array; z: Float32Array };
        for (let i = 0; i < n; i++) {
          coords[i * 3]     = store.x[i];
          coords[i * 3 + 1] = store.y[i];
          coords[i * 3 + 2] = store.z[i];
        }

        // Apply to existing component — NGL refreshes representations automatically
        existingStructure.updatePosition(coords);

        // Update store + surface cache
        setViewerPdbPath(pdbPath);
        surfacePropsCache.delete(pdbPath);
        setStructureLoadTick((tick) => tick + 1);

        // Re-center on ligand if one is detected
        const ligandSele = getLigandSelection();
        if (ligandSele) {
          (proteinComponent as NGL.StructureComponent).autoView(ligandSele);
        } else {
          proteinComponent!.autoView();
        }
      }

      // Always clean up the temp component (whether match or not)
      stage.removeComponent(tempComp);
      return !!atomMatch;
    } catch (err) {
      console.warn('[Viewer] Coordinate update failed, falling back to full reload:', err);
      return false;
    }
  };

  const loadViewerQueueItem = async (item: ViewerQueueItem) => {
    if (item.type === 'conformer' || item.type === 'ligand') {
      console.log(`[Viewer] Queue nav (conformer): ${item.label} — ${item.pdbPath}`);
      await handleLoadStandaloneLigand(item.pdbPath);
      return;
    }

    if (item.ligandPath) {
      console.log(`[Viewer] Queue nav (docking): ${item.label} — ligand=${item.ligandPath}`);
      const currentPdb = state().viewer.pdbPath;
      if (currentPdb !== item.pdbPath || !proteinComponent) {
        await handleLoadPdb(item.pdbPath, true);
      }
      await handleLoadExternalLigand(item.ligandPath);
      return;
    }

    if (isAligned() && alignedComponents.has(state().viewer.pdbQueueIndex)) {
      for (const [i, comp] of alignedComponents.entries()) {
        comp.setVisibility(i === state().viewer.pdbQueueIndex);
      }
      proteinComponent = alignedComponents.get(state().viewer.pdbQueueIndex) || null;
      return;
    }

    // Fast path: update coordinates in-place when topology matches (MD clusters)
    if (cachedProteinTopology && proteinComponent && !isAligned()) {
      setIsLoading(true);
      try {
        const updated = await updateProteinCoordinatesInPlace(item.pdbPath);
        if (updated) {
          console.log('[Viewer] Fast coordinate update:', item.label);
          return;
        }
      } finally {
        setIsLoading(false);
      }
      console.log('[Viewer] Atom count mismatch, full reload:', item.label);
    }

    console.log('[Viewer] Queue navigation to:', item.label, item.pdbPath);
    clearPocketReferenceLigand();
    if (isAligned()) clearAlignment();
    await handleLoadPdb(item.pdbPath, false);
  };

  // React to PDB queue navigation (page-turn arrows + initial load from View 3D)
  // eslint-disable-next-line solid/reactivity -- async load is intentional
  createEffect(async () => {
    const queue = state().viewer.pdbQueue;
    const idx = state().viewer.pdbQueueIndex;
    // Fire on: index change (nav arrows) OR queue just populated (View 3D / structure import)
    const queueJustPopulated = queue.length > 1 && lastQueueLength <= 1;
    const indexChanged = idx !== lastQueueIndex && lastQueueIndex >= 0;
    lastQueueLength = queue.length;
    if (queue.length > 1 && (indexChanged || queueJustPopulated) && stageReady()) {
      lastQueueIndex = idx;
      const item = queue[idx];
      if (!item) { lastQueueIndex = idx; return; }
      await loadViewerQueueItem(item);
    }
    lastQueueIndex = idx;
  });

  // Auto-load files from store when pdbPath/ligandPath change.
  // Guard against re-entrant calls: handleLoadPdb is async and mutates store
  // state (detectedLigands, selectedLigandId) which would re-trigger this effect
  // while proteinComponent is still null, causing an infinite load loop.
  // eslint-disable-next-line solid/reactivity -- async load is intentional
  createEffect(async () => {
    structureLoadTick();
    if (!stageReady()) return;

    const pdbPath = state().viewer.pdbPath;
    const ligandPath = state().viewer.ligandPath;
    const queue = state().viewer.pdbQueue;

    // Skip if already loading this path or already loaded
    if (!pdbPath || proteinComponent || pdbPath === autoLoadPath) return;

    // If a queue is set, let the queue nav effect handle loading (not auto-load)
    if (queue.length > 1) return;

    autoLoadPath = pdbPath;
    const hasExternalLigand = !!ligandPath;

    console.log('[Viewer] Auto-loading PDB:', pdbPath, 'preserveExternalLigand:', hasExternalLigand);
    await handleLoadPdb(pdbPath, hasExternalLigand);

    // Load external ligand after PDB is loaded (if specified)
    if (ligandPath && !ligandComponent) {
      console.log('[Viewer] Auto-loading ligand:', ligandPath);
      await handleLoadExternalLigand(ligandPath);
    }

    // Clear the guard so a new path can be loaded later
    autoLoadPath = null;
  });

  // Auto-load DCD after the topology is present. This is the same path used by the explicit DCD picker.
  // eslint-disable-next-line solid/reactivity -- async load is intentional
  createEffect(async () => {
    structureLoadTick();
    if (!stageReady() || !proteinComponent) return;

    const trajectoryPath = state().viewer.trajectoryPath;
    const topologyPath = state().viewer.pdbPath;
    const trajectoryInfo = state().viewer.trajectoryInfo;

    if (!trajectoryPath || !topologyPath || trajectoryInfo || trajectoryPath === autoLoadTrajectoryPath) return;

    autoLoadTrajectoryPath = trajectoryPath;
    await handleLoadTrajectory(trajectoryPath);
    autoLoadTrajectoryPath = null;
  });

  const handleResetView = () => {
    if (stage) {
      // Reset clipping planes, fog, and camera
      stage.setParameters({
        clipNear: 0,
        clipFar: 100,
        clipDist: 10,
        fogNear: 50,
        fogFar: 100,
      });
      stage.autoView();
    }
  };

  const requestVisibleSurfaceProps = () => {
    if (state().viewer.surfaceColorScheme === 'uniform-grey') return;

    const paths = new Set<string>();
    const pdbPath = state().viewer.pdbPath;
    const ligandPath = state().viewer.ligandPath;

    if (pdbPath && (state().viewer.proteinSurface || (state().viewer.selectedLigandId && state().viewer.ligandSurface))) {
      paths.add(pdbPath);
    }
    if (ligandPath && ligandComponent && state().viewer.ligandSurface) {
      paths.add(ligandPath);
    }

    for (const sourcePath of paths) {
      void ensureSurfacePropsLoaded(sourcePath);
    }
  };

  const handleProteinRepChange = (rep: ProteinRepresentation) => {
    setViewerProteinRep(rep);
    updateAllStyles();
  };

  const handleProteinSurfaceToggle = () => {
    const next = !state().viewer.proteinSurface;
    setViewerProteinSurface(next);
    if (next) {
      requestVisibleSurfaceProps();
    }
    updateAllStyles();
  };

  const handleProteinSurfaceOpacityChange = (opacity: number) => {
    setViewerProteinSurfaceOpacity(opacity);
    updateAllStyles();
  };

  const handleSurfaceColorChange = async (scheme: SurfaceColorScheme) => {
    setViewerSurfaceColorScheme(scheme);
    if (scheme !== 'uniform-grey') {
      requestVisibleSurfaceProps();
    }

    updateAllStyles();
    if (ligandComponent) {
      updateExternalLigandStyle();
    }
  };

  const handleProteinCarbonColorChange = (color: string) => {
    setViewerProteinCarbonColor(color);
    updateAllStyles();
  };

  const handlePocketResiduesToggle = () => {
    setViewerShowPocketResidues(!state().viewer.showPocketResidues);
    updateAllStyles();
  };

  const handlePocketLabelsToggle = () => {
    setViewerShowPocketLabels(!state().viewer.showPocketLabels);
    updateAllStyles();
  };

  const handleHideWaterIonsToggle = () => {
    setViewerHideWaterIons(!state().viewer.hideWaterIons);
    updateAllStyles();
  };

  const handleLigandVisibleToggle = () => {
    setViewerLigandVisible(!state().viewer.ligandVisible);
    updateAllStyles();
  };

  const handleLigandPolarHToggle = () => {
    setViewerLigandPolarHOnly(!state().viewer.ligandPolarHOnly);
    // Update both internal (PDB) and external (SDF) ligand representations
    if (state().viewer.selectedLigandId) {
      updateAllStyles();
    }
    if (ligandComponent) {
      updateExternalLigandStyle();
    }
  };

  const handleExportPdb = async () => {
    const exportComponent = proteinComponent || ligandComponent;
    if (!exportComponent) return;

    try {
      // Get the structure from NGL and export as PDB
      const structure = (exportComponent as NGL.StructureComponent).structure;
      if (!structure) return;

      // Use NGL's built-in PDB writer
      const pdbWriter = new NGL.PdbWriter(structure);
      const pdbString = pdbWriter.getString();

      // Generate a default filename
      const currentPath = state().viewer.ligandPath || state().viewer.pdbPath;
      const baseName = currentPath
        ? currentPath.split('/').pop()?.replace(/\.(pdb|sdf|sdf\.gz|mol|mol2)$/i, '')
        : 'complex';
      const defaultName = `${baseName}_export.pdb`;

      // Save via IPC
      const savedPath = await api.savePdbFile(pdbString, defaultName);
      if (savedPath) {
        console.log('Saved PDB to:', savedPath);
      }
    } catch (err) {
      setError(`Failed to export PDB: ${(err as Error).message}`);
    }
  };

  const handleLigandRepChange = (rep: LigandRepresentation) => {
    setViewerLigandRep(rep);
    if (state().viewer.selectedLigandId) {
      updateAllStyles();
    } else if (ligandComponent) {
      updateExternalLigandStyle();
    }
  };

  const handleLigandSurfaceToggle = () => {
    const next = !state().viewer.ligandSurface;
    setViewerLigandSurface(next);
    if (next) {
      requestVisibleSurfaceProps();
    }
    if (state().viewer.selectedLigandId) {
      updateAllStyles();
    } else if (ligandComponent) {
      updateExternalLigandStyle();
    }
  };

  const handleLigandSurfaceOpacityChange = (opacity: number) => {
    setViewerLigandSurfaceOpacity(opacity);
    if (state().viewer.selectedLigandId) {
      updateAllStyles();
    } else if (ligandComponent) {
      updateExternalLigandStyle();
    }
  };

  const handleLigandCarbonColorChange = (color: string) => {
    setViewerLigandCarbonColor(color);
    if (state().viewer.selectedLigandId) {
      updateAllStyles();
    } else if (ligandComponent) {
      updateExternalLigandStyle();
    }
  };

  const handleInteractionsToggle = () => {
    setViewerShowInteractions(!state().viewer.showInteractions);
    updateAllStyles();
  };

  // === Trajectory functions ===

  const handleLoadTrajectory = async (dcdPath: string) => {
    const viewerSessionKey = state().viewer.sessionKey;
    setIsLoading(true);
    setError(null);

    try {
      const pdbPath = state().viewer.pdbPath;
      if (!pdbPath) {
        setError('No topology PDB loaded');
        return;
      }

      handleTrajectoryPause();
      isFirstFrameLoad = true;
      setViewerTrajectoryInfo(null);
      resetViewerTestState();

      // Get trajectory info from backend
      const infoResult = await api.getTrajectoryInfo(pdbPath, dcdPath);
      if (state().viewer.sessionKey !== viewerSessionKey) return;
      if (!infoResult.ok) {
        setError(infoResult.error.message);
        return;
      }

      setViewerTrajectoryInfo(infoResult.value);
      setViewerTrajectoryPath(dcdPath);
      setViewerCurrentFrame(0);

      console.log('[Viewer] Trajectory info loaded:', dcdPath, 'frames:', infoResult.value.frameCount);

      // Load the first frame
      if (state().viewer.sessionKey === viewerSessionKey) {
        await loadTrajectoryFrame(0);
      }
    } catch (err) {
      console.error('[Viewer] Failed to load trajectory:', err);
      setError(`Failed to load trajectory: ${(err as Error).message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Helper: get the center-of-mass of a selection within the current proteinComponent
  const getSelectionCenter = (sele: string): Vector3 | null => {
    if (!proteinComponent) return null;
    try {
      const structure = (proteinComponent as NGL.StructureComponent).structure;
      if (!structure) return null;
      const selection = new NGL.Selection(sele);
      return structure.atomCenter(selection);
    } catch {
      return null;
    }
  };

  // Helper: get the center to track based on centerTarget setting
  const getTrackingCenter = (): Vector3 | null => {
    const centerTarget = state().viewer.centerTarget;
    if (centerTarget === 'none') return null;

    if (centerTarget === 'ligand' && hasAnyLigand()) {
      const ligandSele = getLigandSelection();
      if (ligandSele) {
        return getSelectionCenter(ligandSele);
      }
    } else if (centerTarget === 'protein') {
      return getSelectionCenter('protein');
    }
    // Fallback: center of all atoms
    if (!proteinComponent) return null;
    try {
      const structure = (proteinComponent as NGL.StructureComponent).structure;
      return structure ? structure.atomCenter() : null;
    } catch {
      return null;
    }
  };

  // Internal: load a specific frame from the backend as PDB and reload in NGL
  const loadTrajectoryFrameInternal = async (frameIndex: number) => {
    const pdbPath = state().viewer.pdbPath;
    const dcdPath = state().viewer.trajectoryPath;
    if (!pdbPath || !dcdPath || !stage) {
      console.error('[Viewer] Missing requirements for frame load:', { pdbPath, dcdPath, stage: !!stage });
      return;
    }

    try {
      // Fast path: update coordinates in-place via atomStore (no PDB parsing, no component recreation)
      // Requires topology already loaded from the first frame
      if (!isFirstFrameLoad && proteinComponent) {
        const coordsResult = await api.getTrajectoryCoords(pdbPath, dcdPath, frameIndex);
        if (!coordsResult.ok) {
          console.error('[Viewer] Failed to load coords:', coordsResult.error.message);
          setError(`Failed to load frame ${frameIndex}: ${coordsResult.error.message}`);
          return;
        }

        const structure = (proteinComponent as NGL.StructureComponent).structure;
        const expectedAtoms = structure.atomCount;

        if (coordsResult.value.atomCount !== expectedAtoms) {
          console.warn(`[Viewer] Atom count mismatch: coords=${coordsResult.value.atomCount} vs NGL=${expectedAtoms}, falling back to full PDB reload`);
          // Fall through to full PDB path below
        } else {
          // Decode base64 float32 coordinates
          const raw = atob(coordsResult.value.coordsBase64);
          const buffer = new ArrayBuffer(raw.length);
          const view = new Uint8Array(buffer);
          for (let i = 0; i < raw.length; i++) view[i] = raw.charCodeAt(i);
          const coords = new Float32Array(buffer);

          // Update positions in-place — triggers bounding box recalc + representation refresh
          structure.updatePosition(coords);

          setViewerCurrentFrame(frameIndex);
          updateViewerTestState(frameIndex, structure, proteinComponent);

          // Track center target without changing camera rotation/zoom
          const centerTarget = state().viewer.centerTarget;
          if (centerTarget !== 'none') {
            const newCenter = getTrackingCenter();
            if (newCenter) {
              stage.viewerControls.center(newCenter);
            }
          }
          return;
        }
      }

      // Full PDB path: first frame (establishes topology) or fallback on atom count mismatch
      const frameResult = await api.getTrajectoryFrame(pdbPath, dcdPath, frameIndex);
      if (!frameResult.ok) {
        console.error('[Viewer] Failed to load frame:', frameResult.error.message);
        setError(`Failed to load frame ${frameIndex}: ${frameResult.error.message}`);
        return;
      }

      const pdbString = frameResult.value.pdbString;
      if (!pdbString || pdbString.length < 100) {
        console.error('[Viewer] PDB string too short:', pdbString?.length);
        setError('Received empty or invalid frame data');
        return;
      }

      console.log(`[Viewer] Loading frame ${frameIndex}, PDB size: ${pdbString.length} bytes`);

      // Save camera orientation before destroying old component
      // This preserves the user's rotation and zoom across frame changes
      const savedOrientation = !isFirstFrameLoad
        ? stage.viewerControls.getOrientation()
        : null;

      // Create a Blob from the PDB string and load it into NGL
      const pdbBlob = new Blob([pdbString], { type: 'text/plain' });

      // Remove old protein component
      if (proteinComponent) {
        stage.removeComponent(proteinComponent);
        proteinComponent = null;
      }

      // Load new frame
      proteinComponent = await stage.loadFile(pdbBlob, {
        ext: 'pdb',
        defaultRepresentation: false
      }) as NGL.Component || null;

      if (!proteinComponent) {
        console.error('[Viewer] Failed to create protein component');
        setError('Failed to load structure into viewer');
        return;
      }

      // Re-apply styling
      updateProteinStyle();

      // Update current frame in state
      setViewerCurrentFrame(frameIndex);
      updateViewerTestState(frameIndex, (proteinComponent as NGL.StructureComponent).structure, proteinComponent);

      // Center view on target
      // First frame: use autoView on target to establish initial rotation + zoom + center
      // Subsequent frames: restore saved rotation/zoom, update only the center position
      //   This prevents tumbling while keeping the molecule tracked
      // NOTE: Stage.autoView() only takes duration; StructureComponent.autoView(sele) takes selection
      if (isFirstFrameLoad) {
        // Establish the initial view centered on the tracking target
        const centerTarget = state().viewer.centerTarget;
        if (centerTarget === 'ligand' && hasAnyLigand()) {
          const ligandSele = getLigandSelection();
          if (ligandSele) {
            (proteinComponent as NGL.StructureComponent).autoView(ligandSele);
          } else {
            proteinComponent.autoView();
          }
        } else if (centerTarget === 'protein') {
          (proteinComponent as NGL.StructureComponent).autoView('protein');
        } else {
          proteinComponent.autoView();
        }
        isFirstFrameLoad = false;
      } else if (savedOrientation) {
        // Restore camera rotation + zoom from before frame change
        stage.viewerControls.orient(savedOrientation);

        // Now update only the center position to track the target
        // This keeps the molecule centered without changing the viewing angle
        const centerTarget = state().viewer.centerTarget;
        if (centerTarget !== 'none') {
          const newCenter = getTrackingCenter();
          if (newCenter) {
            stage.viewerControls.center(newCenter);
          }
        }
      }
    } catch (err) {
      console.error('[Viewer] Error loading frame:', err);
      setError(`Error loading frame: ${(err as Error).message}`);
    }
  };

  // Load a specific frame with mutex to prevent concurrent loads
  const loadTrajectoryFrame = async (frameIndex: number) => {
    if (isFrameLoading) {
      pendingFrameIndex = frameIndex;
      return;
    }

    isFrameLoading = true;
    pendingFrameIndex = null;

    try {
      await loadTrajectoryFrameInternal(frameIndex);
    } finally {
      isFrameLoading = false;

      if (pendingFrameIndex !== null) {
        const next = pendingFrameIndex;
        pendingFrameIndex = null;
        queueMicrotask(() => loadTrajectoryFrame(next));
      }
    }
  };

  // Playback control functions
  const handleTrajectorySeek = async (frame: number) => {
    await loadTrajectoryFrame(frame);
  };

  const handleTrajectoryPlay = () => {
    if (!state().viewer.trajectoryPath) return;

    setViewerIsPlaying(true);

    const info = state().viewer.trajectoryInfo;
    if (!info) return;

    // Increment generation to cancel any prior playback loop
    const currentGeneration = ++playbackGeneration;

    const runPlaybackLoop = async () => {
      while (playbackGeneration === currentGeneration) {
        const speed = state().viewer.playbackSpeed;
        const smoothing = state().viewer.smoothing;
        const currentFrame = state().viewer.currentFrame;
        const totalFrames = info.frameCount;

        let nextFrame = currentFrame + smoothing;

        if (nextFrame >= totalFrames) {
          if (state().viewer.loopPlayback) {
            nextFrame = 0;
          } else {
            handleTrajectoryPause();
            return;
          }
        }

        const frameStart = performance.now();
        await loadTrajectoryFrame(nextFrame);
        const elapsed = performance.now() - frameStart;

        // If cancelled during load, exit
        if (playbackGeneration !== currentGeneration) return;

        // Calculate desired interval, subtract time already spent loading
        // Base rate: 10 fps (each frame is a full IPC round-trip so this is ambitious)
        const baseIntervalMs = 1000 / 10;
        const intervalMs = baseIntervalMs / speed;
        const remainingDelay = Math.max(0, intervalMs - elapsed);

        if (remainingDelay > 0) {
          await new Promise<void>((resolve) => {
            playbackTimer = setTimeout(resolve, remainingDelay);
          });
          playbackTimer = null;
        }
      }
    };

    runPlaybackLoop();
  };

  const handleTrajectoryPause = () => {
    setViewerIsPlaying(false);
    playbackGeneration++;
    pendingFrameIndex = null;

    if (playbackTimer) {
      clearTimeout(playbackTimer);
      playbackTimer = null;
    }
  };

  const handleTrajectoryStepForward = async () => {
    if (!state().viewer.trajectoryPath) return;

    const currentFrame = state().viewer.currentFrame;
    const totalFrames = state().viewer.trajectoryInfo?.frameCount || 1;
    const smoothing = state().viewer.smoothing;

    const nextFrame = Math.min(currentFrame + smoothing, totalFrames - 1);
    await loadTrajectoryFrame(nextFrame);
  };

  const handleTrajectoryStepBackward = async () => {
    if (!state().viewer.trajectoryPath) return;

    const currentFrame = state().viewer.currentFrame;
    const smoothing = state().viewer.smoothing;

    const prevFrame = Math.max(currentFrame - smoothing, 0);
    await loadTrajectoryFrame(prevFrame);
  };

  const handleTrajectoryFirstFrame = async () => {
    if (!state().viewer.trajectoryPath) return;
    await loadTrajectoryFrame(0);
  };

  const handleTrajectoryLastFrame = async () => {
    if (!state().viewer.trajectoryPath) return;
    const totalFrames = state().viewer.trajectoryInfo?.frameCount || 1;
    await loadTrajectoryFrame(totalFrames - 1);
  };

  // Clean up playback on unmount
  onCleanup(() => {
    playbackGeneration++;
    window.removeEventListener('mousemove', handleProjectTableResizeMove);
    window.removeEventListener('mouseup', handleProjectTableResizeEnd);
    window.removeEventListener('keydown', handleViewerKeyDown);
    resizeObserver?.disconnect();
    resizeObserver = null;
    if (playbackTimer) {
      clearTimeout(playbackTimer);
      playbackTimer = null;
    }
    isFrameLoading = false;
    pendingFrameIndex = null;
  });

  const hasTrajectory = () => state().viewer.trajectoryPath !== null;

  // FEP scoring overlay

  // Scoring panel
  const [showScoringPanel, setShowScoringPanel] = createSignal(false);

  // Clustering modal (for trajectory analysis — separate from multi-PDB import)
  const [showClusteringModal, setShowClusteringModal] = createSignal(false);
  const handleOpenClustering = () => setShowClusteringModal(true);
  const handleCloseClustering = () => setShowClusteringModal(false);
  const handleViewCluster = async (centroidFrame: number) => {
    await loadTrajectoryFrame(centroidFrame);
  };

  // Multi-PDB alignment state
  const [isAligned, setIsAligned] = createSignal(false);

  // Apply protein + ligand styling to an arbitrary component (for aligned PDBs)
  const updateComponentStyle = (comp: NGL.Component) => {
    comp.removeAllRepresentations();

    const rep = state().viewer.proteinRep;
    const ligandSele = getLigandSelection();
    const proteinBaseSele = ligandSele
      ? `protein and not water and not ion and not (${ligandSele})`
      : 'protein and not water and not ion';

    // Protein representation
    if (rep === 'spacefill') {
      comp.addRepresentation('spacefill', { sele: proteinBaseSele, colorScheme: 'element' });
    } else {
      comp.addRepresentation(rep, { sele: proteinBaseSele, color: 0x909090 });
    }

    // Ligand representation
    if (ligandSele && state().viewer.ligandVisible) {
      comp.addRepresentation(LIGAND_REP_MAP[state().viewer.ligandRep] || 'ball+stick', {
        sele: ligandSele,
        colorScheme: 'element',
        multipleBond: 'symmetric',
      });
    }

    // Fallback for ligand-only: show hetero atoms
    if (!ligandSele) {
      const hetSele = 'hetero and not water and not ion';
      comp.addRepresentation('ball+stick', { sele: hetSele, colorScheme: 'element', multipleBond: 'symmetric' });
    }
  };

  const clearAlignment = () => {
    if (stage) {
      for (const [_idx, comp] of alignedComponents.entries()) {
        // Don't remove the current proteinComponent
        if (comp !== proteinComponent) {
          stage.removeComponent(comp);
        }
      }
    }
    alignedComponents.clear();
    setIsAligned(false);
  };

  // === Layer panel handlers ===

  // Prepare a raw PDB/CIF for viewing: add hydrogens via PDBFixer
  // Returns prepared path, or raw path if preparation fails
  const prepareStructure = async (rawPath: string): Promise<PreparedPath> => {
    const fileName = rawPath.split('/').pop() || '';
    const baseName = fileName.replace(/\.(pdb|cif)$/i, '');
    const dir = rawPath.substring(0, rawPath.lastIndexOf('/'));
    const preparedPath = `${dir}/${baseName}_prepared.pdb`;

    // Skip files that already have hydrogens or are simulation/docking outputs
    // MD centroid PDBs are full-system trajectory snapshots — hydrogens already present
    if (fileName.includes('_prepared') || fileName.includes('system') ||
        fileName === 'receptor.pdb' || fileName === 'final.pdb' ||
        fileName.startsWith('centroid_')) return rawPath as PreparedPath;
    const exists = await api.fileExists(preparedPath);
    if (exists) return preparedPath as PreparedPath;

    console.log(`[Viewer] Preparing structure: ${fileName}`);
    const result = await api.prepareForViewing(rawPath, preparedPath);
    return (result.ok ? result.value : rawPath) as PreparedPath;
  };

  /** Prepare a ligand SDF: sanitize, add hydrogens, fix bond orders via RDKit. */
  const prepareLigand = async (rawPath: string): Promise<string> => {
    const fileName = rawPath.split('/').pop() || '';
    const baseName = fileName.replace(/\.(sdf|sdf\.gz|mol|mol2)$/i, '');
    const dir = rawPath.substring(0, rawPath.lastIndexOf('/'));
    const preparedPath = `${dir}/${baseName}_prepared.sdf`;

    // Skip files that are already prepared or from docking/scoring output
    if (fileName.includes('_prepared') || fileName.includes('_docked') ||
        fileName === 'all_docked.sdf') return rawPath;
    const exists = await api.fileExists(preparedPath);
    if (exists) return preparedPath;

    console.log(`[Viewer] Preparing ligand: ${fileName}`);
    const result = await api.prepareLigandForViewing(rawPath, preparedPath);
    return result.ok ? result.value : rawPath;
  };

  /** Load a protein PDB into the NGL stage. Only accepts PreparedPath. */
  const stageLoadProtein = (
    path: PreparedPath,
    options: NglLoadOptions,
  ): Promise<NGL.Component | null> => {
    if (!stage) return Promise.resolve(null);
    return stage.loadFile(path, options).then((c) => (c as NGL.Component) || null);
  };

  const joinPath = (dirPath: string, fileName: string) =>
    dirPath.endsWith('/') ? `${dirPath}${fileName}` : `${dirPath}/${fileName}`;

  const parentDir = (filePath: string) => {
    const lastSlash = filePath.lastIndexOf('/');
    if (lastSlash <= 0) return null;
    return filePath.slice(0, lastSlash);
  };

  const pickTopologyCandidate = (pdbFiles: string[], trajectoryFileName: string): string | null => {
    if (pdbFiles.length === 0) return null;

    const legacySystemName = trajectoryFileName.replace(/_trajectory\.dcd$/i, '_system.pdb');
    const preferredNames = [
      'system.pdb',
      legacySystemName,
      'final.pdb',
    ];

    for (const preferred of preferredNames) {
      const match = pdbFiles.find((fileName) => fileName.toLowerCase() === preferred.toLowerCase());
      if (match) return match;
    }

    return pdbFiles.find((fileName) => /_system\.pdb$/i.test(fileName))
      || pdbFiles.find((fileName) => /system\.pdb$/i.test(fileName))
      || pdbFiles[0]
      || null;
  };

  const resolveTrajectoryTopology = async (trajectoryPath: string): Promise<string | null> => {
    const trajectoryDir = parentDir(trajectoryPath);
    if (!trajectoryDir) return null;

    const trajectoryFileName = trajectoryPath.split('/').pop() || '';
    const searchDirs = [trajectoryDir];
    const containerDir = parentDir(trajectoryDir);
    if (containerDir && containerDir !== trajectoryDir) {
      searchDirs.push(containerDir);
    }

    for (const dirPath of searchDirs) {
      const pdbFiles = await api.listPdbInDirectory(dirPath);
      const match = pickTopologyCandidate(pdbFiles, trajectoryFileName);
      if (match) {
        return joinPath(dirPath, match);
      }
    }

    return null;
  };

  const loadImportedStructures = async (selected: string[]) => {
    setIsLoading(true);
    setLoadingStatus('Importing...');
    setError(null);

    try {
      const imported = await Promise.all(selected.map(importToProject));

      const proteinInputs = imported.filter((filePath) => /\.(pdb|cif)$/i.test(filePath));
      const ligandInputs = imported.filter((filePath) => /\.(sdf|sdf\.gz|mol|mol2)$/i.test(filePath));
      if (proteinInputs.length > 0) {
        setLoadingStatus('Preparing protein (adding hydrogens)...');
      }
      const preparedProteins = await Promise.all(proteinInputs.map(prepareStructure));
      if (ligandInputs.length > 0) {
        setLoadingStatus('Preparing ligand (bond orders + hydrogens)...');
      }
      const preparedLigands = await Promise.all(ligandInputs.map(prepareLigand));
      const labelFor = (filePath: string) =>
        filePath.split('/').pop()?.replace('_prepared', '').replace(/(\.sdf\.gz|\.pdb|\.cif|\.sdf|\.mol2|\.mol)$/i, '') || filePath;

      for (const filePath of preparedProteins) {
        const id = nextLayerId();
        const label = labelFor(filePath);

        // Load into NGL (accumulate, no reset)
        if (stage) {
          const comp = await stageLoadProtein(filePath, { defaultRepresentation: false });
          if (comp) {
            layerComponents.set(id, comp);

            // If this is the first protein layer, use it as the main proteinComponent
            if (!proteinComponent) {
              proteinComponent = comp;
              updateProteinStyle();

              // Detect ligands
              api.detectPdbLigands(filePath).then((result) => {
                const ligands = result.ok ? (Array.isArray(result.value) ? result.value : result.value.ligands) : [];
                if (ligands.length > 0) {
                  setViewerDetectedLigands(ligands);
                  setViewerSelectedLigandId(ligands[0].id);
                  updateAllStyles();
                  if (proteinComponent) {
                    const ligandSele = getLigandSelection();
                    if (ligandSele) {
                      (proteinComponent as NGL.StructureComponent).autoView(ligandSele);
                    } else {
                      proteinComponent.autoView();
                    }
                  }
                  tryAutoLoadBindingSiteMap(filePath);
                } else if (proteinComponent) {
                  proteinComponent.autoView();
                }
              });
            } else {
              // Additional protein — apply styling
              updateComponentStyle(comp);
              comp.autoView();
            }
          }
        }

        addViewerLayer({
          id,
          type: 'protein',
          label,
          filePath,
          visible: true,
        });
      }

      const normalizedLigandPaths: string[] = [];
      for (const filePath of preparedLigands) {
        const id = nextLayerId();
        const label = labelFor(filePath);
        const normalizedLigandPath = await normalizeLigandPathForViewer(filePath);
        normalizedLigandPaths.push(normalizedLigandPath);

        if (stage) {
          const comp = await stage.loadFile(normalizedLigandPath, { defaultRepresentation: false, firstModelOnly: true }) as NGL.Component || null;
          if (comp) {
            layerComponents.set(id, comp);
            ligandComponent = comp;
            updateExternalLigandStyle(comp);
            setViewerLigandPath(normalizedLigandPath);
            if (proteinComponent) {
              updateProteinStyle();
            }
            comp.autoView();
          }
        }

        addViewerLayer({
          id,
          type: 'ligand',
          label,
          filePath: normalizedLigandPath,
          visible: true,
        });
      }

      if (preparedProteins.length > 0) {
        setViewerPdbPath(preparedProteins[preparedProteins.length - 1]);
      }

      const allProteinLayers = state().viewer.layers.filter((l) => l.type === 'protein');
      if (allProteinLayers.length > 1) {
        setViewerPdbQueue(allProteinLayers.map((l) => ({
          pdbPath: l.filePath,
          label: l.label,
        })));
      }

      // Build project table family for imported structures
      const allImported = [...preparedProteins, ...normalizedLigandPaths];
      if (allImported.length > 0) {
        const fileTypes = [
          ...preparedProteins.map(() => 'protein' as const),
          ...normalizedLigandPaths.map(() => 'ligand' as const),
        ];
        const { family, rows } = buildImportFamily({ filePaths: allImported, fileTypes });
        addViewerProjectFamily(family, rows);
      }

      return {
        lastPreparedProtein: preparedProteins[preparedProteins.length - 1] || null,
      };
    } catch (err) {
      setError(`Failed to import structure: ${(err as Error).message}`);
      return {
        lastPreparedProtein: null,
      };
    } finally {
      setIsLoading(false);
      setLoadingStatus('');
    }
  };

  const handleImportFiles = async () => {
    const selected = await api.selectStructureFilesMulti();
    if (!selected || selected.length === 0) return;

    const trajectoryFiles = selected.filter((filePath) => /\.dcd$/i.test(filePath));
    const structureFiles = selected.filter((filePath) => !/\.dcd$/i.test(filePath));

    if (trajectoryFiles.length > 1) {
      setError('Select one DCD trajectory at a time.');
      return;
    }

    const importResult = structureFiles.length > 0
      ? await loadImportedStructures(structureFiles)
      : { lastPreparedProtein: null };
    setShowImportOverlay(false);

    const dcdPath = trajectoryFiles[0];
    if (!dcdPath) return;

    const inferredTopologyPath = structureFiles.length === 0
      ? await resolveTrajectoryTopology(dcdPath)
      : null;
    const topologyPath = importResult.lastPreparedProtein || inferredTopologyPath || state().viewer.pdbPath;
    if (!topologyPath) {
      setError('Load a topology structure before opening a DCD trajectory.');
      return;
    }

    if (topologyPath !== state().viewer.pdbPath || !proteinComponent) {
      await handleLoadPdb(topologyPath, false);
    }
    await handleLoadTrajectory(dcdPath);
  };

  const handleFetchViewerPdb = async () => {
    const id = viewerPdbIdInput().trim();
    if (!id) return;
    const projectDir = state().projectDir;
    if (!projectDir) {
      setError('No project selected');
      return;
    }

    setIsFetchingViewerPdb(true);
    setError(null);

    try {
      const result = await api.fetchPdb(id, projectDir);
      if (!result.ok) {
        setError(result.error?.message || 'Failed to fetch PDB');
        return;
      }

      setViewerPdbIdInput('');
      await loadImportedStructures([result.value]);
      setShowImportOverlay(false);
    } catch (err) {
      setError(`PDB fetch error: ${(err as Error).message}`);
    } finally {
      setIsFetchingViewerPdb(false);
    }
  };

  const handleOpenRecentJob = async (jobId: string) => {
    const job = recentJobs().find((entry) => entry.id === jobId);
    if (!job || loadingRecentJobId()) return;

    setLoadingRecentJobId(jobId);
    try {
      await loadProjectJob(job, api);
      setShowImportOverlay(false);
    } finally {
      setLoadingRecentJobId(null);
    }
  };

  const handleProjectTableSort = (familyId: string, columnKey: string) => {
    const family = state().viewer.projectTable?.families.find((entry) => entry.id === familyId);
    if (!family) return;
    const nextDirection = family.sortKey === columnKey && family.sortDirection === 'asc' ? 'desc' : 'asc';
    setViewerProjectFamilySort(familyId, columnKey, nextDirection);
  };

  const handleProjectTableRowSelect = async (rowId: string) => {
    const projectTable = state().viewer.projectTable;
    const row = projectTable?.rows.find((entry) => entry.id === rowId);
    if (!row) return;

    setViewerProjectActiveRow(rowId);

    if (row.loadKind === 'queue' && row.queueIndex !== undefined && row.queueIndex >= 0) {
      const familyQueueRows = projectTable?.rows
        .filter((entry) => entry.familyId === row.familyId && entry.loadKind === 'queue' && entry.queueIndex !== undefined)
        .sort((a, b) => (a.queueIndex ?? 0) - (b.queueIndex ?? 0)) ?? [];

      if (familyQueueRows.length > 0) {
        const restoredQueue = familyQueueRows.map((entry) => entry.item);
        const queueMatches = state().viewer.pdbQueue.length === restoredQueue.length
          && state().viewer.pdbQueue.every((item, index) =>
            item.pdbPath === restoredQueue[index]?.pdbPath
            && item.ligandPath === restoredQueue[index]?.ligandPath
            && item.type === restoredQueue[index]?.type
          );
        if (!queueMatches) {
          setViewerPdbQueue(restoredQueue);
        }
      }

      setViewerTrajectoryPath(row.trajectoryPath ?? null);
      setViewerTrajectoryInfo(row.trajectoryPath ? state().viewer.trajectoryInfo : null);
      setViewerCurrentFrame(0);
      setViewerPdbQueueIndex(row.queueIndex);
      return;
    }

    setViewerPdbQueue([]);
    setViewerTrajectoryPath(row.trajectoryPath ?? null);
    setViewerTrajectoryInfo(null);
    setViewerCurrentFrame(0);

    if (row.loadKind === 'standalone-ligand') {
      setViewerDetectedLigands([]);
      setViewerSelectedLigandId(null);
      setViewerPdbPath(row.item.pdbPath);
      setViewerLigandPath(row.item.pdbPath);
      await handleLoadStandaloneLigand(row.item.pdbPath);
      return;
    }

    setViewerPdbPath(row.item.pdbPath);
    setViewerLigandPath(row.item.ligandPath ?? null);
    clearPocketReferenceLigand();
    await loadViewerQueueItem(row.item);

    if (!row.item.ligandPath && (row.pocketLigandPath || row.pocketSourcePdbPath)) {
      await loadPocketReferenceLigand(row);
    }
  };

  const handleProjectTablePlayTrajectory = async (familyId: string) => {
    const projectTable = state().viewer.projectTable;
    const family = projectTable?.families.find((entry) => entry.id === familyId);
    if (!family?.trajectoryPath) return;
    const initialRow = projectTable?.rows.find((row) => row.familyId === familyId && row.rowKind === 'initial-complex');
    if (!initialRow) return;

    setViewerProjectActiveRow(initialRow.id);
    setViewerPdbQueue([]);
    setViewerPdbPath(initialRow.item.pdbPath);
    setViewerLigandPath(initialRow.item.ligandPath ?? null);
    setViewerTrajectoryPath(family.trajectoryPath);
    setViewerTrajectoryInfo(null);
    setViewerCurrentFrame(0);

    if (!proteinComponent || state().viewer.pdbPath !== initialRow.item.pdbPath) {
      await loadViewerQueueItem(initialRow.item);
    }
  };

  const handleLoadSmiles = async () => {
    const smiles = smilesInput().trim();
    if (!smiles || isLoadingSmiles()) return;

    setIsLoadingSmiles(true);
    setError(null);

    try {
      const defaultDir = await api.getDefaultOutputDir();
      const baseOutputDir = state().customOutputDir || defaultDir;
      const outputDir = `${baseOutputDir}/${state().jobName}/structures`;

      const result = await api.convertSingleMolecule(smiles, outputDir, 'smiles');
      if (!result.ok) {
        setError(result.error?.message || 'Failed to convert SMILES');
        return;
      }

      const sdfPath = result.value.sdfPath;
      const id = nextLayerId();
      const label = result.value.name || (smiles.length > 20 ? `${smiles.substring(0, 20)}...` : smiles);

      if (stage) {
        const comp = await stage.loadFile(sdfPath, { defaultRepresentation: false });
        if (comp) {
          layerComponents.set(id, comp);
          ligandComponent = comp;
          updateExternalLigandStyle(comp);
          comp.autoView();
        }
      }

      addViewerLayer({ id, type: 'ligand', label, filePath: sdfPath, visible: true });
      setViewerLigandPath(sdfPath);

      // Add to project table
      const { family, rows } = buildImportFamily({ filePaths: [sdfPath], fileTypes: ['ligand'] });
      addViewerProjectFamily(family, rows);
      setShowImportOverlay(false);

      setSmilesInput('');
    } catch (err) {
      setError(`SMILES conversion failed: ${(err as Error).message}`);
    } finally {
      setIsLoadingSmiles(false);
    }
  };

  const handleLayerToggleVisibility = (layerId: string) => {
    const layer = state().viewer.layers.find((l) => l.id === layerId);
    if (!layer) return;
    const newVisible = !layer.visible;
    updateViewerLayer(layerId, { visible: newVisible });

    const comp = layerComponents.get(layerId);
    if (comp) {
      comp.setVisibility(newVisible);
    }

    // For ligand layers loaded via queue, toggle ligandComponent
    if (layer.type === 'ligand' && ligandComponent) {
      ligandComponent.setVisibility(newVisible);
    }
  };

  const syncViewerFromRemainingLayers = (remainingLayers: ViewerLayer[]) => {
    const remainingProteins = remainingLayers.filter((l) => l.type === 'protein');
    const remainingLigands = remainingLayers.filter((l) => l.type === 'ligand');
    const nextProtein = remainingProteins[0] || null;
    const nextLigand = remainingLigands[0] || null;
    const nextSelectedLayerId = nextProtein?.id || nextLigand?.id || null;

    if (!nextProtein && !nextLigand) {
      clearViewerSession();
      return;
    }

    setViewerLayerSelected(nextSelectedLayerId);

    if (remainingProteins.length > 1) {
      setViewerPdbQueue(remainingProteins.map((layer) => ({
        pdbPath: layer.filePath,
        label: layer.label,
      })));
    } else {
      setViewerPdbQueue([]);
    }

    if (nextProtein) {
      const nextProteinComp = layerComponents.get(nextProtein.id) || null;
      if (nextProteinComp) {
        proteinComponent = nextProteinComp;
      }
      setViewerPdbPath(nextProtein.filePath);
    } else {
      proteinComponent = null;
      setViewerPdbPath(null);
      setViewerTrajectoryPath(null);
      setViewerTrajectoryInfo(null);
      setViewerCurrentFrame(0);
      setViewerDetectedLigands([]);
      setViewerSelectedLigandId(null);
      clearBindingSiteVolumes();
    }

    if (nextLigand) {
      const nextLigandComp = layerComponents.get(nextLigand.id) || null;
      if (nextLigandComp) {
        ligandComponent = nextLigandComp;
      }
      setViewerLigandPath(nextLigand.filePath);
    } else {
      ligandComponent = null;
      setViewerLigandPath(null);
    }
  };

  const handleLayerRemove = (layerId: string) => {
    const layer = state().viewer.layers.find((l) => l.id === layerId);
    const remainingLayers = state().viewer.layers.filter((l) => l.id !== layerId);
    const comp = layerComponents.get(layerId);
    if (comp && stage) {
      stage.removeComponent(comp);
      layerComponents.delete(layerId);

      if (comp === proteinComponent) {
        proteinComponent = null;
      }
      if (comp === ligandComponent) {
        ligandComponent = null;
      }
    }

    removeViewerLayer(layerId);
    syncViewerFromRemainingLayers(remainingLayers);
  };

  const handleLayerSelect = (layerId: string) => {
    setViewerLayerSelected(layerId);
    const layer = state().viewer.layers.find((l) => l.id === layerId);
    if (!layer) return;

    const comp = layerComponents.get(layerId);
    if (comp) {
      // Update proteinComponent reference for style controls
      if (layer.type === 'protein') {
        proteinComponent = comp;
        setViewerPdbPath(layer.filePath);
      }
      comp.autoView();
    }

    // For ligand layers in a docking group, navigate to that pose
    if (layer.type === 'ligand' && layer.groupId && layer.poseIndex !== undefined) {
      const idx = layer.poseIndex;
      if (idx >= 0 && idx < state().viewer.pdbQueue.length) {
        setViewerPdbQueueIndex(idx);
      }
    }
  };

  const handleGroupRemove = (groupId: string) => {
    const remainingLayers = state().viewer.layers.filter((l) => l.groupId !== groupId);
    // Remove all NGL components for layers in this group
    const groupLayerList = state().viewer.layers.filter((l) => l.groupId === groupId);
    for (const layer of groupLayerList) {
      const comp = layerComponents.get(layer.id);
      if (comp && stage) {
        stage.removeComponent(comp);
        layerComponents.delete(layer.id);
        if (comp === proteinComponent) proteinComponent = null;
        if (comp === ligandComponent) ligandComponent = null;
      }
    }
    removeViewerLayerGroup(groupId);
    syncViewerFromRemainingLayers(remainingLayers);
  };

  const handleClearAll = () => {
    clearViewerSession();
  };

  const handleLayerAlignAll = async () => {
    // Gather all protein layers with NGL components
    const proteinLayers = state().viewer.layers.filter((l) => l.type === 'protein');
    if (proteinLayers.length < 2 || !stage || !proteinComponent) return;

    setIsLoading(true);
    setError(null);

    try {
      const refComp = proteinComponent;

      for (const layer of proteinLayers) {
        const comp = layerComponents.get(layer.id);
        if (!comp || comp === refComp) continue;

        try {
          (comp as NGL.StructureComponent).superpose(refComp as NGL.StructureComponent, true, 'backbone', 'backbone');
        } catch {
          try {
            (comp as NGL.StructureComponent).superpose(refComp as NGL.StructureComponent, false, '', '');
          } catch {
            console.warn(`Could not align ${layer.label}`);
          }
        }
      }

      setIsAligned(true);
      refComp.autoView();
    } catch (err) {
      setError(`Alignment failed: ${(err as Error).message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const proteinLayerCount = () => state().viewer.layers.filter((l) => l.type === 'protein').length;

  // === Binding site interaction maps ===

  const BS_CHANNELS = [
    { key: 'hydrophobic' as const, color: '#22c55e' },
    { key: 'hbondDonor' as const, color: '#3b82f6' },
    { key: 'hbondAcceptor' as const, color: '#ef4444' },
  ];

  const DEFAULT_BS_CHANNEL = { visible: true, isolevel: 0.3, opacity: 0.5 };

  const buildMapState = (data: BindingSiteResultsJson): BindingSiteMapState => ({
    hydrophobic: { ...DEFAULT_BS_CHANNEL },
    hbondDonor: { ...DEFAULT_BS_CHANNEL },
    hbondAcceptor: { ...DEFAULT_BS_CHANNEL },
    hydrophobicDx: data.hydrophobicDx,
    hbondDonorDx: data.hbondDonorDx,
    hbondAcceptorDx: data.hbondAcceptorDx,
    hotspots: data.hotspots || [],
  });

  const clearBindingSiteVolumes = () => {
    if (stage) {
      for (const comp of volumeComponents.values()) {
        stage.removeComponent(comp);
      }
    }
    volumeComponents.clear();
    setViewerBindingSiteMap(null);
  };

  const loadBindingSiteVolumes = async (mapState: BindingSiteMapState) => {
    if (!stage) return;
    const stageRef = stage;  // capture for async closure

    // Clear any existing volumes
    for (const comp of volumeComponents.values()) {
      stageRef.removeComponent(comp);
    }
    volumeComponents.clear();

    const dxPaths: Record<string, string> = {
      hydrophobic: mapState.hydrophobicDx,
      hbondDonor: mapState.hbondDonorDx,
      hbondAcceptor: mapState.hbondAcceptorDx,
    };

    // Load all 3 DX files in parallel
    const results = await Promise.allSettled(
      BS_CHANNELS.map(async (ch) => {
        const comp = await stageRef.loadFile(dxPaths[ch.key], { defaultRepresentation: false });
        return { key: ch.key, color: ch.color, comp };
      })
    );

    for (const r of results) {
      if (r.status === 'fulfilled' && r.value.comp) {
        const { key, color, comp } = r.value;
        const chState = mapState[key];
        comp.addRepresentation('surface', {
          isolevel: chState.isolevel,
          color,
          opacity: chState.opacity,
          wireframe: false,
        });
        comp.setVisibility(chState.visible);
        volumeComponents.set(key, comp);
      } else if (r.status === 'rejected') {
        console.error('[Viewer] Failed to load DX:', r.reason);
      }
    }
  };

  const tryAutoLoadBindingSiteMap = async (pdbPath: string) => {
    const pdbDir = pdbPath.replace(/\/[^/]+$/, '');
    const projectDir = pdbDir.replace(/\/(simulations|docking)\/[^/]+$/, '');
    const projName = projectDir !== pdbDir ? projectDir.split('/').pop() || '' : '';
    const candidateDirs = projectDir !== pdbDir
      ? [`${projectDir}/surfaces/pocket_map_static`, `${projectDir}/surfaces/binding_site_map`]
      : [`${pdbDir}/pocket_map_static`, `${pdbDir}/binding_site_map`];

    try {
      let data: BindingSiteResultsJson | null = null;
      for (const mapDir of candidateDirs) {
        const prefixedPath = projName ? `${mapDir}/${projName}_binding_site_results.json` : null;
        const legacyPath = `${mapDir}/binding_site_results.json`;
        data = (prefixedPath ? await api.readJsonFile(prefixedPath) as BindingSiteResultsJson | null : null)
          || await api.readJsonFile(legacyPath) as BindingSiteResultsJson | null;
        if (data?.hydrophobicDx) break;
      }
      if (!data || !data.hydrophobicDx) return;

      const mapState = buildMapState(data);
      setViewerBindingSiteMap(mapState);
      await loadBindingSiteVolumes(mapState);
    } catch (err) {
      console.error('[Viewer] Failed to auto-load binding site map:', err);
    }
  };

  const handleComputeBindingSiteMap = async () => {
    const pdbPath = state().viewer.pdbPath;
    const selectedId = state().viewer.selectedLigandId;
    if (!pdbPath || !selectedId) return;

    const ligand = state().viewer.detectedLigands.find((l) => l.id === selectedId);
    if (!ligand) return;

    setViewerIsComputingBindingSiteMap(true);

    try {
      // Determine PDB to use: if trajectory loaded, export current frame; otherwise use static PDB
      let targetPdb = pdbPath;
      const trajectoryPath = state().viewer.trajectoryPath;
      if (trajectoryPath) {
        const tmpPath = `/tmp/ember_expand_frame_${Date.now()}.pdb`;
        const exportResult = await api.exportTrajectoryFrame({
          topologyPath: pdbPath,
          trajectoryPath: trajectoryPath,
          frameIndex: state().viewer.currentFrame,
          outputPath: tmpPath,
          stripWaters: true,
        });
        if (exportResult.ok) {
          targetPdb = exportResult.value.pdbPath;
        }
      }

      // Output dir: surfaces/pocket_map_static in the project, or next to the PDB
      const pdbDir = pdbPath.replace(/\/[^/]+$/, '');
      const projectDir = pdbDir.replace(/\/(simulations|docking)\/[^/]+$/, '');
      const outputDir = projectDir !== pdbDir
        ? `${projectDir}/surfaces/pocket_map_static`
        : `${pdbDir}/pocket_map_static`;

      const result = await api.mapBindingSite({
        pdbPath: targetPdb,
        ligandResname: ligand.resname,
        ligandResnum: parseInt(ligand.resnum, 10),
        outputDir,
        sourcePdbPath: pdbPath,
        sourceTrajectoryPath: trajectoryPath || undefined,
      });

      if (result.ok) {
        const mapState = buildMapState(result.value);
        setViewerBindingSiteMap(mapState);
        await loadBindingSiteVolumes(mapState);
      } else {
        setError(`Binding site map failed: ${result.error?.message || 'Unknown error'}`);
      }
    } catch (err) {
      setError(`Binding site map error: ${(err as Error).message}`);
    } finally {
      setViewerIsComputingBindingSiteMap(false);
    }
  };

  // Reactive effect: update volume representations when channel settings change
  createEffect(() => {
    const bsMap = state().viewer.bindingSiteMap;
    if (!bsMap) return;

    for (const ch of BS_CHANNELS) {
      const comp = volumeComponents.get(ch.key);
      if (!comp) continue;

      const chState = bsMap[ch.key];
      comp.setVisibility(chState.visible);

      // Update representation parameters
      if (comp.reprList && comp.reprList.length > 0) {
        const repr = comp.reprList[0];
        repr.setParameters({
          isolevel: chState.isolevel,
          opacity: chState.opacity,
        });
      }
    }
  });

  createEffect(() => {
    const bsMap = state().viewer.bindingSiteMap;
    if (!bsMap || !stageReady() || volumeComponents.size > 0) return;
    void loadBindingSiteVolumes(bsMap);
  });

  // Get display info for detected ligand
  const getDetectedLigandInfo = () => {
    const ligands = state().viewer.detectedLigands;
    const selectedId = state().viewer.selectedLigandId;
    if (ligands.length === 0 || !selectedId) return null;

    const ligand = ligands.find((l) => l.id === selectedId);
    if (!ligand) return null;

    return `${ligand.resname} (Chain ${ligand.chain}, ${ligand.num_atoms} atoms)`;
  };

  const hasAutoDetectedLigand = () => state().viewer.detectedLigands.length > 0;
  const hasExternalLigand = () => state().viewer.ligandPath !== null;
  const hasAnyLigand = () => hasAutoDetectedLigand() || hasExternalLigand();
  const isLigandLikePath = (filePath: string | null) => !!filePath && /\.(sdf(\.gz)?|mol2?)$/i.test(filePath);
  const getStandaloneLigandPath = () => {
    const pdbPath = state().viewer.pdbPath;
    if (isLigandLikePath(pdbPath)) return pdbPath;
    const ligandPath = state().viewer.ligandPath;
    if (!pdbPath && isLigandLikePath(ligandPath)) return ligandPath;
    return null;
  };
  const hasLigandDisplayTarget = () => hasAnyLigand() || getStandaloneLigandPath() !== null || (!!ligandComponent && !proteinComponent);
  const hasProteinLigandContext = () => !!proteinComponent && hasAnyLigand();
  const canSimulateCurrentView = () =>
    !!state().viewer.pdbPath
    && !isLigandLikePath(state().viewer.pdbPath)
    && hasAnyLigand();
  const canExportCurrentView = () => !!state().viewer.pdbPath || !!state().viewer.ligandPath;

  // --- Multi-select alignment state ---
  const [hasAlignment, setHasAlignment] = createSignal(false);
  const [alignSubstructureLabel, setAlignSubstructureLabel] = createSignal<string | null>(null);

  const PROTEIN_ROW_KINDS = new Set(['apo', 'holo', 'initial-complex', 'cluster']);
  const LIGAND_ROW_KINDS = new Set(['ligand', 'prepared-ligand', 'pose', 'input', 'conformer']);

  const selectedRows = () => {
    const pt = state().viewer.projectTable;
    if (!pt) return [];
    const ids = new Set(pt.selectedRowIds || []);
    return pt.rows.filter((r) => ids.has(r.id));
  };

  const selectedProteinRows = () => selectedRows().filter((r) => PROTEIN_ROW_KINDS.has(r.rowKind));
  const selectedLigandRows = () => selectedRows().filter((r) => LIGAND_ROW_KINDS.has(r.rowKind));

  const canAlignProtein = () => selectedProteinRows().length >= 2;

  // L enabled when: 2+ ligands, OR 1+ ligand + 1 protein-complex with a detected bound ligand
  const hasProteinWithBoundLigand = () => {
    const proteins = selectedProteinRows();
    return proteins.length > 0 && (
      !!state().viewer.selectedLigandId ||
      !!state().viewer.ligandPath
    );
  };
  const canAlignLigand = () =>
    selectedLigandRows().length >= 2 ||
    (selectedLigandRows().length >= 1 && hasProteinWithBoundLigand());
  const canAlignSubstructure = () => canAlignLigand();

  const handleToggleRowSelection = async (rowId: string) => {
    const pt = state().viewer.projectTable;
    const row = pt?.rows.find((r) => r.id === rowId);
    // Trajectory rows cannot be multi-selected — fall back to normal click
    if (row?.trajectoryPath) {
      handleProjectTableRowSelect(rowId);
      return;
    }
    const wasSelected = (pt?.selectedRowIds || []).includes(rowId);
    toggleViewerProjectRowSelection(rowId);

    if (!stage || !row) return;

    if (wasSelected) {
      // Unload: remove NGL component
      const comp = layerComponents.get(rowId);
      if (comp) {
        stage.removeComponent(comp);
        layerComponents.delete(rowId);
      }
    } else {
      // Load: add NGL component for this row
      const filePath = row.item.ligandPath || row.item.pdbPath;
      if (!filePath) return;
      const isLigand = row.rowKind === 'ligand' || row.rowKind === 'prepared-ligand'
        || row.rowKind === 'pose' || row.rowKind === 'input' || row.rowKind === 'conformer'
        || row.item.type === 'ligand' || row.item.type === 'conformer';
      try {
        const comp = await stage.loadFile(filePath, { defaultRepresentation: false, firstModelOnly: true });
        if (comp) {
          layerComponents.set(rowId, comp);
          if (isLigand) {
            updateExternalLigandStyle(comp);
          } else {
            updateComponentStyle(comp);
          }
        }
      } catch { /* ignore load errors for toggle */ }
    }
  };

  const handleAlignProtein = () => {
    if (!stage) return;
    const proteins = selectedProteinRows();
    if (proteins.length < 2) return;
    const refRow = state().viewer.projectTable?.activeRowId
      ? proteins.find((r) => r.id === state().viewer.projectTable?.activeRowId) ?? proteins[0]
      : proteins[0];
    const refComp = layerComponents.get(refRow.id) as NGL.StructureComponent | undefined;
    if (!refComp) return;

    for (const row of proteins) {
      if (row.id === refRow.id) continue;
      const mobileComp = layerComponents.get(row.id) as NGL.StructureComponent | undefined;
      if (!mobileComp) continue;
      try {
        NGL.superpose(mobileComp.structure, refComp.structure, false, '.CA', '.CA');
        mobileComp.updateRepresentations({ position: true });
      } catch { /* skip if superpose fails */ }
    }
    stage.autoView();
    setHasAlignment(true);
  };

  const handleAlignLigand = async () => {
    const ligands = selectedLigandRows();
    if (ligands.length < 2) return;
    const pt = state().viewer.projectTable;
    const refRow = pt?.activeRowId
      ? ligands.find((r) => r.id === pt.activeRowId) ?? ligands[0]
      : ligands[0];
    const refPath = refRow.item.ligandPath || refRow.item.pdbPath;
    if (!refPath) return;

    for (const row of ligands) {
      if (row.id === refRow.id) continue;
      const mobilePath = row.item.ligandPath || row.item.pdbPath;
      if (!mobilePath) continue;
      const outPath = mobilePath.replace(/\.sdf(\.gz)?$/i, '_aligned.sdf');
      const result = await api.alignMoleculesMcs(refPath, mobilePath, outPath);
      if (result.ok) {
        // Reload the aligned SDF into the existing NGL component
        const comp = layerComponents.get(row.id);
        if (comp && stage) {
          stage.removeComponent(comp);
          const newComp = await stage.loadFile(outPath, { defaultRepresentation: false, firstModelOnly: true });
          if (newComp) {
            layerComponents.set(row.id, newComp);
            updateExternalLigandStyle(newComp);
          }
        }
      }
    }
    stage?.autoView();
    setHasAlignment(true);
  };

  // Scaffold cycle state
  const [detectedScaffolds, setDetectedScaffolds] = createSignal<Array<{ label: string }>>([]);
  const [scaffoldCycleIndex, setScaffoldCycleIndex] = createSignal(0);

  const handleAlignSubstructure = async () => {
    const ligands = selectedLigandRows();
    if (ligands.length < 2) return;
    const pt = state().viewer.projectTable;
    const refRow = pt?.activeRowId
      ? ligands.find((r) => r.id === pt.activeRowId) ?? ligands[0]
      : ligands[0];
    const refPath = refRow.item.ligandPath || refRow.item.pdbPath;
    if (!refPath) return;

    const mobileRow = ligands.find((r) => r.id !== refRow.id);
    if (!mobileRow) return;
    const mobilePath = mobileRow.item.ligandPath || mobileRow.item.pdbPath;
    if (!mobilePath) return;

    let scaffolds = detectedScaffolds();
    let nextIndex = scaffoldCycleIndex();

    // First call: detect scaffolds
    if (scaffolds.length === 0) {
      const result = await api.alignDetectScaffolds(refPath, mobilePath);
      if (!result.ok || result.value.scaffolds.length === 0) {
        console.log('[Viewer] No shared rigid substructures found');
        return;
      }
      scaffolds = result.value.scaffolds;
      setDetectedScaffolds(scaffolds);
      nextIndex = 0;
    } else {
      nextIndex = (scaffoldCycleIndex() + 1) % scaffolds.length;
    }

    // Align all mobile ligands by this scaffold
    for (const row of ligands) {
      if (row.id === refRow.id) continue;
      const mp = row.item.ligandPath || row.item.pdbPath;
      if (!mp) continue;
      const outPath = mp.replace(/\.sdf(\.gz)?$/i, `_ss${nextIndex}.sdf`);
      const result = await api.alignByScaffold(refPath, mp, nextIndex, outPath);
      if (result.ok) {
        const comp = layerComponents.get(row.id);
        if (comp && stage) {
          stage.removeComponent(comp);
          const newComp = await stage.loadFile(outPath, { defaultRepresentation: false, firstModelOnly: true });
          if (newComp) {
            layerComponents.set(row.id, newComp);
            updateExternalLigandStyle(newComp);
          }
        }
      }
    }

    setScaffoldCycleIndex(nextIndex);
    setAlignSubstructureLabel(`${scaffolds[nextIndex].label} (${nextIndex + 1}/${scaffolds.length})`);
    stage?.autoView();
    setHasAlignment(true);
  };

  const handleResetAlignment = () => {
    setHasAlignment(false);
    setAlignSubstructureLabel(null);
    setDetectedScaffolds([]);
    setScaffoldCycleIndex(0);
  };

  const handleSimulate = () => {
    const pdbPath = state().viewer.pdbPath;
    if (!pdbPath || !canSimulateCurrentView()) return;
    const ligandPath = state().viewer.ligandPath;
    const ligandName = ligandPath
      ? ligandPath.split('/').pop()?.replace(/\.sdf(\.gz)?$/, '') || 'ligand'
      : state().viewer.selectedLigandId || 'ligand';
    setMdReceptorPdb(pdbPath);
    setMdLigandSdf(ligandPath);
    setMdLigandName(ligandName);
    setMdPdbPath(pdbPath);
    batch(() => {
      setMode('md');
      setMdStep('md-configure');
    });
  };

  // Transfer: single-select only, auto-detect what to pre-populate
  const canTransferCurrentView = () => {
    const pt = state().viewer.projectTable;
    if (!pt) return false;
    return (pt.selectedRowIds || []).length === 1;
  };

  const getActiveRow = () => {
    const pt = state().viewer.projectTable;
    if (!pt?.activeRowId) return null;
    return pt.rows.find((r) => r.id === pt.activeRowId) ?? null;
  };

  const handleTransferDock = () => {
    const row = getActiveRow();
    if (!row) return;
    if (PROTEIN_ROW_KINDS.has(row.rowKind)) {
      // Protein → set as dock receptor
      setDockReceptorPdbPath(row.item.pdbPath);
    } else if (row.item.ligandPath) {
      // Ligand → set as dock ligand input
      setDockLigandSdfPaths([row.item.ligandPath]);
    }
    batch(() => {
      setMode('dock');
      setDockStep('dock-load');
    });
  };

  const handleTransferMcmm = () => {
    const row = getActiveRow();
    if (!row) return;
    const filePath = row.item.ligandPath || row.item.pdbPath;
    if (filePath) {
      workflowStore.setConformLigandSdf(filePath);
      const name = filePath.split('/').pop()?.replace(/\.(sdf|mol|pdb|cif)(\.gz)?$/i, '') || 'molecule';
      workflowStore.setConformLigandName(name);
    }
    batch(() => {
      setMode('conform');
      workflowStore.setConformStep('conform-configure');
    });
  };

  const handleTransferSimulate = () => {
    const row = getActiveRow();
    if (!row) return;
    const pdbPath = row.item.pdbPath;
    const ligandPath = row.item.ligandPath || null;
    const ligandName = ligandPath
      ? ligandPath.split('/').pop()?.replace(/\.sdf(\.gz)?$/, '') || 'ligand'
      : 'ligand';
    setMdReceptorPdb(pdbPath);
    setMdLigandSdf(ligandPath);
    setMdLigandName(ligandName);
    setMdPdbPath(pdbPath);
    batch(() => {
      setMode('md');
      setMdStep('md-configure');
    });
  };

  const [showImportOverlay, setShowImportOverlay] = createSignal(false);

  const handleProjectTableImport = () => {
    setShowImportOverlay(true);
  };

  const handleImportOverlayBack = () => {
    setShowImportOverlay(false);
  };

  const viewerLoadPanel = (fillHeight: boolean) => (
    <LayerPanel
      layers={state().viewer.layers}
      layerGroups={state().viewer.layerGroups}
      selectedLayerId={state().viewer.selectedLayerId}
      proteinCount={proteinLayerCount()}
      canClear={hasViewerSession()}
      fillHeight={fillHeight}
      recentJobs={recentJobs()}
      isLoadingRecentJobs={isLoadingRecentJobs()}
      loadingRecentJobId={loadingRecentJobId()}
      pdbIdInput={viewerPdbIdInput()}
      isLoadingImport={isLoading()}
      isLoadingPdbFetch={isFetchingViewerPdb()}
      smilesInput={smilesInput()}
      isLoadingSmiles={isLoadingSmiles()}
      onBrowseFiles={handleImportFiles}
      onAlignAll={handleLayerAlignAll}
      onClearAll={handleClearAll}
      onPdbIdInput={setViewerPdbIdInput}
      onFetchPdb={handleFetchViewerPdb}
      onSmilesInput={setSmilesInput}
      onLoadSmiles={handleLoadSmiles}
      onOpenRecentJob={handleOpenRecentJob}
      onToggleVisibility={handleLayerToggleVisibility}
      onRemoveLayer={handleLayerRemove}
      onSelectLayer={handleLayerSelect}
      onToggleGroupExpanded={toggleViewerLayerGroupExpanded}
      onToggleGroupVisible={(groupId) => {
        const group = state().viewer.layerGroups.find((g) => g.id === groupId);
        const newVisible = group ? !group.visible : true;
        toggleViewerLayerGroupVisible(groupId);
        const groupLayerItems = state().viewer.layers.filter((l) => l.groupId === groupId);
        for (const layer of groupLayerItems) {
          const comp = layerComponents.get(layer.id);
          if (comp) comp.setVisibility(newVisible);
        }
      }}
      onRemoveGroup={handleGroupRemove}
    />
  );

  return (
    <div class="h-full w-full min-w-0 flex flex-col gap-2">
      {/* Detected Ligand Info */}
      <Show when={hasAnyLigand()}>
        <div class="card bg-base-200 p-2">
          <div class="flex items-center gap-2">
            <Show when={hasAutoDetectedLigand()}>
              <div class="badge badge-success badge-sm gap-1" title="X-ray ligand detected in PDB">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-2 w-2" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
                </svg>
                Ligand
              </div>
            </Show>
            <span class="text-xs text-base-content/90 flex-1 truncate">
              <Show when={hasAutoDetectedLigand() && !hasExternalLigand()}>
                {getDetectedLigandInfo()}
              </Show>
              <Show when={hasExternalLigand()}>
                {state().viewer.ligandPath?.split('/').pop()}
              </Show>
            </span>
            <Show when={hasViewerSession()}>
              <button
                class="btn btn-sm btn-error btn-outline ml-auto"
                onClick={handleClearAll}
                title="Close structure"
                data-testid="viewer-close-button"
              >
                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </Show>
          </div>
        </div>
      </Show>

      {/* Trajectory Controls */}
      <Show when={hasTrajectory()}>
        <div class="flex flex-col gap-2">
          <TrajectoryControls
            onSeek={handleTrajectorySeek}
            onPlay={handleTrajectoryPlay}
            onPause={handleTrajectoryPause}
            onStepForward={handleTrajectoryStepForward}
            onStepBackward={handleTrajectoryStepBackward}
            onFirstFrame={handleTrajectoryFirstFrame}
            onLastFrame={handleTrajectoryLastFrame}
          />
          <AnalysisPanel onOpenClustering={handleOpenClustering} />
        </div>
      </Show>

      {/* (Cluster overlay list removed) */}

      {/* Error display */}
      <Show when={error()}>
        <div class="alert alert-error text-xs py-1">
          {error()}
        </div>
      </Show>

      {/* NGL Viewer Canvas + Project Table */}
      <div class="flex-1 min-h-0 min-w-0 w-full flex gap-2">
        <div
          ref={viewerCanvasShellRef}
          class="flex-1 basis-0 min-w-0 w-full relative rounded-lg overflow-hidden border border-base-300 min-h-0"
        >
          <div
            ref={containerRef}
            class="absolute inset-0"
            style={{ width: '100%', height: '100%' }}
          />
          <Show when={isLoading()}>
            <div class="absolute inset-0 bg-base-300/50 flex flex-col items-center justify-center gap-2">
              <span class="loading loading-spinner loading-lg text-primary" />
              <Show when={loadingStatus()}>
                <span class="text-sm text-base-content/70">{loadingStatus()}</span>
              </Show>
            </div>
          </Show>
          <Show when={!hasViewerSession() && !isLoading()}>
            <div class="absolute inset-0 flex items-center justify-center overflow-auto">
              <div class="w-full max-w-md">
                {viewerLoadPanel(false)}
              </div>
            </div>
          </Show>
          <Show when={showImportOverlay() && hasViewerSession()}>
            <div class="absolute inset-0 z-10 bg-base-100/95 flex flex-col overflow-auto">
              <div class="px-3 py-2">
                <button
                  class="btn btn-ghost btn-sm gap-1"
                  onClick={handleImportOverlayBack}
                >
                  <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                  </svg>
                  Back to viewer
                </button>
              </div>
              <div class="flex-1 flex items-center justify-center">
                <div class="w-full max-w-md">
                  {viewerLoadPanel(false)}
                </div>
              </div>
            </div>
          </Show>
        </div>

        <Show when={hasViewerSession() && hasProjectTable() && state().viewer.projectTable}>
          <div
            ref={projectTablePanelRef}
            class="relative shrink-0 min-h-0 h-full"
            style={{ width: `${projectTableWidth()}px` }}
          >
            <div
              class="absolute -left-1 top-0 bottom-0 w-2 cursor-col-resize z-20"
              onMouseDown={handleProjectTableResizeStart}
              data-testid="project-table-resize-handle"
            >
              <div class="pointer-events-none absolute left-1/2 top-0 bottom-0 -translate-x-1/2 w-px bg-base-content/15" />
            </div>
            <ProjectTable
              projectTable={state().viewer.projectTable!}
              panelWidth={projectTableWidth()}
              onSelectRow={handleProjectTableRowSelect}
              onToggleRowSelection={handleToggleRowSelection}
              onToggleFamilyCollapsed={toggleViewerProjectFamilyCollapsed}
              onSortFamily={handleProjectTableSort}
              onPlayTrajectory={handleProjectTablePlayTrajectory}
              onRemoveFamily={removeViewerProjectFamily}
              onRemoveRow={removeViewerProjectRow}
              onRenameRow={renameViewerProjectRow}
              canNavigatePrevious={activeProjectRowIndex() > 0}
              canNavigateNext={activeProjectRowIndex() >= 0 && activeProjectRowIndex() < visibleProjectRows().length - 1}
              onNavigatePrevious={() => void navigateProjectTable(-1)}
              onNavigateNext={() => void navigateProjectTable(1)}
              canTransfer={canTransferCurrentView()}
              canExport={canExportCurrentView()}
              onTransferDock={handleTransferDock}
              onTransferMcmm={handleTransferMcmm}
              onTransferSimulate={handleTransferSimulate}
              onExport={() => void handleExportPdb()}
              onImport={handleProjectTableImport}
              canAlignProtein={canAlignProtein()}
              canAlignLigand={canAlignLigand()}
              canAlignSubstructure={canAlignSubstructure()}
              onAlignProtein={handleAlignProtein}
              onAlignLigand={handleAlignLigand}
              onAlignSubstructure={handleAlignSubstructure}
              alignSubstructureLabel={alignSubstructureLabel()}
              hasAlignment={hasAlignment()}
              onResetAlignment={handleResetAlignment}
            />
          </div>
        </Show>
      </div>

      {/* Style controls - compact layout */}
      <div class="card bg-base-200 p-2">
        <div class="flex flex-col gap-1">
          {/* Protein row */}
          <div class="flex items-center gap-2 flex-wrap">
            <span class="text-xs font-semibold w-14">Protein</span>
            <select
              class="select select-xs select-bordered w-24"
              value={state().viewer.proteinRep}
              onChange={(e) => handleProteinRepChange(e.target.value as ProteinRepresentation)}
              disabled={!state().viewer.pdbPath}
            >
              <option value="cartoon">Cartoon</option>
              <option value="ribbon">Ribbon</option>
              <option value="spacefill">Spacefill</option>
            </select>
            <input
              type="color"
              class="w-6 h-6 cursor-pointer rounded border border-base-300"
              value={state().viewer.proteinCarbonColor}
              onChange={(e) => handleProteinCarbonColorChange(e.target.value)}
              disabled={!state().viewer.pdbPath}
              title="Carbon color"
            />
            <label class="label cursor-pointer gap-1 p-0">
              <input
                type="checkbox"
                class="checkbox checkbox-xs checkbox-primary"
                checked={state().viewer.proteinSurface}
                onChange={handleProteinSurfaceToggle}
                disabled={!state().viewer.pdbPath}
              />
              <span class="text-xs">Surface</span>
            </label>
            <Show when={state().viewer.proteinSurface}>
              <select
                class="select select-xs select-bordered w-32"
                value={state().viewer.surfaceColorScheme}
                onChange={(e) => handleSurfaceColorChange(e.target.value as SurfaceColorScheme)}
                disabled={!state().viewer.pdbPath}
              >
                <option value="uniform-grey">Solid</option>
                <option value="hydrophobic">Hydrophobic</option>
                <option value="electrostatic">Electrostatic</option>
              </select>
              <Show when={surfacePropsLoading()}>
                <span class="loading loading-spinner loading-xs text-primary" title="Computing surface properties..." />
              </Show>
              <select
                class="select select-xs select-bordered w-16"
                value={state().viewer.proteinSurfaceOpacity}
                onChange={(e) => handleProteinSurfaceOpacityChange(parseFloat(e.target.value))}
                disabled={!state().viewer.pdbPath}
                title="Surface opacity"
              >
                <option value="0.1">10%</option>
                <option value="0.2">20%</option>
                <option value="0.3">30%</option>
                <option value="0.4">40%</option>
                <option value="0.5">50%</option>
                <option value="0.6">60%</option>
                <option value="0.7">70%</option>
                <option value="0.8">80%</option>
                <option value="0.9">90%</option>
                <option value="1.0">100%</option>
              </select>
            </Show>
            <label class="label cursor-pointer gap-1 p-0" title="Show pocket residues within 5Å">
              <input
                type="checkbox"
                class="checkbox checkbox-xs checkbox-primary"
                checked={state().viewer.showPocketResidues}
                onChange={handlePocketResiduesToggle}
                disabled={!hasProteinLigandContext()}
              />
              <span class="text-xs">Show Pocket</span>
            </label>
            <label class="label cursor-pointer gap-1 p-0" title="Show amino acid labels on pocket residues (e.g., F2108)">
              <input
                type="checkbox"
                class="checkbox checkbox-xs checkbox-primary"
                checked={state().viewer.showPocketLabels}
                onChange={handlePocketLabelsToggle}
                disabled={!hasProteinLigandContext() || !state().viewer.showPocketResidues}
              />
              <span class="text-xs">Labels</span>
            </label>
            <label class="label cursor-pointer gap-1 p-0" title="Hide water molecules and ions">
              <input
                type="checkbox"
                class="checkbox checkbox-xs checkbox-primary"
                checked={state().viewer.hideWaterIons}
                onChange={handleHideWaterIonsToggle}
                disabled={!state().viewer.pdbPath}
              />
              <span class="text-xs">Hide H₂O/Ions</span>
            </label>
          </div>

          {/* Ligand row */}
          <div class="flex items-center gap-2 flex-wrap">
            <span class="text-xs font-semibold w-14">Ligand</span>
            <label class="label cursor-pointer gap-1 p-0" title="Hide/show ligand">
              <input
                type="checkbox"
                class="checkbox checkbox-xs checkbox-secondary"
                checked={state().viewer.ligandVisible}
                onChange={handleLigandVisibleToggle}
                disabled={!hasLigandDisplayTarget()}
              />
              <span class="text-xs">Show</span>
            </label>
            <select
              class="select select-xs select-bordered w-24"
              value={state().viewer.ligandRep}
              onChange={(e) => handleLigandRepChange(e.target.value as LigandRepresentation)}
              disabled={!hasLigandDisplayTarget()}
            >
              <option value="ball+stick">Ball+Stick</option>
              <option value="stick">Stick</option>
              <option value="spacefill">Spacefill</option>
            </select>
            <input
              type="color"
              class="w-6 h-6 cursor-pointer rounded border border-base-300"
              value={state().viewer.ligandCarbonColor}
              onChange={(e) => handleLigandCarbonColorChange(e.target.value)}
              disabled={!hasLigandDisplayTarget()}
              title="Carbon color"
            />
            <label class="label cursor-pointer gap-1 p-0">
              <input
                type="checkbox"
                class="checkbox checkbox-xs checkbox-secondary"
                checked={state().viewer.ligandSurface}
                onChange={handleLigandSurfaceToggle}
                disabled={!hasLigandDisplayTarget()}
              />
              <span class="text-xs">Surface</span>
            </label>
            <Show when={state().viewer.ligandSurface}>
              <select
                class="select select-xs select-bordered w-32"
                value={state().viewer.surfaceColorScheme}
                onChange={(e) => handleSurfaceColorChange(e.target.value as SurfaceColorScheme)}
                disabled={!hasLigandDisplayTarget()}
              >
                <option value="uniform-grey">Solid</option>
                <option value="hydrophobic">Hydrophobic</option>
                <option value="electrostatic">Electrostatic</option>
              </select>
              <Show when={surfacePropsLoading()}>
                <span class="loading loading-spinner loading-xs text-secondary" title="Computing surface properties..." />
              </Show>
              <select
                class="select select-xs select-bordered w-16"
                value={state().viewer.ligandSurfaceOpacity}
                onChange={(e) => handleLigandSurfaceOpacityChange(parseFloat(e.target.value))}
                disabled={!hasLigandDisplayTarget()}
                title="Surface opacity"
              >
                <option value="0.1">10%</option>
                <option value="0.2">20%</option>
                <option value="0.3">30%</option>
                <option value="0.4">40%</option>
                <option value="0.5">50%</option>
                <option value="0.6">60%</option>
                <option value="0.7">70%</option>
                <option value="0.8">80%</option>
                <option value="0.9">90%</option>
                <option value="1.0">100%</option>
              </select>
            </Show>
            <label class="label cursor-pointer gap-1 p-0" title="Show only polar hydrogens (H bonded to N/O/S)">
              <input
                type="checkbox"
                class="checkbox checkbox-xs checkbox-secondary"
                checked={state().viewer.ligandPolarHOnly}
                onChange={handleLigandPolarHToggle}
                disabled={!hasLigandDisplayTarget()}
              />
              <span class="text-xs">Polar H</span>
            </label>
            <label class="label cursor-pointer gap-1 p-0" title="Show H-bonds, hydrophobic, ionic contacts">
              <input
                type="checkbox"
                class="checkbox checkbox-xs checkbox-accent"
                checked={state().viewer.showInteractions}
                onChange={handleInteractionsToggle}
                disabled={!hasProteinLigandContext()}
              />
              <span class="text-xs">Interactions</span>
            </label>
            <div class="flex-1" />
            {/* Clipping distance */}
            <div class="flex items-center gap-1" title="Clipping distance (how deep into the structure to see)">
              <span class="text-xs text-base-content/60">Clip</span>
              <input
                type="range"
                min="0"
                max="100"
                step="1"
                value="10"
                class="range range-xs w-16"
                onInput={(e) => {
                  if (stage) {
                    const val = parseInt(e.currentTarget.value);
                    stage.setParameters({ clipDist: val });
                  }
                }}
              />
            </div>
            <button
              class="btn btn-xs btn-ghost"
              onClick={handleResetView}
              disabled={!state().viewer.pdbPath}
            >
              Reset
            </button>
          </div>
        </div>
      </div>

      {/* Clustering Modal (trajectory analysis) */}
      <ClusteringModal
        isOpen={showClusteringModal()}
        onClose={handleCloseClustering}
        onViewCluster={handleViewCluster}
      />
    </div>
  );
};

export default ViewerMode;
