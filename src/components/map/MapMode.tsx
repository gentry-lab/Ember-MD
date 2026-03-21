import { Component, For, Show, createSignal, onCleanup } from 'solid-js';
import type { MapMethod } from '../../stores/workflow';
import { workflowStore } from '../../stores/workflow';

const METHOD_LABEL = 'Water Map (GIST)';

const buildDefaultChannels = () => ({
  hydrophobic: { visible: true, isolevel: 0.3, opacity: 0.7 },
  hbondDonor: { visible: true, isolevel: 0.3, opacity: 0.7 },
  hbondAcceptor: { visible: true, isolevel: 0.3, opacity: 0.7 },
});

const MapMode: Component = () => {
  const {
    state,
    openViewerSession,
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
  } = workflowStore;

  let activeListenerCleanup: (() => void) | null = null;

  onCleanup(() => {
    activeListenerCleanup?.();
    activeListenerCleanup = null;
  });

  const api = window.electronAPI;
  const map = () => state().map;
  const result = () => map().result;
  const hasTrajectory = () => !!state().viewer.trajectoryPath;
  const hasPdb = () => !!map().pdbPath;
  const hasLigand = () => !!map().selectedLigandId;

  const pdbName = () => {
    const pdbPath = map().pdbPath;
    if (!pdbPath) return null;
    return pdbPath.split('/').pop() || pdbPath;
  };

  const selectedLigand = () =>
    map().detectedLigands.find((ligand) => ligand.id === map().selectedLigandId) ?? null;

  const resetRunState = (nextStep?: 'map-load' | 'map-configure') => {
    setMapIsComputing(false);
    setMapProgress('', 0);
    setMapError(null);
    setMapResult(null);
    setMapStep(nextStep ?? (map().pdbPath ? 'map-configure' : 'map-load'));
  };

  const handleSelectPdb = async () => {
    const pdbPath = await api.selectPdbFile();
    if (!pdbPath) return;

    setMapPdbPath(pdbPath);
    setMapDetectedLigands([]);
    setMapSelectedLigandId(null);
    setMapIsDetecting(true);
    setMapError(null);
    setMapResult(null);
    setMapProgress('', 0);
    setMapStep('map-configure');

    try {
      const detected = await api.detectPdbLigands(pdbPath);
      if (detected.ok && detected.value.length > 0) {
        setMapDetectedLigands(detected.value);
        setMapSelectedLigandId(detected.value[0].id);
      }
    } catch (err) {
      console.error('[Map] Failed to detect ligands:', err);
    } finally {
      setMapIsDetecting(false);
    }
  };

  const canCompute = () => hasPdb() && hasLigand() && !map().isComputing;

  const getOutputDir = (method: MapMethod): string => {
    const pdbPath = map().pdbPath;
    if (!pdbPath) return '';

    const emberIdx = pdbPath.indexOf('/Ember/');
    if (emberIdx >= 0) {
      const afterEmber = pdbPath.substring(emberIdx + 7);
      const projectName = afterEmber.split('/')[0];
      const projectRoot = pdbPath.substring(0, emberIdx + 7) + projectName;
      return `${projectRoot}/surfaces/pocket_map_${method}`;
    }

    const pdbDir = pdbPath.replace(/\/[^/]+$/, '');
    return `${pdbDir}/pocket_map_${method}`;
  };

  const estimateMdTime = (): number => {
    const info = state().md.systemInfo;
    if (info?.atomCount) {
      return Math.max(5, Math.round((info.atomCount / 20000) * 30));
    }
    return 30;
  };

  const loadMapResult = (
    payload: {
      hydrophobicDx: string;
      hbondDonorDx: string;
      hbondAcceptorDx: string;
      hotspots?: Array<{ type: string; position: number[]; direction: number[]; score: number }>;
    },
    method: MapMethod,
    outputDir: string,
    pdbPath: string,
    trajectoryPath: string | null,
  ) => {
    setMapResult({
      ...buildDefaultChannels(),
      hydrophobicDx: payload.hydrophobicDx,
      hbondDonorDx: payload.hbondDonorDx,
      hbondAcceptorDx: payload.hbondAcceptorDx,
      hotspots: payload.hotspots || [],
      method,
      pdbPath,
      outputDir,
      trajectoryPath,
    });
    setMapProgress('Done', 100);
    setMapStep('map-results');
  };

  const runMdThenComputation = async (method: MapMethod) => {
    const pdbPath = map().pdbPath;
    const ligand = selectedLigand();
    if (!ligand || !pdbPath) return;

    setMapIsComputing(true);
    setMapStep('map-progress');
    setMapProgress('Running short MD simulation for water analysis...', 5);

    activeListenerCleanup?.();
    const cleanup = api.onMdOutput((data) => {
      if (!data.data.includes('PROGRESS:')) return;
      const msg = data.data.replace(/.*PROGRESS:\s*/, '').trim();
      const pctMatch = msg.match(/(\d+)%/);
      if (pctMatch) {
        setMapProgress(msg, Math.min(50, parseInt(pctMatch[1], 10) / 2));
      } else {
        setMapProgress(msg);
      }
    });
    activeListenerCleanup = cleanup;

    try {
      const outputDir = getOutputDir(method);
      const mdOutputDir = `${outputDir}/md_temp`;

      const mdResult = await api.runMdSimulation(
        pdbPath,
        ligand.id,
        mdOutputDir,
        { productionNs: 2, forceFieldPreset: 'ff14sb-tip3p' } as any,
      );

      if (!mdResult.ok) {
        setMapError(`MD simulation failed: ${mdResult.error.message}`);
        setMapStep('map-configure');
        return;
      }

      setMapProgress('Running GIST analysis on trajectory...', 55);
      const trajectoryPath = `${mdOutputDir}/trajectory.dcd`;
      const systemPdb = `${mdOutputDir}/system.pdb`;

      const computed = await api.computePocketMap({
        method: 'solvation',
        pdbPath: systemPdb,
        ligandResname: ligand.resname,
        ligandResnum: parseInt(ligand.resnum as string, 10),
        outputDir,
        trajectoryPath,
        sourcePdbPath: systemPdb,
        sourceTrajectoryPath: trajectoryPath,
      });

      if (computed.ok) {
        loadMapResult(computed.value, method, outputDir, systemPdb, trajectoryPath);
      } else {
        setMapError(computed.error.message);
        setMapStep('map-configure');
      }
    } catch (err) {
      setMapError((err as Error).message);
      setMapStep('map-configure');
    } finally {
      cleanup();
      activeListenerCleanup = null;
      setMapIsComputing(false);
    }
  };

  const runComputation = async (method: MapMethod) => {
    const pdbPath = map().pdbPath;
    const ligand = selectedLigand();
    if (!ligand || !pdbPath) {
      setMapError('No PDB or ligand selected');
      return;
    }

    setMapIsComputing(true);
    setMapStep('map-progress');
    setMapProgress('Initializing...', 0);

    activeListenerCleanup?.();
    const cleanup = api.onMdOutput((data) => {
      if (!data.data.includes('PROGRESS:')) return;
      const msg = data.data.replace(/.*PROGRESS:\s*/, '').trim();
      const pctMatch = msg.match(/(\d+)%/);
      if (pctMatch) {
        setMapProgress(msg, parseInt(pctMatch[1], 10));
      } else {
        setMapProgress(msg);
      }
    });
    activeListenerCleanup = cleanup;

    try {
      const outputDir = getOutputDir(method);
      const trajectoryPath = method === 'solvation' ? state().viewer.trajectoryPath : null;

      const computed = await api.computePocketMap({
        method,
        pdbPath,
        ligandResname: ligand.resname,
        ligandResnum: parseInt(ligand.resnum as string, 10),
        outputDir,
        trajectoryPath: trajectoryPath || undefined,
        sourcePdbPath: pdbPath,
        sourceTrajectoryPath: trajectoryPath || undefined,
      });

      if (computed.ok) {
        loadMapResult(computed.value, method, outputDir, pdbPath, trajectoryPath);
      } else {
        setMapError(computed.error.message);
        setMapStep('map-configure');
      }
    } catch (err) {
      setMapError((err as Error).message);
      setMapStep('map-configure');
    } finally {
      cleanup();
      activeListenerCleanup = null;
      setMapIsComputing(false);
    }
  };

  const handleCompute = () => {
    setMapMethod('solvation');
    setMapError(null);

    if (!hasTrajectory()) {
      setMapShowMdConfirm(true, estimateMdTime());
      return;
    }

    runComputation('solvation');
  };

  const handleConfirmMd = () => {
    setMapShowMdConfirm(false);
    runMdThenComputation('solvation');
  };

  const handleOpenResults = () => {
    const mapResult = result();
    if (!mapResult?.pdbPath) return;

    openViewerSession({
      pdbPath: mapResult.pdbPath,
      trajectoryPath: mapResult.trajectoryPath,
      bindingSiteMap: {
        hydrophobic: { ...mapResult.hydrophobic },
        hbondDonor: { ...mapResult.hbondDonor },
        hbondAcceptor: { ...mapResult.hbondAcceptor },
        hydrophobicDx: mapResult.hydrophobicDx,
        hbondDonorDx: mapResult.hbondDonorDx,
        hbondAcceptorDx: mapResult.hbondAcceptorDx,
        hotspots: mapResult.hotspots,
        method: mapResult.method,
      },
    });
  };

  const handleOpenFolder = () => {
    const outputDir = result()?.outputDir;
    if (outputDir) {
      api.openFolder(outputDir);
    }
  };

  const renderResults = () => {
    const mapResult = result();
    if (!mapResult) return null;

    return (
      <div class="h-full flex flex-col">
        <div class="text-center mb-3">
          <h2 class="text-xl font-bold">Map Results</h2>
          <p class="text-sm text-base-content/70">
            {METHOD_LABEL} {mapResult.hotspots.length > 0 ? `• ${mapResult.hotspots.length} hotspots` : ''}
          </p>
        </div>

        <div class="flex-1 min-h-0 overflow-auto">
          <div class="card bg-base-200 shadow-lg">
            <div class="card-body p-4">
              <div class="flex items-center justify-between mb-2">
                <h3 class="text-sm font-semibold">Computed Pocket Map</h3>
                <span class="badge badge-primary badge-sm">{METHOD_LABEL}</span>
              </div>

              <div class="mb-3 rounded-lg bg-base-300 px-3 py-2">
                <p class="text-[10px] uppercase tracking-wider text-base-content/60">Saved Run</p>
                <p class="text-xs font-mono break-all">{mapResult.outputDir}</p>
              </div>

              <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                <div class="rounded border border-base-300 bg-base-100/60 p-3">
                  <p class="text-[10px] uppercase tracking-wider text-base-content/60 mb-1">Structure</p>
                  <p class="font-mono break-all">{mapResult.pdbPath}</p>
                </div>
                <div class="rounded border border-base-300 bg-base-100/60 p-3">
                  <p class="text-[10px] uppercase tracking-wider text-base-content/60 mb-1">Trajectory</p>
                  <p class="font-mono break-all">{mapResult.trajectoryPath || 'None'}</p>
                </div>
              </div>

              <div class="mt-4 flex items-center gap-4 py-1">
                <div class="flex items-center gap-1">
                  <span class="w-2 h-2 rounded-full inline-block" style={{ background: '#22c55e' }} />
                  <span class="text-[10px] text-base-content/60">Hydrophobic</span>
                </div>
                <div class="flex items-center gap-1">
                  <span class="w-2 h-2 rounded-full inline-block" style={{ background: '#3b82f6' }} />
                  <span class="text-[10px] text-base-content/60">H-Donor</span>
                </div>
                <div class="flex items-center gap-1">
                  <span class="w-2 h-2 rounded-full inline-block" style={{ background: '#ef4444' }} />
                  <span class="text-[10px] text-base-content/60">H-Acceptor</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="flex justify-between mt-3">
          <button class="btn btn-ghost btn-sm" onClick={() => resetRunState()}>
            <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 17l-5-5m0 0l5-5m-5 5h12" />
            </svg>
            New Map
          </button>
          <div class="flex gap-2">
            <button class="btn btn-ghost btn-sm" onClick={handleOpenFolder}>
              Open Folder
            </button>
            <button class="btn btn-primary btn-sm" onClick={handleOpenResults} disabled={!mapResult.pdbPath}>
              View Results
              <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    );
  };

  if (map().step === 'map-results' && result()) {
    return renderResults();
  }

  return (
    <div class="h-full flex flex-col gap-4">
      <div class="flex items-center gap-3 flex-wrap">
        <Show
          when={hasPdb()}
          fallback={
            <button class="btn btn-primary btn-sm gap-2" onClick={handleSelectPdb}>
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
              Load Structure
            </button>
          }
        >
          <button
            class="btn btn-ghost btn-sm gap-2 font-mono"
            onClick={handleSelectPdb}
            title="Load a different structure"
          >
            <svg class="w-4 h-4 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            {pdbName()}
          </button>

          <Show when={map().isDetecting} fallback={
            <Show when={map().detectedLigands.length > 0}>
              <div class="flex items-center gap-2">
                <span class="text-xs text-base-content/50 font-semibold">Ligand</span>
                <Show
                  when={map().detectedLigands.length === 1}
                  fallback={
                    <select
                      class="select select-bordered select-sm font-mono text-xs"
                      value={map().selectedLigandId ?? ''}
                      onChange={(e) => setMapSelectedLigandId(e.currentTarget.value)}
                    >
                      <For each={map().detectedLigands}>
                        {(ligand) => (
                          <option value={ligand.id}>
                            {ligand.resname} {ligand.resnum} ({ligand.num_atoms} atoms)
                          </option>
                        )}
                      </For>
                    </select>
                  }
                >
                  <span class="badge badge-ghost font-mono">
                    {selectedLigand()?.resname} {selectedLigand()?.resnum}
                  </span>
                </Show>
              </div>
            </Show>
          }>
            <span class="flex items-center gap-1.5 text-xs text-base-content/50">
              <span class="loading loading-spinner loading-xs" />
              Detecting ligands...
            </span>
          </Show>

          <Show when={!map().isDetecting && map().detectedLigands.length === 0 && hasPdb()}>
            <span class="text-xs text-warning">No ligands detected in this structure</span>
          </Show>

          <Show when={hasTrajectory()}>
            <span class="badge badge-sm badge-success badge-outline gap-1 ml-auto">
              <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
              </svg>
              MD trajectory available
            </span>
          </Show>
        </Show>
      </div>

      {/* Water Map description + compute button */}
      <div class="card bg-base-200 shadow-lg">
        <div class="card-body p-4">
          <div class="flex items-center justify-between mb-2">
            <div>
              <h3 class="text-sm font-bold">Water Thermodynamics</h3>
              <p class="text-[10px] text-primary font-medium">GIST (Grid Inhomogeneous Solvation Theory)</p>
            </div>
            <Show when={hasTrajectory()}>
              <span class="badge badge-sm badge-success badge-outline gap-1">
                <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                </svg>
                Trajectory loaded
              </span>
            </Show>
          </div>
          <p class="text-xs text-base-content/70 leading-relaxed mb-3">
            Analyzes water behavior from an MD trajectory to identify hydration sites that should be displaced or exploited. Computes solute-water energy, water-water energy, and translational/orientational entropy, decomposed into hydrophobic, H-bond donor, and H-bond acceptor channels.
          </p>
          <div class="flex items-center gap-4 text-[11px] text-base-content/60 mb-3">
            <span>~2-10 min (analysis only)</span>
            <Show when={!hasTrajectory()}>
              <span>~30-60 min (including short MD)</span>
            </Show>
          </div>
          <button
            class="btn btn-primary btn-sm w-full"
            disabled={!canCompute()}
            onClick={handleCompute}
          >
            <Show
              when={!map().isComputing}
              fallback={
                <>
                  <span class="loading loading-spinner loading-xs" />
                  Computing...
                </>
              }
            >
              {hasTrajectory() ? 'Compute Water Map' : 'Run Simulation + Water Map'}
            </Show>
          </button>
        </div>
      </div>

      <Show when={map().isComputing}>
        <div class="flex items-center gap-3">
          <progress class="progress progress-primary flex-1" value={map().progressPct} max="100" />
          <span class="text-xs text-base-content/60 min-w-[100px] text-right">{map().progress}</span>
        </div>
      </Show>

      <Show when={map().error}>
        <div class="alert alert-error py-2">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span class="text-xs">{map().error}</span>
        </div>
      </Show>

      <Show when={map().showMdConfirm}>
        <div class="modal modal-open">
          <div class="modal-box max-w-sm">
            <h3 class="font-bold text-sm mb-3">Run MD Simulation?</h3>
            <p class="text-xs text-base-content/70 mb-3">
              Water thermodynamics analysis requires an MD trajectory. No trajectory is currently loaded, so a short <strong>2-5 ns</strong> explicit-water simulation will be run first.
            </p>

            <div class="bg-base-200 rounded-lg p-3 mb-4">
              <div class="flex items-center justify-between">
                <span class="text-xs text-base-content/60">Estimated time</span>
                <span class="text-sm font-mono font-bold">~{map().estimatedTimeMin} min</span>
              </div>
              <div class="flex items-center justify-between mt-1">
                <span class="text-xs text-base-content/60">GPU</span>
                <span class="text-xs font-mono">Metal / OpenCL (Apple Silicon)</span>
              </div>
            </div>

            <p class="text-xs text-base-content/50 mb-4">
              The simulation will use the same force field and parameters as Simulate mode. Results are saved to the project&apos;s surfaces directory.
            </p>

            <div class="modal-action">
              <button class="btn btn-sm" onClick={() => setMapShowMdConfirm(false)}>
                Cancel
              </button>
              <button class="btn btn-sm btn-primary" onClick={handleConfirmMd}>
                Run Simulation
              </button>
            </div>
          </div>
          <div class="modal-backdrop" onClick={() => setMapShowMdConfirm(false)} />
        </div>
      </Show>
    </div>
  );
};

export default MapMode;
