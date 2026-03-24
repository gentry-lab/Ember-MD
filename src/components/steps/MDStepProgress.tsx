import { Component, onMount, createSignal, createMemo, For, Show } from 'solid-js';
import path from 'path';
import { workflowStore } from '../../stores/workflow';
import { useMdOutput } from '../../hooks/useElectronApi';
import { MDStage } from '../../../shared/types/md';
import { buildMdRunFolderName, estimateChargeTime } from '../../utils/jobName';
import { projectPaths } from '../../utils/projectPaths';
import TerminalOutput from '../shared/TerminalOutput';

interface StageInfo {
  id: MDStage;
  label: string;
  description: string;
}

const STAGES: StageInfo[] = [
  { id: 'building', label: 'Building', description: 'Setting up system topology' },
  { id: 'parameterizing', label: 'AM1-BCC', description: 'Computing partial charges (sqm)' },
  { id: 'min_restrained', label: 'Min (restr)', description: 'Restrained minimization (heavy atoms)' },
  { id: 'min_unrestrained', label: 'Min (free)', description: 'Unrestrained minimization' },
  { id: 'heating', label: 'Heating', description: 'NVT→NPT heating with backbone restraints' },
  { id: 'npt_restrained', label: 'NPT Equil', description: 'NPT equilibration with backbone restraints' },
  { id: 'release', label: 'Release', description: 'Gradual restraint release' },
  { id: 'equilibration', label: 'Free Equil', description: 'Unrestrained NPT equilibration' },
  { id: 'production', label: 'Production', description: 'Running production MD' },
  { id: 'clustering', label: 'Clustering', description: 'Clustering trajectory into 10 centroids' },
  { id: 'scoring', label: 'Scoring', description: 'Scoring clusters (Vina + CORDIAL)' },
  { id: 'report', label: 'Report', description: 'Generating analysis report' },
];

const MDStepProgress: Component = () => {
  const {
    state,
    appendLog,
    setMdStep,
    setMdCurrentStage,
    setMdStageProgress,
    setMdSystemInfo,
    setMdResult,
    setMdClusteringResultsFromIpc,
    setMdClusterScores,
    setMdTorsionAnalysis,
    setIsRunning,
    setIsPaused,
    setCurrentPhase,
    setError,
    clearLogs,
  } = workflowStore;

  const [hasStarted, setHasStarted] = createSignal(false);
  const [productionNs, setProductionNs] = createSignal<{ current: number; total: number } | null>(null);
  const [chargeEstimate, setChargeEstimate] = createSignal<string | null>(null);

  const api = window.electronAPI;

  // Parse progress from MD output
  useMdOutput((data) => {
    appendLog(data.data);

    // Parse PROGRESS:stage:value lines
    // Production format: PROGRESS:production:1.5/10.0 (current_ns/total_ns)
    // Parameterizing format: PROGRESS:parameterizing:0:59 (progress:atom_count)
    // Other stages: PROGRESS:stage:percentage
    const progressMatch = data.data.match(/PROGRESS:(\w+):(\S+)/);
    if (progressMatch) {
      const stage = progressMatch[1] as MDStage;
      const value = progressMatch[2];
      setMdCurrentStage(stage);

      if (stage === 'production' && value.includes('/')) {
        // Parse ns format: current/total
        const [current, total] = value.split('/').map(parseFloat);
        setProductionNs({ current, total });
        // Convert to percentage for progress bar
        setMdStageProgress(total > 0 ? (current / total) * 100 : 0);
      } else if (stage === 'parameterizing' && value.includes(':')) {
        // Parse parameterizing format: progress:atom_count
        // Full match is PROGRESS:parameterizing:0:59
        const fullMatch = data.data.match(/PROGRESS:parameterizing:(\d+):(\d+)/);
        if (fullMatch) {
          setMdStageProgress(parseFloat(fullMatch[1]));
          const atoms = parseInt(fullMatch[2]);
          // Estimate based on empirical sqm profiling
          setChargeEstimate(`${atoms} atoms, est. ${estimateChargeTime(atoms)}`);
        } else {
          setMdStageProgress(parseFloat(value));
        }
      } else {
        // Regular percentage format
        setMdStageProgress(parseFloat(value));
        if (stage !== 'production') {
          setProductionNs(null);
        }
      }
    }

    // Parse SYSTEM_INFO:atomCount:volume lines
    const systemMatch = data.data.match(/SYSTEM_INFO:(\d+):(\d+)/);
    if (systemMatch) {
      setMdSystemInfo({
        atomCount: parseInt(systemMatch[1]),
        boxVolumeA3: parseInt(systemMatch[2]),
      });
    }

    // Parse SUCCESS:path line — simulation done, auto-trigger scoring
    const successMatch = data.data.match(/SUCCESS:(.+)/);
    if (successMatch) {
      const trajectoryPath = successMatch[1].trim();
      const outputDir = path.dirname(trajectoryPath);
      const trajBasename = path.basename(trajectoryPath);
      // New layout: unprefixed (trajectory.dcd). Legacy: {prefix}_trajectory.dcd
      const isUnprefixed = trajBasename === 'trajectory.dcd';
      const jobPrefix = isUnprefixed ? '' : (trajBasename.replace(/_trajectory\.dcd$/, '') || '');
      const pf = (name: string) => jobPrefix ? `${jobPrefix}_${name}` : name;
      const mdResult = {
        systemPdbPath: path.join(outputDir, pf('system.pdb')),
        trajectoryPath: trajectoryPath,
        equilibratedPdbPath: path.join(outputDir, pf('equilibrated.pdb')),
        finalPdbPath: path.join(outputDir, pf('final.pdb')),
        energyCsvPath: path.join(outputDir, pf('energy.csv')),
      };
      setMdResult(mdResult);

      // Auto-trigger clustering + scoring + report
      runPostSimulation(mdResult.systemPdbPath, trajectoryPath, outputDir);
    }

    // Parse post-simulation scoring progress
    const scoringMatch = data.data.match(/PROGRESS:(clustering|scoring_split|scoring_vina|scoring_cordial|scoring):(\d+)/);
    if (scoringMatch) {
      const step = scoringMatch[1];
      const pct = parseInt(scoringMatch[2]);
      if (step === 'clustering') {
        setMdCurrentStage('clustering');
      } else if (step.startsWith('scoring')) {
        setMdCurrentStage('scoring');
      }
      setMdStageProgress(pct);
    }
  });

  const runPostSimulation = async (topologyPath: string, trajectoryPath: string, resultsDir: string) => {
    const isLigandOnly = state().md.inputMode === 'ligand_only';
    const isApo = state().md.inputMode === 'apo';

    // Determine the run root: if resultsDir ends with /results, parent is run root; otherwise resultsDir IS the run root
    const simRunRoot = path.basename(resultsDir) === 'results' ? path.dirname(resultsDir) : resultsDir;
    const analysisDir = path.join(simRunRoot, 'analysis');
    // Try inputs/ subdir first, fall back to original state paths (MD runner may not create inputs/)
    const inputsLigandPath = path.join(simRunRoot, 'inputs', 'ligand.sdf');
    const inputsReceptorPath = path.join(simRunRoot, 'inputs', 'receptor.pdb');
    const hasInputsDir = await api.fileExists(path.join(simRunRoot, 'inputs'));
    const inputLigandSdf = hasInputsDir ? inputsLigandPath : (state().md.ligandSdf || inputsLigandPath);
    const inputReceptorPdb = (isLigandOnly || isApo) ? undefined
      : (hasInputsDir ? inputsReceptorPath : (state().md.receptorPdb || inputsReceptorPath));

    // Step 1: Cluster every completed simulation. Holo runs attach scores to the canonical clustering output.
    setMdCurrentStage('clustering');
    setMdStageProgress(0);
    setMdClusteringResultsFromIpc(null);
    setMdClusterScores([]);
    setMdTorsionAnalysis(null);

    if (!isLigandOnly && !isApo) {
      setMdCurrentStage('clustering');
      appendLog('\n=== Auto-scoring: clustering + scoring ===\n');

      try {
        const scoreResult = await api.scoreMdClusters({
          topologyPath,
          trajectoryPath,
          outputDir: analysisDir,
          inputLigandSdf,
          inputReceptorPdb,
          numClusters: 10,
          enableVina: !isLigandOnly,
          enableCordial: !isLigandOnly,
        });

        if (scoreResult.ok) {
          setMdClusteringResultsFromIpc(scoreResult.value.clusteringResults);
          setMdClusterScores(scoreResult.value.clusters);
          appendLog(`\nScored ${scoreResult.value.clusters.length} clusters\n`);
        } else {
          const message = `Cluster scoring failed: ${scoreResult.error.message}`;
          appendLog(`\n${message}\n`);
          setError(message);
          setCurrentPhase('error');
          setIsRunning(false);
          return;
        }
      } catch (err) {
        const message = `Cluster scoring error: ${(err as Error).message}`;
        appendLog(`\n${message}\n`);
        setError(message);
        setCurrentPhase('error');
        setIsRunning(false);
        return;
      }
    } else {
      appendLog(`\n=== Auto-clustering ${isApo ? 'apo' : 'ligand-only'} simulation ===\n`);

      try {
        const clusterResult = await api.clusterTrajectory({
          topologyPath,
          trajectoryPath,
          numClusters: 10,
          method: 'kmeans',
          rmsdSelection: isApo ? 'backbone' : 'ligand',
          stripWaters: true,
          outputDir: path.join(analysisDir, 'clustering'),
        });

        if (clusterResult.ok) {
          setMdClusteringResultsFromIpc(clusterResult.value);
          appendLog(`\nClustered ${clusterResult.value.clusters.length} centroids\n`);
        } else {
          appendLog(`\nClustering failed: ${clusterResult.error.message}\n`);
        }
      } catch (err) {
        appendLog(`\nClustering error: ${(err as Error).message}\n`);
      }
    }

    // Step 2: Generate report
    setMdCurrentStage('report');
    setMdStageProgress(0);
    appendLog('\n=== Generating analysis report ===\n');

    try {
      const simInfo: Record<string, string> = {};
      const si = state().md.systemInfo;
      if (si) simInfo.atoms = si.atomCount.toLocaleString();
      simInfo.temperature = `${state().md.config.temperatureK} K`;
      simInfo.duration = `${state().md.config.productionNs} ns`;
      simInfo.forceField = state().md.config.forceFieldPreset || 'ff19SB/OPC';
      if (state().md.benchmarkResult) {
        simInfo.performance = `${state().md.benchmarkResult!.nsPerDay.toFixed(1)} ns/day`;
      }
      simInfo.jobName = state().jobName.trim() || 'job';

      await api.generateMdReport({
        topologyPath,
        trajectoryPath,
        outputDir: analysisDir,
        ligandSdf: inputLigandSdf,
        simInfo,
      });

      const torsionResult = await api.loadMdTorsionAnalysis({ analysisDir });
      if (torsionResult.ok) {
        setMdTorsionAnalysis(torsionResult.value);
      }
    } catch (err) {
      appendLog(`\nReport generation error: ${(err as Error).message}\n`);
    }

    // Done — transition to results
    setCurrentPhase('complete');
    setIsRunning(false);
  };

  const runSimulation = async () => {
    const isLigandOnly = state().md.inputMode === 'ligand_only';
    const isApo = state().md.inputMode === 'apo';
    if (!isLigandOnly && !isApo && !state().md.receptorPdb) return;
    if (!isApo && !state().md.ligandSdf) return;

    setIsRunning(true);
    setCurrentPhase('generation');
    clearLogs();
    setMdClusteringResultsFromIpc(null);
    setMdClusterScores([]);
    setMdTorsionAnalysis(null);
    setMdCurrentStage('building');
    setMdStageProgress(0);

    // Use global job name as project folder, run folder inside simulations/
    const globalJobName = state().jobName.trim();
    const defaultDir = await api.getDefaultOutputDir();
    const baseOutputDir = state().customOutputDir || defaultDir;
    const paths = projectPaths(baseOutputDir, globalJobName);

    const compoundId = state().md.config.compoundId?.trim() || '';
    const runFolder = buildMdRunFolderName({
      forceFieldPreset: state().md.config.forceFieldPreset,
      temperatureK: state().md.config.temperatureK,
      productionNs: state().md.config.productionNs,
      compoundId,
    });

    // Deduplicate: append _run2, _run3, etc. if folder exists
    let finalRunFolder = runFolder;
    let n = 1;
    while (await api.fileExists(paths.simulations(finalRunFolder).root)) {
      n++;
      finalRunFolder = `${runFolder}_run${n}`;
    }
    const outputDir = paths.simulations(finalRunFolder).root;

    try {
      console.log(`[MD] Starting simulation: ${state().md.config.forceFieldPreset}, ${state().md.config.productionNs}ns → ${outputDir}`);
      const result = await api.runMdSimulation(
        state().md.receptorPdb,
        state().md.ligandSdf || '',
        outputDir,
        state().md.config,
        isLigandOnly,
        isApo
      );

      if (!result.ok) {
        // Don't overwrite if user already cancelled
        if (state().currentPhase !== 'idle') {
          setError(result.error.message);
          setCurrentPhase('error');
        }
        setIsRunning(false);
      }
    } catch (err) {
      if (state().currentPhase !== 'idle') {
        setError((err as Error).message);
        setCurrentPhase('error');
      }
      setIsRunning(false);
    }
  };

  onMount(() => {
    if (!state().isRunning && !hasStarted()) {
      setHasStarted(true);
      runSimulation();
    }
  });

  const handleBack = () => {
    if (!state().isRunning) {
      setCurrentPhase('idle');
      setError(null);
      clearLogs();
      setMdCurrentStage(null);
      setMdStageProgress(0);
      setMdResult(null);
      setMdStep('md-configure');
    }
  };

  const currentStageIndex = createMemo(() => {
    const current = state().md.currentStage;
    if (!current) return -1;
    return STAGES.findIndex(s => s.id === current);
  });

  const getStageStatus = (index: number): 'done' | 'active' | 'pending' => {
    const currentIdx = currentStageIndex();
    if (currentIdx === -1) return 'pending';
    if (index < currentIdx) return 'done';
    if (index === currentIdx) return 'active';
    return 'pending';
  };

  const overallProgress = createMemo(() => {
    const currentIdx = currentStageIndex();
    if (currentIdx === -1) return 0;
    const stageProgress = state().md.stageProgress;
    // Each stage contributes equally (except production which might be longer)
    const baseProgress = (currentIdx / STAGES.length) * 100;
    const stageContribution = (stageProgress / 100) * (100 / STAGES.length);
    return Math.min(100, baseProgress + stageContribution);
  });

  return (
    <div class="h-full flex flex-col">
      {/* Title + Status */}
      <div class="flex items-center justify-between mb-3">
        <div>
          <h2 class="text-xl font-bold gradient-text">
            {state().currentPhase === 'complete' ? 'Simulation Complete' : 'Running MD Simulation'}
          </h2>
          <p class="text-sm text-base-content/90">
            {state().currentPhase === 'complete'
              ? `${state().md.config.productionNs} ns simulation finished`
              : state().md.currentStage === 'production' && productionNs()
                ? `Production: ${productionNs()!.current.toFixed(1)} / ${productionNs()!.total.toFixed(1)} ns`
                : state().md.currentStage === 'parameterizing' && chargeEstimate()
                  ? `Computing AM1-BCC charges (${chargeEstimate()})`
                  : state().md.currentStage
                    ? STAGES.find(s => s.id === state().md.currentStage)?.description || 'Processing...'
                    : 'Initializing...'}
          </p>
        </div>
        <div class="flex items-center gap-2">
          <Show when={state().md.systemInfo}>
            <span class="text-xs text-base-content/80">{state().md.systemInfo!.atomCount.toLocaleString()} atoms</span>
          </Show>
          {state().isRunning && (
            <>
              <Show
                when={!state().isPaused}
                fallback={
                  <button
                    class="btn btn-circle btn-xs btn-info"
                    title="Resume"
                    onClick={async () => {
                      await window.electronAPI.resumeMdSimulation();
                      setIsPaused(false);
                      appendLog('\n--- Simulation resumed ---\n');
                    }}
                  >
                    <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M8 5v14l11-7z" />
                    </svg>
                  </button>
                }
              >
                <button
                  class="btn btn-circle btn-xs btn-ghost"
                  title="Pause"
                  onClick={async () => {
                    await window.electronAPI.pauseMdSimulation();
                    setIsPaused(true);
                    appendLog('\n--- Simulation paused ---\n');
                  }}
                >
                  <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M6 4h4v16H6zm8 0h4v16h-4z" />
                  </svg>
                </button>
              </Show>
              <button
                class="btn btn-circle btn-xs btn-ghost text-error"
                title="Stop"
                onClick={async () => {
                  await window.electronAPI.cancelMdSimulation();
                  setIsRunning(false);
                  setIsPaused(false);
                  setCurrentPhase('idle');
                  appendLog('\n--- Simulation cancelled by user ---\n');
                }}
              >
                <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M6 6h12v12H6z" />
                </svg>
              </button>
            </>
          )}
          {state().isRunning && !state().isPaused && (
            <span class="loading loading-spinner loading-sm text-primary" />
          )}
          {state().isPaused && (
            <span class="badge badge-warning badge-sm">Paused</span>
          )}
          {state().currentPhase === 'complete' && (
            <span class="badge badge-success badge-sm">Done</span>
          )}
          {state().currentPhase === 'error' && (
            <span class="badge badge-error badge-sm">Error</span>
          )}
        </div>
      </div>

      {/* Stage progress indicator */}
      <div class="mb-3 p-3 bg-base-200 rounded-lg">
        <div class="flex justify-between mb-2">
          <For each={STAGES}>
            {(stage, index) => {
              const status = () => getStageStatus(index());
              return (
                <div class="flex flex-col items-center" style={{ width: `${100 / STAGES.length}%` }}>
                  <div
                    class={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold ${
                      status() === 'done'
                        ? 'bg-success text-success-content'
                        : status() === 'active'
                        ? 'bg-primary text-primary-content'
                        : 'bg-base-300 text-base-content/90'
                    }`}
                  >
                    {status() === 'done' ? (
                      <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7" />
                      </svg>
                    ) : (
                      index() + 1
                    )}
                  </div>
                  <span
                    class={`text-[10px] mt-1 text-center ${
                      status() === 'active' ? 'text-primary font-semibold' : 'text-base-content/80'
                    }`}
                  >
                    {stage.label}
                  </span>
                </div>
              );
            }}
          </For>
        </div>
        {/* Progress bar */}
        <progress
          class={`progress w-full h-2 ${
            state().currentPhase === 'error'
              ? 'progress-error'
              : state().currentPhase === 'complete'
              ? 'progress-success'
              : state().isPaused
              ? 'progress-warning'
              : 'progress-primary'
          }`}
          value={overallProgress()}
          max="100"
         />
        <div class="flex justify-between text-xs text-base-content/80 mt-1">
          <Show
            when={state().md.currentStage === 'production' && productionNs()}
            fallback={
              <>
                <span>0%</span>
                <span>{overallProgress().toFixed(0)}%</span>
                <span>100%</span>
              </>
            }
          >
            <span>0 ns</span>
            <span class="font-semibold text-primary">{productionNs()!.current.toFixed(1)} ns</span>
            <span>{productionNs()!.total.toFixed(1)} ns</span>
          </Show>
        </div>
      </div>

      {/* Error message */}
      {state().errorMessage && (
        <div class="alert alert-error py-2 mb-2">
          <svg class="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span class="text-sm">{state().errorMessage}</span>
        </div>
      )}

      <TerminalOutput title="OpenMM Output" logs={state().logs} />

      {/* Navigation */}
      <div class="flex justify-between mt-3">
        <button class="btn btn-ghost btn-sm" onClick={handleBack} disabled={state().isRunning}>
          <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 17l-5-5m0 0l5-5m-5 5h12" />
          </svg>
          Back
        </button>

        {state().currentPhase === 'complete' && (
          <button class="btn btn-primary" onClick={() => setMdStep('md-results')}>
            View Results
            <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </button>
        )}

        {state().currentPhase === 'error' && (
          <button
            class="btn btn-error btn-sm"
            onClick={() => {
              setCurrentPhase('idle');
              clearLogs();
              setError(null);
              setHasStarted(false);
              runSimulation();
            }}
          >
            Retry
          </button>
        )}
      </div>
    </div>
  );
};

export default MDStepProgress;
