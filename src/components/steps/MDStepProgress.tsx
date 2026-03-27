// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, onMount, onCleanup, createSignal, createMemo, For, Show } from 'solid-js';
import path from 'path';
import { workflowStore } from '../../stores/workflow';
import { useMdOutput } from '../../hooks/useElectronApi';
import { MDStage } from '../../../shared/types/md';
import { buildMdRunFolderName, estimateChargeTime } from '../../utils/jobName';
import { projectPathsFromProjectDir } from '../../utils/projectPaths';
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
    setMdOutputDir,
    setIsRunning,
    setIsPaused,
    setCurrentPhase,
    setError,
    clearLogs,
  } = workflowStore;

  const [hasStarted, setHasStarted] = createSignal(false);
  const [productionNs, setProductionNs] = createSignal<{ current: number; total: number; nsPerDay?: number; etaSeconds?: number; timestamp?: string } | null>(null);
  const [chargeEstimate, setChargeEstimate] = createSignal<string | null>(null);
  const [showPrepHint, setShowPrepHint] = createSignal(false);

  // Pause/resume transition state
  const [pauseTransition, setPauseTransition] = createSignal<'idle' | 'pausing' | 'resuming'>('idle');
  let lastOutputTime = Date.now();
  let awaitingResume = false;

  // Stop modal state
  const [showStopModal, setShowStopModal] = createSignal(false);
  const [showDiscardConfirm, setShowDiscardConfirm] = createSignal(false);
  const [isStopping, setIsStopping] = createSignal(false);

  // Adjust runtime state
  const [adjustTarget, setAdjustTarget] = createSignal<number | null>(null);
  const [isAdjusting, setIsAdjusting] = createSignal(false);

  // After 8s in building/parameterizing, show a "this is normal" hint
  let prepHintTimer: ReturnType<typeof setTimeout> | undefined;
  const startPrepHintTimer = () => {
    clearTimeout(prepHintTimer);
    setShowPrepHint(false);
    prepHintTimer = setTimeout(() => setShowPrepHint(true), 8000);
  };
  const clearPrepHintTimer = () => {
    clearTimeout(prepHintTimer);
    setShowPrepHint(false);
  };
  onCleanup(clearPrepHintTimer);

  const api = window.electronAPI;

  // Parse progress from MD output
  useMdOutput((data) => {
    appendLog(data.data);
    lastOutputTime = Date.now();

    // If we're waiting for resume confirmation, first PROGRESS line confirms it
    if (awaitingResume && data.data.includes('PROGRESS:')) {
      awaitingResume = false;
      setIsPaused(false);
      setPauseTransition('idle');
    }

    // Parse PROGRESS:stage:value lines
    // Production format: PROGRESS:production:1.5/10.0 (current_ns/total_ns)
    // Parameterizing format: PROGRESS:parameterizing:0:59 (progress:atom_count)
    // Other stages: PROGRESS:stage:percentage
    const progressMatch = data.data.match(/PROGRESS:(\w+):(\S+)/);
    if (progressMatch) {
      const stage = progressMatch[1] as MDStage;
      const value = progressMatch[2];
      setMdCurrentStage(stage);

      // Manage the "still working" hint for early prep stages
      if (stage === 'building' || stage === 'parameterizing') {
        if (!showPrepHint()) startPrepHintTimer();
      } else {
        clearPrepHintTimer();
      }

      if (stage === 'production' && value.includes('/')) {
        // Parse ns format: current/total or current/total:nsPerDay:etaSeconds
        const parts = value.split(':');
        const [current, total] = parts[0].split('/').map(parseFloat);
        const nsPerDay = parts.length > 1 ? parseFloat(parts[1]) : undefined;
        const etaSeconds = parts.length > 2 ? parseFloat(parts[2]) : undefined;
        const timestamp = parts.length > 3 ? parts[3] : undefined;
        setProductionNs({ current, total, nsPerDay, etaSeconds, timestamp });
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
      void runPostSimulation(mdResult.systemPdbPath, trajectoryPath, outputDir);
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
    const projectDir = state().projectDir;
    if (!projectDir) {
      setError('No project selected');
      setCurrentPhase('error');
      setIsRunning(false);
      return;
    }
    const paths = projectPathsFromProjectDir(projectDir);

    const compoundId = state().md.config.compoundId?.trim() || '';
    const runFolder = buildMdRunFolderName({
      forceFieldPreset: state().md.config.forceFieldPreset,
      temperatureK: state().md.config.temperatureK,
      productionNs: state().md.config.productionNs,
      compoundId,
      inputMode: state().md.inputMode,
    });

    // Deduplicate: append _run2, _run3, etc. if folder exists
    let finalRunFolder = runFolder;
    let n = 1;
    while (await api.fileExists(paths.simulations(finalRunFolder).root)) {
      n++;
      finalRunFolder = `${runFolder}_run${n}`;
    }
    const outputDir = paths.simulations(finalRunFolder).root;
    setMdOutputDir(outputDir);

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
    const phase = state().currentPhase;
    if (!state().isRunning && !hasStarted() && phase !== 'complete' && phase !== 'error') {
      setHasStarted(true);
      void runSimulation();
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
                ? `Production: ${productionNs()!.current.toFixed(1)} / ${productionNs()!.total.toFixed(1)} ns${productionNs()!.nsPerDay ? ` (${productionNs()!.nsPerDay!.toFixed(1)} ns/day)` : ''}`
                : state().md.currentStage === 'parameterizing' && chargeEstimate()
                  ? `Computing AM1-BCC charges (${chargeEstimate()})`
                  : state().md.currentStage
                    ? STAGES.find(s => s.id === state().md.currentStage)?.description || 'Processing...'
                    : 'Initializing...'}
          </p>
          <Show when={showPrepHint() && state().isRunning && (state().md.currentStage === 'building' || state().md.currentStage === 'parameterizing')}>
            <p class="text-xs text-warning mt-0.5">
              {state().md.currentStage === 'parameterizing'
                ? 'AM1-BCC charge generation can be slow for larger ligands — this is normal'
                : 'System setup is still running — receptor prep and solvation can take a moment'}
            </p>
          </Show>
        </div>
        <div class="flex items-center gap-2">
          <Show when={state().md.systemInfo}>
            <span class="text-xs text-base-content/80">{state().md.systemInfo!.atomCount.toLocaleString()} atoms</span>
          </Show>
          {state().isRunning && (
            <>
              <Show
                when={!state().isPaused && pauseTransition() !== 'pausing'}
                fallback={
                  <button
                    class="btn btn-circle btn-xs btn-info"
                    title="Resume"
                    disabled={pauseTransition() !== 'idle'}
                    onClick={async () => {
                      setPauseTransition('resuming');
                      awaitingResume = true;
                      await window.electronAPI.resumeMdSimulation();
                      appendLog('\n--- Simulation resumed ---\n');
                      // Fallback: if no PROGRESS line arrives within 2s, assume resumed
                      setTimeout(() => {
                        if (awaitingResume) {
                          awaitingResume = false;
                          setIsPaused(false);
                          setPauseTransition('idle');
                        }
                      }, 2000);
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
                  disabled={pauseTransition() !== 'idle'}
                  onClick={async () => {
                    setPauseTransition('pausing');
                    await window.electronAPI.pauseMdSimulation();
                    appendLog('\n--- Simulation paused ---\n');
                    // Confirm pause by detecting output silence (300ms no new output)
                    const checkPaused = () => {
                      if (Date.now() - lastOutputTime >= 300) {
                        setIsPaused(true);
                        setPauseTransition('idle');
                      } else {
                        setTimeout(checkPaused, 100);
                      }
                    };
                    // Fallback: 2s timeout forces state regardless
                    setTimeout(() => {
                      if (pauseTransition() === 'pausing') {
                        setIsPaused(true);
                        setPauseTransition('idle');
                      }
                    }, 2000);
                    setTimeout(checkPaused, 300);
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
                onClick={() => setShowStopModal(true)}
              >
                <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M6 6h12v12H6z" />
                </svg>
              </button>
            </>
          )}
          {state().isRunning && pauseTransition() === 'pausing' && (
            <span class="badge badge-warning badge-sm animate-pulse">Pausing...</span>
          )}
          {state().isRunning && pauseTransition() === 'resuming' && (
            <span class="badge badge-info badge-sm animate-pulse">Resuming...</span>
          )}
          {state().isRunning && !state().isPaused && pauseTransition() === 'idle' && (
            <span class="loading loading-spinner loading-sm text-primary" />
          )}
          {state().isPaused && pauseTransition() === 'idle' && (
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
        <Show when={state().md.currentStage === 'production' && productionNs()?.nsPerDay}>
          <div class="flex justify-between text-[10px] text-base-content/50 mt-0.5">
            <span>
              {productionNs()!.nsPerDay!.toFixed(1)} ns/day
              <Show when={productionNs()!.timestamp}>
                {' '}&bull; {productionNs()!.timestamp}
              </Show>
            </span>
            <Show when={productionNs()!.etaSeconds! > 0}>
              <span>
                {(() => {
                  const s = productionNs()!.etaSeconds!;
                  const eta = new Date(Date.now() + s * 1000);
                  const timeStr = eta.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                  const isToday = eta.toDateString() === new Date().toDateString();
                  const dateStr = isToday ? '' : ` ${eta.toLocaleDateString([], { month: 'short', day: 'numeric' })}`;
                  if (s < 60) return `done ~${timeStr}${dateStr} (~${Math.round(s)}s)`;
                  if (s < 3600) return `done ~${timeStr}${dateStr} (~${Math.round(s / 60)}m)`;
                  const h = Math.floor(s / 3600);
                  const m = Math.round((s % 3600) / 60);
                  return `done ~${timeStr}${dateStr} (~${h}h ${m}m)`;
                })()}
              </span>
            </Show>
          </div>
        </Show>
        <Show when={state().isRunning && state().md.currentStage === 'production' && productionNs()}>
          <div class="flex items-center gap-2 mt-1.5">
            <span class="text-[10px] text-base-content/50">Adjust runtime:</span>
            <input
              type="number"
              class="input input-xs input-bordered w-16 text-xs"
              min={0.1}
              step={1}
              value={adjustTarget() ?? productionNs()!.total}
              onInput={(e) => setAdjustTarget(parseFloat(e.currentTarget.value) || null)}
            />
            <span class="text-[10px] text-base-content/50">ns</span>
            <button
              class="btn btn-xs btn-outline btn-primary"
              disabled={isAdjusting() || !adjustTarget() || adjustTarget() === productionNs()!.total}
              onClick={async () => {
                const target = adjustTarget();
                if (!target || target <= 0) return;
                setIsAdjusting(true);
                await window.electronAPI.extendMdSimulation(target);
                appendLog(`\n--- Adjusted runtime to ${target.toFixed(1)} ns ---\n`);
                setAdjustTarget(null);
                setIsAdjusting(false);
              }}
            >
              Set
            </button>
          </div>
        </Show>
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
              void runSimulation();
            }}
          >
            Retry
          </button>
        )}
      </div>

      {/* Stop confirmation modal */}
      <Show when={showStopModal()}>
        <div class="modal modal-open">
          <div class="modal-box max-w-sm">
            <h3 class="font-bold text-sm mb-3">Stop Simulation?</h3>

            <Show when={!showDiscardConfirm()}>
              <Show when={state().md.currentStage === 'production'}>
                <p class="text-xs text-base-content/70 mb-3">
                  The trajectory written so far can be analyzed for partial results.
                </p>
                <button
                  class="btn btn-primary btn-sm w-full mb-2"
                  disabled={isStopping()}
                  onClick={async () => {
                    setIsStopping(true);
                    // Resume if paused — macOS won't deliver SIGTERM to a SIGSTOP'd process
                    if (state().isPaused) {
                      await window.electronAPI.resumeMdSimulation();
                    }
                    await window.electronAPI.cancelMdSimulation();
                    appendLog('\n--- Simulation stopped: collecting partial results ---\n');
                    setIsRunning(false);
                    setIsPaused(false);
                    setShowStopModal(false);
                    setIsStopping(false);

                    // Collect results from partial trajectory
                    const outputDir = state().md.outputDir;
                    if (outputDir) {
                      const systemPdb = path.join(outputDir, 'system.pdb');
                      const trajectory = path.join(outputDir, 'trajectory.dcd');
                      const [sysExists, trajExists] = await Promise.all([
                        api.fileExists(systemPdb),
                        api.fileExists(trajectory),
                      ]);

                      if (sysExists && trajExists) {
                        const mdResult = {
                          systemPdbPath: systemPdb,
                          trajectoryPath: trajectory,
                          equilibratedPdbPath: path.join(outputDir, 'equilibrated.pdb'),
                          finalPdbPath: '',
                          energyCsvPath: path.join(outputDir, 'energy.csv'),
                        };
                        setMdResult(mdResult);
                        void runPostSimulation(mdResult.systemPdbPath, trajectory, outputDir);
                      } else {
                        setCurrentPhase('idle');
                      }
                    } else {
                      setCurrentPhase('idle');
                    }
                  }}
                >
                  Collect Results
                </button>
              </Show>

              <Show when={state().md.currentStage !== 'production'}>
                <p class="text-xs text-base-content/70 mb-3">
                  No production data has been written yet.
                </p>
              </Show>

              <button
                class="btn btn-error btn-sm w-full mb-2"
                disabled={isStopping()}
                onClick={() => setShowDiscardConfirm(true)}
              >
                Discard
              </button>

              <div class="modal-action">
                <button class="btn btn-sm" onClick={() => { setShowStopModal(false); setShowDiscardConfirm(false); }}>
                  Cancel
                </button>
              </div>
            </Show>

            <Show when={showDiscardConfirm()}>
              <p class="text-sm text-error mb-3">
                Are you sure? All simulation data will be deleted.
              </p>
              <div class="modal-action">
                <button class="btn btn-sm" onClick={() => setShowDiscardConfirm(false)}>
                  Back
                </button>
                <button
                  class="btn btn-error btn-sm"
                  disabled={isStopping()}
                  onClick={async () => {
                    setIsStopping(true);
                    if (state().isPaused) {
                      await window.electronAPI.resumeMdSimulation();
                    }
                    await window.electronAPI.cancelMdSimulation();
                    appendLog('\n--- Simulation discarded ---\n');

                    const outputDir = state().md.outputDir;
                    if (outputDir) {
                      await window.electronAPI.deleteDirectory(outputDir);
                    }

                    setIsRunning(false);
                    setIsPaused(false);
                    setCurrentPhase('idle');
                    setShowStopModal(false);
                    setShowDiscardConfirm(false);
                    setIsStopping(false);
                  }}
                >
                  Delete Run
                </button>
              </div>
            </Show>
          </div>
          <div class="modal-backdrop" onClick={() => { setShowStopModal(false); setShowDiscardConfirm(false); }} />
        </div>
      </Show>
    </div>
  );
};

export default MDStepProgress;
