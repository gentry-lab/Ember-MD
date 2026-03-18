import { Component, onMount, onCleanup, createSignal, Show } from 'solid-js';
import path from 'path';
import { workflowStore } from '../../stores/workflow';
import { buildDockFolderName } from '../../utils/jobName';
import TerminalOutput from '../shared/TerminalOutput';

const DockStepProgress: Component = () => {
  const {
    state,
    appendLog,
    setDockStep,
    setDockTotalLigands,
    setDockCompletedLigands,
    setDockResults,
    setDockOutputDir,
    setDockCordialScored,
    setIsRunning,
    setCurrentPhase,
    setError,
    clearLogs,
  } = workflowStore;

  const [hasStarted, setHasStarted] = createSignal(false);
  const [cordialRunning, setCordialRunning] = createSignal(false);

  const api = window.electronAPI;

  // Subscribe to dock output events
  onMount(() => {
    const cleanup = api.onDockOutput((data) => {
      appendLog(data.data);

      // Parse DOCKING: X/N lines for authoritative progress count
      const dockingMatch = data.data.match(/DOCKING:\s*(\d+)\s*\/\s*(\d+)/);
      if (dockingMatch) {
        setDockCompletedLigands(parseInt(dockingMatch[1]));
      }
    });
    onCleanup(cleanup);
  });

  const runDocking = async () => {
    const dock = state().dock;
    if (!dock.receptorPrepared) return;
    if (!dock.referenceLigandPath) return;
    if (dock.ligandSdfPaths.length === 0) return;

    setIsRunning(true);
    setCurrentPhase('generation');
    clearLogs();
    setDockTotalLigands(dock.ligandSdfPaths.length);
    setDockCompletedLigands(0);

    // Build output directory path
    const defaultDir = await api.getDefaultOutputDir();
    const baseOutputDir = state().customOutputDir || defaultDir;
    const jobName = state().jobName.trim();
    const projectDir = path.join(baseOutputDir, jobName);
    const dockFolder = buildDockFolderName({
      referenceLigandId: dock.referenceLigandId,
      numLigands: dock.ligandSdfPaths.length,
    });
    const outputDir = path.join(projectDir, dockFolder);
    setDockOutputDir(outputDir);

    try {
      let ligandPaths = [...dock.ligandSdfPaths];

      // Preprocessing: protonation enumeration
      if (dock.protonationConfig.enabled && ligandPaths.length > 0) {
        appendLog('--- Enumerating protonation states... ---\n');
        const protonDir = path.join(projectDir, 'protonated');
        const protonResult = await api.enumerateProtonation(
          ligandPaths, protonDir,
          dock.protonationConfig.phMin, dock.protonationConfig.phMax
        );
        if (protonResult.ok && protonResult.value.outputPaths.length > 0) {
          ligandPaths = protonResult.value.outputPaths;
          appendLog(`  ${ligandPaths.length} protonation variants generated\n\n`);
        } else {
          appendLog('  Protonation unchanged (Dimorphite-DL may not be installed)\n\n');
        }
      }

      // Preprocessing: conformer generation
      if (dock.conformerConfig.method !== 'none' && ligandPaths.length > 0) {
        appendLog('--- Generating conformers... ---\n');
        const confDir = path.join(projectDir, 'conformers');
        const confResult = await api.generateConformers(
          ligandPaths, confDir,
          dock.conformerConfig.maxConformers,
          dock.conformerConfig.rmsdCutoff,
          dock.conformerConfig.energyWindow
        );
        if (confResult.ok && confResult.value.outputPaths.length > 0) {
          ligandPaths = confResult.value.outputPaths;
          appendLog(`  ${ligandPaths.length} conformers generated\n\n`);
        } else {
          appendLog('  Conformer generation skipped\n\n');
        }
      }

      // Update total count after preprocessing may have expanded the ligand set
      setDockTotalLigands(ligandPaths.length);

      const result = await api.runVinaDocking(
        dock.receptorPrepared,
        dock.referenceLigandPath,
        ligandPaths,
        outputDir,
        dock.config
      );

      if (!result.ok) {
        if (state().currentPhase !== 'idle') {
          setError(result.error.message);
          setCurrentPhase('error');
        }
        setIsRunning(false);
        return;
      }

      // Docking complete -- run CORDIAL rescoring if enabled and available
      if (dock.cordialConfig.enabled && dock.cordialAvailable) {
        setCordialRunning(true);
        appendLog('\n--- Rescoring with CORDIAL... ---\n');

        try {
          const cordialResult = await api.runCordialScoring(
            outputDir,
            dock.cordialConfig.batchSize
          );
          if (cordialResult.ok) {
            setDockCordialScored(true);
            appendLog(`\nCORDIAL rescoring complete: ${cordialResult.value.count} poses scored\n`);
          } else {
            appendLog(`\nCORDIAL rescoring failed: ${cordialResult.error.message}\n`);
          }
        } catch (err) {
          appendLog(`\nCORDIAL rescoring error: ${(err as Error).message}\n`);
        }
        setCordialRunning(false);
      }

      // Parse results
      const parseResult = await api.parseDockResults(outputDir);
      if (parseResult.ok) {
        setDockResults(parseResult.value);
      } else {
        appendLog(`\nWarning: Failed to parse results: ${parseResult.error.message}\n`);
      }

      setCurrentPhase('complete');
      setIsRunning(false);
    } catch (err) {
      if (state().currentPhase !== 'idle') {
        setError((err as Error).message);
        setCurrentPhase('error');
      }
      setIsRunning(false);
    }
  };


  onMount(() => {
    // Only start docking if we haven't already started or completed
    const phase = state().currentPhase;
    if (!state().isRunning && !hasStarted() && phase !== 'complete') {
      setHasStarted(true);
      runDocking();
    }
  });

  const handleBack = () => {
    if (!state().isRunning) {
      setCurrentPhase('idle');
      setError(null);
      clearLogs();
      setDockCompletedLigands(0);
      setDockResults([]);
      setDockStep('dock-configure');
    }
  };

  const totalLigandsDisplay = () => state().dock.totalLigands || state().dock.ligandSdfPaths.length;

  const dockProgress = () => {
    const total = totalLigandsDisplay();
    const completed = state().dock.completedLigands;
    if (total <= 0) return 0;
    return Math.min(100, (completed / total) * 100);
  };

  return (
    <div class="h-full flex flex-col">
      {/* Title + Status */}
      <div class="flex items-center justify-between mb-3">
        <div>
          <h2 class="text-xl font-bold">
            {state().currentPhase === 'complete'
              ? 'Docking Complete'
              : cordialRunning()
              ? 'Rescoring with CORDIAL...'
              : 'Running Docking'}
          </h2>
          <p class="text-sm text-base-content/90">
            {state().currentPhase === 'complete'
              ? `${state().dock.results.length} poses generated`
              : cordialRunning()
              ? 'Neural network rescoring in progress'
              : totalLigandsDisplay() > 0
              ? `Docking ${state().dock.completedLigands} / ${totalLigandsDisplay()} ligands`
              : 'Initializing...'}
          </p>
        </div>
        <div class="flex items-center gap-2">
          <Show when={state().isRunning}>
            <button
              class="btn btn-circle btn-xs btn-ghost text-error"
              title="Cancel"
              onClick={async () => {
                await api.cancelVinaDocking();
                setIsRunning(false);
                setCurrentPhase('idle');
                appendLog('\n--- Docking cancelled by user ---\n');
              }}
            >
              <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M6 6h12v12H6z" />
              </svg>
            </button>
            <span class="loading loading-spinner loading-sm text-primary"></span>
          </Show>
          <Show when={state().currentPhase === 'complete'}>
            <span class="badge badge-success badge-sm">Done</span>
          </Show>
          <Show when={state().currentPhase === 'error'}>
            <span class="badge badge-error badge-sm">Error</span>
          </Show>
        </div>
      </div>

      {/* Progress bar */}
      <div class="mb-3 p-3 bg-base-200 rounded-lg">
        <div class="flex justify-between text-xs text-base-content/80 mb-1">
          <span>
            {state().dock.completedLigands} / {totalLigandsDisplay()} ligands
          </span>
          <span>{dockProgress().toFixed(0)}%</span>
        </div>
        <progress
          class={`progress w-full h-2 ${
            state().currentPhase === 'error'
              ? 'progress-error'
              : state().currentPhase === 'complete'
              ? 'progress-success'
              : 'progress-primary'
          }`}
          value={dockProgress()}
          max="100"
        ></progress>
        {/* CORDIAL secondary indicator */}
        <Show when={cordialRunning()}>
          <div class="mt-2 flex items-center gap-2">
            <span class="loading loading-dots loading-xs text-secondary"></span>
            <span class="text-xs text-secondary">Rescoring with CORDIAL...</span>
          </div>
        </Show>
      </div>

      {/* Error message */}
      <Show when={state().errorMessage}>
        <div class="alert alert-error py-2 mb-2">
          <svg class="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span class="text-sm">{state().errorMessage}</span>
        </div>
      </Show>

      <TerminalOutput title="Vina Docking Output" logs={state().logs} />

      {/* Navigation */}
      <div class="flex justify-between mt-3">
        <button class="btn btn-ghost btn-sm" onClick={handleBack} disabled={state().isRunning}>
          <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 17l-5-5m0 0l5-5m-5 5h12" />
          </svg>
          Back
        </button>

        <Show when={state().currentPhase === 'complete'}>
          <button class="btn btn-primary" onClick={() => setDockStep('dock-results')}>
            View Results
            <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </button>
        </Show>

        <Show when={state().currentPhase === 'error'}>
          <button
            class="btn btn-error btn-sm"
            onClick={() => {
              setCurrentPhase('idle');
              clearLogs();
              setError(null);
              setHasStarted(false);
              runDocking();
            }}
          >
            Retry
          </button>
        </Show>
      </div>
    </div>
  );
};

export default DockStepProgress;
