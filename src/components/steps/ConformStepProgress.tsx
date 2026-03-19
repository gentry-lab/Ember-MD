import { Component, onMount, onCleanup, createSignal, Show } from 'solid-js';
import { workflowStore } from '../../stores/workflow';
import { projectPaths } from '../../utils/projectPaths';
import TerminalOutput from '../shared/TerminalOutput';

const ConformStepProgress: Component = () => {
  const {
    state,
    appendLog,
    setConformStep,
    setConformOutputDir,
    setConformPaths,
    setConformRunning,
    setCurrentPhase,
    setError,
    clearLogs,
  } = workflowStore;

  const [hasStarted, setHasStarted] = createSignal(false);
  const api = window.electronAPI;

  onMount(() => {
    const cleanup = api.onConformOutput((data) => {
      appendLog(data.data);
    });
    onCleanup(cleanup);
  });

  const runConformers = async () => {
    const conform = state().conform;
    if (!conform.ligandSdfPath) return;

    setConformRunning(true);
    setCurrentPhase('generation');
    clearLogs();

    // Build output directory
    const defaultDir = await api.getDefaultOutputDir();
    const baseOutputDir = state().customOutputDir || defaultDir;
    const jobName = state().jobName.trim();
    const paths = projectPaths(baseOutputDir, jobName);
    const ligandName = conform.ligandName || 'ligand';
    const outputDir = paths.conformers(ligandName);
    setConformOutputDir(outputDir);

    try {
      await api.createDirectory(outputDir);

      const mcmmOpts = conform.config.method === 'mcmm' ? {
        steps: conform.config.mcmmSteps,
        temperature: conform.config.mcmmTemperature,
        sampleAmides: conform.config.sampleAmides,
      } : undefined;

      const result = await api.runConformGeneration(
        conform.ligandSdfPath,
        outputDir,
        conform.config.maxConformers,
        conform.config.rmsdCutoff,
        conform.config.energyWindow,
        conform.config.method,
        mcmmOpts
      );

      if (result.ok) {
        setConformPaths(result.value.conformerPaths);
        setCurrentPhase('complete');
        appendLog(`\n${result.value.conformerPaths.length} conformers generated\n`);
      } else {
        setError(result.error.message);
        setCurrentPhase('error');
      }
    } catch (err) {
      setError((err as Error).message);
      setCurrentPhase('error');
    }

    setConformRunning(false);
  };

  onMount(() => {
    const phase = state().currentPhase;
    if (!state().conform.isRunning && !hasStarted() && phase !== 'complete') {
      setHasStarted(true);
      runConformers();
    }
  });

  const handleBack = () => {
    if (!state().conform.isRunning) {
      setCurrentPhase('idle');
      setError(null);
      clearLogs();
      setConformStep('conform-configure');
    }
  };

  return (
    <div class="h-full flex flex-col">
      <div class="flex items-center justify-between mb-3">
        <div>
          <h2 class="text-xl font-bold">
            {state().currentPhase === 'complete' ? 'Search Complete' : 'Running Conformer Search'}
          </h2>
          <p class="text-sm text-base-content/90">
            {state().currentPhase === 'complete'
              ? `${state().conform.conformerPaths.length} conformers found`
              : `${state().conform.config.method.toUpperCase()} — ${state().conform.ligandName}`}
          </p>
        </div>
        <div class="flex items-center gap-2">
          <Show when={state().conform.isRunning}>
            <span class="loading loading-spinner loading-sm text-primary" />
          </Show>
          <Show when={state().currentPhase === 'complete'}>
            <span class="badge badge-success badge-sm">Done</span>
          </Show>
          <Show when={state().currentPhase === 'error'}>
            <span class="badge badge-error badge-sm">Error</span>
          </Show>
        </div>
      </div>

      <Show when={state().errorMessage}>
        <div class="alert alert-error py-2 mb-2">
          <span class="text-sm">{state().errorMessage}</span>
        </div>
      </Show>

      <TerminalOutput title="Conformer Search Output" logs={state().logs} />

      <div class="flex justify-between mt-3">
        <button class="btn btn-ghost btn-sm" onClick={handleBack} disabled={state().conform.isRunning}>
          <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 17l-5-5m0 0l5-5m-5 5h12" />
          </svg>
          Back
        </button>
        <Show when={state().currentPhase === 'complete'}>
          <button class="btn btn-primary" onClick={() => setConformStep('conform-results')}>
            View Results
            <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </button>
        </Show>
      </div>
    </div>
  );
};

export default ConformStepProgress;
