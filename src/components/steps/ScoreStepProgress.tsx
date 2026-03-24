// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, Show, createSignal, onCleanup, onMount } from 'solid-js';
import path from 'path';
import { workflowStore } from '../../stores/workflow';
import { buildXrayRunFolderName } from '../../utils/jobName';
import { projectPaths } from '../../utils/projectPaths';
import TerminalOutput from '../shared/TerminalOutput';

const ScoreStepProgress: Component = () => {
  const {
    state,
    appendLog,
    clearLogs,
    setCurrentPhase,
    setError,
    setIsRunning,
    setScoreOutputDir,
    setScorePdfPaths,
    setScoreLastResult,
    setScoreRunning,
    setScoreStep,
  } = workflowStore;
  const api = window.electronAPI;
  const [hasStarted, setHasStarted] = createSignal(false);

  onMount(() => {
    const cleanup = api.onXrayOutput((data) => {
      appendLog(data.data);
    });
    onCleanup(cleanup);
  });

  const runAnalysis = async () => {
    const inputDir = state().score.inputDir;
    if (!inputDir) return;

    setScoreRunning(true);
    setIsRunning(true);
    setCurrentPhase('generation');
    setError(null);
    clearLogs();

    const defaultDir = await api.getDefaultOutputDir();
    const baseOutputDir = state().customOutputDir || defaultDir;
    const paths = projectPaths(baseOutputDir, state().jobName.trim());
    const inputFolderName = path.basename(inputDir);
    const runFolder = buildXrayRunFolderName(inputFolderName);
    const outputDir = path.join(paths.xray(runFolder), 'reports');

    setScoreOutputDir(outputDir);

    try {
      await api.createDirectory(outputDir);
      const result = await api.runXrayAnalysis(inputDir, outputDir);

      if (result.ok) {
        setScoreLastResult(result.value);
        setScorePdfPaths(result.value.pdfPaths);
        setCurrentPhase('complete');
        appendLog(`\nGenerated ${result.value.pdfPaths.length} PDF report${result.value.pdfPaths.length === 1 ? '' : 's'}.\n`);
      } else {
        setError(result.error.message);
        setCurrentPhase('error');
      }
    } catch (err) {
      setError((err as Error).message);
      setCurrentPhase('error');
    }

    setScoreRunning(false);
    setIsRunning(false);
  };

  onMount(() => {
    if (!state().score.isRunning && !hasStarted() && state().currentPhase !== 'complete') {
      setHasStarted(true);
      runAnalysis();
    }
  });

  const handleBack = () => {
    if (state().score.isRunning) return;
    setCurrentPhase('idle');
    setError(null);
    clearLogs();
    setScoreStep('score-load');
  };

  return (
    <div class="h-full flex flex-col">
      <div class="flex items-center justify-between mb-3">
        <div>
          <h2 class="text-xl font-bold">
            {state().currentPhase === 'complete' ? 'X-ray Analysis Complete' : 'Running X-ray Pose Analysis'}
          </h2>
          <p class="text-sm text-base-content/90">
            {state().score.inputDir || 'Waiting for input folder'}
          </p>
        </div>
        <div class="flex items-center gap-2">
          <Show when={state().score.isRunning}>
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

      <TerminalOutput title="X-ray Analyzer Output" logs={state().logs} />

      <div class="flex justify-between mt-3">
        <button class="btn btn-ghost btn-sm" onClick={handleBack} disabled={state().score.isRunning}>
          <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 17l-5-5m0 0l5-5m-5 5h12" />
          </svg>
          Back
        </button>
        <Show when={state().currentPhase === 'complete'}>
          <button class="btn btn-primary" onClick={() => setScoreStep('score-results')}>
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

export default ScoreStepProgress;
