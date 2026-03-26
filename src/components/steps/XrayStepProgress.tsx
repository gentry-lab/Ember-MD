// Copyright (c) 2026 Ember Contributors. MIT License.
import path from 'path';
import { Component, Show, createSignal, onCleanup, onMount } from 'solid-js';
import { workflowStore } from '../../stores/workflow';
import { buildXrayRunFolderName } from '../../utils/jobName';
import { projectPathsFromProjectDir } from '../../utils/projectPaths';
import TerminalOutput from '../shared/TerminalOutput';

const XrayStepProgress: Component = () => {
  const {
    state,
    appendLog,
    clearLogs,
    setCurrentPhase,
    setError,
    setIsRunning,
    setXrayRunning,
    setXrayOutputDir,
    setXrayResult,
    setXrayStep,
  } = workflowStore;
  const api = window.electronAPI;
  const [hasStarted, setHasStarted] = createSignal(false);

  onMount(() => {
    const cleanup = api.onXrayOutput((data) => appendLog(data.data));
    onCleanup(cleanup);
  });

  const runAnalysis = async () => {
    const inputDir = state().xray.inputDir;
    const projectDir = state().projectDir;
    if (!inputDir || !projectDir) {
      setError('Project and input folder are required to run X-ray analysis.');
      setCurrentPhase('error');
      return;
    }

    setXrayRunning(true);
    setIsRunning(true);
    setCurrentPhase('generation');
    setError(null);
    clearLogs();

    const paths = projectPathsFromProjectDir(projectDir);
    const runFolder = buildXrayRunFolderName(path.basename(inputDir));
    const outputDir = paths.xray(runFolder).results;
    setXrayOutputDir(outputDir);
    setXrayResult(null);

    try {
      const result = await api.runXrayAnalysis(inputDir, outputDir);
      if (result.ok) {
        setXrayResult(result.value);
        setCurrentPhase('complete');
        appendLog(`\nX-ray analysis complete: ${result.value.pdfPaths.length} report${result.value.pdfPaths.length === 1 ? '' : 's'} generated.\n`);
      } else {
        setError(result.error.message);
        setCurrentPhase('error');
      }
    } catch (error) {
      setError((error as Error).message);
      setCurrentPhase('error');
    }

    setXrayRunning(false);
    setIsRunning(false);
  };

  onMount(() => {
    if (!state().xray.isRunning && !hasStarted() && state().currentPhase !== 'complete') {
      setHasStarted(true);
      void runAnalysis();
    }
  });

  const handleBack = () => {
    if (state().xray.isRunning) return;
    setCurrentPhase('idle');
    setError(null);
    clearLogs();
    setXrayStep('xray-load');
  };

  return (
    <div class="h-full flex flex-col">
      <div class="flex items-center justify-between mb-3">
        <div>
          <h2 class="text-xl font-bold">
            {state().currentPhase === 'complete' ? 'X-ray Analysis Complete' : 'Running X-ray Analyzer'}
          </h2>
          <p class="text-sm text-base-content/90">
            {state().xray.inputDir ? path.basename(state().xray.inputDir) : 'Preparing analysis'}
          </p>
        </div>
        <div class="flex items-center gap-2">
          <Show when={state().xray.isRunning}>
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

      <TerminalOutput title="X-ray Output" logs={state().logs} />

      <div class="flex justify-between mt-3">
        <button class="btn btn-ghost btn-sm" onClick={handleBack} disabled={state().xray.isRunning}>
          <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 17l-5-5m0 0l5-5m-5 5h12" />
          </svg>
          Back
        </button>
        <Show when={state().currentPhase === 'complete'}>
          <button class="btn btn-primary" onClick={() => setXrayStep('xray-results')}>
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

export default XrayStepProgress;
