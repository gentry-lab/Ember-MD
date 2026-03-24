import { Component, Show } from 'solid-js';
import { workflowStore } from '../../stores/workflow';

const ScoreStepLoad: Component = () => {
  const {
    state,
    setScoreInputDir,
    setScoreOutputDir,
    setScorePdfPaths,
    setScoreLastResult,
    setScoreStep,
    setError,
  } = workflowStore;
  const api = window.electronAPI;

  const handleSelectFolder = async () => {
    const dirPath = await api.selectFolder();
    if (!dirPath) return;
    setScoreInputDir(dirPath);
    setScoreOutputDir(null);
    setScorePdfPaths([]);
    setScoreLastResult(null);
    setError(null);
  };

  const handleClear = () => {
    setScoreInputDir(null);
    setScoreOutputDir(null);
    setScorePdfPaths([]);
    setScoreLastResult(null);
    setError(null);
  };

  return (
    <div class="h-full flex flex-col">
      <div class="text-center mb-3">
        <h2 class="text-xl font-bold">Score X-ray Pose</h2>
        <p class="text-sm text-base-content/90">Import a folder containing matching `*.pdb` and `*.mtz` pairs</p>
      </div>

      <div class="flex-1 min-h-0 overflow-auto flex flex-col items-center gap-4">
        <div class="card bg-base-200 shadow-lg w-full max-w-xl">
          <div class="card-body p-4">
            <Show
              when={!state().score.inputDir}
              fallback={
                <div class="space-y-3">
                  <div class="p-3 bg-base-300 rounded-lg">
                    <p class="text-xs font-semibold">{state().score.inputDir?.split('/').pop() || 'Selected folder'}</p>
                    <p class="text-[10px] font-mono text-base-content/70 break-all">{state().score.inputDir}</p>
                  </div>
                  <button class="btn btn-ghost btn-xs w-full" onClick={handleClear}>Clear</button>
                </div>
              }
            >
              <div class="space-y-3">
                <button class="btn btn-outline btn-sm w-full" onClick={handleSelectFolder}>
                  Import Folder
                </button>

                <div class="rounded-lg bg-base-300/70 p-3 text-xs text-base-content/80 leading-relaxed">
                  The analyzer matches files by stem, with fallback rules for common suffixes and compound IDs.
                  Output PDFs are written into the current Ember project under `xray/`.
                </div>
              </div>
            </Show>
          </div>
        </div>

        <Show when={state().errorMessage}>
          <div class="alert alert-error py-2 w-full max-w-xl">
            <span class="text-sm">{state().errorMessage}</span>
          </div>
        </Show>
      </div>

      <div class="flex justify-end mt-3 flex-shrink-0">
        <button
          class="btn btn-primary"
          onClick={() => setScoreStep('score-progress')}
          disabled={!state().score.inputDir}
        >
          Continue
          <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
          </svg>
        </button>
      </div>
    </div>
  );
};

export default ScoreStepLoad;
