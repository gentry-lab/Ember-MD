// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, For, Show } from 'solid-js';
import { workflowStore } from '../../stores/workflow';

const ScoreStepResults: Component = () => {
  const { state, resetScore } = workflowStore;
  const api = window.electronAPI;

  const handleOpenOutput = () => {
    if (state().score.outputDir) {
      api.openFolder(state().score.outputDir);
    }
  };

  const handleOpenInput = () => {
    if (state().score.inputDir) {
      api.openFolder(state().score.inputDir);
    }
  };

  const handleOpenPdf = (pdfPath: string) => {
    api.openFolder(pdfPath);
  };

  return (
    <div class="h-full flex flex-col">
      <div class="text-center mb-3">
        <h2 class="text-xl font-bold">X-ray Pose Results</h2>
        <p class="text-sm text-base-content/90">
          {state().score.pdfPaths.length} PDF report{state().score.pdfPaths.length === 1 ? '' : 's'} generated
        </p>
      </div>

      <div class="flex-1 min-h-0 overflow-auto">
        <div class="card bg-base-200 shadow-lg">
          <div class="card-body p-4 space-y-4">
            <div class="grid gap-3 md:grid-cols-2">
              <div class="rounded-lg bg-base-300 p-3">
                <p class="text-[10px] uppercase tracking-wide text-base-content/60 mb-1">Input Folder</p>
                <p class="text-xs font-semibold">{state().score.inputDir?.split('/').pop() || 'Unknown'}</p>
                <p class="text-[10px] font-mono text-base-content/70 break-all">{state().score.inputDir}</p>
              </div>
              <div class="rounded-lg bg-base-300 p-3">
                <p class="text-[10px] uppercase tracking-wide text-base-content/60 mb-1">Output Folder</p>
                <p class="text-xs font-semibold">{state().score.outputDir?.split('/').pop() || 'Unknown'}</p>
                <p class="text-[10px] font-mono text-base-content/70 break-all">{state().score.outputDir}</p>
              </div>
            </div>

            <div class="flex gap-2">
              <button class="btn btn-xs btn-ghost" onClick={handleOpenInput} disabled={!state().score.inputDir}>
                Open Input Folder
              </button>
              <button class="btn btn-xs btn-ghost" onClick={handleOpenOutput} disabled={!state().score.outputDir}>
                Open Output Folder
              </button>
            </div>

            <Show
              when={state().score.pdfPaths.length > 0}
              fallback={
                <div class="rounded-lg border border-warning/30 bg-warning/10 p-3 text-sm">
                  The analyzer completed, but no PDFs were generated.
                </div>
              }
            >
              <div class="overflow-x-auto">
                <table class="table table-xs table-zebra w-full">
                  <thead>
                    <tr>
                      <th class="w-12">#</th>
                      <th>PDF</th>
                      <th class="w-24 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    <For each={state().score.pdfPaths}>
                      {(pdfPath, i) => (
                        <tr>
                          <td class="font-mono text-xs">{i() + 1}</td>
                          <td class="font-mono text-[10px] break-all">{pdfPath.split('/').pop()}</td>
                          <td class="text-right">
                            <button class="btn btn-xs btn-primary" onClick={() => handleOpenPdf(pdfPath)}>
                              Open PDF
                            </button>
                          </td>
                        </tr>
                      )}
                    </For>
                  </tbody>
                </table>
              </div>
            </Show>
          </div>
        </div>
      </div>

      <div class="flex justify-between mt-3 flex-shrink-0">
        <button class="btn btn-ghost btn-sm" onClick={() => resetScore()}>
          New Job
        </button>
      </div>
    </div>
  );
};

export default ScoreStepResults;
