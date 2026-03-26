// Copyright (c) 2026 Ember Contributors. MIT License.
import path from 'path';
import { Component, For, Show } from 'solid-js';
import { workflowStore } from '../../stores/workflow';

const XrayStepResults: Component = () => {
  const { state, resetXray } = workflowStore;
  const api = window.electronAPI;

  const result = () => state().xray.result;
  const pdfPaths = () => result()?.pdfPaths || [];

  return (
    <div class="h-full flex flex-col">
      <div class="flex items-center justify-between mb-3">
        <div>
          <h2 class="text-xl font-bold">X-ray Analyzer Results</h2>
          <p class="text-sm text-base-content/90">
            {pdfPaths().length} report{pdfPaths().length === 1 ? '' : 's'} generated
          </p>
        </div>
        <button
          class="btn btn-xs btn-ghost"
          onClick={() => result()?.outputDir && api.openFolder(result()!.outputDir)}
          disabled={!result()?.outputDir}
        >
          Open Folder
        </button>
      </div>

      <div class="flex-1 min-h-0 overflow-auto">
        <div class="card bg-base-200 shadow-lg">
          <div class="card-body p-4 space-y-4">
            <Show when={result()} fallback={
              <div class="rounded-lg border border-warning/30 bg-warning/10 p-3 text-sm">
                No X-ray analysis results are available.
              </div>
            }>
              <div class="grid gap-3 md:grid-cols-2">
                <div class="rounded-lg border border-base-300 bg-base-100/70 p-3">
                  <div class="text-[10px] uppercase tracking-wide text-base-content/55">Input folder</div>
                  <div class="mt-1 text-xs font-mono break-all">{result()?.inputDir}</div>
                </div>
                <div class="rounded-lg border border-base-300 bg-base-100/70 p-3">
                  <div class="text-[10px] uppercase tracking-wide text-base-content/55">Results folder</div>
                  <div class="mt-1 text-xs font-mono break-all">{result()?.outputDir}</div>
                </div>
              </div>

              <Show when={pdfPaths().length > 0} fallback={
                <div class="rounded-lg border border-warning/30 bg-warning/10 p-3 text-sm">
                  Analysis finished but no PDF reports were found in the output folder.
                </div>
              }>
                <div class="space-y-2">
                  <div class="text-xs font-semibold uppercase tracking-wide text-base-content/60">Reports</div>
                  <div class="space-y-2">
                    <For each={pdfPaths()}>{(pdfPath, index) => (
                      <button
                        class="w-full rounded-lg border border-base-300 bg-base-100/80 px-3 py-2 text-left hover:bg-base-100"
                        onClick={() => api.openFolder(pdfPath)}
                      >
                        <div class="flex items-center justify-between gap-3">
                          <div class="min-w-0">
                            <div class="text-xs font-semibold truncate">
                              {index() + 1}. {path.basename(pdfPath)}
                            </div>
                            <div class="text-[10px] text-base-content/55 truncate">{pdfPath}</div>
                          </div>
                          <span class="text-[10px] font-semibold text-primary">Reveal</span>
                        </div>
                      </button>
                    )}</For>
                  </div>
                </div>
              </Show>
            </Show>
          </div>
        </div>
      </div>

      <div class="flex justify-between mt-3">
        <button class="btn btn-ghost btn-sm" onClick={() => resetXray()}>
          New Job
        </button>
      </div>
    </div>
  );
};

export default XrayStepResults;
