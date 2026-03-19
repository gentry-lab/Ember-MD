import { Component, For, Show } from 'solid-js';
import { workflowStore } from '../../stores/workflow';
import type { ViewerQueueItem } from '../../stores/workflow';

const ConformStepResults: Component = () => {
  const {
    state,
    setConformStep,
    setMode,
    setViewerPdbPath,
    setViewerPdbQueue,
    setViewerPdbQueueIndex,
    resetViewer,
  } = workflowStore;
  const api = window.electronAPI;

  const conformers = () => state().conform.conformerPaths;

  const handleView3D = () => {
    const paths = conformers();
    if (paths.length === 0) return;

    const queue: ViewerQueueItem[] = paths.map((sdf, i) => ({
      pdbPath: sdf,
      label: `Conformer ${i + 1}`,
      type: 'conformer' as const,
    }));

    resetViewer();
    setViewerPdbPath(queue[0].pdbPath);
    setViewerPdbQueue(queue);
    setViewerPdbQueueIndex(0);
    setMode('viewer');
  };

  const handleOpenFolder = () => {
    const dir = state().conform.outputDir;
    if (dir) api.openFolder(dir);
  };

  return (
    <div class="h-full flex flex-col">
      <div class="text-center mb-3">
        <h2 class="text-xl font-bold">Conformer Results</h2>
        <p class="text-sm text-base-content/90">
          {conformers().length} conformers — {state().conform.ligandName}
        </p>
      </div>

      <div class="flex-1 min-h-0 overflow-auto">
        <div class="card bg-base-200 shadow-lg">
          <div class="card-body p-4">
            <div class="flex items-center justify-between mb-2">
              <h3 class="text-sm font-semibold">Generated Conformers</h3>
              <div class="flex gap-2">
                <button class="btn btn-xs btn-ghost" onClick={handleOpenFolder} title="Open folder">
                  <svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9a2 2 0 00-2 2v5a2 2 0 01-2 2z" />
                  </svg>
                </button>
              </div>
            </div>

            <div class="overflow-x-auto">
              <table class="table table-xs table-zebra w-full">
                <thead>
                  <tr>
                    <th class="w-12">#</th>
                    <th>File</th>
                  </tr>
                </thead>
                <tbody>
                  <For each={conformers()}>
                    {(sdfPath, i) => (
                      <tr>
                        <td class="font-mono text-xs">{i() + 1}</td>
                        <td class="font-mono text-[10px] break-all">{sdfPath.split('/').pop()}</td>
                      </tr>
                    )}
                  </For>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div class="flex justify-between mt-3 flex-shrink-0">
        <button class="btn btn-ghost btn-sm" onClick={() => setConformStep('conform-progress')}>
          <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 17l-5-5m0 0l5-5m-5 5h12" />
          </svg>
          Back
        </button>
        <Show when={conformers().length > 0}>
          <button class="btn btn-primary" onClick={handleView3D}>
            View 3D
            <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
          </button>
        </Show>
      </div>
    </div>
  );
};

export default ConformStepResults;
