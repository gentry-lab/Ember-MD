import { Component, For, Show, createMemo } from 'solid-js';
import { workflowStore } from '../../stores/workflow';
import { buildConformerProjectTable, buildConformerViewerQueue } from '../../utils/viewerQueue';

const ConformStepResults: Component = () => {
  const {
    state,
    setConformStep,
    openViewerSession,
    resetConform,
  } = workflowStore;
  const api = window.electronAPI;

  const conformers = () => state().conform.conformerPaths;
  const energies = () => state().conform.conformerEnergies;
  const method = () => state().conform.config.method;

  const hasEnergies = createMemo(() => Object.keys(energies()).length > 0);

  const energyForPath = (sdfPath: string): number | null => {
    const e = energies();
    // Try exact path match first, then basename match
    if (sdfPath in e) return e[sdfPath];
    for (const key of Object.keys(e)) {
      if (key.endsWith('/' + sdfPath.split('/').pop())) return e[key];
    }
    return null;
  };

  const handleView3D = () => {
    const paths = conformers();
    if (paths.length === 0) return;
    const queue = buildConformerViewerQueue(paths);
    const projectTable = buildConformerProjectTable({
      familyId: `conform:${state().jobName || 'current'}`,
      title: state().conform.outputName || state().conform.ligandName || 'Conformer job',
      inputPath: state().conform.ligandSdfPath,
      conformerPaths: paths,
      conformerEnergies: energies(),
    });
    openViewerSession({
      pdbPath: queue[0].pdbPath,
      pdbQueue: queue,
      pdbQueueIndex: 0,
      projectTable,
    });
  };

  const handleOpenFolder = () => {
    const dir = state().conform.outputDir;
    if (dir) api.openFolder(dir);
  };

  const methodLabel = () => {
    const m = method();
    if (m === 'crest') return 'CREST (GFN2-xTB)';
    if (m === 'mcmm') return 'MCMM (Sage 2.3.0)';
    return 'ETKDG';
  };

  return (
    <div class="h-full flex flex-col">
      <div class="text-center mb-3">
        <h2 class="text-xl font-bold">Conformer Results</h2>
        <p class="text-sm text-base-content/90">
          {conformers().length} conformer{conformers().length !== 1 ? 's' : ''} via {methodLabel()} — {state().conform.outputName || state().conform.ligandName}
        </p>
      </div>

      <div class="flex-1 min-h-0 overflow-auto">
        <div class="card bg-base-200 shadow-lg">
          <div class="card-body p-4">
            <div class="flex items-center justify-between mb-2">
              <h3 class="text-sm font-semibold">Generated Conformers</h3>
              <div class="flex gap-2">
                <button class="btn btn-xs btn-ghost" onClick={handleOpenFolder} title="Open folder">
                  Open Folder
                </button>
              </div>
            </div>

            <div class="overflow-x-auto">
              <table class="table table-xs table-zebra w-full">
                <thead>
                  <tr>
                    <th class="w-12">#</th>
                    <th>File</th>
                    <Show when={hasEnergies()}>
                      <th class="text-right w-28">Energy (kcal/mol)</th>
                    </Show>
                  </tr>
                </thead>
                <tbody>
                  <For each={conformers()}>
                    {(sdfPath, i) => {
                      const energy = energyForPath(sdfPath);
                      return (
                        <tr>
                          <td class="font-mono text-xs">{i() + 1}</td>
                          <td class="font-mono text-[10px] break-all">{sdfPath.split('/').pop()}</td>
                          <Show when={hasEnergies()}>
                            <td class="text-right font-mono text-xs">
                              {energy != null ? (
                                i() === 0 && energy === 0 ? (
                                  <span class="text-success">0.0 (min)</span>
                                ) : (
                                  <span class={energy > 5 ? 'text-warning' : ''}>{energy.toFixed(2)}</span>
                                )
                              ) : '-'}
                            </td>
                          </Show>
                        </tr>
                      );
                    }}
                  </For>
                </tbody>
              </table>
            </div>

            <Show when={hasEnergies()}>
              <p class="text-[10px] text-base-content/50 mt-2">
                Relative to lowest-energy conformer (kcal/mol).
                {method() === 'crest'
                  ? ' GFN2-xTB + ALPB solvation.'
                  : state().conform.config.xtbRerank
                    ? ' GFN2-xTB + ALPB solvation (reranked).'
                    : method() === 'mcmm'
                      ? ' Sage 2.3.0 + OBC2 implicit solvent.'
                      : ' MMFF94s.'}
              </p>
            </Show>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div class="flex justify-between mt-3 flex-shrink-0">
        <button class="btn btn-ghost btn-sm" onClick={() => resetConform()}>
          New Job
        </button>
        <Show when={conformers().length > 0}>
          <button class="btn btn-primary" onClick={handleView3D}>
            View 3D
          </button>
        </Show>
      </div>
    </div>
  );
};

export default ConformStepResults;
