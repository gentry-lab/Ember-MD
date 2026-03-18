import { Component, Show, For, createSignal, createMemo } from 'solid-js';
import { workflowStore } from '../../stores/workflow';
import { DockResult } from '../../../shared/types/dock';
import path from 'path';

type SortField = 'ligandName' | 'vinaAffinity' | 'cordialExpectedPkd' | 'cordialPHighAffinity' | 'qed' | 'coreRmsd';
type SortDirection = 'asc' | 'desc';

const PAGE_SIZE = 25;

const DockStepResults: Component = () => {
  const { state, setMode, setMdStep, setViewerPdbPath, setMdReceptorPdb, setMdLigandSdf, setMdLigandName, setMdConfig, resetDock } = workflowStore;
  const api = window.electronAPI;

  const [sortField, setSortField] = createSignal<SortField>('vinaAffinity');
  const [sortDirection, setSortDirection] = createSignal<SortDirection>('asc');
  const [selectedIndex, setSelectedIndex] = createSignal<number | null>(null);
  const [currentPage, setCurrentPage] = createSignal(0);
  const [bestOnly, setBestOnly] = createSignal(false);

  const results = () => state().dock.results;
  const uniqueLigandCount = createMemo(() => new Set(results().map(r => r.ligandName)).size);
  const outputDir = () => state().dock.dockingOutputDir;
  const coreConstrained = () => state().dock.config.coreConstrained;
  const cordialScored = () => state().dock.cordialScored;

  const receptorPdb = () => {
    const dir = outputDir();
    return dir ? path.join(dir, 'receptor_prepared.pdb') : '';
  };

  const handleSort = (field: SortField) => {
    if (sortField() === field) {
      setSortDirection(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      // Default sort direction per field type
      setSortDirection(field === 'ligandName' ? 'asc' : field === 'cordialExpectedPkd' || field === 'cordialPHighAffinity' || field === 'qed' ? 'desc' : 'asc');
    }
    setCurrentPage(0);
  };

  const sortIndicator = (field: SortField) => {
    if (sortField() !== field) return '';
    return sortDirection() === 'asc' ? ' \u25B2' : ' \u25BC';
  };

  const sortedResults = createMemo(() => {
    const items = [...results()];
    const field = sortField();
    const dir = sortDirection();

    items.sort((a, b) => {
      let cmp = 0;
      if (field === 'ligandName') {
        cmp = a.ligandName.localeCompare(b.ligandName);
      } else {
        const va = a[field] ?? 0;
        const vb = b[field] ?? 0;
        cmp = (va as number) - (vb as number);
      }
      return dir === 'asc' ? cmp : -cmp;
    });

    return items;
  });

  const totalPages = createMemo(() => Math.max(1, Math.ceil(sortedResults().length / PAGE_SIZE)));

  const pagedResults = createMemo(() => {
    const start = currentPage() * PAGE_SIZE;
    return sortedResults().slice(start, start + PAGE_SIZE);
  });

  const selectedPose = createMemo(() => {
    const idx = selectedIndex();
    if (idx === null) return null;
    return sortedResults()[idx] ?? null;
  });

  const handleRowClick = (globalIndex: number) => {
    setSelectedIndex(globalIndex === selectedIndex() ? null : globalIndex);
  };

  const globalIndex = (pageLocalIndex: number) => currentPage() * PAGE_SIZE + pageLocalIndex;

  const handleExportCsv = async () => {
    const dir = outputDir();
    if (!dir) return;
    const csvPath = path.join(dir, bestOnly() ? 'results_best.csv' : 'results_all.csv');
    await api.exportDockCsv(dir, csvPath, bestOnly());
  };

  const handleOpenFolder = () => {
    const dir = outputDir();
    if (dir) api.openFolder(dir);
  };

  const handleNewDocking = () => {
    resetDock();
  };

  const handleView3D = () => {
    const pose = selectedPose();
    if (!pose) return;
    const { setViewerLigandPath } = workflowStore;
    setViewerPdbPath(receptorPdb());
    setViewerLigandPath(pose.outputSdf);
    setMode('viewer');
  };

  const handleSimulate = () => {
    const pose = selectedPose();
    if (!pose) return;
    setMdReceptorPdb(receptorPdb());
    setMdLigandSdf(pose.outputSdf);
    setMdLigandName(pose.ligandName);
    setMdConfig({ restrainLigandNs: 2 });
    setMode('md');
    setMdStep('md-configure');
  };

  return (
    <div class="h-full flex flex-col">
      {/* Title */}
      <div class="text-center mb-3">
        <h2 class="text-xl font-bold">Docking Complete</h2>
        <p class="text-sm text-base-content/90">
          {results().length} poses from {uniqueLigandCount()} ligands
        </p>
      </div>

      {/* Results table */}
      <div class="flex-1 min-h-0 overflow-auto">
        <table class="table table-xs w-full">
          <thead class="sticky top-0 bg-base-200 z-10">
            <tr>
              <th class="w-8"></th>
              <th class="cursor-pointer select-none" onClick={() => handleSort('ligandName')}>
                Ligand{sortIndicator('ligandName')}
              </th>
              <th class="cursor-pointer select-none text-right" onClick={() => handleSort('vinaAffinity')}>
                Vina (kcal/mol){sortIndicator('vinaAffinity')}
              </th>
              <Show when={cordialScored()}>
                <th class="cursor-pointer select-none text-right" onClick={() => handleSort('cordialExpectedPkd')}>
                  CORDIAL pKd{sortIndicator('cordialExpectedPkd')}
                </th>
                <th class="cursor-pointer select-none text-right" onClick={() => handleSort('cordialPHighAffinity')}>
                  {"P(\u22656)"}{sortIndicator('cordialPHighAffinity')}
                </th>
              </Show>
              <th class="cursor-pointer select-none text-right" onClick={() => handleSort('qed')}>
                QED{sortIndicator('qed')}
              </th>
              <Show when={coreConstrained()}>
                <th class="cursor-pointer select-none text-right" onClick={() => handleSort('coreRmsd')}>
                  Core RMSD (A){sortIndicator('coreRmsd')}
                </th>
              </Show>
            </tr>
          </thead>
          <tbody>
            <For each={pagedResults()}>
              {(row, localIdx) => {
                const gIdx = () => globalIndex(localIdx());
                return (
                  <tr
                    class={`cursor-pointer hover:bg-base-300 ${selectedIndex() === gIdx() ? 'bg-primary/10' : ''}`}
                    onClick={() => handleRowClick(gIdx())}
                  >
                    <td>
                      <input
                        type="radio"
                        class="radio radio-xs radio-primary"
                        checked={selectedIndex() === gIdx()}
                        onChange={() => handleRowClick(gIdx())}
                      />
                    </td>
                    <td class="font-mono text-xs truncate max-w-[200px]">{row.ligandName}</td>
                    <td class="text-right font-mono text-xs">{row.vinaAffinity.toFixed(1)}</td>
                    <Show when={cordialScored()}>
                      <td class="text-right font-mono text-xs">
                        {row.cordialExpectedPkd != null ? row.cordialExpectedPkd.toFixed(2) : '-'}
                      </td>
                      <td class="text-right font-mono text-xs">
                        {row.cordialPHighAffinity != null ? (row.cordialPHighAffinity * 100).toFixed(0) + '%' : '-'}
                      </td>
                    </Show>
                    <td class="text-right font-mono text-xs">{row.qed.toFixed(2)}</td>
                    <Show when={coreConstrained()}>
                      <td class="text-right font-mono text-xs">
                        {row.coreRmsd != null ? row.coreRmsd.toFixed(2) : '-'}
                      </td>
                    </Show>
                  </tr>
                );
              }}
            </For>
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <Show when={totalPages() > 1}>
        <div class="flex justify-center items-center gap-2 mt-2">
          <button
            class="btn btn-ghost btn-xs"
            disabled={currentPage() === 0}
            onClick={() => setCurrentPage(p => p - 1)}
          >
            Prev
          </button>
          <span class="text-xs text-base-content/70">
            Page {currentPage() + 1} of {totalPages()}
          </span>
          <button
            class="btn btn-ghost btn-xs"
            disabled={currentPage() >= totalPages() - 1}
            onClick={() => setCurrentPage(p => p + 1)}
          >
            Next
          </button>
        </div>
      </Show>

      {/* Action buttons */}
      <div class="flex justify-between items-center mt-3">
        <div class="flex gap-2 items-center">
          <button class="btn btn-outline btn-sm" onClick={handleExportCsv}>
            <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Export CSV
          </button>
          <label class="label cursor-pointer gap-1">
            <input
              type="checkbox"
              class="checkbox checkbox-xs"
              checked={bestOnly()}
              onChange={e => setBestOnly((e.target as HTMLInputElement).checked)}
            />
            <span class="text-xs text-base-content/70">Best per ligand</span>
          </label>
        </div>
        <div class="flex gap-2">
          <button class="btn btn-outline btn-sm" onClick={handleOpenFolder}>
            <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9a2 2 0 00-2 2v5a2 2 0 01-2 2z" />
            </svg>
            Open Folder
          </button>
          <button class="btn btn-primary btn-sm" onClick={handleNewDocking}>
            New Docking
            <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
            </svg>
          </button>
        </div>
      </div>

      {/* Selected pose actions bar */}
      <Show when={selectedPose()}>
        {(pose) => (
        <div class="flex justify-center gap-3 mt-3 p-2 bg-base-200 rounded-lg">
          <span class="text-xs text-base-content/70 self-center">
            Selected: <span class="font-mono font-medium">{pose().ligandName}</span>
            {' '}({pose().vinaAffinity.toFixed(1)} kcal/mol)
          </span>
          <button class="btn btn-outline btn-sm" onClick={handleView3D}>
            <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
            View 3D
          </button>
          <button class="btn btn-primary btn-sm" onClick={handleSimulate}>
            <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Simulate
          </button>
        </div>
        )}
      </Show>
    </div>
  );
};

export default DockStepResults;
