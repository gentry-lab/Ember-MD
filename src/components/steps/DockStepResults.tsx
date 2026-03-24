// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, Show, For, createSignal, createMemo, createEffect, onMount, onCleanup, batch } from 'solid-js';
import { workflowStore } from '../../stores/workflow';
import { buildDockingProjectTable, buildDockingViewerQueue } from '../../utils/viewerQueue';
import path from 'path';

type SortField = 'ligandName' | 'vinaAffinity' | 'xtbEnergyKcal' | 'cordialPHighAffinity' | 'cordialPVeryHighAffinity' | 'qed';
type SortDirection = 'asc' | 'desc';

const PAGE_SIZE = 25;

const DockStepResults: Component = () => {
  const { state, openViewerSession, setMode, setMdStep, setMdReceptorPdb, setMdLigandSdf, setMdLigandName, setMdPdbPath, setMdConfig, resetDock } = workflowStore;
  const api = window.electronAPI;

  const results = () => state().dock.results;
  const cordialScored = () => state().dock.cordialScored;
  const hasXtbEnergy = createMemo(() => results().some(r => r.xtbEnergyKcal != null));
  const [sortField, setSortField] = createSignal<SortField>(cordialScored() ? 'cordialPHighAffinity' : 'vinaAffinity');
  const [sortDirection, setSortDirection] = createSignal<SortDirection>(cordialScored() ? 'desc' : 'asc');
  const [selectedIndex, setSelectedIndex] = createSignal<number | null>(0);
  const [currentPage, setCurrentPage] = createSignal(0);
  const [bestOnly, setBestOnly] = createSignal(false);
  const [thumbnailUrl, setThumbnailUrl] = createSignal<string | null>(null);
  const dockedResults = createMemo(() => results().filter(r => !r.isReferencePose));
  const referenceCount = createMemo(() => results().filter(r => r.isReferencePose).length);
  const uniqueLigandCount = createMemo(() => new Set(dockedResults().map(r => r.ligandName)).size);
  const outputDir = () => state().dock.dockingOutputDir;
  const scoreValue = (row: any) => row.vinaAffinity ?? row.vinaScoreOnlyAffinity ?? null;
  const formatScore = (row: any) => {
    const value = scoreValue(row);
    return value == null ? '-' : value.toFixed(1);
  };

  const receptorPdb = () => {
    const dir = outputDir();
    return dir ? path.join(dir, 'inputs', 'receptor.pdb') : '';
  };

  const handleSort = (field: SortField) => {
    if (sortField() === field) {
      setSortDirection(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection(field === 'ligandName' ? 'asc' : field === 'cordialPHighAffinity' || field === 'cordialPVeryHighAffinity' || field === 'qed' ? 'desc' : 'asc');
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
      if (a.isReferencePose !== b.isReferencePose) {
        return a.isReferencePose ? 1 : -1;
      }
      let cmp: number;
      if (field === 'ligandName') {
        cmp = a.ligandName.localeCompare(b.ligandName);
      } else if (field === 'vinaAffinity') {
        const va = scoreValue(a) ?? Number.POSITIVE_INFINITY;
        const vb = scoreValue(b) ?? Number.POSITIVE_INFINITY;
        cmp = va - vb;
      } else {
        const va = (a as any)[field] ?? 0;
        const vb = (b as any)[field] ?? 0;
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

  // Generate thumbnail when selection changes
  // eslint-disable-next-line solid/reactivity
  createEffect(async () => {
    const pose = selectedPose();
    if (!pose) {
      setThumbnailUrl(null);
      return;
    }
    setThumbnailUrl(null);
    const url = await api.generateThumbnail(pose.outputSdf);
    if (selectedPose()?.outputSdf === pose.outputSdf) {
      setThumbnailUrl(url);
    }
  });

  const selectIndex = (idx: number) => {
    setSelectedIndex(idx);
    // Ensure the selected row is on the visible page
    const page = Math.floor(idx / PAGE_SIZE);
    if (page !== currentPage()) setCurrentPage(page);
  };

  const handleRowClick = (globalIndex: number, e: MouseEvent) => {
    if (e.metaKey || e.ctrlKey) {
      // Cmd/Ctrl+click: deselect if already selected, otherwise select
      setSelectedIndex(globalIndex === selectedIndex() ? null : globalIndex);
    } else {
      // Normal click: select (clicking same row keeps it selected)
      setSelectedIndex(globalIndex);
    }
  };

  const handlePrev = () => {
    const idx = selectedIndex();
    if (idx === null) {
      selectIndex(0);
    } else if (idx > 0) {
      selectIndex(idx - 1);
    }
  };

  const handleNext = () => {
    const idx = selectedIndex();
    const max = sortedResults().length - 1;
    if (idx === null) {
      selectIndex(0);
    } else if (idx < max) {
      selectIndex(idx + 1);
    }
  };

  // Keyboard navigation: arrow keys
  const handleKeyDown = (e: KeyboardEvent) => {
    if (state().mode !== 'dock') return;
    if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
      e.preventDefault();
      handlePrev();
    } else if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
      e.preventDefault();
      handleNext();
    } else if (e.key === 'Enter' && selectedPose()) {
      e.preventDefault();
      handleView3D();
    }
  };

  onMount(() => window.addEventListener('keydown', handleKeyDown));
  onCleanup(() => window.removeEventListener('keydown', handleKeyDown));

  const globalIndex = (pageLocalIndex: number) => currentPage() * PAGE_SIZE + pageLocalIndex;

  const handleExportCsv = async () => {
    const dir = outputDir();
    if (!dir) return;
    const csvPath = path.join(dir, 'results', bestOnly() ? 'results_best.csv' : 'results_all.csv');
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
    const receptor = receptorPdb();
    const viewerReceptor = state().dock.receptorPrepared || receptor;

    const allResults = sortedResults();
    const queue = buildDockingViewerQueue(
      viewerReceptor,
      allResults.map((r) => ({
        name: r.ligandName,
        path: r.outputSdf,
        affinity: scoreValue(r) ?? undefined,
      })),
    );

    const selIdx = selectedIndex();
    const projectTable = buildDockingProjectTable({
      familyId: `dock:${state().jobName || 'current'}`,
      title: outputDir()?.split('/').pop() || 'Docking job',
      receptorPdb: viewerReceptor,
      holoPdb: state().dock.receptorPdbPath,
      preparedLigandPath: state().dock.preparedLigandPath || null,
      referenceLigandPath: state().dock.referenceLigandPath,
      poses: allResults,
      poseQueue: queue,
      selectedQueueIndex: selIdx !== null && selIdx >= 0 ? selIdx : 0,
    });
    openViewerSession({
      pdbPath: viewerReceptor,
      ligandPath: pose.outputSdf,
      pdbQueue: queue,
      pdbQueueIndex: selIdx !== null && selIdx >= 0 ? selIdx : 0,
      projectTable,
    });
  };

  const handleSimulate = () => {
    const pose = selectedPose();
    if (!pose) return;
    const receptor = receptorPdb();
    setMdReceptorPdb(receptor);
    setMdLigandSdf(pose.outputSdf);
    setMdLigandName(pose.ligandName);
    setMdPdbPath(receptor || null);
    batch(() => {
      setMode('md');
      setMdStep('md-configure');
    });
  };

  return (
    <div class="h-full flex flex-col">
      {/* Header */}
      <div class="mb-2 flex flex-col gap-2">
        <div class="flex flex-wrap gap-2 self-start">
          <button class="btn btn-ghost btn-xs" onClick={handleOpenFolder}>Open Folder</button>
          <button class="btn btn-ghost btn-xs" onClick={handleExportCsv}>Export CSV</button>
          <button class="btn btn-ghost btn-xs" onClick={handleNewDocking}>New Job</button>
        </div>
        <div class="text-center">
          <h2 class="text-xl font-bold">Docking Complete</h2>
          <p class="text-sm text-base-content/70">
            {dockedResults().length} docked poses from {uniqueLigandCount()} ligands
            <Show when={referenceCount() > 0}>
              {' '}+ {referenceCount()} reference pose
            </Show>
          </p>
        </div>
      </div>

      {/* Main content: table + preview side panel */}
      <div class="flex-1 flex gap-3 min-h-0">
        {/* Left: results table */}
        <div class="flex-1 min-w-0 flex flex-col">
          <div class="flex-1 min-h-0 overflow-auto">
            <table class="table table-xs w-full">
              <thead class="sticky top-0 bg-base-200 z-10">
                <tr>
                  <th class="w-6" />
                  <th class="cursor-pointer select-none text-xs font-semibold" onClick={() => handleSort('ligandName')}>
                    Ligand{sortIndicator('ligandName')}
                  </th>
                  <th class="cursor-pointer select-none text-right text-xs font-semibold w-16" onClick={() => handleSort('vinaAffinity')}>
                    Vina{sortIndicator('vinaAffinity')}
                  </th>
                  <Show when={hasXtbEnergy()}>
                    <th class="cursor-pointer select-none text-right text-xs font-semibold w-16" onClick={() => handleSort('xtbEnergyKcal')}>
                      xTB{sortIndicator('xtbEnergyKcal')}
                    </th>
                  </Show>
                  <Show when={cordialScored()}>
                    <th class="cursor-pointer select-none text-right text-xs font-semibold w-20" onClick={() => handleSort('cordialPHighAffinity')}>
                      {"P(< 1\u00B5M)"}{sortIndicator('cordialPHighAffinity')}
                    </th>
                    <th class="cursor-pointer select-none text-right text-xs font-semibold w-20" onClick={() => handleSort('cordialPVeryHighAffinity')}>
                      {"P(< 100nM)"}{sortIndicator('cordialPVeryHighAffinity')}
                    </th>
                  </Show>
                  <th class="cursor-pointer select-none text-right text-xs font-semibold w-12" onClick={() => handleSort('qed')}>
                    QED{sortIndicator('qed')}
                  </th>
                </tr>
              </thead>
              <tbody>
                <For each={pagedResults()}>
                  {(row, localIdx) => {
                    const gIdx = () => globalIndex(localIdx());
                    return (
                      <tr
                        class={`cursor-pointer hover:bg-base-300 ${selectedIndex() === gIdx() ? 'bg-primary/10' : ''}`}
                        onClick={(e) => handleRowClick(gIdx(), e)}
                      >
                        <td
                          class="cursor-pointer"
                          onClick={(e) => { e.stopPropagation(); setSelectedIndex(gIdx()); }}
                        >
                          <div class={`w-3 h-3 rounded-full border-2 ${
                            selectedIndex() === gIdx()
                              ? 'border-primary bg-primary'
                              : 'border-base-content/30'
                          }`} />
                        </td>
                        <td class="font-mono text-xs truncate max-w-[160px]">
                          {row.ligandName}
                          <Show when={row.isReferencePose}>
                            <span class="badge badge-ghost badge-xs ml-2">Ref</span>
                          </Show>
                        </td>
                        <td class="text-right font-mono text-xs">{formatScore(row)}</td>
                        <Show when={hasXtbEnergy()}>
                          <td class="text-right font-mono text-xs">
                            {row.xtbEnergyKcal != null ? row.xtbEnergyKcal.toFixed(1) : '-'}
                          </td>
                        </Show>
                        <Show when={cordialScored()}>
                          <td class="text-right font-mono text-xs">
                            {row.cordialPHighAffinity != null ? (row.cordialPHighAffinity * 100).toFixed(0) + '%' : '-'}
                          </td>
                          <td class="text-right font-mono text-xs">
                            {row.cordialPVeryHighAffinity != null ? (row.cordialPVeryHighAffinity * 100).toFixed(0) + '%' : '-'}
                          </td>
                        </Show>
                        <td class="text-right font-mono text-xs">{row.qed.toFixed(2)}</td>
                      </tr>
                    );
                  }}
                </For>
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <Show when={totalPages() > 1}>
            <div class="flex justify-center items-center gap-2 mt-1">
              <button class="btn btn-ghost btn-xs" disabled={currentPage() === 0} onClick={() => setCurrentPage(p => p - 1)}>Prev</button>
              <span class="text-xs text-base-content/60 font-medium">
                {currentPage() + 1} / {totalPages()}
              </span>
              <button class="btn btn-ghost btn-xs" disabled={currentPage() >= totalPages() - 1} onClick={() => setCurrentPage(p => p + 1)}>Next</button>
            </div>
          </Show>
        </div>

        {/* Right: compound preview panel */}
        <div class="w-56 flex-shrink-0 flex flex-col">
          <Show
            when={selectedPose()}
            fallback={
              <div class="flex-1 flex items-center justify-center text-xs text-base-content/40 text-center px-4">
                Select a pose to preview
              </div>
            }
          >
            {(pose) => (
              <div class="flex flex-col gap-2 h-full">
                {/* Navigation arrows + counter */}
                <div class="flex items-center justify-between">
                  <button
                    class="btn btn-ghost btn-xs btn-square"
                    disabled={selectedIndex() === 0 || selectedIndex() === null}
                    onClick={handlePrev}
                    title="Previous compound"
                  >
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                    </svg>
                  </button>
                  <span class="text-xs text-base-content/50 font-mono font-medium">
                    {(selectedIndex() ?? 0) + 1} / {sortedResults().length}
                  </span>
                  <button
                    class="btn btn-ghost btn-xs btn-square"
                    disabled={selectedIndex() === null || selectedIndex()! >= sortedResults().length - 1}
                    onClick={handleNext}
                    title="Next compound"
                  >
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                    </svg>
                  </button>
                </div>

                {/* Thumbnail */}
                <div class="bg-base-200 rounded-lg flex items-center justify-center aspect-square">
                  <Show
                    when={thumbnailUrl()}
                    fallback={
                      <div class="text-base-content/30 text-[10px]">
                        <Show when={selectedIndex() !== null} fallback="No preview">
                          <span class="loading loading-spinner loading-sm" />
                        </Show>
                      </div>
                    }
                  >
                    <img src={thumbnailUrl()!} class="w-full h-full object-contain p-2 rounded-lg" alt="2D structure" />
                  </Show>
                </div>

                {/* Scores */}
                <div class="space-y-1">
                  <div class="flex items-center gap-2">
                    <p class="text-xs font-mono font-semibold truncate" title={pose().ligandName}>
                      {pose().ligandName}
                    </p>
                    <Show when={pose().isReferencePose}>
                      <span class="badge badge-ghost badge-xs">Reference</span>
                    </Show>
                  </div>
                  <div class="flex justify-between text-xs">
                    <span class="text-base-content/60">{pose().isReferencePose ? 'Vina score_only' : 'Vina'}</span>
                    <span class="font-mono font-semibold">{formatScore(pose())} {scoreValue(pose()) != null ? 'kcal/mol' : ''}</span>
                  </div>
                  <Show when={pose().refinementEnergy != null}>
                    <div class="flex justify-between text-xs">
                      <span class="text-base-content/60">Refinement</span>
                      <span class="font-mono font-semibold">{pose().refinementEnergy!.toFixed(1)} kcal/mol</span>
                    </div>
                  </Show>
                  <Show when={cordialScored() && pose().cordialPHighAffinity != null}>
                    <div class="flex justify-between text-xs">
                      <span class="text-base-content/60">{"P(< 1\u00B5M)"}</span>
                      <span class="font-mono font-semibold">{((pose().cordialPHighAffinity ?? 0) * 100).toFixed(0)}%</span>
                    </div>
                    <div class="flex justify-between text-xs">
                      <span class="text-base-content/60">{"P(< 100nM)"}</span>
                      <span class="font-mono font-semibold">{((pose().cordialPVeryHighAffinity ?? 0) * 100).toFixed(0)}%</span>
                    </div>
                  </Show>
                  <div class="flex justify-between text-xs">
                    <span class="text-base-content/60">QED</span>
                    <span class="font-mono font-semibold">{pose().qed.toFixed(2)}</span>
                  </div>
                </div>

                {/* Actions */}
                <div class="mt-auto flex flex-col gap-1.5">
                  <button class="btn btn-primary btn-sm w-full" onClick={handleView3D}>
                    View 3D
                  </button>
                  <button class="btn btn-outline btn-sm w-full" onClick={handleSimulate}>
                    Simulate
                  </button>
                </div>
              </div>
            )}
          </Show>
        </div>
      </div>

      {/* Scoring legend */}
      <div class="mt-2 pt-2 border-t border-base-300">
        <div class="text-xs text-base-content/50 leading-relaxed max-w-[420px]">
          <span class="font-semibold text-base-content/70">Vina</span> = docked affinity
          {' '}<span class="mx-1 text-base-content/30">|</span>{' '}
          <span class="font-semibold text-base-content/70">Vina score_only</span> = reference-pose rescoring in the prepared pocket
          <Show when={cordialScored()}>
            {' '}<span class="mx-1 text-base-content/30">|</span>{' '}
            <span class="font-semibold text-base-content/70">{"P(< 1\u00B5M)"}</span> / <span class="font-semibold text-base-content/70">{"P(< 100nM)"}</span> = ML-predicted binding probability (CORDIAL)
          </Show>
          {' '}<span class="mx-1 text-base-content/30">|</span>{' '}
          <span class="font-semibold text-base-content/70">QED</span> = drug-likeness (0-1)
        </div>
      </div>
    </div>
  );
};

export default DockStepResults;
