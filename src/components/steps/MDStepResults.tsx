// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, Show, For, createSignal, createMemo, onMount, onCleanup, batch } from 'solid-js';
import { workflowStore, ViewerQueueItem } from '../../stores/workflow';
import path from 'path';
import MDTorsionPanel from './MDTorsionPanel';
import { buildMdProjectTable } from '../../utils/viewerQueue';

type SortField = 'clusterId' | 'population' | 'vinaRescore' | 'cordialPHighAffinity' | 'cordialPVeryHighAffinity';
type SortDirection = 'asc' | 'desc';
type ResultsTab = 'clusters' | 'dihedrals';

type DisplayCluster = {
  clusterId: number;
  frameCount: number;
  population: number;
  centroidFrame: number;
  centroidPdbPath?: string;
  vinaRescore?: number;
  cordialExpectedPkd?: number;
  cordialPHighAffinity?: number;
  cordialPVeryHighAffinity?: number;
};

const MDStepResults: Component = () => {
  const { state, openViewerSession, addViewerProjectFamily, resetMd } = workflowStore;
  const api = window.electronAPI;

  const [sortField, setSortField] = createSignal<SortField>('population');
  const [sortDirection, setSortDirection] = createSignal<SortDirection>('desc');
  const [selectedIndex, setSelectedIndex] = createSignal<number | null>(0);
  const [activeTab, setActiveTab] = createSignal<ResultsTab>('clusters');

  const result = () => state().md.result;
  const systemInfo = () => state().md.systemInfo;
  const clusteringResults = () => state().md.clusteringResults;
  const clusterScores = () => state().md.clusterScores;
  const torsionAnalysis = () => state().md.torsionAnalysis;

  const clusters = createMemo<DisplayCluster[]>(() => {
    const scoreMap = new Map(clusterScores().map(cluster => [cluster.clusterId, cluster]));
    const baseClusters = clusteringResults()?.clusters;
    if (baseClusters && baseClusters.length > 0) {
      return baseClusters.map((cluster) => ({
        ...cluster,
        ...scoreMap.get(cluster.clusterId),
        centroidPdbPath: cluster.centroidPdbPath || scoreMap.get(cluster.clusterId)?.centroidPdbPath,
      }));
    }
    return clusterScores();
  });

  const hasVina = createMemo(() => clusters().some(c => c.vinaRescore != null));
  const hasCordial = createMemo(() => clusters().some(c => c.cordialPHighAffinity != null));

  const handleSort = (field: SortField) => {
    if (sortField() === field) {
      setSortDirection(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection(
        field === 'clusterId' ? 'asc'
          : field === 'vinaRescore' ? 'asc'
          : 'desc'
      );
    }
  };

  const sortIndicator = (field: SortField) => {
    if (sortField() !== field) return '';
    return sortDirection() === 'asc' ? ' \u25B2' : ' \u25BC';
  };

  const sortedClusters = createMemo(() => {
    const items = [...clusters()];
    const field = sortField();
    const dir = sortDirection();

    items.sort((a, b) => {
      let va: number | null = null;
      let vb: number | null = null;

      switch (field) {
        case 'clusterId': va = a.clusterId; vb = b.clusterId; break;
        case 'population': va = a.population; vb = b.population; break;
        case 'vinaRescore': va = a.vinaRescore ?? null; vb = b.vinaRescore ?? null; break;
        case 'cordialPHighAffinity': va = a.cordialPHighAffinity ?? null; vb = b.cordialPHighAffinity ?? null; break;
        case 'cordialPVeryHighAffinity': va = a.cordialPVeryHighAffinity ?? null; vb = b.cordialPVeryHighAffinity ?? null; break;
      }

      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      return dir === 'asc' ? va - vb : vb - va;
    });

    return items;
  });

  const selectedCluster = createMemo(() => {
    const idx = selectedIndex();
    if (idx == null) return null;
    return sortedClusters()[idx] ?? null;
  });

  const selectIndex = (idx: number) => {
    setSelectedIndex(idx);
  };

  const handleRowClick = (index: number, e: MouseEvent) => {
    if (e.metaKey || e.ctrlKey) {
      setSelectedIndex(index === selectedIndex() ? null : index);
    } else {
      setSelectedIndex(index);
    }
  };

  const handlePrev = () => {
    const idx = selectedIndex();
    if (idx === null) { selectIndex(0); }
    else if (idx > 0) { selectIndex(idx - 1); }
  };

  const handleNext = () => {
    const idx = selectedIndex();
    const max = sortedClusters().length - 1;
    if (idx === null) { selectIndex(0); }
    else if (idx < max) { selectIndex(idx + 1); }
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    if (state().mode !== 'md' || activeTab() !== 'clusters') return;
    if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
      e.preventDefault();
      handlePrev();
    } else if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
      e.preventDefault();
      handleNext();
    } else if (e.key === 'Enter' && selectedCluster()) {
      e.preventDefault();
      openClusterInViewer(selectedCluster()!);
    }
  };

  onMount(() => window.addEventListener('keydown', handleKeyDown));
  onCleanup(() => window.removeEventListener('keydown', handleKeyDown));

  /* ── Navigation helpers ── */
  const getSimulationRunRoot = () => {
    if (!result()) return null;
    const trajectoryDir = path.dirname(result()!.trajectoryPath);
    return path.basename(trajectoryDir) === 'results' ? path.dirname(trajectoryDir) : trajectoryDir;
  };

  const openTrajectoryInViewer = () => {
    if (!result()) return;
    const projectTable = buildMdProjectTable({
      familyId: `md:${state().jobName || 'current'}`,
      title: getSimulationRunRoot()?.split('/').pop() || 'Simulation job',
      systemPdb: result()!.systemPdbPath,
      trajectoryPath: result()!.trajectoryPath,
      queueBackedClusters: false,
      clusters: sortedClusters(),
    });
    openViewerSession({
      pdbPath: result()!.systemPdbPath,
      trajectoryPath: result()!.trajectoryPath,
    });
    addViewerProjectFamily(projectTable.families[0], projectTable.rows);
  };

  const openClusterInViewer = (cluster: DisplayCluster) => {
    if (!cluster.centroidPdbPath) return;
    const queue: ViewerQueueItem[] = sortedClusters()
      .filter(c => c.centroidPdbPath)
      .map(c => ({
        pdbPath: c.centroidPdbPath!,
        label: `Cluster ${c.clusterId + 1} (${c.population.toFixed(0)}%)`,
      }));
    const queueIndex = queue.findIndex(q => q.pdbPath === cluster.centroidPdbPath);
    const projectTable = buildMdProjectTable({
      familyId: `md:${state().jobName || 'current'}`,
      title: getSimulationRunRoot()?.split('/').pop() || 'Simulation job',
      systemPdb: result()!.systemPdbPath,
      trajectoryPath: result()?.trajectoryPath,
      queueBackedClusters: true,
      clusters: sortedClusters(),
    });
    const activeRowId = projectTable.rows.find((row) => row.queueIndex === (queueIndex >= 0 ? queueIndex : 0))?.id
      ?? projectTable.activeRowId;
    openViewerSession({
      pdbPath: cluster.centroidPdbPath,
      pdbQueue: queue,
      pdbQueueIndex: queueIndex >= 0 ? queueIndex : 0,
    });
    addViewerProjectFamily(projectTable.families[0], projectTable.rows);
    workflowStore.setViewerProjectActiveRow(activeRowId);
  };

  const openAllClustersInViewer = () => {
    const withPdb = sortedClusters().filter(c => c.centroidPdbPath);
    if (withPdb.length === 0) return;
    const queue: ViewerQueueItem[] = withPdb.map(c => ({
      pdbPath: c.centroidPdbPath!,
      label: `Cluster ${c.clusterId + 1} (${c.population.toFixed(0)}%)`,
    }));
    const projectTable = buildMdProjectTable({
      familyId: `md:${state().jobName || 'current'}`,
      title: getSimulationRunRoot()?.split('/').pop() || 'Simulation job',
      systemPdb: result()!.systemPdbPath,
      trajectoryPath: result()?.trajectoryPath,
      queueBackedClusters: true,
      clusters: sortedClusters(),
    });
    const activeRowId = projectTable.rows.find((row) => row.queueIndex === 0)?.id ?? projectTable.activeRowId;
    openViewerSession({
      pdbPath: queue[0].pdbPath,
      pdbQueue: queue,
      pdbQueueIndex: 0,
    });
    addViewerProjectFamily(projectTable.families[0], projectTable.rows);
    workflowStore.setViewerProjectActiveRow(activeRowId);
  };

  const handleOpenFolder = () => {
    const dir = getSimulationRunRoot();
    if (dir) api.openFolder(dir);
  };

  const openReport = () => {
    const runRoot = getSimulationRunRoot();
    if (!runRoot) return;
    const reportPath = path.join(runRoot, 'analysis', 'full_report.pdf');
    api.openFolder(reportPath);
  };

  const handleCopyTable = async () => {
    const lines = sortedClusters().map((c) =>
      [
        c.clusterId + 1,
        c.population.toFixed(1) + '%',
        c.frameCount,
        c.centroidFrame,
        c.vinaRescore != null ? c.vinaRescore.toFixed(1) : '-',
        c.cordialPHighAffinity != null ? (c.cordialPHighAffinity * 100).toFixed(0) + '%' : '-',
        c.cordialPVeryHighAffinity != null ? (c.cordialPVeryHighAffinity * 100).toFixed(0) + '%' : '-',
      ].join('\t')
    );
    const header = 'Cluster\tPop%\tFrames\tCentroid\tVina\tP(<1uM)\tP(<100nM)';
    try {
      await navigator.clipboard.writeText([header, ...lines].join('\n'));
    } catch { /* noop */ }
  };

  return (
    <div class="h-full flex flex-col">
      {/* ── Header ── */}
      <div class="mb-2 flex flex-col gap-2">
        <div class="flex flex-wrap gap-2 self-start">
          <button class="btn btn-ghost btn-xs" onClick={handleOpenFolder}>Open Folder</button>
          <button class="btn btn-ghost btn-xs" onClick={openReport}>Open Report</button>
          <button class="btn btn-ghost btn-xs" onClick={handleCopyTable}>Copy Table</button>
          <button class="btn btn-ghost btn-xs" onClick={() => resetMd()}>New Job</button>
        </div>
        <div class="text-center">
          <h2 class="text-xl font-bold">Simulation Complete</h2>
          <p class="text-sm text-base-content/70">
            {clusters().length} clusters from {state().md.config.productionNs} ns
            <Show when={systemInfo()}>
              {' '}&bull; {systemInfo()!.atomCount.toLocaleString()} atoms
            </Show>
            <Show when={state().md.benchmarkResult}>
              {' '}&bull; {state().md.benchmarkResult!.nsPerDay.toFixed(1)} ns/day
            </Show>
          </p>
        </div>
      </div>

      {/* ── Tab bar (only if torsion data exists) ── */}
      <Show when={torsionAnalysis()}>
        <div class="mb-2 flex justify-center">
          <div role="tablist" class="tabs tabs-boxed tabs-sm">
            <button
              class={`tab ${activeTab() === 'clusters' ? 'tab-active' : ''}`}
              onClick={() => setActiveTab('clusters')}
            >
              Clusters
            </button>
            <button
              class={`tab ${activeTab() === 'dihedrals' ? 'tab-active' : ''}`}
              onClick={() => setActiveTab('dihedrals')}
            >
              Dihedrals
            </button>
          </div>
        </div>
      </Show>

      {/* ── Clusters tab ── */}
      <Show when={activeTab() === 'clusters'}>
        <Show
          when={clusters().length > 0}
          fallback={
            <div class="flex-1 flex items-center justify-center text-base-content/50 text-sm">
              No clusters available
            </div>
          }
        >
          <div class="flex-1 flex gap-3 min-h-0">
            {/* Left: cluster table */}
            <div class="flex-1 min-w-0 flex flex-col">
              <div class="flex-1 min-h-0 overflow-auto">
                <table class="table table-xs w-full">
                  <thead class="sticky top-0 bg-base-200 z-10">
                    <tr>
                      <th class="w-6" />
                      <th class="cursor-pointer select-none text-xs font-semibold" onClick={() => handleSort('clusterId')}>
                        Cluster{sortIndicator('clusterId')}
                      </th>
                      <th class="cursor-pointer select-none text-right text-xs font-semibold w-16" onClick={() => handleSort('population')}>
                        Pop%{sortIndicator('population')}
                      </th>
                      <Show when={hasVina()}>
                        <th class="cursor-pointer select-none text-right text-xs font-semibold w-16" onClick={() => handleSort('vinaRescore')}>
                          Vina{sortIndicator('vinaRescore')}
                        </th>
                      </Show>
                      <Show when={hasCordial()}>
                        <th class="cursor-pointer select-none text-right text-xs font-semibold w-20" onClick={() => handleSort('cordialPHighAffinity')}>
                          {"P(< 1\u00B5M)"}{sortIndicator('cordialPHighAffinity')}
                        </th>
                        <th class="cursor-pointer select-none text-right text-xs font-semibold w-20" onClick={() => handleSort('cordialPVeryHighAffinity')}>
                          {"P(< 100nM)"}{sortIndicator('cordialPVeryHighAffinity')}
                        </th>
                      </Show>
                    </tr>
                  </thead>
                  <tbody>
                    <For each={sortedClusters()}>
                      {(cluster, index) => (
                        <tr
                          class={`cursor-pointer hover:bg-base-300 ${
                            selectedIndex() === index() ? 'bg-primary/10' : ''
                          }`}
                          onClick={(e) => handleRowClick(index(), e)}
                          onDblClick={() => openClusterInViewer(cluster)}
                        >
                          <td
                            class="cursor-pointer"
                            onClick={(e) => { e.stopPropagation(); setSelectedIndex(index()); }}
                          >
                            <div class={`w-3 h-3 rounded-full border-2 ${
                              selectedIndex() === index()
                                ? 'border-primary bg-primary'
                                : 'border-base-content/30'
                            }`} />
                          </td>
                          <td class="font-mono text-xs">
                            {cluster.clusterId + 1}
                          </td>
                          <td class="text-right font-mono text-xs">
                            {cluster.population.toFixed(1)}%
                          </td>
                          <Show when={hasVina()}>
                            <td class="text-right font-mono text-xs">
                              {cluster.vinaRescore != null ? cluster.vinaRescore.toFixed(1) : '-'}
                            </td>
                          </Show>
                          <Show when={hasCordial()}>
                            <td class="text-right font-mono text-xs">
                              {cluster.cordialPHighAffinity != null
                                ? `${(cluster.cordialPHighAffinity * 100).toFixed(0)}%`
                                : '-'}
                            </td>
                            <td class="text-right font-mono text-xs">
                              {cluster.cordialPVeryHighAffinity != null
                                ? `${(cluster.cordialPVeryHighAffinity * 100).toFixed(0)}%`
                                : '-'}
                            </td>
                          </Show>
                        </tr>
                      )}
                    </For>
                  </tbody>
                </table>
              </div>
            </div>

            {/* Right: cluster detail + dihedral profile */}
            <div class="w-72 flex-shrink-0 flex flex-col min-h-0 overflow-auto">
              <Show
                when={selectedCluster()}
                fallback={
                  <div class="flex-1 flex items-center justify-center text-xs text-base-content/40 text-center px-4">
                    Select a cluster to preview
                  </div>
                }
              >
                {(cluster) => (
                  <div class="flex flex-col gap-2 h-full">
                    {/* Navigation arrows + counter */}
                    <div class="flex items-center justify-between">
                      <button
                        class="btn btn-ghost btn-xs btn-square"
                        disabled={selectedIndex() === 0 || selectedIndex() === null}
                        onClick={handlePrev}
                        title="Previous cluster"
                      >
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                        </svg>
                      </button>
                      <span class="text-xs text-base-content/50 font-mono font-medium">
                        Cluster {cluster().clusterId + 1} &bull; {cluster().population.toFixed(1)}%
                      </span>
                      <button
                        class="btn btn-ghost btn-xs btn-square"
                        disabled={selectedIndex() === null || selectedIndex()! >= sortedClusters().length - 1}
                        onClick={handleNext}
                        title="Next cluster"
                      >
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                        </svg>
                      </button>
                    </div>

                    {/* Scores row */}
                    <div class="flex flex-wrap gap-x-3 gap-y-0.5 text-xs">
                      <Show when={cluster().vinaRescore != null}>
                        <span><span class="text-base-content/60">Vina</span> <span class="font-mono font-semibold">{cluster().vinaRescore!.toFixed(1)}</span></span>
                      </Show>
                      <Show when={cluster().cordialPHighAffinity != null}>
                        <span><span class="text-base-content/60">{"P(<1\u00B5M)"}</span> <span class="font-mono font-semibold">{((cluster().cordialPHighAffinity ?? 0) * 100).toFixed(0)}%</span></span>
                      </Show>
                      <span><span class="text-base-content/60">Frames</span> <span class="font-mono font-semibold">{cluster().frameCount}</span></span>
                    </div>

                    {/* Per-cluster dihedral angles */}
                    <Show when={torsionAnalysis()?.data.torsions.length}>
                      <div class="bg-base-100 rounded-lg border border-base-300 p-2">
                        <p class="text-[10px] text-base-content/50 font-semibold uppercase tracking-wider mb-1">Dihedrals at centroid</p>
                        <div class="overflow-auto max-h-40">
                          <table class="table table-xs">
                            <thead>
                              <tr>
                                <th class="text-[10px]">Bond</th>
                                <th class="text-right text-[10px]">Angle</th>
                                <th class="text-right text-[10px]">Mean</th>
                              </tr>
                            </thead>
                            <tbody>
                              <For each={torsionAnalysis()!.data.torsions}>
                                {(torsion) => {
                                  const cv = () => torsion.clusterValues.find(v => v.clusterId === cluster().clusterId);
                                  return (
                                    <tr>
                                      <td class="font-mono text-[10px]">{torsion.label}</td>
                                      <td class="text-right font-mono text-[10px]">
                                        {cv() ? `${cv()!.angle.toFixed(1)}°` : '-'}
                                      </td>
                                      <td class="text-right font-mono text-[10px] text-base-content/50">
                                        {torsion.circularMean.toFixed(1)}°
                                      </td>
                                    </tr>
                                  );
                                }}
                              </For>
                            </tbody>
                          </table>
                        </div>
                      </div>
                    </Show>

                    {/* Actions */}
                    <div class="mt-auto flex flex-col gap-1.5">
                      <button
                        class="btn btn-primary btn-sm w-full"
                        onClick={() => openClusterInViewer(cluster())}
                        disabled={!cluster().centroidPdbPath}
                      >
                        View 3D
                      </button>
                      <button
                        class="btn btn-outline btn-sm w-full"
                        onClick={openTrajectoryInViewer}
                      >
                        Play Trajectory
                      </button>
                      <button
                        class="btn btn-outline btn-sm w-full"
                        onClick={openAllClustersInViewer}
                      >
                        View All Clusters
                      </button>
                    </div>
                  </div>
                )}
              </Show>
            </div>
          </div>
        </Show>
      </Show>

      {/* ── Dihedrals tab ── */}
      <Show when={activeTab() === 'dihedrals' && torsionAnalysis()}>
        <div class="flex-1 overflow-auto min-h-0">
          <MDTorsionPanel analysis={torsionAnalysis()!} />
        </div>
      </Show>

      {/* ── Scoring legend ── */}
      <Show when={activeTab() === 'clusters' && (hasVina() || hasCordial())}>
        <div class="mt-2 pt-2 border-t border-base-300">
          <div class="text-xs text-base-content/50 leading-relaxed max-w-[420px]">
            <span class="font-semibold text-base-content/70">Vina</span> = pocket rescore of cluster centroid
            <Show when={hasCordial()}>
              {' '}<span class="mx-1 text-base-content/30">|</span>{' '}
              <span class="font-semibold text-base-content/70">{"P(< 1\u00B5M)"}</span> / <span class="font-semibold text-base-content/70">{"P(< 100nM)"}</span> = ML-predicted binding probability (CORDIAL)
            </Show>
          </div>
        </div>
      </Show>
    </div>
  );
};

export default MDStepResults;
