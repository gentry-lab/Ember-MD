// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, Show, For, createSignal, createMemo } from 'solid-js';
import { workflowStore, ViewerQueueItem } from '../../stores/workflow';
import path from 'path';
import MDTorsionPanel from './MDTorsionPanel';
import { buildMdProjectTable } from '../../utils/viewerQueue';

type SortField = 'clusterId' | 'population' | 'vinaRescore' | 'cordialPHighAffinity' | 'cordialPVeryHighAffinity';
type SortDirection = 'asc' | 'desc';
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
  const { state, openViewerSession, resetMd } = workflowStore;
  const api = window.electronAPI;

  const [sortField, setSortField] = createSignal<SortField>('population');
  const [sortDirection, setSortDirection] = createSignal<SortDirection>('desc');
  const [selectedIndex, setSelectedIndex] = createSignal<number | null>(null);

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
      projectTable,
    });
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
    openViewerSession({
      pdbPath: cluster.centroidPdbPath,
      pdbQueue: queue,
      pdbQueueIndex: queueIndex >= 0 ? queueIndex : 0,
      projectTable: {
        ...projectTable,
        activeRowId: projectTable.rows.find((row) => row.queueIndex === (queueIndex >= 0 ? queueIndex : 0))?.id
          ?? projectTable.activeRowId,
      },
    });
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
    openViewerSession({
      pdbPath: queue[0].pdbPath,
      pdbQueue: queue,
      pdbQueueIndex: 0,
      projectTable: {
        ...projectTable,
        activeRowId: projectTable.rows.find((row) => row.queueIndex === 0)?.id ?? projectTable.activeRowId,
      },
    });
  };

  const handleOpenFolder = () => {
    const dir = getSimulationRunRoot();
    if (dir) {
      api.openFolder(dir);
    }
  };

  const openReport = () => {
    const runRoot = getSimulationRunRoot();
    if (!runRoot) return;
    const reportPath = path.join(runRoot, 'analysis', 'full_report.pdf');
    api.openFolder(reportPath);
  };

  return (
    <div class="h-full flex flex-col gap-3">
      {/* Header */}
      <div class="flex items-center justify-between mb-3">
        <div>
          <h2 class="text-lg font-bold">Simulation Complete</h2>
          <div class="flex gap-3 text-xs text-base-content/70">
            <span>{state().md.config.productionNs} ns</span>
            <Show when={systemInfo()}>
              <span>{systemInfo()!.atomCount.toLocaleString()} atoms</span>
            </Show>
            <Show when={state().md.benchmarkResult}>
              <span>{state().md.benchmarkResult!.nsPerDay.toFixed(1)} ns/day</span>
            </Show>
          </div>
        </div>
        <div class="flex gap-1.5">
          <button class="btn btn-outline btn-xs" onClick={openTrajectoryInViewer}>
            Play Trajectory
          </button>
          <button class="btn btn-outline btn-xs" onClick={openReport}>
            Open Report
          </button>
          <Show when={clusters().length > 0}>
            <button class="btn btn-outline btn-xs" onClick={openAllClustersInViewer}>
              View All Clusters
            </button>
          </Show>
        </div>
      </div>

      {/* Scored clusters table */}
      <Show
        when={clusters().length > 0}
        fallback={
          <div class="flex-1 flex items-center justify-center text-base-content/50 text-sm">
            No clusters available
          </div>
        }
      >
        <div class="flex-1 min-h-0 flex flex-col">
          <div class="flex-1 overflow-auto min-h-0">
            <table class="table table-xs table-pin-rows">
              <thead>
                <tr class="bg-base-200">
                  <th class="cursor-pointer select-none" onClick={() => handleSort('clusterId')}>
                    Cluster{sortIndicator('clusterId')}
                  </th>
                  <th class="cursor-pointer select-none text-right" onClick={() => handleSort('population')}>
                    Pop%{sortIndicator('population')}
                  </th>
                  <Show when={hasVina()}>
                    <th class="cursor-pointer select-none text-right" onClick={() => handleSort('vinaRescore')}>
                      Vina{sortIndicator('vinaRescore')}
                    </th>
                  </Show>
                  <Show when={hasCordial()}>
                    <th class="cursor-pointer select-none text-right" onClick={() => handleSort('cordialPHighAffinity')}>
                      P(&lt;1uM){sortIndicator('cordialPHighAffinity')}
                    </th>
                    <th class="cursor-pointer select-none text-right" onClick={() => handleSort('cordialPVeryHighAffinity')}>
                      P(&lt;100nM){sortIndicator('cordialPVeryHighAffinity')}
                    </th>
                  </Show>
                </tr>
              </thead>
              <tbody>
                <For each={sortedClusters()}>
                  {(cluster, index) => (
                    <tr
                      class={`cursor-pointer hover:bg-base-200 ${
                        selectedIndex() === index() ? 'bg-primary/10' : ''
                      }`}
                      onClick={() => setSelectedIndex(index())}
                      onDblClick={() => openClusterInViewer(cluster)}
                    >
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

          <Show when={selectedCluster()}>
            <div class="border-t border-base-300 pt-2 mt-2 flex items-center justify-between">
              <div class="text-xs text-base-content/70">
                <span class="font-semibold">Cluster {selectedCluster()!.clusterId + 1}</span>
                <span class="mx-2">|</span>
                <span>{selectedCluster()!.population.toFixed(1)}% of trajectory</span>
                <span class="mx-2">|</span>
                <span>{selectedCluster()!.frameCount} frames</span>
                <span class="mx-2">|</span>
                <span>Centroid: frame {selectedCluster()!.centroidFrame}</span>
              </div>
              <button
                class="btn btn-primary btn-xs"
                onClick={() => openClusterInViewer(selectedCluster()!)}
              >
                View 3D
              </button>
            </div>
          </Show>
        </div>
      </Show>

      <Show when={torsionAnalysis()}>
        <MDTorsionPanel analysis={torsionAnalysis()!} />
      </Show>

      {/* Bottom actions */}
      <div class="flex justify-between">
        <button class="btn btn-ghost btn-sm" onClick={handleOpenFolder}>
          <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9a2 2 0 00-2 2v5a2 2 0 01-2 2z" />
          </svg>
          Open Folder
        </button>
        <button class="btn btn-primary btn-sm" onClick={() => resetMd()}>
          New Job
          <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
          </svg>
        </button>
      </div>
    </div>
  );
};

export default MDStepResults;
