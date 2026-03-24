import { Component, Show, createSignal, For } from 'solid-js';
import path from 'path';
import {
  workflowStore,
  ClusteringConfig,
  ClusterResult,
} from '../../stores/workflow';

interface ClusteringModalProps {
  isOpen: boolean;
  onClose: () => void;
  onViewCluster: (centroidFrame: number) => void;
}

const ClusteringModal: Component<ClusteringModalProps> = (props) => {
  const {
    state,
    setViewerClusteringConfig,
    setViewerClusteringResults,
    setViewerIsClustering,
  } = workflowStore;

  const api = window.electronAPI;
  const [logs, setLogs] = createSignal<string>('');
  const [error, setError] = createSignal<string | null>(null);

  const config = () => state().viewer.clusteringConfig;
  const results = () => state().viewer.clusteringResults;
  const isClustering = () => state().viewer.isClustering;
  const pdbPath = () => state().viewer.pdbPath;
  const trajectoryPath = () => state().viewer.trajectoryPath;

  const handleRunClustering = async () => {
    if (!pdbPath() || !trajectoryPath()) {
      setError('Please load a PDB and trajectory first');
      return;
    }

    setError(null);
    setLogs('Starting clustering...\n');
    setViewerIsClustering(true);
    setViewerClusteringResults(null);

    // Set up log listener
    const removeListener = api.onMdOutput((data) => {
      setLogs((prev) => {
        const combined = prev + data.data;
        return combined.length > 50000 ? combined.slice(-50000) : combined;
      });
    });

    try {
      // Determine output directory
      const trajDir = path.dirname(trajectoryPath()!);
      const runRoot = path.basename(trajDir) === 'results' ? path.dirname(trajDir) : trajDir;
      const outputDir = path.join(runRoot, 'analysis', 'clustering');

      const result = await api.clusterTrajectory({
        topologyPath: pdbPath()!,
        trajectoryPath: trajectoryPath()!,
        numClusters: config().numClusters,
        method: config().method,
        rmsdSelection: config().rmsdSelection,
        stripWaters: config().stripWaters,
        outputDir,
      });

      if (result.ok) {
        setViewerClusteringResults({
          clusters: result.value.clusters,
          frameAssignments: result.value.frameAssignments,
        });
        setLogs((prev) => prev + '\nClustering complete!\n');
      } else {
        setError(result.error.message);
      }
    } catch (err) {
      setError(`Clustering failed: ${(err as Error).message}`);
    } finally {
      removeListener();
      setViewerIsClustering(false);
    }
  };

  const handleMethodChange = (e: Event) => {
    const target = e.target as HTMLSelectElement;
    setViewerClusteringConfig({ method: target.value as ClusteringConfig['method'] });
  };

  const handleSelectionChange = (e: Event) => {
    const target = e.target as HTMLSelectElement;
    setViewerClusteringConfig({ rmsdSelection: target.value as ClusteringConfig['rmsdSelection'] });
  };

  const handleNumClustersChange = (e: Event) => {
    const target = e.target as HTMLInputElement;
    setViewerClusteringConfig({ numClusters: parseInt(target.value, 10) || 5 });
  };

  const handleStripWatersChange = () => {
    setViewerClusteringConfig({ stripWaters: !config().stripWaters });
  };

  return (
    <Show when={props.isOpen}>
      <div class="modal modal-open">
        <div class="modal-box max-w-2xl">
          <h3 class="font-bold text-lg mb-4">Cluster Ensemble</h3>

          {/* Configuration */}
          <div class="flex flex-col gap-3 mb-4">
            <div class="form-control">
              <label class="label py-1">
                <span class="label-text">Number of clusters</span>
              </label>
              <input
                type="number"
                class="input input-bordered input-sm w-24"
                min="2"
                max="50"
                value={config().numClusters}
                onInput={handleNumClustersChange}
                disabled={isClustering()}
              />
            </div>

            <div class="form-control">
              <label class="label py-1">
                <span class="label-text">Clustering method</span>
              </label>
              <select
                class="select select-bordered select-sm w-48"
                value={config().method}
                onChange={handleMethodChange}
                disabled={isClustering()}
              >
                <option value="kmeans">K-means</option>
                <option value="dbscan">DBSCAN (auto-clusters)</option>
                <option value="hierarchical">Hierarchical</option>
              </select>
            </div>

            <div class="form-control">
              <label class="label py-1">
                <span class="label-text">RMSD selection</span>
              </label>
              <select
                class="select select-bordered select-sm w-48"
                value={config().rmsdSelection}
                onChange={handleSelectionChange}
                disabled={isClustering()}
              >
                <option value="ligand">Ligand heavy atoms</option>
                <option value="backbone">Protein backbone</option>
                <option value="all">All heavy atoms</option>
              </select>
            </div>

            <label class="label cursor-pointer justify-start gap-2 py-1">
              <input
                type="checkbox"
                class="checkbox checkbox-sm checkbox-primary"
                checked={config().stripWaters}
                onChange={handleStripWatersChange}
                disabled={isClustering()}
              />
              <span class="label-text">Strip waters and ions from output PDBs</span>
            </label>
          </div>

          {/* Run button */}
          <div class="mb-4">
            <button
              class="btn btn-primary btn-sm"
              onClick={handleRunClustering}
              disabled={isClustering() || !pdbPath() || !trajectoryPath()}
            >
              <Show when={isClustering()} fallback="Run Clustering">
                <span class="loading loading-spinner loading-xs mr-1" />
                Clustering...
              </Show>
            </button>
          </div>

          {/* Error display */}
          <Show when={error()}>
            <div class="alert alert-error text-sm mb-4">
              {error()}
            </div>
          </Show>

          {/* Logs */}
          <Show when={logs()}>
            <div class="mb-4">
              <label class="label py-1">
                <span class="label-text text-xs">Output</span>
              </label>
              <div class="bg-base-300 rounded p-2 text-xs font-mono h-32 overflow-auto">
                <pre class="whitespace-pre-wrap">{logs()}</pre>
              </div>
            </div>
          </Show>

          {/* Results */}
          <Show when={results()}>
            <div class="divider text-xs">Results</div>
            <div class="overflow-x-auto">
              <table class="table table-xs">
                <thead>
                  <tr>
                    <th>Cluster</th>
                    <th>Frames</th>
                    <th>Population</th>
                    <th>Centroid</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  <For each={results()!.clusters}>
                    {(cluster: ClusterResult) => (
                      <tr>
                        <td class="font-mono">{cluster.clusterId}</td>
                        <td>{cluster.frameCount}</td>
                        <td>{cluster.population.toFixed(1)}%</td>
                        <td class="font-mono">Frame {cluster.centroidFrame}</td>
                        <td>
                          <button
                            class="btn btn-xs btn-ghost"
                            onClick={() => props.onViewCluster(cluster.centroidFrame)}
                          >
                            View
                          </button>
                        </td>
                      </tr>
                    )}
                  </For>
                </tbody>
              </table>
            </div>
          </Show>

          {/* Modal actions */}
          <div class="modal-action">
            <button class="btn btn-sm" onClick={() => props.onClose()}>
              Close
            </button>
          </div>
        </div>
        <div class="modal-backdrop" onClick={() => props.onClose()} />
      </div>
    </Show>
  );
};

export default ClusteringModal;
