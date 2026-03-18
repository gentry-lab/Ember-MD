import { Component, Show, For, createSignal } from 'solid-js';
import { workflowStore, ViewerQueueItem } from '../../stores/workflow';
import { ClusterResultData } from '../../../shared/types/ipc';
import path from 'path';

const MDStepResults: Component = () => {
  const { state, setMode, setMdStep, setViewerPdbPath, setViewerPdbQueue, setViewerTrajectoryPath, resetMd, resetViewer } = workflowStore;
  const api = window.electronAPI;

  // Analysis state
  const [isAnalyzing, setIsAnalyzing] = createSignal(false);
  const [analysisProgress, setAnalysisProgress] = createSignal(0);
  const [analysisStep, setAnalysisStep] = createSignal('');
  const [reportPath, setReportPath] = createSignal<string | null>(null);
  const [analysisDir, setAnalysisDir] = createSignal<string | null>(null);
  const [analysisError, setAnalysisError] = createSignal<string | null>(null);
  const [clusterResults, setClusterResults] = createSignal<ClusterResultData[]>([]);

  const result = () => state().md.result;
  const systemInfo = () => state().md.systemInfo;
  const jobName = () => state().jobName.trim() || 'job';
  const isLigandOnly = () => state().md.inputMode === 'ligand_only';

  // Step name mapping for progress display
  const stepNames: Record<string, string> = {
    'analyze_contacts': 'Computing contacts...',
    'analyze_rmsd': 'RMSD analysis...',
    'analyze_rmsf': 'RMSF analysis...',
    'analyze_sse': 'Secondary structure...',
    'analyze_hbonds': 'H-bond analysis...',
    'analyze_ligand_props': 'Ligand properties...',
    'analyze_torsions': 'Torsion analysis...',
    'clustering': 'Clustering trajectory...',
    'compile_pdf': 'Compiling report...',
    'done': 'Complete',
  };

  const runAnalysis = async () => {
    if (!result()) return;

    setIsAnalyzing(true);
    setAnalysisError(null);
    setAnalysisProgress(0);
    setAnalysisStep('Starting analysis...');
    setReportPath(null);
    setClusterResults([]);

    const outputDir = path.dirname(result()!.trajectoryPath);
    const analysisOutputDir = path.join(outputDir, 'analysis');

    // Listen for progress updates from MD_OUTPUT
    const cleanup = api.onMdOutput((data) => {
      const text = data.data;
      // Parse PROGRESS: lines
      const progressMatch = text.match(/PROGRESS:(\w+):(\d+)/);
      if (progressMatch) {
        const step = progressMatch[1];
        const pct = parseInt(progressMatch[2], 10);
        setAnalysisProgress(pct);
        setAnalysisStep(stepNames[step] || step);
      }
    });

    try {
      // Build sim info from current state
      const simInfo: Record<string, string> = {};
      const si = systemInfo();
      if (si) {
        simInfo.atoms = si.atomCount.toLocaleString();
      }
      simInfo.temperature = `${state().md.config.temperatureK} K`;
      simInfo.duration = `${state().md.config.productionNs} ns`;
      simInfo.forceField = state().md.config.forceFieldPreset || 'ff19SB/OPC';
      if (state().md.benchmarkResult) {
        simInfo.performance = `${state().md.benchmarkResult!.nsPerDay.toFixed(1)} ns/day`;
      }
      simInfo.jobName = jobName();

      const reportResult = await api.generateMdReport({
        topologyPath: result()!.systemPdbPath,
        trajectoryPath: result()!.trajectoryPath,
        outputDir: analysisOutputDir,
        ligandSelection: undefined,
        simInfo,
      });

      if (reportResult.ok) {
        setReportPath(reportResult.value.reportPath);
        setAnalysisDir(reportResult.value.analysisDir);
        if (reportResult.value.clusteringResults) {
          setClusterResults(reportResult.value.clusteringResults);
        }
        setAnalysisProgress(100);
        setAnalysisStep('Complete');
        console.log('[MD Results] Analysis complete:', reportResult.value.reportPath);
      } else {
        setAnalysisError(reportResult.error.message);
        console.error('[MD Results] Analysis failed:', reportResult.error.message);
      }
    } catch (err) {
      setAnalysisError((err as Error).message);
    } finally {
      setIsAnalyzing(false);
      cleanup();
    }
  };

  const openReport = () => {
    const rp = reportPath();
    if (rp) {
      api.openFolder(rp); // macOS 'open' works for files too
    }
  };

  const openAnalysisFolder = () => {
    const ad = analysisDir();
    if (ad) {
      api.openFolder(ad);
    }
  };

  const openInViewer = (pdbPath: string) => {
    resetViewer();
    setViewerPdbPath(pdbPath);
    setMode('viewer');
  };

  const openClustersInViewer = () => {
    const clusters = clusterResults().filter(c => c.centroidPdbPath);
    if (clusters.length === 0) return;
    resetViewer();

    const queue: ViewerQueueItem[] = clusters.map(c => ({
      pdbPath: c.centroidPdbPath!,
      label: `Cluster ${c.clusterId + 1} (${c.population.toFixed(0)}%)`,
    }));

    setViewerPdbQueue(queue);
    setViewerPdbPath(queue[0].pdbPath);
    setMode('viewer');
  };

  const openTrajectoryInViewer = () => {
    if (!result()) return;
    resetViewer();
    setViewerPdbPath(result()!.systemPdbPath);
    setViewerTrajectoryPath(result()!.trajectoryPath);
    setMode('viewer');
  };

  const handleOpenFolder = () => {
    if (result()) {
      const dir = path.dirname(result()!.trajectoryPath);
      api.openFolder(dir);
    }
  };

  const handleNewSimulation = () => {
    resetMd();
  };

  return (
    <div class="h-full flex flex-col">
      {/* Title */}
      <div class="text-center mb-3">
        <h2 class="text-xl font-bold">Simulation Complete</h2>
        <p class="text-sm text-base-content/90">
          {isLigandOnly() ? 'Ligand-only' : 'Protein-ligand'} MD simulation finished
        </p>
      </div>

      {/* Main content */}
      <div class="flex-1 grid grid-cols-3 gap-3 min-h-0">
        {/* Left column - Summary */}
        <div class="card bg-base-200 shadow-lg">
          <div class="card-body p-4">
            <h3 class="text-sm font-semibold mb-2">Summary</h3>
            <div class="space-y-2 text-xs">
              <div class="flex justify-between py-1 border-b border-base-300">
                <span class="text-base-content/85">Duration</span>
                <span class="font-mono font-medium">{state().md.config.productionNs} ns</span>
              </div>
              <Show when={systemInfo()}>
                <div class="flex justify-between py-1 border-b border-base-300">
                  <span class="text-base-content/85">Atoms</span>
                  <span class="font-mono">{systemInfo()!.atomCount.toLocaleString()}</span>
                </div>
              </Show>
              <div class="flex justify-between py-1 border-b border-base-300">
                <span class="text-base-content/85">Ligand</span>
                <span class="font-mono">{state().md.ligandName || 'Unknown'}</span>
              </div>
              <Show when={state().md.benchmarkResult}>
                <div class="flex justify-between py-1">
                  <span class="text-base-content/85">Performance</span>
                  <span class="font-mono">{state().md.benchmarkResult!.nsPerDay.toFixed(1)} ns/day</span>
                </div>
              </Show>
            </div>
          </div>
        </div>

        {/* Middle column - Output files */}
        <div class="card bg-base-200 shadow-lg overflow-y-auto">
          <div class="card-body p-4">
            <h3 class="text-sm font-semibold mb-2">Output Files</h3>
            <Show when={result()}>
              <div class="space-y-1.5">
                {/* System PDB */}
                <div class="flex items-center gap-2 p-1.5 bg-base-300 rounded">
                  <div class="flex-1 min-w-0">
                    <p class="text-[11px] font-medium truncate">{jobName()}_system.pdb</p>
                    <p class="text-[9px] text-base-content/60">Full solvated system</p>
                  </div>
                  <button
                    class="btn btn-ghost btn-xs"
                    onClick={() => openInViewer(result()!.systemPdbPath)}
                    title="Open in 3D Viewer"
                  >
                    View
                  </button>
                </div>

                {/* Final PDB */}
                <div class="flex items-center gap-2 p-1.5 bg-base-300 rounded">
                  <div class="flex-1 min-w-0">
                    <p class="text-[11px] font-medium truncate">{jobName()}_final.pdb</p>
                    <p class="text-[9px] text-base-content/60">Final production frame</p>
                  </div>
                  <button
                    class="btn btn-ghost btn-xs"
                    onClick={() => openInViewer(result()!.finalPdbPath)}
                    title="Open in 3D Viewer"
                  >
                    View
                  </button>
                </div>

                {/* Trajectory */}
                <div class="flex items-center gap-2 p-1.5 bg-base-300 rounded">
                  <div class="flex-1 min-w-0">
                    <p class="text-[11px] font-medium truncate">{jobName()}_trajectory.dcd</p>
                    <p class="text-[9px] text-base-content/60">Full trajectory</p>
                  </div>
                  <button
                    class="btn btn-ghost btn-xs"
                    onClick={openTrajectoryInViewer}
                    title="Play in 3D Viewer"
                  >
                    Play
                  </button>
                </div>

                {/* Energy CSV */}
                <div class="flex items-center gap-2 p-1.5 bg-base-300 rounded">
                  <div class="flex-1 min-w-0">
                    <p class="text-[11px] font-medium truncate">{jobName()}_energy.csv</p>
                    <p class="text-[9px] text-base-content/60">Energy, temperature, volume</p>
                  </div>
                </div>
              </div>
            </Show>
          </div>
        </div>

        {/* Right column - Analysis */}
        <div class="card bg-base-200 shadow-lg overflow-y-auto">
          <div class="card-body p-4">
            <h3 class="text-sm font-semibold mb-2">Analysis</h3>

            {/* Before analysis: show Analyze button */}
            <Show when={!isAnalyzing() && !reportPath() && !analysisError()}>
              <div class="flex-1 flex items-center justify-center">
                <div class="text-center">
                  <button class="btn btn-primary btn-sm" onClick={runAnalysis}>
                    <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    Analyze
                  </button>
                  <p class="text-[9px] text-base-content/60 mt-2 max-w-[160px]">
                    RMSD, RMSF, contacts, H-bonds, SSE, torsions, clustering
                  </p>
                </div>
              </div>
            </Show>

            {/* During analysis: progress bar */}
            <Show when={isAnalyzing()}>
              <div class="flex-1 flex items-center justify-center">
                <div class="w-full text-center">
                  <span class="loading loading-spinner loading-sm text-primary" />
                  <p class="text-[10px] text-base-content/80 mt-2 font-medium">{analysisStep()}</p>
                  <progress
                    class="progress progress-primary w-full mt-2"
                    value={analysisProgress()}
                    max="100"
                  />
                  <p class="text-[9px] text-base-content/60 mt-1">{analysisProgress()}%</p>
                </div>
              </div>
            </Show>

            {/* Error state */}
            <Show when={analysisError() && !isAnalyzing()}>
              <div class="text-[10px] text-error p-2 bg-error/10 rounded mb-2">
                {analysisError()}
              </div>
              <button class="btn btn-ghost btn-xs" onClick={runAnalysis}>
                Retry
              </button>
            </Show>

            {/* After analysis: report + clusters */}
            <Show when={reportPath()}>
              <div class="space-y-2">
                {/* Open Report button */}
                <button class="btn btn-primary btn-sm w-full" onClick={openReport}>
                  <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  Open Report
                </button>

                <button class="btn btn-ghost btn-xs w-full" onClick={openAnalysisFolder}>
                  Open Analysis Folder
                </button>

                {/* Cluster results */}
                <Show when={clusterResults().length > 0}>
                  <div class="border-t border-base-300 pt-2 mt-2">
                    <p class="text-[10px] font-semibold mb-1">Clusters</p>
                    <div class="space-y-1">
                      <For each={clusterResults()}>
                        {(cluster) => (
                          <div class="flex items-center gap-1.5 p-1 bg-base-300 rounded">
                            <div class="w-5 h-5 rounded bg-primary text-primary-content flex items-center justify-center text-[9px] font-bold flex-shrink-0">
                              {cluster.clusterId + 1}
                            </div>
                            <div class="flex-1 min-w-0">
                              <p class="text-[10px]">
                                {cluster.population.toFixed(0)}%
                                <span class="text-base-content/60 ml-1">({cluster.frameCount} frames)</span>
                              </p>
                            </div>
                          </div>
                        )}
                      </For>
                    </div>
                    <button
                      class="btn btn-outline btn-xs w-full mt-1.5"
                      onClick={openClustersInViewer}
                    >
                      View Clusters
                    </button>
                  </div>
                </Show>

                {/* Re-run option */}
                <button class="btn btn-ghost btn-xs w-full mt-1" onClick={runAnalysis}>
                  Re-run Analysis
                </button>
              </div>
            </Show>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div class="flex justify-between mt-3">
        <button class="btn btn-outline btn-sm" onClick={handleOpenFolder}>
          <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9a2 2 0 00-2 2v5a2 2 0 01-2 2z" />
          </svg>
          Open Folder
        </button>
        <div class="flex gap-2">
          <button class="btn btn-ghost btn-sm" onClick={() => setMdStep('md-configure')}>
            <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 17l-5-5m0 0l5-5m-5 5h12" />
            </svg>
            Back
          </button>
          <button class="btn btn-primary btn-sm" onClick={handleNewSimulation}>
            New Simulation
            <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
};

export default MDStepResults;
