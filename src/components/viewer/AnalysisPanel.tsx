// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, Show, createSignal } from 'solid-js';
import path from 'path';
import { workflowStore } from '../../stores/workflow';
import type { AnalysisResult, RmsdAnalysisResult, RmsfAnalysisResult, HbondAnalysisResult } from '../../../shared/types/ipc';

interface AnalysisPanelProps {
  onOpenClustering: () => void;
}

const AnalysisPanel: Component<AnalysisPanelProps> = (props) => {
  const { state } = workflowStore;
  const api = window.electronAPI;

  const [isAnalyzing, setIsAnalyzing] = createSignal(false);
  const [analysisType, setAnalysisType] = createSignal<string | null>(null);
  const [logs, setLogs] = createSignal<string>('');
  const [error, setError] = createSignal<string | null>(null);
  const [rmsdResult, setRmsdResult] = createSignal<AnalysisResult | null>(null);
  const [rmsfResult, setRmsfResult] = createSignal<AnalysisResult | null>(null);
  const [hbondsResult, setHbondsResult] = createSignal<AnalysisResult | null>(null);
  const [contactsResult, setContactsResult] = createSignal<AnalysisResult | null>(null);
  const [reportPath, setReportPath] = createSignal<string | null>(null);

  const pdbPath = () => state().viewer.pdbPath;
  const trajectoryPath = () => state().viewer.trajectoryPath;
  const hasTrajectory = () => trajectoryPath() !== null;

  const getOutputDir = () => {
    const trajPath = trajectoryPath();
    if (!trajPath) return null;
    const trajDir = path.dirname(trajPath);
    const runRoot = path.basename(trajDir) === 'results' ? path.dirname(trajDir) : trajDir;
    return path.join(runRoot, 'analysis');
  };

  const runAnalysis = async (type: 'rmsd' | 'rmsf' | 'hbonds' | 'contacts') => {
    if (!pdbPath() || !trajectoryPath()) return;

    setError(null);
    setLogs(`Running ${type.toUpperCase()} analysis...\n`);
    setIsAnalyzing(true);
    setAnalysisType(type);

    const removeListener = api.onMdOutput((data) => {
      setLogs((prev) => {
        const combined = prev + data.data;
        return combined.length > 50000 ? combined.slice(-50000) : combined;
      });
    });

    try {
      const outputDir = getOutputDir();
      if (!outputDir) return;

      const result = await api.analyzeTrajectory({
        topologyPath: pdbPath()!,
        trajectoryPath: trajectoryPath()!,
        analysisType: type,
        outputDir: path.join(outputDir, type),
      });

      if (result.ok) {
        setLogs((prev) => prev + '\nAnalysis complete!\n');
        if (type === 'rmsd') setRmsdResult(result.value);
        if (type === 'rmsf') setRmsfResult(result.value);
        if (type === 'hbonds') setHbondsResult(result.value);
        if (type === 'contacts') setContactsResult(result.value);
      } else {
        setError(result.error.message);
      }
    } catch (err) {
      setError(`Analysis failed: ${(err as Error).message}`);
    } finally {
      removeListener();
      setIsAnalyzing(false);
      setAnalysisType(null);
    }
  };

  const runFullReport = async () => {
    if (!pdbPath() || !trajectoryPath()) return;

    setError(null);
    setLogs('Generating full MD analysis report...\n');
    setIsAnalyzing(true);
    setAnalysisType('report');

    const removeListener = api.onMdOutput((data) => {
      setLogs((prev) => {
        const combined = prev + data.data;
        return combined.length > 50000 ? combined.slice(-50000) : combined;
      });
    });

    try {
      const outputDir = getOutputDir();
      if (!outputDir) return;

      const result = await api.generateMdReport({
        topologyPath: pdbPath()!,
        trajectoryPath: trajectoryPath()!,
        outputDir,
      });

      if (result.ok) {
        setLogs((prev) => prev + '\nReport generated!\n');
        setReportPath(result.value.reportPath);
      } else {
        setError(result.error.message);
      }
    } catch (err) {
      setError(`Report generation failed: ${(err as Error).message}`);
    } finally {
      removeListener();
      setIsAnalyzing(false);
      setAnalysisType(null);
    }
  };

  const openReport = async () => {
    const path = reportPath();
    if (path) {
      await api.openFolder(path); // macOS 'open' works for files too
    }
  };

  return (
    <Show when={hasTrajectory()}>
      <div class="card bg-base-200 p-2">
        <div class="flex flex-col gap-2">
          {/* Analysis buttons row */}
          <div class="flex items-center gap-2 flex-wrap">
            <span class="text-xs font-semibold">Analysis:</span>

            <button
              class="btn btn-xs btn-outline"
              onClick={() => props.onOpenClustering()}
              disabled={isAnalyzing()}
              title="Cluster trajectory frames"
            >
              Cluster
            </button>

            <button
              class="btn btn-xs btn-outline"
              onClick={() => runAnalysis('rmsd')}
              disabled={isAnalyzing()}
              title="Calculate RMSD over time"
            >
              <Show when={analysisType() === 'rmsd'} fallback="RMSD">
                <span class="loading loading-spinner loading-xs" />
              </Show>
            </button>

            <button
              class="btn btn-xs btn-outline"
              onClick={() => runAnalysis('rmsf')}
              disabled={isAnalyzing()}
              title="Calculate per-residue RMSF"
            >
              <Show when={analysisType() === 'rmsf'} fallback="RMSF">
                <span class="loading loading-spinner loading-xs" />
              </Show>
            </button>

            <button
              class="btn btn-xs btn-outline"
              onClick={() => runAnalysis('hbonds')}
              disabled={isAnalyzing()}
              title="Analyze hydrogen bonds"
            >
              <Show when={analysisType() === 'hbonds'} fallback="H-bonds">
                <span class="loading loading-spinner loading-xs" />
              </Show>
            </button>

            <button
              class="btn btn-xs btn-outline"
              onClick={() => runAnalysis('contacts')}
              disabled={isAnalyzing()}
              title="Analyze protein-ligand contacts"
            >
              <Show when={analysisType() === 'contacts'} fallback="Contacts">
                <span class="loading loading-spinner loading-xs" />
              </Show>
            </button>

            <div class="flex-1" />

            <button
              class="btn btn-xs btn-primary"
              onClick={runFullReport}
              disabled={isAnalyzing()}
              title="Generate full HTML report"
            >
              <Show when={analysisType() === 'report'} fallback="Full Report">
                <span class="loading loading-spinner loading-xs" />
              </Show>
            </button>
          </div>

          {/* Error display */}
          <Show when={error()}>
            <div class="alert alert-error text-xs py-1">
              {error()}
            </div>
          </Show>

          {/* Results summary */}
          <Show when={rmsdResult() || rmsfResult() || hbondsResult() || contactsResult() || reportPath()}>
            <div class="text-xs flex flex-wrap gap-2 items-center">
              <Show when={rmsdResult()}>
                <div class="badge badge-sm badge-success gap-1">
                  RMSD: {(rmsdResult()?.data as RmsdAnalysisResult | undefined)?.stats?.proteinMean?.toFixed(2) || '?'} Å
                </div>
              </Show>
              <Show when={rmsfResult()}>
                <div class="badge badge-sm badge-success gap-1">
                  RMSF: {(rmsfResult()?.data as RmsfAnalysisResult | undefined)?.stats?.mean?.toFixed(2) || '?'} Å
                </div>
              </Show>
              <Show when={hbondsResult()}>
                <div class="badge badge-sm badge-success gap-1">
                  H-bonds: {(hbondsResult()?.data as HbondAnalysisResult | undefined)?.totalUnique || '?'}
                </div>
              </Show>
              <Show when={contactsResult()}>
                <div class="badge badge-sm badge-success gap-1">
                  Contacts: {((contactsResult()?.data as { residues?: unknown[] } | undefined)?.residues?.length) ?? '?'}
                </div>
              </Show>
              <Show when={reportPath()}>
                <button
                  class="btn btn-xs btn-ghost"
                  onClick={openReport}
                >
                  Open Report
                </button>
              </Show>
            </div>
          </Show>

          {/* Logs (collapsible) */}
          <Show when={logs() && isAnalyzing()}>
            <div class="collapse collapse-arrow bg-base-300 rounded">
              <input type="checkbox" class="peer" checked />
              <div class="collapse-title text-xs py-1 min-h-0">
                Output
              </div>
              <div class="collapse-content">
                <pre class="text-xs font-mono whitespace-pre-wrap max-h-32 overflow-auto">
                  {logs()}
                </pre>
              </div>
            </div>
          </Show>
        </div>
      </div>
    </Show>
  );
};

export default AnalysisPanel;
