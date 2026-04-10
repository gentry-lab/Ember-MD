// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, Show, For, createSignal, onCleanup, onMount } from 'solid-js';
import { workflowStore } from '../../stores/workflow';
import { projectPathsFromProjectDir } from '../../utils/projectPaths';
import { buildScoreRunFolderName } from '../../utils/jobName';
import TerminalOutput from '../shared/TerminalOutput';
import StopConfirmModal from '../shared/StopConfirmModal';
import type { BatchScoreRequest, BatchScoreEntryResult, ScoreTrajectoryRequest } from '../../../shared/types/ipc';

const ScoreStepProgress: Component = () => {
  const {
    state,
    appendLog,
    clearLogs,
    setCurrentPhase,
    setError,
    setIsRunning,
    setScoreRunning,
    setScoreOutputDir,
    applyScoreBatchResults,
    updateScoreEntry,
    setScoreStep,
  } = workflowStore;
  const api = window.electronAPI;
  const [hasStarted, setHasStarted] = createSignal(false);
  const [showStopConfirm, setShowStopConfirm] = createSignal(false);

  onMount(() => {
    const cleanup = api.onScoreOutput((data) => {
      // Detect CORDIAL availability early so column shows during scoring
      if (data.data.includes('SCORE_CORDIAL_AVAILABLE')) {
        workflowStore.setScoreCordialAvailable(true);
      }
      // Fast-path: only parse lines if structured marker present
      if (data.data.includes('SCORE_ENTRY_RESULT')) {
        for (const line of data.data.split('\n')) {
          const match = line.match(/^SCORE_ENTRY_RESULT:([^:]+):(.+)$/);
          if (match) {
            try {
              const result: BatchScoreEntryResult = JSON.parse(match[2]);
              const matchedEntry = state().score.entries.find((entry) => entry.id === result.id)
                ?? state().score.entries.find((entry) =>
                  entry.pdbPath === result.pdbPath && entry.selectedLigandId === result.ligandId,
                );

              if (!matchedEntry) {
                appendLog(`[Score][UI] Warning: received result for unknown entry id=${result.id} pdb=${result.pdbPath}\n`);
                continue;
              }

              if (matchedEntry.id !== result.id) {
                appendLog(`[Score][UI] Fallback-matched result by pdbPath for ${matchedEntry.name}\n`);
              }

              updateScoreEntry(matchedEntry.id, {
                status: result.status === 'done' ? 'done' : 'error',
                preparedReceptorPath: result.preparedReceptorPath,
                extractedLigandSdfPath: result.extractedLigandSdfPath,
                vinaScore: result.vinaScore,
                cordialExpectedPkd: result.cordialExpectedPkd,
                cordialPHighAffinity: result.cordialPHighAffinity,
                qed: result.qed,
                errorMessage: result.errorMessage,
              });
            } catch { /* ignore parse errors */ }
          }
        }
      }
      appendLog(data.data);
    });
    onCleanup(cleanup);
  });

  const runScoring = async () => {
    const trajectoryConfig = state().score.trajectoryConfig;
    const entries = state().score.entries.filter((e) => e.status !== 'error');

    if (entries.length === 0 && !trajectoryConfig) return;

    setScoreRunning(true);
    setIsRunning(true);
    setCurrentPhase('generation');
    setError(null);
    clearLogs();

    // Mark all valid entries as pending
    for (const entry of entries) {
      updateScoreEntry(entry.id, { status: 'scoring' });
    }

    // Build output directory
    const projectDir = state().projectDir;
    if (!projectDir) {
      setError('No project selected');
      setCurrentPhase('error');
      setScoreRunning(false);
      setIsRunning(false);
      return;
    }
    const paths = projectPathsFromProjectDir(projectDir);
    const runFolder = buildScoreRunFolderName(
      state().score.descriptor,
      trajectoryConfig ? 'trajectory' : 'batch',
    );
    const jobDir = paths.scoring(runFolder).root;
    setScoreOutputDir(jobDir);

    try {
      let result;

      if (trajectoryConfig) {
        // Trajectory scoring: cluster + score centroids
        const trajRequest: ScoreTrajectoryRequest = {
          trajectoryPath: trajectoryConfig.trajectoryPath,
          topologyPath: trajectoryConfig.topologyPath,
          ligandSdfPath: trajectoryConfig.ligandSdfPath,
          numClusters: trajectoryConfig.numClusters,
          jobDir,
        };
        result = await api.scoreTrajectory(trajRequest);
      } else {
        // Batch PDB scoring
        const request: BatchScoreRequest = {
          entries: entries.map((e) => ({
            id: e.id,
            name: e.name,
            pdbPath: e.pdbPath,
            ligandId: e.selectedLigandId,
            isPrepared: e.isPrepared,
          })),
          jobDir,
        };
        result = await api.scoreBatch(request);
      }

      if (result.ok) {
        if (trajectoryConfig) {
          // For trajectory results, create score entries from the returned centroids
          const { setScoreEntries } = workflowStore;
          const newEntries = result.value.entries.map((r: BatchScoreEntryResult) => ({
            id: r.id,
            pdbPath: r.pdbPath,
            name: r.name,
            detectedLigands: [],
            selectedLigandId: r.ligandId,
            isPrepared: true,
            preparedReceptorPath: r.preparedReceptorPath,
            extractedLigandSdfPath: r.extractedLigandSdfPath,
            vinaScore: r.vinaScore,
            cordialExpectedPkd: r.cordialExpectedPkd,
            cordialPHighAffinity: r.cordialPHighAffinity,
            qed: r.qed,
            status: r.status,
            errorMessage: r.errorMessage,
          }));
          setScoreEntries(newEntries);
          workflowStore.setScoreCordialAvailable(result.value.cordialAvailable);
        } else {
          const mergeSummary = applyScoreBatchResults(result.value.entries, result.value.cordialAvailable);
          appendLog(
            `\n[Score][UI] Merged ${mergeSummary.updatedEntries}/${result.value.entries.length} results ` +
            `(id=${mergeSummary.matchedById}, path=${mergeSummary.matchedByPath}, unmatched=${mergeSummary.unmatchedResults}).\n`,
          );
          if (result.value.entries.length > 0 && mergeSummary.updatedEntries === 0) {
            appendLog('[Score][UI] Warning: batch returned results, but no rows were updated in the table.\n');
          }
        }
        setCurrentPhase('complete');
        appendLog(`\nScoring complete: ${result.value.entries.filter((e) => e.status === 'done').length} scored successfully.\n`);
      } else {
        setError(result.error.message);
        setCurrentPhase('error');
      }
    } catch (err) {
      setError((err as Error).message);
      setCurrentPhase('error');
    }

    setScoreRunning(false);
    setIsRunning(false);
  };

  onMount(() => {
    if (!state().score.isRunning && !hasStarted()) {
      setHasStarted(true);
      void runScoring();
    }
  });

  const handleBack = () => {
    if (state().score.isRunning) return;
    setCurrentPhase('idle');
    setError(null);
    clearLogs();
    // Reset entry statuses back to pending
    for (const entry of state().score.entries) {
      if (entry.status !== 'error' || entry.selectedLigandId) {
        updateScoreEntry(entry.id, {
          status: 'pending',
          vinaScore: null,
          cordialExpectedPkd: null,
          cordialPHighAffinity: null,
          qed: null,
          errorMessage: null,
        });
      }
    }
    setScoreStep('score-load');
  };

  const handleCancel = async () => {
    try {
      await api.cancelScoreBatch();
    } catch { /* ignore */ }
  };

  const scoredEntries = () => state().score.entries.filter((e) => e.status !== 'error' || e.vinaScore !== null);
  const totalValid = () => state().score.entries.filter((e) => e.selectedLigandId).length;
  const completedCount = () => state().score.entries.filter((e) => e.status === 'done' || e.status === 'error').length;

  return (
    <div class="h-full flex flex-col">
      <div class="flex items-center justify-between mb-3">
        <div>
          <h2 class="text-xl font-bold">
            {state().currentPhase === 'complete' ? 'Scoring Complete' : 'Scoring Complexes'}
          </h2>
          <p class="text-sm text-base-content/90">
            {state().score.isRunning
              ? `${completedCount()} of ${totalValid()} scored`
              : state().currentPhase === 'complete'
                ? `${state().score.entries.filter((e) => e.status === 'done').length} complexes scored`
                : 'Waiting...'}
          </p>
        </div>
        <div class="flex items-center gap-2">
          <Show when={state().score.isRunning}>
            <span class="loading loading-spinner loading-sm text-primary" />
          </Show>
          <Show when={state().currentPhase === 'complete'}>
            <span class="badge badge-success badge-sm">Done</span>
          </Show>
          <Show when={state().currentPhase === 'error'}>
            <span class="badge badge-error badge-sm">Error</span>
          </Show>
        </div>
      </div>

      <Show when={state().errorMessage}>
        <div class="alert alert-error py-2 mb-2">
          <span class="text-sm">{state().errorMessage}</span>
        </div>
      </Show>

      {/* Progress table */}
      <Show when={scoredEntries().length > 0}>
        <div class="overflow-x-auto max-h-40 overflow-y-auto mb-2">
          <table class="table table-xs table-zebra w-full">
            <thead>
              <tr>
                <th>#</th>
                <th>Name</th>
                <th>Status</th>
                <th class="text-right">Vina</th>
                <Show when={state().score.cordialAvailable}>
                  <th class="text-right">CORDIAL</th>
                </Show>
                <th class="text-right">QED</th>
              </tr>
            </thead>
            <tbody>
              <For each={scoredEntries()}>{(entry, i) => (
                <tr>
                  <td class="font-mono text-xs">{i() + 1}</td>
                  <td class="text-xs font-medium">{entry.name}</td>
                  <td class="text-xs">
                    <Show when={entry.status === 'scoring'}>
                      <span class="loading loading-spinner loading-xs" />
                    </Show>
                    <Show when={entry.status === 'done'}>
                      <span class="badge badge-success badge-xs">Done</span>
                    </Show>
                    <Show when={entry.status === 'error'}>
                      <span class="badge badge-error badge-xs" title={entry.errorMessage || ''}>Error</span>
                    </Show>
                    <Show when={entry.status === 'pending'}>
                      <span class="text-base-content/50">Pending</span>
                    </Show>
                  </td>
                  <td class="text-right font-mono text-xs">
                    {entry.vinaScore != null ? entry.vinaScore.toFixed(1) : '--'}
                  </td>
                  <Show when={state().score.cordialAvailable}>
                    <td class="text-right font-mono text-xs">
                      {entry.cordialExpectedPkd != null ? entry.cordialExpectedPkd.toFixed(1) : '--'}
                    </td>
                  </Show>
                  <td class="text-right font-mono text-xs">
                    {entry.qed != null ? entry.qed.toFixed(3) : '--'}
                  </td>
                </tr>
              )}</For>
            </tbody>
          </table>
        </div>
      </Show>

      <TerminalOutput title="Scoring Output" logs={state().logs} />

      <div class="flex justify-between mt-3">
        <button class="btn btn-ghost btn-sm" onClick={handleBack} disabled={state().score.isRunning}>
          <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 17l-5-5m0 0l5-5m-5 5h12" />
          </svg>
          Back
        </button>
        <div class="flex gap-2">
          <Show when={state().score.isRunning}>
            <button class="btn btn-error btn-sm" onClick={() => setShowStopConfirm(true)}>Cancel</button>
          </Show>
          <Show when={state().currentPhase === 'complete'}>
            <button class="btn btn-primary" onClick={() => setScoreStep('score-results')}>
              View Results
              <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </button>
          </Show>
        </div>
      </div>

      <StopConfirmModal
        isOpen={showStopConfirm()}
        title="Stop Scoring?"
        message="Are you sure you want to cancel? Scores already computed will be preserved."
        onConfirm={() => {
          setShowStopConfirm(false);
          void handleCancel();
        }}
        onCancel={() => setShowStopConfirm(false)}
      />
    </div>
  );
};

export default ScoreStepProgress;
