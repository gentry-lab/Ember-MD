// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, For, Show, createSignal, createMemo } from 'solid-js';
import { workflowStore } from '../../stores/workflow';
import MoleculeDetailPanel from '../shared/MoleculeDetailPanel';
import type { DetailScore } from '../shared/MoleculeDetailPanel';

type SortKey = 'name' | 'vinaScore' | 'cordialExpectedPkd' | 'qed';
type SortDir = 'asc' | 'desc';

const ScoreStepResults: Component = () => {
  const { state, resetScore } = workflowStore;
  const api = window.electronAPI;
  const [sortKey, setSortKey] = createSignal<SortKey>('vinaScore');
  const [sortDir, setSortDir] = createSignal<SortDir>('asc');
  const [selectedIndex, setSelectedIndex] = createSignal<number | null>(null);

  const scoredEntries = () => {
    const entries = state().score.entries.filter((e) => e.status === 'done' || e.vinaScore != null);
    const key = sortKey();
    const dir = sortDir();
    return [...entries].sort((a, b) => {
      let aVal: string | number | null;
      let bVal: string | number | null;
      if (key === 'name') {
        aVal = a.name.toLowerCase();
        bVal = b.name.toLowerCase();
      } else {
        aVal = a[key];
        bVal = b[key];
      }
      if (aVal == null && bVal == null) return 0;
      if (aVal == null) return 1;
      if (bVal == null) return -1;
      const cmp = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
      return dir === 'asc' ? cmp : -cmp;
    });
  };

  const errorEntries = () => state().score.entries.filter((e) => e.status === 'error');

  const hasCordial = () => state().score.cordialAvailable;

  const handleSort = (key: SortKey) => {
    if (sortKey() === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  };

  const sortIcon = (key: SortKey) => {
    if (sortKey() !== key) return '';
    return sortDir() === 'asc' ? ' \u25B2' : ' \u25BC';
  };

  const vinaColor = (score: number | null) => {
    if (score == null) return '';
    if (score <= -7) return 'text-success';
    if (score <= -5) return 'text-warning';
    return 'text-error';
  };

  const qedColor = (qed: number | null) => {
    if (qed == null) return '';
    if (qed >= 0.5) return 'text-success';
    if (qed >= 0.3) return 'text-warning';
    return 'text-error';
  };

  const selectedEntry = createMemo(() => {
    const idx = selectedIndex();
    if (idx == null) return null;
    return scoredEntries()[idx] ?? null;
  });

  const selectedScores = createMemo((): DetailScore[] => {
    const entry = selectedEntry();
    if (!entry) return [];
    const scores: DetailScore[] = [];
    if (entry.vinaScore != null) scores.push({ label: 'Vina', value: entry.vinaScore, unit: 'kcal/mol' });
    if (entry.cordialExpectedPkd != null) scores.push({ label: 'CORDIAL pKd', value: entry.cordialExpectedPkd });
    if (entry.qed != null) scores.push({ label: 'QED', value: entry.qed });
    return scores;
  });

  const handleRowClick = (idx: number) => {
    setSelectedIndex(selectedIndex() === idx ? null : idx);
  };

  const handleExportCsv = async () => {
    const entries = state().score.entries;
    const entriesJson = JSON.stringify(entries.map((e) => ({
      name: e.name,
      vinaScore: e.vinaScore,
      cordialExpectedPkd: e.cordialExpectedPkd,
      cordialPHighAffinity: e.cordialPHighAffinity,
      qed: e.qed,
      isPrepared: e.isPrepared,
      status: e.status,
    })));
    const outputDir = state().score.outputDir;
    if (!outputDir) return;
    const csvPath = `${outputDir}/results/score_results.csv`;
    try {
      await api.exportScoreCsv(entriesJson, csvPath);
      void api.openFolder(csvPath);
    } catch { /* ignore */ }
  };

  const handleOpenFolder = () => {
    const dir = state().score.outputDir;
    if (dir) void api.openFolder(dir);
  };

  return (
    <div class="h-full flex flex-col">
      <div class="flex items-center justify-between mb-3">
        <div>
          <h2 class="text-xl font-bold">Score Results</h2>
          <p class="text-sm text-base-content/90">
            {scoredEntries().length} complex{scoredEntries().length === 1 ? '' : 'es'} scored
            {errorEntries().length > 0 ? `, ${errorEntries().length} error${errorEntries().length === 1 ? '' : 's'}` : ''}
          </p>
        </div>
        <div class="flex gap-2">
          <button class="btn btn-xs btn-outline" onClick={handleExportCsv}>
            Export CSV
          </button>
          <button class="btn btn-xs btn-ghost" onClick={handleOpenFolder} disabled={!state().score.outputDir}>
            Open Folder
          </button>
        </div>
      </div>

      <div class="flex-1 min-h-0 flex gap-3">
        {/* Left: results table */}
        <div class="flex-1 min-w-0 overflow-auto">
          <Show
            when={scoredEntries().length > 0}
            fallback={
              <div class="rounded-lg border border-warning/30 bg-warning/10 p-3 text-sm">
                No complexes were scored successfully.
              </div>
            }
          >
            <table class="table table-xs table-zebra w-full">
              <thead>
                <tr>
                  <th class="w-10">#</th>
                  <th class="cursor-pointer hover:text-primary" onClick={() => handleSort('name')}>
                    Name{sortIcon('name')}
                  </th>
                  <th class="text-right cursor-pointer hover:text-primary" onClick={() => handleSort('vinaScore')}>
                    Vina{sortIcon('vinaScore')}
                  </th>
                  <Show when={hasCordial()}>
                    <th class="text-right cursor-pointer hover:text-primary" onClick={() => handleSort('cordialExpectedPkd')}>
                      CORDIAL{sortIcon('cordialExpectedPkd')}
                    </th>
                  </Show>
                  <th class="text-right cursor-pointer hover:text-primary" onClick={() => handleSort('qed')}>
                    QED{sortIcon('qed')}
                  </th>
                  <th class="text-center">Prep</th>
                </tr>
              </thead>
              <tbody>
                <For each={scoredEntries()}>{(entry, i) => (
                  <tr
                    class={`cursor-pointer hover:bg-base-300 ${selectedIndex() === i() ? 'bg-primary/10' : ''}`}
                    onClick={() => handleRowClick(i())}
                  >
                    <td class="font-mono text-xs">{i() + 1}</td>
                    <td class="text-xs font-medium">{entry.name}</td>
                    <td class={`text-right font-mono text-xs ${vinaColor(entry.vinaScore)}`}>
                      {entry.vinaScore != null ? entry.vinaScore.toFixed(1) : '--'}
                    </td>
                    <Show when={hasCordial()}>
                      <td class="text-right font-mono text-xs">
                        {entry.cordialExpectedPkd != null ? entry.cordialExpectedPkd.toFixed(1) : '--'}
                      </td>
                    </Show>
                    <td class={`text-right font-mono text-xs ${qedColor(entry.qed)}`}>
                      {entry.qed != null ? entry.qed.toFixed(3) : '--'}
                    </td>
                    <td class="text-center">
                      <Show when={entry.isPrepared}>
                        <span class="badge badge-success badge-xs">Yes</span>
                      </Show>
                      <Show when={!entry.isPrepared}>
                        <span class="badge badge-info badge-xs">Auto</span>
                      </Show>
                    </td>
                  </tr>
                )}</For>
              </tbody>
            </table>
          </Show>

          <Show when={errorEntries().length > 0}>
            <div class="mt-3 rounded-lg border border-error/30 bg-error/10 p-3">
              <p class="text-xs font-semibold mb-1">{errorEntries().length} failed</p>
              <For each={errorEntries()}>{(entry) => (
                <p class="text-[10px] text-base-content/70">
                  {entry.name}: {entry.errorMessage || 'Unknown error'}
                </p>
              )}</For>
            </div>
          </Show>
        </div>

        {/* Right: molecule detail panel */}
        <Show when={selectedEntry()}>
          <MoleculeDetailPanel
            sdfPath={selectedEntry()!.extractedLigandSdfPath}
            scores={selectedScores()}
            label={selectedEntry()!.name}
            sublabel={selectedEntry()!.isPrepared ? 'Prepared' : 'Auto-prepared'}
          />
        </Show>
      </div>

      <div class="flex justify-between mt-3 flex-shrink-0">
        <button class="btn btn-ghost btn-sm" onClick={() => resetScore()}>
          New Job
        </button>
      </div>
    </div>
  );
};

export default ScoreStepResults;
