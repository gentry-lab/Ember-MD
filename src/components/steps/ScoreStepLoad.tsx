// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, Show, createEffect, createSignal } from 'solid-js';
import { workflowStore } from '../../stores/workflow';
import DropZone from '../shared/DropZone';

const ScoreStepLoad: Component = () => {
  const {
    state,
    setScoreInputDir,
    setScoreOutputDir,
    setScorePdfPaths,
    setScoreLastResult,
    setScoreStep,
    setError,
  } = workflowStore;
  const api = window.electronAPI;
  const [isScanning, setIsScanning] = createSignal(false);
  const [scanSummary, setScanSummary] = createSignal<{
    pdbCount: number;
    mtzCount: number;
    pairedCount: number;
    unpairedPdbCount: number;
  } | null>(null);

  createEffect(async () => {
    const dirPath = state().score.inputDir;
    if (!dirPath || scanSummary() || isScanning()) return;
    setIsScanning(true);
    try {
      const result = await api.scanXrayDirectory(dirPath);
      if (result.ok) {
        setScanSummary(result.value);
      }
    } finally {
      setIsScanning(false);
    }
  });

  const loadFolderFromPath = async (dirPath: string) => {
    setIsScanning(true);
    setScanSummary(null);
    setScoreInputDir(dirPath);
    setScoreOutputDir(null);
    setScorePdfPaths([]);
    setScoreLastResult(null);
    setError(null);
    try {
      const result = await api.scanXrayDirectory(dirPath);
      if (result.ok) {
        setScanSummary(result.value);
      } else {
        setError(result.error.message);
      }
    } finally {
      setIsScanning(false);
    }
  };

  const handleSelectFolder = async () => {
    const dirPath = await api.selectFolder();
    if (!dirPath) return;
    loadFolderFromPath(dirPath);
  };

  const handleClear = () => {
    setScoreInputDir(null);
    setScoreOutputDir(null);
    setScorePdfPaths([]);
    setScoreLastResult(null);
    setScanSummary(null);
    setError(null);
  };

  return (
    <div class="h-full flex flex-col">
      <div class="text-center mb-3">
        <h2 class="text-xl font-bold">Analyze X-ray Pose</h2>
        <p class="text-sm text-base-content/90">
          Compare the reported ligand pose in the PDB against the MTZ electron density map.
        </p>
      </div>

      <div class="flex-1 min-h-0 overflow-auto flex flex-col items-center gap-4">
        <DropZone
          accept={['.pdb', '.cif', '.mtz']}
          acceptFolders
          onFiles={(paths) => {
            const first = paths[0];
            const basename = first.substring(first.lastIndexOf('/') + 1);
            const hasExtension = /\.\w+$/.test(basename);
            loadFolderFromPath(hasExtension ? first.substring(0, first.lastIndexOf('/')) : first);
          }}
          hoverLabel="Drop folder with PDB + MTZ files"
        >
        <div class="card bg-base-200 shadow-lg w-full max-w-md">
          <div class="card-body p-4">
            <Show
              when={!state().score.inputDir}
              fallback={
                <div class="space-y-3">
                  <div class="p-3 bg-base-300 rounded-lg">
                    <p class="text-xs font-semibold">{state().score.inputDir?.split('/').pop() || 'Selected folder'}</p>
                    <p class="text-[10px] font-mono text-base-content/70 break-all">{state().score.inputDir}</p>
                  </div>
                  <Show when={isScanning()}>
                    <div class="text-xs text-base-content/60 flex items-center gap-2">
                      <span class="loading loading-spinner loading-xs" />
                      Scanning PDB/MTZ pairs...
                    </div>
                  </Show>
                  <Show when={!isScanning() && scanSummary()}>
                    <div class="rounded-lg bg-base-300/70 p-3 space-y-1">
                      <p class="text-xs font-semibold">
                        {scanSummary()!.pairedCount} PDB/MTZ pair{scanSummary()!.pairedCount === 1 ? '' : 's'} matched
                      </p>
                      <p class="text-[10px] text-base-content/70">
                        {scanSummary()!.pdbCount} structure{scanSummary()!.pdbCount === 1 ? '' : 's'} and {scanSummary()!.mtzCount} MTZ file{scanSummary()!.mtzCount === 1 ? '' : 's'} found
                        {scanSummary()!.unpairedPdbCount > 0 ? `, ${scanSummary()!.unpairedPdbCount} PDB-only` : ''}
                      </p>
                    </div>
                  </Show>
                  <button class="btn btn-ghost btn-xs w-full" onClick={handleClear}>Clear</button>
                </div>
              }
            >
              <div class="space-y-3">
                <button class="btn btn-outline btn-sm w-full" onClick={handleSelectFolder}>
                  Import Folder
                </button>

                <p class="text-[10px] text-base-content/60 leading-relaxed">
                  Select a folder containing matching `*.pdb` and `*.mtz` files. PDFs will be written into the current Ember project under `xray/`.
                </p>
              </div>
            </Show>
          </div>
        </div>
        </DropZone>

        <Show when={state().errorMessage}>
          <div class="alert alert-error py-2 w-full max-w-md">
            <span class="text-sm">{state().errorMessage}</span>
          </div>
        </Show>
      </div>

      <div class="mt-4 flex items-center gap-3 flex-shrink-0">
        <div class="flex-1 flex items-center gap-2 text-xs text-base-content/85 bg-base-200 rounded-lg px-3 py-2">
          <svg class="w-4 h-4 text-info flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>
            Electron density maps (.mtz) and structure files (.pdb) are analyzed algorithmically for alignment.
          </span>
        </div>
        <button
          class="btn btn-primary"
          onClick={() => setScoreStep('score-progress')}
          disabled={!state().score.inputDir}
        >
          Continue
          <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
          </svg>
        </button>
      </div>
    </div>
  );
};

export default ScoreStepLoad;
