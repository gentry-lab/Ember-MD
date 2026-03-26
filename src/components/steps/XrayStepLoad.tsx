// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, Show, createSignal } from 'solid-js';
import { workflowStore } from '../../stores/workflow';

const XrayStepLoad: Component = () => {
  const {
    state,
    setError,
    setXrayInputDir,
    setXrayScanResult,
    setXrayOutputDir,
    setXrayResult,
    setXrayStep,
  } = workflowStore;
  const api = window.electronAPI;
  const [isScanning, setIsScanning] = createSignal(false);

  const scanDirectory = async (dirPath: string) => {
    setIsScanning(true);
    setError(null);
    setXrayInputDir(dirPath);
    setXrayOutputDir(null);
    setXrayResult(null);
    try {
      const result = await api.scanXrayDirectory(dirPath);
      if (result.ok) {
        setXrayScanResult(result.value);
      } else {
        setXrayScanResult(null);
        setError(result.error.message);
      }
    } catch (error) {
      setXrayScanResult(null);
      setError((error as Error).message);
    }
    setIsScanning(false);
  };

  const handleSelectFolder = async () => {
    const dirPath = await api.selectFolder();
    if (!dirPath) return;
    await scanDirectory(dirPath);
  };

  const canContinue = () => (state().xray.scanResult?.pairedCount || 0) > 0;

  return (
    <div class="h-full flex flex-col">
      <div class="text-center mb-3">
        <h2 class="text-xl font-bold">X-ray Analyzer</h2>
        <p class="text-sm text-base-content/90">
          Select a folder containing PDB/CIF structures and matching MTZ files to generate validation reports.
        </p>
      </div>

      <div class="flex-1 min-h-0 overflow-auto flex items-center justify-center">
        <div class="card bg-base-200 shadow-lg w-full max-w-xl">
          <div class="card-body p-5 space-y-4">
            <div class="space-y-2">
              <div class="text-xs font-semibold uppercase tracking-wide text-base-content/60">Input folder</div>
              <div class="rounded-lg border border-base-300 bg-base-100/70 px-3 py-2 text-sm font-mono break-all min-h-[3rem] flex items-center">
                <Show when={state().xray.inputDir} fallback={<span class="text-base-content/45">No folder selected</span>}>
                  {state().xray.inputDir}
                </Show>
              </div>
            </div>

            <div class="flex gap-2">
              <button class="btn btn-primary btn-sm flex-1" onClick={handleSelectFolder} disabled={isScanning()}>
                <Show when={isScanning()} fallback={state().xray.inputDir ? 'Choose Different Folder' : 'Choose Folder'}>
                  <span class="loading loading-spinner loading-sm" />
                </Show>
              </button>
              <button
                class="btn btn-outline btn-sm"
                onClick={() => state().xray.inputDir && void scanDirectory(state().xray.inputDir!)}
                disabled={!state().xray.inputDir || isScanning()}
              >
                Rescan
              </button>
            </div>

            <Show when={state().xray.scanResult}>
              <div class="rounded-lg border border-base-300 bg-base-100/70 p-4">
                <div class="text-xs font-semibold uppercase tracking-wide text-base-content/60 mb-2">Scan summary</div>
                <div class="grid grid-cols-2 gap-2 text-sm">
                  <div class="rounded border border-base-300 bg-base-100 px-3 py-2">
                    <div class="text-[10px] text-base-content/55 uppercase">Structures</div>
                    <div class="font-semibold">{state().xray.scanResult?.pdbCount ?? 0}</div>
                  </div>
                  <div class="rounded border border-base-300 bg-base-100 px-3 py-2">
                    <div class="text-[10px] text-base-content/55 uppercase">MTZ files</div>
                    <div class="font-semibold">{state().xray.scanResult?.mtzCount ?? 0}</div>
                  </div>
                  <div class="rounded border border-base-300 bg-base-100 px-3 py-2">
                    <div class="text-[10px] text-base-content/55 uppercase">Matched pairs</div>
                    <div class={`font-semibold ${canContinue() ? 'text-success' : 'text-warning'}`}>
                      {state().xray.scanResult?.pairedCount ?? 0}
                    </div>
                  </div>
                  <div class="rounded border border-base-300 bg-base-100 px-3 py-2">
                    <div class="text-[10px] text-base-content/55 uppercase">Unpaired structures</div>
                    <div class="font-semibold">{state().xray.scanResult?.unpairedPdbCount ?? 0}</div>
                  </div>
                </div>
                <Show when={!canContinue()}>
                  <p class="mt-3 text-xs text-warning">
                    At least one matched PDB/CIF and MTZ pair is required before analysis can start.
                  </p>
                </Show>
              </div>
            </Show>

            <Show when={state().errorMessage}>
              <div class="alert alert-error py-2">
                <span class="text-sm">{state().errorMessage}</span>
              </div>
            </Show>
          </div>
        </div>
      </div>

      <div class="flex justify-end mt-3">
        <button class="btn btn-primary" onClick={() => setXrayStep('xray-progress')} disabled={!canContinue()}>
          Run Analysis
        </button>
      </div>
    </div>
  );
};

export default XrayStepLoad;
