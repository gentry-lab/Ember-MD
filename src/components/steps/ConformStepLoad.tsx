// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, Show, createSignal, createMemo } from 'solid-js';
import { workflowStore } from '../../stores/workflow';
import { sanitizeConformOutputName } from '../../utils/jobName';
import DropZone from '../shared/DropZone';

const ConformStepLoad: Component = () => {
  const {
    state,
    setConformStep,
    setConformLigandSdf,
    setConformLigandName,
    setConformOutputName,
    setError,
  } = workflowStore;
  const api = window.electronAPI;

  const [isLoading, setIsLoading] = createSignal(false);
  const [smilesText, setSmilesText] = createSignal('');

  const detectedSmiles = createMemo(() =>
    smilesText().split('\n').map(l => l.trim()).filter(l => l.length > 0)
  );

  const loadLigandFromPath = (sdfPath: string) => {
    const name = sdfPath.split('/').pop()?.replace(/(\.sdf(\.gz)?|\.mol2?|\.mol)$/i, '') || 'ligand';
    setConformLigandSdf(sdfPath);
    setConformLigandName(name);
    setConformOutputName(sanitizeConformOutputName(name));
  };

  const handleSelectFile = async () => {
    const sdfPath = await api.selectSdfFile();
    if (!sdfPath) return;
    loadLigandFromPath(sdfPath);
  };

  const handleConvertSmiles = async () => {
    const smiles = detectedSmiles();
    if (smiles.length === 0) return;
    setIsLoading(true);
    setError(null);
    try {
      const defaultDir = await api.getDefaultOutputDir();
      const baseDir = state().customOutputDir || defaultDir;
      const tmpDir = `${baseDir}/${state().jobName}/conformers/_tmp`;
      await api.createDirectory(tmpDir);
      const result = await api.convertSmilesList(smiles, tmpDir);
      if (result.ok && result.value.length > 0) {
        const first = result.value[0];
        setConformLigandSdf(first.sdfPath);
        setConformLigandName(first.filename || 'smiles_mol');
        setConformOutputName(sanitizeConformOutputName(first.filename || 'smiles_mol'));
      } else {
        setError(result.ok ? 'No molecules converted' : (result.error?.message || 'SMILES conversion failed'));
      }
    } catch (err) {
      setError((err as Error).message);
    }
    setIsLoading(false);
  };

  const handleClear = () => {
    setConformLigandSdf(null);
    setConformLigandName(null);
    setConformOutputName('');
    setSmilesText('');
    setError(null);
  };

  const canContinue = () => !!state().conform.ligandSdfPath;

  return (
    <div class="h-full flex flex-col">
      <div class="text-center mb-3">
        <h2 class="text-xl font-bold">Load Molecule for MCMM</h2>
        <p class="text-sm text-base-content/90">Import a ligand file or enter SMILES</p>
      </div>

      <div class="flex-1 min-h-0 overflow-auto flex flex-col items-center gap-4">
        <DropZone
          accept={['.sdf', '.mol', '.mol2']}
          onFiles={(paths) => loadLigandFromPath(paths[0])}
          disabled={isLoading()}
          hoverLabel="Drop ligand (.sdf, .mol, .mol2)"
        >
        <div class="card bg-base-200 shadow-lg w-full max-w-md">
          <div class="card-body p-4">
            <Show
              when={!state().conform.ligandSdfPath}
              fallback={
                <div class="space-y-3">
                  <div class="p-3 bg-base-300 rounded-lg">
                    <p class="text-xs font-semibold">{state().conform.ligandName}</p>
                    <p class="text-[10px] font-mono text-base-content/70 break-all truncate">{state().conform.ligandSdfPath}</p>
                  </div>
                  <button class="btn btn-ghost btn-xs w-full" onClick={handleClear}>Clear</button>
                </div>
              }
            >
              <div class="space-y-3">
                <button class="btn btn-outline btn-sm w-full" onClick={handleSelectFile} disabled={isLoading()}>
                  Import (.sdf, .mol, .mol2)
                </button>

                <div>
                  <div class="flex items-center justify-between mb-1">
                    <span class="text-[10px] text-base-content/50">or enter SMILES</span>
                    <Show when={detectedSmiles().length > 0}>
                      <span class="text-[10px] font-mono text-success">
                        {detectedSmiles().length} input{detectedSmiles().length !== 1 ? 's' : ''}
                      </span>
                    </Show>
                  </div>
                  <textarea
                    class="textarea textarea-bordered text-xs font-mono w-full resize-none leading-relaxed"
                    placeholder="Enter SMILES strings (one compound per line)"
                    value={smilesText()}
                    onInput={(e) => setSmilesText(e.currentTarget.value)}
                    rows={4}
                  />
                </div>

                <button
                  class="btn btn-primary btn-sm w-full"
                  onClick={handleConvertSmiles}
                  disabled={isLoading() || detectedSmiles().length === 0}
                >
                  {isLoading() ? <span class="loading loading-spinner loading-xs" /> : 'Enter SMILES'}
                </button>
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
            Finds the global energy minimum using standard force fields. CREST uses xTB calculations throughout the entire conformational search.
          </span>
        </div>
        <button class="btn btn-primary" onClick={() => setConformStep('conform-configure')} disabled={!canContinue()}>
          Continue
          <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
          </svg>
        </button>
      </div>
    </div>
  );
};

export default ConformStepLoad;
