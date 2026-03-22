import { Component, Show, createSignal, createMemo } from 'solid-js';
import { workflowStore } from '../../stores/workflow';
import { sanitizeConformOutputName } from '../../utils/jobName';

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
  const [inputTab, setInputTab] = createSignal<'sdf' | 'smiles'>('smiles');

  const detectedSmiles = createMemo(() =>
    smilesText().split('\n').map(l => l.trim()).filter(l => l.length > 0)
  );

  const handleSelectSdf = async () => {
    const sdfPath = await api.selectSdfFile();
    if (!sdfPath) return;
    const name = sdfPath.split('/').pop()?.replace(/(\.sdf(\.gz)?|\.mol2?|\.mol)$/i, '') || 'ligand';
    setConformLigandSdf(sdfPath);
    setConformLigandName(name);
    setConformOutputName(sanitizeConformOutputName(name));
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

  const canContinue = () => !!state().conform.ligandSdfPath;

  return (
    <div class="h-full flex flex-col">
      <div class="text-center mb-3">
        <h2 class="text-xl font-bold">Load Molecule for MCMM</h2>
        <p class="text-sm text-base-content/90">Select one ligand structure or paste a SMILES string</p>
      </div>

      <div class="flex-1 min-h-0 overflow-auto flex flex-col items-center gap-4">
        {/* Input tabs */}
        <div class="tabs tabs-boxed tabs-sm">
          <button class={`tab ${inputTab() === 'sdf' ? 'tab-active' : ''}`} onClick={() => setInputTab('sdf')}>
            Structure File
          </button>
          <button class={`tab ${inputTab() === 'smiles' ? 'tab-active' : ''}`} onClick={() => setInputTab('smiles')}>
            SMILES
          </button>
        </div>

        <div class="card bg-base-200 shadow-lg w-full max-w-md">
          <div class="card-body p-4">
            <Show when={inputTab() === 'sdf'}>
              <div class="space-y-2">
                <button class="btn btn-primary btn-sm w-full" onClick={handleSelectSdf} disabled={isLoading()}>
                  Select Structure File
                </button>
                <p class="text-[10px] text-base-content/70">
                  Accepted formats: `.sdf`, `.sdf.gz`, `.mol`, `.mol2`
                </p>
              </div>
            </Show>

            <Show when={inputTab() === 'smiles'}>
              <div class="space-y-2">
                <div class="flex items-center justify-between">
                  <span class="text-[10px] text-base-content/70">One SMILES per line</span>
                  <span class={`text-[10px] font-mono ${detectedSmiles().length > 0 ? 'text-success' : 'text-base-content/50'}`}>
                    {detectedSmiles().length} molecule{detectedSmiles().length !== 1 ? 's' : ''} detected
                  </span>
                </div>
                <textarea
                  class="textarea textarea-bordered text-xs font-mono w-full resize-none leading-relaxed"
                  placeholder={"CCO\nc1ccccc1\nCC(=O)Oc1ccccc1C(=O)O"}
                  value={smilesText()}
                  onInput={(e) => setSmilesText(e.currentTarget.value)}
                  rows={5}
                />
                <button
                  class="btn btn-primary btn-sm w-full"
                  onClick={handleConvertSmiles}
                  disabled={isLoading() || detectedSmiles().length === 0}
                >
                  {isLoading() ? <span class="loading loading-spinner loading-xs" /> : `Convert ${detectedSmiles().length} molecule${detectedSmiles().length !== 1 ? 's' : ''}`}
                </button>
              </div>
            </Show>

            <Show when={state().conform.ligandSdfPath}>
              <div class="mt-3 p-2 bg-base-300 rounded-lg">
                <p class="text-xs font-semibold">{state().conform.ligandName}</p>
                <p class="text-[10px] font-mono text-base-content/70 break-all">{state().conform.ligandSdfPath}</p>
              </div>
            </Show>
          </div>
        </div>

        <Show when={state().errorMessage}>
          <div class="alert alert-error py-2 w-full max-w-md">
            <span class="text-sm">{state().errorMessage}</span>
          </div>
        </Show>
      </div>

      {/* Navigation */}
      <div class="flex justify-end mt-3 flex-shrink-0">
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
