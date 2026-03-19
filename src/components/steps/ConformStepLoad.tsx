import { Component, Show, createSignal } from 'solid-js';
import { workflowStore } from '../../stores/workflow';

const ConformStepLoad: Component = () => {
  const {
    state,
    setConformStep,
    setConformLigandSdf,
    setConformLigandName,
    setError,
  } = workflowStore;
  const api = window.electronAPI;

  const [isLoading, setIsLoading] = createSignal(false);
  const [smilesInput, setSmilesInput] = createSignal('');
  const [inputTab, setInputTab] = createSignal<'sdf' | 'smiles'>('sdf');

  const handleSelectSdf = async () => {
    const sdfPath = await api.selectSdfFile();
    if (!sdfPath) return;
    const name = sdfPath.split('/').pop()?.replace(/\.sdf(\.gz)?$/, '') || 'ligand';
    setConformLigandSdf(sdfPath);
    setConformLigandName(name);
  };

  const handleSmiles = async () => {
    const smiles = smilesInput().trim();
    if (!smiles) return;
    setIsLoading(true);
    try {
      const defaultDir = await api.getDefaultOutputDir();
      const baseDir = state().customOutputDir || defaultDir;
      const tmpDir = `${baseDir}/${state().jobName}/conformers/_tmp`;
      await api.createDirectory(tmpDir);
      const result = await api.convertSingleMolecule(smiles, tmpDir, 'smiles');
      if (result.ok) {
        setConformLigandSdf(result.value.sdfPath);
        setConformLigandName(result.value.name || 'smiles_mol');
      } else {
        setError(result.error?.message || 'SMILES conversion failed');
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
        <h2 class="text-xl font-bold">Load Molecule</h2>
        <p class="text-sm text-base-content/90">Select a molecule for conformer search</p>
      </div>

      <div class="flex-1 min-h-0 overflow-auto flex flex-col items-center gap-4">
        {/* Input tabs */}
        <div class="tabs tabs-boxed tabs-sm">
          <button class={`tab ${inputTab() === 'sdf' ? 'tab-active' : ''}`} onClick={() => setInputTab('sdf')}>
            SDF File
          </button>
          <button class={`tab ${inputTab() === 'smiles' ? 'tab-active' : ''}`} onClick={() => setInputTab('smiles')}>
            SMILES
          </button>
        </div>

        <div class="card bg-base-200 shadow-lg w-full max-w-md">
          <div class="card-body p-4">
            <Show when={inputTab() === 'sdf'}>
              <button class="btn btn-primary btn-sm w-full" onClick={handleSelectSdf} disabled={isLoading()}>
                Select SDF File
              </button>
            </Show>

            <Show when={inputTab() === 'smiles'}>
              <div class="flex gap-2">
                <input
                  type="text"
                  class="input input-bordered input-sm flex-1 font-mono text-xs"
                  placeholder="Paste SMILES..."
                  value={smilesInput()}
                  onInput={(e) => setSmilesInput(e.currentTarget.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleSmiles(); }}
                />
                <button class="btn btn-primary btn-sm" onClick={handleSmiles} disabled={!smilesInput().trim() || isLoading()}>
                  {isLoading() ? <span class="loading loading-spinner loading-xs" /> : 'Go'}
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
