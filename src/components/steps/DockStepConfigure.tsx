import { Component, Show, onMount, createSignal, createMemo } from 'solid-js';
import { workflowStore } from '../../stores/workflow';
import { buildDockFolderName } from '../../utils/jobName';

const DockStepConfigure: Component = () => {
  const {
    state,
    setDockStep,
    setDockConfig,
    setDockCordialConfig,
    setDockProtonationConfig,
    setDockStereoisomerConfig,
    setDockConformerConfig,
    setDockCordialAvailable,
  } = workflowStore;
  const api = window.electronAPI;
  const [cordialChecked, setCordialChecked] = createSignal(false);

  onMount(async () => {
    try {
      const available = await api.checkCordialInstalled();
      setDockCordialAvailable(available);
      setDockCordialConfig({ enabled: available });
    } catch {
      setDockCordialAvailable(false);
      setDockCordialConfig({ enabled: false });
    }
    setCordialChecked(true);
  });

  const outputFolderName = createMemo(() =>
    buildDockFolderName({
      referenceLigandId: state().dock.referenceLigandId,
      numLigands: state().dock.ligandMolecules.length,
    })
  );

  const totalLigands = createMemo(() => state().dock.ligandMolecules.length);
  const totalPoses = createMemo(() => state().dock.config.numPoses * totalLigands());

  const canStart = createMemo(() => {
    const c = state().dock.config;
    return c.exhaustiveness >= 1 && c.exhaustiveness <= 64
      && c.numPoses >= 1 && c.numPoses <= 20
      && c.autoboxAdd >= 2 && c.autoboxAdd <= 8
      && totalLigands() > 0;
  });

  return (
    <div class="h-full flex flex-col">
      <div class="text-center mb-3">
        <h2 class="text-xl font-bold">Configure Docking</h2>
        <p class="text-sm text-base-content/90">AutoDock Vina parameters</p>
      </div>

      <div class="flex-1 min-h-0 overflow-auto grid grid-cols-2 gap-4 content-start">
        {/* Left — Summary */}
        <div class="flex flex-col gap-3">
          <div class="card bg-base-200 shadow-lg">
            <div class="card-body p-4">
              <h3 class="text-sm font-semibold mb-2">Output</h3>
              <div class="bg-base-300 rounded-lg px-3 py-2">
                <span class="text-xs font-mono break-all">{state().jobName}/docking/{outputFolderName()}</span>
              </div>
              <div class="space-y-1.5 text-xs mt-3">
                <div class="flex justify-between">
                  <span>Ligands</span>
                  <span class="font-mono font-bold">{totalLigands()}</span>
                </div>
                <div class="flex justify-between">
                  <span>Poses / ligand</span>
                  <span class="font-mono">{state().dock.config.numPoses}</span>
                </div>
                <div class="flex justify-between border-t border-base-300 pt-1.5">
                  <span>Total poses</span>
                  <span class="font-mono font-bold">{totalPoses()}</span>
                </div>
              </div>
            </div>
          </div>

          <div class="card bg-base-200 shadow-lg">
            <div class="card-body p-4">
              <h3 class="text-sm font-semibold mb-2">ML Rescoring</h3>
              <Show
                when={cordialChecked()}
                fallback={
                  <div class="flex items-center gap-2">
                    <span class="loading loading-spinner loading-xs" />
                    <span class="text-xs">Checking CORDIAL...</span>
                  </div>
                }
              >
                <label class="label cursor-pointer py-0">
                  <span class={`label-text text-xs ${!state().dock.cordialAvailable ? 'opacity-50' : ''}`}>
                    CORDIAL rescoring
                  </span>
                  <input
                    type="checkbox"
                    class="checkbox checkbox-sm checkbox-primary"
                    checked={state().dock.cordialConfig.enabled}
                    disabled={!state().dock.cordialAvailable}
                    onChange={(e) => setDockCordialConfig({ enabled: e.currentTarget.checked })}
                  />
                </label>
                <Show when={!state().dock.cordialAvailable}>
                  <p class="text-[10px] text-base-content/70 ml-1 mt-0.5">Not installed</p>
                </Show>
              </Show>
            </div>
          </div>
        </div>

        {/* Right — Parameters */}
        <div class="card bg-base-200 shadow-lg">
          <div class="card-body p-4 overflow-auto">
            <h3 class="text-sm font-semibold mb-3">Parameters</h3>

            <div class="grid grid-cols-2 gap-3">
              {/* Exhaustiveness */}
              <div class="form-control">
                <label class="label py-0.5">
                  <span class="label-text text-xs">Exhaustiveness</span>
                </label>
                <input
                  type="number"
                  class="input input-bordered input-sm w-full"
                  value={state().dock.config.exhaustiveness}
                  min={1} max={32}
                  onInput={(e) => setDockConfig({ exhaustiveness: Number(e.currentTarget.value) || 8 })}
                />
              </div>

              {/* Poses */}
              <div class="form-control">
                <label class="label py-0.5">
                  <span class="label-text text-xs">Poses per ligand</span>
                </label>
                <input
                  type="number"
                  class="input input-bordered input-sm w-full"
                  value={state().dock.config.numPoses}
                  min={1} max={20}
                  onInput={(e) => setDockConfig({ numPoses: Number(e.currentTarget.value) || 9 })}
                />
              </div>

              {/* Autobox margin */}
              <div class="form-control">
                <label class="label py-0.5">
                  <span class="label-text text-xs">{"Autobox margin (\u00C5)"}</span>
                </label>
                <input
                  type="number"
                  class="input input-bordered input-sm w-full"
                  value={state().dock.config.autoboxAdd}
                  min={2} max={8} step={0.5}
                  onInput={(e) => setDockConfig({ autoboxAdd: Number(e.currentTarget.value) || 4 })}
                />
              </div>

              {/* Seed */}
              <div class="form-control">
                <label class="label py-0.5">
                  <span class="label-text text-xs">Random seed</span>
                </label>
                <input
                  type="number"
                  class="input input-bordered input-sm w-full"
                  value={state().dock.config.seed}
                  min={0}
                  onInput={(e) => setDockConfig({ seed: Number(e.currentTarget.value) || 0 })}
                  placeholder="0 = random"
                />
              </div>
            </div>

            {/* Divider */}
            <div class="border-t border-base-300 my-3" />

            {/* Toggles */}
            <div class="space-y-2">
              <label class="label cursor-pointer py-0.5">
                <span class="label-text text-xs flex items-center gap-1.5">
                  Core-constrained alignment (MCS)
                  <span class="badge badge-xs badge-warning font-normal" title="Experimental — works best with congeneric series sharing a common scaffold">exp</span>
                </span>
                <input
                  type="checkbox"
                  class="checkbox checkbox-sm checkbox-primary"
                  checked={state().dock.config.coreConstrained}
                  onChange={(e) => setDockConfig({ coreConstrained: e.currentTarget.checked })}
                />
              </label>

              <label class="label cursor-pointer py-0.5">
                <span class="label-text text-xs">Protonation enumeration</span>
                <input
                  type="checkbox"
                  class="checkbox checkbox-sm checkbox-primary"
                  checked={state().dock.protonationConfig.enabled}
                  onChange={(e) => setDockProtonationConfig({ enabled: e.currentTarget.checked })}
                />
              </label>
              <Show when={state().dock.protonationConfig.enabled}>
                <div class="flex gap-2 ml-6">
                  <div class="form-control flex-1">
                    <label class="label py-0"><span class="label-text text-[10px]">pH min</span></label>
                    <input type="number" class="input input-bordered input-xs w-full"
                      value={state().dock.protonationConfig.phMin} min={0} max={14} step={0.1}
                      onInput={(e) => setDockProtonationConfig({ phMin: Number(e.currentTarget.value) || 6.4 })}
                    />
                  </div>
                  <div class="form-control flex-1">
                    <label class="label py-0"><span class="label-text text-[10px]">pH max</span></label>
                    <input type="number" class="input input-bordered input-xs w-full"
                      value={state().dock.protonationConfig.phMax} min={0} max={14} step={0.1}
                      onInput={(e) => setDockProtonationConfig({ phMax: Number(e.currentTarget.value) || 8.4 })}
                    />
                  </div>
                </div>
              </Show>

              <label class="label cursor-pointer py-0.5">
                <span class="label-text text-xs">Sample enantiomers</span>
                <input
                  type="checkbox"
                  class="checkbox checkbox-sm checkbox-primary"
                  checked={state().dock.stereoisomerConfig.enabled}
                  onChange={(e) => setDockStereoisomerConfig({ enabled: e.currentTarget.checked })}
                />
              </label>

              <label class="label cursor-pointer py-0.5">
                <span class="label-text text-xs">Conformer generation (ETKDG)</span>
                <input
                  type="checkbox"
                  class="checkbox checkbox-sm checkbox-primary"
                  checked={state().dock.conformerConfig.method !== 'none'}
                  onChange={(e) => setDockConformerConfig({ method: e.currentTarget.checked ? 'etkdg' : 'none' })}
                />
              </label>
              <Show when={state().dock.conformerConfig.method !== 'none'}>
                <div class="flex gap-2 ml-6">
                  <div class="form-control flex-1">
                    <label class="label py-0"><span class="label-text text-[10px]">Max conformers</span></label>
                    <input type="number" class="input input-bordered input-xs w-full"
                      value={state().dock.conformerConfig.maxConformers} min={1} max={100}
                      onInput={(e) => setDockConformerConfig({ maxConformers: Number(e.currentTarget.value) || 5 })}
                    />
                  </div>
                  <div class="form-control flex-1">
                    <label class="label py-0"><span class="label-text text-[10px]">{"RMSD cutoff (\u00C5)"}</span></label>
                    <input type="number" class="input input-bordered input-xs w-full"
                      value={state().dock.conformerConfig.rmsdCutoff} min={0.1} max={5.0} step={0.1}
                      onInput={(e) => setDockConformerConfig({ rmsdCutoff: Number(e.currentTarget.value) || 1.0 })}
                    />
                  </div>
                </div>
              </Show>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div class="flex justify-between mt-3 flex-shrink-0">
        <button class="btn btn-ghost btn-sm" onClick={() => setDockStep('dock-load')}>
          <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 17l-5-5m0 0l5-5m-5 5h12" />
          </svg>
          Back
        </button>
        <button class="btn btn-primary" onClick={() => setDockStep('dock-progress')} disabled={!canStart()}>
          Start Docking
          <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
          </svg>
        </button>
      </div>
    </div>
  );
};

export default DockStepConfigure;
