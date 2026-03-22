import { Component, Show, onMount, createSignal, createMemo } from 'solid-js';
import path from 'path';
import { workflowStore } from '../../stores/workflow';
import { buildDockFolderName, buildDockConformRunFolderName } from '../../utils/jobName';
import { projectPaths } from '../../utils/projectPaths';

const DockStepConfigure: Component = () => {
  const {
    state,
    setDockStep,
    setDockConfig,
    setDockCordialConfig,
    setDockProtonationConfig,
    setDockStereoisomerConfig,
    setDockConformerConfig,
    setDockRefinementConfig,
    setDockXtbConfig,
    setDockCordialAvailable,
    setDockCachedConformerPaths,
  } = workflowStore;
  const api = window.electronAPI;
  const [cordialChecked, setCordialChecked] = createSignal(false);
  const [cachedConformers, setCachedConformers] = createSignal<string[]>([]);
  const [cachedConformersUsed, setCachedConformersUsed] = createSignal(false);

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

    // Scan for cached MCMM/ETKDG conformers from previous runs
    try {
      const defaultDir = await api.getDefaultOutputDir();
      const baseOutputDir = state().customOutputDir || defaultDir;
      const jobName = state().jobName.trim();
      const dockFolder = buildDockFolderName({
        referenceLigandId: state().dock.referenceLigandId,
        numLigands: state().dock.ligandMolecules.length,
      });
      const paths = projectPaths(baseOutputDir, jobName);
      const dockPaths = paths.docking(dockFolder);

      for (const method of ['mcmm', 'etkdg'] as const) {
        const conformerRunFolder = buildDockConformRunFolderName({
          referenceLigandId: state().dock.referenceLigandId,
          numLigands: state().dock.ligandMolecules.length,
          method,
          maxConformers: state().dock.conformerConfig.maxConformers,
        });
        const canonicalDir = paths.conformers(conformerRunFolder);
        const legacyDir = path.join(dockPaths.prep, method);

        try {
          const canonicalSdfs = await api.listSdfInDirectory(canonicalDir);
          if (canonicalSdfs.length > 0) {
            setCachedConformers(canonicalSdfs.map((file) => path.join(canonicalDir, file)));
            return;
          }
        } catch { /* canonical dir doesn't exist */ }

        try {
          const legacySdfs = await api.listSdfInDirectory(legacyDir);
          if (legacySdfs.length > 0) {
            setCachedConformers(legacySdfs.map((file) => path.join(legacyDir, file)));
            return;
          }
        } catch { /* legacy dir doesn't exist */ }
      }
    } catch { /* scan failed, no cached conformers */ }
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

      <div class="flex-1 min-h-0 overflow-auto grid grid-cols-2 gap-3 content-start">
        {/* Top Left — Output Summary */}
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

        {/* Top Right — Docking Parameters */}
        <div class="card bg-base-200 shadow-lg">
          <div class="card-body p-4">
            <h3 class="text-sm font-semibold mb-2">Docking</h3>
            <div class="grid grid-cols-2 gap-3">
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
            </div>
            <label class="label cursor-pointer py-0.5 mt-1">
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
          </div>
        </div>

        {/* Bottom Left — Ligand Preparation */}
        <div class="card bg-base-200 shadow-lg">
          <div class="card-body p-4 overflow-auto">
            <h3 class="text-sm font-semibold mb-2">Ligand Preparation</h3>
            <div class="space-y-2">
              {/* Protonation */}
              <label class="label cursor-pointer py-0.5">
                <span class="label-text text-xs">Protonation states (Molscrub)</span>
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

              {/* Stereoisomers */}
              <label class="label cursor-pointer py-0.5">
                <span class="label-text text-xs">Enumerate enantiomers (RDKit)</span>
                <input
                  type="checkbox"
                  class="checkbox checkbox-sm checkbox-primary"
                  checked={state().dock.stereoisomerConfig.enabled}
                  onChange={(e) => setDockStereoisomerConfig({ enabled: e.currentTarget.checked })}
                />
              </label>

              <div class="border-t border-base-300 my-1" />

              {/* Conformer Sampling */}
              <h4 class="text-xs font-semibold text-base-content/70">Conformer Sampling</h4>
              <div class="flex items-center justify-between py-0.5">
                <span class="label-text text-xs">
                  Method
                  <Show when={state().dock.conformerConfig.method === 'mcmm'}>
                    <span class="text-base-content/50"> (Sage 2.3.0 + OBC2)</span>
                  </Show>
                  <Show when={state().dock.conformerConfig.method === 'etkdg'}>
                    <span class="text-base-content/50"> (RDKit geometric optimization)</span>
                  </Show>
                </span>
                <select
                  class="select select-bordered select-xs w-28"
                  value={state().dock.conformerConfig.method}
                  onChange={(e) => {
                    const method = e.currentTarget.value as 'none' | 'etkdg' | 'mcmm';
                    const updates: Record<string, unknown> = { method };
                    if (method === 'mcmm' && state().dock.conformerConfig.maxConformers <= 10) {
                      updates.maxConformers = 50;
                    }
                    setDockConformerConfig(updates);
                  }}
                >
                  <option value="none">Simple</option>
                  <option value="etkdg">ETKDG</option>
                  <option value="mcmm">MCMM</option>
                </select>
              </div>
              <Show when={state().dock.conformerConfig.method !== 'none'}>
                <div class="flex gap-2 ml-6">
                  <div class="form-control flex-1">
                    <label class="label py-0"><span class="label-text text-[10px]">Max conformers</span></label>
                    <input type="number" class="input input-bordered input-xs w-full"
                      value={state().dock.conformerConfig.maxConformers} min={1} max={500}
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
              <Show when={state().dock.conformerConfig.method === 'mcmm'}>
                <div class="flex gap-2 ml-6">
                  <div class="form-control flex-1">
                    <label class="label py-0"><span class="label-text text-[10px]">Search steps</span></label>
                    <input type="number" class="input input-bordered input-xs w-full"
                      value={state().dock.conformerConfig.mcmmSteps} min={10} max={5000}
                      onInput={(e) => setDockConformerConfig({ mcmmSteps: Number(e.currentTarget.value) || 1000 })}
                    />
                  </div>
                  <div class="form-control flex-1">
                    <label class="label py-0"><span class="label-text text-[10px]">Temperature (K)</span></label>
                    <input type="number" class="input input-bordered input-xs w-full"
                      value={state().dock.conformerConfig.mcmmTemperature} min={100} max={1000}
                      onInput={(e) => setDockConformerConfig({ mcmmTemperature: Number(e.currentTarget.value) || 298 })}
                    />
                  </div>
                </div>
                <label class="label cursor-pointer py-0 ml-6">
                  <span class="label-text text-[10px]">Sample amide cis/trans</span>
                  <input
                    type="checkbox"
                    class="checkbox checkbox-xs checkbox-primary"
                    checked={state().dock.conformerConfig.sampleAmides}
                    onChange={(e) => setDockConformerConfig({ sampleAmides: e.currentTarget.checked })}
                  />
                </label>
              </Show>
              <Show when={state().dock.conformerConfig.method === 'none' && cachedConformers().length > 0 && !cachedConformersUsed()}>
                <div class="flex items-center gap-2 ml-6 mt-1 bg-info/10 rounded px-2 py-1">
                  <span class="text-[10px]">{cachedConformers().length} cached conformers found</span>
                  <button class="btn btn-xs btn-primary" onClick={() => {
                    setDockCachedConformerPaths(cachedConformers());
                    setCachedConformersUsed(true);
                  }}>Use</button>
                </div>
              </Show>
              <Show when={cachedConformersUsed()}>
                <div class="flex items-center gap-2 ml-6 mt-1 text-[10px] text-success">
                  Using {state().dock.cachedConformerPaths.length} cached conformers
                </div>
              </Show>

              <div class="border-t border-base-300 my-2" />
              <p class="text-[10px] text-base-content/50">
                {state().dock.conformerConfig.method === 'crest'
                  ? 'CREST conformers already at GFN2-xTB level.'
                  : 'Ligands minimized with GFN2-xTB before docking.'}
              </p>
            </div>
          </div>
        </div>

        {/* Bottom Right — Post-Docking */}
        <div class="card bg-base-200 shadow-lg">
          <div class="card-body p-4">
            <h3 class="text-sm font-semibold mb-2">Post-Docking</h3>
            <div class="space-y-2">
              <label class="label cursor-pointer py-0.5">
                <span class="label-text text-xs">Pocket refinement</span>
                <input
                  type="checkbox"
                  class="checkbox checkbox-sm checkbox-primary"
                  checked={state().dock.refinementConfig.enabled}
                  onChange={(e) => setDockRefinementConfig({ enabled: e.currentTarget.checked })}
                />
              </label>
              <Show when={state().dock.refinementConfig.enabled}>
                <div class="flex items-center justify-between py-0.5 ml-6">
                  <span class="label-text text-[10px]">
                    Charges
                    <span class="text-base-content/50">
                      {state().dock.refinementConfig.chargeMethod === 'am1bcc'
                        ? ' (NAGL AM1-BCC — MD quality)'
                        : ' (Gasteiger — fast)'}
                    </span>
                  </span>
                  <select
                    class="select select-bordered select-xs w-24"
                    value={state().dock.refinementConfig.chargeMethod}
                    onChange={(e) => setDockRefinementConfig({ chargeMethod: e.currentTarget.value as 'gasteiger' | 'am1bcc' })}
                  >
                    <option value="am1bcc">AM1-BCC</option>
                    <option value="gasteiger">Gasteiger</option>
                  </select>
                </div>
                <p class="text-[10px] text-base-content/50 ml-1">Sage 2.3.0 + OBC2 implicit solvent</p>
              </Show>

              <div class="border-t border-base-300 my-2" />

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
