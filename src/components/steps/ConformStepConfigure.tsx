import { Component, Show } from 'solid-js';
import { workflowStore } from '../../stores/workflow';
import type { ConformerMethod } from '../../../shared/types/dock';

const ConformStepConfigure: Component = () => {
  const {
    state,
    setConformStep,
    setConformConfig,
  } = workflowStore;

  return (
    <div class="h-full flex flex-col">
      <div class="text-center mb-3">
        <h2 class="text-xl font-bold">Configure Conformer Search</h2>
        <p class="text-sm text-base-content/90">{state().conform.ligandName}</p>
      </div>

      <div class="flex-1 min-h-0 overflow-auto flex flex-col items-center gap-4">
        <div class="card bg-base-200 shadow-lg w-full max-w-md">
          <div class="card-body p-4">
            <h3 class="text-sm font-semibold mb-2">Method</h3>

            <div class="flex items-center justify-between py-0.5">
              <span class="label-text text-xs">Conformer search</span>
              <select
                class="select select-bordered select-xs w-28"
                value={state().conform.config.method}
                onChange={(e) => {
                  const method = e.currentTarget.value as ConformerMethod;
                  const updates: Record<string, unknown> = { method };
                  if (method === 'mcmm' && state().conform.config.maxConformers <= 10) {
                    updates.maxConformers = 50;
                  }
                  setConformConfig(updates);
                }}
              >
                <option value="etkdg">ETKDG</option>
                <option value="mcmm">MCMM</option>
              </select>
            </div>

            <Show when={state().conform.config.method === 'mcmm'}>
              <p class="text-[10px] text-base-content/50 ml-1 mt-0.5">Sage 2.3.0 + OBC2 implicit solvent</p>
            </Show>

            <div class="border-t border-base-300 my-2" />

            <h3 class="text-sm font-semibold mb-2">Parameters</h3>

            <div class="flex gap-3">
              <div class="form-control flex-1">
                <label class="label py-0"><span class="label-text text-[10px]">Max conformers</span></label>
                <input type="number" class="input input-bordered input-xs w-full"
                  value={state().conform.config.maxConformers} min={1} max={500}
                  onInput={(e) => setConformConfig({ maxConformers: Number(e.currentTarget.value) || 50 })}
                />
              </div>
              <div class="form-control flex-1">
                <label class="label py-0"><span class="label-text text-[10px]">{"RMSD cutoff (\u00C5)"}</span></label>
                <input type="number" class="input input-bordered input-xs w-full"
                  value={state().conform.config.rmsdCutoff} min={0.1} max={5.0} step={0.1}
                  onInput={(e) => setConformConfig({ rmsdCutoff: Number(e.currentTarget.value) || 1.0 })}
                />
              </div>
            </div>

            <Show when={state().conform.config.method === 'mcmm'}>
              <div class="flex gap-3 mt-1">
                <div class="form-control flex-1">
                  <label class="label py-0"><span class="label-text text-[10px]">Search steps</span></label>
                  <input type="number" class="input input-bordered input-xs w-full"
                    value={state().conform.config.mcmmSteps} min={10} max={5000}
                    onInput={(e) => setConformConfig({ mcmmSteps: Number(e.currentTarget.value) || 1000 })}
                  />
                </div>
                <div class="form-control flex-1">
                  <label class="label py-0"><span class="label-text text-[10px]">Temperature (K)</span></label>
                  <input type="number" class="input input-bordered input-xs w-full"
                    value={state().conform.config.mcmmTemperature} min={100} max={1000}
                    onInput={(e) => setConformConfig({ mcmmTemperature: Number(e.currentTarget.value) || 298 })}
                  />
                </div>
              </div>
              <label class="label cursor-pointer py-0 mt-1">
                <span class="label-text text-[10px]">Sample amide cis/trans</span>
                <input
                  type="checkbox"
                  class="checkbox checkbox-xs checkbox-primary"
                  checked={state().conform.config.sampleAmides}
                  onChange={(e) => setConformConfig({ sampleAmides: e.currentTarget.checked })}
                />
              </label>
            </Show>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div class="flex justify-between mt-3 flex-shrink-0">
        <button class="btn btn-ghost btn-sm" onClick={() => setConformStep('conform-load')}>
          <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 17l-5-5m0 0l5-5m-5 5h12" />
          </svg>
          Back
        </button>
        <button class="btn btn-primary" onClick={() => setConformStep('conform-progress')}>
          Start Search
          <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
          </svg>
        </button>
      </div>
    </div>
  );
};

export default ConformStepConfigure;
