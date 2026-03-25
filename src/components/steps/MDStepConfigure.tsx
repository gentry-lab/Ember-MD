// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, Show, createMemo, createSignal, For, onMount } from 'solid-js';
import { workflowStore } from '../../stores/workflow';
import { useMdOutput } from '../../hooks/useElectronApi';
import { MD_COMMON_PARAMS, MD_PRESET_PARAMS, MDForceFieldPreset } from '../../../shared/types/md';
import { buildMdRunFolderName, sanitizeCompoundId, estimateChargeTime } from '../../utils/jobName';
import DurationDial from '../shared/DurationDial';

const MDStepConfigure: Component = () => {
  const {
    state,
    setMdConfig,
    setMdStep,
    setMdBenchmarkResult,
    setMdIsBenchmarking,
    setMdSystemInfo,
  } = workflowStore;
  const api = window.electronAPI;
  const [benchmarkStatus, setBenchmarkStatus] = createSignal<string | null>(null);
  const [showProtocol, setShowProtocol] = createSignal(false);
  // Track inputs used for last benchmark to enable caching
  const [lastBenchmarkInputs, setLastBenchmarkInputs] = createSignal<{
    ligandSdf: string;
    receptorPdb: string | null;
    preset: string;
    ligandOnly: boolean;
  } | null>(null);

  // Listen for MD output during benchmarking to show progress
  useMdOutput((data) => {
    // Always log MD output to console for debugging
    console.log(`[MD ${data.type}]`, data.data.trim());

    if (!state().md.isBenchmarking) return;
    const text = data.data;

    // Parse PROGRESS lines for user-friendly status
    const progressMatch = text.match(/PROGRESS:(\w+):(\S+)/);
    if (progressMatch) {
      const stage = progressMatch[1];
      if (stage === 'building') setBenchmarkStatus('Building system...');
      else if (stage === 'parameterizing') {
        // Parse atom count from PROGRESS:parameterizing:0:59
        const atomMatch = text.match(/PROGRESS:parameterizing:\d+:(\d+)/);
        const atoms = atomMatch ? parseInt(atomMatch[1]) : 0;
        setBenchmarkStatus(atoms ? `Computing AM1-BCC charges (${atoms} atoms, est. ${estimateChargeTime(atoms)})...` : 'Computing AM1-BCC charges...');
      }
      else if (stage === 'benchmark') setBenchmarkStatus('Running benchmark...');
    }

    // Parse SYSTEM_INFO
    const sysMatch = text.match(/SYSTEM_INFO:(\d+):(\d+)/);
    if (sysMatch) {
      setBenchmarkStatus(`System: ${parseInt(sysMatch[1]).toLocaleString()} atoms`);
    }
  });

  // Compute the output folder name reactively (project/run structure)
  const outputFolderName = createMemo(() => {
    const runFolder = buildMdRunFolderName({
      forceFieldPreset: state().md.config.forceFieldPreset,
      temperatureK: state().md.config.temperatureK,
      productionNs: state().md.config.productionNs,
      compoundId: state().md.config.compoundId,
    });
    return `${state().jobName}/${runFolder}`;
  });

  const isLigandOnly = () => state().md.inputMode === 'ligand_only';
  const isApo = () => state().md.inputMode === 'apo';
  const hasProtein = () => !isLigandOnly();

  // Set smart default duration on first render if still at initial default
  onMount(() => {
    if (state().md.config.productionNs === 10) {
      setMdConfig({ productionNs: isLigandOnly() ? 10 : 100 });
    }
  });

  const handleCancelBenchmark = async () => {
    await api.cancelMdBenchmark();
    setMdIsBenchmarking(false);
    setBenchmarkStatus(null);
  };

  const handleRunBenchmark = async () => {
    if (!isApo() && !state().md.ligandSdf) return;
    if (!isLigandOnly() && !state().md.receptorPdb) return;

    // Check if we already have a cached benchmark for the same inputs
    const currentInputs = {
      ligandSdf: state().md.ligandSdf!,
      receptorPdb: state().md.receptorPdb,
      preset: state().md.config.forceFieldPreset,
      ligandOnly: isLigandOnly(),
    };
    const cached = lastBenchmarkInputs();
    if (cached && state().md.benchmarkResult &&
        cached.ligandSdf === currentInputs.ligandSdf &&
        cached.receptorPdb === currentInputs.receptorPdb &&
        cached.preset === currentInputs.preset &&
        cached.ligandOnly === currentInputs.ligandOnly) {
      // Inputs unchanged — just recalculate estimate from cached ns/day
      const nsPerDay = state().md.benchmarkResult!.nsPerDay;
      const productionNs = state().md.config.productionNs;
      const estimatedHours = (productionNs / nsPerDay) * 24;
      setMdBenchmarkResult({
        nsPerDay,
        estimatedHours,
        systemInfo: state().md.benchmarkResult!.systemInfo,
      });
      return;
    }

    setMdIsBenchmarking(true);
    setMdBenchmarkResult(null);
    setBenchmarkStatus('Initializing...');
    let benchmarkSucceeded = false;

    try {
      // Create temp output dir for benchmark using working directory
      const defaultDir = await api.getDefaultOutputDir();
      const baseOutputDir = state().customOutputDir || defaultDir;
      const benchmarkDir = `${baseOutputDir}/${state().jobName}/.md_benchmark_temp`;

      console.log('[Benchmark] Starting with:', {
        receptorPdb: state().md.receptorPdb,
        ligandSdf: state().md.ligandSdf,
        benchmarkDir,
        preset: state().md.config.forceFieldPreset,
        ligandOnly: isLigandOnly(),
        inputMode: state().md.inputMode,
      });

      const result = await api.runMdBenchmark(
        state().md.receptorPdb,
        state().md.ligandSdf!,
        benchmarkDir,
        state().md.config.forceFieldPreset,
        isLigandOnly()
      );

      console.log('[Benchmark] Result:', JSON.stringify(result));

      if (result.ok) {
        const nsPerDay = result.value.nsPerDay;
        const productionNs = state().md.config.productionNs;
        const estimatedHours = (productionNs / nsPerDay) * 24;

        setMdBenchmarkResult({
          nsPerDay,
          estimatedHours,
          systemInfo: result.value.systemInfo,
        });
        setMdSystemInfo(result.value.systemInfo);
        setLastBenchmarkInputs(currentInputs);
        benchmarkSucceeded = true;
      } else {
        setBenchmarkStatus(`Error: ${result.error.message}`);
        // Keep status visible for 5 seconds on error
        setTimeout(() => setBenchmarkStatus(null), 5000);
        return;
      }
    } finally {
      setMdIsBenchmarking(false);
      if (benchmarkSucceeded) {
        setBenchmarkStatus(null);
      }
    }
  };

  const handleBack = () => {
    if (state().md.isBenchmarking) {
      handleCancelBenchmark();
    }
    setMdBenchmarkResult(null);
    setMdSystemInfo(null);
    setMdStep('md-load');
  };

  const handleContinue = () => {
    if (state().md.isBenchmarking) return;
    setMdStep('md-progress');
  };

  const formatDuration = (hours: number): string => {
    if (hours < 1) {
      return `${Math.round(hours * 60)} min`;
    } else if (hours < 24) {
      return `${hours.toFixed(1)} hours`;
    } else {
      const days = Math.floor(hours / 24);
      const remainingHours = hours % 24;
      return `${days}d ${remainingHours.toFixed(0)}h`;
    }
  };

  // Recalculate estimated time when production duration changes
  const estimatedTime = createMemo(() => {
    const benchmark = state().md.benchmarkResult;
    if (!benchmark) return null;
    const productionNs = state().md.config.productionNs;
    const hours = (productionNs / benchmark.nsPerDay) * 24;
    return formatDuration(hours);
  });

  return (
    <div class="h-full flex flex-col">
      {/* Title */}
      <div class="text-center mb-3">
        <h2 class="text-xl font-bold gradient-text">
          {isLigandOnly() ? 'Configure Ligand MD' : isApo() ? 'Configure Apo Simulation' : 'Configure MD Simulation'}
        </h2>
        <p class="text-sm text-base-content/90">
          {isLigandOnly() ? 'Small molecule in solvent' : isApo() ? 'Protein-only in solvent (no ligand)' : 'Set simulation parameters and estimate runtime'}
        </p>
      </div>

      {/* Main content */}
      <div class="flex-1 min-h-0 overflow-auto grid grid-cols-2 gap-4 content-start">
        {/* Left column - Output preview + benchmark */}
        <div class="flex flex-col gap-3">
          {/* Output Preview */}
          <div class="card bg-base-200 shadow-lg">
            <div class="card-body p-4">
              <h3 class="text-sm font-semibold mb-2">Output Preview</h3>
              <div class="bg-base-300 rounded-lg p-3">
                <span class="text-[10px] text-base-content/80 block mb-1">Output folder</span>
                <span class="text-xs font-mono text-base-content/80 break-all">{outputFolderName()}</span>
              </div>
              <p class="text-[10px] text-base-content/80 mt-2">
                Folder name updates based on force field and duration
              </p>
            </div>
          </div>

          {/* Production Duration Dial + Benchmark */}
          <div class="card bg-base-200 shadow-lg flex-1">
            <div class="card-body p-4 flex flex-col items-center">
              <h3 class="text-sm font-semibold mb-1 self-start">Production Duration</h3>
              <DurationDial
                value={state().md.config.productionNs}
                min={0.1}
                max={1000}
                onChange={(v) => setMdConfig({ productionNs: v })}
                disabled={state().md.isBenchmarking}
              />
              <div class="flex flex-col items-center gap-2 w-full mt-1">
                <Show
                  when={!state().md.isBenchmarking}
                  fallback={
                    <button
                      class="btn btn-error btn-sm"
                      onClick={handleCancelBenchmark}
                    >
                      <span class="loading loading-spinner loading-xs" />
                      Cancel
                    </button>
                  }
                >
                  <button
                    class="btn btn-secondary btn-sm"
                    onClick={handleRunBenchmark}
                  >
                    Estimate Runtime
                  </button>
                </Show>
                <Show when={benchmarkStatus()}>
                  <p class="text-[10px] text-base-content/60" data-testid="benchmark-status">
                    {benchmarkStatus()}
                  </p>
                </Show>
              </div>

              {/* Benchmark Results */}
              <Show when={state().md.benchmarkResult}>
                <div class="bg-success/10 border border-success rounded-lg p-3 w-full mt-2" data-testid="benchmark-results">
                  <h4 class="font-semibold text-success text-xs mb-2">Benchmark Results</h4>
                  <div class="grid grid-cols-2 gap-2 text-xs">
                    <span class="text-base-content/85">Throughput:</span>
                    <span class="font-mono font-bold">{state().md.benchmarkResult!.nsPerDay.toFixed(1)} ns/day</span>
                    <span class="text-base-content/85">Est. Runtime:</span>
                    <span class="font-mono font-bold">{estimatedTime()}</span>
                    <span class="text-base-content/85">Atom Count:</span>
                    <span class="font-mono">{state().md.benchmarkResult!.systemInfo.atomCount.toLocaleString()}</span>
                    <span class="text-base-content/85">Box Volume:</span>
                    <span class="font-mono">{state().md.benchmarkResult!.systemInfo.boxVolumeA3.toLocaleString()} A^3</span>
                  </div>
                </div>
              </Show>
            </div>
          </div>
        </div>

        {/* Right column - Parameters */}
        <div class="card bg-base-200 shadow-lg">
          <div class="card-body p-4">
            {/* Force Field Preset Dropdown */}
            <div class="mb-4">
              <h3 class="text-sm font-semibold mb-2">Force Field Preset</h3>
              <select
                class="select select-sm w-full"
                value={state().md.config.forceFieldPreset}
                onChange={(e) => setMdConfig({ forceFieldPreset: e.currentTarget.value as MDForceFieldPreset })}
              >
                <For each={Object.keys(MD_PRESET_PARAMS) as MDForceFieldPreset[]}>{(id) => (
                  <option value={id}>
                    {MD_PRESET_PARAMS[id].label}{MD_PRESET_PARAMS[id].recommended ? ' ★' : ''}
                  </option>
                )}</For>
              </select>
              <p class="text-[10px] text-base-content/85 mt-1">
                {MD_PRESET_PARAMS[state().md.config.forceFieldPreset].description}
              </p>
            </div>

            {/* Compound Identifier */}
            <div class="mb-4">
              <h3 class="text-sm font-semibold mb-2">Compound Identifier (optional)</h3>
              <input
                type="text"
                class="input input-bordered input-sm w-full font-mono text-xs"
                placeholder="e.g., imatinib, compound-7a"
                value={state().md.config.compoundId}
                onInput={(e) => setMdConfig({ compoundId: sanitizeCompoundId(e.currentTarget.value) })}
              />
              <p class="text-[10px] text-base-content/80 mt-1">
                Added to the run folder name for identification
              </p>
            </div>

            <h3 class="text-sm font-semibold mb-3">Simulation Parameters</h3>
            <div class="space-y-2 text-xs">
              <div class="flex justify-between py-1.5 border-b border-base-300">
                <span class="text-base-content/90">Protein FF</span>
                <span class="font-mono">
                  {isLigandOnly() ? 'None (ligand only)' : MD_PRESET_PARAMS[state().md.config.forceFieldPreset].forceFieldProtein}
                </span>
              </div>
              <div class="flex justify-between py-1.5 border-b border-base-300">
                <span class="text-base-content/90">Water Model</span>
                <span class="font-mono">{MD_PRESET_PARAMS[state().md.config.forceFieldPreset].forceFieldWater}</span>
              </div>
              <div class="flex justify-between py-1.5 border-b border-base-300">
                <span class="text-base-content/90">Ligand FF</span>
                <span class="font-mono">{MD_COMMON_PARAMS.forceFieldLigand}</span>
              </div>
              <div class="flex justify-between items-center py-1.5 border-b border-base-300">
                <span class="text-base-content/90">Temperature</span>
                <div class="flex items-center gap-1">
                  <input
                    type="number"
                    class="input input-bordered input-xs w-20 text-right font-mono"
                    value={state().md.config.temperatureK}
                    min={200}
                    max={500}
                    step={10}
                    onInput={(e) => setMdConfig({ temperatureK: Number(e.currentTarget.value) || 300 })}
                  />
                  <span class="font-mono">K</span>
                </div>
              </div>
              <div class="flex justify-between items-center py-1.5 border-b border-base-300">
                <span class="text-base-content/90">Salt</span>
                <div class="flex items-center gap-1">
                  <input
                    type="number"
                    class="input input-bordered input-xs w-20 text-right font-mono"
                    value={state().md.config.saltConcentrationM * 1000}
                    min={0}
                    max={1000}
                    step={10}
                    onInput={(e) => setMdConfig({ saltConcentrationM: (Number(e.currentTarget.value) || 150) / 1000 })}
                  />
                  <span class="font-mono">mM NaCl</span>
                </div>
              </div>
            </div>

            {/* Advanced Settings Accordion */}
            <div class="collapse collapse-arrow bg-base-300 rounded-lg mt-2">
              <input type="checkbox" />
              <div class="collapse-title text-xs font-semibold py-2 min-h-0">
                Advanced Settings
              </div>
              <div class="collapse-content px-3 pb-3">
                <div class="space-y-2 text-xs">
                  <div class="flex justify-between py-1.5 border-b border-base-300">
                    <span class="text-base-content/90">Padding</span>
                    <span class="font-mono">{MD_COMMON_PARAMS.paddingNm} nm</span>
                  </div>
                  <div class="flex justify-between py-1.5 border-b border-base-300">
                    <span class="text-base-content/90">Timestep</span>
                    <span class="font-mono">{MD_COMMON_PARAMS.timestepFs} fs (HMR)</span>
                  </div>
                  <div class="flex justify-between py-1.5 border-b border-base-300">
                    <span class="text-base-content/90">Box Shape</span>
                    <span class="font-mono">Auto (minimize volume)</span>
                  </div>
                  <div class="flex justify-between py-1.5 border-b border-base-300">
                    <span class="text-base-content/90">Integrator</span>
                    <span class="font-mono">{MD_COMMON_PARAMS.integrator}</span>
                  </div>
                  <div class="flex justify-between py-1.5">
                    <span class="text-base-content/90">Equilibration</span>
                    <span class="font-mono">~{isLigandOnly() ? '170' : MD_COMMON_PARAMS.equilibrationPs} ps</span>
                  </div>
                </div>

                {/* Equilibration protocol */}
                <button
                  class="mt-2 flex items-center gap-1 text-[10px] text-base-content/60 hover:text-base-content/80"
                  onClick={() => setShowProtocol(!showProtocol())}
                >
                  <svg class={`w-3 h-3 transition-transform ${showProtocol() ? 'rotate-90' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                  </svg>
                  Equilibration protocol (~{isLigandOnly() ? '170' : '270'} ps)
                </button>
                <Show when={showProtocol()}>
                  <div class="mt-1 p-2 bg-base-300 rounded text-[10px] text-base-content/85">
                    <Show
                      when={!isLigandOnly()}
                      fallback={
                        <ol class="list-decimal list-inside space-y-0.5">
                          <li>Restrained minimization (heavy atoms, 10 kcal/mol/A^2)</li>
                          <li>Unrestrained minimization</li>
                          <li>NVT→NPT heating: 10K→{state().md.config.temperatureK}K (70 ps)</li>
                          <li>NPT equilibration (50 ps)</li>
                          <li>Unrestrained NPT equilibration (50 ps)</li>
                        </ol>
                      }
                    >
                      <ol class="list-decimal list-inside space-y-0.5">
                        <li>Restrained minimization (heavy atoms, 10 kcal/mol/A^2)</li>
                        <li>Unrestrained minimization</li>
                        <li>NVT→NPT heating: 10K→{state().md.config.temperatureK}K with backbone restraints (70 ps)</li>
                        <li>NPT equilibration with backbone restraints (50 ps)</li>
                        <li>Gradual restraint release (100 ps)</li>
                        <li>Unrestrained NPT equilibration (50 ps)</li>
                      </ol>
                    </Show>
                  </div>
                </Show>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div class="flex justify-between mt-3 flex-shrink-0">
        <button class="btn btn-ghost btn-sm" onClick={handleBack}>
          <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 17l-5-5m0 0l5-5m-5 5h12" />
          </svg>
          Back
        </button>
        <button
          class="btn btn-primary"
          onClick={handleContinue}
          disabled={state().md.isBenchmarking}
        >
          {state().md.isBenchmarking ? 'Benchmarking...' : 'Start Simulation'}
          <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
          </svg>
        </button>
      </div>
    </div>
  );
};

export default MDStepConfigure;
