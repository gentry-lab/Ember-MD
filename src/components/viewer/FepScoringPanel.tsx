import { Component, Show, createSignal, createEffect, For } from 'solid-js';
import { workflowStore } from '../../stores/workflow';
import type { FepSnapshotResult, FepScoringResult } from '../../../shared/types/ipc';

type FepStep = 'range' | 'configure' | 'progress' | 'results';

interface FepScoringPanelProps {
  onBack?: () => void;
}

const FepScoringPanel: Component<FepScoringPanelProps> = (props) => {
  const { state } = workflowStore;
  const api = window.electronAPI;

  // Step state
  const [step, setStep] = createSignal<FepStep>('range');

  // Range selection
  const [rmsdData, setRmsdData] = createSignal<{ timeNs: number[]; rmsd: number[] } | null>(null);
  const [isLoadingRmsd, setIsLoadingRmsd] = createSignal(false);
  const [startNs, setStartNs] = createSignal(0);
  const [endNs, setEndNs] = createSignal(0);
  const [totalTimeNs, setTotalTimeNs] = createSignal(0);

  // Configuration
  const [numSnapshots, setNumSnapshots] = createSignal(5);
  const [speedPreset, setSpeedPreset] = createSignal<'fast' | 'accurate'>('fast');

  // Progress
  const [isRunning, setIsRunning] = createSignal(false);
  const [progressText, setProgressText] = createSignal('');
  const [overallPct, setOverallPct] = createSignal(0);
  const [incrementalResults, setIncrementalResults] = createSignal<FepSnapshotResult[]>([]);
  const [logs, setLogs] = createSignal('');

  // Results
  const [finalResult, setFinalResult] = createSignal<FepScoringResult | null>(null);
  const [error, setError] = createSignal<string | null>(null);

  // Canvas ref for RMSD plot
  // eslint-disable-next-line no-unassigned-vars -- SolidJS ref pattern
  let canvasRef: HTMLCanvasElement | undefined;

  // Load RMSD data on mount
  createEffect(() => {
    if (step() === 'range' && !rmsdData() && !isLoadingRmsd()) {
      loadRmsdData();
    }
  });

  // Redraw canvas when data or range changes
  createEffect(() => {
    const data = rmsdData();
    const s = startNs();
    const e = endNs();
    if (data && canvasRef) {
      drawRmsdPlot(canvasRef, data, s, e);
    }
  });

  const loadRmsdData = async () => {
    const pdbPath = state().viewer.pdbPath;
    const trajPath = state().viewer.trajectoryPath;
    if (!pdbPath || !trajPath) return;

    setIsLoadingRmsd(true);
    try {
      const trajDir = trajPath.split('/').slice(0, -1).join('/');
      const outputDir = `${trajDir}/analysis/rmsd`;

      const result = await api.analyzeTrajectory({
        topologyPath: pdbPath,
        trajectoryPath: trajPath,
        analysisType: 'rmsd',
        outputDir,
      });

      if (result.ok && result.value.type === 'rmsd') {
        const data = result.value.data as { timeNs?: number[]; rmsdProtein?: number[] };
        const timeNs = data.timeNs || [];
        const rmsd = data.rmsdProtein || [];
        setRmsdData({ timeNs, rmsd });
        if (timeNs.length > 0) {
          const total = timeNs[timeNs.length - 1];
          setTotalTimeNs(total);
          // Default: select last 25% as equilibrated range
          setStartNs(Math.round(total * 0.75 * 10) / 10);
          setEndNs(Math.round(total * 10) / 10);
        }
      }
    } catch (err) {
      console.error('[FEP] Failed to load RMSD:', err);
    } finally {
      setIsLoadingRmsd(false);
    }
  };

  const drawRmsdPlot = (canvas: HTMLCanvasElement, data: { timeNs: number[]; rmsd: number[] }, selStart: number, selEnd: number) => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    const pad = { top: 20, right: 20, bottom: 35, left: 50 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    ctx.clearRect(0, 0, w, h);

    if (data.timeNs.length === 0) return;

    const tMin = 0;
    const tMax = data.timeNs[data.timeNs.length - 1];
    const rMax = Math.max(...data.rmsd) * 1.1;
    const rMin = 0;

    const toX = (t: number) => pad.left + (t - tMin) / (tMax - tMin) * plotW;
    const toY = (r: number) => pad.top + plotH - (r - rMin) / (rMax - rMin) * plotH;

    // Selected region highlight
    const selX0 = toX(selStart);
    const selX1 = toX(selEnd);
    ctx.fillStyle = 'rgba(59, 130, 246, 0.15)';
    ctx.fillRect(selX0, pad.top, selX1 - selX0, plotH);

    // Axes
    ctx.strokeStyle = '#888';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = '#aaa';
    ctx.font = '11px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('Time (ns)', pad.left + plotW / 2, h - 5);
    ctx.save();
    ctx.translate(12, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('RMSD (nm)', 0, 0);
    ctx.restore();

    // Tick labels
    ctx.font = '10px system-ui';
    const nTicksX = 5;
    for (let i = 0; i <= nTicksX; i++) {
      const t = tMin + (tMax - tMin) * i / nTicksX;
      const x = toX(t);
      ctx.fillText(t.toFixed(1), x, pad.top + plotH + 15);
    }
    const nTicksY = 4;
    ctx.textAlign = 'right';
    for (let i = 0; i <= nTicksY; i++) {
      const r = rMin + (rMax - rMin) * i / nTicksY;
      const y = toY(r);
      ctx.fillText(r.toFixed(2), pad.left - 5, y + 3);
    }

    // RMSD line
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < data.timeNs.length; i++) {
      const x = toX(data.timeNs[i]);
      const y = toY(data.rmsd[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Selection handles
    for (const xPos of [selX0, selX1]) {
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(xPos, pad.top);
      ctx.lineTo(xPos, pad.top + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  };

  // Fast: 5 elec + 4 steric = 9 unique lambda windows. Accurate: 7 + 5 = 12.
  const windowsPerLeg = () => speedPreset() === 'fast' ? 9 : 12;

  const estimatedTimeHours = () => {
    const ns = numSnapshots();
    const wpl = windowsPerLeg();
    const nsPerWindow = speedPreset() === 'fast' ? 1.0 : 2.0;
    return Math.round(ns * 2 * wpl * nsPerWindow * 10 / 8.7) / 10;
  };

  const handleStartFep = async () => {
    const pdbPath = state().viewer.pdbPath;
    const trajPath = state().viewer.trajectoryPath;
    if (!pdbPath || !trajPath) return;

    setStep('progress');
    setIsRunning(true);
    setError(null);
    setLogs('');
    setIncrementalResults([]);
    setOverallPct(0);
    setProgressText('Starting FEP calculation...');

    const trajDir = trajPath.split('/').slice(0, -1).join('/');
    // Output to fep/ in the project root (walk up from simulations/{run}/)
    const projectDir = trajDir.replace(/\/(simulations)\/[^/]+$/, '');
    const outputDir = projectDir !== trajDir
      ? `${projectDir}/fep`
      : `${trajDir}/fep_scoring`;

    // eslint-disable-next-line solid/reactivity -- callback registered once at run start, not reactive
    const removeListener = api.onMdOutput((data) => {
      const text = data.data;
      // Cap logs to last 50KB to avoid unbounded DOM growth during multi-hour runs
      setLogs((prev) => {
        const combined = prev + text;
        return combined.length > 50000 ? combined.slice(-50000) : combined;
      });

      // Parse progress lines
      const lines = text.split('\n');
      for (const line of lines) {
        if (line.startsWith('PROGRESS:')) {
          const parts = line.split(':');
          if (parts.length >= 6) {
            const snapIdx = parseInt(parts[2]);
            const leg = parts[3];
            const winIdx = parseInt(parts[4]);
            const wpl = windowsPerLeg();
            setProgressText(`Snapshot ${snapIdx + 1}/${numSnapshots()}: ${leg} leg, window ${winIdx + 1}`);
            const totalWindows = numSnapshots() * 2 * wpl;
            const completedWindows = snapIdx * 2 * wpl
              + (leg === 'solvent' ? wpl : 0)
              + winIdx;
            setOverallPct(Math.min(100, Math.round(100 * completedWindows / totalWindows)));
          }
        } else if (line.startsWith('FEP_RESULT:')) {
          try {
            const result = JSON.parse(line.substring(11));
            setIncrementalResults((prev) => [...prev, result]);
          } catch (_e) {
            console.warn('[FEP] Failed to parse result line:', line);
          }
        }
      }
    });

    try {
      console.log(`[FEP] Starting scoring: ${speedPreset()}, ${numSnapshots()} snapshots, ${startNs()}-${endNs()}ns → ${outputDir}`);
      const result = await api.runFepScoring({
        topologyPath: pdbPath,
        trajectoryPath: trajPath,
        startNs: startNs(),
        endNs: endNs(),
        numSnapshots: numSnapshots(),
        speedPreset: speedPreset(),
        outputDir,
        forceFieldPreset: 'ff19sb-opc',
      });

      if (result.ok) {
        setFinalResult(result.value);
        setStep('results');
      } else {
        if (result.error.type === 'FEP_SCORING_CANCELLED') {
          setError('FEP scoring was cancelled.');
        } else {
          setError(result.error.message);
        }
      }
    } catch (err) {
      setError(`FEP scoring failed: ${(err as Error).message}`);
    } finally {
      removeListener();
      setIsRunning(false);
    }
  };

  const handleCancel = async () => {
    try {
      await api.cancelFepScoring();
    } catch { /* cancel is best-effort */ }
  };

  const handleOpenFolder = () => {
    const result = finalResult();
    if (result?.outputDir) {
      api.openFolder(result.outputDir);
    }
  };

  // Range step — slider handlers
  const handleStartChange = (e: Event) => {
    const val = parseFloat((e.target as HTMLInputElement).value);
    if (val < endNs() - 0.5) setStartNs(val);
  };

  const handleEndChange = (e: Event) => {
    const val = parseFloat((e.target as HTMLInputElement).value);
    if (val > startNs() + 0.5) setEndNs(val);
  };

  return (
    <div class="h-full overflow-auto flex flex-col">
      {/* Header */}
      <div class="flex items-center gap-3 p-3 border-b border-base-300">
        <Show when={props.onBack}>
          <button
            class="btn btn-sm btn-ghost"
            onClick={() => props.onBack?.()}
            disabled={isRunning()}
          >
            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
            </svg>
            Back
          </button>
        </Show>
        <h2 class="text-sm font-bold flex items-center gap-2">
          ABFE Free Energy Scoring
          <span class="badge badge-warning badge-xs">experimental</span>
        </h2>
      </div>

      {/* Content */}
      <div class="flex-1 p-4 overflow-auto">
        {/* Step 1: Range Selection */}
        <Show when={step() === 'range'}>
          <div class="max-w-2xl mx-auto">
            <h3 class="text-sm font-semibold mb-2">Select Equilibrated Range</h3>
            <p class="text-xs text-base-content/70 mb-3">
              Choose a time range from the equilibrated portion of your trajectory.
              The RMSD plot below helps identify when the system is equilibrated.
            </p>

            <Show when={isLoadingRmsd()}>
              <div class="flex items-center justify-center py-12">
                <span class="loading loading-spinner loading-md" />
                <span class="ml-2 text-sm">Computing RMSD...</span>
              </div>
            </Show>

            <Show when={rmsdData()}>
              <div class="bg-base-200 rounded-lg p-3 mb-3">
                <canvas
                  ref={canvasRef}
                  width={600}
                  height={200}
                  class="w-full"
                  style={{ "max-height": "200px" }}
                />
              </div>

              <div class="flex flex-col gap-2 mb-3">
                <div class="flex items-center gap-3">
                  <label class="text-xs w-16">Start (ns)</label>
                  <input
                    type="range"
                    class="range range-xs range-primary flex-1"
                    min={0}
                    max={totalTimeNs()}
                    step={0.1}
                    value={startNs()}
                    onInput={handleStartChange}
                  />
                  <span class="text-xs font-mono w-14 text-right">{startNs().toFixed(1)}</span>
                </div>
                <div class="flex items-center gap-3">
                  <label class="text-xs w-16">End (ns)</label>
                  <input
                    type="range"
                    class="range range-xs range-accent flex-1"
                    min={0}
                    max={totalTimeNs()}
                    step={0.1}
                    value={endNs()}
                    onInput={handleEndChange}
                  />
                  <span class="text-xs font-mono w-14 text-right">{endNs().toFixed(1)}</span>
                </div>
              </div>

              <div class="text-xs text-base-content/80 mb-4">
                Selected: <span class="font-semibold">{startNs().toFixed(1)} – {endNs().toFixed(1)} ns</span>
                {' '}({(endNs() - startNs()).toFixed(1)} ns)
              </div>

              <div class="flex justify-end">
                <button
                  class="btn btn-sm btn-primary"
                  onClick={() => setStep('configure')}
                  disabled={endNs() - startNs() < 0.5}
                >
                  Next
                </button>
              </div>
            </Show>

            <Show when={!rmsdData() && !isLoadingRmsd()}>
              <div class="text-xs text-error">
                Could not load RMSD data. Make sure a trajectory is loaded.
              </div>
            </Show>
          </div>
        </Show>

        {/* Step 2: Configuration */}
        <Show when={step() === 'configure'}>
          <div class="max-w-md mx-auto">
            <h3 class="text-sm font-semibold mb-3">FEP Configuration</h3>

            <div class="form-control mb-4">
              <label class="label py-1">
                <span class="label-text text-xs">Number of snapshots</span>
              </label>
              <div class="flex gap-3">
                <For each={[3, 5, 7]}>
                  {(n) => (
                    <label class="flex items-center gap-1.5 cursor-pointer">
                      <input
                        type="radio"
                        name="snapshots"
                        class="radio radio-xs radio-primary"
                        checked={numSnapshots() === n}
                        onChange={() => setNumSnapshots(n)}
                      />
                      <span class="text-xs">{n}</span>
                    </label>
                  )}
                </For>
              </div>
            </div>

            <div class="form-control mb-4">
              <label class="label py-1">
                <span class="label-text text-xs">Speed preset</span>
              </label>
              <div class="flex gap-4">
                <label class="flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="radio"
                    name="speed"
                    class="radio radio-xs radio-primary"
                    checked={speedPreset() === 'fast'}
                    onChange={() => setSpeedPreset('fast')}
                  />
                  <span class="text-xs">Fast (8 windows/leg)</span>
                </label>
                <label class="flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="radio"
                    name="speed"
                    class="radio radio-xs radio-primary"
                    checked={speedPreset() === 'accurate'}
                    onChange={() => setSpeedPreset('accurate')}
                  />
                  <span class="text-xs">Accurate (12 windows/leg)</span>
                </label>
              </div>
            </div>

            <div class="bg-base-200 rounded-lg p-3 mb-4 text-xs">
              <div class="flex justify-between">
                <span>Estimated time:</span>
                <span class="font-semibold">~{estimatedTimeHours()} hours</span>
              </div>
              <div class="flex justify-between mt-1 text-base-content/70">
                <span>Range:</span>
                <span>{startNs().toFixed(1)} – {endNs().toFixed(1)} ns</span>
              </div>
              <div class="flex justify-between mt-1 text-base-content/70">
                <span>Total windows:</span>
                <span>{numSnapshots() * 2 * windowsPerLeg()}</span>
              </div>
            </div>

            <div class="flex justify-between">
              <button class="btn btn-sm btn-ghost" onClick={() => setStep('range')}>
                Back
              </button>
              <button class="btn btn-sm btn-primary" onClick={handleStartFep}>
                Start FEP
              </button>
            </div>
          </div>
        </Show>

        {/* Step 3: Progress */}
        <Show when={step() === 'progress'}>
          <div class="max-w-2xl mx-auto">
            <h3 class="text-sm font-semibold mb-3">FEP Calculation in Progress</h3>

            <Show when={error()}>
              <div class="alert alert-error text-xs mb-3">
                {error()}
              </div>
            </Show>

            <Show when={isRunning()}>
              <div class="mb-3">
                <div class="flex justify-between text-xs mb-1">
                  <span>{progressText()}</span>
                  <span>{overallPct()}%</span>
                </div>
                <progress class="progress progress-primary w-full" value={overallPct()} max="100" />
              </div>
            </Show>

            {/* Incremental results table */}
            <Show when={incrementalResults().length > 0}>
              <div class="overflow-x-auto mb-3">
                <table class="table table-xs">
                  <thead>
                    <tr>
                      <th>Snapshot</th>
                      <th>Time (ns)</th>
                      <th>dG_complex</th>
                      <th>dG_solvent</th>
                      <th>dG_bind</th>
                    </tr>
                  </thead>
                  <tbody>
                    <For each={incrementalResults()}>
                      {(r) => (
                        <tr>
                          <td>{r.snapshotIndex + 1}</td>
                          <td>{r.timeNs.toFixed(2)}</td>
                          <td>{r.deltaG_complex.toFixed(2)}</td>
                          <td>{r.deltaG_solvent.toFixed(2)}</td>
                          <td class="font-semibold">{r.deltaG_bind.toFixed(2)} +/- {r.uncertainty.toFixed(2)}</td>
                        </tr>
                      )}
                    </For>
                  </tbody>
                </table>
              </div>
            </Show>

            {/* Log output */}
            <details class="collapse collapse-arrow bg-base-200 mb-3">
              <summary class="collapse-title text-xs font-medium py-2 min-h-0">
                Log output
              </summary>
              <div class="collapse-content">
                <pre class="text-xs font-mono max-h-48 overflow-auto whitespace-pre-wrap bg-base-300 p-2 rounded">
                  {logs() || 'Waiting for output...'}
                </pre>
              </div>
            </details>

            <Show when={isRunning()}>
              <div class="flex justify-end">
                <button class="btn btn-sm btn-error btn-outline" onClick={handleCancel}>
                  Cancel
                </button>
              </div>
            </Show>

            <Show when={!isRunning() && error()}>
              <div class="flex justify-end gap-2">
                <Show when={props.onBack}>
                  <button class="btn btn-sm btn-ghost" onClick={() => props.onBack?.()}>
                    Close
                  </button>
                </Show>
                <button class="btn btn-sm btn-primary" onClick={() => { setError(null); setStep('configure'); }}>
                  Retry
                </button>
              </div>
            </Show>
          </div>
        </Show>

        {/* Step 4: Results */}
        <Show when={step() === 'results'}>
          <div class="max-w-2xl mx-auto">
            <h3 class="text-sm font-semibold mb-3">FEP Results</h3>

            <Show when={finalResult()}>
              {(result) => (
                <>
                  <div class="bg-primary/10 rounded-lg p-4 mb-4 text-center">
                    <div class="text-xs text-base-content/70 mb-1">Binding Free Energy</div>
                    <div class="text-2xl font-bold">
                      dG_bind = {result().meanDeltaG.toFixed(2)} +/- {result().sem.toFixed(2)} kcal/mol
                    </div>
                  </div>

                  <div class="overflow-x-auto mb-4">
                    <table class="table table-xs">
                      <thead>
                        <tr>
                          <th>Snapshot</th>
                          <th>Time (ns)</th>
                          <th>dG_complex</th>
                          <th>dG_solvent</th>
                          <th>dG_bind +/- uncertainty</th>
                        </tr>
                      </thead>
                      <tbody>
                        <For each={result().snapshots}>
                          {(r) => (
                            <tr>
                              <td>{r.snapshotIndex + 1}</td>
                              <td>{r.timeNs.toFixed(2)}</td>
                              <td>{r.deltaG_complex.toFixed(2)}</td>
                              <td>{r.deltaG_solvent.toFixed(2)}</td>
                              <td class="font-semibold">{r.deltaG_bind.toFixed(2)} +/- {r.uncertainty.toFixed(2)}</td>
                            </tr>
                          )}
                        </For>
                      </tbody>
                    </table>
                  </div>

                  <div class="flex justify-end gap-2">
                    <Show when={props.onBack}>
                      <button class="btn btn-sm btn-ghost" onClick={() => props.onBack?.()}>
                        Close
                      </button>
                    </Show>
                    <button class="btn btn-sm btn-primary" onClick={handleOpenFolder}>
                      Open Folder
                    </button>
                  </div>
                </>
              )}
            </Show>
          </div>
        </Show>
      </div>
    </div>
  );
};

export default FepScoringPanel;
