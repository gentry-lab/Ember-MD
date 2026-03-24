// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, For, Show, createEffect, createMemo, createSignal } from 'solid-js';
import type { MdLigandDepictionBond, MdTorsionAnalysis, MdTorsionEntry } from '../../../shared/types/ipc';

interface MDTorsionPanelProps {
  analysis: MdTorsionAnalysis;
}

const PLOT_WIDTH = 420;
const PLOT_HEIGHT = 210;
const PLOT_MARGIN = 24;
const VIEW_PADDING = 1.2;

const formatAngle = (value: number | null | undefined) => {
  if (value == null || Number.isNaN(value)) return '-';
  return `${value.toFixed(1)}°`;
};

const MDTorsionPanel: Component<MDTorsionPanelProps> = (props) => {
  const [selectedTorsionId, setSelectedTorsionId] = createSignal<string | null>(null);
  const [activeTab, setActiveTab] = createSignal<'trajectory' | 'clusters'>('trajectory');
  const [copyStatus, setCopyStatus] = createSignal<string | null>(null);

  const torsions = () => props.analysis.data.torsions;
  const depiction = () => props.analysis.depiction;

  createEffect(() => {
    const currentId = selectedTorsionId();
    const rows = torsions();
    if (!rows.length) {
      setSelectedTorsionId(null);
      return;
    }
    if (!currentId || !rows.some((row) => row.torsionId === currentId)) {
      setSelectedTorsionId(rows[0].torsionId);
    }
  });

  const selectedTorsion = createMemo<MdTorsionEntry | null>(() => {
    const torsionId = selectedTorsionId();
    if (!torsionId) return null;
    return torsions().find((row) => row.torsionId === torsionId) ?? null;
  });

  const torsionBondIds = createMemo(() => new Map(torsions().map((row) => [row.bondId, row.torsionId])));

  const viewBox = createMemo(() => {
    const data = depiction();
    if (!data) return '0 0 10 10';
    const { minX, maxX, minY, maxY } = data.bounds;
    const width = Math.max(2, maxX - minX);
    const height = Math.max(2, maxY - minY);
    const padX = width * 0.15;
    const padY = height * 0.15;
    return `${minX - padX} ${minY - padY} ${width + padX * 2} ${height + padY * 2}`;
  });

  const clusterValues = createMemo(() => {
    const row = selectedTorsion();
    return row ? [...row.clusterValues].sort((a, b) => a.clusterId - b.clusterId) : [];
  });

  const makeTrajectoryPoints = createMemo(() => {
    const row = selectedTorsion();
    const frames = props.analysis.sampledFrameIndices;
    if (!row || row.trajectoryAngles.length === 0 || frames.length === 0) return '';
    const minFrame = frames[0];
    const maxFrame = frames[frames.length - 1];
    const frameSpan = Math.max(1, maxFrame - minFrame);
    const points = row.trajectoryAngles.map((angle, idx) => {
      const x = PLOT_MARGIN + ((frames[idx] - minFrame) / frameSpan) * (PLOT_WIDTH - PLOT_MARGIN * 2);
      const y = PLOT_MARGIN + ((180 - (angle + 180)) / 360) * (PLOT_HEIGHT - PLOT_MARGIN * 2);
      return `${x},${y}`;
    });
    return points.join(' ');
  });

  const makeClusterPoints = createMemo(() => {
    const values = clusterValues();
    if (values.length === 0) return '';
    const span = Math.max(1, values.length - 1);
    return values.map((value, idx) => {
      const x = PLOT_MARGIN + (idx / span) * (PLOT_WIDTH - PLOT_MARGIN * 2);
      const y = PLOT_MARGIN + ((180 - (value.angle + 180)) / 360) * (PLOT_HEIGHT - PLOT_MARGIN * 2);
      return `${x},${y}`;
    }).join(' ');
  });

  const yTicks = [-180, -90, 0, 90, 180];

  const copyText = async (text: string, message: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopyStatus(message);
      window.setTimeout(() => setCopyStatus(null), 1500);
    } catch (err) {
      console.error('Failed to copy torsion text:', err);
      setCopyStatus('Copy failed');
      window.setTimeout(() => setCopyStatus(null), 1500);
    }
  };

  const copySelectedTorsion = () => {
    const row = selectedTorsion();
    if (!row) return;
    const lines = [
      `torsionId: ${row.torsionId}`,
      `bondId: ${row.bondId}`,
      `label: ${row.label}`,
      `quartet: ${row.atomNames.join(' - ')}`,
      `mean: ${formatAngle(row.circularMean)}`,
      `std: ${formatAngle(row.circularStd)}`,
      `range: ${formatAngle(row.min)} to ${formatAngle(row.max)}`,
      `trajectoryAngles: ${row.trajectoryAngles.join(', ')}`,
    ];
    if (row.clusterValues.length > 0) {
      lines.push('clusterValues:');
      for (const value of row.clusterValues) {
        lines.push(`  cluster ${value.clusterId + 1}: ${formatAngle(value.angle)} (${value.population.toFixed(1)}%)`);
      }
    }
    copyText(lines.join('\n'), 'Copied torsion');
  };

  const copyAllTorsions = () => {
    const lines = torsions().map((row) => (
      [
        row.torsionId,
        row.label,
        row.bondId,
        row.centralBondAtomIndices.join('-'),
        formatAngle(row.circularMean),
        formatAngle(row.circularStd),
      ].join('\t')
    ));
    copyText(['torsionId\tlabel\tbondId\tcentralBond\tmean\tstd', ...lines].join('\n'), 'Copied torsion table');
  };

  const handleBondClick = (bond: MdLigandDepictionBond) => {
    const torsionId = torsionBondIds().get(bond.bondId);
    if (torsionId) {
      setSelectedTorsionId(torsionId);
    }
  };

  const isSelectedBond = (bondId: string) => selectedTorsion()?.bondId === bondId;
  const isInteractiveBond = (bondId: string) => torsionBondIds().has(bondId);

  return (
    <Show when={props.analysis.ligandPresent && torsions().length > 0 && depiction()}>
      <div class="card bg-base-200 shadow-sm">
        <div class="card-body p-4 gap-3">
          <div class="flex items-center justify-between gap-3 flex-wrap">
            <div>
              <h3 class="text-sm font-semibold">Ligand Dihedrals</h3>
              <p class="text-xs text-base-content/70">
                {props.analysis.nRotatableBonds} rotatable bonds • {props.analysis.nSampledFrames} sampled frames
              </p>
            </div>
            <div class="flex items-center gap-2 flex-wrap">
              <div role="tablist" class="tabs tabs-boxed tabs-xs">
                <button
                  class={`tab ${activeTab() === 'trajectory' ? 'tab-active' : ''}`}
                  onClick={() => setActiveTab('trajectory')}
                >
                  Trajectory
                </button>
                <button
                  class={`tab ${activeTab() === 'clusters' ? 'tab-active' : ''}`}
                  onClick={() => setActiveTab('clusters')}
                >
                  Clusters
                </button>
              </div>
              <button class="btn btn-outline btn-xs" onClick={copySelectedTorsion}>
                Copy Selected
              </button>
              <button class="btn btn-outline btn-xs" onClick={copyAllTorsions}>
                Copy Table
              </button>
            </div>
          </div>

          <Show when={copyStatus()}>
            <div class="text-xs text-success">{copyStatus()}</div>
          </Show>

          <div class="grid grid-cols-1 xl:grid-cols-[minmax(260px,320px)_minmax(0,1fr)] gap-4">
            <div class="bg-base-100 rounded-lg border border-base-300 p-3">
              <svg viewBox={viewBox()} class="w-full h-64">
                <For each={depiction()!.bonds}>
                  {(bond) => {
                    const interactive = isInteractiveBond(bond.bondId);
                    const selected = isSelectedBond(bond.bondId);
                    let stroke = '#94a3b8';
                    let strokeWidth = 1.4;

                    if (interactive) {
                      stroke = '#2563eb';
                      strokeWidth = 1.8;
                    }
                    if (selected) {
                      stroke = '#dc2626';
                      strokeWidth = 2.4;
                    }

                    return (
                      <g
                        class={interactive ? 'cursor-pointer' : ''}
                        onClick={() => handleBondClick(bond)}
                      >
                        <Show when={interactive}>
                          <line
                            x1={bond.x1}
                            y1={bond.y1}
                            x2={bond.x2}
                            y2={bond.y2}
                            stroke="transparent"
                            stroke-width="10"
                            vector-effect="non-scaling-stroke"
                            stroke-linecap="round"
                            pointer-events="stroke"
                          />
                        </Show>
                        <line
                          x1={bond.x1}
                          y1={bond.y1}
                          x2={bond.x2}
                          y2={bond.y2}
                          stroke={stroke}
                          stroke-width={strokeWidth}
                          vector-effect="non-scaling-stroke"
                          stroke-linecap="round"
                          pointer-events="none"
                        />
                      </g>
                    );
                  }}
                </For>
                <For each={depiction()!.atoms}>
                  {(atom) => (
                    <Show when={atom.showLabel}>
                      <text
                        x={atom.x}
                        y={atom.y}
                        text-anchor="middle"
                        dominant-baseline="middle"
                        font-size={`${0.48 * VIEW_PADDING}`}
                        fill="#0f172a"
                        class="select-none"
                      >
                        {atom.symbol}
                      </text>
                    </Show>
                  )}
                </For>
              </svg>
              <p class="mt-2 text-xs text-base-content/70">
                Click a highlighted bond or select a torsion row to sync the diagram.
              </p>
            </div>

            <div class="flex flex-col gap-3 min-w-0">
              <div class="bg-base-100 rounded-lg border border-base-300 p-3">
                <div class="overflow-auto max-h-48">
                  <table class="table table-xs">
                    <thead>
                      <tr>
                        <th>Torsion</th>
                        <th class="text-right">Mean</th>
                        <th class="text-right">Std</th>
                      </tr>
                    </thead>
                    <tbody>
                      <For each={torsions()}>
                        {(row) => (
                          <tr
                            class={`cursor-pointer ${selectedTorsionId() === row.torsionId ? 'bg-primary/10' : ''}`}
                            onClick={() => setSelectedTorsionId(row.torsionId)}
                          >
                            <td class="font-mono text-[11px]">{row.label}</td>
                            <td class="text-right font-mono text-[11px]">{formatAngle(row.circularMean)}</td>
                            <td class="text-right font-mono text-[11px]">{formatAngle(row.circularStd)}</td>
                          </tr>
                        )}
                      </For>
                    </tbody>
                  </table>
                </div>
              </div>

              <Show when={selectedTorsion()}>
                <div class="bg-base-100 rounded-lg border border-base-300 p-3 flex flex-col gap-3 min-w-0">
                  <div class="flex items-center justify-between gap-2 flex-wrap">
                    <div>
                      <div class="font-mono text-sm">{selectedTorsion()!.label}</div>
                      <div class="text-xs text-base-content/70">
                        {selectedTorsion()!.torsionId} • bond {selectedTorsion()!.centralBondAtomIndices.join('-')}
                      </div>
                    </div>
                    <div class="text-xs text-base-content/70">
                      median {formatAngle(selectedTorsion()!.median)}
                    </div>
                  </div>

                  <svg viewBox={`0 0 ${PLOT_WIDTH} ${PLOT_HEIGHT}`} class="w-full h-56 bg-base-200 rounded">
                    <For each={yTicks}>
                      {(tick) => {
                        const y = PLOT_MARGIN + ((180 - (tick + 180)) / 360) * (PLOT_HEIGHT - PLOT_MARGIN * 2);
                        return (
                          <>
                            <line x1={PLOT_MARGIN} y1={y} x2={PLOT_WIDTH - PLOT_MARGIN} y2={y} stroke="#cbd5e1" stroke-width="1" />
                            <text x={8} y={y + 4} font-size="10" fill="#475569">{tick}</text>
                          </>
                        );
                      }}
                    </For>

                    <Show when={activeTab() === 'trajectory' && makeTrajectoryPoints()}>
                      <polyline
                        fill="none"
                        stroke="#2563eb"
                        stroke-width="2"
                        points={makeTrajectoryPoints()}
                      />
                    </Show>

                    <Show when={activeTab() === 'clusters' && makeClusterPoints()}>
                      <polyline
                        fill="none"
                        stroke="#0f766e"
                        stroke-width="2"
                        points={makeClusterPoints()}
                      />
                      <For each={clusterValues()}>
                        {(value, index) => {
                          const span = Math.max(1, clusterValues().length - 1);
                          const x = PLOT_MARGIN + (index() / span) * (PLOT_WIDTH - PLOT_MARGIN * 2);
                          const y = PLOT_MARGIN + ((180 - (value.angle + 180)) / 360) * (PLOT_HEIGHT - PLOT_MARGIN * 2);
                          return <circle cx={x} cy={y} r="3.5" fill="#0f766e" />;
                        }}
                      </For>
                    </Show>
                  </svg>

                  <Show
                    when={activeTab() === 'trajectory'}
                    fallback={
                      <div class="overflow-auto max-h-40">
                        <table class="table table-xs">
                          <thead>
                            <tr>
                              <th>Cluster</th>
                              <th class="text-right">Pop%</th>
                              <th class="text-right">Angle</th>
                            </tr>
                          </thead>
                          <tbody>
                            <For each={clusterValues()}>
                              {(value) => (
                                <tr>
                                  <td class="font-mono text-[11px]">{value.clusterId + 1}</td>
                                  <td class="text-right font-mono text-[11px]">{value.population.toFixed(1)}%</td>
                                  <td class="text-right font-mono text-[11px]">{formatAngle(value.angle)}</td>
                                </tr>
                              )}
                            </For>
                          </tbody>
                        </table>
                      </div>
                    }
                  >
                    <div class="text-xs text-base-content/70">
                      Sampled every {props.analysis.stride} frame(s) across {props.analysis.nSampledFrames} points.
                    </div>
                  </Show>
                </div>
              </Show>
            </div>
          </div>
        </div>
      </div>
    </Show>
  );
};

export default MDTorsionPanel;
