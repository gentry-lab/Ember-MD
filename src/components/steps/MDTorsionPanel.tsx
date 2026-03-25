// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, For, Show, createEffect, createMemo, createSignal } from 'solid-js';
import type { MdLigandDepictionBond, MdTorsionAnalysis, MdTorsionEntry } from '../../../shared/types/ipc';

interface MDTorsionPanelProps {
  analysis: MdTorsionAnalysis;
}

/* ── Plot layout ── */
const PLOT_WIDTH = 420;
const PLOT_HEIGHT = 240;
const M = { top: 14, right: 14, bottom: 34, left: 42 };
const PL = M.left;
const PR = PLOT_WIDTH - M.right;
const PT = M.top;
const PB = PLOT_HEIGHT - M.bottom;
const PW = PR - PL;
const PH = PB - PT;

/* ── Bond rendering ── */
const BOND_W = 2.0;
const BOND_W_ACTIVE = 2.8;
const BOND_W_SEL = 3.2;
const GLOW_W = 12;
const HIT_W = 14;
const VIEW_PAD = 1.2;

/* ── Colors ── */
const C_BOND = '#94a3b8';
const C_ACTIVE = '#3b82f6';
const C_SEL = '#ef4444';
const C_GLOW = 'rgba(239, 68, 68, 0.22)';
const C_TRAJ = '#2563eb';
const C_CLUST = '#0f766e';

const Y_TICKS = [-180, -90, 0, 90, 180];

const angleToY = (a: number) => PT + ((180 - a) / 360) * PH;

const fmtAngle = (v: number | null | undefined) => {
  if (v == null || Number.isNaN(v)) return '-';
  return `${v.toFixed(1)}°`;
};

const MDTorsionPanel: Component<MDTorsionPanelProps> = (props) => {
  const [selectedTorsionId, setSelectedTorsionId] = createSignal<string | null>(null);
  const [activeTab, setActiveTab] = createSignal<'trajectory' | 'clusters'>('trajectory');
  const [copyStatus, setCopyStatus] = createSignal<string | null>(null);

  const torsions = () => props.analysis.data.torsions;
  const depiction = () => props.analysis.depiction;

  /* Auto-select first torsion */
  createEffect(() => {
    const id = selectedTorsionId();
    const rows = torsions();
    if (!rows.length) { setSelectedTorsionId(null); return; }
    if (!id || !rows.some((r) => r.torsionId === id)) {
      setSelectedTorsionId(rows[0].torsionId);
    }
  });

  const selectedTorsion = createMemo<MdTorsionEntry | null>(() => {
    const id = selectedTorsionId();
    return id ? torsions().find((r) => r.torsionId === id) ?? null : null;
  });

  const torsionBondIds = createMemo(() =>
    new Map(torsions().map((r) => [r.bondId, r.torsionId]))
  );

  /* ── Molecule viewBox ── */
  const viewBox = createMemo(() => {
    const d = depiction();
    if (!d) return '0 0 10 10';
    const { minX, maxX, minY, maxY } = d.bounds;
    const w = Math.max(2, maxX - minX);
    const h = Math.max(2, maxY - minY);
    const px = w * 0.18;
    const py = h * 0.18;
    return `${minX - px} ${minY - py} ${w + px * 2} ${h + py * 2}`;
  });

  /* Double-bond offset scaled to average bond length */
  const bondGap = createMemo(() => {
    const bonds = depiction()?.bonds;
    if (!bonds || bonds.length === 0) return 0.08;
    let total = 0;
    for (const b of bonds) total += Math.hypot(b.x2 - b.x1, b.y2 - b.y1);
    return (total / bonds.length) * 0.085;
  });

  /* ── Cluster data ── */
  const clusterValues = createMemo(() => {
    const row = selectedTorsion();
    return row ? [...row.clusterValues].sort((a, b) => a.clusterId - b.clusterId) : [];
  });

  /* ── Trajectory polyline ── */
  const trajectoryPoints = createMemo(() => {
    const row = selectedTorsion();
    const frames = props.analysis.sampledFrameIndices;
    if (!row || !row.trajectoryAngles.length || !frames.length) return '';
    const lo = frames[0];
    const hi = frames[frames.length - 1];
    const span = Math.max(1, hi - lo);
    return row.trajectoryAngles.map((a, i) => {
      const x = PL + ((frames[i] - lo) / span) * PW;
      return `${x},${angleToY(a)}`;
    }).join(' ');
  });

  /* ── Clipboard ── */
  const copyText = async (text: string, msg: string) => {
    try { await navigator.clipboard.writeText(text); setCopyStatus(msg); }
    catch { setCopyStatus('Copy failed'); }
    window.setTimeout(() => setCopyStatus(null), 1500);
  };

  const copySelectedTorsion = () => {
    const row = selectedTorsion();
    if (!row) return;
    const lines = [
      `torsionId: ${row.torsionId}`, `bondId: ${row.bondId}`,
      `label: ${row.label}`, `quartet: ${row.atomNames.join(' - ')}`,
      `mean: ${fmtAngle(row.circularMean)}`, `std: ${fmtAngle(row.circularStd)}`,
      `range: ${fmtAngle(row.min)} to ${fmtAngle(row.max)}`,
      `trajectoryAngles: ${row.trajectoryAngles.join(', ')}`,
    ];
    if (row.clusterValues.length) {
      lines.push('clusterValues:');
      for (const v of row.clusterValues)
        lines.push(`  cluster ${v.clusterId + 1}: ${fmtAngle(v.angle)} (${v.population.toFixed(1)}%)`);
    }
    copyText(lines.join('\n'), 'Copied torsion');
  };

  const copyAllTorsions = () => {
    const lines = torsions().map((r) =>
      [r.torsionId, r.label, r.bondId, r.centralBondAtomIndices.join('-'),
        fmtAngle(r.circularMean), fmtAngle(r.circularStd)].join('\t')
    );
    copyText(['torsionId\tlabel\tbondId\tcentralBond\tmean\tstd', ...lines].join('\n'), 'Copied torsion table');
  };

  const handleBondClick = (bond: MdLigandDepictionBond) => {
    const id = torsionBondIds().get(bond.bondId);
    if (id) setSelectedTorsionId(id);
  };

  /* Reactive helpers — must be called inside JSX for SolidJS tracking */
  const isSel = (bondId: string) => selectedTorsion()?.bondId === bondId;
  const isAct = (bondId: string) => torsionBondIds().has(bondId);

  /* ── Bond line renderer ── */
  const renderBondLines = (bond: MdLigandDepictionBond) => {
    const dx = bond.x2 - bond.x1;
    const dy = bond.y2 - bond.y1;
    const len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len;
    const ny = dx / len;
    const g = bondGap();

    /* stroke/width as functions so JSX attributes track signals */
    const sc = () => isSel(bond.bondId) ? C_SEL : isAct(bond.bondId) ? C_ACTIVE : C_BOND;
    const sw = () => isSel(bond.bondId) ? BOND_W_SEL : isAct(bond.bondId) ? BOND_W_ACTIVE : BOND_W;

    if ((bond.order >= 2 && !bond.isAromatic) || bond.isAromatic) {
      const dash = bond.isAromatic ? '3 2' : undefined;
      return (
        <>
          <line x1={bond.x1 + nx * g} y1={bond.y1 + ny * g}
                x2={bond.x2 + nx * g} y2={bond.y2 + ny * g}
                stroke={sc()} stroke-width={sw()}
                vector-effect="non-scaling-stroke" stroke-linecap="round" pointer-events="none" />
          <line x1={bond.x1 - nx * g} y1={bond.y1 - ny * g}
                x2={bond.x2 - nx * g} y2={bond.y2 - ny * g}
                stroke={sc()} stroke-width={sw()}
                stroke-dasharray={dash}
                vector-effect="non-scaling-stroke" stroke-linecap="round" pointer-events="none" />
        </>
      );
    }
    return (
      <line x1={bond.x1} y1={bond.y1} x2={bond.x2} y2={bond.y2}
            stroke={sc()} stroke-width={sw()}
            vector-effect="non-scaling-stroke" stroke-linecap="round" pointer-events="none" />
    );
  };

  /* ═══════════════════════════ JSX ═══════════════════════════ */
  return (
    <Show when={props.analysis.ligandPresent && torsions().length > 0 && depiction()}>
      <div class="card bg-base-200 shadow-sm">
        <div class="card-body p-4 gap-3">

          {/* ── Header ── */}
          <div class="flex items-center justify-between gap-3 flex-wrap">
            <div>
              <h3 class="text-sm font-semibold">Ligand Dihedrals</h3>
              <p class="text-xs text-base-content/70">
                {props.analysis.nRotatableBonds} rotatable bonds • {props.analysis.nSampledFrames} sampled frames
              </p>
            </div>
            <div class="flex items-center gap-2 flex-wrap">
              <div role="tablist" class="tabs tabs-boxed tabs-xs">
                <button class={`tab ${activeTab() === 'trajectory' ? 'tab-active' : ''}`}
                        onClick={() => setActiveTab('trajectory')}>Trajectory</button>
                <button class={`tab ${activeTab() === 'clusters' ? 'tab-active' : ''}`}
                        onClick={() => setActiveTab('clusters')}>Clusters</button>
              </div>
              <button class="btn btn-outline btn-xs" onClick={copySelectedTorsion}>Copy Selected</button>
              <button class="btn btn-outline btn-xs" onClick={copyAllTorsions}>Copy Table</button>
            </div>
          </div>

          <Show when={copyStatus()}>
            <div class="text-xs text-success">{copyStatus()}</div>
          </Show>

          {/* ── Main grid ── */}
          <div class="grid grid-cols-1 xl:grid-cols-[minmax(260px,320px)_minmax(0,1fr)] gap-4">

            {/* ── Molecule depiction ── */}
            <div class="bg-base-100 rounded-lg border border-base-300 p-3">
              <svg viewBox={viewBox()} preserveAspectRatio="xMidYMid meet" class="w-full h-64">
                {/* Underglow layer — behind everything */}
                <For each={depiction()!.bonds}>
                  {(bond) => (
                    <Show when={isSel(bond.bondId)}>
                      <line x1={bond.x1} y1={bond.y1} x2={bond.x2} y2={bond.y2}
                            stroke={C_GLOW} stroke-width={GLOW_W}
                            vector-effect="non-scaling-stroke" stroke-linecap="round"
                            pointer-events="none" />
                    </Show>
                  )}
                </For>

                {/* Bond lines + hit areas */}
                <For each={depiction()!.bonds}>
                  {(bond) => (
                    <g class={isAct(bond.bondId) ? 'cursor-pointer' : ''}
                       onClick={() => handleBondClick(bond)}>
                      <Show when={isAct(bond.bondId)}>
                        <line x1={bond.x1} y1={bond.y1} x2={bond.x2} y2={bond.y2}
                              stroke="transparent" stroke-width={HIT_W}
                              vector-effect="non-scaling-stroke" stroke-linecap="round"
                              pointer-events="stroke" />
                      </Show>
                      {renderBondLines(bond)}
                    </g>
                  )}
                </For>

                {/* Atom labels with white halo */}
                <For each={depiction()!.atoms}>
                  {(atom) => (
                    <Show when={atom.showLabel}>
                      <text x={atom.x} y={atom.y}
                            text-anchor="middle" dominant-baseline="middle"
                            font-size={`${0.48 * VIEW_PAD}`}
                            fill="#0f172a"
                            stroke="white" stroke-width={`${0.18 * VIEW_PAD}`}
                            paint-order="stroke" stroke-linejoin="round"
                            class="select-none">
                        {atom.symbol}
                      </text>
                    </Show>
                  )}
                </For>
              </svg>
              <p class="mt-2 text-xs text-base-content/70">
                Click a <span class="font-semibold" style={{ color: C_ACTIVE }}>blue</span> bond to view its dihedral.
              </p>
            </div>

            {/* ── Right column ── */}
            <div class="flex flex-col gap-3 min-w-0">

              {/* Torsion table */}
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
                          <tr class={`cursor-pointer ${selectedTorsionId() === row.torsionId ? 'bg-primary/10' : ''}`}
                              onClick={() => setSelectedTorsionId(row.torsionId)}>
                            <td class="font-mono text-[11px]">{row.label}</td>
                            <td class="text-right font-mono text-[11px]">{fmtAngle(row.circularMean)}</td>
                            <td class="text-right font-mono text-[11px]">{fmtAngle(row.circularStd)}</td>
                          </tr>
                        )}
                      </For>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* ── Plot + detail ── */}
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
                      median {fmtAngle(selectedTorsion()!.median)}
                    </div>
                  </div>

                  {/* Plot SVG */}
                  <svg viewBox={`0 0 ${PLOT_WIDTH} ${PLOT_HEIGHT}`}
                       preserveAspectRatio="xMidYMid meet"
                       class="w-full bg-base-200 rounded overflow-hidden"
                       style={{ "aspect-ratio": `${PLOT_WIDTH} / ${PLOT_HEIGHT}` }}>

                    {/* Plot border */}
                    <rect x={PL} y={PT} width={PW} height={PH}
                          fill="none" stroke="#e2e8f0" stroke-width="1" />

                    {/* Y-axis grid + labels */}
                    <For each={Y_TICKS}>
                      {(tick) => {
                        const y = angleToY(tick);
                        return (
                          <>
                            <line x1={PL} y1={y} x2={PR} y2={y}
                                  stroke="#cbd5e1" stroke-width="1" />
                            <text x={M.left - 4} y={y + 3.5}
                                  font-size="10" fill="#475569" text-anchor="end">
                              {tick}°
                            </text>
                          </>
                        );
                      }}
                    </For>

                    {/* ── Trajectory view ── */}
                    <Show when={activeTab() === 'trajectory' && trajectoryPoints()}>
                      <polyline fill="none" stroke={C_TRAJ} stroke-width="1.5"
                                points={trajectoryPoints()} opacity="0.85" />
                      {/* X-axis frame labels */}
                      {(() => {
                        const frames = props.analysis.sampledFrameIndices;
                        if (!frames.length) return null;
                        return (
                          <>
                            <text x={PL} y={PB + 16} font-size="9" fill="#64748b" text-anchor="start">
                              {frames[0]}
                            </text>
                            <text x={PR} y={PB + 16} font-size="9" fill="#64748b" text-anchor="end">
                              {frames[frames.length - 1]}
                            </text>
                            <text x={(PL + PR) / 2} y={PB + 16} font-size="9" fill="#64748b" text-anchor="middle">
                              Frame
                            </text>
                          </>
                        );
                      })()}
                    </Show>

                    {/* ── Cluster lollipop view ── */}
                    <Show when={activeTab() === 'clusters' && clusterValues().length > 0}>
                      {(() => {
                        const vals = clusterValues();
                        const n = vals.length;
                        const span = Math.max(1, n - 1);
                        const zeroY = angleToY(0);
                        return (
                          <>
                            {/* 0° reference dashed line */}
                            <line x1={PL} y1={zeroY} x2={PR} y2={zeroY}
                                  stroke="#94a3b8" stroke-width="1" stroke-dasharray="4 3" />

                            <For each={vals}>
                              {(v, i) => {
                                const x = n === 1
                                  ? (PL + PR) / 2
                                  : PL + (i() / span) * PW;
                                const y = angleToY(v.angle);
                                const r = 4 + (v.population / 100) * 6;
                                return (
                                  <>
                                    {/* Stem from 0° to angle */}
                                    <line x1={x} y1={zeroY} x2={x} y2={y}
                                          stroke={C_CLUST} stroke-width="2" opacity="0.45" />
                                    {/* Dot (radius scaled by population) */}
                                    <circle cx={x} cy={y} r={r}
                                            fill={C_CLUST} opacity="0.85" />
                                    {/* Angle label above dot */}
                                    <text x={x} y={y - r - 3}
                                          font-size="9" fill="#0f766e" text-anchor="middle"
                                          font-weight="600">
                                      {v.angle.toFixed(0)}°
                                    </text>
                                    {/* Cluster ID below plot */}
                                    <text x={x} y={PB + 14}
                                          font-size="9" fill="#64748b" text-anchor="middle">
                                      C{v.clusterId + 1}
                                    </text>
                                    {/* Population below ID */}
                                    <text x={x} y={PB + 24}
                                          font-size="8" fill="#94a3b8" text-anchor="middle">
                                      {v.population.toFixed(0)}%
                                    </text>
                                  </>
                                );
                              }}
                            </For>
                          </>
                        );
                      })()}
                    </Show>
                  </svg>

                  {/* Below-plot info */}
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
                              {(v) => (
                                <tr>
                                  <td class="font-mono text-[11px]">{v.clusterId + 1}</td>
                                  <td class="text-right font-mono text-[11px]">{v.population.toFixed(1)}%</td>
                                  <td class="text-right font-mono text-[11px]">{fmtAngle(v.angle)}</td>
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
