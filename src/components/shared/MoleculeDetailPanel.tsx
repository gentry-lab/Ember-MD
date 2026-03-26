// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, Show, For, createSignal, createEffect, on } from 'solid-js';
import type { MoleculeDetailsResult } from '../../../shared/types/ipc';

export interface DetailScore {
  label: string;
  value: string | number | null | undefined;
  unit?: string;
}

export interface DetailDihedral {
  label: string;
  angle: number;
}

interface MoleculeDetailPanelProps {
  sdfPath: string | null;
  referenceSdfPath?: string | null;
  scores?: DetailScore[];
  dihedrals?: DetailDihedral[];
  label?: string;
  sublabel?: string;
}

const MoleculeDetailPanel: Component<MoleculeDetailPanelProps> = (props) => {
  const api = window.electronAPI;
  const [details, setDetails] = createSignal<MoleculeDetailsResult | null>(null);
  const [loading, setLoading] = createSignal(false);

  createEffect(on(
    () => props.sdfPath,
    async (sdfPath) => {
      setDetails(null);
      if (!sdfPath) return;
      setLoading(true);
      try {
        const result = await api.getMoleculeDetails(sdfPath, props.referenceSdfPath || undefined);
        if (result.ok) {
          setDetails(result.value);
        }
      } catch { /* ignore */ }
      setLoading(false);
    },
  ));

  const formatCoord = (v: number) => v.toFixed(1);

  return (
    <div class="w-56 flex-shrink-0 bg-base-200 rounded-lg p-3 space-y-2 overflow-y-auto">
      {/* Label */}
      <Show when={props.label}>
        <div>
          <p class="text-xs font-semibold truncate">{props.label}</p>
          <Show when={props.sublabel}>
            <p class="text-[10px] text-base-content/60 truncate">{props.sublabel}</p>
          </Show>
        </div>
      </Show>

      {/* Thumbnail */}
      <Show when={loading()}>
        <div class="flex items-center justify-center h-32">
          <span class="loading loading-spinner loading-sm" />
        </div>
      </Show>
      <Show when={!loading() && details()?.thumbnail}>
        <div class="flex justify-center">
          <img
            src={details()!.thumbnail!}
            alt="2D structure"
            class="rounded border border-base-300 bg-white"
            style={{ width: '200px', height: '200px', 'object-fit': 'contain' }}
          />
        </div>
      </Show>
      <Show when={!loading() && !details()?.thumbnail && props.sdfPath}>
        <div class="flex items-center justify-center h-32 text-xs text-base-content/40">
          No 2D image
        </div>
      </Show>

      {/* Scores */}
      <Show when={props.scores && props.scores.length > 0}>
        <div class="space-y-0.5">
          <For each={props.scores}>{(score) => (
            <Show when={score.value != null}>
              <div class="flex justify-between text-[11px]">
                <span class="text-base-content/70">{score.label}</span>
                <span class="font-mono font-medium">
                  {typeof score.value === 'number' ? score.value.toFixed(score.unit === 'kcal/mol' ? 1 : score.unit === 'deg' ? 1 : 3) : score.value}
                  <Show when={score.unit}>
                    <span class="text-base-content/50 ml-0.5">{score.unit}</span>
                  </Show>
                </span>
              </div>
            </Show>
          )}</For>
        </div>
      </Show>

      {/* RMSD + Centroid from molecule details */}
      <Show when={details() && (details()!.rmsd != null || details()!.centroid != null)}>
        <div class="border-t border-base-300 pt-1.5 space-y-0.5">
          <Show when={details()!.rmsd != null}>
            <div class="flex justify-between text-[11px]">
              <span class="text-base-content/70">RMSD</span>
              <span class="font-mono font-medium">
                {details()!.rmsd!.toFixed(2)}
                <span class="text-base-content/50 ml-0.5">A</span>
              </span>
            </div>
          </Show>
          <Show when={details()!.centroid != null}>
            <div class="flex justify-between text-[11px]">
              <span class="text-base-content/70">Centroid</span>
              <span class="font-mono font-medium text-[10px]">
                ({formatCoord(details()!.centroid!.x)}, {formatCoord(details()!.centroid!.y)}, {formatCoord(details()!.centroid!.z)})
              </span>
            </div>
          </Show>
        </div>
      </Show>

      {/* Dihedrals (MD only) */}
      <Show when={props.dihedrals && props.dihedrals.length > 0}>
        <div class="border-t border-base-300 pt-1.5">
          <p class="text-[10px] font-semibold text-base-content/70 mb-1">Rotatable Bonds</p>
          <div class="space-y-0.5 max-h-32 overflow-y-auto">
            <For each={props.dihedrals}>{(d) => (
              <div class="flex justify-between text-[10px]">
                <span class="text-base-content/60 truncate mr-2">{d.label}</span>
                <span class="font-mono font-medium flex-shrink-0">{d.angle.toFixed(1)}&deg;</span>
              </div>
            )}</For>
          </div>
        </div>
      </Show>

      {/* MW / LogP from details */}
      <Show when={details() && (details()!.mw > 0 || details()!.logp !== 0)}>
        <div class="border-t border-base-300 pt-1.5 space-y-0.5">
          <Show when={details()!.mw > 0}>
            <div class="flex justify-between text-[10px]">
              <span class="text-base-content/50">MW</span>
              <span class="font-mono text-base-content/60">{details()!.mw.toFixed(1)}</span>
            </div>
          </Show>
          <Show when={details()!.logp !== 0}>
            <div class="flex justify-between text-[10px]">
              <span class="text-base-content/50">LogP</span>
              <span class="font-mono text-base-content/60">{details()!.logp.toFixed(2)}</span>
            </div>
          </Show>
        </div>
      </Show>
    </div>
  );
};

export default MoleculeDetailPanel;
