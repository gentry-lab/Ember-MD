// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, Show, createSignal } from 'solid-js';
import { workflowStore } from '../../stores/workflow';

interface ScoringPanelProps {
  pdbPath: string;
  ligandSdfPath?: string;
  onClose: () => void;
}

interface ScoreResult {
  vinaRescore?: number;
  xtbStrainKcal?: number;
  cordialExpectedPkd?: number;
  cordialPHighAffinity?: number;
  cordialPVeryHighAffinity?: number;
}

const ScoringPanel: Component<ScoringPanelProps> = (props) => {
  const api = window.electronAPI;

  const [isScoring, setIsScoring] = createSignal(false);
  const [scores, setScores] = createSignal<ScoreResult | null>(null);
  const [error, setError] = createSignal<string | null>(null);
  const [progress, setProgress] = createSignal('');

  const runScoring = async () => {
    setIsScoring(true);
    setError(null);
    setScores(null);
    setProgress('Scoring complex...');

    try {
      const result = await api.scoreComplex(props.pdbPath, props.ligandSdfPath);
      if (result.ok) {
        setScores(result.value);
      } else {
        setError(result.error.message);
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsScoring(false);
      setProgress('');
    }
  };

  return (
    <div class="bg-base-200 rounded-lg p-3 text-xs">
      <div class="flex items-center justify-between mb-2">
        <span class="font-semibold text-sm">Score Complex</span>
        <button class="btn btn-ghost btn-xs" onClick={props.onClose}>
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <Show
        when={scores()}
        fallback={
          <div>
            <Show when={error()}>
              <div class="alert alert-error py-1 mb-2 text-xs">{error()}</div>
            </Show>
            <Show when={isScoring()}>
              <div class="flex items-center gap-2 mb-2">
                <span class="loading loading-spinner loading-xs" />
                <span>{progress()}</span>
              </div>
            </Show>
            <Show when={!isScoring()}>
              <p class="text-base-content/60 mb-2">
                Score with Vina rescore, xTB strain energy, and CORDIAL
              </p>
              <button
                class="btn btn-primary btn-xs w-full"
                onClick={runScoring}
              >
                Score
              </button>
            </Show>
          </div>
        }
      >
        <div class="space-y-1.5">
          <Show when={scores()!.vinaRescore != null}>
            <div class="flex justify-between">
              <span class="text-base-content/70">Vina rescore</span>
              <span class="font-mono">{scores()!.vinaRescore!.toFixed(1)} kcal/mol</span>
            </div>
          </Show>
          <Show when={scores()!.xtbStrainKcal != null}>
            <div class="flex justify-between">
              <span class="text-base-content/70">xTB strain</span>
              <span class={`font-mono ${
                scores()!.xtbStrainKcal! > 8 ? 'text-error'
                  : scores()!.xtbStrainKcal! > 5 ? 'text-warning' : ''
              }`}>
                {scores()!.xtbStrainKcal!.toFixed(1)} kcal/mol
              </span>
            </div>
          </Show>
          <Show when={scores()!.cordialPHighAffinity != null}>
            <div class="flex justify-between">
              <span class="text-base-content/70">{"P(< 1\u00B5M)"}</span>
              <span class="font-mono">
                {(scores()!.cordialPHighAffinity! * 100).toFixed(0)}%
              </span>
            </div>
          </Show>
          <Show when={scores()!.cordialPVeryHighAffinity != null}>
            <div class="flex justify-between">
              <span class="text-base-content/70">{"P(< 100nM)"}</span>
              <span class="font-mono">
                {(scores()!.cordialPVeryHighAffinity! * 100).toFixed(0)}%
              </span>
            </div>
          </Show>
          <Show when={scores()!.cordialExpectedPkd != null}>
            <div class="flex justify-between">
              <span class="text-base-content/70">Expected pKd</span>
              <span class="font-mono">{scores()!.cordialExpectedPkd!.toFixed(1)}</span>
            </div>
          </Show>
          <button
            class="btn btn-ghost btn-xs w-full mt-2"
            onClick={() => { setScores(null); setError(null); }}
          >
            Rescore
          </button>
        </div>
      </Show>
    </div>
  );
};

export default ScoringPanel;
