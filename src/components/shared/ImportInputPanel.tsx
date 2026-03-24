import { Component, Show } from 'solid-js';
import type { JSXElement } from 'solid-js';

interface ImportInputPanelProps {
  importButtonLabel: string;
  onImport: () => void;
  importDisabled?: boolean;
  importLoading?: boolean;
  showPdbFetch?: boolean;
  pdbIdValue?: string;
  onPdbIdInput?: (value: string) => void;
  onFetchPdb?: () => void;
  fetchDisabled?: boolean;
  fetchLoading?: boolean;
  showSmiles?: boolean;
  smilesValue?: string;
  onSmilesInput?: (value: string) => void;
  onSubmitSmiles?: () => void;
  smilesDisabled?: boolean;
  smilesLoading?: boolean;
  smilesCount?: number;
  statusText?: string | null;
  beforeInputs?: JSXElement;
  afterInputs?: JSXElement;
}

const ImportInputPanel: Component<ImportInputPanelProps> = (props) => (
  <div class="space-y-3">
    <Show when={props.beforeInputs}>
      {props.beforeInputs}
    </Show>

    <Show when={props.statusText}>
      <p class="text-[10px] text-base-content/60 text-center">{props.statusText}</p>
    </Show>

    <button
      class="btn btn-outline btn-sm w-full"
      onClick={props.onImport}
      disabled={props.importDisabled}
    >
      <Show when={props.importLoading} fallback={props.importButtonLabel}>
        <span class="loading loading-spinner loading-xs" />
      </Show>
    </button>

    <Show when={props.showPdbFetch}>
      <div>
        <span class="text-[10px] text-base-content/50">or enter PDB ID</span>
        <div class="flex gap-1 mt-1">
          <input
            type="text"
            class="input input-bordered input-sm flex-1 font-mono uppercase"
            placeholder="e.g. 8TCE"
            value={props.pdbIdValue || ''}
            onInput={(e) => props.onPdbIdInput?.(e.currentTarget.value)}
            onKeyDown={(e) => e.key === 'Enter' && props.onFetchPdb?.()}
            maxLength={4}
          />
          <button
            class="btn btn-primary btn-sm"
            onClick={() => props.onFetchPdb?.()}
            disabled={props.fetchDisabled}
          >
            <Show when={props.fetchLoading} fallback={'Fetch'}>
              <span class="loading loading-spinner loading-xs" />
            </Show>
          </button>
        </div>
      </div>
    </Show>

    <Show when={props.showSmiles}>
      <div>
        <div class="flex items-center justify-between mb-1">
          <span class="text-[10px] text-base-content/50">or enter SMILES</span>
          <Show when={(props.smilesCount || 0) > 0}>
            <span class="text-[10px] font-mono text-success">
              {props.smilesCount} input{props.smilesCount === 1 ? '' : 's'}
            </span>
          </Show>
        </div>
        <textarea
          class="textarea textarea-bordered text-xs font-mono w-full resize-none leading-relaxed"
          placeholder="Enter SMILES strings (one compound per line)"
          value={props.smilesValue || ''}
          onInput={(e) => props.onSmilesInput?.(e.currentTarget.value)}
          rows={4}
        />
      </div>
    </Show>

    <Show when={props.showSmiles}>
      <button
        class="btn btn-primary btn-sm w-full"
        onClick={() => props.onSubmitSmiles?.()}
        disabled={props.smilesDisabled}
      >
        <Show when={props.smilesLoading} fallback={'Enter SMILES'}>
          <span class="loading loading-spinner loading-xs" />
        </Show>
      </button>
    </Show>

    <Show when={props.afterInputs}>
      {props.afterInputs}
    </Show>
  </div>
);

export default ImportInputPanel;
