import { Component, Show, For } from 'solid-js';
import { workflowStore, BindingSiteMapChannel } from '../../stores/workflow';

interface BindingSiteMapPanelProps {
  onCompute: () => void;
  onClear: () => void;
}

const CHANNELS = [
  { key: 'hydrophobic' as const, label: 'Hydrophobic', color: '#22c55e' },
  { key: 'hbondDonor' as const, label: 'H-Donor', color: '#3b82f6' },
  { key: 'hbondAcceptor' as const, label: 'H-Acceptor', color: '#ef4444' },
];

const BindingSiteMapPanel: Component<BindingSiteMapPanelProps> = (props) => {
  const { state, setViewerBindingSiteChannel } = workflowStore;

  const bsMap = () => state().viewer.bindingSiteMap;
  const isComputing = () => state().viewer.isComputingBindingSiteMap;

  const getChannel = (key: 'hydrophobic' | 'hbondDonor' | 'hbondAcceptor'): BindingSiteMapChannel | null => {
    const map = bsMap();
    if (!map) return null;
    return map[key];
  };

  const hotspotCount = () => {
    const map = bsMap();
    return map ? map.hotspots.length : 0;
  };

  return (
    <div class="flex flex-col gap-1">
      <Show when={!bsMap()}>
        <button
          class="btn btn-xs btn-outline btn-accent gap-1"
          onClick={props.onCompute}
          disabled={isComputing()}
          title="Compute binding site interaction maps around the ligand"
        >
          {isComputing() ? (
            <>
              <span class="loading loading-spinner loading-xs" />
              Computing...
            </>
          ) : (
            <>
              <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Expand
            </>
          )}
        </button>
      </Show>

      <Show when={bsMap()}>
        <div class="card bg-base-300/50 p-1.5">
          <div class="flex items-center justify-between mb-1">
            <span class="text-xs font-semibold">Interaction Maps</span>
            <div class="flex items-center gap-1">
              <Show when={hotspotCount() > 0}>
                <span class="badge badge-xs badge-accent">{hotspotCount()} hotspots</span>
              </Show>
              <button
                class="btn btn-xs btn-ghost btn-square"
                onClick={props.onClear}
                title="Clear interaction maps"
              >
                <svg class="w-3 h-3" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd" />
                </svg>
              </button>
            </div>
          </div>

          <For each={CHANNELS}>
            {(ch) => {
              const channel = () => getChannel(ch.key);
              return (
                <Show when={channel()}>
                  <div class="flex items-center gap-1.5 py-0.5">
                    {/* Color swatch + toggle */}
                    <label class="flex items-center gap-1 cursor-pointer">
                      <input
                        type="checkbox"
                        class="checkbox checkbox-xs"
                        checked={channel()!.visible}
                        onChange={(e) =>
                          setViewerBindingSiteChannel(ch.key, { visible: e.currentTarget.checked })
                        }
                      />
                      <span
                        class="w-2 h-2 rounded-full inline-block"
                        style={{ background: ch.color }}
                      />
                      <span class="text-xs w-16">{ch.label}</span>
                    </label>

                    {/* Isolevel slider */}
                    <input
                      type="range"
                      class="range range-xs flex-1"
                      min="0.05"
                      max="0.95"
                      step="0.05"
                      value={channel()!.isolevel}
                      onInput={(e) =>
                        setViewerBindingSiteChannel(ch.key, {
                          isolevel: parseFloat(e.currentTarget.value),
                        })
                      }
                      title={`Isolevel: ${channel()!.isolevel.toFixed(2)}`}
                    />

                    {/* Opacity dropdown */}
                    <select
                      class="select select-xs select-bordered w-14"
                      value={channel()!.opacity}
                      onChange={(e) =>
                        setViewerBindingSiteChannel(ch.key, {
                          opacity: parseFloat(e.currentTarget.value),
                        })
                      }
                    >
                      <option value="0.3">30%</option>
                      <option value="0.5">50%</option>
                      <option value="0.7">70%</option>
                      <option value="1.0">100%</option>
                    </select>
                  </div>
                </Show>
              );
            }}
          </For>
        </div>
      </Show>
    </div>
  );
};

export default BindingSiteMapPanel;
