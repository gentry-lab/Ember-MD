import { Component, For, Show, createSignal, onMount } from 'solid-js';
import { workflowStore } from '../../stores/workflow';
import { generateJobName, sanitizeJobName } from '../../utils/jobName';
import type { ProjectInfo } from '../../../shared/types/ipc';

const MDStepHome: Component = () => {
  const {
    state,
    setMdStep,
    setJobName,
  } = workflowStore;

  const api = window.electronAPI;

  const [projects, setProjects] = createSignal<ProjectInfo[]>([]);
  const [isLoading, setIsLoading] = createSignal(true);
  const [newProjectName, setNewProjectName] = createSignal(generateJobName());

  onMount(async () => {
    setIsLoading(true);
    try {
      const result = await api.scanProjects();
      setProjects(result);
    } catch (err) {
      console.error('Failed to scan projects:', err);
    }
    setIsLoading(false);
  });

  const handleNewSimulation = () => {
    const name = newProjectName().trim();
    if (!name) return;
    setJobName(name);
    setMdStep('md-load');
  };

  const handleRerollName = () => {
    setNewProjectName(generateJobName());
  };

  const handleOpenProject = (project: ProjectInfo) => {
    setJobName(project.name);
    setMdStep('md-load');
  };

  const formatDate = (ms: number) => {
    const d = new Date(ms);
    const now = new Date();
    const diffDays = Math.floor((now.getTime() - ms) / 86400000);
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays}d ago`;
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  };

  return (
    <div class="h-full flex flex-col">
      {/* Title */}
      <div class="text-center mb-4">
        <h2 class="text-2xl font-bold">Ember</h2>
        <p class="text-sm text-base-content/70">GPU-accelerated molecular dynamics on Apple Silicon</p>
      </div>

      <Show when={!isLoading()} fallback={
        <div class="flex-1 flex items-center justify-center">
          <span class="loading loading-spinner loading-md"></span>
        </div>
      }>
        <div class="flex-1 flex items-center justify-center">
          <div class="card bg-base-200 shadow-lg w-80">
            <div class="card-body p-5">

              {/* Recent projects */}
              <Show when={projects().length > 0}>
                <p class="text-[10px] text-base-content/70 font-semibold uppercase tracking-wider mb-1.5">Recent Projects</p>
                <div class="max-h-52 overflow-y-auto -mx-1 mb-3 space-y-0.5">
                  <For each={projects()}>
                    {(project) => (
                      <button
                        class="w-full flex items-center gap-2 px-2.5 py-2 rounded-lg hover:bg-base-300 active:bg-base-300/80 text-left transition-colors"
                        onClick={() => handleOpenProject(project)}
                      >
                        <svg class="w-4 h-4 text-primary/60 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                        </svg>
                        <div class="flex-1 min-w-0 flex items-baseline gap-1">
                          <span class="text-xs font-medium truncate">{project.name}</span>
                          <span class="text-[10px] text-base-content/50 flex-shrink-0">({project.runs.length})</span>
                        </div>
                        <span class="text-[10px] text-base-content/60 flex-shrink-0">
                          {formatDate(project.lastModified)}
                        </span>
                      </button>
                    )}
                  </For>
                </div>
                <div class="border-t border-base-300 mb-3"></div>
              </Show>

              {/* New project */}
              <Show when={projects().length === 0}>
                <div class="flex items-center justify-center mb-4">
                  <svg class="w-10 h-10 text-base-content/20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                  </svg>
                </div>
              </Show>

              <div class="flex items-center gap-1 mb-2">
                <input
                  type="text"
                  class="input input-bordered input-sm flex-1 font-mono text-xs"
                  value={newProjectName()}
                  onInput={(e) => setNewProjectName(sanitizeJobName(e.currentTarget.value))}
                  onKeyDown={(e) => e.key === 'Enter' && handleNewSimulation()}
                  placeholder="project-name"
                />
                <button
                  class="btn btn-ghost btn-sm btn-square"
                  onClick={handleRerollName}
                  title="Random name"
                >
                  <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                </button>
              </div>
              <button
                class="btn btn-primary btn-sm w-full"
                onClick={handleNewSimulation}
                disabled={!newProjectName().trim()}
              >
                New Simulation
              </button>
            </div>
          </div>
        </div>
      </Show>
    </div>
  );
};

export default MDStepHome;
