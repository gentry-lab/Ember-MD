import { Component, JSX, Show, For, createSignal, createEffect, onMount } from 'solid-js';
import { workflowStore, WorkflowMode, ViewerQueueItem } from '../../stores/workflow';
import HelpModal from '../HelpModal';
import AboutModal from '../AboutModal';
import { generateJobName, sanitizeJobName } from '../../utils/jobName';
import type { ProjectInfo, ProjectArtifact } from '../../../shared/types/ipc';

interface StepInfo {
  id: string;
  label: string;
  icon: string;
}

const viewerSteps: StepInfo[] = [
  { id: 'viewer-load', label: 'Load', icon: '1' },
  { id: 'viewer-view', label: 'View', icon: '2' },
];

const dockSteps: StepInfo[] = [
  { id: 'dock-load', label: 'Load', icon: '1' },
  { id: 'dock-configure', label: 'Configure', icon: '2' },
  { id: 'dock-progress', label: 'Dock', icon: '3' },
  { id: 'dock-results', label: 'Results', icon: '4' },
];

const mdSteps: StepInfo[] = [
  { id: 'md-load', label: 'Load', icon: '1' },
  { id: 'md-configure', label: 'Configure', icon: '2' },
  { id: 'md-progress', label: 'Simulate', icon: '3' },
  { id: 'md-results', label: 'Results', icon: '4' },
];

const dockStepOrder = dockSteps.map((s) => s.id);
const mdStepOrder = mdSteps.map((s) => s.id);

type PickerView = 'list' | 'rename' | 'delete';

interface WizardLayoutProps {
  children: JSX.Element;
}

const WizardLayout: Component<WizardLayoutProps> = (props) => {
  const {
    state, setMode, setJobName, setProjectReady,
    resetViewer, setViewerPdbPath, setViewerLigandPath,
    setViewerPdbQueue, setViewerTrajectoryPath,
  } = workflowStore;
  const [showHelp, setShowHelp] = createSignal(false);
  const [showAbout, setShowAbout] = createSignal(false);
  const api = window.electronAPI;

  // Project picker state
  const [projects, setProjects] = createSignal<ProjectInfo[]>([]);
  const [isLoadingProjects, setIsLoadingProjects] = createSignal(true);
  const [newProjectName, setNewProjectName] = createSignal(generateJobName());

  // Rename/delete state
  const [pickerView, setPickerView] = createSignal<PickerView>('list');
  const [targetProject, setTargetProject] = createSignal<ProjectInfo | null>(null);
  const [renameTo, setRenameTo] = createSignal('');
  const [renameError, setRenameError] = createSignal<string | null>(null);
  const [deleteConfirmText, setDeleteConfirmText] = createSignal('');
  const [deleteFileCount, setDeleteFileCount] = createSignal<{ fileCount: number; totalSizeMb: number } | null>(null);
  const [isProcessing, setIsProcessing] = createSignal(false);

  // Artifacts state
  const [artifacts, setArtifacts] = createSignal<ProjectArtifact[]>([]);

  const loadArtifacts = async () => {
    if (!state().projectReady || !state().jobName) {
      setArtifacts([]);
      return;
    }
    try {
      const result = await api.scanProjectArtifacts(state().jobName);
      setArtifacts(result);
      if (result.length > 0) {
        console.log(`[Nav] Scanned ${result.length} artifacts for ${state().jobName}:`, result.map((a: ProjectArtifact) => `${a.type}:${a.label}`));
      }
    } catch (err) {
      console.error('[Nav] Failed to scan artifacts:', err);
    }
  };

  // Reload artifacts when project becomes ready or jobName changes
  // Track only the specific fields to avoid re-triggering on every state change
  let lastArtifactProject = '';
  createEffect(() => {
    const ready = state().projectReady;
    const name = state().jobName;
    // Only re-scan if the project identity actually changed
    if (ready && name && name !== lastArtifactProject) {
      lastArtifactProject = name;
      loadArtifacts();
    }
  });

  interface ArtifactGroup {
    label: string;
    items: ProjectArtifact[];
  }

  const groupedArtifacts = (): ArtifactGroup[] => {
    const groups: ArtifactGroup[] = [];
    const a = artifacts();
    const prepared = a.filter((x) => x.type === 'prepared');
    const docking = a.filter((x) => x.type === 'docking');
    const sims = a.filter((x) => x.type === 'simulation');
    const trajs = a.filter((x) => x.type === 'trajectory');
    const clusters = a.filter((x) => x.type === 'cluster');

    if (prepared.length) groups.push({ label: 'Prepared', items: prepared });
    if (docking.length) groups.push({ label: 'Docking', items: docking });
    if (sims.length) groups.push({ label: 'Simulations', items: sims });
    if (trajs.length) groups.push({ label: 'Trajectories', items: trajs });
    if (clusters.length) groups.push({ label: 'Clusters', items: clusters });
    return groups;
  };

  const handleLoadArtifact = (artifact: ProjectArtifact) => {
    console.log(`[Nav] Load artifact: ${artifact.type} — ${artifact.label}`, artifact.path);

    // Close dropdown by blurring
    if (document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
    }

    resetViewer();

    switch (artifact.type) {
      case 'prepared':
      case 'simulation': {
        setViewerPdbPath(artifact.path);
        setMode('viewer');
        break;
      }
      case 'docking': {
        if (artifact.poses && artifact.poses.length > 0 && artifact.receptorPdb) {
          const queue: ViewerQueueItem[] = artifact.poses.map((p) => ({
            pdbPath: artifact.receptorPdb!,
            ligandPath: p.path,
            label: `${p.name}${p.affinity != null ? ` (${p.affinity.toFixed(1)} kcal/mol)` : ''}`,
          }));
          setViewerPdbQueue(queue);
          setViewerPdbPath(queue[0].pdbPath);
          setViewerLigandPath(queue[0].ligandPath || null);
        }
        setMode('viewer');
        break;
      }
      case 'trajectory': {
        if (artifact.systemPdb) {
          setViewerPdbPath(artifact.systemPdb);
          setViewerTrajectoryPath(artifact.path);
        }
        setMode('viewer');
        break;
      }
      case 'cluster': {
        // Build queue from cluster centroid PDBs via scan
        // For now, load the pooled/first centroid PDB directly
        setViewerPdbPath(artifact.path);
        setMode('viewer');
        break;
      }
    }
  };

  const loadProjects = async () => {
    setIsLoadingProjects(true);
    try {
      const result = await api.scanProjects();
      setProjects(result);
    } catch (err) {
      console.error('Failed to scan projects:', err);
    }
    setIsLoadingProjects(false);
  };

  onMount(loadProjects);

  const handleSelectProject = async (project: ProjectInfo) => {
    console.log(`[Nav] Select project: ${project.name} (${project.runs.length} runs)`);
    setJobName(project.name);
    await api.ensureProject(project.name);
    setProjectReady(true);
    loadArtifacts();
  };

  const handleNewProject = async () => {
    const name = newProjectName().trim();
    if (!name) return;
    console.log(`[Nav] New project: ${name}`);
    setJobName(name);
    await api.ensureProject(name);
    setProjectReady(true);
    loadArtifacts();
  };

  const handleRerollName = () => {
    setNewProjectName(generateJobName());
  };

  // Click project name in header → go back to picker
  const handleProjectNameClick = () => {
    if (state().isRunning) return;
    resetPickerView();
    setProjectReady(false);
    loadProjects();
  };

  const resetPickerView = () => {
    setPickerView('list');
    setTargetProject(null);
    setRenameTo('');
    setRenameError(null);
    setDeleteConfirmText('');
    setDeleteFileCount(null);
  };

  // Rename flow
  const handleStartRename = (e: MouseEvent, project: ProjectInfo) => {
    e.stopPropagation();
    setTargetProject(project);
    setRenameTo(project.name);
    setRenameError(null);
    setPickerView('rename');
  };

  const handleConfirmRename = async () => {
    const project = targetProject();
    const newName = renameTo().trim();
    if (!project || !newName) return;
    if (newName === project.name) { resetPickerView(); return; }

    setRenameError(null);
    setIsProcessing(true);
    try {
      const result = await api.renameProject(project.name, newName);
      if (result.ok) {
        resetPickerView();
        await loadProjects();
      } else {
        setRenameError(result.error?.message || 'Rename failed');
      }
    } catch (err) {
      setRenameError((err as Error).message);
    }
    setIsProcessing(false);
  };

  // Delete flow
  const handleStartDelete = async (e: MouseEvent, project: ProjectInfo) => {
    e.stopPropagation();
    setTargetProject(project);
    setDeleteConfirmText('');
    setDeleteFileCount(null);
    setPickerView('delete');

    // Load file count in background
    try {
      const info = await api.getProjectFileCount(project.name);
      setDeleteFileCount(info);
    } catch {
      setDeleteFileCount({ fileCount: 0, totalSizeMb: 0 });
    }
  };

  const handleConfirmDelete = async () => {
    const project = targetProject();
    if (!project || deleteConfirmText() !== 'delete') return;

    setIsProcessing(true);
    try {
      const result = await api.deleteProject(project.name);
      if (result.ok) {
        resetPickerView();
        await loadProjects();
      }
    } catch (err) {
      console.error('Delete failed:', err);
    }
    setIsProcessing(false);
  };

  const formatDate = (ms: number) => {
    const now = new Date();
    const diffDays = Math.floor((now.getTime() - ms) / 86400000);
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays}d ago`;
    const d = new Date(ms);
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  };

  const getStepStatus = (stepId: string): 'done' | 'active' | 'pending' => {
    if (state().mode === 'viewer') {
      const hasPdb = state().viewer.pdbPath !== null;
      if (stepId === 'viewer-load') return hasPdb ? 'done' : 'active';
      if (stepId === 'viewer-view') return hasPdb ? 'active' : 'pending';
      return 'pending';
    }
    if (state().mode === 'dock') {
      const currentStep = state().dockStep;
      const currentIndex = dockStepOrder.indexOf(currentStep);
      const stepIndex = dockStepOrder.indexOf(stepId);
      if (stepIndex < currentIndex) return 'done';
      if (stepIndex === currentIndex) return 'active';
      return 'pending';
    }
    const currentStep = state().mdStep;
    const currentIndex = mdStepOrder.indexOf(currentStep);
    const stepIndex = mdStepOrder.indexOf(stepId);
    if (stepIndex < currentIndex) return 'done';
    if (stepIndex === currentIndex) return 'active';
    return 'pending';
  };

  const canSwitchMode = () => {
    return !state().isRunning && state().projectReady;
  };

  const handleModeSwitch = (newMode: WorkflowMode) => {
    if (canSwitchMode() && newMode !== state().mode) {
      console.log(`[Nav] Mode switch: ${state().mode} → ${newMode}`);
      setMode(newMode);
    }
  };

  return (
    <div class="h-screen flex flex-col bg-base-100 overflow-hidden">
      {/* Draggable title bar area for macOS traffic lights */}
      <div class="h-6 bg-base-200 flex-shrink-0" style={{ "-webkit-app-region": "drag" }} />
      {/* Header + Steps combined */}
      <header class="bg-base-200 border-b border-base-300 px-4 py-2 flex items-center justify-between flex-shrink-0">
        <div class="flex items-center gap-3">
          {/* Mode selector segmented control */}
          <div class="tabs tabs-boxed bg-base-300 p-0.5">
            <button
              class={`tab tab-sm ${state().mode === 'viewer' ? 'tab-active' : ''}`}
              onClick={() => handleModeSwitch('viewer')}
              disabled={!canSwitchMode()}
            >
              View
            </button>
            <button
              class={`tab tab-sm ${state().mode === 'dock' ? 'tab-active' : ''}`}
              onClick={() => handleModeSwitch('dock')}
              disabled={!canSwitchMode()}
            >
              Dock
            </button>
            <button
              class={`tab tab-sm ${state().mode === 'md' ? 'tab-active' : ''}`}
              onClick={() => handleModeSwitch('md')}
              disabled={!canSwitchMode()}
            >
              Simulate
            </button>
            <button
              class={`tab tab-sm ${state().mode === 'score' ? 'tab-active' : ''}`}
              onClick={() => handleModeSwitch('score')}
              disabled={!canSwitchMode()}
            >
              Score
            </button>
          </div>
          {/* Help button */}
          <button
            class="btn btn-ghost btn-xs btn-circle"
            onClick={() => setShowHelp(true)}
            title="Help"
          >
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </button>
          {/* About button */}
          <button
            class="btn btn-ghost btn-xs btn-circle"
            onClick={() => setShowAbout(true)}
            title="About Ember"
          >
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </button>
        </div>

        {/* Project name + artifacts dropdown */}
        <Show when={state().projectReady}>
          <div class="flex items-center gap-1">
            <svg class="w-3.5 h-3.5 text-base-content/50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
            </svg>
            <button
              class="btn btn-ghost btn-xs h-auto py-0.5 font-mono text-xs"
              onClick={handleProjectNameClick}
              disabled={state().isRunning}
              title="Switch project"
            >
              {state().jobName}
            </button>
            {/* Artifacts dropdown */}
            <Show when={artifacts().length > 0}>
              <div class="dropdown dropdown-end">
                <label tabindex="0" class="btn btn-ghost btn-xs text-base-content/60 gap-0.5 px-1.5">
                  <span class="text-[10px]">{artifacts().length}</span>
                  <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                  </svg>
                </label>
                <ul tabindex="0" class="dropdown-content z-[1] menu menu-sm p-2 shadow bg-base-200 rounded-box w-64 max-h-60 overflow-y-auto">
                  <For each={groupedArtifacts()}>
                    {(group) => (
                      <>
                        <li class="menu-title"><span class="text-[10px] uppercase tracking-wider">{group.label}</span></li>
                        <For each={group.items}>
                          {(artifact) => (
                            <li><a class="text-xs" onClick={() => handleLoadArtifact(artifact)}>
                              {artifact.label}
                            </a></li>
                          )}
                        </For>
                      </>
                    )}
                  </For>
                </ul>
              </div>
            </Show>
          </div>
        </Show>

        {/* Step indicators — dock mode */}
        <Show when={state().mode === 'dock' && state().projectReady}>
          <ul class="steps steps-horizontal">
            <For each={dockSteps}>{(step) => {
              const status = getStepStatus(step.id);
              return (
                <li
                  class={`step step-sm ${status === 'done' || status === 'active' ? 'step-primary' : ''}`}
                  data-content={status === 'done' ? '✓' : step.icon}
                >
                  <span class={`text-xs ${status === 'active' ? 'font-semibold' : 'text-base-content/90'}`}>
                    {step.label}
                  </span>
                </li>
              );
            }}</For>
          </ul>
        </Show>

        {/* Step indicators — MD mode */}
        <Show when={state().mode === 'md' && state().projectReady}>
          <ul class="steps steps-horizontal">
            <For each={mdSteps}>{(step) => {
              const status = getStepStatus(step.id);
              return (
                <li
                  class={`step step-sm ${status === 'done' || status === 'active' ? 'step-primary' : ''}`}
                  data-content={status === 'done' ? '✓' : step.icon}
                >
                  <span class={`text-xs ${status === 'active' ? 'font-semibold' : 'text-base-content/90'}`}>
                    {step.label}
                  </span>
                </li>
              );
            }}</For>
          </ul>
        </Show>

        {/* Step indicators — View mode */}
        <Show when={state().mode === 'viewer' && state().projectReady}>
          <ul class="steps steps-horizontal">
            <For each={viewerSteps}>{(step) => {
              const status = getStepStatus(step.id);
              return (
                <li
                  class={`step step-sm ${status === 'done' || status === 'active' ? 'step-primary' : ''}`}
                  data-content={status === 'done' ? '✓' : step.icon}
                >
                  <span class={`text-xs ${status === 'active' ? 'font-semibold' : 'text-base-content/90'}`}>
                    {step.label}
                  </span>
                </li>
              );
            }}</For>
          </ul>
        </Show>
      </header>

      {/* Main content */}
      <main class="flex-1 min-h-0 overflow-auto relative">
        <div class="h-full max-w-4xl mx-auto px-4 py-3">{props.children}</div>

        {/* Project selection overlay — blocks all content until a project is picked */}
        <Show when={!state().projectReady}>
          <div class="absolute inset-0 z-30 bg-base-100 flex items-center justify-center">
            <Show when={!isLoadingProjects()} fallback={
              <span class="loading loading-spinner loading-md" />
            }>
              <div class="card bg-base-200 shadow-lg w-80">
                <div class="card-body p-5">

                  {/* === List view (default) === */}
                  <Show when={pickerView() === 'list'}>
                    <div class="text-center mb-3">
                      <h2 class="text-xl font-bold">Ember</h2>
                      <p class="text-xs text-base-content/60">GPU-accelerated molecular dynamics</p>
                    </div>

                    {/* Recent projects */}
                    <Show when={projects().length > 0}>
                      <p class="text-[10px] text-base-content/70 font-semibold uppercase tracking-wider mb-1.5">Recent Projects</p>
                      <div class="max-h-52 overflow-y-auto -mx-1 mb-3 space-y-0.5">
                        <For each={projects()}>
                          {(project) => (
                            <div class="group flex items-center rounded-lg hover:bg-base-300 transition-colors">
                              <button
                                class="flex-1 flex items-center gap-2 px-2.5 py-2 text-left min-w-0"
                                onClick={() => handleSelectProject(project)}
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
                              {/* Rename/delete buttons — visible on hover */}
                              <div class="flex-shrink-0 flex gap-0.5 pr-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                <button
                                  class="btn btn-ghost btn-xs btn-square"
                                  onClick={(e) => handleStartRename(e, project)}
                                  title="Rename project"
                                >
                                  <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                                  </svg>
                                </button>
                                <button
                                  class="btn btn-ghost btn-xs btn-square text-error"
                                  onClick={(e) => handleStartDelete(e, project)}
                                  title="Delete project"
                                >
                                  <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                  </svg>
                                </button>
                              </div>
                            </div>
                          )}
                        </For>
                      </div>
                      <div class="border-t border-base-300 mb-3" />
                    </Show>

                    {/* Empty state icon */}
                    <Show when={projects().length === 0}>
                      <div class="flex items-center justify-center mb-4">
                        <svg class="w-10 h-10 text-base-content/20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                        </svg>
                      </div>
                    </Show>

                    {/* New project input */}
                    <div class="flex items-center gap-1 mb-2">
                      <input
                        type="text"
                        class="input input-bordered input-sm flex-1 font-mono text-xs"
                        value={newProjectName()}
                        onInput={(e) => setNewProjectName(sanitizeJobName(e.currentTarget.value))}
                        onKeyDown={(e) => e.key === 'Enter' && handleNewProject()}
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
                      onClick={handleNewProject}
                      disabled={!newProjectName().trim()}
                    >
                      New Project
                    </button>
                  </Show>

                  {/* === Rename view === */}
                  <Show when={pickerView() === 'rename' && targetProject()}>
                    <p class="text-xs font-semibold mb-3">Rename Project</p>
                    <p class="text-[10px] text-base-content/60 mb-2">
                      Renaming <span class="font-mono font-medium">{targetProject()!.name}</span> will update the project folder and all output files.
                    </p>
                    <input
                      type="text"
                      class="input input-bordered input-sm w-full font-mono text-xs mb-2"
                      value={renameTo()}
                      onInput={(e) => { setRenameTo(sanitizeJobName(e.currentTarget.value)); setRenameError(null); }}
                      onKeyDown={(e) => e.key === 'Enter' && handleConfirmRename()}
                      autofocus
                    />
                    <Show when={renameError()}>
                      <p class="text-[10px] text-error mb-2">{renameError()}</p>
                    </Show>
                    <div class="flex gap-2">
                      <button class="btn btn-sm flex-1" onClick={resetPickerView} disabled={isProcessing()}>
                        Cancel
                      </button>
                      <button
                        class="btn btn-primary btn-sm flex-1"
                        onClick={handleConfirmRename}
                        disabled={!renameTo().trim() || renameTo() === targetProject()!.name || isProcessing()}
                      >
                        {isProcessing() ? <span class="loading loading-spinner loading-xs" /> : 'Rename'}
                      </button>
                    </div>
                  </Show>

                  {/* === Delete view === */}
                  <Show when={pickerView() === 'delete' && targetProject()}>
                    <p class="text-xs font-semibold text-error mb-3">Delete Project</p>
                    <p class="text-[10px] text-base-content/60 mb-2">
                      This will permanently delete <span class="font-mono font-medium">{targetProject()!.name}</span> and all its data.
                    </p>
                    <Show when={deleteFileCount()} fallback={
                      <div class="flex items-center gap-2 mb-3">
                        <span class="loading loading-spinner loading-xs" />
                        <span class="text-[10px] text-base-content/50">Counting files...</span>
                      </div>
                    }>
                      <div class="bg-error/10 rounded-lg px-3 py-2 mb-3">
                        <p class="text-xs font-medium text-error">
                          {deleteFileCount()!.fileCount} files ({deleteFileCount()!.totalSizeMb} MB) will be removed
                        </p>
                      </div>
                    </Show>
                    <p class="text-[10px] text-base-content/60 mb-1">
                      Type <span class="font-mono font-bold">delete</span> to confirm:
                    </p>
                    <input
                      type="text"
                      class="input input-bordered input-sm w-full font-mono text-xs mb-3"
                      value={deleteConfirmText()}
                      onInput={(e) => setDeleteConfirmText(e.currentTarget.value.toLowerCase())}
                      onKeyDown={(e) => e.key === 'Enter' && deleteConfirmText() === 'delete' && handleConfirmDelete()}
                      placeholder="delete"
                      autofocus
                    />
                    <div class="flex gap-2">
                      <button class="btn btn-sm flex-1" onClick={resetPickerView} disabled={isProcessing()}>
                        Cancel
                      </button>
                      <button
                        class="btn btn-error btn-sm flex-1"
                        onClick={handleConfirmDelete}
                        disabled={deleteConfirmText() !== 'delete' || isProcessing()}
                      >
                        {isProcessing() ? <span class="loading loading-spinner loading-xs" /> : 'Delete'}
                      </button>
                    </div>
                  </Show>

                </div>
              </div>
            </Show>
          </div>
        </Show>
      </main>

      {/* Help Modal */}
      <HelpModal
        isOpen={showHelp()}
        onClose={() => setShowHelp(false)}
      />
      {/* About Modal */}
      <AboutModal
        isOpen={showAbout()}
        onClose={() => setShowAbout(false)}
      />
    </div>
  );
};

export default WizardLayout;
