import { Component, JSX, Show, For, createSignal, createEffect, onMount } from 'solid-js';
import { workflowStore, WorkflowMode, ViewerQueueItem } from '../../stores/workflow';
import HelpModal from '../HelpModal';
import AboutModal from '../AboutModal';
import { generateJobName, sanitizeJobName } from '../../utils/jobName';
import type { ProjectInfo, ProjectJob } from '../../../shared/types/ipc';
import { theme, toggleTheme } from '../../utils/theme';

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

const conformSteps: StepInfo[] = [
  { id: 'conform-load', label: 'Load', icon: '1' },
  { id: 'conform-configure', label: 'Configure', icon: '2' },
  { id: 'conform-progress', label: 'Generate', icon: '3' },
  { id: 'conform-results', label: 'Results', icon: '4' },
];

const dockStepOrder = dockSteps.map((s) => s.id);
const mdStepOrder = mdSteps.map((s) => s.id);
const conformStepOrder = conformSteps.map((s) => s.id);

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

  // Job selector state
  const [jobs, setJobs] = createSignal<ProjectJob[]>([]);
  const [activeJobId, setActiveJobId] = createSignal<string | null>(null);

  const loadJobs = async () => {
    if (!state().projectReady || !state().jobName) {
      setJobs([]);
      return;
    }
    try {
      const result = await api.scanProjectArtifacts(state().jobName);
      setJobs(result);
      if (result.length > 0) {
        console.log(`[Nav] Scanned ${result.length} jobs for ${state().jobName}:`, result.map((j: ProjectJob) => j.id));
      }
    } catch (err) {
      console.error('[Nav] Failed to scan jobs:', err);
    }
  };

  // Reload jobs when project identity changes
  let lastJobProject = '';
  createEffect(() => {
    const ready = state().projectReady;
    const name = state().jobName;
    if (ready && name && name !== lastJobProject) {
      lastJobProject = name;
      loadJobs();
    }
  });

  const activeJob = () => jobs().find((j) => j.id === activeJobId()) || null;

  const [jobDropdownOpen, setJobDropdownOpen] = createSignal(false);

  const jobSelectorLabel = () => {
    const count = jobs().length;
    if (count === 0) return 'No jobs';
    return `Load Previous Job (${count})`;
  };

  const handleSelectJob = async (job: ProjectJob) => {
    console.log(`[Nav] Select job: ${job.id} — ${job.label}`, job.path);
    setActiveJobId(job.id);

    // Close dropdown by blurring
    if (document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
    }

    const {
      setDockOutputDir, setDockResults, setDockReceptorPdbPath,
      setDockConfig, setDockStep, setDockCordialScored,
      setMdResult, setMdOutputDir, setMdStep,
    } = workflowStore;

    if (job.type === 'docking') {
      // Re-hydrate docking results from disk, navigate to results screen
      try {
        const parseResult = await api.parseDockResults(job.path);
        if (parseResult.ok) {
          setDockOutputDir(job.path);
          setDockResults(parseResult.value);
          if (job.receptorPdb) setDockReceptorPdbPath(job.receptorPdb);
          // Check if any result has CORDIAL scores
          const hasCordial = parseResult.value.some((r: any) => r.cordialExpectedPkd != null);
          setDockCordialScored(hasCordial);
          setMode('dock');
          setDockStep('dock-results');
        } else {
          // Fallback: open in viewer
          resetViewer();
          if (job.poses && job.poses.length > 0 && job.receptorPdb) {
            const queue: ViewerQueueItem[] = job.poses.map((p) => ({
              pdbPath: job.receptorPdb!,
              ligandPath: p.path,
              label: `${p.name}${p.affinity != null ? ` (${p.affinity.toFixed(1)} kcal/mol)` : ''}`,
            }));
            setViewerPdbQueue(queue);
            setViewerPdbPath(queue[0].pdbPath);
            setViewerLigandPath(queue[0].ligandPath || null);
          }
          setMode('viewer');
        }
      } catch {
        // On error, fall back to viewer
        resetViewer();
        setMode('viewer');
      }
    } else if (job.type === 'simulation') {
      // Re-hydrate MD result from job paths, navigate to results screen
      const systemPdb = job.systemPdb || '';
      const trajectoryDcd = job.trajectoryDcd || '';
      const finalPdb = job.finalPdb || systemPdb;

      if (systemPdb && trajectoryDcd) {
        setMdResult({
          systemPdbPath: systemPdb,
          trajectoryPath: trajectoryDcd,
          equilibratedPdbPath: systemPdb,
          finalPdbPath: finalPdb,
          energyCsvPath: trajectoryDcd.replace('trajectory.dcd', 'energy.csv'),
        });
        setMdOutputDir(job.path);
        setMode('md');
        setMdStep('md-results');
      } else {
        // No trajectory — open final PDB in viewer
        resetViewer();
        if (finalPdb) setViewerPdbPath(finalPdb);
        setMode('viewer');
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
    loadJobs();
  };

  const handleNewProject = async () => {
    const name = newProjectName().trim();
    if (!name) return;
    console.log(`[Nav] New project: ${name}`);
    setJobName(name);
    await api.ensureProject(name);
    setProjectReady(true);
    loadJobs();
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
    if (state().mode === 'conform') {
      const currentStep = state().conformStep;
      const currentIndex = conformStepOrder.indexOf(currentStep);
      const stepIndex = conformStepOrder.indexOf(stepId);
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
      {/* Header: mode tabs (left) | project + job (center) | step indicators (right) */}
      <header class="bg-base-200 border-b border-base-300 px-4 py-2 flex items-center flex-shrink-0">
        {/* Left: Mode tabs + help/about */}
        <div class="flex items-center gap-3 flex-shrink-0">
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
              class={`tab tab-sm ${state().mode === 'conform' ? 'tab-active' : ''}`}
              onClick={() => handleModeSwitch('conform')}
              disabled={!canSwitchMode()}
            >
              Conform
            </button>
            <button
              class={`tab tab-sm ${state().mode === 'map' ? 'tab-active' : ''}`}
              onClick={() => handleModeSwitch('map')}
              disabled={!canSwitchMode()}
            >
              Map
            </button>
            <button
              class={`tab tab-sm ${state().mode === 'md' ? 'tab-active' : ''}`}
              onClick={() => handleModeSwitch('md')}
              disabled={!canSwitchMode()}
            >
              Simulate
            </button>
            <button
              class={`tab tab-sm ${state().mode === 'score' ? 'tab-active' : ''} opacity-40`}
              disabled={true}
              title="FEP Scoring (coming soon)"
            >
              FEP
            </button>
          </div>
          <button
            class="btn btn-ghost btn-xs btn-circle"
            onClick={() => setShowHelp(true)}
            title="Help"
          >
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </button>
          <button
            class="btn btn-ghost btn-xs btn-circle"
            onClick={() => setShowAbout(true)}
            title="About Ember"
          >
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </button>
          <button
            class="btn btn-ghost btn-xs btn-circle"
            onClick={toggleTheme}
            title="Toggle dark mode"
          >
            <Show when={theme() === 'business'} fallback={
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M12 3v1m0 16v1m8.66-9h-1M4.34 12h-1m15.07-6.07-.71.71M6.34 17.66l-.71.71M17.66 17.66l.71.71M6.34 6.34l.71.71M12 5a7 7 0 100 14A7 7 0 0012 5z" />
              </svg>
            }>
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" />
              </svg>
            </Show>
          </button>
        </div>

        {/* Center: Project name + Job selector — stacked vertically */}
        <Show when={state().projectReady}>
          <div class="flex-1 flex justify-center">
            <div class="flex flex-col items-center gap-0.5">
              <button
                class="btn btn-ghost btn-xs h-auto py-0.5 font-mono text-sm gap-1"
                onClick={handleProjectNameClick}
                disabled={state().isRunning}
                title="Switch project"
              >
                <svg class="w-3.5 h-3.5 text-base-content/50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                </svg>
                {state().jobName}
              </button>
              <Show when={jobs().length > 0}>
                <div class="relative">
                  <button
                    class="btn btn-ghost btn-xs gap-1 px-2 text-xs text-base-content/70"
                    onClick={() => setJobDropdownOpen(!jobDropdownOpen())}
                  >
                    {jobSelectorLabel()}
                    <svg class={`w-3 h-3 transition-transform ${jobDropdownOpen() ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>
                  <Show when={jobDropdownOpen()}>
                    <div class="fixed inset-0 z-[998]" onClick={() => setJobDropdownOpen(false)} />
                    <ul class="absolute top-full left-1/2 -translate-x-1/2 mt-1 z-[999] menu p-2 shadow-lg bg-base-200 border border-base-300 rounded-box w-72 max-h-72 overflow-y-auto">
                      <Show when={jobs().some((j) => j.type === 'docking')}>
                        <li class="menu-title"><span class="text-xs uppercase tracking-wider font-semibold">Docking</span></li>
                        <For each={jobs().filter((j) => j.type === 'docking')}>
                          {(job) => (
                            <li>
                              <a
                                class={`text-sm ${activeJobId() === job.id ? 'active' : ''}`}
                                onClick={() => { setJobDropdownOpen(false); handleSelectJob(job); }}
                              >
                                {job.label}
                              </a>
                            </li>
                          )}
                        </For>
                      </Show>
                      <Show when={jobs().some((j) => j.type === 'simulation')}>
                        <li class="menu-title"><span class="text-xs uppercase tracking-wider font-semibold">Simulations</span></li>
                        <For each={jobs().filter((j) => j.type === 'simulation')}>
                          {(job) => (
                            <li>
                              <a
                                class={`text-sm ${activeJobId() === job.id ? 'active' : ''}`}
                                onClick={() => { setJobDropdownOpen(false); handleSelectJob(job); }}
                              >
                                {job.label}
                              </a>
                            </li>
                          )}
                        </For>
                      </Show>
                    </ul>
                  </Show>
                </div>
              </Show>
            </div>
          </div>
        </Show>

        {/* Right: Step indicators (all modes) */}
        <div class="flex-shrink-0">
        {/* Step indicators — dock mode */}
        <Show when={state().mode === 'dock' && state().projectReady}>
          <ul class="steps steps-horizontal">
            <For each={dockSteps}>{(step) => {
              const status = () => getStepStatus(step.id);
              return (
                <li
                  class={`step step-sm ${status() === 'done' || status() === 'active' ? 'step-primary' : ''}`}
                  data-content={status() === 'done' ? '✓' : step.icon}
                >
                  <span class={`text-xs ${status() === 'active' ? 'font-semibold' : 'text-base-content/90'}`}>
                    {step.label}
                  </span>
                </li>
              );
            }}</For>
          </ul>
        </Show>

        {/* Step indicators — Conform mode */}
        <Show when={state().mode === 'conform' && state().projectReady}>
          <ul class="steps steps-horizontal">
            <For each={conformSteps}>{(step) => {
              const status = () => getStepStatus(step.id);
              return (
                <li
                  class={`step step-sm ${status() === 'done' || status() === 'active' ? 'step-primary' : ''}`}
                  data-content={status() === 'done' ? '✓' : step.icon}
                >
                  <span class={`text-xs ${status() === 'active' ? 'font-semibold' : 'text-base-content/90'}`}>
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
              const status = () => getStepStatus(step.id);
              return (
                <li
                  class={`step step-sm ${status() === 'done' || status() === 'active' ? 'step-primary' : ''}`}
                  data-content={status() === 'done' ? '✓' : step.icon}
                >
                  <span class={`text-xs ${status() === 'active' ? 'font-semibold' : 'text-base-content/90'}`}>
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
              const status = () => getStepStatus(step.id);
              return (
                <li
                  class={`step step-sm ${status() === 'done' || status() === 'active' ? 'step-primary' : ''}`}
                  data-content={status() === 'done' ? '✓' : step.icon}
                >
                  <span class={`text-xs ${status() === 'active' ? 'font-semibold' : 'text-base-content/90'}`}>
                    {step.label}
                  </span>
                </li>
              );
            }}</For>
          </ul>
        </Show>

        {/* Step indicators — Score mode */}
        <Show when={state().mode === 'score' && state().projectReady}>
          <span class="text-xs text-base-content/60">FEP Scoring</span>
        </Show>

        {/* Step indicators — Map mode */}
        <Show when={state().mode === 'map' && state().projectReady}>
          <span class="text-xs text-base-content/60">Pocket Mapping</span>
        </Show>
        </div>
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
              <div class="card bg-base-200 shadow-lg w-96">
                <div class="card-body p-6">

                  {/* === List view (default) === */}
                  <Show when={pickerView() === 'list'}>
                    <div class="text-center mb-4">
                      <h2 class="text-2xl font-bold">Ember</h2>
                      <p class="text-sm text-base-content/60">GPU-accelerated molecular dynamics</p>
                    </div>

                    {/* Recent projects */}
                    <Show when={projects().length > 0}>
                      <p class="text-xs text-base-content/70 font-semibold uppercase tracking-wider mb-2">Recent Projects</p>
                      <div class="max-h-60 overflow-y-auto -mx-1 mb-4 space-y-1">
                        <For each={projects()}>
                          {(project) => (
                            <div class="group flex items-center bg-base-100 rounded-lg border border-base-300 hover:border-primary hover:shadow-sm transition-all">
                              <button
                                class="flex-1 flex items-center gap-2.5 px-3 py-3 text-left min-w-0"
                                onClick={() => handleSelectProject(project)}
                              >
                                <svg class="w-5 h-5 text-primary flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                                </svg>
                                <div class="flex-1 min-w-0 flex items-baseline gap-1.5">
                                  <span class="text-sm font-semibold truncate">{project.name}</span>
                                  <span class="text-xs text-base-content/50 flex-shrink-0">({project.runs.length})</span>
                                </div>
                                <span class="text-xs text-base-content/60 flex-shrink-0 mr-1">
                                  {formatDate(project.lastModified)}
                                </span>
                                <svg class="w-4 h-4 text-base-content/30 group-hover:text-primary flex-shrink-0 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                                </svg>
                              </button>
                              {/* Rename/delete buttons — visible on hover */}
                              <div class="flex-shrink-0 flex gap-0.5 pr-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
                                <button
                                  class="btn btn-ghost btn-xs btn-square"
                                  onClick={(e) => handleStartRename(e, project)}
                                  title="Rename project"
                                >
                                  <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                                  </svg>
                                </button>
                                <button
                                  class="btn btn-ghost btn-xs btn-square text-error"
                                  onClick={(e) => handleStartDelete(e, project)}
                                  title="Delete project"
                                >
                                  <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                  </svg>
                                </button>
                              </div>
                            </div>
                          )}
                        </For>
                      </div>
                    </Show>

                    {/* Empty state icon */}
                    <Show when={projects().length === 0}>
                      <div class="flex items-center justify-center mb-4">
                        <svg class="w-12 h-12 text-base-content/20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                        </svg>
                      </div>
                    </Show>

                    {/* New project input */}
                    <div class="border-t border-base-300 pt-3 opacity-70 hover:opacity-100 transition-opacity">
                      <p class="text-[10px] text-base-content/40 font-semibold uppercase tracking-wider mb-1.5">New Project</p>
                      <div class="flex items-center gap-1.5">
                        <input
                          type="text"
                          class="input input-bordered input-xs flex-1 font-mono text-xs"
                          value={newProjectName()}
                          onInput={(e) => setNewProjectName(sanitizeJobName(e.currentTarget.value))}
                          onKeyDown={(e) => e.key === 'Enter' && handleNewProject()}
                          placeholder="project-name"
                        />
                        <button
                          class="btn btn-ghost btn-xs btn-square"
                          onClick={handleRerollName}
                          title="Random name"
                        >
                          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                          </svg>
                        </button>
                        <button
                          class="btn btn-ghost btn-xs"
                          onClick={handleNewProject}
                          disabled={!newProjectName().trim()}
                        >
                          Create
                        </button>
                      </div>
                    </div>
                  </Show>

                  {/* === Rename view === */}
                  <Show when={pickerView() === 'rename' && targetProject()}>
                    <p class="text-sm font-semibold mb-3">Rename Project</p>
                    <p class="text-xs text-base-content/60 mb-2">
                      Renaming <span class="font-mono font-semibold">{targetProject()!.name}</span> will update the project folder and all output files.
                    </p>
                    <input
                      type="text"
                      class="input input-bordered w-full font-mono text-sm mb-2"
                      value={renameTo()}
                      onInput={(e) => { setRenameTo(sanitizeJobName(e.currentTarget.value)); setRenameError(null); }}
                      onKeyDown={(e) => e.key === 'Enter' && handleConfirmRename()}
                      autofocus
                    />
                    <Show when={renameError()}>
                      <p class="text-xs text-error mb-2">{renameError()}</p>
                    </Show>
                    <div class="flex gap-2">
                      <button class="btn flex-1" onClick={resetPickerView} disabled={isProcessing()}>
                        Cancel
                      </button>
                      <button
                        class="btn btn-primary flex-1"
                        onClick={handleConfirmRename}
                        disabled={!renameTo().trim() || renameTo() === targetProject()!.name || isProcessing()}
                      >
                        {isProcessing() ? <span class="loading loading-spinner loading-xs" /> : 'Rename'}
                      </button>
                    </div>
                  </Show>

                  {/* === Delete view === */}
                  <Show when={pickerView() === 'delete' && targetProject()}>
                    <p class="text-sm font-semibold text-error mb-3">Delete Project</p>
                    <p class="text-xs text-base-content/60 mb-2">
                      This will permanently delete <span class="font-mono font-semibold">{targetProject()!.name}</span> and all its data.
                    </p>
                    <Show when={deleteFileCount()} fallback={
                      <div class="flex items-center gap-2 mb-3">
                        <span class="loading loading-spinner loading-xs" />
                        <span class="text-xs text-base-content/50">Counting files...</span>
                      </div>
                    }>
                      <div class="bg-error/10 rounded-lg px-3 py-2 mb-3">
                        <p class="text-sm font-semibold text-error">
                          {deleteFileCount()!.fileCount} files ({deleteFileCount()!.totalSizeMb} MB) will be removed
                        </p>
                      </div>
                    </Show>
                    <p class="text-xs text-base-content/60 mb-1">
                      Type <span class="font-mono font-bold">delete</span> to confirm:
                    </p>
                    <input
                      type="text"
                      class="input input-bordered w-full font-mono text-sm mb-3"
                      value={deleteConfirmText()}
                      onInput={(e) => setDeleteConfirmText(e.currentTarget.value.toLowerCase())}
                      onKeyDown={(e) => e.key === 'Enter' && deleteConfirmText() === 'delete' && handleConfirmDelete()}
                      placeholder="delete"
                      autofocus
                    />
                    <div class="flex gap-2">
                      <button class="btn flex-1" onClick={resetPickerView} disabled={isProcessing()}>
                        Cancel
                      </button>
                      <button
                        class="btn btn-error flex-1"
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
