// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, JSX, Show } from 'solid-js';

interface HelpModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const HelpModal: Component<HelpModalProps> = (props) => {
  return (
    <Show when={props.isOpen}>
      <div class="fixed inset-0 z-50 flex items-center justify-center">
        {/* Backdrop */}
        <div
          class="absolute inset-0 bg-black/50"
          onClick={() => props.onClose()}
        />

        {/* Modal */}
        <div class="relative bg-base-100 rounded-lg shadow-xl w-full max-w-2xl mx-4 max-h-[80vh] flex flex-col">
          {/* Header */}
          <div class="flex items-center justify-between px-4 py-3 border-b border-base-300">
            <h3 class="text-lg font-bold">Ember Quick Reference</h3>
            <button
              class="btn btn-ghost btn-sm btn-circle"
              onClick={() => props.onClose()}
            >
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Content */}
          <div class="flex-1 overflow-y-auto p-4">
            <QuickReferenceContent />
          </div>
        </div>
      </div>
    </Show>
  );
};

const QuickReferenceContent: Component = () => (
  <div class="space-y-4">
    <HelpSection title="HOW THE APP IS ORGANIZED" icon="info">
      <p class="text-sm text-base-content/80 leading-relaxed">
        The tabs across the top are the main workflows. The project name in the center is your
        working folder, and the step tracker on the right shows where you are inside the current
        workflow. Most runs save into the current project automatically and can be reopened later in
        View.
      </p>
    </HelpSection>

    <HelpSection title="VIEW" icon="output">
      <p class="text-sm text-base-content/80 leading-relaxed">
        Use View to inspect structures and reopen saved results. This is the 3D workspace for
        imported PDB/CIF files, docking poses, MD trajectories, cluster centroids, overlays, and
        analysis panels.
      </p>
    </HelpSection>

    <HelpSection title="MCMM" icon="input">
      <p class="text-sm text-base-content/80 leading-relaxed">
        MCMM is the ligand conformer workflow. Start from an SDF/MOL file or paste SMILES, choose a
        search method such as ETKDG, MCMM, or CREST, then generate and rank low-energy conformers for
        a single molecule.
      </p>
    </HelpSection>

    <HelpSection title="DOCK" icon="auto">
      <p class="text-sm text-base-content/80 leading-relaxed">
        Dock prepares a receptor from a bound structure, uses the selected reference ligand to define
        the pocket, then docks one or more ligands into that site. This is the workflow to use when
        you want ranked poses and post-docking scoring.
      </p>
    </HelpSection>

    <HelpSection title="X-RAY" icon="warning">
      <p class="text-sm text-base-content/80 leading-relaxed">
        X-ray scans a folder of experimental structures and matching MTZ files, pairs the inputs, and
        runs the crystallographic validation analysis for each matched structure.
      </p>
    </HelpSection>

    <HelpSection title="SCORE" icon="warning">
      <p class="text-sm text-base-content/80 leading-relaxed">
        Score is for rescoring existing protein-ligand complexes. Import one or more PDB/CIF
        structures and Ember will detect ligands, prepare the complexes, and report Vina, CORDIAL,
        and QED metrics without running a full docking job.
      </p>
    </HelpSection>

    <HelpSection title="SIMULATE" icon="output">
      <p class="text-sm text-base-content/80 leading-relaxed">
        Simulate runs molecular dynamics. You can start from a holo complex, an apo protein, or a
        ligand-only system; Ember prepares the system, solvates it, equilibrates it, runs production
        MD, then saves trajectories, analyses, and clustered end-state summaries.
      </p>
    </HelpSection>

    <HelpSection title="PROJECTS AND JOBS" icon="info">
      <p class="text-sm text-base-content/80 leading-relaxed">
        The project control in the center lets you switch, move, or open the current project folder.
        Each workflow creates its own job outputs inside that project, so you can generate conformers,
        dock ligands, score complexes, and run simulations under the same project and review them later
        in View.
      </p>
    </HelpSection>
  </div>
);

interface HelpSectionProps {
  title: string;
  icon: 'input' | 'output' | 'auto' | 'warning' | 'info';
  children: JSX.Element;
}

const HelpSection: Component<HelpSectionProps> = (props) => {
  const iconColor = () => {
    switch (props.icon) {
      case 'input': return 'text-primary';
      case 'output': return 'text-secondary';
      case 'auto': return 'text-success';
      case 'warning': return 'text-warning';
      case 'info': return 'text-info';
    }
  };

  const getIcon = () => {
    switch (props.icon) {
      case 'input':
        return (
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
          </svg>
        );
      case 'output':
        return (
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
        );
      case 'auto':
        return (
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        );
      case 'warning':
        return (
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        );
      case 'info':
        return (
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
    }
  };

  return (
    <div class="bg-base-200 rounded-lg p-3">
      <div class={`flex items-center gap-2 mb-2 ${iconColor()}`}>
        {getIcon()}
        <span class="font-semibold text-sm">{props.title}</span>
      </div>
      {props.children}
    </div>
  );
};

export default HelpModal;
