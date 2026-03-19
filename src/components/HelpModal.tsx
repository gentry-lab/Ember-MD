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
            <MDHelp />
          </div>
        </div>
      </div>
    </Show>
  );
};

const MDHelp: Component = () => (
  <div class="space-y-4">
    <HelpSection title="INPUT MODES" icon="input">
      <ul class="list-disc list-inside text-sm text-base-content/80 space-y-1">
        <li><strong>Protein + Ligand:</strong> X-ray PDB complex or docking output</li>
        <li><strong>Ligand Only:</strong> SMILES paste, MOL/SDF file (small molecule in solvent)</li>
      </ul>
      <p class="text-xs text-base-content/60 mt-2">
        X-ray PDB: auto-detects ligands, extracts to SDF, prepares receptor with PDBFixer.
        Ligand-only: optional "Protonate pH 7.4" button for physiological protonation state.
      </p>
    </HelpSection>

    <HelpSection title="OUTPUTS" icon="output">
      <ul class="list-disc list-inside text-sm text-base-content/80 space-y-1">
        <li><code class="bg-base-300 px-1 rounded">{'{job}'}_system.pdb</code> - Solvated system</li>
        <li><code class="bg-base-300 px-1 rounded">{'{job}'}_trajectory.dcd</code> - Production trajectory</li>
        <li><code class="bg-base-300 px-1 rounded">{'{job}'}_energy.csv</code> - Energy timeseries</li>
        <li><code class="bg-base-300 px-1 rounded">{'{job}'}_checkpoint.chk</code> - Crash recovery</li>
        <li><code class="bg-base-300 px-1 rounded">clusters/</code> - Top 5 conformer cluster centroids (auto)</li>
      </ul>
    </HelpSection>

    <HelpSection title="AUTO PROCESSING" icon="auto">
      <div class="flex flex-wrap gap-2">
        <span class="badge badge-success gap-1">
          <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
          </svg>
          PDBFixer + hydrogens
        </span>
        <span class="badge badge-success gap-1">
          <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
          </svg>
          Solvation + ions
        </span>
        <span class="badge badge-success gap-1">
          <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
          </svg>
          Restrained equilibration
        </span>
        <span class="badge badge-success gap-1">
          <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
          </svg>
          Conformer clustering
        </span>
      </div>
    </HelpSection>

    <HelpSection title="FORCE FIELD PRESETS" icon="info">
      <div class="text-sm text-base-content/80 space-y-1">
        <p><strong>Fast:</strong> ff14SB + TIP3P — well-tested, faster</p>
        <p><strong>Accurate (default):</strong> ff19SB + OPC (4-site water) — higher accuracy</p>
        <p class="text-xs text-base-content/60 mt-2">
          OpenFF Sage 2.3.0 for ligand parameterization. 4fs HMR timestep.
          ~270ps equilibration (protein+ligand) or ~170ps (ligand-only).
          Checkpoint saved every 0.5 ns.
        </p>
      </div>
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
