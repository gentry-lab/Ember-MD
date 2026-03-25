// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, Show, createSignal, onMount } from 'solid-js';
import type { UpdateInfo } from '../utils/updateCheck';

interface AboutModalProps {
  isOpen: boolean;
  onClose: () => void;
  updateInfo?: UpdateInfo | null;
}

const AboutModal: Component<AboutModalProps> = (props) => {
  const [appVersion, setAppVersion] = createSignal('');
  const [copied, setCopied] = createSignal(false);
  const brewUpdateCommand = 'brew update && brew upgrade --cask ember-md';

  onMount(async () => {
    try { setAppVersion(await window.electronAPI.getAppVersion()); }
    catch { setAppVersion('unknown'); }
  });

  const copyBrewUpdateCommand = async () => {
    try {
      await navigator.clipboard.writeText(brewUpdateCommand);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy Homebrew update command:', err);
    }
  };

  return (
    <Show when={props.isOpen}>
      <div class="fixed inset-0 z-50 flex items-center justify-center">
        <div class="absolute inset-0 bg-black/50" onClick={() => props.onClose()} />
        <div class="relative bg-base-100 rounded-lg shadow-xl w-full max-w-lg mx-4 max-h-[80vh] flex flex-col">
          {/* Header */}
          <div class="flex items-center justify-between px-4 py-3 border-b border-base-300">
            <h3 class="text-lg font-bold">About Ember</h3>
            <button class="btn btn-ghost btn-sm btn-circle" onClick={() => props.onClose()}>
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Content */}
          <div class="flex-1 overflow-y-auto p-4 space-y-4">
            {/* App Info */}
            <div class="text-center mb-4">
              <h2 class="text-2xl font-bold">Ember</h2>
              <p class="text-sm text-base-content/70">Molecular Dynamics on Apple Silicon</p>
              <p class="text-xs text-base-content/50 mt-1">Version {appVersion()}</p>
            </div>

            <Show when={props.updateInfo}>
              <div class="bg-amber-500/10 border border-amber-500/30 rounded-lg px-3 py-3 space-y-2">
                <p class="text-sm font-semibold text-amber-400">Update available: {props.updateInfo!.version}</p>
                <div
                  class="relative group cursor-pointer rounded-md bg-base-300/60 hover:bg-base-300/80 transition-colors"
                  onClick={() => { void copyBrewUpdateCommand(); }}
                >
                  <pre class="px-3 py-2.5 pr-10 text-xs font-mono text-base-content/90 select-all overflow-x-auto">{brewUpdateCommand}</pre>
                  <span class="absolute right-2 top-1/2 -translate-y-1/2 text-[10px] text-base-content/50 group-hover:text-base-content/80 transition-colors">
                    {copied() ? 'Copied!' : 'Click to copy'}
                  </span>
                </div>
                <p class="text-[11px] text-base-content/50">To update Ember, paste this command into Terminal on your Mac.</p>
              </div>
            </Show>

            <p class="text-sm text-base-content/80">
              GPU-accelerated molecular dynamics simulations using AMBER force fields,
              optimized for Apple M-series chips via bundled OpenCL, Metal, and CPU OpenMM backends.
            </p>

            {/* Ember License */}
            <div class="bg-base-200 rounded-lg p-3">
              <h4 class="text-sm font-semibold mb-1">License</h4>
              <p class="text-xs text-base-content/70">
                MIT License. Copyright (c) 2026 Ember Contributors.
              </p>
              <p class="text-[10px] text-base-content/50 mt-1">
                Permission is hereby granted, free of charge, to any person obtaining a copy
                of this software and associated documentation files, to deal in the Software
                without restriction, including without limitation the rights to use, copy, modify,
                merge, publish, distribute, sublicense, and/or sell copies of the Software.
                THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
              </p>
            </div>

            {/* Dependency Licenses */}
            <div class="bg-base-200 rounded-lg p-3">
              <h4 class="text-sm font-semibold mb-2">Open Source Dependencies</h4>
              <div class="space-y-2 text-xs text-base-content/70 max-h-64 overflow-y-auto">
                {/* --- Simulation Engine --- */}
                <LicenseEntry
                  name="OpenMM"
                  license="MIT / LGPL"
                  url="https://openmm.org"
                  desc="Molecular simulation engine"
                />
                <LicenseEntry
                  name="OpenMM OpenCL Platform"
                  license="MIT / LGPL"
                  url="https://openmm.org"
                  desc="Bundled OpenCL backend used on macOS through the system OpenCL stack"
                />
                <LicenseEntry
                  name="Ember-Metal"
                  license="MIT / LGPL"
                  url="https://github.com/philipturner/openmm-metal"
                  desc="Bundled native Metal GPU backend for OpenMM, including Amoeba, Drude, and RPMD plugins"
                />
                {/* --- Force Fields --- */}
                <LicenseEntry
                  name="AMBER Force Fields (ff14SB, ff19SB)"
                  license="Public Domain (UCSF)"
                  desc="Protein force field parameters"
                />
                <LicenseEntry
                  name="OPC Water Model"
                  license="Academic (Izadi et al. 2014)"
                  desc="4-site water model for accurate solvation"
                />
                <LicenseEntry
                  name="OpenFF Sage 2.3.0"
                  license="MIT"
                  url="https://openforcefield.org"
                  desc="Small molecule force field (ligand parameterization)"
                />
                <LicenseEntry
                  name="OpenMM Force Fields"
                  license="MIT"
                  url="https://github.com/openmm/openmmforcefields"
                  desc="Force field XML definitions for OpenMM"
                />
                {/* --- Docking --- */}
                <LicenseEntry
                  name="AutoDock Vina"
                  license="Apache-2.0"
                  url="https://github.com/ccsb-scripps/AutoDock-Vina"
                  desc="Molecular docking engine"
                />
                <LicenseEntry
                  name="Meeko"
                  license="Apache-2.0"
                  url="https://github.com/forlilab/Meeko"
                  desc="PDBQT ligand/receptor preparation for Vina"
                />
                {/* --- Cheminformatics --- */}
                <LicenseEntry
                  name="RDKit"
                  license="BSD-3-Clause"
                  url="https://rdkit.org"
                  desc="Cheminformatics toolkit"
                />
                <LicenseEntry
                  name="Molscrub"
                  license="Apache-2.0"
                  url="https://github.com/forlilab/Molscrub"
                  desc="Ligand protonation state enumeration"
                />
                <LicenseEntry
                  name="Open Babel"
                  license="GPL-2.0"
                  url="https://openbabel.org"
                  desc="Chemical file format conversion"
                />
                {/* --- Structure Preparation --- */}
                <LicenseEntry
                  name="PDBFixer"
                  license="MIT"
                  url="https://github.com/openmm/pdbfixer"
                  desc="PDB structure preparation and repair"
                />
                <LicenseEntry
                  name="PROPKA"
                  license="BSD-3-Clause"
                  url="https://github.com/jensengroup/propka"
                  desc="Protein pKa prediction for protonation states"
                />
                <LicenseEntry
                  name="AmberTools"
                  license="LGPL"
                  url="https://ambermd.org/AmberTools.php"
                  desc="AM1-BCC charges and structure utilities, including sqm, reduce, and cpptraj"
                />
                {/* --- Semiempirical QM --- */}
                <LicenseEntry
                  name="GFN2-xTB"
                  license="LGPL-3.0"
                  url="https://github.com/grimme-lab/xtb"
                  desc="Semiempirical tight-binding method for conformer ranking and strain energy"
                />
                <LicenseEntry
                  name="CREST"
                  license="LGPL-3.0"
                  url="https://github.com/crest-lab/crest"
                  desc="Conformer-rotamer ensemble sampling via xTB metadynamics"
                />
                <LicenseEntry
                  name="CORDIAL"
                  license="Apache-2.0"
                  desc="Neural-network rescoring for docked poses and MD cluster centroids"
                />
                <LicenseEntry
                  name="PyTorch"
                  license="BSD-3-Clause"
                  url="https://pytorch.org"
                  desc="Tensor runtime used by bundled CORDIAL models"
                />
                {/* --- Analysis --- */}
                <LicenseEntry
                  name="MDAnalysis"
                  license="LGPL-2.1+ / LGPL-3+"
                  url="https://mdanalysis.org"
                  desc="Trajectory analysis toolkit"
                />
                <LicenseEntry
                  name="NumPy"
                  license="BSD-3-Clause"
                  url="https://numpy.org"
                  desc="Core numerical array library used across simulation, scoring, and analysis"
                />
                <LicenseEntry
                  name="Matplotlib"
                  license="PSF / BSD"
                  url="https://matplotlib.org"
                  desc="Scientific plotting for analysis reports"
                />
                <LicenseEntry
                  name="SciPy"
                  license="BSD-3-Clause"
                  url="https://scipy.org"
                  desc="Scientific computing (clustering, spatial algorithms)"
                />
                {/* --- Visualization & Desktop --- */}
                <LicenseEntry
                  name="NGL Viewer"
                  license="MIT"
                  url="https://nglviewer.org"
                  desc="WebGL molecular visualization"
                />
                <LicenseEntry
                  name="Electron"
                  license="MIT"
                  url="https://electronjs.org"
                  desc="Desktop application framework"
                />
                <LicenseEntry
                  name="SolidJS"
                  license="MIT"
                  url="https://solidjs.com"
                  desc="Reactive UI framework"
                />
                <LicenseEntry
                  name="Tailwind CSS / DaisyUI"
                  license="MIT"
                  url="https://daisyui.com"
                  desc="Utility-first CSS framework and component library"
                />
                <LicenseEntry
                  name="VkFFT"
                  license="MIT"
                  url="https://github.com/DTolm/VkFFT"
                  desc="GPU FFT library (used by openmm-metal)"
                />
              </div>
            </div>

            <p class="text-[10px] text-base-content/50 text-center">
              Built with open-source tools for computational chemistry research.
            </p>
          </div>
        </div>
      </div>
    </Show>
  );
};

const LicenseEntry: Component<{ name: string; license: string; url?: string; desc: string }> = (props) => (
  <div class="border-b border-base-300 pb-1.5">
    <div class="flex items-center justify-between">
      <span class="font-medium text-base-content/90">{props.name}</span>
      <span class="badge badge-ghost badge-xs">{props.license}</span>
    </div>
    <p class="text-[10px] text-base-content/60">{props.desc}</p>
  </div>
);

export default AboutModal;
