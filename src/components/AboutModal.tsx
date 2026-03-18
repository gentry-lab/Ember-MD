import { Component, Show } from 'solid-js';

interface AboutModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const AboutModal: Component<AboutModalProps> = (props) => {
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
              <p class="text-xs text-base-content/50 mt-1">Version 1.0.0</p>
            </div>

            <p class="text-sm text-base-content/80">
              GPU-accelerated molecular dynamics simulations using AMBER force fields,
              optimized for Apple M-series chips via OpenCL/Metal.
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
                <LicenseEntry
                  name="OpenMM"
                  license="MIT / LGPL"
                  url="https://openmm.org"
                  desc="Molecular simulation engine"
                />
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
                  name="OpenFF Sage 2.0"
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
                <LicenseEntry
                  name="openmm-metal"
                  license="MIT / LGPL"
                  url="https://github.com/philipturner/openmm-metal"
                  desc="Metal GPU backend for OpenMM (Philip Turner)"
                />
                <LicenseEntry
                  name="RDKit"
                  license="BSD-3-Clause"
                  url="https://rdkit.org"
                  desc="Cheminformatics toolkit"
                />
                <LicenseEntry
                  name="Open Babel"
                  license="GPL-2.0"
                  url="https://openbabel.org"
                  desc="Chemical file format conversion and bond order perception"
                />
                <LicenseEntry
                  name="PDBFixer"
                  license="MIT"
                  url="https://github.com/openmm/pdbfixer"
                  desc="PDB structure preparation and repair"
                />
                <LicenseEntry
                  name="MDAnalysis"
                  license="GPL-2.0"
                  url="https://mdanalysis.org"
                  desc="Trajectory analysis toolkit"
                />
                <LicenseEntry
                  name="AmberTools"
                  license="LGPL"
                  url="https://ambermd.org/AmberTools.php"
                  desc="AM1-BCC partial charge computation (sqm)"
                />
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
