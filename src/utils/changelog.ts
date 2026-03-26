// Copyright (c) 2026 Ember Contributors. MIT License.
/**
 * Rolling changelog for the "What's New" section.
 * Add newest versions at the top. Each entry is shown in the About modal
 * and in the first-launch popup after an update.
 */

export interface ChangelogEntry {
  version: string;
  date: string;
  highlights: string[];
}

export const changelog: ChangelogEntry[] = [
  {
    version: '0.3.13',
    date: '2026-03-26',
    highlights: [
      'Score tab: batch-score protein-ligand PDBs with Vina, CORDIAL, and QED',
      'Molecule detail panels with 2D thumbnails, RMSD, and centroid coordinates across all results',
      'MD cluster detail now shows ligand thumbnail and per-cluster dihedral angles',
      'Conformer results detail panel with RMSD to lowest-energy conformer',
      'Renamed MCMM tab to Conformers',
      'Single working directory project browser with metadata-backed project imports',
      'What\'s New changelog in About modal',
    ],
  },
  {
    version: '0.3.12',
    date: '2026-03-20',
    highlights: [
      'Configurable working directory for project storage',
      'Quit guard prevents closing during active jobs',
      'Job safety improvements for simulation and docking',
    ],
  },
  {
    version: '0.3.11',
    date: '2026-03-15',
    highlights: [
      'Drag-and-drop file imports across all modes',
      'Tab switching allowed during running jobs',
      'Project browser and folder management improvements',
    ],
  },
  {
    version: '0.3.10',
    date: '2026-03-10',
    highlights: [
      'Fixed OpenMM CPU platform crash on bundled app',
    ],
  },
  {
    version: '0.3.9',
    date: '2026-03-05',
    highlights: [
      'Auto-optimize solvation box: 46% faster MD throughput',
    ],
  },
];

/**
 * Returns changelog entries that are newer than the given version.
 * If lastSeenVersion is null, returns all entries.
 */
export function getNewEntries(lastSeenVersion: string | null): ChangelogEntry[] {
  if (!lastSeenVersion) return changelog;
  const lastParts = lastSeenVersion.replace(/^v/, '').split('.').map(Number);
  return changelog.filter((entry) => {
    const parts = entry.version.replace(/^v/, '').split('.').map(Number);
    for (let i = 0; i < 3; i++) {
      const a = parts[i] || 0;
      const b = lastParts[i] || 0;
      if (a > b) return true;
      if (a < b) return false;
    }
    return false;
  });
}
