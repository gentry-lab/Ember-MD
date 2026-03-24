/**
 * Centralized path construction for the project directory layout.
 *
 * ~/Ember/{projectName}/
 *   .ember-project        — Project ID file
 *   structures/            — Imported structure files (.pdb/.cif/.sdf/.mol/.mol2) + prepared copies
 *   surfaces/              — Computed surface caches
 *   surfaces/binding_site_map/ — Interaction grids (OpenDX)
 *   docking/{run}/         — Self-contained docking jobs
 *     inputs/              — receptor.pdb, reference_ligand.sdf, ligands/
 *     prep/                — Intermediates (extraction, protonation, conformers)
 *     results/             — all_docked.sdf, cordial_scores.json, poses/
 *   simulations/{run}/     — Self-contained MD simulation jobs
 *     inputs/              — receptor.pdb, ligand.sdf
 *     results/             — system.pdb, trajectory.dcd, final.pdb, energy.csv
 *     analysis/            — clustering/, rmsd/, rmsf/, hbonds/, contacts/
 *   xray/{run}/            — X-ray pose validation jobs
 *     reports/             — xray_analysis_*.pdf outputs
 *   fep/                   — FEP scoring results
 */

import path from 'path';

export interface DockingPaths {
  root: string;
  inputs: string;
  inputsLigands: string;
  prep: string;
  results: string;
  resultsPoses: string;
}

export interface SimulationPaths {
  root: string;
  inputs: string;
  results: string;
  analysis: string;
  analysisClustering: string;
}

export function projectPaths(baseDir: string, projectName: string) {
  const root = path.join(baseDir, projectName);
  return {
    root,
    structures: path.join(root, 'structures'),
    surfaces: path.join(root, 'surfaces'),
    bindingSiteMap: path.join(root, 'surfaces', 'binding_site_map'),
    docking: (runFolder: string): DockingPaths => {
      const base = path.join(root, 'docking', runFolder);
      return {
        root: base,
        inputs: path.join(base, 'inputs'),
        inputsLigands: path.join(base, 'inputs', 'ligands'),
        prep: path.join(base, 'prep'),
        results: path.join(base, 'results'),
        resultsPoses: path.join(base, 'results', 'poses'),
      };
    },
    simulations: (runFolder: string): SimulationPaths => {
      const base = path.join(root, 'simulations', runFolder);
      return {
        root: base,
        inputs: path.join(base, 'inputs'),
        results: path.join(base, 'results'),
        analysis: path.join(base, 'analysis'),
        analysisClustering: path.join(base, 'analysis', 'clustering'),
      };
    },
    conformers: (runFolder: string) => path.join(root, 'conformers', runFolder),
    xray: (runFolder: string) => path.join(root, 'xray', runFolder),
    fep: path.join(root, 'fep'),

    // === Legacy aliases (for reading old projects) ===
    /** @deprecated Use structures instead */
    raw: path.join(root, 'raw'),
    /** @deprecated Receptors now live in each job's inputs/ */
    prepared: path.join(root, 'prepared'),
    /** @deprecated Ligands now live in each docking job's inputs/ligands/ and prep/ */
    ligands: {
      sdf: path.join(root, 'ligands', 'sdf'),
      smiles: path.join(root, 'ligands', 'smiles'),
      images: path.join(root, 'ligands', 'images'),
    },
  };
}
