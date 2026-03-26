// Copyright (c) 2026 Ember Contributors. MIT License.
/**
 * Centralized path construction for the project directory layout.
 *
 * {project}/
 *   .ember-project         — Project ID file
 *   structures/            — Imported structure and support files
 *   surfaces/              — Computed surface caches
 *   docking/{run}/         — inputs/, prep/, results/
 *   simulations/{run}/     — inputs/, results/, analysis/
 *   conformers/{run}/      — inputs/, results/
 *   scoring/{run}/         — inputs/, results/
 *   xray/{run}/            — inputs/, results/
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

export interface ConformerPaths {
  root: string;
  inputs: string;
  results: string;
}

export interface ScoringPaths {
  root: string;
  inputs: string;
  results: string;
}

export interface XrayPaths {
  root: string;
  inputs: string;
  results: string;
}

export interface ProjectPaths {
  root: string;
  structures: string;
  surfaces: string;
  bindingSiteMap: string;
  docking: (runFolder: string) => DockingPaths;
  simulations: (runFolder: string) => SimulationPaths;
  conformers: (runFolder: string) => ConformerPaths;
  scoring: (runFolder: string) => ScoringPaths;
  xray: (runFolder: string) => XrayPaths;
  fep: string;
}

export function projectPaths(baseDir: string, projectName: string): ProjectPaths {
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
    conformers: (runFolder: string): ConformerPaths => {
      const base = path.join(root, 'conformers', runFolder);
      return {
        root: base,
        inputs: path.join(base, 'inputs'),
        results: path.join(base, 'results'),
      };
    },
    scoring: (runFolder: string): ScoringPaths => {
      const base = path.join(root, 'scoring', runFolder);
      return {
        root: base,
        inputs: path.join(base, 'inputs'),
        results: path.join(base, 'results'),
      };
    },
    xray: (runFolder: string): XrayPaths => {
      const base = path.join(root, 'xray', runFolder);
      return {
        root: base,
        inputs: path.join(base, 'inputs'),
        results: path.join(base, 'results'),
      };
    },
    fep: path.join(root, 'fep'),
  };
}

export function projectPathsFromProjectDir(projectDir: string): ProjectPaths {
  return projectPaths(path.dirname(projectDir), path.basename(projectDir));
}
