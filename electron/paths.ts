/**
 * Path resolution and environment detection for Ember.
 * All functions are pure (fs.existsSync + env vars), no IPC or state mutation.
 */
import { app } from 'electron';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import * as yaml from 'yaml';
import type { SamplingConfig, RunParameters } from '../shared/types/ipc';

// Bundled installation path (set by .deb package)
const BUNDLED_INSTALL_PATH = '/opt/fraggen';

export function isBundledInstall(): boolean {
  return fs.existsSync(path.join(BUNDLED_INSTALL_PATH, 'scripts')) &&
         fs.existsSync(path.join(BUNDLED_INSTALL_PATH, 'python310'));
}

export function isDevRuntime(): boolean {
  return !app.isPackaged;
}

export function getFragGenRoot(): string {
  if (process.env.FRAGGEN_ROOT) {
    return process.env.FRAGGEN_ROOT;
  }

  if (isDevRuntime()) {
    const localScripts = path.join(__dirname, '..', '..', 'deps', 'staging', 'scripts');
    if (fs.existsSync(localScripts)) {
      return localScripts;
    }

    const devBundledScripts = path.join(path.resolve(__dirname, '..', '..'), 'bundle-mac', 'extra-resources', 'scripts');
    if (fs.existsSync(devBundledScripts)) {
      return devBundledScripts;
    }
  } else {
    const bundledScriptsPath = path.join(BUNDLED_INSTALL_PATH, 'scripts');
    if (fs.existsSync(bundledScriptsPath)) {
      return bundledScriptsPath;
    }

    const bundledScripts = path.join(process.resourcesPath, 'scripts');
    if (fs.existsSync(bundledScripts)) {
      return bundledScripts;
    }
  }

  const homeDir = process.env.HOME || process.env.USERPROFILE || '';
  if (process.platform === 'darwin') {
    return path.join(homeDir, 'fraggen_workspace', 'FragGen');
  } else {
    const candidates = [
      path.join(homeDir, 'FragGen'),
      path.join(homeDir, 'fraggen'),
      path.join(homeDir, 'fraggen_workspace', 'FragGen'),
      '/opt/FragGen',
    ];
    for (const candidate of candidates) {
      if (fs.existsSync(candidate)) {
        return candidate;
      }
    }
    return path.join(homeDir, 'FragGen');
  }
}

export function getCondaPythonPath(): string | null {
  if (process.env.FRAGGEN_PYTHON) {
    if (fs.existsSync(process.env.FRAGGEN_PYTHON)) {
      return process.env.FRAGGEN_PYTHON;
    }
  }

  if (!isDevRuntime()) {
    const bundledPython = path.join(process.resourcesPath, 'python', 'bin', 'python');
    if (fs.existsSync(bundledPython)) {
      return bundledPython;
    }

    const bundledPython310 = path.join(BUNDLED_INSTALL_PATH, 'python310', 'bin', 'python');
    if (fs.existsSync(bundledPython310)) {
      return bundledPython310;
    }
  }

  const homeDir = process.env.HOME || process.env.USERPROFILE || '';

  const envNames = process.platform === 'darwin'
    ? ['openmm-metal', 'fraggen']
    : ['fraggen', 'openmm-metal'];
  const condaDirs = [
    path.join(homeDir, 'miniconda3'),
    path.join(homeDir, 'anaconda3'),
    path.join(homeDir, 'miniforge3'),
    path.join(homeDir, 'mambaforge'),
  ];
  const candidates: string[] = [];
  for (const envName of envNames) {
    for (const condaDir of condaDirs) {
      candidates.push(path.join(condaDir, 'envs', envName, 'bin', 'python'));
    }
  }
  if (process.platform !== 'darwin') {
    candidates.push('/opt/conda/envs/fraggen/bin/python');
  }

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  if (isDevRuntime()) {
    const devBundledPython = path.join(path.resolve(__dirname, '..', '..'), 'bundle-mac', 'extra-resources', 'python', 'bin', 'python');
    if (fs.existsSync(devBundledPython)) {
      return devBundledPython;
    }
  } else {
    const bundledCondaPath = path.join(process.resourcesPath, 'conda/fraggen/bin/python');
    if (fs.existsSync(bundledCondaPath)) {
      return bundledCondaPath;
    }
  }

  return null;
}

export function getDevExtraResourcesPath(...parts: string[]): string {
  return path.join(path.resolve(__dirname, '..', '..'), 'bundle-mac', 'extra-resources', ...parts);
}

export function getQupkakePythonPath(): string | null {
  const envPython = process.env.QUPKAKE_PYTHON;
  if (envPython && fs.existsSync(envPython)) {
    return envPython;
  }

  const bundledPython = path.join(process.resourcesPath, 'qupkake-python', 'bin', 'python');
  if (fs.existsSync(bundledPython)) {
    return bundledPython;
  }

  const devBundledPython = getDevExtraResourcesPath('qupkake-python', 'bin', 'python');
  if (fs.existsSync(devBundledPython)) {
    return devBundledPython;
  }

  const homeDir = process.env.HOME || process.env.USERPROFILE || '';
  const condaDirs = [
    path.join(homeDir, 'miniconda3'),
    path.join(homeDir, 'anaconda3'),
    path.join(homeDir, 'miniforge3'),
    path.join(homeDir, 'mambaforge'),
  ];
  for (const condaDir of condaDirs) {
    const candidate = path.join(condaDir, 'envs', 'qupkake', 'bin', 'python');
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  return null;
}

export function getQupkakeRoot(): string | null {
  const envRoot = process.env.QUPKAKE_ROOT;
  if (envRoot && fs.existsSync(envRoot)) {
    return envRoot;
  }

  const bundledRoot = path.join(process.resourcesPath, 'qupkake-fork');
  if (fs.existsSync(path.join(bundledRoot, 'qupkake'))) {
    return bundledRoot;
  }

  const devBundledRoot = getDevExtraResourcesPath('qupkake-fork');
  if (fs.existsSync(path.join(devBundledRoot, 'qupkake'))) {
    return devBundledRoot;
  }

  const repoVendorRoot = path.join(path.resolve(__dirname, '..', '..'), 'vendor', 'QupKake');
  if (fs.existsSync(path.join(repoVendorRoot, 'qupkake'))) {
    return repoVendorRoot;
  }

  return null;
}

export function getQupkakeXtbPath(): string | null {
  const envXtb = process.env.QUPKAKE_XTBPATH || process.env.XTBPATH;
  if (envXtb && fs.existsSync(envXtb)) {
    return envXtb;
  }

  const bundledXtb = path.join(process.resourcesPath, 'qupkake-xtb', 'bin', 'xtb');
  if (fs.existsSync(bundledXtb)) {
    return bundledXtb;
  }

  const devBundledXtb = getDevExtraResourcesPath('qupkake-xtb', 'bin', 'xtb');
  if (fs.existsSync(devBundledXtb)) {
    return devBundledXtb;
  }

  const repoXtbCandidates = [
    path.join(path.resolve(__dirname, '..', '..'), 'vendor', 'xtb-env', 'bin', 'xtb'),
    path.join(path.resolve(__dirname, '..', '..'), 'vendor', 'xtb-6.4.1', 'install', 'bin', 'xtb'),
    path.join(path.resolve(__dirname, '..', '..'), 'vendor', 'xtb-6.4.1', 'install-openblas', 'bin', 'xtb'),
  ];
  for (const candidate of repoXtbCandidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  const qupkakePython = getQupkakePythonPath();
  if (qupkakePython) {
    const envXtbCandidate = path.join(path.dirname(qupkakePython), 'xtb');
    if (fs.existsSync(envXtbCandidate)) {
      return envXtbCandidate;
    }
  }

  return null;
}

export function getQupkakeValidationLigand(): string | null {
  const explicitLigand = process.env.QUPKAKE_VALIDATE_LIGAND;
  if (explicitLigand && fs.existsSync(explicitLigand)) {
    return explicitLigand;
  }

  const emberRoot = path.join(os.homedir(), 'Ember');
  if (!fs.existsSync(emberRoot)) {
    return null;
  }

  try {
    const projects = fs.readdirSync(emberRoot).sort();
    for (const project of projects) {
      const dockingRoot = path.join(emberRoot, project, 'docking');
      if (!fs.existsSync(dockingRoot)) continue;
      const jobs = fs.readdirSync(dockingRoot).sort();
      for (const job of jobs) {
        const prepDir = path.join(dockingRoot, job, 'prep');
        if (!fs.existsSync(prepDir)) continue;
        const files = fs.readdirSync(prepDir).sort();
        for (const file of files) {
          if (!/\.(sdf(\.gz)?|mol2?|MOL2?)$/i.test(file)) continue;
          return path.join(prepDir, file);
        }
      }
    }
  } catch (error) {
    console.warn('[QupKake] Failed to scan validation ligands:', (error as Error).message);
  }

  return null;
}

export function detectBabelDataDir(): string | null {
  const bundledBabelBase = path.join(BUNDLED_INSTALL_PATH, 'openbabel', 'share', 'openbabel');
  if (fs.existsSync(bundledBabelBase)) {
    try {
      const dirs = fs.readdirSync(bundledBabelBase);
      for (const dir of dirs) {
        const fullPath = path.join(bundledBabelBase, dir);
        if (fs.existsSync(path.join(fullPath, 'space-groups.txt'))) {
          return fullPath;
        }
      }
    } catch {
      // Ignore errors
    }
  }

  const candidates = [
    '/usr/share/openbabel/3.1.1',
    '/usr/share/openbabel/3.1.0',
    '/usr/share/openbabel/3.0.0',
    '/usr/local/share/openbabel/3.1.1',
    '/usr/local/share/openbabel/3.1.0',
    '/opt/homebrew/share/openbabel/3.1.1',
  ];

  for (const candidate of candidates) {
    if (fs.existsSync(path.join(candidate, 'space-groups.txt'))) {
      return candidate;
    }
  }

  const basePaths = ['/usr/share/openbabel', '/usr/local/share/openbabel', '/opt/homebrew/share/openbabel'];
  for (const basePath of basePaths) {
    if (fs.existsSync(basePath)) {
      try {
        const dirs = fs.readdirSync(basePath);
        for (const dir of dirs) {
          const fullPath = path.join(basePath, dir);
          if (fs.existsSync(path.join(fullPath, 'space-groups.txt'))) {
            return fullPath;
          }
        }
      } catch {
        // Ignore errors
      }
    }
  }

  return null;
}

export function getSurfaceGenPythonPath(): string | null {
  if (process.env.FRAGGEN_SURFACE_PYTHON) {
    if (fs.existsSync(process.env.FRAGGEN_SURFACE_PYTHON)) {
      return process.env.FRAGGEN_SURFACE_PYTHON;
    }
  }

  const bundledPython36 = path.join(BUNDLED_INSTALL_PATH, 'python36', 'bin', 'python');
  if (fs.existsSync(bundledPython36)) {
    return bundledPython36;
  }

  const homeDir = process.env.HOME || process.env.USERPROFILE || '';

  const candidates = [
    path.join(homeDir, 'miniconda3', 'envs', 'surface_gen', 'bin', 'python'),
    path.join(homeDir, 'anaconda3', 'envs', 'surface_gen', 'bin', 'python'),
    path.join(homeDir, 'miniforge3', 'envs', 'surface_gen', 'bin', 'python'),
    path.join(homeDir, 'mambaforge', 'envs', 'surface_gen', 'bin', 'python'),
  ];

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  return null;
}

export function getCordialRoot(): string | null {
  if (process.env.CORDIAL_ROOT && fs.existsSync(process.env.CORDIAL_ROOT)) {
    return process.env.CORDIAL_ROOT;
  }

  const appBundledCordial = path.join(process.resourcesPath, 'cordial');
  if (fs.existsSync(appBundledCordial) &&
      fs.existsSync(path.join(appBundledCordial, 'weights')) &&
      fs.existsSync(path.join(appBundledCordial, 'modules'))) {
    return appBundledCordial;
  }

  const devBundledCordial = path.join(path.resolve(__dirname, '..', '..'), 'bundle-mac', 'extra-resources', 'cordial');
  if (fs.existsSync(devBundledCordial) &&
      fs.existsSync(path.join(devBundledCordial, 'weights')) &&
      fs.existsSync(path.join(devBundledCordial, 'modules'))) {
    return devBundledCordial;
  }

  const bundledCordial = path.join(BUNDLED_INSTALL_PATH, 'cordial');
  if (fs.existsSync(bundledCordial) &&
      fs.existsSync(path.join(bundledCordial, 'weights')) &&
      fs.existsSync(path.join(bundledCordial, 'modules'))) {
    return bundledCordial;
  }

  const projectRoot = path.dirname(__dirname);
  const projectCordial = path.join(projectRoot, 'CORDIAL');
  if (fs.existsSync(projectCordial) &&
      fs.existsSync(path.join(projectCordial, 'weights')) &&
      fs.existsSync(path.join(projectCordial, 'modules'))) {
    return projectCordial;
  }

  const homeDir = process.env.HOME || process.env.USERPROFILE || '';

  const candidates = [
    path.join(homeDir, 'Desktop', 'FragGen', 'CORDIAL'),
    path.join(homeDir, 'Desktop', 'CORDIAL'),
    path.join(homeDir, 'CORDIAL'),
    path.join(homeDir, 'cordial'),
    path.join(homeDir, 'projects', 'CORDIAL'),
    '/opt/CORDIAL',
  ];

  for (const candidate of candidates) {
    if (fs.existsSync(candidate) &&
        fs.existsSync(path.join(candidate, 'weights')) &&
        fs.existsSync(path.join(candidate, 'modules'))) {
      return candidate;
    }
  }

  return null;
}

export function getFragGenScript(fraggenRoot: string): string {
  return path.join(fraggenRoot, 'gen_from_pdb.py');
}

export function getFragBase(fraggenRoot: string): string {
  const bundledFragBase = path.join(BUNDLED_INSTALL_PATH, 'models', 'data', 'fragment_base.pkl');
  if (fs.existsSync(bundledFragBase)) {
    return bundledFragBase;
  }
  return path.join(fraggenRoot, 'data', 'fragment_base.pkl');
}

export function getModelCheckpointDir(fraggenRoot: string): string {
  const bundledCkpt = path.join(BUNDLED_INSTALL_PATH, 'models', 'ckpt');
  if (fs.existsSync(bundledCkpt)) {
    return bundledCkpt;
  }
  return path.join(fraggenRoot, 'ckpt');
}

export function getBaseConfigs(fraggenRoot: string): Record<string, string> {
  return {
    dihedral: path.join(fraggenRoot, 'configs', 'sample_dihedral.yml'),
    cartesian: path.join(fraggenRoot, 'configs', 'sample_cartesian.yml'),
    geomopt: path.join(fraggenRoot, 'configs', 'sample_geomopt.yml'),
  };
}

export function generateRuntimeConfig(
  baseConfigPath: string,
  sampling: SamplingConfig,
  outputDir: string
): string {
  const baseConfig = yaml.parse(fs.readFileSync(baseConfigPath, 'utf-8'));

  baseConfig.sample = {
    seed: sampling.seed,
    mask_init: true,
    num_samples: sampling.numSamples,
    beam_size: sampling.beamSize,
    max_steps: sampling.maxSteps,
    threshold: {
      focal_threshold: sampling.threshold.focalThreshold,
      pos_threshold: sampling.threshold.posThreshold,
      element_threshold: sampling.threshold.elementThreshold,
    },
    initial_num_steps: 1,
    next_threshold: {
      focal_threshold: sampling.nextThreshold.focalThreshold,
      pos_threshold: sampling.nextThreshold.posThreshold,
      element_threshold: sampling.nextThreshold.elementThreshold,
    },
    queue_same_smi_tolorance: sampling.queueSameSmiTolerance,
  };

  const runtimeConfigPath = path.join(outputDir, 'runtime_config.yml');
  fs.writeFileSync(runtimeConfigPath, yaml.stringify(baseConfig), 'utf-8');

  return runtimeConfigPath;
}

export function saveRunParameters(params: RunParameters, outputDir: string): string {
  const paramsPath = path.join(outputDir, 'run_parameters.json');
  fs.writeFileSync(paramsPath, JSON.stringify(params, null, 2), 'utf-8');
  return paramsPath;
}

/** Resolved paths object returned by initializePaths() */
export interface ResolvedPaths {
  fraggenRoot: string;
  condaPythonPath: string | null;
  condaEnvBin: string | null;
  surfaceGenPythonPath: string | null;
}

export function initializePaths(): ResolvedPaths {
  const fraggenRoot = getFragGenRoot();
  const condaPythonPath = getCondaPythonPath();
  const condaEnvBin = condaPythonPath ? path.dirname(condaPythonPath) : null;

  // Prepend conda env bin to PATH so all child processes find sqm, obabel, etc.
  if (condaEnvBin) {
    process.env.PATH = `${condaEnvBin}:${process.env.PATH || ''}`;
  }

  const surfaceGenPythonPath = getSurfaceGenPythonPath();

  try {
    console.log('=== FragGen Path Configuration ===');
    console.log('Bundled installation:', isBundledInstall() ? 'Yes' : 'No');
    console.log('FragGen root:', fraggenRoot);
    console.log('Model checkpoint dir:', getModelCheckpointDir(fraggenRoot));
    console.log('Fragment base:', getFragBase(fraggenRoot));
    console.log('Conda Python (fraggen):', condaPythonPath);
    console.log('QupKake Python:', getQupkakePythonPath() || 'Not found');
    console.log('QupKake root:', getQupkakeRoot() || 'Not found');
    console.log('QupKake xTB:', getQupkakeXtbPath() || 'Not found');
    console.log('Conda Python (surface_gen):', surfaceGenPythonPath);
    console.log('Docking backend: Vina (Python API)');
    console.log('CORDIAL root:', getCordialRoot() || 'Not found');
    console.log('OpenBabel data:', detectBabelDataDir() || 'Not found');
    console.log('==================================');
  } catch {
    // Ignore EPIPE — stdout may be closed when launched from npm start
  }

  return { fraggenRoot, condaPythonPath, condaEnvBin, surfaceGenPythonPath };
}
