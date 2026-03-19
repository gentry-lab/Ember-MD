import { app, BrowserWindow, ipcMain, dialog } from 'electron';
import * as path from 'path';
import { spawn, ChildProcess } from 'child_process';
import * as fs from 'fs';

// Prevent EPIPE crashes when stdout/stderr pipe is closed (e.g. npm start)
process.stdout?.on('error', () => {});
process.stderr?.on('error', () => {});

// --- Session log file (~/Ember/logs/ember-<timestamp>.log) ---
const logDir = path.join(require('os').homedir(), 'Ember', 'logs');
fs.mkdirSync(logDir, { recursive: true });
const sessionTs = new Date().toISOString().replace(/[:.]/g, '-').replace('T', '_').slice(0, 19);
const logPath = path.join(logDir, `ember-${sessionTs}.log`);
const logStream = fs.createWriteStream(logPath, { flags: 'a' });

const formatLog = (level: string, args: any[]): string => {
  const ts = new Date().toISOString();
  const msg = args.map(a => typeof a === 'string' ? a : JSON.stringify(a, null, 0)).join(' ');
  return `${ts} [${level}] ${msg}\n`;
};

const origLog = console.log.bind(console);
const origWarn = console.warn.bind(console);
const origError = console.error.bind(console);
console.log = (...args: any[]) => { origLog(...args); logStream.write(formatLog('LOG', args)); };
console.warn = (...args: any[]) => { origWarn(...args); logStream.write(formatLog('WARN', args)); };
console.error = (...args: any[]) => { origError(...args); logStream.write(formatLog('ERROR', args)); };
import * as yaml from 'yaml';
import { Ok, Err, Result } from '../shared/types/result';
import { AppError } from '../shared/types/errors';
import {
  PrepPdbOptions,
  PrepPdbResult,
  SurfaceResult,
  GenerationOptions,
  GenerationResult,
  RunParameters,
  SamplingConfig,
  IpcChannels,
  GenerationStats,
} from '../shared/types/ipc';
import * as os from 'os';
import * as zlib from 'zlib';

let mainWindow: BrowserWindow | null = null;

// Track spawned processes for cleanup
const childProcesses = new Set<ChildProcess>();

// Filter noisy stderr from Metal backend and Python deprecation warnings
function filterMdStderr(text: string): string {
  const lines = text.split('\n').filter(line => {
    if (line.startsWith('[Metal Transform STEP')) return false;
    if (line.startsWith('[Metal Debug]')) return false;
    if (line.startsWith('[Metal]') && (
      line.includes('wrapped ') || line.includes('mutable param') ||
      line.includes('Functions needing') || line.includes('Added _mm_ts') ||
      line.includes('Applied ') || line.includes('Renamed ') ||
      line.includes('Rewrote ') || line.includes('Compiled kernel library')
    )) return false;
    if (line.includes('pkg_resources is deprecated')) return false;
    if (line.includes('FutureWarning')) return false;
    if (line.match(/^\s+from pkg_resources/)) return false;
    if (line.match(/^\s+if isinstance\(obj, functools\._lru_cache_wrapper\)/)) return false;
    return true;
  });
  return lines.join('\n');
}

function killAllChildProcesses() {
  for (const proc of childProcesses) {
    if (!proc.killed) {
      proc.kill('SIGTERM');
    }
  }
  childProcesses.clear();
}

// Bundled installation path (set by .deb package)
const BUNDLED_INSTALL_PATH = '/opt/fraggen';

// Check if running from bundled installation
function isBundledInstall(): boolean {
  return fs.existsSync(path.join(BUNDLED_INSTALL_PATH, 'scripts')) &&
         fs.existsSync(path.join(BUNDLED_INSTALL_PATH, 'python310'));
}

// FragGen paths - configurable via environment or auto-detected
function getFragGenRoot(): string {
  // Check environment variable first
  if (process.env.FRAGGEN_ROOT) {
    return process.env.FRAGGEN_ROOT;
  }

  // Check for bundled installation
  const bundledScriptsPath = path.join(BUNDLED_INSTALL_PATH, 'scripts');
  if (fs.existsSync(bundledScriptsPath)) {
    return bundledScriptsPath;
  }

  // Check bundled scripts inside .app (electron-builder extraResources)
  const bundledScripts = path.join(process.resourcesPath, 'scripts');
  if (fs.existsSync(bundledScripts)) {
    return bundledScripts;
  }

  // Check local deps/staging/scripts/ relative to the app (dev mode)
  // __dirname is electron-dist/electron/, so ../.. reaches project root
  const localScripts = path.join(__dirname, '..', '..', 'deps', 'staging', 'scripts');
  if (fs.existsSync(localScripts)) {
    return localScripts;
  }

  // Default paths by platform
  const homeDir = process.env.HOME || process.env.USERPROFILE || '';
  if (process.platform === 'darwin') {
    return path.join(homeDir, 'fraggen_workspace', 'FragGen');
  } else {
    // Linux - check common locations
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

function getCondaPythonPath(): string | null {
  // Check bundled Python inside .app (electron-builder extraResources)
  const bundledPython = path.join(process.resourcesPath, 'python', 'bin', 'python');
  if (fs.existsSync(bundledPython)) {
    return bundledPython;
  }

  // Check environment variable
  if (process.env.FRAGGEN_PYTHON) {
    if (fs.existsSync(process.env.FRAGGEN_PYTHON)) {
      return process.env.FRAGGEN_PYTHON;
    }
  }

  // Check for bundled installation (.deb package)
  const bundledPython310 = path.join(BUNDLED_INSTALL_PATH, 'python310', 'bin', 'python');
  if (fs.existsSync(bundledPython310)) {
    return bundledPython310;
  }

  const homeDir = process.env.HOME || process.env.USERPROFILE || '';

  // Platform-specific conda paths — try openmm-metal first (Mac), then fraggen (Linux)
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

  // Check bundled conda (for Electron packaged app - legacy)
  const bundledCondaPath = path.join(process.resourcesPath, 'conda/fraggen/bin/python');
  if (fs.existsSync(bundledCondaPath)) {
    return bundledCondaPath;
  }

  return null;
}

function detectBabelDataDir(): string | null {
  // Check for bundled OpenBabel first (.deb package)
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

  // Try to find Open Babel data directory
  const candidates = [
    '/usr/share/openbabel/3.1.1',
    '/usr/share/openbabel/3.1.0',
    '/usr/share/openbabel/3.0.0',
    '/usr/local/share/openbabel/3.1.1',
    '/usr/local/share/openbabel/3.1.0',
    '/opt/homebrew/share/openbabel/3.1.1',  // macOS Homebrew
  ];

  for (const candidate of candidates) {
    if (fs.existsSync(path.join(candidate, 'space-groups.txt'))) {
      return candidate;
    }
  }

  // Try to find any openbabel directory with space-groups.txt
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

function getSurfaceGenPythonPath(): string | null {
  // Check environment variable first (set by launcher script)
  if (process.env.FRAGGEN_SURFACE_PYTHON) {
    if (fs.existsSync(process.env.FRAGGEN_SURFACE_PYTHON)) {
      return process.env.FRAGGEN_SURFACE_PYTHON;
    }
  }

  // Check for bundled installation (.deb package)
  const bundledPython36 = path.join(BUNDLED_INSTALL_PATH, 'python36', 'bin', 'python');
  if (fs.existsSync(bundledPython36)) {
    return bundledPython36;
  }

  // Surface generation requires Python 3.6 with pymesh2 (surface_gen env)
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

// Global state
let condaPythonPath: string | null = null;
let condaEnvBin: string | null = null; // bin/ dir of the conda env, for PATH
let surfaceGenPythonPath: string | null = null;
let fraggenRoot: string = '';

// Build spawn env with conda bin on PATH so child processes find sqm, obabel, etc.
function getSpawnEnv(): NodeJS.ProcessEnv {
  if (!condaEnvBin) return { ...process.env };
  const currentPath = process.env.PATH || '';
  return { ...process.env, PATH: `${condaEnvBin}:${currentPath}` };
}

function initializePaths(): void {
  fraggenRoot = getFragGenRoot();
  condaPythonPath = getCondaPythonPath();
  condaEnvBin = condaPythonPath ? path.dirname(condaPythonPath) : null;
  // Prepend conda env bin to PATH so all child processes find sqm, obabel, etc.
  if (condaEnvBin) {
    process.env.PATH = `${condaEnvBin}:${process.env.PATH || ''}`;
  }
  surfaceGenPythonPath = getSurfaceGenPythonPath();

  try {
    console.log('=== FragGen Path Configuration ===');
    console.log('Bundled installation:', isBundledInstall() ? 'Yes' : 'No');
    console.log('FragGen root:', fraggenRoot);
    console.log('Model checkpoint dir:', getModelCheckpointDir());
    console.log('Fragment base:', getFragBase());
    console.log('Conda Python (fraggen):', condaPythonPath);
    console.log('Conda Python (surface_gen):', surfaceGenPythonPath);
    console.log('Docking backend: Vina (Python API)');
    console.log('CORDIAL root:', getCordialRoot() || 'Not found');
    console.log('OpenBabel data:', detectBabelDataDir() || 'Not found');
    console.log('==================================');
  } catch {
    // Ignore EPIPE — stdout may be closed when launched from npm start
  }
}

function getFragGenScript(): string {
  return path.join(fraggenRoot, 'gen_from_pdb.py');
}

function getFragBase(): string {
  // Check bundled models directory first
  const bundledFragBase = path.join(BUNDLED_INSTALL_PATH, 'models', 'data', 'fragment_base.pkl');
  if (fs.existsSync(bundledFragBase)) {
    return bundledFragBase;
  }
  // Fall back to repo structure
  return path.join(fraggenRoot, 'data', 'fragment_base.pkl');
}

function getModelCheckpointDir(): string {
  // Check bundled models directory first
  const bundledCkpt = path.join(BUNDLED_INSTALL_PATH, 'models', 'ckpt');
  if (fs.existsSync(bundledCkpt)) {
    return bundledCkpt;
  }
  // Fall back to repo structure
  return path.join(fraggenRoot, 'ckpt');
}

function getBaseConfigs(): Record<string, string> {
  return {
    dihedral: path.join(fraggenRoot, 'configs', 'sample_dihedral.yml'),
    cartesian: path.join(fraggenRoot, 'configs', 'sample_cartesian.yml'),
    geomopt: path.join(fraggenRoot, 'configs', 'sample_geomopt.yml'),
  };
}

/**
 * Generate a runtime config file with custom sampling parameters
 */
function generateRuntimeConfig(
  baseConfigPath: string,
  sampling: SamplingConfig,
  outputDir: string
): string {
  // Read base config
  const baseConfig = yaml.parse(fs.readFileSync(baseConfigPath, 'utf-8'));

  // Override sampling parameters
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

  // Write runtime config
  const runtimeConfigPath = path.join(outputDir, 'runtime_config.yml');
  fs.writeFileSync(runtimeConfigPath, yaml.stringify(baseConfig), 'utf-8');

  return runtimeConfigPath;
}

/**
 * Save run parameters to JSON log file
 */
function saveRunParameters(params: RunParameters, outputDir: string): string {
  const paramsPath = path.join(outputDir, 'run_parameters.json');
  fs.writeFileSync(paramsPath, JSON.stringify(params, null, 2), 'utf-8');
  return paramsPath;
}

function createWindow(): void {
  // Initialize paths on startup
  initializePaths();

  mainWindow = new BrowserWindow({
    width: 1125,
    height: 950,
    minWidth: 600,
    minHeight: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
    // Use hiddenInset on macOS, default on Linux
    ...(process.platform === 'darwin' ? { titleBarStyle: 'hiddenInset' } : {}),
    backgroundColor: '#1e1e2e',
  });

  mainWindow.loadFile(path.join(__dirname, '../../dist-webpack/index.html'));

  if (process.env.NODE_ENV === 'development') {
    mainWindow.webContents.openDevTools();
  }

  // Capture renderer console.log/warn/error into the session log file
  mainWindow.webContents.on('console-message', (_e, level, message, line, sourceId) => {
    const tag = ['LOG', 'WARN', 'ERROR'][level] || 'LOG';
    const src = sourceId ? `${sourceId}:${line}` : '';
    logStream.write(formatLog(`RENDERER:${tag}`, [message, src ? `(${src})` : '']));
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.on('ready', createWindow);
app.on('window-all-closed', () => {
  killAllChildProcesses();
  if (process.platform !== 'darwin') app.quit();
});
app.on('before-quit', () => {
  killAllChildProcesses();
});
app.on('activate', () => {
  if (mainWindow === null) createWindow();
});

// === IPC Handlers ===

// Select multiple PDB files
ipcMain.handle(IpcChannels.SELECT_PDB_FILES_MULTI, async (): Promise<string[]> => {
  if (!mainWindow) return [];
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile', 'multiSelections'],
    filters: [
      { name: 'Structure Files', extensions: ['pdb', 'cif'] },
      { name: 'All Files', extensions: ['*'] },
    ],
  });
  return result.filePaths;
});

// Select PDB file
ipcMain.handle(IpcChannels.SELECT_PDB_FILE, async (_event, defaultPath?: string): Promise<string | null> => {
  if (!mainWindow) return null;
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    defaultPath: defaultPath || undefined,
    filters: [
      { name: 'Structure Files', extensions: ['pdb', 'cif'] },
      { name: 'All Files', extensions: ['*'] },
    ],
  });
  return result.filePaths[0] || null;
});

// Prep PDB (extract pocket + ligand from complex)
ipcMain.handle(
  IpcChannels.PREP_PDB,
  async (
    event,
    pdbPath: string,
    outputDir: string,
    options?: PrepPdbOptions
  ): Promise<Result<PrepPdbResult, AppError>> => {
    return new Promise((resolve) => {
      if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
        resolve(Err({
          type: 'PYTHON_NOT_FOUND',
          message: 'Python not found. Please install miniconda and create fraggen environment.',
        }));
        return;
      }

      const scriptPath = path.join(fraggenRoot, 'prep_pdb_gui.py');
      if (!fs.existsSync(scriptPath)) {
        resolve(Err({
          type: 'SCRIPT_NOT_FOUND',
          path: scriptPath,
          message: `Prep script not found: ${scriptPath}`,
        }));
        return;
      }

      const args = [scriptPath, '--input_pdb', pdbPath, '--output_dir', outputDir];
      if (options?.ligandName) args.push('--ligand_name', options.ligandName);
      if (options?.pocketRadius) args.push('--pocket_radius', String(options.pocketRadius));

      const python = spawn(condaPythonPath, args);
      childProcesses.add(python);

      python.stdout.on('data', (data: Buffer) => {
        event.sender.send(IpcChannels.PREP_OUTPUT, { type: 'stdout', data: data.toString() });
      });

      python.stderr.on('data', (data: Buffer) => {
        event.sender.send(IpcChannels.PREP_OUTPUT, { type: 'stderr', data: data.toString() });
      });

      python.on('close', (code: number | null) => {
        childProcesses.delete(python);
        if (code === 0) {
          resolve(Ok({
            pocketPdb: path.join(outputDir, 'pocket.pdb'),
            ligandPdb: path.join(outputDir, 'ligand.pdb'),
            outputDir,
          }));
        } else {
          resolve(Err({
            type: 'PREP_FAILED',
            message: `Process exited with code ${code}`,
          }));
        }
      });

      python.on('error', (error: Error) => {
        childProcesses.delete(python);
        resolve(Err({
          type: 'PREP_FAILED',
          message: error.message,
        }));
      });
    });
  }
);

// Generate surface PLY file
ipcMain.handle(
  IpcChannels.GENERATE_SURFACE,
  async (
    event,
    pocketPdb: string,
    ligandPdb: string,
    outputPly: string
  ): Promise<Result<SurfaceResult, AppError>> => {
    return new Promise((resolve) => {
      // Surface generation requires Python 3.6 with pymesh2 (surface_gen env)
      if (!surfaceGenPythonPath || !fs.existsSync(surfaceGenPythonPath)) {
        resolve(Err({
          type: 'PYTHON_NOT_FOUND',
          message: 'Surface generation requires surface_gen conda environment (Python 3.6 with pymesh2). Run: conda create -n surface_gen python=3.6 && conda activate surface_gen && conda install -c conda-forge pymesh2',
        }));
        return;
      }

      const scriptPath = path.join(fraggenRoot, 'generate_pocket_surface.py');
      if (!fs.existsSync(scriptPath)) {
        resolve(Err({
          type: 'SCRIPT_NOT_FOUND',
          path: scriptPath,
          message: `Surface script not found: ${scriptPath}`,
        }));
        return;
      }

      if (!ligandPdb || !fs.existsSync(ligandPdb)) {
        resolve(Err({
          type: 'FILE_NOT_FOUND',
          path: ligandPdb || 'undefined',
          message: `Ligand file required for surface generation but not found: ${ligandPdb}`,
        }));
        return;
      }

      const args = [
        scriptPath,
        '--pdb_file', pocketPdb,
        '--ligand_file', ligandPdb,
        '--output', outputPly,
      ];

      console.log('Surface generation args:', args);
      console.log('Using Python:', surfaceGenPythonPath);

      const python = spawn(surfaceGenPythonPath, args);
      childProcesses.add(python);

      python.stdout.on('data', (data: Buffer) => {
        event.sender.send(IpcChannels.SURFACE_OUTPUT, { type: 'stdout', data: data.toString() });
      });

      python.stderr.on('data', (data: Buffer) => {
        event.sender.send(IpcChannels.SURFACE_OUTPUT, { type: 'stderr', data: data.toString() });
      });

      python.on('close', (code: number | null) => {
        childProcesses.delete(python);
        if (code === 0) {
          resolve(Ok({ surfaceFile: outputPly }));
        } else {
          resolve(Err({
            type: 'SURFACE_FAILED',
            message: `Process exited with code ${code}`,
          }));
        }
      });

      python.on('error', (error: Error) => {
        childProcesses.delete(python);
        resolve(Err({
          type: 'SURFACE_FAILED',
          message: error.message,
        }));
      });
    });
  }
);

// Run FragGen generation
ipcMain.handle(
  IpcChannels.RUN_GENERATION,
  async (event, options: GenerationOptions): Promise<Result<GenerationResult, AppError>> => {
    const {
      surfacePly, pocketPdb, ligandPdb, outputDir, modelVariant, device, sampling, pocketRadius,
      generationMode = 'denovo', anchorSdfPath = null
    } = options;

    return new Promise((resolve) => {
      console.log('=== RUN FRAGGEN GENERATION ===');
      console.log('Surface PLY:', surfacePly);
      console.log('Pocket PDB:', pocketPdb);
      console.log('Ligand PDB:', ligandPdb);
      console.log('Output dir:', outputDir);
      console.log('Model variant:', modelVariant);
      console.log('Device:', device);
      console.log('Generation mode:', generationMode);
      console.log('Anchor SDF path:', anchorSdfPath);
      console.log('Sampling config:', JSON.stringify(sampling, null, 2));

      if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
        resolve(Err({
          type: 'PYTHON_NOT_FOUND',
          message: 'Python not found. Please install miniconda and create fraggen environment.',
        }));
        return;
      }

      const fraggenScript = getFragGenScript();
      if (!fs.existsSync(fraggenScript)) {
        resolve(Err({
          type: 'SCRIPT_NOT_FOUND',
          path: fraggenScript,
          message: `FragGen script not found: ${fraggenScript}`,
        }));
        return;
      }

      const baseConfigs = getBaseConfigs();
      const baseConfigPath = baseConfigs[modelVariant] || baseConfigs.dihedral;
      if (!fs.existsSync(baseConfigPath)) {
        resolve(Err({
          type: 'FILE_NOT_FOUND',
          path: baseConfigPath,
          message: `Base config not found: ${baseConfigPath}`,
        }));
        return;
      }

      const fragBase = getFragBase();
      if (!fs.existsSync(fragBase)) {
        resolve(Err({
          type: 'FILE_NOT_FOUND',
          path: fragBase,
          message: `Fragment database not found: ${fragBase}`,
        }));
        return;
      }

      // Ensure output directory exists
      fs.mkdirSync(outputDir, { recursive: true });

      // Generate runtime config with custom sampling parameters
      const runtimeConfigPath = generateRuntimeConfig(baseConfigPath, sampling, outputDir);
      console.log('Generated runtime config:', runtimeConfigPath);

      // Save run parameters log
      const runParams: RunParameters = {
        timestamp: new Date().toISOString(),
        inputPdb: pocketPdb,
        modelVariant,
        device,
        pocketRadius,
        sampling,
        outputDir,
        pocketPdb,
        ligandPdb,
        surfacePly,
      };
      const paramsFile = saveRunParameters(runParams, outputDir);
      console.log('Saved run parameters:', paramsFile);

      const args = [
        fraggenScript,
        '--config', runtimeConfigPath,
        '--device', device || 'cpu',
        '--surf_file', surfacePly,
        '--pdb_file', pocketPdb,
        '--frag_base', fragBase,
        '--save_dir', outputDir,
      ];

      // Add anchor mode parameter
      if (generationMode === 'grow') {
        args.push('--anchor_mode', 'grow');

        // Use custom anchor SDF if provided, otherwise use extracted ligand
        if (anchorSdfPath && fs.existsSync(anchorSdfPath)) {
          args.push('--sdf_file', anchorSdfPath);
        } else if (ligandPdb && fs.existsSync(ligandPdb)) {
          // For extracted ligand, the ligandPdb is the anchor
          args.push('--sdf_file', ligandPdb);
        }
      } else {
        args.push('--anchor_mode', 'denovo');

        // Add ligand PDB if provided (for pocket center reference)
        if (ligandPdb && fs.existsSync(ligandPdb)) {
          args.push('--sdf_file', ligandPdb);
        }
      }

      console.log('Running command:', condaPythonPath, args.join(' '));

      const python = spawn(condaPythonPath, args, { cwd: fraggenRoot });
      childProcesses.add(python);
      let stderrOutput = '';

      python.stdout.on('data', (data: Buffer) => {
        const text = data.toString();
        console.log('stdout:', text);
        event.sender.send(IpcChannels.GENERATION_OUTPUT, { type: 'stdout', data: text });
      });

      python.stderr.on('data', (data: Buffer) => {
        const text = data.toString();
        console.error('stderr:', text);
        stderrOutput += text;
        event.sender.send(IpcChannels.GENERATION_OUTPUT, { type: 'stderr', data: text });
      });

      python.on('close', (code: number | null) => {
        childProcesses.delete(python);
        console.log('Generation finished with code:', code);
        if (code === 0) {
          // Determine SDF directory path
          const ligandBaseName = ligandPdb
            ? path.basename(ligandPdb, path.extname(ligandPdb))
            : path.basename(pocketPdb, '.pdb');
          const sdfDir = path.join(outputDir, ligandBaseName, 'SDF');

          resolve(Ok({ outputDir, sdfDir, paramsFile }));
        } else {
          // Include relevant part of stderr in error message
          const lastLines = stderrOutput.split('\n').slice(-5).join('\n');
          resolve(Err({
            type: 'GENERATION_FAILED',
            message: `Process exited with code ${code}: ${lastLines}`,
          }));
        }
      });

      python.on('error', (error: Error) => {
        childProcesses.delete(python);
        console.error('Python process error:', error);
        resolve(Err({
          type: 'GENERATION_FAILED',
          message: error.message,
        }));
      });
    });
  }
);

// Generate 2D thumbnail from SDF
ipcMain.handle(
  IpcChannels.GENERATE_THUMBNAIL,
  async (_event, sdfPath: string): Promise<string | null> => {
    return new Promise((resolve) => {
      console.log('[Thumbnail] Generating for:', sdfPath);

      if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
        console.log('[Thumbnail] Python not found');
        resolve(null);
        return;
      }

      const scriptPath = path.join(fraggenRoot, 'generate_2d_thumbnail.py');
      if (!fs.existsSync(scriptPath)) {
        console.log('[Thumbnail] Script not found:', scriptPath);
        resolve(null);
        return;
      }

      const python = spawn(condaPythonPath, [scriptPath, '--sdf_file', sdfPath]);
      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data: Buffer) => {
        stdout += data.toString();
      });

      python.stderr.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      python.on('close', (code: number | null) => {
        if (code === 0 && stdout.includes('data:image')) {
          console.log('[Thumbnail] Success');
          resolve(stdout.trim());
        } else {
          console.log('[Thumbnail] Failed - code:', code, 'stderr:', stderr);
          resolve(null);
        }
      });

      python.on('error', (err) => {
        console.log('[Thumbnail] Spawn error:', err);
        resolve(null);
      });
    });
  }
);

// Generate results CSV with SMILES and properties
ipcMain.handle(
  IpcChannels.GENERATE_RESULTS_CSV,
  async (event, sdfDir: string, outputCsv: string): Promise<Result<string, AppError>> => {
    return new Promise((resolve) => {
      if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
        resolve(Err({
          type: 'PYTHON_NOT_FOUND',
          message: 'Python not found',
        }));
        return;
      }

      const scriptPath = path.join(fraggenRoot, 'generate_results_csv.py');
      if (!fs.existsSync(scriptPath)) {
        resolve(Err({
          type: 'SCRIPT_NOT_FOUND',
          path: scriptPath,
          message: `CSV generator script not found: ${scriptPath}`,
        }));
        return;
      }

      const args = [scriptPath, '--sdf_dir', sdfDir, '--output', outputCsv];
      const python = spawn(condaPythonPath, args);
      childProcesses.add(python);

      python.stdout.on('data', (data: Buffer) => {
        event.sender.send(IpcChannels.GENERATION_OUTPUT, { type: 'stdout', data: data.toString() });
      });

      python.stderr.on('data', (data: Buffer) => {
        event.sender.send(IpcChannels.GENERATION_OUTPUT, { type: 'stderr', data: data.toString() });
      });

      python.on('close', (code: number | null) => {
        childProcesses.delete(python);
        if (code === 0) {
          resolve(Ok(outputCsv));
        } else {
          resolve(Err({
            type: 'GENERATION_FAILED',
            message: `CSV generation failed with code ${code}`,
          }));
        }
      });

      python.on('error', (error: Error) => {
        childProcesses.delete(python);
        resolve(Err({
          type: 'GENERATION_FAILED',
          message: error.message,
        }));
      });
    });
  }
);

// File operations
ipcMain.handle(
  IpcChannels.FILE_EXISTS,
  async (_event, filePath: string): Promise<boolean> => {
    return fs.existsSync(filePath);
  }
);

ipcMain.handle(
  IpcChannels.GET_FILE_INFO,
  async (_event, filePath: string): Promise<{ exists: boolean; size?: number; modified?: Date }> => {
    try {
      const stats = fs.statSync(filePath);
      return { exists: true, size: stats.size, modified: stats.mtime };
    } catch {
      return { exists: false };
    }
  }
);

ipcMain.handle(
  IpcChannels.CREATE_DIRECTORY,
  async (_event, dirPath: string): Promise<Result<void, AppError>> => {
    try {
      fs.mkdirSync(dirPath, { recursive: true });
      return Ok(undefined);
    } catch (error) {
      return Err({
        type: 'DIRECTORY_ERROR',
        path: dirPath,
        message: (error as Error).message,
      });
    }
  }
);

ipcMain.handle(
  IpcChannels.LIST_SDF_FILES,
  async (_event, dirPath: string): Promise<string[]> => {
    try {
      if (!fs.existsSync(dirPath)) {
        return [];
      }
      const files = fs.readdirSync(dirPath)
        .filter((f) => f.endsWith('.sdf'))
        .sort((a, b) => {
          const aNum = parseInt(a.split('.')[0]);
          const bNum = parseInt(b.split('.')[0]);
          if (isNaN(aNum) || isNaN(bNum)) return a.localeCompare(b);
          return aNum - bNum;
        });
      return files;
    } catch (error) {
      console.error('Error listing SDF files:', error);
      return [];
    }
  }
);

ipcMain.handle(
  IpcChannels.OPEN_FOLDER,
  async (_event, folderPath: string): Promise<void> => {
    const { execFile } = require('child_process');
    // Use platform-appropriate command to open folder
    const openCmd = process.platform === 'darwin' ? 'open' : 'xdg-open';
    execFile(openCmd, [folderPath], (error: Error | null) => {
      if (error) console.error('Failed to open folder:', error);
    });
  }
);

ipcMain.handle(
  IpcChannels.GET_AVAILABLE_DEVICES,
  async (): Promise<string[]> => {
    const devices = ['cpu'];

    if (process.platform === 'darwin') {
      // macOS: MPS for Apple Silicon (FragGen/PyTorch)
      devices.push('mps');
      // macOS: Metal for MD (OpenMM Metal backend, if installed)
      devices.push('metal');
    } else if (process.platform === 'linux') {
      // Linux: Check for NVIDIA GPU via nvidia-smi
      try {
        const { execSync } = require('child_process');
        execSync('nvidia-smi', { stdio: 'ignore' });
        devices.push('cuda');
        console.log('CUDA device detected via nvidia-smi');
      } catch (err) {
        console.log('No NVIDIA GPU detected:', err);
      }
    }

    console.log('Available devices:', devices);
    return devices;
  }
);

// Select output folder
ipcMain.handle(IpcChannels.SELECT_OUTPUT_FOLDER, async (): Promise<string | null> => {
  if (!mainWindow) return null;
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory', 'createDirectory'],
    title: 'Select Output Folder',
  });
  return result.filePaths[0] || null;
});

// === Stats Management ===

function getStatsPath(): string {
  return path.join(os.homedir(), '.fraggen', 'stats.json');
}

function ensureStatsDir(): void {
  const statsDir = path.join(os.homedir(), '.fraggen');
  if (!fs.existsSync(statsDir)) {
    fs.mkdirSync(statsDir, { recursive: true });
  }
}

function getDefaultStats(): GenerationStats {
  return {
    totalMoleculesGenerated: 0,
    sessionsCount: 0,
    lastGenerationDate: null,
  };
}

ipcMain.handle(IpcChannels.GET_STATS, async (): Promise<GenerationStats> => {
  try {
    const statsPath = getStatsPath();
    if (fs.existsSync(statsPath)) {
      const content = fs.readFileSync(statsPath, 'utf-8');
      return JSON.parse(content) as GenerationStats;
    }
    return getDefaultStats();
  } catch (error) {
    console.error('Error reading stats:', error);
    return getDefaultStats();
  }
});

ipcMain.handle(
  IpcChannels.UPDATE_STATS,
  async (_event, moleculeCount: number): Promise<GenerationStats> => {
    try {
      ensureStatsDir();
      const statsPath = getStatsPath();
      let stats = getDefaultStats();

      if (fs.existsSync(statsPath)) {
        try {
          const content = fs.readFileSync(statsPath, 'utf-8');
          stats = JSON.parse(content) as GenerationStats;
        } catch {
          // Use default if parse fails
        }
      }

      stats.totalMoleculesGenerated += moleculeCount;
      stats.sessionsCount += 1;
      stats.lastGenerationDate = new Date().toISOString();

      fs.writeFileSync(statsPath, JSON.stringify(stats, null, 2), 'utf-8');
      console.log('Updated stats:', stats);

      return stats;
    } catch (error) {
      console.error('Error updating stats:', error);
      return getDefaultStats();
    }
  }
);

ipcMain.handle(
  IpcChannels.CHECK_JOB_EXISTS,
  async (_event, outputFolder: string, jobName: string): Promise<boolean> => {
    const jobPath = path.join(outputFolder, jobName);
    return fs.existsSync(jobPath);
  }
);

// Validate anchor SDF for fragment growing mode
ipcMain.handle(
  IpcChannels.VALIDATE_ANCHOR_SDF,
  async (_event, sdfPath: string): Promise<Result<{
    valid: boolean;
    atomCount: number;
    has3DCoords: boolean;
  }, AppError>> => {
    return new Promise((resolve) => {
      if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
        resolve(Err({
          type: 'PYTHON_NOT_FOUND',
          message: 'Python not found. Please install miniconda and create fraggen environment.',
        }));
        return;
      }

      if (!fs.existsSync(sdfPath)) {
        resolve(Err({
          type: 'FILE_NOT_FOUND',
          path: sdfPath,
          message: `Anchor file not found: ${sdfPath}`,
        }));
        return;
      }

      // Use inline Python to validate the SDF or PDB
      const pythonCode = `
import sys
import json

try:
    from rdkit import Chem
    import numpy as np

    file_path = '${sdfPath.replace(/'/g, "\\'")}'
    ext = file_path.lower().split('.')[-1]
    mol = None

    # Load based on file extension
    if ext == 'pdb':
        mol = Chem.MolFromPDBFile(file_path, removeHs=False)
    elif ext in ('sdf', 'mol'):
        mol = Chem.SDMolSupplier(file_path, removeHs=False)[0]
    else:
        # Try SDF first, then PDB
        mol = Chem.SDMolSupplier(file_path, removeHs=False)[0]
        if mol is None:
            mol = Chem.MolFromPDBFile(file_path, removeHs=False)

    if mol is None:
        print(json.dumps({"valid": False, "atomCount": 0, "has3DCoords": False, "error": "Could not parse file"}))
        sys.exit(0)

    atom_count = mol.GetNumAtoms()

    # Check for 3D coordinates
    has_3d = False
    try:
        conf = mol.GetConformer()
        positions = conf.GetPositions()
        # Check if Z coordinates are not all zero (2D structure check)
        z_coords = positions[:, 2]
        has_3d = not np.allclose(z_coords, 0, atol=0.01)
    except:
        has_3d = False

    print(json.dumps({"valid": True, "atomCount": atom_count, "has3DCoords": has_3d}))
except Exception as e:
    print(json.dumps({"valid": False, "atomCount": 0, "has3DCoords": False, "error": str(e)}))
`;

      const python = spawn(condaPythonPath, ['-c', pythonCode]);
      childProcesses.add(python);

      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data: Buffer) => {
        stdout += data.toString();
      });

      python.stderr.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      python.on('close', (code: number | null) => {
        childProcesses.delete(python);
        if (code === 0 && stdout.trim()) {
          try {
            const result = JSON.parse(stdout.trim());
            if (result.error) {
              resolve(Err({
                type: 'VALIDATION_FAILED',
                message: result.error,
              }));
            } else {
              resolve(Ok({
                valid: result.valid,
                atomCount: result.atomCount,
                has3DCoords: result.has3DCoords,
              }));
            }
          } catch (e) {
            resolve(Err({
              type: 'PARSE_FAILED',
              message: `Failed to parse validation result: ${stdout}`,
            }));
          }
        } else {
          resolve(Err({
            type: 'VALIDATION_FAILED',
            message: stderr || 'Validation failed',
          }));
        }
      });

      python.on('error', (error: Error) => {
        childProcesses.delete(python);
        resolve(Err({
          type: 'VALIDATION_FAILED',
          message: error.message,
        }));
      });
    });
  }
);

// === Docking Handlers (Vina + CORDIAL) ===

// Select CSV file
ipcMain.handle(IpcChannels.SELECT_CSV_FILE, async (): Promise<string | null> => {
  if (!mainWindow) return null;
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'CSV Files', extensions: ['csv'] },
      { name: 'All Files', extensions: ['*'] },
    ],
  });
  return result.filePaths[0] || null;
});

// Select SDF/MOL file (for Viewer mode, single molecule input)
ipcMain.handle(IpcChannels.SELECT_SDF_FILE, async (): Promise<string | null> => {
  if (!mainWindow) return null;
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Molecule Files', extensions: ['sdf', 'mol', 'mol2'] },
      { name: 'Compressed SDF', extensions: ['sdf.gz', 'gz'] },
      { name: 'All Files', extensions: ['*'] },
    ],
  });
  return result.filePaths[0] || null;
});

// Save PDB file (for Viewer mode export)
ipcMain.handle(
  IpcChannels.SAVE_PDB_FILE,
  async (_event, content: string, defaultName?: string): Promise<string | null> => {
    if (!mainWindow) return null;
    const result = await dialog.showSaveDialog(mainWindow, {
      defaultPath: defaultName || 'complex.pdb',
      filters: [
        { name: 'PDB Files', extensions: ['pdb'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    });
    if (result.canceled || !result.filePath) return null;

    try {
      await fs.promises.writeFile(result.filePath, content, 'utf-8');
      return result.filePath;
    } catch (err) {
      console.error('Failed to save PDB:', err);
      return null;
    }
  }
);

// Parse FragGen results CSV (legacy — kept for backwards compat)
ipcMain.handle(
  'parse-fraggen-csv',
  async (_event, csvPath: string): Promise<Result<{
    molecules: Array<{
      filename: string;
      smiles: string;
      qed: number;
      saScore: number;
      sdfPath: string;
    }>;
    qedRange: { min: number; max: number };
    sdfDirectory: string;
  }, AppError>> => {
    try {
      const csvContent = fs.readFileSync(csvPath, 'utf-8');
      const lines = csvContent.trim().split('\n');

      if (lines.length < 2) {
        return Err({ type: 'PARSE_FAILED', message: 'CSV file is empty or has no data rows' });
      }

      const header = lines[0].split(',').map(h => h.trim().toLowerCase());
      const filenameIdx = header.findIndex(h => h === 'filename' || h === 'name');
      const smilesIdx = header.findIndex(h => h === 'smiles');
      const qedIdx = header.findIndex(h => h === 'qed');
      const saIdx = header.findIndex(h => h === 'sa_score' || h === 'sascore');

      if (filenameIdx === -1) {
        return Err({ type: 'PARSE_FAILED', message: 'CSV missing filename column' });
      }

      const csvDir = path.dirname(csvPath);
      // Try to find the SDF directory - check common patterns
      let sdfDirectory = '';
      const possibleSdfDirs = [
        path.join(csvDir, 'SDF'),
        path.join(csvDir, '..', 'SDF'),
        path.join(csvDir, 'ligand', 'SDF'),
      ];

      // Also check for ligand subdirectories (FragGen output structure)
      try {
        const parentDirContents = fs.readdirSync(csvDir);
        for (const item of parentDirContents) {
          const itemPath = path.join(csvDir, item);
          const sdfPath = path.join(itemPath, 'SDF');
          if (fs.statSync(itemPath).isDirectory() && fs.existsSync(sdfPath)) {
            possibleSdfDirs.unshift(sdfPath);
          }
        }
      } catch (e) {
        // Ignore errors
      }

      for (const dir of possibleSdfDirs) {
        if (fs.existsSync(dir)) {
          sdfDirectory = dir;
          break;
        }
      }

      const molecules: Array<{
        filename: string;
        smiles: string;
        qed: number;
        saScore: number;
        sdfPath: string;
      }> = [];

      for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',').map(v => v.trim());
        const filename = values[filenameIdx] || '';
        if (!filename) continue;

        const qed = qedIdx >= 0 ? parseFloat(values[qedIdx]) : 0.5;
        const saScore = saIdx >= 0 ? parseFloat(values[saIdx]) : 3;

        // Try to find the SDF file
        let sdfPath = '';
        if (sdfDirectory) {
          const sdfFile = path.join(sdfDirectory, filename.endsWith('.sdf') ? filename : `${filename}.sdf`);
          if (fs.existsSync(sdfFile)) {
            sdfPath = sdfFile;
          }
        }

        molecules.push({
          filename: filename.replace('.sdf', ''),
          smiles: smilesIdx >= 0 ? values[smilesIdx] : '',
          qed: isNaN(qed) ? 0.5 : qed,
          saScore: isNaN(saScore) ? 3 : saScore,
          sdfPath,
        });
      }

      const qeds = molecules.map(m => m.qed);
      const qedRange = {
        min: qeds.length > 0 ? Math.min(...qeds) : 0,
        max: qeds.length > 0 ? Math.max(...qeds) : 1,
      };

      return Ok({ molecules, qedRange, sdfDirectory });
    } catch (error) {
      return Err({ type: 'PARSE_FAILED', message: (error as Error).message });
    }
  }
);


// Get CPU count for parallel docking
ipcMain.handle(IpcChannels.GET_CPU_COUNT, async (): Promise<number> => {
  return os.cpus().length;
});

// === Node.js-Managed Parallel Docking ===

interface DockingResult {
  ligand: string;
  success: boolean;
  output?: string;
  error?: string;
}

interface VinaDockConfig {
  exhaustiveness: number;
  numPoses: number;
  autoboxAdd: number;
  numCpus: number;
  seed: number;
  coreConstrained: boolean;
}

// Track active docking processes for cancellation
const dockingProcesses = new Set<ChildProcess>();

/**
 * Concurrency-limited parallel execution helper with staggered starts.
 * Executes async functions with a maximum number of concurrent operations.
 * Adds a delay between starting each new job to avoid resource contention.
 */
async function runWithConcurrency<T, R>(
  items: T[],
  concurrency: number,
  fn: (item: T, index: number) => Promise<R>,
  onProgress?: (completed: number, total: number, result: R) => void,
  staggerDelayMs: number = 0
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let completed = 0;
  let running = 0;
  let index = 0;
  let lastStartTime = 0;

  return new Promise((resolve) => {
    const runNext = async () => {
      while (running < concurrency && index < items.length) {
        const currentIndex = index++;
        running++;

        // Stagger job starts to avoid resource contention (e.g., Open Babel init)
        if (staggerDelayMs > 0) {
          const now = Date.now();
          const elapsed = now - lastStartTime;
          if (elapsed < staggerDelayMs) {
            await new Promise(r => setTimeout(r, staggerDelayMs - elapsed));
          }
          lastStartTime = Date.now();
        }

        fn(items[currentIndex], currentIndex)
          .then((result) => {
            results[currentIndex] = result;
            completed++;
            running--;
            onProgress?.(completed, items.length, result);
            runNext();
          })
          .catch((error) => {
            results[currentIndex] = error;
            completed++;
            running--;
            onProgress?.(completed, items.length, error);
            runNext();
          });
      }

      if (completed === items.length) {
        resolve(results);
      }
    };

    runNext();
  });
}

/**
 * Dock a single ligand using Vina Python API script.
 * Returns a promise that resolves when docking completes.
 */
function dockSingleLigandVina(
  ligandPath: string,
  receptor: string,
  reference: string,
  outputDir: string,
  config: VinaDockConfig
): Promise<DockingResult> {
  return new Promise((resolve) => {
    const name = path.basename(ligandPath, '.sdf');
    const scriptPath = path.join(fraggenRoot, 'run_vina_docking.py');

    // Derive project name from output path: .../docking/{runFolder}
    const vinaProjectName = path.basename(path.resolve(outputDir, '../..'));

    const args = [
      scriptPath,
      '--receptor', receptor,
      '--ligand', ligandPath,
      '--reference', reference,
      '--output_dir', outputDir,
      '--exhaustiveness', String(config.exhaustiveness),
      '--num_poses', String(config.numPoses),
      '--autobox_add', String(config.autoboxAdd),
      '--cpu', '1',  // Each Vina process uses 1 CPU, concurrency handled by Node.js
      '--project_name', vinaProjectName,
    ];

    if (config.seed > 0) {
      args.push('--seed', String(config.seed));
    }

    if (config.coreConstrained) {
      args.push('--core_constrain', '--reference_sdf', reference);
    }

    // Set BABEL_DATADIR to help Open Babel find its data files
    const babelDataDir = process.env.BABEL_DATADIR || detectBabelDataDir();
    const env = {
      ...process.env,
      ...(babelDataDir ? { BABEL_DATADIR: babelDataDir } : {}),
    };

    const python = spawn(condaPythonPath!, args, { env });
    childProcesses.add(python);
    dockingProcesses.add(python);

    let stdout = '';
    let stderr = '';

    python.stdout.on('data', (data: Buffer) => {
      stdout += data.toString();
    });

    python.stderr.on('data', (data: Buffer) => {
      stderr += data.toString();
    });

    python.on('close', (code: number | null) => {
      childProcesses.delete(python);
      dockingProcesses.delete(python);
      if (code === 0) {
        const match = stdout.match(/SUCCESS:([^:]+):(.+)/);
        resolve({
          ligand: name,
          success: true,
          output: match ? match[2].trim() : undefined,
        });
      } else {
        resolve({
          ligand: name,
          success: false,
          error: stderr.slice(0, 200) || 'Unknown error',
        });
      }
    });

    python.on('error', (err: Error) => {
      childProcesses.delete(python);
      dockingProcesses.delete(python);
      resolve({ ligand: name, success: false, error: err.message });
    });
  });
}

// Cancel Vina docking — kill all docking child processes
ipcMain.handle(IpcChannels.CANCEL_VINA_DOCKING, async (): Promise<void> => {
  for (const proc of dockingProcesses) {
    if (!proc.killed) {
      proc.kill('SIGTERM');
    }
  }
  dockingProcesses.clear();
});

// Run Vina docking — Node.js-managed parallel execution
ipcMain.handle(
  IpcChannels.RUN_VINA_DOCKING,
  async (
    event,
    receptorPdb: string,
    referenceLigand: string,
    ligandSdfPaths: string[],
    outputDir: string,
    config: VinaDockConfig
  ): Promise<Result<string, AppError>> => {
    if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Python not found. Please install miniconda and create the openmm-metal environment.',
      });
    }

    const scriptPath = path.join(fraggenRoot, 'run_vina_docking.py');
    if (!fs.existsSync(scriptPath)) {
      return Err({
        type: 'SCRIPT_NOT_FOUND',
        path: scriptPath,
        message: `Vina docking script not found: ${scriptPath}`,
      });
    }

    // Create output directory and inputs/ subdir
    fs.mkdirSync(outputDir, { recursive: true });
    const inputsDir = path.join(outputDir, 'inputs');
    fs.mkdirSync(inputsDir, { recursive: true });

    // Copy receptor and reference ligand to inputs/ (no project prefix)
    const projectName = path.basename(path.resolve(outputDir, '../..'));
    const receptorOutputPath = path.join(inputsDir, 'receptor.pdb');
    const referenceOutputPath = path.join(inputsDir, 'reference_ligand.sdf');
    fs.copyFileSync(receptorPdb, receptorOutputPath);
    fs.copyFileSync(referenceLigand, referenceOutputPath);

    // Write ligands list to inputs/ligands.json
    const ligandsJsonPath = path.join(inputsDir, 'ligands.json');
    fs.writeFileSync(ligandsJsonPath, JSON.stringify(ligandSdfPaths, null, 2));

    // Vina is CPU-only — concurrency = CPU count (each process uses 1 CPU)
    const concurrency = config.numCpus > 0 ? config.numCpus : os.cpus().length;

    // Emit header
    event.sender.send(IpcChannels.DOCK_OUTPUT, {
      type: 'stdout',
      data: `=== Vina Parallel Docking ===\nWorkers: ${concurrency}\nLigands: ${ligandSdfPaths.length}\nReceptor: ${receptorPdb}\nReference: ${referenceLigand}\nOutput: ${outputDir}\n\n`
    });

    console.log(`Starting Vina parallel docking: ${ligandSdfPaths.length} ligands, ${concurrency} workers`);

    let successful = 0;
    let failed = 0;

    // Run first ligand sequentially before opening parallel workers
    if (ligandSdfPaths.length > 0) {
      event.sender.send(IpcChannels.DOCK_OUTPUT, {
        type: 'stdout',
        data: `Preparing first ligand...\n`
      });

      const firstResult = await dockSingleLigandVina(
        ligandSdfPaths[0],
        receptorPdb,
        referenceLigand,
        outputDir,
        config
      );

      if (firstResult.success) successful++;
      else failed++;

      const firstStatusLine = firstResult.success
        ? `DOCKING: 1/${ligandSdfPaths.length} - ${firstResult.ligand} - OK\n  ${firstResult.output}\n`
        : `DOCKING: 1/${ligandSdfPaths.length} - ${firstResult.ligand} - FAILED\n  ${firstResult.error}\n`;

      console.log(firstStatusLine.trim());
      event.sender.send(IpcChannels.DOCK_OUTPUT, {
        type: 'stdout',
        data: firstStatusLine
      });

      await new Promise(r => setTimeout(r, 100));
    }

    // Process remaining ligands in parallel
    const remainingLigands = ligandSdfPaths.slice(1);

    if (remainingLigands.length > 0) {
      event.sender.send(IpcChannels.DOCK_OUTPUT, {
        type: 'stdout',
        data: `\nDocking ${remainingLigands.length} remaining ligands (${concurrency} workers)...\n\n`
      });

      await runWithConcurrency(
        remainingLigands,
        concurrency,
        async (ligandPath) => {
          return dockSingleLigandVina(ligandPath, receptorPdb, referenceLigand, outputDir, config);
        },
        (completed, total, result) => {
          if (result.success) successful++;
          else failed++;

          const statusLine = result.success
            ? `DOCKING: ${completed + 1}/${ligandSdfPaths.length} - ${result.ligand} - OK\n  ${result.output}\n`
            : `DOCKING: ${completed + 1}/${ligandSdfPaths.length} - ${result.ligand} - FAILED\n  ${result.error}\n`;

          console.log(statusLine.trim());
          event.sender.send(IpcChannels.DOCK_OUTPUT, {
            type: 'stdout',
            data: statusLine
          });
        },
        100  // minimal stagger — OBabel already initialized by first ligand
      );
    }

    event.sender.send(IpcChannels.DOCK_OUTPUT, {
      type: 'stdout',
      data: `\n=== COMPLETE ===\nSuccessful: ${successful}/${ligandSdfPaths.length}\nFailed: ${failed}\n`
    });

    console.log(`Vina docking complete: ${successful} successful, ${failed} failed`);

    if (failed === ligandSdfPaths.length) {
      return Err({ type: 'DOCKING_FAILED', message: 'All docking jobs failed' });
    }

    // Post-processing: move docked files into results/poses/ and create pooled SDF
    try {
      const resultsDir = path.join(outputDir, 'results');
      const posesDir = path.join(resultsDir, 'poses');
      fs.mkdirSync(posesDir, { recursive: true });

      const allFiles = fs.readdirSync(outputDir);
      const dockedGzFiles = allFiles.filter((f) => f.endsWith('_docked.sdf.gz'));

      // Move *_docked.sdf.gz into results/poses/
      for (const f of dockedGzFiles) {
        fs.renameSync(path.join(outputDir, f), path.join(posesDir, f));
      }

      // Copy X-ray reference ligand into poses dir for comparison scoring
      if (referenceLigand && fs.existsSync(referenceLigand)) {
        try {
          const refSdf = fs.readFileSync(referenceLigand, 'utf-8');
          // Write as uncompressed SDF (parser handles both .sdf and .sdf.gz)
          const refName = path.basename(outputDir) + '_xray_reference_docked.sdf';
          fs.writeFileSync(path.join(posesDir, refName), refSdf);
          event.sender.send(IpcChannels.DOCK_OUTPUT, {
            type: 'stdout',
            data: 'Including X-ray reference ligand for comparison\n',
          });
        } catch (e) {
          console.error('Failed to copy reference ligand:', e);
        }
      }

      // Create pooled SDF: extract first model from each .sdf.gz
      const pooledParts: string[] = [];
      for (const f of dockedGzFiles) {
        try {
          const gzData = fs.readFileSync(path.join(posesDir, f));
          const sdfText = zlib.gunzipSync(gzData).toString('utf-8');
          // First model ends at first $$$$
          const delimIdx = sdfText.indexOf('$$$$');
          if (delimIdx >= 0) {
            pooledParts.push(sdfText.substring(0, delimIdx + 4));
          }
        } catch (e) {
          console.error(`Failed to extract first model from ${f}:`, e);
        }
      }

      if (pooledParts.length > 0) {
        const pooledPath = path.join(resultsDir, 'all_docked.sdf');
        fs.writeFileSync(pooledPath, pooledParts.join('\n') + '\n');
        console.log(`Pooled ${pooledParts.length} best poses into ${pooledPath}`);
      }
    } catch (e) {
      console.error('Post-processing (pooling) failed:', e);
      // Non-fatal — docking results are still available
    }

    return Ok(outputDir);
  }
);

// Parse Vina docking results — reads *_docked.sdf.gz files from output dir
ipcMain.handle(
  IpcChannels.PARSE_DOCK_RESULTS,
  async (_event, outputDir: string): Promise<Result<Array<{
    ligandName: string;
    smiles: string;
    qed: number;
    vinaAffinity: number;
    poseIndex: number;
    outputSdf: string;
    parentMolecule: string;
    protonationVariant: number | null;
    conformerIndex: number | null;
    cordialExpectedPkd?: number;
    cordialPHighAffinity?: number;
    cordialPVeryHighAffinity?: number;
    coreRmsd?: number;
  }>, AppError>> => {
    try {
      if (!fs.existsSync(outputDir)) {
        return Err({ type: 'DIRECTORY_ERROR', path: outputDir, message: 'Output directory not found' });
      }

      // Check new layout first (results/poses/), then legacy (poses/), then top-level
      const newPosesDir = path.join(outputDir, 'results', 'poses');
      const legacyPosesDir = path.join(outputDir, 'poses');
      const searchDir = fs.existsSync(newPosesDir) ? newPosesDir : fs.existsSync(legacyPosesDir) ? legacyPosesDir : outputDir;
      const files = fs.readdirSync(searchDir);
      const dockedFiles = files.filter((f) => f.endsWith('_docked.sdf.gz'));

      if (dockedFiles.length === 0) {
        return Err({ type: 'FILE_NOT_FOUND', path: outputDir, message: 'No docked SDF files found' });
      }

      // Parse all docked SDF files in parallel
      const parsePromises = dockedFiles.map(async (sdfFile) => {
        const sdfPath = path.join(searchDir, sdfFile);
        const name = sdfFile.replace('_docked.sdf.gz', '');
        const props = await parseSdfProperties(sdfPath);
        return {
          ligandName: name,
          smiles: props.smiles || '',
          qed: props.qed,
          vinaAffinity: props.vinaAffinity,
          poseIndex: 0,
          outputSdf: sdfPath,
          parentMolecule: name,
          protonationVariant: null,
          conformerIndex: null,
          coreRmsd: props.coreRmsd,
        };
      });
      const results: Array<any> = await Promise.all(parsePromises);

      // Load CORDIAL scores if available (check results/ first, then top-level)
      const newCordialPath = path.join(outputDir, 'results', 'cordial_scores.json');
      const cordialJsonPath = fs.existsSync(newCordialPath) ? newCordialPath : path.join(outputDir, 'cordial_scores.json');
      if (fs.existsSync(cordialJsonPath)) {
        try {
          const cordialData = JSON.parse(fs.readFileSync(cordialJsonPath, 'utf-8'));
          const cordialByName = new Map<string, { expectedPkd: number; pHighAffinity: number; pVeryHighAffinity: number }>();
          for (const entry of cordialData) {
            const entryName = entry.source_name;
            const pHighAffinity = entry.cordial_p_high_affinity;
            const existing = cordialByName.get(entryName);
            if (!existing || pHighAffinity > existing.pHighAffinity) {
              cordialByName.set(entryName, {
                expectedPkd: entry.cordial_expected_pkd,
                pHighAffinity,
                pVeryHighAffinity: entry.cordial_p_very_high_affinity || 0,
              });
            }
          }
          for (const result of results) {
            const scores = cordialByName.get(result.ligandName);
            if (scores) {
              result.cordialExpectedPkd = scores.expectedPkd;
              result.cordialPHighAffinity = scores.pHighAffinity;
              result.cordialPVeryHighAffinity = scores.pVeryHighAffinity;
            }
          }
        } catch (e) {
          console.error('Failed to load CORDIAL scores:', e);
        }
      }

      // Sort by Vina affinity ascending (most negative = best)
      results.sort((a: any, b: any) => a.vinaAffinity - b.vinaAffinity);

      return Ok(results);
    } catch (error) {
      return Err({ type: 'PARSE_FAILED', message: (error as Error).message });
    }
  }
);

// List SDF files in directory
ipcMain.handle(
  IpcChannels.LIST_SDF_IN_DIRECTORY,
  async (_event, dirPath: string): Promise<string[]> => {
    try {
      if (!fs.existsSync(dirPath)) {
        return [];
      }
      const files = fs.readdirSync(dirPath)
        .filter((f) => f.endsWith('.sdf'))
        .sort((a, b) => {
          const aNum = parseInt(a.split('.')[0]);
          const bNum = parseInt(b.split('.')[0]);
          if (isNaN(aNum) || isNaN(bNum)) return a.localeCompare(b);
          return aNum - bNum;
        });
      return files;
    } catch (error) {
      console.error('Error listing SDF files:', error);
      return [];
    }
  }
);

// Detect ligands in PDB file
ipcMain.handle(
  IpcChannels.DETECT_PDB_LIGANDS,
  async (_event, pdbPath: string): Promise<Result<Array<{
    id: string;
    resname: string;
    chain: string;
    resnum: string;
    num_atoms: number;
    centroid: { x: number; y: number; z: number };
  }>, AppError>> => {
    return new Promise((resolve) => {
      if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
        resolve(Err({
          type: 'PYTHON_NOT_FOUND',
          message: 'Python not found',
        }));
        return;
      }

      const scriptPath = path.join(fraggenRoot, 'detect_pdb_ligands.py');
      if (!fs.existsSync(scriptPath)) {
        resolve(Err({
          type: 'SCRIPT_NOT_FOUND',
          path: scriptPath,
          message: `Ligand detection script not found: ${scriptPath}`,
        }));
        return;
      }

      const python = spawn(condaPythonPath, [
        scriptPath,
        '--pdb', pdbPath,
        '--mode', 'detect'
      ]);

      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data: Buffer) => {
        stdout += data.toString();
      });

      python.stderr.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      python.on('close', (code: number | null) => {
        if (code === 0) {
          try {
            const ligands = JSON.parse(stdout);
            resolve(Ok(ligands));
          } catch (e) {
            resolve(Err({
              type: 'PARSE_FAILED',
              message: `Failed to parse ligand detection output: ${(e as Error).message}`,
            }));
          }
        } else {
          resolve(Err({
            type: 'PARSE_FAILED',
            message: `Ligand detection failed: ${stderr}`,
          }));
        }
      });

      python.on('error', (error: Error) => {
        resolve(Err({
          type: 'PARSE_FAILED',
          message: error.message,
        }));
      });
    });
  }
);

// Extract ligand from PDB
ipcMain.handle(
  IpcChannels.EXTRACT_LIGAND,
  async (_event, pdbPath: string, ligandId: string, outputPath: string): Promise<Result<string, AppError>> => {
    return new Promise((resolve) => {
      if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
        resolve(Err({
          type: 'PYTHON_NOT_FOUND',
          message: 'Python not found',
        }));
        return;
      }

      const scriptPath = path.join(fraggenRoot, 'detect_pdb_ligands.py');
      if (!fs.existsSync(scriptPath)) {
        resolve(Err({
          type: 'SCRIPT_NOT_FOUND',
          path: scriptPath,
          message: `Ligand extraction script not found: ${scriptPath}`,
        }));
        return;
      }

      fs.mkdirSync(path.dirname(outputPath), { recursive: true });

      const python = spawn(condaPythonPath, [
        scriptPath,
        '--pdb', pdbPath,
        '--mode', 'extract',
        '--ligand_id', ligandId,
        '--output', outputPath
      ]);

      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data: Buffer) => {
        stdout += data.toString();
      });

      python.stderr.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      python.on('close', (code: number | null) => {
        if (code === 0) {
          resolve(Ok(outputPath));
        } else {
          resolve(Err({
            type: 'PARSE_FAILED',
            message: `Ligand extraction failed: ${stderr}`,
          }));
        }
      });

      python.on('error', (error: Error) => {
        resolve(Err({
          type: 'PARSE_FAILED',
          message: error.message,
        }));
      });
    });
  }
);

// Prepare receptor (remove ligand, add hydrogens, optionally retain waters)
ipcMain.handle(
  IpcChannels.PREPARE_RECEPTOR,
  async (_event, pdbPath: string, ligandId: string, outputPath: string, waterDistance: number = 0): Promise<Result<string, AppError>> => {
    return new Promise((resolve) => {
      if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
        resolve(Err({
          type: 'PYTHON_NOT_FOUND',
          message: 'Python not found',
        }));
        return;
      }

      const scriptPath = path.join(fraggenRoot, 'detect_pdb_ligands.py');
      if (!fs.existsSync(scriptPath)) {
        resolve(Err({
          type: 'SCRIPT_NOT_FOUND',
          path: scriptPath,
          message: `Receptor preparation script not found: ${scriptPath}`,
        }));
        return;
      }

      fs.mkdirSync(path.dirname(outputPath), { recursive: true });

      const prepArgs = [
        scriptPath,
        '--pdb', pdbPath,
        '--mode', 'prepare_receptor',
        '--ligand_id', ligandId,
        '--output', outputPath,
        '--water_distance', String(waterDistance || 0),
      ];

      const python = spawn(condaPythonPath, prepArgs);

      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data: Buffer) => {
        stdout += data.toString();
      });

      python.stderr.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      python.on('close', (code: number | null) => {
        if (code === 0) {
          resolve(Ok(outputPath));
        } else {
          resolve(Err({
            type: 'PARSE_FAILED',
            message: `Receptor preparation failed: ${stderr}`,
          }));
        }
      });

      python.on('error', (error: Error) => {
        resolve(Err({
          type: 'PARSE_FAILED',
          message: error.message,
        }));
      });
    });
  }
);

// Export docking results to CSV — done in Node.js (no Python needed)
ipcMain.handle(
  IpcChannels.EXPORT_DOCK_CSV,
  async (_event, outputDir: string, csvOutput: string, bestOnly: boolean): Promise<Result<string, AppError>> => {
    try {
      // Re-parse results (check results/poses/ first, then poses/, then top-level)
      const csvNewPosesDir = path.join(outputDir, 'results', 'poses');
      const csvLegacyPosesDir = path.join(outputDir, 'poses');
      const csvSearchDir = fs.existsSync(csvNewPosesDir) ? csvNewPosesDir : fs.existsSync(csvLegacyPosesDir) ? csvLegacyPosesDir : outputDir;
      const files = fs.readdirSync(csvSearchDir);
      const dockedFiles = files.filter((f) => f.endsWith('_docked.sdf.gz'));

      const rows: string[] = [];
      const header = ['ligand', 'vina_affinity', 'qed', 'smiles', 'sdf_path'];

      // Check for CORDIAL scores (results/ first, then top-level)
      const csvNewCordialPath = path.join(outputDir, 'results', 'cordial_scores.json');
      const cordialJsonPath = fs.existsSync(csvNewCordialPath) ? csvNewCordialPath : path.join(outputDir, 'cordial_scores.json');
      const hasCordial = fs.existsSync(cordialJsonPath);
      if (hasCordial) {
        header.push('cordial_pkd', 'cordial_p_high_affinity');
      }

      rows.push(header.join(','));

      for (const sdfFile of dockedFiles) {
        const sdfPath = path.join(csvSearchDir, sdfFile);
        const name = sdfFile.replace('_docked.sdf.gz', '');
        const props = await parseSdfProperties(sdfPath);

        const row = [
          name,
          String(props.vinaAffinity),
          String(props.qed),
          `"${(props.smiles || '').replace(/"/g, '""')}"`,
          sdfPath,
        ];

        rows.push(row.join(','));
      }

      fs.writeFileSync(csvOutput, rows.join('\n'));
      return Ok(csvOutput);
    } catch (error) {
      return Err({ type: 'EXPORT_FAILED', message: (error as Error).message });
    }
  }
);

// Export protein-ligand complex PDB
ipcMain.handle(
  IpcChannels.EXPORT_COMPLEX_PDB,
  async (_event, receptorPdb: string, ligandSdf: string, poseIndex: number, outputPath: string): Promise<Result<string, AppError>> => {
    return new Promise((resolve) => {
      if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
        resolve(Err({
          type: 'PYTHON_NOT_FOUND',
          message: 'Python not found',
        }));
        return;
      }

      const scriptPath = path.join(fraggenRoot, 'export_complex_pdb.py');
      if (!fs.existsSync(scriptPath)) {
        resolve(Err({
          type: 'SCRIPT_NOT_FOUND',
          path: scriptPath,
          message: `Complex export script not found: ${scriptPath}`,
        }));
        return;
      }

      const python = spawn(condaPythonPath, [
        scriptPath,
        '--receptor', receptorPdb,
        '--ligand_sdf', ligandSdf,
        '--pose', String(poseIndex),
        '--output', outputPath,
      ]);

      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data: Buffer) => {
        stdout += data.toString();
      });

      python.stderr.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      python.on('close', (code: number | null) => {
        if (code === 0) {
          resolve(Ok(outputPath));
        } else {
          resolve(Err({
            type: 'PARSE_FAILED',
            message: `Complex export failed: ${stderr}`,
          }));
        }
      });

      python.on('error', (error: Error) => {
        resolve(Err({
          type: 'PARSE_FAILED',
          message: error.message,
        }));
      });
    });
  }
);

// === Multi-input Ligand Source Handlers ===

// Select folder dialog
ipcMain.handle(IpcChannels.SELECT_FOLDER, async (): Promise<string | null> => {
  if (!mainWindow) return null;
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
    title: 'Select SDF Directory',
  });
  return result.filePaths[0] || null;
});

// Scan SDF directory for ligand files
ipcMain.handle(
  IpcChannels.SCAN_SDF_DIRECTORY,
  async (_event, dirPath: string, outputDir: string): Promise<Result<Array<{
    filename: string;
    smiles: string;
    qed: number;
    saScore: number;
    sdfPath: string;
  }>, AppError>> => {
    return new Promise((resolve) => {
      if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
        resolve(Err({
          type: 'PYTHON_NOT_FOUND',
          message: 'Python not found. Please install miniconda and create fraggen environment.',
        }));
        return;
      }

      const scriptPath = path.join(fraggenRoot, 'scan_sdf_directory.py');
      if (!fs.existsSync(scriptPath)) {
        resolve(Err({
          type: 'SCRIPT_NOT_FOUND',
          path: scriptPath,
          message: `SDF directory scanner script not found: ${scriptPath}`,
        }));
        return;
      }

      const python = spawn(condaPythonPath, [
        scriptPath,
        '--directory', dirPath,
        '--output_dir', outputDir
      ]);

      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data: Buffer) => {
        stdout += data.toString();
      });

      python.stderr.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      python.on('close', (code: number | null) => {
        if (code === 0) {
          try {
            const molecules = JSON.parse(stdout);
            resolve(Ok(molecules));
          } catch (e) {
            resolve(Err({
              type: 'PARSE_FAILED',
              message: `Failed to parse SDF scan results: ${(e as Error).message}`,
            }));
          }
        } else {
          resolve(Err({
            type: 'PARSE_FAILED',
            message: `SDF scan failed: ${stderr}`,
          }));
        }
      });

      python.on('error', (error: Error) => {
        resolve(Err({
          type: 'PARSE_FAILED',
          message: error.message,
        }));
      });
    });
  }
);

// Parse SMILES CSV and generate 3D structures
ipcMain.handle(
  IpcChannels.PARSE_SMILES_CSV,
  async (event, csvPath: string, outputDir: string): Promise<Result<Array<{
    filename: string;
    smiles: string;
    qed: number;
    saScore: number;
    sdfPath: string;
  }>, AppError>> => {
    return new Promise((resolve) => {
      if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
        resolve(Err({
          type: 'PYTHON_NOT_FOUND',
          message: 'Python not found. Please install miniconda and create fraggen environment.',
        }));
        return;
      }

      const scriptPath = path.join(fraggenRoot, 'smiles_to_3d.py');
      if (!fs.existsSync(scriptPath)) {
        resolve(Err({
          type: 'SCRIPT_NOT_FOUND',
          path: scriptPath,
          message: `SMILES to 3D converter script not found: ${scriptPath}`,
        }));
        return;
      }

      fs.mkdirSync(outputDir, { recursive: true });

      const python = spawn(condaPythonPath, [
        scriptPath,
        '--input_csv', csvPath,
        '--output_dir', outputDir
      ]);
      childProcesses.add(python);

      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data: Buffer) => {
        const text = data.toString();
        stdout += text;
        // Forward progress to UI
        event.sender.send(IpcChannels.DOCK_OUTPUT, { type: 'stdout', data: text });
      });

      python.stderr.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      python.on('close', (code: number | null) => {
        childProcesses.delete(python);
        if (code === 0) {
          try {
            // JSON is always the last line of stdout (single-line json.dumps)
            const lines = stdout.trim().split('\n');
            const lastLine = lines[lines.length - 1];
            if (lastLine && (lastLine.startsWith('[') || lastLine.startsWith('{'))) {
              const molecules = JSON.parse(lastLine);
              resolve(Ok(molecules));
            } else {
              resolve(Err({
                type: 'PARSE_FAILED',
                message: 'No molecule data returned from SMILES conversion',
              }));
            }
          } catch (e) {
            resolve(Err({
              type: 'PARSE_FAILED',
              message: `Failed to parse SMILES conversion results: ${(e as Error).message}`,
            }));
          }
        } else {
          resolve(Err({
            type: 'PARSE_FAILED',
            message: `SMILES conversion failed: ${stderr}`,
          }));
        }
      });

      python.on('error', (error: Error) => {
        childProcesses.delete(python);
        resolve(Err({
          type: 'PARSE_FAILED',
          message: error.message,
        }));
      });
    });
  }
);

// Convert single SMILES or MOL file to 3D SDF + thumbnail
ipcMain.handle(
  IpcChannels.CONVERT_SINGLE_MOLECULE,
  async (_event, input: string, outputDir: string, inputType: 'smiles' | 'mol_file'): Promise<Result<any, AppError>> => {
    return new Promise((resolve) => {
      if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
        resolve(Err({ type: 'PYTHON_NOT_FOUND', message: 'Python not found' }));
        return;
      }

      const scriptPath = path.join(fraggenRoot, 'smiles_to_sdf.py');
      if (!fs.existsSync(scriptPath)) {
        resolve(Err({ type: 'SCRIPT_NOT_FOUND', path: scriptPath, message: `Script not found: ${scriptPath}` }));
        return;
      }

      fs.mkdirSync(outputDir, { recursive: true });

      const args = [scriptPath, '--output_dir', outputDir];
      if (inputType === 'smiles') {
        args.push('--smiles', input);
      } else {
        args.push('--mol_file', input);
      }

      const python = spawn(condaPythonPath, args);
      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data: Buffer) => { stdout += data.toString(); });
      python.stderr.on('data', (data: Buffer) => { stderr += data.toString(); });

      python.on('close', (code: number | null) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdout.trim());
            if (result.error) {
              resolve(Err({ type: 'PARSE_FAILED', message: result.error }));
            } else {
              resolve(Ok(result));
            }
          } catch (e) {
            resolve(Err({ type: 'PARSE_FAILED', message: `Failed to parse output: ${stdout}` }));
          }
        } else {
          resolve(Err({ type: 'PARSE_FAILED', message: stderr || 'Conversion failed' }));
        }
      });

      python.on('error', (error: Error) => {
        resolve(Err({ type: 'PARSE_FAILED', message: error.message }));
      });
    });
  }
);

// Extract X-ray ligand from PDB and convert to SDF
ipcMain.handle(
  IpcChannels.EXTRACT_XRAY_LIGAND,
  async (_event, pdbPath: string, ligandId: string, outputDir: string, smiles?: string): Promise<Result<any, AppError>> => {
    return new Promise((resolve) => {
      if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
        resolve(Err({ type: 'PYTHON_NOT_FOUND', message: 'Python not found' }));
        return;
      }

      const scriptPath = path.join(fraggenRoot, 'extract_xray_ligand.py');
      if (!fs.existsSync(scriptPath)) {
        resolve(Err({ type: 'SCRIPT_NOT_FOUND', path: scriptPath, message: `Script not found: ${scriptPath}` }));
        return;
      }

      fs.mkdirSync(outputDir, { recursive: true });

      const args = [scriptPath, '--pdb', pdbPath, '--ligand_id', ligandId, '--output_dir', outputDir];
      if (smiles) {
        args.push('--smiles', smiles);
      }

      const python = spawn(condaPythonPath, args);
      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data: Buffer) => { stdout += data.toString(); });
      python.stderr.on('data', (data: Buffer) => { stderr += data.toString(); });

      python.on('close', (code: number | null) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdout.trim());
            if (result.error && result.needsSmiles) {
              // Not a failure — just needs user to provide SMILES
              resolve(Ok(result));
            } else if (result.error) {
              resolve(Err({ type: 'PARSE_FAILED', message: result.error }));
            } else {
              resolve(Ok(result));
            }
          } catch (e) {
            resolve(Err({ type: 'PARSE_FAILED', message: `Failed to parse output: ${stdout}` }));
          }
        } else {
          resolve(Err({ type: 'PARSE_FAILED', message: stderr || 'Extraction failed' }));
        }
      });

      python.on('error', (error: Error) => {
        resolve(Err({ type: 'PARSE_FAILED', message: error.message }));
      });
    });
  }
);

// Enumerate protonation states using Dimorphite-DL
ipcMain.handle(
  IpcChannels.ENUMERATE_PROTONATION,
  async (
    event,
    ligandSdfPaths: string[],
    outputDir: string,
    phMin: number,
    phMax: number
  ): Promise<Result<{
    protonatedPaths: string[];
    parentMapping: Record<string, string>;
  }, AppError>> => {
    return new Promise((resolve) => {
      if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
        resolve(Err({
          type: 'PYTHON_NOT_FOUND',
          message: 'Python not found. Please install miniconda and create fraggen environment.',
        }));
        return;
      }

      const scriptPath = path.join(fraggenRoot, 'enumerate_protonation.py');
      if (!fs.existsSync(scriptPath)) {
        // Graceful degradation: if script not found, return original paths
        event.sender.send(IpcChannels.DOCK_OUTPUT, {
          type: 'stdout',
          data: 'Warning: enumerate_protonation.py not found, skipping protonation\n'
        });
        const parentMapping: Record<string, string> = {};
        for (const p of ligandSdfPaths) {
          const name = path.basename(p, '.sdf');
          parentMapping[name] = name;
        }
        resolve(Ok({
          protonatedPaths: ligandSdfPaths,
          parentMapping,
        }));
        return;
      }

      // Ensure output directory exists (caller provides the full path)
      fs.mkdirSync(outputDir, { recursive: true });

      // Write ligand list to JSON file for the script
      const ligandListPath = path.join(outputDir, 'ligand_list_for_protonation.json');
      fs.writeFileSync(ligandListPath, JSON.stringify(ligandSdfPaths, null, 2));

      const args = [
        scriptPath,
        '--ligand_list', ligandListPath,
        '--output_dir', outputDir,
        '--ph_min', String(phMin),
        '--ph_max', String(phMax),
      ];

      event.sender.send(IpcChannels.DOCK_OUTPUT, {
        type: 'stdout',
        data: `=== Protonation Enumeration ===\npH range: ${phMin}-${phMax}\nInput molecules: ${ligandSdfPaths.length}\n\n`
      });

      const python = spawn(condaPythonPath, args);
      childProcesses.add(python);

      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data: Buffer) => {
        const text = data.toString();
        stdout += text;
        event.sender.send(IpcChannels.DOCK_OUTPUT, { type: 'stdout', data: text });
      });

      python.stderr.on('data', (data: Buffer) => {
        const text = data.toString();
        stderr += text;
        // Only show warnings in output, not debug messages
        if (text.includes('Warning') || text.includes('ERROR')) {
          event.sender.send(IpcChannels.DOCK_OUTPUT, { type: 'stderr', data: text });
        }
      });

      python.on('close', (code: number | null) => {
        childProcesses.delete(python);
        if (code === 0) {
          try {
            // JSON is always the last line of stdout (single-line json.dumps)
            const lines = stdout.trim().split('\n');
            const lastLine = lines[lines.length - 1];
            if (lastLine && lastLine.startsWith('{')) {
              const result = JSON.parse(lastLine);
              resolve(Ok({
                protonatedPaths: result.protonated_paths || [],
                parentMapping: result.parent_mapping || {},
              }));
            } else {
              // No output - may mean Dimorphite-DL not installed, fallback to original
              event.sender.send(IpcChannels.DOCK_OUTPUT, {
                type: 'stdout',
                data: 'Warning: No protonation output, using original molecules\n'
              });
              const parentMapping: Record<string, string> = {};
              for (const p of ligandSdfPaths) {
                const name = path.basename(p, '.sdf');
                parentMapping[name] = name;
              }
              resolve(Ok({
                protonatedPaths: ligandSdfPaths,
                parentMapping,
              }));
            }
          } catch (e) {
            resolve(Err({
              type: 'PARSE_FAILED',
              message: `Failed to parse protonation results: ${(e as Error).message}`,
            }));
          }
        } else {
          // Non-zero exit - check if it's because Dimorphite-DL not installed
          if (stderr.includes('dimorphite_dl') || stderr.includes('ModuleNotFoundError')) {
            event.sender.send(IpcChannels.DOCK_OUTPUT, {
              type: 'stdout',
              data: 'Warning: Dimorphite-DL not installed, skipping protonation\n' +
                    'Install with: pip install dimorphite_dl\n\n'
            });
            const parentMapping: Record<string, string> = {};
            for (const p of ligandSdfPaths) {
              const name = path.basename(p, '.sdf');
              parentMapping[name] = name;
            }
            resolve(Ok({
              protonatedPaths: ligandSdfPaths,
              parentMapping,
            }));
          } else {
            resolve(Err({
              type: 'PROTONATION_FAILED',
              message: `Protonation failed: ${stderr.slice(0, 200)}`,
            }));
          }
        }
      });

      python.on('error', (error: Error) => {
        childProcesses.delete(python);
        resolve(Err({
          type: 'PROTONATION_FAILED',
          message: error.message,
        }));
      });
    });
  }
);

// Enumerate stereoisomers for unspecified stereocenters using RDKit
ipcMain.handle(
  IpcChannels.ENUMERATE_STEREOISOMERS,
  async (
    event,
    ligandSdfPaths: string[],
    outputDir: string,
    maxStereoisomers: number
  ): Promise<Result<{
    stereoisomerPaths: string[];
    parentMapping: Record<string, string>;
  }, AppError>> => {
    return new Promise((resolve) => {
      if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
        resolve(Err({
          type: 'PYTHON_NOT_FOUND',
          message: 'Python not found.',
        }));
        return;
      }

      const scriptPath = path.join(fraggenRoot, 'enumerate_stereoisomers.py');
      if (!fs.existsSync(scriptPath)) {
        event.sender.send(IpcChannels.DOCK_OUTPUT, {
          type: 'stdout',
          data: 'Warning: enumerate_stereoisomers.py not found, skipping\n'
        });
        const parentMapping: Record<string, string> = {};
        for (const p of ligandSdfPaths) {
          const name = path.basename(p, '.sdf');
          parentMapping[name] = name;
        }
        resolve(Ok({ stereoisomerPaths: ligandSdfPaths, parentMapping }));
        return;
      }

      fs.mkdirSync(outputDir, { recursive: true });

      const ligandListPath = path.join(outputDir, 'ligand_list_for_stereo.json');
      fs.writeFileSync(ligandListPath, JSON.stringify(ligandSdfPaths, null, 2));

      const args = [
        scriptPath,
        '--ligand_list', ligandListPath,
        '--output_dir', outputDir,
        '--max_stereoisomers', String(maxStereoisomers),
      ];

      event.sender.send(IpcChannels.DOCK_OUTPUT, {
        type: 'stdout',
        data: `=== Stereoisomer Enumeration ===\nMax per molecule: ${maxStereoisomers}\nInput molecules: ${ligandSdfPaths.length}\n\n`
      });

      const python = spawn(condaPythonPath, args);
      childProcesses.add(python);

      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data: Buffer) => {
        const text = data.toString();
        stdout += text;
        event.sender.send(IpcChannels.DOCK_OUTPUT, { type: 'stdout', data: text });
      });

      python.stderr.on('data', (data: Buffer) => {
        const text = data.toString();
        stderr += text;
        if (text.includes('Warning') || text.includes('ERROR')) {
          event.sender.send(IpcChannels.DOCK_OUTPUT, { type: 'stderr', data: text });
        }
      });

      python.on('close', (code: number | null) => {
        childProcesses.delete(python);
        if (code === 0) {
          try {
            const lines = stdout.trim().split('\n');
            const lastLine = lines[lines.length - 1];
            if (lastLine && lastLine.startsWith('{')) {
              const result = JSON.parse(lastLine);
              resolve(Ok({
                stereoisomerPaths: result.stereoisomer_paths || [],
                parentMapping: result.parent_mapping || {},
              }));
            } else {
              resolve(Ok({ stereoisomerPaths: ligandSdfPaths, parentMapping: {} }));
            }
          } catch (e) {
            resolve(Err({
              type: 'PARSE_FAILED',
              message: `Failed to parse stereoisomer results: ${(e as Error).message}`,
            }));
          }
        } else {
          resolve(Err({
            type: 'UNKNOWN',
            message: `Stereoisomer enumeration failed (exit ${code}): ${stderr.slice(0, 300)}`,
          }));
        }
      });
    });
  }
);

// Generate conformers using RDKit ETKDG or MCMM
ipcMain.handle(
  IpcChannels.GENERATE_CONFORMERS,
  async (
    event,
    ligandSdfPaths: string[],
    outputDir: string,
    maxConformers: number,
    rmsdCutoff: number,
    energyWindow: number,
    method?: string,
    mcmmOptions?: { steps: number; temperature: number; sampleAmides: boolean }
  ): Promise<Result<{
    conformerPaths: string[];
    parentMapping: Record<string, string>;
  }, AppError>> => {
    return new Promise((resolve) => {
      if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
        resolve(Err({
          type: 'PYTHON_NOT_FOUND',
          message: 'Python not found. Please install miniconda and create fraggen environment.',
        }));
        return;
      }

      const scriptPath = path.join(fraggenRoot, 'generate_conformers.py');
      if (!fs.existsSync(scriptPath)) {
        // Graceful degradation: if script not found, return original paths
        event.sender.send(IpcChannels.DOCK_OUTPUT, {
          type: 'stdout',
          data: 'Warning: generate_conformers.py not found, skipping conformer generation\n'
        });
        const parentMapping: Record<string, string> = {};
        for (const p of ligandSdfPaths) {
          const name = path.basename(p, '.sdf');
          parentMapping[name] = name;
        }
        resolve(Ok({
          conformerPaths: ligandSdfPaths,
          parentMapping,
        }));
        return;
      }

      // Use the caller-provided outputDir directly (no extra subdirectory)
      const conformerDir = outputDir;
      fs.mkdirSync(conformerDir, { recursive: true });

      // Write ligand list to JSON file for the script
      const ligandListPath = path.join(outputDir, 'ligand_list_for_conformers.json');
      fs.writeFileSync(ligandListPath, JSON.stringify(ligandSdfPaths, null, 2));

      const effectiveMethod = method || 'etkdg';
      const args = [
        scriptPath,
        '--ligand_list', ligandListPath,
        '--output_dir', conformerDir,
        '--max_conformers', String(maxConformers),
        '--rmsd_cutoff', String(rmsdCutoff),
        '--energy_window', String(energyWindow),
        '--method', effectiveMethod,
      ];

      if (effectiveMethod === 'mcmm' && mcmmOptions) {
        args.push('--mcmm_steps', String(mcmmOptions.steps));
        args.push('--mcmm_temperature', String(mcmmOptions.temperature));
        if (mcmmOptions.sampleAmides) {
          args.push('--sample_amides');
        }
      }

      const methodLabel = effectiveMethod.toUpperCase();
      event.sender.send(IpcChannels.DOCK_OUTPUT, {
        type: 'stdout',
        data: `=== Conformer Generation (${methodLabel}) ===\nMax conformers: ${maxConformers}\nRMSD cutoff: ${rmsdCutoff} A\nEnergy window: ${energyWindow} kcal/mol\nInput molecules: ${ligandSdfPaths.length}\n\n`
      });

      const python = spawn(condaPythonPath, args);
      childProcesses.add(python);

      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data: Buffer) => {
        const text = data.toString();
        stdout += text;
        event.sender.send(IpcChannels.DOCK_OUTPUT, { type: 'stdout', data: text });
      });

      python.stderr.on('data', (data: Buffer) => {
        const text = data.toString();
        stderr += text;
        // Only show warnings in output, not debug messages
        if (text.includes('Warning') || text.includes('ERROR')) {
          event.sender.send(IpcChannels.DOCK_OUTPUT, { type: 'stderr', data: text });
        }
      });

      python.on('close', (code: number | null) => {
        childProcesses.delete(python);
        if (code === 0) {
          try {
            // JSON is always the last line of stdout (single-line json.dumps)
            const lines = stdout.trim().split('\n');
            const lastLine = lines[lines.length - 1];
            if (lastLine && lastLine.startsWith('{')) {
              const result = JSON.parse(lastLine);
              resolve(Ok({
                conformerPaths: result.conformer_paths || [],
                parentMapping: result.parent_mapping || {},
              }));
            } else {
              // No output - fallback to original
              event.sender.send(IpcChannels.DOCK_OUTPUT, {
                type: 'stdout',
                data: 'Warning: No conformer output, using original molecules\n'
              });
              const parentMapping: Record<string, string> = {};
              for (const p of ligandSdfPaths) {
                const name = path.basename(p, '.sdf');
                parentMapping[name] = name;
              }
              resolve(Ok({
                conformerPaths: ligandSdfPaths,
                parentMapping,
              }));
            }
          } catch (e) {
            resolve(Err({
              type: 'PARSE_FAILED',
              message: `Failed to parse conformer results: ${(e as Error).message}`,
            }));
          }
        } else {
          resolve(Err({
            type: 'CONFORMER_FAILED',
            message: `Conformer generation failed: ${stderr.slice(0, 200)}`,
          }));
        }
      });

      python.on('error', (error: Error) => {
        childProcesses.delete(python);
        resolve(Err({
          type: 'CONFORMER_FAILED',
          message: error.message,
        }));
      });
    });
  }
);

// === Standalone Conformer Generation Handler ===
ipcMain.handle(
  'conform:generate',
  async (
    event,
    ligandSdfPath: string,
    outputDir: string,
    maxConformers: number,
    rmsdCutoff: number,
    energyWindow: number,
    method: string,
    mcmmOptions?: { steps: number; temperature: number; sampleAmides: boolean }
  ): Promise<Result<{
    conformerPaths: string[];
    parentMapping: Record<string, string>;
  }, AppError>> => {
    return new Promise((resolve) => {
      if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
        resolve(Err({
          type: 'PYTHON_NOT_FOUND',
          message: 'Python not found. Please install miniconda and create fraggen environment.',
        }));
        return;
      }

      const scriptPath = path.join(fraggenRoot, 'generate_conformers.py');
      if (!fs.existsSync(scriptPath)) {
        event.sender.send('conform:output', {
          type: 'stdout',
          data: 'Warning: generate_conformers.py not found, skipping conformer generation\n'
        });
        const name = path.basename(ligandSdfPath, '.sdf');
        resolve(Ok({
          conformerPaths: [ligandSdfPath],
          parentMapping: { [name]: name },
        }));
        return;
      }

      fs.mkdirSync(outputDir, { recursive: true });

      // Wrap single ligand path in array for the script
      const ligandSdfPaths = [ligandSdfPath];
      const ligandListPath = path.join(outputDir, 'ligand_list_for_conformers.json');
      fs.writeFileSync(ligandListPath, JSON.stringify(ligandSdfPaths, null, 2));

      const effectiveMethod = method || 'etkdg';
      const args = [
        scriptPath,
        '--ligand_list', ligandListPath,
        '--output_dir', outputDir,
        '--max_conformers', String(maxConformers),
        '--rmsd_cutoff', String(rmsdCutoff),
        '--energy_window', String(energyWindow),
        '--method', effectiveMethod,
      ];

      if (effectiveMethod === 'mcmm' && mcmmOptions) {
        args.push('--mcmm_steps', String(mcmmOptions.steps));
        args.push('--mcmm_temperature', String(mcmmOptions.temperature));
        if (mcmmOptions.sampleAmides) {
          args.push('--sample_amides');
        }
      }

      const methodLabel = effectiveMethod.toUpperCase();
      event.sender.send('conform:output', {
        type: 'stdout',
        data: `=== Conformer Generation (${methodLabel}) ===\nMax conformers: ${maxConformers}\nRMSD cutoff: ${rmsdCutoff} A\nEnergy window: ${energyWindow} kcal/mol\nInput: ${path.basename(ligandSdfPath)}\n\n`
      });

      const python = spawn(condaPythonPath, args);
      childProcesses.add(python);

      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data: Buffer) => {
        const text = data.toString();
        stdout += text;
        event.sender.send('conform:output', { type: 'stdout', data: text });
      });

      python.stderr.on('data', (data: Buffer) => {
        const text = data.toString();
        stderr += text;
        if (text.includes('Warning') || text.includes('ERROR')) {
          event.sender.send('conform:output', { type: 'stderr', data: text });
        }
      });

      python.on('close', (code: number | null) => {
        childProcesses.delete(python);
        if (code === 0) {
          try {
            const lines = stdout.trim().split('\n');
            const lastLine = lines[lines.length - 1];
            if (lastLine && lastLine.startsWith('{')) {
              const result = JSON.parse(lastLine);
              resolve(Ok({
                conformerPaths: result.conformer_paths || [],
                parentMapping: result.parent_mapping || {},
              }));
            } else {
              event.sender.send('conform:output', {
                type: 'stdout',
                data: 'Warning: No conformer output, using original molecule\n'
              });
              const name = path.basename(ligandSdfPath, '.sdf');
              resolve(Ok({
                conformerPaths: [ligandSdfPath],
                parentMapping: { [name]: name },
              }));
            }
          } catch (e) {
            resolve(Err({
              type: 'PARSE_FAILED',
              message: `Failed to parse conformer results: ${(e as Error).message}`,
            }));
          }
        } else {
          resolve(Err({
            type: 'CONFORMER_FAILED',
            message: `Conformer generation failed: ${stderr.slice(0, 200)}`,
          }));
        }
      });

      python.on('error', (error: Error) => {
        childProcesses.delete(python);
        resolve(Err({
          type: 'CONFORMER_FAILED',
          message: error.message,
        }));
      });
    });
  }
);

// === MD Simulation Handlers ===

interface MDConfig {
  productionNs: number;
  forceFieldPreset: string;
  compoundId?: string;
  temperatureK?: number;
  saltConcentrationM?: number;
  paddingNm?: number;
  restrainLigandNs?: number;
  seed?: number;
}

interface MDBenchmarkResult {
  nsPerDay: number;
  systemInfo: {
    atomCount: number;
    boxVolumeA3: number;
  };
}

interface MDLoadedLigand {
  name: string;
  sdfPath: string;
  smiles: string;
  vinaAffinity: number;
  qed: number;
  mw?: number;
  logp?: number;
  cordialPHighAffinity?: number;
  cordialExpectedPkd?: number;
  thumbnail?: string;
}

interface MDDockOutput {
  receptorPdb: string;
  ligands: MDLoadedLigand[];
}

// Helper function to parse SDF properties using Python script
function parseSdfProperties(sdfPath: string): Promise<{
  success: boolean;
  error?: string;
  smiles?: string;
  vinaAffinity: number;
  qed: number;
  mw: number;
  logp: number;
  thumbnail?: string;
  coreRmsd?: number;
}> {
  return new Promise((resolve) => {
    if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
      resolve({
        success: false,
        error: 'Python not found',
        vinaAffinity: 0,
        qed: 0,
        mw: 0,
        logp: 0,
      });
      return;
    }

    const scriptPath = path.join(fraggenRoot, 'parse_sdf_properties.py');
    if (!fs.existsSync(scriptPath)) {
      resolve({
        success: false,
        error: 'parse_sdf_properties.py not found',
        vinaAffinity: 0,
        qed: 0,
        mw: 0,
        logp: 0,
      });
      return;
    }

    const python = spawn(condaPythonPath, [scriptPath, '--sdf_file', sdfPath]);
    let stdout = '';
    let stderr = '';

    python.stdout.on('data', (data: Buffer) => {
      stdout += data.toString();
    });

    python.stderr.on('data', (data: Buffer) => {
      stderr += data.toString();
    });

    python.on('close', (code) => {
      if (code === 0 && stdout.trim()) {
        try {
          const result = JSON.parse(stdout.trim());
          resolve({
            success: result.success,
            error: result.error,
            smiles: result.smiles,
            vinaAffinity: result.vinaAffinity || result.minimizedAffinity || 0,
            qed: result.qed || 0,
            mw: result.mw || 0,
            logp: result.logp || 0,
            thumbnail: result.thumbnail,
            coreRmsd: result.coreRMSD != null ? parseFloat(result.coreRMSD) : undefined,
          });
        } catch (e) {
          resolve({
            success: false,
            error: 'Failed to parse JSON output',
            vinaAffinity: 0,
            qed: 0,
            mw: 0,
            logp: 0,
          });
        }
      } else {
        resolve({
          success: false,
          error: stderr || 'Script failed',
          vinaAffinity: 0,
          qed: 0,
          mw: 0,
          logp: 0,
        });
      }
    });

    python.on('error', (err) => {
      resolve({
        success: false,
        error: err.message,
        vinaAffinity: 0,
        qed: 0,
        mw: 0,
        logp: 0,
      });
    });
  });
}

// Load docking output directory for MD
ipcMain.handle(
  IpcChannels.LOAD_DOCK_OUTPUT_FOR_MD,
  async (_event, dirPath: string): Promise<Result<MDDockOutput, AppError>> => {
    try {
      // Check if directory exists
      if (!fs.existsSync(dirPath)) {
        return Err({
          type: 'DIRECTORY_ERROR',
          path: dirPath,
          message: 'Directory does not exist',
        });
      }

      // Look for receptor: inputs/receptor.pdb (new), *_receptor_prepared.pdb (legacy)
      const files0 = fs.readdirSync(dirPath);
      const prefixedReceptor = files0.find((f) => f.endsWith('_receptor_prepared.pdb'));

      const receptorCandidates = [
        // New layout: inputs/receptor.pdb
        path.join(dirPath, 'inputs', 'receptor.pdb'),
        // Legacy: {projectName}_receptor_prepared.pdb in docking dir
        ...(prefixedReceptor ? [path.join(dirPath, prefixedReceptor)] : []),
        // Very old legacy: receptor_prepared.pdb in docking dir or parent dirs
        path.join(dirPath, 'receptor_prepared.pdb'),
        path.join(path.dirname(dirPath), 'receptor_prepared.pdb'),
        path.join(path.dirname(path.dirname(dirPath)), 'receptor_prepared.pdb'),
      ];

      let receptorPdb = '';
      for (const candidate of receptorCandidates) {
        if (fs.existsSync(candidate)) {
          receptorPdb = candidate;
          break;
        }
      }

      if (!receptorPdb) {
        return Err({
          type: 'FILE_NOT_FOUND',
          path: path.join(dirPath, 'receptor_prepared.pdb'),
          message: 'receptor_prepared.pdb not found in docking output directory or parent directories. Please ensure you ran docking with a prepared receptor.',
        });
      }

      // Find all *_docked.sdf.gz or *.sdf files (check results/poses/ first, then poses/, then top-level)
      const newPosesSubDir = path.join(dirPath, 'results', 'poses');
      const legacyPosesSubDir = path.join(dirPath, 'poses');
      const sdfSearchDir = fs.existsSync(newPosesSubDir) ? newPosesSubDir : fs.existsSync(legacyPosesSubDir) ? legacyPosesSubDir : dirPath;
      const files = fs.readdirSync(sdfSearchDir);
      const dockedFiles = files.filter((f) => f.endsWith('_docked.sdf.gz'));
      // Also check top-level for regular SDF files (legacy or manual)
      const topFiles = sdfSearchDir !== dirPath ? fs.readdirSync(dirPath) : files;
      const regularSdfFiles = topFiles.filter((f) => f.endsWith('.sdf') && !f.endsWith('.sdf.gz') && !f.includes('_all_docked'));

      // Use docked files if available, otherwise fall back to regular SDF files
      const sdfFiles = dockedFiles.length > 0 ? dockedFiles : regularSdfFiles;

      if (sdfFiles.length === 0) {
        return Err({
          type: 'FILE_NOT_FOUND',
          path: dirPath,
          message: 'No SDF files found in directory. Expected *_docked.sdf.gz or *.sdf files.',
        });
      }

      // Look for parent FragGen directory for legacy thumbnails
      let thumbnailDir = '';
      const parentDir = path.dirname(dirPath);
      const grandparentDir = path.dirname(parentDir);

      const thumbnailCandidates = [
        path.join(parentDir, 'thumbnails'),
        path.join(grandparentDir, 'thumbnails'),
        path.join(parentDir, '..', 'thumbnails'),
        path.join(grandparentDir, '..', 'thumbnails'),
      ];

      for (const candidate of thumbnailCandidates) {
        if (fs.existsSync(candidate)) {
          thumbnailDir = candidate;
          break;
        }
      }

      // Parse each SDF file directly for properties
      // Process in parallel for better performance
      const ligands: MDLoadedLigand[] = [];
      const parsePromises = sdfFiles.map(async (sdfFile) => {
        const isDockedFile = sdfFile.endsWith('_docked.sdf.gz');
        const name = isDockedFile
          ? sdfFile.replace('_docked.sdf.gz', '')
          : sdfFile.replace('.sdf', '');
        const sdfPath = path.join(isDockedFile ? sdfSearchDir : dirPath, sdfFile);

        // Parse SDF for all properties (SMILES, scores, QED, thumbnail)
        const props = await parseSdfProperties(sdfPath);

        // Check for legacy thumbnail file
        let thumbnailPath: string | undefined;
        if (thumbnailDir) {
          const thumbFile = path.join(thumbnailDir, `${name}.png`);
          if (fs.existsSync(thumbFile)) {
            thumbnailPath = thumbFile;
          }
        }

        return {
          name,
          sdfPath,
          smiles: props.smiles || '',
          vinaAffinity: props.vinaAffinity,
          qed: props.qed,
          mw: props.mw,
          logp: props.logp,
          thumbnail: props.thumbnail,
        };
      });

      const parsedLigands = await Promise.all(parsePromises);
      ligands.push(...parsedLigands);

      // Load CORDIAL scores if available (check results/ first, then top-level)
      const newCordialJsonPath = path.join(dirPath, 'results', 'cordial_scores.json');
      const cordialJsonPath = fs.existsSync(newCordialJsonPath) ? newCordialJsonPath : path.join(dirPath, 'cordial_scores.json');
      if (fs.existsSync(cordialJsonPath)) {
        try {
          const cordialData = JSON.parse(fs.readFileSync(cordialJsonPath, 'utf-8'));
          // cordialData is array of { source_name, pose_index, cordial_expected_pkd, cordial_p_high_affinity, ... }
          // Group by source_name and get best score (pose_index 0 is usually best)
          const cordialByName = new Map<string, { pHighAffinity: number; expectedPkd: number }>();
          for (const entry of cordialData) {
            const name = entry.source_name;
            const pHighAffinity = entry.cordial_p_high_affinity;
            const expectedPkd = entry.cordial_expected_pkd;
            // Keep the best score (highest pHighAffinity) for each ligand
            const existing = cordialByName.get(name);
            if (!existing || pHighAffinity > existing.pHighAffinity) {
              cordialByName.set(name, { pHighAffinity, expectedPkd });
            }
          }
          // Merge into ligands
          for (const ligand of ligands) {
            const cordialScores = cordialByName.get(ligand.name);
            if (cordialScores) {
              ligand.cordialPHighAffinity = cordialScores.pHighAffinity;
              ligand.cordialExpectedPkd = cordialScores.expectedPkd;
            }
          }
        } catch (e) {
          console.error('Failed to load CORDIAL scores:', e);
        }
      }

      // Sort by vinaAffinity ascending (most negative = best)
      ligands.sort((a, b) => a.vinaAffinity - b.vinaAffinity);

      return Ok({
        receptorPdb,
        ligands,
      });
    } catch (error) {
      return Err({
        type: 'PARSE_FAILED',
        message: (error as Error).message,
      });
    }
  }
);

// Run MD benchmark to estimate performance
ipcMain.handle(
  IpcChannels.RUN_MD_BENCHMARK,
  async (
    event,
    receptorPdb: string | null,
    ligandSdf: string,
    outputDir: string,
    forceFieldPreset: string = 'ff19sb-opc',
    ligandOnly: boolean = false
  ): Promise<Result<MDBenchmarkResult, AppError>> => {
    return new Promise((resolve) => {
      if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
        resolve(Err({
          type: 'PYTHON_NOT_FOUND',
          message: 'Python not found. Please install miniconda and create fraggen environment.',
        }));
        return;
      }

      const scriptPath = path.join(fraggenRoot, 'run_md_simulation.py');
      if (!fs.existsSync(scriptPath)) {
        resolve(Err({
          type: 'SCRIPT_NOT_FOUND',
          path: scriptPath,
          message: `MD simulation script not found: ${scriptPath}`,
        }));
        return;
      }

      fs.mkdirSync(outputDir, { recursive: true });

      const args = [scriptPath];

      if (!ligandOnly && receptorPdb) {
        args.push('--receptor', receptorPdb);
      }

      args.push(
        '--ligand', ligandSdf,
        '--output_dir', outputDir,
        '--force_field_preset', forceFieldPreset || 'ff19sb-opc',
        '--benchmark_only',
      );

      if (ligandOnly) {
        args.push('--ligand_only');
      }

      // Kill any previous benchmark still running
      if (currentBenchmarkProcess && !currentBenchmarkProcess.killed) {
        currentBenchmarkProcess.kill('SIGTERM');
      }

      const python = spawn(condaPythonPath, args);
      childProcesses.add(python);
      currentBenchmarkProcess = python;

      let stdout = '';
      let systemInfo = { atomCount: 0, boxVolumeA3: 0 };
      let nsPerDay = 0;

      python.stdout.on('data', (data: Buffer) => {
        const text = data.toString();
        stdout += text;
        event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stdout', data: text });

        // Parse SYSTEM_INFO:atomCount:volume
        const systemMatch = text.match(/SYSTEM_INFO:(\d+):(\d+)/);
        if (systemMatch) {
          systemInfo.atomCount = parseInt(systemMatch[1]);
          systemInfo.boxVolumeA3 = parseInt(systemMatch[2]);
        }

        // Parse BENCHMARK:nsPerDay
        const benchmarkMatch = text.match(/BENCHMARK:([\d.]+)/);
        if (benchmarkMatch) {
          nsPerDay = parseFloat(benchmarkMatch[1]);
        }
      });

      python.stderr.on('data', (data: Buffer) => {
        const filtered = filterMdStderr(data.toString());
        if (filtered.trim()) {
          event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stderr', data: filtered });
        }
      });

      python.on('close', (code: number | null) => {
        childProcesses.delete(python);
        if (currentBenchmarkProcess === python) currentBenchmarkProcess = null;
        if (code === 0 && nsPerDay > 0) {
          resolve(Ok({
            nsPerDay,
            systemInfo,
          }));
        } else {
          resolve(Err({
            type: 'BENCHMARK_FAILED',
            message: `Benchmark failed with code ${code}`,
          }));
        }
      });

      python.on('error', (error: Error) => {
        childProcesses.delete(python);
        if (currentBenchmarkProcess === python) currentBenchmarkProcess = null;
        resolve(Err({
          type: 'BENCHMARK_FAILED',
          message: error.message,
        }));
      });
    });
  }
);

let currentBenchmarkProcess: ChildProcess | null = null;
let currentMdProcess: ChildProcess | null = null;

// Run full MD simulation
ipcMain.handle(
  IpcChannels.RUN_MD_SIMULATION,
  async (
    event,
    receptorPdb: string | null,
    ligandSdf: string,
    outputDir: string,
    config: MDConfig,
    ligandOnly: boolean = false
  ): Promise<Result<string, AppError>> => {
    return new Promise((resolve) => {
      if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
        resolve(Err({
          type: 'PYTHON_NOT_FOUND',
          message: 'Python not found. Please install miniconda and create fraggen environment.',
        }));
        return;
      }

      const scriptPath = path.join(fraggenRoot, 'run_md_simulation.py');
      if (!fs.existsSync(scriptPath)) {
        resolve(Err({
          type: 'SCRIPT_NOT_FOUND',
          path: scriptPath,
          message: `MD simulation script not found: ${scriptPath}`,
        }));
        return;
      }

      fs.mkdirSync(outputDir, { recursive: true });

      const args = [
        scriptPath,
        '--ligand', ligandSdf,
        '--output_dir', outputDir,
        '--production_ns', String(config.productionNs),
        '--force_field_preset', config.forceFieldPreset || 'ff19sb-opc',
        '--temperature', String(config.temperatureK || 300),
        '--salt_concentration', String(config.saltConcentrationM || 0.15),
        '--padding', String(config.paddingNm || 1.2),
      ];

      if (config.restrainLigandNs && config.restrainLigandNs > 0) {
        args.push('--restrain_ligand_ns', String(config.restrainLigandNs));
      }

      if (config.seed && config.seed > 0) {
        args.push('--seed', String(config.seed));
      }

      if (ligandOnly) {
        args.push('--ligand_only');
      } else if (receptorPdb) {
        args.push('--receptor', receptorPdb);
      }

      // Kill any running benchmark before starting simulation
      if (currentBenchmarkProcess && !currentBenchmarkProcess.killed) {
        currentBenchmarkProcess.kill('SIGTERM');
        currentBenchmarkProcess = null;
      }

      console.log('Running MD simulation:', condaPythonPath, args.join(' '));

      const python = spawn(condaPythonPath, args);
      childProcesses.add(python);
      currentMdProcess = python;

      let trajectoryPath = '';

      python.stdout.on('data', (data: Buffer) => {
        const text = data.toString();
        event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stdout', data: text });

        // Parse SUCCESS:path
        const successMatch = text.match(/SUCCESS:(.+)/);
        if (successMatch) {
          trajectoryPath = successMatch[1].trim();
        }
      });

      python.stderr.on('data', (data: Buffer) => {
        const filtered = filterMdStderr(data.toString());
        if (filtered.trim()) {
          event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stderr', data: filtered });
        }
      });

      python.on('close', (code: number | null) => {
        childProcesses.delete(python);
        currentMdProcess = null;
        if (code === 0 && trajectoryPath) {
          resolve(Ok(trajectoryPath));
        } else {
          resolve(Err({
            type: 'SIMULATION_FAILED',
            message: `Simulation failed with code ${code}`,
          }));
        }
      });

      python.on('error', (error: Error) => {
        childProcesses.delete(python);
        currentMdProcess = null;
        resolve(Err({
          type: 'SIMULATION_FAILED',
          message: error.message,
        }));
      });
    });
  }
);

// Cancel running MD simulation
// Cancel running benchmark
ipcMain.handle(IpcChannels.CANCEL_MD_BENCHMARK, async (): Promise<void> => {
  if (currentBenchmarkProcess && !currentBenchmarkProcess.killed) {
    currentBenchmarkProcess.kill('SIGTERM');
    currentBenchmarkProcess = null;
  }
});

ipcMain.handle(IpcChannels.CANCEL_MD_SIMULATION, async (): Promise<void> => {
  if (currentMdProcess && !currentMdProcess.killed) {
    currentMdProcess.kill('SIGTERM');
    currentMdProcess = null;
  }
});

// Pause running MD simulation (SIGSTOP)
ipcMain.handle(IpcChannels.PAUSE_MD_SIMULATION, async (): Promise<void> => {
  if (currentMdProcess && !currentMdProcess.killed) {
    currentMdProcess.kill('SIGSTOP');
  }
});

// Resume paused MD simulation (SIGCONT)
ipcMain.handle(IpcChannels.RESUME_MD_SIMULATION, async (): Promise<void> => {
  if (currentMdProcess && !currentMdProcess.killed) {
    currentMdProcess.kill('SIGCONT');
  }
});

// Get the default output directory (~/Ember)
ipcMain.handle('get-default-output-dir', async (): Promise<string> => {
  const emberDir = path.join(app.getPath('home'), 'Ember');
  if (!fs.existsSync(emberDir)) {
    fs.mkdirSync(emberDir, { recursive: true });
  }
  return emberDir;
});

// Ensure a project directory exists with a .ember-project ID file
ipcMain.handle(
  IpcChannels.ENSURE_PROJECT,
  async (_event: any, projectName: string): Promise<Result<string, AppError>> => {
    try {
      const emberDir = path.join(app.getPath('home'), 'Ember');
      const projectDir = path.join(emberDir, projectName);
      const idFile = path.join(projectDir, '.ember-project');

      fs.mkdirSync(projectDir, { recursive: true });

      if (!fs.existsSync(idFile)) {
        fs.writeFileSync(idFile, JSON.stringify({
          name: projectName,
          created: new Date().toISOString(),
        }, null, 2));
      }

      return Ok(projectDir);
    } catch (err: any) {
      return Err({
        type: 'DIRECTORY_ERROR',
        path: path.join(app.getPath('home'), 'Ember', projectName),
        message: `Failed to create project: ${err.message}`,
      });
    }
  }
);

// Import a PDB/CIF into a project's structures/ directory (was raw/)
ipcMain.handle(
  IpcChannels.IMPORT_STRUCTURE,
  async (_event, sourcePath: string, projectDir: string): Promise<Result<string, AppError>> => {
    try {
      const structuresDir = path.join(projectDir, 'structures');
      fs.mkdirSync(structuresDir, { recursive: true });
      const filename = path.basename(sourcePath);
      const destPath = path.join(structuresDir, filename);
      // Don't re-copy if already in the project
      if (path.resolve(sourcePath) !== path.resolve(destPath)) {
        fs.copyFileSync(sourcePath, destPath);
      }
      return Ok(destPath);
    } catch (error) {
      return Err({
        type: 'IMPORT_FAILED',
        message: `Failed to import structure: ${(error as Error).message}`,
      });
    }
  }
);

// Prepare a PDB for viewing: add missing hydrogens via PDBFixer
ipcMain.handle(
  IpcChannels.PREPARE_FOR_VIEWING,
  async (_event, rawPdbPath: string, preparedPath: string): Promise<Result<string, AppError>> => {
    return new Promise((resolve) => {
      if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
        // No Python — just use raw file
        resolve(Ok(rawPdbPath));
        return;
      }

      const scriptPath = path.join(fraggenRoot, 'detect_pdb_ligands.py');
      if (!fs.existsSync(scriptPath)) {
        resolve(Ok(rawPdbPath));
        return;
      }

      fs.mkdirSync(path.dirname(preparedPath), { recursive: true });

      const python = spawn(condaPythonPath, [
        scriptPath,
        '--pdb', rawPdbPath,
        '--mode', 'add_hydrogens',
        '--output', preparedPath,
      ]);

      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data: Buffer) => {
        stdout += data.toString();
      });

      python.stderr.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      python.on('close', (code: number | null) => {
        if (code === 0 && fs.existsSync(preparedPath)) {
          resolve(Ok(preparedPath));
        } else {
          // Fall back to raw file on any failure
          console.error('[prepare-for-viewing] Failed:', stderr);
          resolve(Ok(rawPdbPath));
        }
      });

      python.on('error', () => {
        resolve(Ok(rawPdbPath));
      });
    });
  }
);

// Project browser: scan ~/Ember for projects and runs
// Primary: .ember-project ID file → project detected regardless of run state
// Secondary: scan simulations/ + docking/ for runs
// Tertiary: legacy fallback for old directories without .ember-project
ipcMain.handle(IpcChannels.SCAN_PROJECTS, async (): Promise<any[]> => {
  const emberDir = path.join(app.getPath('home'), 'Ember');
  if (!fs.existsSync(emberDir)) return [];

  const projects: any[] = [];
  const legacyPattern = /^(.+?)_(ff14sb-TIP3P|ff19sb-OPC|ff19sb-OPC3|charmm36-mTIP3P)_MD-/;

  try {
    const entries = fs.readdirSync(emberDir, { withFileTypes: true });
    const legacyGroups: Record<string, any[]> = {};

    // Helper: check a directory for simulation or docking output files and return run info
    const scanRunDir = (runPath: string, folderName: string): any | null => {
      try {
        const runFiles = fs.readdirSync(runPath);
        // Check results/ subdir for new layout
        const resultsPath = path.join(runPath, 'results');
        const resultsFiles = fs.existsSync(resultsPath) ? fs.readdirSync(resultsPath) : [];
        // New layout: unprefixed files in results/ or top-level; Legacy: prefixed files
        const hasSimOutput = runFiles.some((f: string) => f.endsWith('_system.pdb') || f.endsWith('_trajectory.dcd') || f === 'simulation.log' || f === 'system.pdb' || f === 'trajectory.dcd')
          || resultsFiles.some((f: string) => f === 'system.pdb' || f === 'trajectory.dcd');
        // New layout: docked files in results/poses/; Legacy: in poses/ or top-level
        const newPosesPath = path.join(runPath, 'results', 'poses');
        const legacyPosesPath = path.join(runPath, 'poses');
        const hasDockOutput = runFiles.some((f: string) => f.endsWith('_docked.sdf.gz') || f.endsWith('_docked.sdf'))
          || (fs.existsSync(newPosesPath) && fs.readdirSync(newPosesPath).some((f: string) => f.endsWith('_docked.sdf.gz')))
          || (fs.existsSync(legacyPosesPath) && fs.readdirSync(legacyPosesPath).some((f: string) => f.endsWith('_docked.sdf.gz')));
        // New layout: inputs/ dir with receptor.pdb indicates a docking run even without results yet
        const hasInputs = fs.existsSync(path.join(runPath, 'inputs', 'receptor.pdb'));
        if (!hasSimOutput && !hasDockOutput && !hasInputs) return null;
        const stat = fs.statSync(runPath);
        return {
          folderName,
          path: runPath,
          lastModified: stat.mtimeMs,
          type: hasSimOutput ? 'simulation' : 'docking',
          hasTrajectory: runFiles.some((f: string) => f.endsWith('_trajectory.dcd') || f === 'trajectory.dcd')
            || resultsFiles.some((f: string) => f === 'trajectory.dcd'),
          hasFinalPdb: runFiles.some((f: string) => f.endsWith('_final.pdb') || f === 'final.pdb')
            || resultsFiles.some((f: string) => f === 'final.pdb'),
        };
      } catch { return null; }
    };

    // Scan runs inside a project directory (simulations/, docking/, or legacy direct children)
    const scanProjectRuns = (entryPath: string): any[] => {
      const runs: any[] = [];
      try {
        const subEntries = fs.readdirSync(entryPath, { withFileTypes: true });
        for (const sub of subEntries) {
          if (!sub.isDirectory() || sub.name.startsWith('.')) continue;

          // New layout: runs live under simulations/ and docking/ subdirectories
          if (sub.name === 'simulations' || sub.name === 'docking') {
            const typeDir = path.join(entryPath, sub.name);
            try {
              const typeEntries = fs.readdirSync(typeDir, { withFileTypes: true });
              for (const runEntry of typeEntries) {
                if (!runEntry.isDirectory() || runEntry.name.startsWith('.')) continue;
                const runInfo = scanRunDir(path.join(typeDir, runEntry.name), `${sub.name}/${runEntry.name}`);
                if (runInfo) runs.push(runInfo);
              }
            } catch { /* skip unreadable */ }
            continue;
          }

          // Legacy layout: runs directly under project dir
          const runInfo = scanRunDir(path.join(entryPath, sub.name), sub.name);
          if (runInfo) runs.push(runInfo);
        }
      } catch { /* skip unreadable */ }
      return runs;
    };

    for (const entry of entries) {
      if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
      const entryPath = path.join(emberDir, entry.name);

      // Primary: .ember-project ID file → project always detected
      const idFile = path.join(entryPath, '.ember-project');
      if (fs.existsSync(idFile)) {
        const runs = scanProjectRuns(entryPath);
        runs.sort((a: any, b: any) => b.lastModified - a.lastModified);
        const stat = fs.statSync(entryPath);
        projects.push({
          name: entry.name,
          path: entryPath,
          runs,
          lastModified: runs.length > 0 ? runs[0].lastModified : stat.mtimeMs,
        });
        continue;
      }

      // Fallback: detect by run output (for projects created before .ember-project)
      const runs = scanProjectRuns(entryPath);
      if (runs.length > 0) {
        runs.sort((a: any, b: any) => b.lastModified - a.lastModified);
        projects.push({
          name: entry.name,
          path: entryPath,
          runs,
          lastModified: runs[0].lastModified,
        });
      } else {
        // Legacy grouped layout: folder name matches ff pattern
        const match = entry.name.match(legacyPattern);
        if (match) {
          const projectName = match[1];
          const files = fs.readdirSync(entryPath);
          const hasSimOutput = files.some((f: string) => f.endsWith('_system.pdb') || f.endsWith('_trajectory.dcd') || f === 'simulation.log' || f === 'system.pdb' || f === 'trajectory.dcd');
          if (hasSimOutput) {
            if (!legacyGroups[projectName]) legacyGroups[projectName] = [];
            const stat = fs.statSync(entryPath);
            legacyGroups[projectName].push({
              folderName: entry.name,
              path: entryPath,
              lastModified: stat.mtimeMs,
              type: 'simulation',
              hasTrajectory: files.some((f: string) => f.endsWith('_trajectory.dcd') || f === 'trajectory.dcd'),
              hasFinalPdb: files.some((f: string) => f.endsWith('_final.pdb') || f === 'final.pdb'),
            });
          }
        }
      }
    }

    for (const [name, runs] of Object.entries(legacyGroups)) {
      runs.sort((a: any, b: any) => b.lastModified - a.lastModified);
      projects.push({
        name,
        path: path.join(emberDir, name),
        runs,
        lastModified: runs[0].lastModified,
      });
    }

    projects.sort((a, b) => b.lastModified - a.lastModified);
  } catch (err) {
    console.error('Error scanning projects:', err);
  }
  return projects;
});

// Project browser: scan a run directory for output files
ipcMain.handle(IpcChannels.SCAN_RUN_FILES, async (_event: any, runDir: string): Promise<any> => {
  const result: any = {
    systemPdb: null,
    trajectory: null,
    finalPdb: null,
    equilibratedPdb: null,
    energyCsv: null,
  };
  try {
    if (!fs.existsSync(runDir)) return result;
    const files = fs.readdirSync(runDir);

    // Check results/ subdir first (new layout)
    const resultsDir = path.join(runDir, 'results');
    const resultsFiles = fs.existsSync(resultsDir) ? fs.readdirSync(resultsDir) : [];

    // New layout: unprefixed files in results/
    for (const f of resultsFiles) {
      if (f === 'system.pdb' && !result.systemPdb) result.systemPdb = path.join(resultsDir, f);
      else if (f === 'trajectory.dcd' && !result.trajectory) result.trajectory = path.join(resultsDir, f);
      else if (f === 'final.pdb' && !result.finalPdb) result.finalPdb = path.join(resultsDir, f);
      else if (f === 'equilibrated.pdb' && !result.equilibratedPdb) result.equilibratedPdb = path.join(resultsDir, f);
      else if (f === 'energy.csv' && !result.energyCsv) result.energyCsv = path.join(resultsDir, f);
    }

    // Legacy: prefixed files at top level; also check unprefixed at top level
    for (const f of files) {
      if ((f.endsWith('_system.pdb') || f === 'system.pdb') && !result.systemPdb) result.systemPdb = path.join(runDir, f);
      else if ((f.endsWith('_trajectory.dcd') || f === 'trajectory.dcd') && !result.trajectory) result.trajectory = path.join(runDir, f);
      else if ((f.endsWith('_final.pdb') || f === 'final.pdb') && !result.finalPdb) result.finalPdb = path.join(runDir, f);
      else if ((f.endsWith('_equilibrated.pdb') || f === 'equilibrated.pdb') && !result.equilibratedPdb) result.equilibratedPdb = path.join(runDir, f);
      else if ((f.endsWith('_energy.csv') || f === 'energy.csv') && !result.energyCsv) result.energyCsv = path.join(runDir, f);
    }
  } catch (err) {
    console.error('Error scanning run files:', err);
  }
  return result;
});

// Get file count and total size for a project (for delete confirmation)
ipcMain.handle(IpcChannels.GET_PROJECT_FILE_COUNT, async (_event: any, projectName: string): Promise<{ fileCount: number; totalSizeMb: number }> => {
  const emberDir = path.join(app.getPath('home'), 'Ember');
  const projectDir = path.join(emberDir, projectName);
  if (!fs.existsSync(projectDir)) return { fileCount: 0, totalSizeMb: 0 };

  let fileCount = 0;
  let totalSize = 0;
  const walk = (dir: string) => {
    try {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          walk(full);
        } else {
          fileCount++;
          try { totalSize += fs.statSync(full).size; } catch { /* skip */ }
        }
      }
    } catch { /* skip unreadable */ }
  };
  walk(projectDir);
  return { fileCount, totalSizeMb: Math.round(totalSize / (1024 * 1024) * 10) / 10 };
});

// Rename a project directory
ipcMain.handle(IpcChannels.RENAME_PROJECT, async (_event: any, oldName: string, newName: string): Promise<any> => {
  const emberDir = path.join(app.getPath('home'), 'Ember');
  const oldDir = path.join(emberDir, oldName);
  const newDir = path.join(emberDir, newName);

  if (!fs.existsSync(oldDir)) {
    return { ok: false, error: { type: 'NOT_FOUND', message: `Project "${oldName}" not found` } };
  }
  if (fs.existsSync(newDir)) {
    return { ok: false, error: { type: 'ALREADY_EXISTS', message: `Project "${newName}" already exists` } };
  }

  try {
    // Rename the project directory
    fs.renameSync(oldDir, newDir);

    // Rename prefixed files inside the project (e.g., projectName_system.pdb → newName_system.pdb)
    const renamePrefix = (dir: string) => {
      try {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
          const full = path.join(dir, entry.name);
          if (entry.isDirectory()) {
            renamePrefix(full);
          } else if (entry.name.startsWith(oldName + '_')) {
            const newFileName = newName + entry.name.substring(oldName.length);
            fs.renameSync(full, path.join(dir, newFileName));
          }
        }
      } catch { /* skip unreadable */ }
    };
    renamePrefix(newDir);

    // Update .ember-project ID file with new name
    const idFile = path.join(newDir, '.ember-project');
    if (fs.existsSync(idFile)) {
      try {
        const idData = JSON.parse(fs.readFileSync(idFile, 'utf-8'));
        idData.name = newName;
        fs.writeFileSync(idFile, JSON.stringify(idData, null, 2));
      } catch { /* non-critical — project still works without updated ID */ }
    }

    return { ok: true, value: undefined };
  } catch (err: any) {
    return { ok: false, error: { type: 'RENAME_FAILED', message: err.message || 'Failed to rename project' } };
  }
});

// Delete a project directory entirely
ipcMain.handle(IpcChannels.DELETE_PROJECT, async (_event: any, projectName: string): Promise<any> => {
  const emberDir = path.join(app.getPath('home'), 'Ember');
  const projectDir = path.join(emberDir, projectName);

  if (!fs.existsSync(projectDir)) {
    return { ok: false, error: { type: 'NOT_FOUND', message: `Project "${projectName}" not found` } };
  }

  try {
    fs.rmSync(projectDir, { recursive: true, force: true });
    return { ok: true, value: undefined };
  } catch (err: any) {
    return { ok: false, error: { type: 'DELETE_FAILED', message: err.message || 'Failed to delete project' } };
  }
});

// Shared helper: extract Vina affinity from first model of an .sdf.gz file
const extractVinaAffinity = (sdfGzPath: string): number | undefined => {
  try {
    const gzData = fs.readFileSync(sdfGzPath);
    const text = zlib.gunzipSync(gzData).toString('utf-8');
    const match = text.match(/>  <minimizedAffinity>\s*\n([\-\d.]+)/);
    if (match) return parseFloat(match[1]);
  } catch { /* ignore */ }
  return undefined;
};

// Scan project artifacts as ProjectJob[] — docking runs and simulation runs with their outputs
ipcMain.handle(
  IpcChannels.SCAN_PROJECT_ARTIFACTS,
  async (_event: any, projectName: string): Promise<any[]> => {
    const emberDir = path.join(app.getPath('home'), 'Ember');
    const projectDir = path.join(emberDir, projectName);
    if (!fs.existsSync(projectDir)) return [];

    const jobs: any[] = [];

    // Strip project name prefix from filenames for cleaner labels
    const stripPrefix = (name: string) => {
      if (name.startsWith(projectName + '_')) return name.substring(projectName.length + 1);
      return name;
    };

    // 1. Scan prepared/ for receptor PDBs (still returned as type: 'prepared')
    const preparedDir = path.join(projectDir, 'prepared');
    if (fs.existsSync(preparedDir)) {
      try {
        const prepFiles = fs.readdirSync(preparedDir).filter((f) => f.endsWith('.pdb'));
        for (const f of prepFiles) {
          jobs.push({
            type: 'prepared',
            label: stripPrefix(f.replace('.pdb', '')),
            path: path.join(preparedDir, f),
          });
        }
      } catch { /* skip */ }
    }

    // 2. Scan docking/ for docking run folders
    const dockingDir = path.join(projectDir, 'docking');
    if (fs.existsSync(dockingDir)) {
      try {
        const dockRuns = fs.readdirSync(dockingDir, { withFileTypes: true });
        for (const run of dockRuns) {
          if (!run.isDirectory() || run.name.startsWith('.')) continue;
          const runPath = path.join(dockingDir, run.name);
          const runFiles = fs.readdirSync(runPath);

          // Find receptor PDB: inputs/receptor.pdb (new) first, then *_receptor_prepared.pdb (legacy)
          let receptorPdb: string | undefined;
          const newReceptor = path.join(runPath, 'inputs', 'receptor.pdb');
          if (fs.existsSync(newReceptor)) {
            receptorPdb = newReceptor;
          } else {
            const legacyReceptorFile = runFiles.find((f) => f.includes('_receptor_prepared') && f.endsWith('.pdb'));
            if (legacyReceptorFile) receptorPdb = path.join(runPath, legacyReceptorFile);
          }

          // Find docked files: results/poses/ (new) first, then poses/ (legacy), then top-level
          const newPosesDir = path.join(runPath, 'results', 'poses');
          const legacyPosesDir = path.join(runPath, 'poses');
          const posesSearchDir = fs.existsSync(newPosesDir) ? newPosesDir : fs.existsSync(legacyPosesDir) ? legacyPosesDir : runPath;
          const poseFiles = fs.readdirSync(posesSearchDir).filter((f) => f.endsWith('_docked.sdf.gz'));

          if (poseFiles.length > 0) {
            const poses = poseFiles.map((f) => {
              const posePath = path.join(posesSearchDir, f);
              const name = f.replace('_docked.sdf.gz', '');
              return {
                name: stripPrefix(name),
                path: posePath,
                affinity: extractVinaAffinity(posePath),
              };
            });

            // Sort by affinity (best = most negative first)
            poses.sort((a, b) => (a.affinity ?? 0) - (b.affinity ?? 0));

            jobs.push({
              id: `dock:${run.name}`,
              type: 'docking',
              folder: run.name,
              label: `${run.name} (${poses.length} poses)`,
              path: runPath,
              receptorPdb,
              poses,
            });
          }
        }
      } catch { /* skip */ }
    }

    // 3. Scan simulations/ for MD run folders
    const simsDir = path.join(projectDir, 'simulations');
    if (fs.existsSync(simsDir)) {
      try {
        const simRuns = fs.readdirSync(simsDir, { withFileTypes: true });
        for (const run of simRuns) {
          if (!run.isDirectory() || run.name.startsWith('.')) continue;
          const runPath = path.join(simsDir, run.name);
          const runFiles = fs.readdirSync(runPath);

          // Check results/ subdir first (new layout), then top-level (legacy)
          const resultsDir = path.join(runPath, 'results');
          const resultsFiles = fs.existsSync(resultsDir) ? fs.readdirSync(resultsDir) : [];

          // Find final PDB: results/final.pdb (new), then *_final.pdb (legacy)
          let finalPdbPath: string | undefined;
          if (resultsFiles.includes('final.pdb')) {
            finalPdbPath = path.join(resultsDir, 'final.pdb');
          } else {
            const legacyFinal = runFiles.find((f) => f.endsWith('_final.pdb'));
            if (legacyFinal) finalPdbPath = path.join(runPath, legacyFinal);
          }

          // Find system PDB + trajectory DCD
          let systemPdbPath: string | undefined;
          let trajectoryDcdPath: string | undefined;

          // New layout: results/
          if (resultsFiles.includes('system.pdb')) systemPdbPath = path.join(resultsDir, 'system.pdb');
          if (resultsFiles.includes('trajectory.dcd')) trajectoryDcdPath = path.join(resultsDir, 'trajectory.dcd');

          // Legacy: *_system.pdb, *_trajectory.dcd at top level
          if (!systemPdbPath) {
            const legacySys = runFiles.find((f) => f.endsWith('_system.pdb'));
            if (legacySys) systemPdbPath = path.join(runPath, legacySys);
          }
          if (!trajectoryDcdPath) {
            const legacyTraj = runFiles.find((f) => f.endsWith('_trajectory.dcd'));
            if (legacyTraj) trajectoryDcdPath = path.join(runPath, legacyTraj);
          }

          // Consolidate all outputs into one job per simulation folder
          if (finalPdbPath || (systemPdbPath && trajectoryDcdPath)) {
            // Count clusters
            let clusterCount = 0;
            let clusterDirPath: string | undefined;
            const newClusterDir = path.join(runPath, 'analysis', 'clustering');
            const legacyClusterDir = path.join(runPath, 'clustering');
            const resolvedClusterDir = fs.existsSync(newClusterDir) ? newClusterDir : fs.existsSync(legacyClusterDir) ? legacyClusterDir : null;
            if (resolvedClusterDir) {
              const clusterFiles = fs.readdirSync(resolvedClusterDir).filter((f: string) => f.match(/cluster_\d+_centroid\.pdb/));
              if (clusterFiles.length > 0) {
                clusterCount = clusterFiles.length;
                clusterDirPath = resolvedClusterDir;
              }
            }

            // Build descriptive label
            const parts = [];
            if (trajectoryDcdPath) parts.push('trajectory');
            if (clusterCount > 0) parts.push(`${clusterCount} clusters`);
            const suffix = parts.length > 0 ? ` (${parts.join(', ')})` : '';

            jobs.push({
              id: `sim:${run.name}`,
              type: 'simulation',
              folder: run.name,
              label: `${run.name}${suffix}`,
              path: runPath,
              systemPdb: systemPdbPath,
              trajectoryDcd: trajectoryDcdPath,
              finalPdb: finalPdbPath,
              hasTrajectory: !!trajectoryDcdPath,
              clusterCount,
              clusterDir: clusterDirPath,
            });
          }
        }
      } catch { /* skip */ }
    }

    return jobs;
  }
);

// Select an Ember job folder via dialog, validate, and return as ProjectJob
ipcMain.handle(
  IpcChannels.SELECT_EMBER_JOB_FOLDER,
  async (): Promise<any | null> => {
    const emberDir = path.join(app.getPath('home'), 'Ember');
    const result = await dialog.showOpenDialog({
      title: 'Select Ember Job Folder',
      defaultPath: emberDir,
      properties: ['openDirectory'],
    });
    if (result.canceled || result.filePaths.length === 0) return null;

    const folderPath = result.filePaths[0];
    const folderName = path.basename(folderPath);
    const folderFiles = fs.readdirSync(folderPath);

    // Check for docking job: inputs/receptor.pdb + results/poses/*_docked.sdf.gz
    const newReceptor = path.join(folderPath, 'inputs', 'receptor.pdb');
    const legacyReceptor = folderFiles.find((f) => f.includes('_receptor_prepared') && f.endsWith('.pdb'));
    const receptorPdb = fs.existsSync(newReceptor) ? newReceptor : (legacyReceptor ? path.join(folderPath, legacyReceptor) : undefined);

    const newPosesDir = path.join(folderPath, 'results', 'poses');
    const legacyPosesDir = path.join(folderPath, 'poses');
    const posesSearchDir = fs.existsSync(newPosesDir) ? newPosesDir : fs.existsSync(legacyPosesDir) ? legacyPosesDir : folderPath;

    let poseFiles: string[] = [];
    try {
      poseFiles = fs.readdirSync(posesSearchDir).filter((f) => f.endsWith('_docked.sdf.gz'));
    } catch { /* ignore */ }

    if (receptorPdb && poseFiles.length > 0) {
      const poses = poseFiles.map((f) => ({
        name: f.replace('_docked.sdf.gz', ''),
        path: path.join(posesSearchDir, f),
        affinity: extractVinaAffinity(path.join(posesSearchDir, f)),
      }));
      poses.sort((a, b) => (a.affinity ?? 0) - (b.affinity ?? 0));
      return {
        id: `dock:${folderName}`,
        type: 'docking',
        folder: folderName,
        label: `${folderName} (${poses.length} poses)`,
        path: folderPath,
        receptorPdb,
        poses,
      };
    }

    // Check for simulation job: system.pdb + trajectory.dcd (or results/ subdir)
    const resultsDir = path.join(folderPath, 'results');
    const resultsFiles = fs.existsSync(resultsDir) ? fs.readdirSync(resultsDir) : [];
    const runFiles = folderFiles;

    let systemPdb: string | undefined;
    let trajectoryDcd: string | undefined;

    if (resultsFiles.includes('system.pdb')) systemPdb = path.join(resultsDir, 'system.pdb');
    if (resultsFiles.includes('trajectory.dcd')) trajectoryDcd = path.join(resultsDir, 'trajectory.dcd');
    if (!systemPdb) {
      const legacy = runFiles.find((f) => f.endsWith('_system.pdb') || f === 'system.pdb');
      if (legacy) systemPdb = path.join(folderPath, legacy);
    }
    if (!trajectoryDcd) {
      const legacy = runFiles.find((f) => f.endsWith('_trajectory.dcd') || f === 'trajectory.dcd');
      if (legacy) trajectoryDcd = path.join(folderPath, legacy);
    }

    if (systemPdb) {
      return {
        id: `sim:${folderName}`,
        type: 'simulation',
        folder: folderName,
        label: folderName,
        path: folderPath,
        systemPdb,
        trajectoryDcd,
        hasTrajectory: !!trajectoryDcd,
      };
    }

    return null;
  }
);

// Read image file and return as data URL (for thumbnail display in Electron)
ipcMain.handle(
  'read-image-as-data-url',
  async (_event, imagePath: string): Promise<string | null> => {
    try {
      if (!fs.existsSync(imagePath)) return null;
      const data = fs.readFileSync(imagePath);
      const ext = path.extname(imagePath).toLowerCase();
      const mimeType = ext === '.png' ? 'image/png' : 'image/jpeg';
      return `data:${mimeType};base64,${data.toString('base64')}`;
    } catch {
      return null;
    }
  }
);

// === JSON File Reading (for CORDIAL scores) ===

ipcMain.handle('read-json-file', async (_event, jsonPath: string): Promise<unknown | null> => {
  try {
    if (!fs.existsSync(jsonPath)) return null;
    const content = fs.readFileSync(jsonPath, 'utf-8');
    return JSON.parse(content);
  } catch {
    return null;
  }
});

// === Text File Writing (for logs) ===

ipcMain.handle(IpcChannels.WRITE_TEXT_FILE, async (_event, filePath: string, content: string): Promise<Result<string, AppError>> => {
  try {
    // Ensure parent directory exists
    const dir = path.dirname(filePath);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(filePath, content, 'utf-8');
    return Ok(filePath);
  } catch (err) {
    return Err({
      type: 'FILE_WRITE_ERROR',
      path: filePath,
      message: `Failed to write file: ${(err as Error).message}`,
    });
  }
});

// === CORDIAL Rescoring Handlers ===

function getCordialRoot(): string | null {
  // Check environment variable first
  if (process.env.CORDIAL_ROOT && fs.existsSync(process.env.CORDIAL_ROOT)) {
    return process.env.CORDIAL_ROOT;
  }

  // Check for bundled installation (.deb package)
  const bundledCordial = path.join(BUNDLED_INSTALL_PATH, 'cordial');
  if (fs.existsSync(bundledCordial) &&
      fs.existsSync(path.join(bundledCordial, 'weights')) &&
      fs.existsSync(path.join(bundledCordial, 'modules'))) {
    return bundledCordial;
  }

  // Check relative to app directory (development / project root)
  // __dirname is electron-dist/, so one level up is project root
  const projectRoot = path.dirname(__dirname);
  const projectCordial = path.join(projectRoot, 'CORDIAL');
  if (fs.existsSync(projectCordial) &&
      fs.existsSync(path.join(projectCordial, 'weights')) &&
      fs.existsSync(path.join(projectCordial, 'modules'))) {
    return projectCordial;
  }

  const homeDir = process.env.HOME || process.env.USERPROFILE || '';

  // Check common locations
  const candidates = [
    path.join(homeDir, 'Desktop', 'FragGen', 'CORDIAL'),
    path.join(homeDir, 'Desktop', 'CORDIAL'),
    path.join(homeDir, 'CORDIAL'),
    path.join(homeDir, 'cordial'),
    path.join(homeDir, 'projects', 'CORDIAL'),
    '/opt/CORDIAL',
  ];

  for (const candidate of candidates) {
    // Check if the directory exists and has the expected structure
    if (fs.existsSync(candidate) &&
        fs.existsSync(path.join(candidate, 'weights')) &&
        fs.existsSync(path.join(candidate, 'modules'))) {
      return candidate;
    }
  }

  return null;
}

// Post-docking pocket refinement (OpenMM + Sage 2.3.0 + OBC2)
ipcMain.handle(
  'dock:refine-poses',
  async (
    event,
    receptorPdb: string,
    posesDir: string,
    outputDir: string,
    maxIterations: number
  ): Promise<Result<{ refinedCount: number; outputDir: string }, AppError>> => {
    return new Promise((resolve) => {
      if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
        resolve(Err({ type: 'PYTHON_NOT_FOUND', message: 'Python not found' }));
        return;
      }

      const scriptPath = path.join(fraggenRoot, 'refine_poses.py');
      if (!fs.existsSync(scriptPath)) {
        event.sender.send(IpcChannels.DOCK_OUTPUT, {
          type: 'stdout',
          data: 'Warning: refine_poses.py not found, skipping refinement\n'
        });
        resolve(Ok({ refinedCount: 0, outputDir }));
        return;
      }

      fs.mkdirSync(outputDir, { recursive: true });

      const args = [
        scriptPath,
        '--receptor_pdb', receptorPdb,
        '--poses_dir', posesDir,
        '--output_dir', outputDir,
        '--max_iterations', String(maxIterations),
      ];

      event.sender.send(IpcChannels.DOCK_OUTPUT, {
        type: 'stdout',
        data: `=== Pocket Refinement (Sage 2.3.0 + OBC2) ===\nReceptor: ${path.basename(receptorPdb)}\nPoses: ${posesDir}\nMax iterations: ${maxIterations}\n\n`
      });

      const envVars = { ...process.env };
      if (condaEnvBin) {
        envVars.PATH = `${condaEnvBin}:${envVars.PATH || ''}`;
      }

      const python = spawn(condaPythonPath, args, { env: envVars });
      childProcesses.add(python);

      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data: Buffer) => {
        const text = data.toString();
        stdout += text;
        event.sender.send(IpcChannels.DOCK_OUTPUT, { type: 'stdout', data: text });
      });

      python.stderr.on('data', (data: Buffer) => {
        const text = data.toString();
        stderr += text;
        // Show all stderr for refinement (critical for debugging OpenMM/PDBFixer issues)
        event.sender.send(IpcChannels.DOCK_OUTPUT, { type: 'stderr', data: text });
      });

      python.on('close', (code: number | null) => {
        childProcesses.delete(python);
        if (code === 0) {
          try {
            const lines = stdout.trim().split('\n');
            const lastLine = lines[lines.length - 1];
            if (lastLine && lastLine.startsWith('{')) {
              const result = JSON.parse(lastLine);
              resolve(Ok({
                refinedCount: result.refined_count || 0,
                outputDir: result.output_dir || outputDir,
              }));
            } else {
              resolve(Ok({ refinedCount: 0, outputDir }));
            }
          } catch (e) {
            resolve(Err({
              type: 'PARSE_FAILED',
              message: `Failed to parse refinement results: ${(e as Error).message}`,
            }));
          }
        } else {
          resolve(Err({
            type: 'REFINEMENT_FAILED',
            message: `Pose refinement failed (exit ${code}): ${stderr.slice(0, 300)}`,
          }));
        }
      });
    });
  }
);

// Check if CORDIAL is installed
ipcMain.handle(IpcChannels.CHECK_CORDIAL_INSTALLED, async (): Promise<boolean> => {
  const cordialRoot = getCordialRoot();
  if (!cordialRoot) return false;

  // Check for required files
  const weightsDir = path.join(cordialRoot, 'weights');
  const normFile = path.join(cordialRoot, 'resources', 'normalization', 'full.train.norm.pkl');

  if (!fs.existsSync(weightsDir) || !fs.existsSync(normFile)) return false;

  // Check for at least one model file
  try {
    const files = fs.readdirSync(weightsDir);
    return files.some(f => f.endsWith('.model'));
  } catch {
    return false;
  }
});

// Run CORDIAL scoring on docked poses
ipcMain.handle(
  IpcChannels.RUN_CORDIAL_SCORING,
  async (
    event,
    dockOutputDir: string,
    batchSize: number = 32
  ): Promise<Result<{ scoresFile: string; count: number }, AppError>> => {
    const cordialRoot = getCordialRoot();
    if (!cordialRoot) {
      return Err({
        type: 'CORDIAL_FAILED',
        message: 'CORDIAL not found. Set CORDIAL_ROOT environment variable or clone to ~/Desktop/CORDIAL',
      });
    }

    const pythonPath = getCondaPythonPath();
    if (!pythonPath) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Conda environment not found. Make sure the openmm-metal environment is set up.',
      });
    }

    // score_cordial.py lives in project-root/scripts/, not in deps/staging/scripts/
    const projectRoot = path.resolve(__dirname, '..', '..');
    const scriptPath = path.join(projectRoot, 'scripts', 'score_cordial.py');
    if (!fs.existsSync(scriptPath)) {
      return Err({
        type: 'SCRIPT_NOT_FOUND',
        path: scriptPath,
        message: `CORDIAL scoring script not found: ${scriptPath}`,
      });
    }

    // Write results to results/ subdir (new layout), creating it if needed
    const cordialResultsDir = path.join(dockOutputDir, 'results');
    fs.mkdirSync(cordialResultsDir, { recursive: true });
    const outputCsv = path.join(cordialResultsDir, 'cordial_scores.csv');

    return new Promise((resolve) => {
      event.sender.send(IpcChannels.DOCK_OUTPUT, {
        type: 'stdout',
        data: `=== CORDIAL Rescoring ===\nCORDIAL Root: ${cordialRoot}\nDock Output: ${dockOutputDir}\nBatch Size: ${batchSize}\n\n`,
      });

      const args = [
        scriptPath,
        '--dock_dir', dockOutputDir,
        '--cordial_root', cordialRoot,
        '--output', outputCsv,
        '--batch_size', String(batchSize),
      ];

      const proc = spawn(pythonPath, args, {
        cwd: cordialRoot,
        env: {
          ...process.env,
          PYTHONPATH: cordialRoot,
        },
      });

      childProcesses.add(proc);

      proc.stdout?.on('data', (data: Buffer) => {
        const text = data.toString();
        event.sender.send(IpcChannels.DOCK_OUTPUT, { type: 'stdout', data: text });
      });

      proc.stderr?.on('data', (data: Buffer) => {
        const text = data.toString();
        // Don't treat warnings as errors
        if (!text.toLowerCase().includes('error') && !text.toLowerCase().includes('traceback')) {
          event.sender.send(IpcChannels.DOCK_OUTPUT, { type: 'stdout', data: text });
        } else {
          event.sender.send(IpcChannels.DOCK_OUTPUT, { type: 'stderr', data: text });
        }
      });

      proc.on('close', (code) => {
        childProcesses.delete(proc);

        if (code === 0 && fs.existsSync(outputCsv)) {
          // Count results
          try {
            const content = fs.readFileSync(outputCsv, 'utf-8');
            const lines = content.trim().split('\n');
            const count = Math.max(0, lines.length - 1); // Subtract header

            event.sender.send(IpcChannels.DOCK_OUTPUT, {
              type: 'stdout',
              data: `\n=== CORDIAL Scoring Complete ===\nScored ${count} poses\nOutput: ${outputCsv}\n`,
            });

            resolve(Ok({ scoresFile: outputCsv, count }));
          } catch (err) {
            resolve(Err({
              type: 'CORDIAL_FAILED',
              message: `Error reading CORDIAL output: ${err}`,
            }));
          }
        } else {
          resolve(Err({
            type: 'CORDIAL_FAILED',
            message: `CORDIAL scoring failed with exit code ${code}`,
          }));
        }
      });

      proc.on('error', (err) => {
        childProcesses.delete(proc);
        resolve(Err({
          type: 'CORDIAL_FAILED',
          message: `Failed to start CORDIAL scoring: ${err.message}`,
        }));
      });
    });
  }
);

// === Trajectory Viewer Handlers ===

// Select DCD file dialog
ipcMain.handle(IpcChannels.SELECT_DCD_FILE, async (): Promise<string | null> => {
  const result = await dialog.showOpenDialog(mainWindow!, {
    title: 'Select Trajectory File',
    filters: [
      { name: 'DCD Trajectory', extensions: ['dcd'] },
      { name: 'All Files', extensions: ['*'] },
    ],
    properties: ['openFile'],
  });

  if (result.canceled || result.filePaths.length === 0) {
    return null;
  }

  return result.filePaths[0];
});

// List PDB files in a directory (for topology auto-detection)
ipcMain.handle(
  IpcChannels.LIST_PDB_IN_DIRECTORY,
  async (_event, dirPath: string): Promise<string[]> => {
    try {
      if (!fs.existsSync(dirPath)) {
        return [];
      }
      const files = fs.readdirSync(dirPath)
        .filter((f) => f.toLowerCase().endsWith('.pdb'))
        .sort();
      return files;
    } catch (error) {
      console.error('Error listing PDB files:', error);
      return [];
    }
  }
);

// Get trajectory info (frame count, timestep, duration)
ipcMain.handle(
  IpcChannels.GET_TRAJECTORY_INFO,
  async (
    _event,
    topologyPath: string,
    trajectoryPath: string
  ): Promise<Result<{
    frameCount: number;
    timestepPs: number;
    totalTimeNs: number;
  }, AppError>> => {
    if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Python not found. Please install miniconda and create fraggen environment.',
      });
    }

    if (!fs.existsSync(topologyPath)) {
      return Err({
        type: 'FILE_NOT_FOUND',
        path: topologyPath,
        message: `Topology file not found: ${topologyPath}`,
      });
    }

    if (!fs.existsSync(trajectoryPath)) {
      return Err({
        type: 'FILE_NOT_FOUND',
        path: trajectoryPath,
        message: `Trajectory file not found: ${trajectoryPath}`,
      });
    }

    const pythonCode = `
import sys
import json

try:
    import MDAnalysis as mda

    topology_path = '${topologyPath.replace(/'/g, "\\'")}'
    trajectory_path = '${trajectoryPath.replace(/'/g, "\\'")}'

    u = mda.Universe(topology_path, trajectory_path)

    frame_count = len(u.trajectory)

    # Get timestep in picoseconds (MDAnalysis stores in ps)
    timestep_ps = 0
    if frame_count > 1:
        u.trajectory[0]
        t0 = u.trajectory.time
        u.trajectory[1]
        t1 = u.trajectory.time
        timestep_ps = t1 - t0
    elif hasattr(u.trajectory, 'dt'):
        timestep_ps = u.trajectory.dt
    else:
        # Default to 10 ps if cannot determine
        timestep_ps = 10.0

    total_time_ns = (frame_count * timestep_ps) / 1000.0

    result = {
        "frameCount": frame_count,
        "timestepPs": timestep_ps,
        "totalTimeNs": total_time_ns
    }

    print(json.dumps(result))
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
`;

    return new Promise((resolve) => {
      const python = spawn(condaPythonPath!, ['-c', pythonCode]);
      childProcesses.add(python);

      let stdout = '';
      let stderr = '';

      python.stdout?.on('data', (data: Buffer) => {
        stdout += data.toString();
      });

      python.stderr?.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      python.on('close', (code: number | null) => {
        childProcesses.delete(python);
        if (code === 0 && stdout.trim()) {
          try {
            const result = JSON.parse(stdout.trim());
            if (result.error) {
              resolve(Err({
                type: 'TRAJECTORY_READ_FAILED',
                message: result.error,
              }));
            } else {
              resolve(Ok({
                frameCount: result.frameCount,
                timestepPs: result.timestepPs,
                totalTimeNs: result.totalTimeNs,
              }));
            }
          } catch (e) {
            resolve(Err({
              type: 'PARSE_FAILED',
              message: `Failed to parse trajectory info: ${stdout}`,
            }));
          }
        } else {
          resolve(Err({
            type: 'TRAJECTORY_READ_FAILED',
            message: stderr || 'Failed to read trajectory info',
          }));
        }
      });

      python.on('error', (error: Error) => {
        childProcesses.delete(python);
        resolve(Err({
          type: 'TRAJECTORY_READ_FAILED',
          message: error.message,
        }));
      });
    });
  }
);

// Get a specific trajectory frame as PDB string
ipcMain.handle(
  IpcChannels.GET_TRAJECTORY_FRAME,
  async (
    _event,
    topologyPath: string,
    trajectoryPath: string,
    frameIndex: number
  ): Promise<Result<{ pdbString: string }, AppError>> => {
    if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Python not found.',
      });
    }

    const pythonCode = `
import sys
import json
import warnings
import tempfile
import os
warnings.filterwarnings('ignore')

try:
    import MDAnalysis as mda

    topology_path = '${topologyPath.replace(/'/g, "\\'")}'
    trajectory_path = '${trajectoryPath.replace(/'/g, "\\'")}'
    frame_index = ${frameIndex}

    u = mda.Universe(topology_path, trajectory_path)

    # Apply PBC transformations to center system in view
    try:
        from MDAnalysis import transformations as trans

        protein = u.select_atoms('protein')
        # Try to get ligand for centering together
        lig_sele = 'not protein and not resname WAT HOH TIP3 TIP4 NA CL SOL K MG and not element H'
        ligand = u.select_atoms(lig_sele)
        if len(ligand) == 0:
            ligand = u.select_atoms('(resname LIG UNL UNK MOL) and not element H')

        if len(protein) > 0:
            if len(ligand) > 0:
                complex_group = protein + ligand
            else:
                complex_group = protein

            workflow = [
                trans.unwrap(complex_group),
                trans.center_in_box(protein, center='mass'),
                trans.wrap(complex_group, compound='fragments'),
            ]
            u.trajectory.add_transformations(*workflow)
        elif len(ligand) > 0:
            # Ligand-only system: center on ligand
            workflow = [
                trans.unwrap(ligand),
                trans.center_in_box(ligand, center='mass'),
                trans.wrap(ligand, compound='fragments'),
            ]
            u.trajectory.add_transformations(*workflow)
    except Exception:
        pass  # If PBC transforms fail, continue without them

    u.trajectory[frame_index]

    # Write frame to temporary PDB file then read it back
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        temp_path = f.name

    u.atoms.write(temp_path)

    with open(temp_path, 'r') as f:
        pdb_string = f.read()

    os.unlink(temp_path)

    # Output as base64 to avoid JSON issues with special characters
    import base64
    encoded = base64.b64encode(pdb_string.encode()).decode()
    print(json.dumps({"pdbBase64": encoded}))
except Exception as e:
    import traceback
    print(json.dumps({"error": str(e) + "\\n" + traceback.format_exc()}))
    sys.exit(1)
`;

    return new Promise((resolve) => {
      const python = spawn(condaPythonPath!, ['-c', pythonCode]);
      childProcesses.add(python);

      let stdout = '';
      let stderr = '';

      python.stdout?.on('data', (data: Buffer) => {
        stdout += data.toString();
      });

      python.stderr?.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      python.on('close', (code: number | null) => {
        childProcesses.delete(python);
        if (code === 0 && stdout.trim()) {
          try {
            const result = JSON.parse(stdout.trim());
            if (result.error) {
              resolve(Err({
                type: 'TRAJECTORY_READ_FAILED',
                message: result.error,
              }));
            } else {
              // Decode base64 PDB string
              const pdbString = Buffer.from(result.pdbBase64, 'base64').toString('utf-8');
              resolve(Ok({ pdbString }));
            }
          } catch (e) {
            resolve(Err({
              type: 'PARSE_FAILED',
              message: `Failed to parse frame data: ${stdout.substring(0, 200)}`,
            }));
          }
        } else {
          resolve(Err({
            type: 'TRAJECTORY_READ_FAILED',
            message: stderr || 'Failed to read trajectory frame',
          }));
        }
      });
    });
  }
);

// Get trajectory frame coordinates only (no PDB parsing — for atomStore updates)
ipcMain.handle(
  IpcChannels.GET_TRAJECTORY_COORDS,
  async (
    _event,
    topologyPath: string,
    trajectoryPath: string,
    frameIndex: number
  ): Promise<Result<{ coordsBase64: string; atomCount: number }, AppError>> => {
    if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Python not found.',
      });
    }

    const pythonCode = `
import sys
import json
import warnings
import base64
import numpy as np
warnings.filterwarnings('ignore')

try:
    import MDAnalysis as mda

    topology_path = '${topologyPath.replace(/'/g, "\\'")}'
    trajectory_path = '${trajectoryPath.replace(/'/g, "\\'")}'
    frame_index = ${frameIndex}

    u = mda.Universe(topology_path, trajectory_path)

    # Apply PBC transformations (same as GET_TRAJECTORY_FRAME)
    try:
        from MDAnalysis import transformations as trans

        protein = u.select_atoms('protein')
        lig_sele = 'not protein and not resname WAT HOH TIP3 TIP4 NA CL SOL K MG and not element H'
        ligand = u.select_atoms(lig_sele)
        if len(ligand) == 0:
            ligand = u.select_atoms('(resname LIG UNL UNK MOL) and not element H')

        if len(protein) > 0:
            if len(ligand) > 0:
                complex_group = protein + ligand
            else:
                complex_group = protein

            workflow = [
                trans.unwrap(complex_group),
                trans.center_in_box(protein, center='mass'),
                trans.wrap(complex_group, compound='fragments'),
            ]
            u.trajectory.add_transformations(*workflow)
        elif len(ligand) > 0:
            workflow = [
                trans.unwrap(ligand),
                trans.center_in_box(ligand, center='mass'),
                trans.wrap(ligand, compound='fragments'),
            ]
            u.trajectory.add_transformations(*workflow)
    except Exception:
        pass

    u.trajectory[frame_index]

    # Flatten positions to [x0, y0, z0, x1, y1, z1, ...] as float32
    coords = u.atoms.positions.astype(np.float32).flatten()
    encoded = base64.b64encode(coords.tobytes()).decode()
    print(json.dumps({"coordsBase64": encoded, "atomCount": len(u.atoms)}))
except Exception as e:
    import traceback
    print(json.dumps({"error": str(e) + "\\n" + traceback.format_exc()}))
    sys.exit(1)
`;

    return new Promise((resolve) => {
      const python = spawn(condaPythonPath!, ['-c', pythonCode]);
      childProcesses.add(python);

      let stdout = '';
      let stderr = '';

      python.stdout?.on('data', (data: Buffer) => {
        stdout += data.toString();
      });

      python.stderr?.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      python.on('close', (code: number | null) => {
        childProcesses.delete(python);
        if (code === 0 && stdout.trim()) {
          try {
            const result = JSON.parse(stdout.trim());
            if (result.error) {
              resolve(Err({
                type: 'TRAJECTORY_READ_FAILED',
                message: result.error,
              }));
            } else {
              resolve(Ok({
                coordsBase64: result.coordsBase64,
                atomCount: result.atomCount,
              }));
            }
          } catch (e) {
            resolve(Err({
              type: 'PARSE_FAILED',
              message: `Failed to parse coords data: ${stdout.substring(0, 200)}`,
            }));
          }
        } else {
          resolve(Err({
            type: 'TRAJECTORY_READ_FAILED',
            message: stderr || 'Failed to read trajectory coordinates',
          }));
        }
      });
    });
  }
);

// Cluster trajectory using MDAnalysis
ipcMain.handle(
  IpcChannels.CLUSTER_TRAJECTORY,
  async (
    event,
    options: {
      topologyPath: string;
      trajectoryPath: string;
      numClusters: number;
      method: 'kmeans' | 'dbscan' | 'hierarchical';
      rmsdSelection: 'ligand' | 'backbone' | 'all';
      stripWaters: boolean;
      outputDir: string;
    }
  ): Promise<Result<{
    clusters: Array<{
      clusterId: number;
      frameCount: number;
      population: number;
      centroidFrame: number;
      centroidPdbPath?: string;
    }>;
    frameAssignments: number[];
    outputDir: string;
  }, AppError>> => {
    if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Python not found. Please install miniconda and create fraggen environment.',
      });
    }

    const scriptPath = path.join(fraggenRoot, 'cluster_trajectory.py');
    if (!fs.existsSync(scriptPath)) {
      return Err({
        type: 'SCRIPT_NOT_FOUND',
        path: scriptPath,
        message: `Clustering script not found: ${scriptPath}`,
      });
    }

    // Ensure output directory exists
    if (!fs.existsSync(options.outputDir)) {
      fs.mkdirSync(options.outputDir, { recursive: true });
    }

    return new Promise((resolve) => {
      event.sender.send(IpcChannels.MD_OUTPUT, {
        type: 'stdout',
        data: `=== Trajectory Clustering ===\nMethod: ${options.method}\nClusters: ${options.numClusters}\nRMSD Selection: ${options.rmsdSelection}\n\n`,
      });

      const args = [
        scriptPath,
        '--topology', options.topologyPath,
        '--trajectory', options.trajectoryPath,
        '--n_clusters', String(options.numClusters),
        '--method', options.method,
        '--selection', options.rmsdSelection,
        '--output_dir', options.outputDir,
      ];

      if (options.stripWaters) {
        args.push('--strip_waters');
      }

      const proc = spawn(condaPythonPath!, args);
      childProcesses.add(proc);

      let stderrOutput = '';

      proc.stdout?.on('data', (data: Buffer) => {
        const text = data.toString();
        try { event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stdout', data: text }); } catch {}
      });

      proc.stderr?.on('data', (data: Buffer) => {
        const text = data.toString();
        stderrOutput += text;
        try { event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stderr', data: text }); } catch {}
      });

      proc.on('close', (code: number | null) => {
        childProcesses.delete(proc);

        const resultFile = path.join(options.outputDir, 'clustering_results.json');
        if (code === 0 && fs.existsSync(resultFile)) {
          try {
            const content = fs.readFileSync(resultFile, 'utf-8');
            const result = JSON.parse(content);
            resolve(Ok({
              clusters: result.clusters,
              frameAssignments: result.frameAssignments,
              outputDir: options.outputDir,
            }));
          } catch (err) {
            resolve(Err({
              type: 'CLUSTERING_FAILED',
              message: `Error reading clustering results: ${err}`,
            }));
          }
        } else {
          // Include stderr in error message for debugging
          const stderrTail = stderrOutput.trim().split('\n').slice(-10).join('\n');
          resolve(Err({
            type: 'CLUSTERING_FAILED',
            message: `Clustering failed (exit code ${code}):\n${stderrTail}`,
          }));
        }
      });

      proc.on('error', (err: Error) => {
        childProcesses.delete(proc);
        resolve(Err({
          type: 'CLUSTERING_FAILED',
          message: `Failed to start clustering: ${err.message}`,
        }));
      });
    });
  }
);

// Scan a cluster directory for centroid PDB files
ipcMain.handle(
  IpcChannels.SCAN_CLUSTER_DIRECTORY,
  async (
    _event,
    directoryPath: string
  ): Promise<Result<{
    clusters: Array<{
      clusterId: number;
      pdbPath: string;
      population: number;
    }>;
    clusteringResultsPath?: string;
  }, AppError>> => {
    if (!fs.existsSync(directoryPath)) {
      return Err({
        type: 'DIRECTORY_NOT_FOUND',
        message: `Directory not found: ${directoryPath}`,
      });
    }

    try {
      // Find cluster centroid PDB files (pattern: cluster_N_centroid.pdb)
      const files = fs.readdirSync(directoryPath);
      const clusterPattern = /^cluster_(\d+)_centroid\.pdb$/;

      const clusterFiles: Array<{ clusterId: number; pdbPath: string }> = [];
      for (const file of files) {
        const match = file.match(clusterPattern);
        if (match) {
          clusterFiles.push({
            clusterId: parseInt(match[1], 10),
            pdbPath: path.join(directoryPath, file),
          });
        }
      }

      if (clusterFiles.length === 0) {
        return Err({
          type: 'NO_CLUSTERS_FOUND',
          message: 'No cluster centroid PDB files found (expected pattern: cluster_N_centroid.pdb)',
        });
      }

      // Sort by cluster ID
      clusterFiles.sort((a, b) => a.clusterId - b.clusterId);

      // Try to read clustering_results.json for population info
      const resultsPath = path.join(directoryPath, 'clustering_results.json');
      let populationMap = new Map<number, number>();
      let clusteringResultsPath: string | undefined;

      if (fs.existsSync(resultsPath)) {
        try {
          const content = fs.readFileSync(resultsPath, 'utf-8');
          const results = JSON.parse(content);
          clusteringResultsPath = resultsPath;

          if (results.clusters && Array.isArray(results.clusters)) {
            for (const cluster of results.clusters) {
              if (typeof cluster.clusterId === 'number' && typeof cluster.population === 'number') {
                populationMap.set(cluster.clusterId, cluster.population);
              }
            }
          }
        } catch {
          // Ignore JSON parsing errors, just use default populations
        }
      }

      // Build result with populations
      const clusters = clusterFiles.map(cf => ({
        clusterId: cf.clusterId,
        pdbPath: cf.pdbPath,
        population: populationMap.get(cf.clusterId) ?? 0,
      }));

      return Ok({ clusters, clusteringResultsPath });
    } catch (err) {
      return Err({
        type: 'SCAN_FAILED',
        message: `Failed to scan cluster directory: ${err}`,
      });
    }
  }
);

// Load and align cluster PDB files
ipcMain.handle(
  IpcChannels.LOAD_ALIGNED_CLUSTERS,
  async (
    _event,
    directoryPath: string,
    clusterIds: number[]
  ): Promise<Result<{
    clusters: Array<{
      clusterId: number;
      pdbPath: string;
      alignedPath: string;
      population: number;
    }>;
  }, AppError>> => {
    if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Python not found. Please install miniconda and create fraggen environment.',
      });
    }

    const scriptPath = path.join(fraggenRoot, 'align_clusters.py');
    if (!fs.existsSync(scriptPath)) {
      return Err({
        type: 'SCRIPT_NOT_FOUND',
        path: scriptPath,
        message: `Alignment script not found: ${scriptPath}`,
      });
    }

    // Get cluster files and populations from directory
    const resultsPath = path.join(directoryPath, 'clustering_results.json');
    let populationMap = new Map<number, number>();

    if (fs.existsSync(resultsPath)) {
      try {
        const content = fs.readFileSync(resultsPath, 'utf-8');
        const results = JSON.parse(content);
        if (results.clusters && Array.isArray(results.clusters)) {
          for (const cluster of results.clusters) {
            if (typeof cluster.clusterId === 'number' && typeof cluster.population === 'number') {
              populationMap.set(cluster.clusterId, cluster.population);
            }
          }
        }
      } catch {
        // Ignore errors
      }
    }

    // Build list of PDB paths to align
    const pdbPaths: string[] = [];
    for (const clusterId of clusterIds) {
      const pdbPath = path.join(directoryPath, `cluster_${clusterId}_centroid.pdb`);
      if (fs.existsSync(pdbPath)) {
        pdbPaths.push(pdbPath);
      }
    }

    if (pdbPaths.length === 0) {
      return Err({
        type: 'NO_CLUSTERS_FOUND',
        message: 'No cluster PDB files found for the specified cluster IDs',
      });
    }

    // Create output directory for aligned PDBs
    const alignedDir = path.join(directoryPath, 'aligned');

    return new Promise((resolve) => {
      const args = [
        scriptPath,
        '--pdb_files', ...pdbPaths,
        '--output_dir', alignedDir,
      ];

      const proc = spawn(condaPythonPath!, args);
      childProcesses.add(proc);

      let stdout = '';
      let stderr = '';

      proc.stdout?.on('data', (data: Buffer) => {
        stdout += data.toString();
      });

      proc.stderr?.on('data', (data: Buffer) => {
        stderr += data.toString();
        // Log alignment progress
        console.log(data.toString().trim());
      });

      proc.on('close', (code: number | null) => {
        childProcesses.delete(proc);

        if (code === 0) {
          try {
            const result = JSON.parse(stdout);
            // Map the aligned PDB paths back to clusters
            const clusters = result.alignedPdbs.map((item: { originalPath: string; alignedPath: string }) => {
              const filename = path.basename(item.originalPath);
              const match = filename.match(/cluster_(\d+)_centroid\.pdb/);
              const clusterId = match ? parseInt(match[1], 10) : 0;
              return {
                clusterId,
                pdbPath: item.originalPath,
                alignedPath: item.alignedPath,
                population: populationMap.get(clusterId) ?? 0,
              };
            });
            resolve(Ok({ clusters }));
          } catch (err) {
            resolve(Err({
              type: 'ALIGNMENT_FAILED',
              message: `Failed to parse alignment results: ${err}`,
            }));
          }
        } else {
          resolve(Err({
            type: 'ALIGNMENT_FAILED',
            message: stderr || `Alignment failed with exit code ${code}`,
          }));
        }
      });

      proc.on('error', (err: Error) => {
        childProcesses.delete(proc);
        resolve(Err({
          type: 'ALIGNMENT_FAILED',
          message: `Failed to start alignment: ${err.message}`,
        }));
      });
    });
  }
);

// Export a single frame from trajectory as PDB
ipcMain.handle(
  IpcChannels.EXPORT_TRAJECTORY_FRAME,
  async (
    _event,
    options: {
      topologyPath: string;
      trajectoryPath: string;
      frameIndex: number;
      outputPath: string;
      stripWaters?: boolean;
    }
  ): Promise<Result<{ pdbPath: string }, AppError>> => {
    if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Python not found. Please install miniconda and create fraggen environment.',
      });
    }

    const scriptPath = path.join(fraggenRoot, 'export_frame.py');
    if (!fs.existsSync(scriptPath)) {
      return Err({
        type: 'SCRIPT_NOT_FOUND',
        path: scriptPath,
        message: `Export frame script not found: ${scriptPath}`,
      });
    }

    return new Promise((resolve) => {
      const args = [
        scriptPath,
        '--topology', options.topologyPath,
        '--trajectory', options.trajectoryPath,
        '--frame', String(options.frameIndex),
        '--output', options.outputPath,
      ];

      if (options.stripWaters) {
        args.push('--strip_waters');
      }

      const proc = spawn(condaPythonPath!, args);
      childProcesses.add(proc);

      let stderr = '';

      proc.stderr?.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      proc.on('close', (code: number | null) => {
        childProcesses.delete(proc);

        if (code === 0 && fs.existsSync(options.outputPath)) {
          resolve(Ok({ pdbPath: options.outputPath }));
        } else {
          resolve(Err({
            type: 'EXPORT_FAILED',
            message: stderr || `Failed to export frame with exit code ${code}`,
          }));
        }
      });

      proc.on('error', (err: Error) => {
        childProcesses.delete(proc);
        resolve(Err({
          type: 'EXPORT_FAILED',
          message: `Failed to start export: ${err.message}`,
        }));
      });
    });
  }
);

// Analyze trajectory (RMSD, RMSF, H-bonds, contacts)
ipcMain.handle(
  IpcChannels.ANALYZE_TRAJECTORY,
  async (
    event,
    options: {
      topologyPath: string;
      trajectoryPath: string;
      analysisType: 'rmsd' | 'rmsf' | 'hbonds' | 'contacts';
      outputDir: string;
      ligandSelection?: string;
    }
  ): Promise<Result<{
    type: string;
    plotPath: string;
    csvPath?: string;
    data: unknown;
  }, AppError>> => {
    if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Python not found. Please install miniconda and create fraggen environment.',
      });
    }

    const scriptMap: Record<string, string> = {
      rmsd: 'analyze_rmsd.py',
      rmsf: 'analyze_rmsf.py',
      hbonds: 'analyze_hbonds.py',
      contacts: 'analyze_contacts.py',
    };

    const scriptPath = path.join(fraggenRoot, scriptMap[options.analysisType]);
    if (!fs.existsSync(scriptPath)) {
      return Err({
        type: 'SCRIPT_NOT_FOUND',
        path: scriptPath,
        message: `Analysis script not found: ${scriptPath}`,
      });
    }

    // Ensure output directory exists
    if (!fs.existsSync(options.outputDir)) {
      fs.mkdirSync(options.outputDir, { recursive: true });
    }

    return new Promise((resolve) => {
      event.sender.send(IpcChannels.MD_OUTPUT, {
        type: 'stdout',
        data: `=== ${options.analysisType.toUpperCase()} Analysis ===\n`,
      });

      const args = [
        scriptPath,
        '--topology', options.topologyPath,
        '--trajectory', options.trajectoryPath,
        '--output_dir', options.outputDir,
      ];

      if (options.ligandSelection) {
        args.push('--ligand_selection', options.ligandSelection);
      }

      const proc = spawn(condaPythonPath!, args);
      childProcesses.add(proc);

      proc.stdout?.on('data', (data: Buffer) => {
        const text = data.toString();
        event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stdout', data: text });
      });

      proc.stderr?.on('data', (data: Buffer) => {
        const text = data.toString();
        event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stderr', data: text });
      });

      proc.on('close', (code: number | null) => {
        childProcesses.delete(proc);

        const resultFile = path.join(options.outputDir, `${options.analysisType}_results.json`);
        if (code === 0 && fs.existsSync(resultFile)) {
          try {
            const content = fs.readFileSync(resultFile, 'utf-8');
            const result = JSON.parse(content);
            resolve(Ok(result));
          } catch (err) {
            resolve(Err({
              type: 'ANALYSIS_FAILED',
              message: `Error reading analysis results: ${err}`,
            }));
          }
        } else {
          resolve(Err({
            type: 'ANALYSIS_FAILED',
            message: `Analysis failed with exit code ${code}`,
          }));
        }
      });

      proc.on('error', (err: Error) => {
        childProcesses.delete(proc);
        resolve(Err({
          type: 'ANALYSIS_FAILED',
          message: `Failed to start analysis: ${err.message}`,
        }));
      });
    });
  }
);

// Generate comprehensive MD analysis report (PDF pipeline)
ipcMain.handle(
  IpcChannels.GENERATE_MD_REPORT,
  async (
    event,
    options: {
      topologyPath: string;
      trajectoryPath: string;
      outputDir: string;
      ligandSelection?: string;
      simInfo?: Record<string, string>;
    }
  ): Promise<Result<{
    reportPath: string;
    analysisDir: string;
    sectionPdfs: string[];
    clusteringResults?: Array<{
      clusterId: number;
      frameCount: number;
      population: number;
      centroidFrame: number;
      centroidPdbPath?: string;
    }>;
  }, AppError>> => {
    if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Python not found. Please install miniconda and create fraggen environment.',
      });
    }

    const scriptPath = path.join(fraggenRoot, 'generate_md_report.py');
    if (!fs.existsSync(scriptPath)) {
      return Err({
        type: 'SCRIPT_NOT_FOUND',
        path: scriptPath,
        message: `Report generation script not found: ${scriptPath}`,
      });
    }

    // Ensure output directory exists
    if (!fs.existsSync(options.outputDir)) {
      fs.mkdirSync(options.outputDir, { recursive: true });
    }

    return new Promise((resolve) => {
      event.sender.send(IpcChannels.MD_OUTPUT, {
        type: 'stdout',
        data: `=== Generating MD Analysis Report ===\nOutput: ${options.outputDir}\n\n`,
      });

      const args = [
        scriptPath,
        '--topology', options.topologyPath,
        '--trajectory', options.trajectoryPath,
        '--output_dir', options.outputDir,
      ];

      if (options.ligandSelection) {
        args.push('--ligand_selection', options.ligandSelection);
      }

      if (options.simInfo) {
        args.push('--sim_info', JSON.stringify(options.simInfo));
      }

      const proc = spawn(condaPythonPath!, args);
      childProcesses.add(proc);

      proc.stdout?.on('data', (data: Buffer) => {
        const text = data.toString();
        event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stdout', data: text });
      });

      proc.stderr?.on('data', (data: Buffer) => {
        const text = data.toString();
        event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stderr', data: text });
      });

      proc.on('close', (code: number | null) => {
        childProcesses.delete(proc);

        const reportPath = path.join(options.outputDir, 'full_report.pdf');
        if (code === 0 && fs.existsSync(reportPath)) {
          // Collect section PDFs
          const sectionPdfs: string[] = [];
          const subdirs = ['contacts', 'rmsd', 'rmsf', 'sse', 'hbonds', 'ligand_props', 'torsions'];
          for (const subdir of subdirs) {
            const dirPath = path.join(options.outputDir, subdir);
            if (fs.existsSync(dirPath)) {
              const files = fs.readdirSync(dirPath).filter((f: string) => f.endsWith('.pdf'));
              for (const f of files) {
                sectionPdfs.push(path.join(dirPath, f));
              }
            }
          }

          // Read clustering results if available
          let clusteringResults;
          const clusteringFile = path.join(options.outputDir, 'clustering', 'clustering_results.json');
          if (fs.existsSync(clusteringFile)) {
            try {
              const clusterData = JSON.parse(fs.readFileSync(clusteringFile, 'utf-8'));
              clusteringResults = clusterData.clusters;
            } catch {
              // ignore parse errors
            }
          }

          resolve(Ok({
            reportPath,
            analysisDir: options.outputDir,
            sectionPdfs,
            clusteringResults,
          }));
        } else {
          resolve(Err({
            type: 'REPORT_FAILED',
            message: `Report generation failed with exit code ${code}`,
          }));
        }
      });

      proc.on('error', (err: Error) => {
        childProcesses.delete(proc);
        resolve(Err({
          type: 'REPORT_FAILED',
          message: `Failed to start report generation: ${err.message}`,
        }));
      });
    });
  }
);

// Map binding site interaction potentials around a ligand
ipcMain.handle(
  IpcChannels.MAP_BINDING_SITE,
  async (
    event,
    options: {
      pdbPath: string;
      ligandResname: string;
      ligandResnum: number;
      outputDir: string;
      boxPadding?: number;
      gridSpacing?: number;
    }
  ): Promise<Result<{
    hydrophobicDx: string;
    hbondDonorDx: string;
    hbondAcceptorDx: string;
    hotspots: Array<{ type: string; position: number[]; direction: number[]; score: number }>;
    gridDimensions: number[];
    ligandCom: number[];
  }, AppError>> => {
    if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Python not found. Please install miniconda and create the openmm-metal environment.',
      });
    }

    const scriptPath = path.join(fraggenRoot, 'map_binding_site.py');
    if (!fs.existsSync(scriptPath)) {
      return Err({
        type: 'SCRIPT_NOT_FOUND',
        path: scriptPath,
        message: `Binding site mapping script not found: ${scriptPath}`,
      });
    }

    if (!fs.existsSync(options.outputDir)) {
      fs.mkdirSync(options.outputDir, { recursive: true });
    }

    return new Promise((resolve) => {
      event.sender.send(IpcChannels.MD_OUTPUT, {
        type: 'stdout',
        data: '=== Binding Site Interaction Map ===\n',
      });

      // Derive project name from output path: .../surfaces/binding_site_map
      const bsmProjectName = path.basename(path.resolve(options.outputDir, '../..'));

      const args = [
        scriptPath,
        '--pdb_path', options.pdbPath,
        '--ligand_resname', options.ligandResname,
        '--ligand_resnum', String(options.ligandResnum),
        '--output_dir', options.outputDir,
        '--project_name', bsmProjectName,
      ];

      if (options.boxPadding !== undefined) {
        args.push('--box_padding', String(options.boxPadding));
      }
      if (options.gridSpacing !== undefined) {
        args.push('--grid_spacing', String(options.gridSpacing));
      }

      const proc = spawn(condaPythonPath!, args);
      childProcesses.add(proc);

      proc.stdout?.on('data', (data: Buffer) => {
        const text = data.toString();
        event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stdout', data: text });
      });

      proc.stderr?.on('data', (data: Buffer) => {
        const text = data.toString();
        event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stderr', data: text });
      });

      proc.on('close', (code: number | null) => {
        childProcesses.delete(proc);

        // Try prefixed results first, fall back to unprefixed
        const prefixedResultFile = path.join(options.outputDir, `${bsmProjectName}_binding_site_results.json`);
        const legacyResultFile = path.join(options.outputDir, 'binding_site_results.json');
        const resultFile = fs.existsSync(prefixedResultFile) ? prefixedResultFile : legacyResultFile;
        if (code === 0 && fs.existsSync(resultFile)) {
          try {
            const content = fs.readFileSync(resultFile, 'utf-8');
            const result = JSON.parse(content);
            resolve(Ok(result));
          } catch (err) {
            resolve(Err({
              type: 'BINDING_SITE_MAP_FAILED',
              message: `Error reading binding site results: ${err}`,
            }));
          }
        } else {
          resolve(Err({
            type: 'BINDING_SITE_MAP_FAILED',
            message: `Binding site mapping failed with exit code ${code}`,
          }));
        }
      });

      proc.on('error', (err: Error) => {
        childProcesses.delete(proc);
        resolve(Err({
          type: 'BINDING_SITE_MAP_FAILED',
          message: `Failed to start binding site mapping: ${err.message}`,
        }));
      });
    });
  }
);

// Compute surface properties (hydrophobic, electrostatic) for a PDB
ipcMain.handle(
  IpcChannels.COMPUTE_SURFACE_PROPS,
  async (
    _event,
    pdbPath: string,
    outputDir: string
  ): Promise<Result<{
    atomCount: number;
    hydrophobic: number[];
    electrostatic: number[];
    cachedPath: string;
  }, AppError>> => {
    if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Python not found.',
      });
    }

    const scriptPath = path.join(fraggenRoot, 'compute_surface_props.py');
    if (!fs.existsSync(scriptPath)) {
      return Err({
        type: 'SCRIPT_NOT_FOUND',
        path: scriptPath,
        message: `Surface properties script not found: ${scriptPath}`,
      });
    }

    // Cache path: outputDir/surface_properties.json
    fs.mkdirSync(outputDir, { recursive: true });
    const cachePath = path.join(outputDir, 'surface_properties.json');

    // Return cached if exists
    if (fs.existsSync(cachePath)) {
      try {
        const cached = JSON.parse(fs.readFileSync(cachePath, 'utf-8'));
        return Ok({ ...cached, cachedPath: cachePath });
      } catch {
        // Corrupted cache, recompute
      }
    }

    return new Promise((resolve) => {
      const proc = spawn(condaPythonPath!, [
        scriptPath,
        '--pdb_path', pdbPath,
        '--output_path', cachePath,
      ]);
      childProcesses.add(proc);

      let stderr = '';

      proc.stderr?.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      proc.on('close', (code: number | null) => {
        childProcesses.delete(proc);
        if (code === 0 && fs.existsSync(cachePath)) {
          try {
            const result = JSON.parse(fs.readFileSync(cachePath, 'utf-8'));
            resolve(Ok({ ...result, cachedPath: cachePath }));
          } catch (err) {
            resolve(Err({
              type: 'SURFACE_PROPS_FAILED',
              message: `Failed to parse surface properties: ${err}`,
            }));
          }
        } else {
          resolve(Err({
            type: 'SURFACE_PROPS_FAILED',
            message: `Surface property computation failed (exit ${code}): ${stderr.slice(0, 300)}`,
          }));
        }
      });

      proc.on('error', (err: Error) => {
        childProcesses.delete(proc);
        resolve(Err({
          type: 'SURFACE_PROPS_FAILED',
          message: err.message,
        }));
      });
    });
  }
);

// === Pocket Map (unified 3-method dispatch) ===

ipcMain.handle(
  IpcChannels.COMPUTE_POCKET_MAP,
  async (
    event,
    options: {
      method: 'static' | 'solvation' | 'probe';
      pdbPath: string;
      ligandResname: string;
      ligandResnum: number;
      outputDir: string;
      trajectoryPath?: string;
      boxPadding?: number;
      gridSpacing?: number;
    }
  ): Promise<Result<{
    hydrophobicDx: string;
    hbondDonorDx: string;
    hbondAcceptorDx: string;
    hotspots: Array<{ type: string; position: number[]; direction: number[]; score: number }>;
    gridDimensions: number[];
    ligandCom: number[];
    method?: string;
  }, AppError>> => {
    if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Python not found. Please install miniconda and create the openmm-metal environment.',
      });
    }

    // Determine script and args based on method
    let scriptName: string;
    let scriptArgs: string[];
    const projectName = path.basename(path.resolve(options.outputDir, '../..'));

    switch (options.method) {
      case 'static': {
        scriptName = 'map_binding_site.py';
        scriptArgs = [
          '--pdb_path', options.pdbPath,
          '--ligand_resname', options.ligandResname,
          '--ligand_resnum', String(options.ligandResnum),
          '--output_dir', options.outputDir,
          '--project_name', projectName,
          '--scoring', 'energy',
        ];
        if (options.boxPadding !== undefined) {
          scriptArgs.push('--box_padding', String(options.boxPadding));
        }
        if (options.gridSpacing !== undefined) {
          scriptArgs.push('--grid_spacing', String(options.gridSpacing));
        }
        break;
      }
      case 'solvation': {
        if (!options.trajectoryPath || !fs.existsSync(options.trajectoryPath)) {
          return Err({
            type: 'POCKET_MAP_FAILED',
            message: 'Solvation method requires an MD trajectory. Run a simulation first.',
          });
        }
        scriptName = 'analyze_gist.py';
        scriptArgs = [
          '--pdb_path', options.pdbPath,
          '--trajectory_path', options.trajectoryPath,
          '--ligand_resname', options.ligandResname,
          '--ligand_resnum', String(options.ligandResnum),
          '--output_dir', options.outputDir,
          '--project_name', projectName,
        ];
        if (options.boxPadding !== undefined) {
          scriptArgs.push('--box_padding', String(options.boxPadding));
        }
        if (options.gridSpacing !== undefined) {
          scriptArgs.push('--grid_spacing', String(options.gridSpacing));
        }
        break;
      }
      case 'probe': {
        scriptName = 'run_probe_md.py';
        scriptArgs = [
          '--pdb_path', options.pdbPath,
          '--ligand_resname', options.ligandResname,
          '--ligand_resnum', String(options.ligandResnum),
          '--output_dir', options.outputDir,
          '--project_name', projectName,
        ];
        break;
      }
      default:
        return Err({
          type: 'POCKET_MAP_FAILED',
          message: `Unknown pocket map method: ${options.method}`,
        });
    }

    const scriptPath = path.join(fraggenRoot, scriptName);
    if (!fs.existsSync(scriptPath)) {
      return Err({
        type: 'SCRIPT_NOT_FOUND',
        path: scriptPath,
        message: `Pocket map script not found: ${scriptPath}`,
      });
    }

    if (!fs.existsSync(options.outputDir)) {
      fs.mkdirSync(options.outputDir, { recursive: true });
    }

    return new Promise((resolve) => {
      event.sender.send(IpcChannels.MD_OUTPUT, {
        type: 'stdout',
        data: `=== Pocket Map: ${options.method} ===\n`,
      });

      const proc = spawn(condaPythonPath!, [scriptPath, ...scriptArgs]);
      childProcesses.add(proc);

      proc.stdout?.on('data', (data: Buffer) => {
        const text = data.toString();
        event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stdout', data: text });
      });

      proc.stderr?.on('data', (data: Buffer) => {
        const text = data.toString();
        event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stderr', data: text });
      });

      proc.on('close', (code: number | null) => {
        childProcesses.delete(proc);

        // Find results JSON (try prefixed first, fall back to unprefixed)
        const prefixedResultFile = path.join(options.outputDir, `${projectName}_binding_site_results.json`);
        const legacyResultFile = path.join(options.outputDir, 'binding_site_results.json');
        const resultFile = fs.existsSync(prefixedResultFile) ? prefixedResultFile : legacyResultFile;

        if (code === 0 && fs.existsSync(resultFile)) {
          try {
            const content = fs.readFileSync(resultFile, 'utf-8');
            const result = JSON.parse(content);
            // Tag with the method so the frontend knows which produced this
            result.method = options.method;
            resolve(Ok(result));
          } catch (err) {
            resolve(Err({
              type: 'POCKET_MAP_FAILED',
              message: `Error reading pocket map results: ${err}`,
            }));
          }
        } else {
          resolve(Err({
            type: 'POCKET_MAP_FAILED',
            message: `Pocket map (${options.method}) failed with exit code ${code}`,
          }));
        }
      });

      proc.on('error', (err: Error) => {
        childProcesses.delete(proc);
        resolve(Err({
          type: 'POCKET_MAP_FAILED',
          message: `Failed to start pocket map: ${err.message}`,
        }));
      });
    });
  }
);

// === FEP Scoring ===

let currentFepProcess: ChildProcess | null = null;

ipcMain.handle(
  IpcChannels.RUN_FEP_SCORING,
  async (
    event,
    options: {
      topologyPath: string;
      trajectoryPath: string;
      startNs: number;
      endNs: number;
      numSnapshots: number;
      speedPreset: 'fast' | 'accurate';
      outputDir: string;
      forceFieldPreset: string;
      ligandSdf?: string;
    }
  ): Promise<Result<{
    snapshots: Array<{
      snapshotIndex: number;
      frameIndex: number;
      timeNs: number;
      deltaG_complex: number;
      deltaG_solvent: number;
      deltaG_bind: number;
      uncertainty: number;
    }>;
    meanDeltaG: number;
    sem: number;
    outputDir: string;
  }, AppError>> => {
    if (!condaPythonPath || !fs.existsSync(condaPythonPath)) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Python not found. Please install miniconda and create the openmm-metal environment.',
      });
    }

    const scriptPath = path.join(fraggenRoot, 'run_abfe.py');
    if (!fs.existsSync(scriptPath)) {
      return Err({
        type: 'SCRIPT_NOT_FOUND',
        path: scriptPath,
        message: `ABFE FEP script not found: ${scriptPath}`,
      });
    }

    if (!fs.existsSync(options.outputDir)) {
      fs.mkdirSync(options.outputDir, { recursive: true });
    }

    return new Promise((resolve) => {
      event.sender.send(IpcChannels.MD_OUTPUT, {
        type: 'stdout',
        data: `=== ABFE Free Energy Perturbation ===\nPreset: ${options.speedPreset}\nSnapshots: ${options.numSnapshots}\nRange: ${options.startNs.toFixed(1)} - ${options.endNs.toFixed(1)} ns\n\n`,
      });

      const args = [
        scriptPath,
        '--topology', options.topologyPath,
        '--trajectory', options.trajectoryPath,
        '--start_ns', String(options.startNs),
        '--end_ns', String(options.endNs),
        '--num_snapshots', String(options.numSnapshots),
        '--speed_preset', options.speedPreset,
        '--output_dir', options.outputDir,
        '--force_field_preset', options.forceFieldPreset,
      ];

      if (options.ligandSdf) {
        args.push('--ligand_sdf', options.ligandSdf);
      }

      const proc = spawn(condaPythonPath!, args);
      currentFepProcess = proc;
      childProcesses.add(proc);

      let stderrOutput = '';

      proc.stdout?.on('data', (data: Buffer) => {
        const text = data.toString();
        try { event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stdout', data: text }); } catch {}
      });

      proc.stderr?.on('data', (data: Buffer) => {
        const text = data.toString();
        // Cap stderr to last 2KB — only used for error reporting on failure
        stderrOutput = (stderrOutput + text).slice(-2048);
        const filtered = filterMdStderr(text);
        if (filtered.trim()) {
          try { event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stderr', data: filtered }); } catch {}
        }
      });

      proc.on('close', (code: number | null) => {
        childProcesses.delete(proc);
        if (currentFepProcess === proc) currentFepProcess = null;

        const resultFile = path.join(options.outputDir, 'fep_results.json');
        if (code === 0 && fs.existsSync(resultFile)) {
          try {
            const content = fs.readFileSync(resultFile, 'utf-8');
            const result = JSON.parse(content);
            resolve(Ok(result));
          } catch (err) {
            resolve(Err({
              type: 'FEP_SCORING_FAILED',
              message: `Error reading FEP results: ${err}`,
            }));
          }
        } else if (code === null || code === 137 || code === 143) {
          resolve(Err({
            type: 'FEP_SCORING_CANCELLED',
            message: 'FEP scoring was cancelled.',
          }));
        } else {
          resolve(Err({
            type: 'FEP_SCORING_FAILED',
            message: stderrOutput.slice(-500) || `FEP scoring failed with exit code ${code}`,
          }));
        }
      });

      proc.on('error', (err: Error) => {
        childProcesses.delete(proc);
        if (currentFepProcess === proc) currentFepProcess = null;
        resolve(Err({
          type: 'FEP_SCORING_FAILED',
          message: err.message,
        }));
      });
    });
  }
);

ipcMain.handle(IpcChannels.CANCEL_FEP_SCORING, async () => {
  if (currentFepProcess && !currentFepProcess.killed) {
    currentFepProcess.kill('SIGTERM');
    currentFepProcess = null;
  }
});
