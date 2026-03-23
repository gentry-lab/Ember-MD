/**
 * Docking-related IPC handlers.
 *
 * Extracted from main.ts — covers parallel Vina docking, result parsing,
 * receptor/ligand preparation, pose refinement, CORDIAL rescoring,
 * QupKake pKa prediction, xTB scoring, and MD cluster scoring.
 */
import { ipcMain } from 'electron';
import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import * as zlib from 'zlib';
import { Ok, Err, Result } from '../../shared/types/result';
import { AppError } from '../../shared/types/errors';
import type { PreparedComplexManifest } from '../../shared/types/dock';
import { IpcChannels } from '../../shared/types/ipc';
import type {
  ClusteringResult,
  QupkakeCapabilityResult,
  LigandPkaResult,
  ScoredClusterResult,
} from '../../shared/types/ipc';
import * as appState from '../app-state';
import {
  childProcesses,
  loadAndMergeCordialScores,
  getSpawnEnv as _getSpawnEnv,
  spawnPythonScript as _spawnPythonScriptRaw,
  getQupkakeSpawnEnv as _getQupkakeSpawnEnv,
} from '../spawn';
import {
  getQupkakeXtbPath,
  getCordialRoot,
  detectBabelDataDir,
} from '../paths';

// ---------------------------------------------------------------------------
// Module-level state
// ---------------------------------------------------------------------------

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

interface PreparedComplexRunResult {
  manifestPath: string;
  preparedReceptorPdb: string;
  preparedReferenceLigandSdf: string;
  manifest: PreparedComplexManifest;
}

/** Track active docking processes for cancellation */
const dockingProcesses = new Set<ChildProcess>();

// ---------------------------------------------------------------------------
// Local convenience wrappers that close over appState
// ---------------------------------------------------------------------------

function getSpawnEnv(): NodeJS.ProcessEnv {
  return _getSpawnEnv(appState.condaEnvBin);
}

function getQupkakeSpawnEnv(): NodeJS.ProcessEnv {
  return _getQupkakeSpawnEnv(appState.condaEnvBin);
}

function spawnPythonScript(
  args: string[],
  options?: {
    env?: NodeJS.ProcessEnv;
    cwd?: string;
    onStdout?: (text: string) => void;
    onStderr?: (text: string) => void;
  }
): Promise<{ stdout: string; stderr: string; code: number }> {
  return _spawnPythonScriptRaw(appState.condaPythonPath, appState.condaEnvBin, args, options);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function rebuildDockingPool(resultsDir: string, posesDir: string): void {
  if (!fs.existsSync(posesDir)) return;
  const poseFiles = fs.readdirSync(posesDir)
    .filter((f) => f.endsWith('_docked.sdf.gz'))
    .sort((a, b) => a.localeCompare(b));
  const pooledParts: string[] = [];
  for (const file of poseFiles) {
    try {
      const gzData = fs.readFileSync(path.join(posesDir, file));
      const sdfText = zlib.gunzipSync(gzData).toString('utf-8');
      const delimIdx = sdfText.indexOf('$$$$');
      if (delimIdx >= 0) {
        pooledParts.push(sdfText.substring(0, delimIdx + 4));
      }
    } catch (error) {
      console.error(`Failed to include ${file} in pooled SDF:`, error);
    }
  }

  if (pooledParts.length > 0) {
    const pooledPath = path.join(resultsDir, 'all_docked.sdf');
    fs.writeFileSync(pooledPath, pooledParts.join('\n') + '\n');
  }
}

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
    const scriptPath = path.join(appState.fraggenRoot, 'run_vina_docking.py');

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

    const python = spawn(appState.condaPythonPath!, args, { env });
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

const resolveVinaScriptPath = (): string => path.join(appState.fraggenRoot, 'run_vina_docking.py');

const runVinaScoreOnly = async (
  receptorPath: string,
  ligandPath: string,
  referencePath: string,
  options?: {
    outputSdfGz?: string;
    autoboxAdd?: number;
    cpu?: number;
    seed?: number;
    onStdout?: (text: string) => void;
    onStderr?: (text: string) => void;
  },
): Promise<Result<number, AppError>> => {
  const vinaScript = resolveVinaScriptPath();
  if (!fs.existsSync(vinaScript)) {
    return Err({
      type: 'SCRIPT_NOT_FOUND',
      path: vinaScript,
      message: `Vina script not found: ${vinaScript}`,
    });
  }

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ember_vina_score_'));
  const outputSdfGz = options?.outputSdfGz || path.join(tmpDir, 'scored.sdf.gz');

  try {
    fs.mkdirSync(path.dirname(outputSdfGz), { recursive: true });
    const babelDataDir = process.env.BABEL_DATADIR || detectBabelDataDir();
    const env = {
      ...getSpawnEnv(),
      ...(babelDataDir ? { BABEL_DATADIR: babelDataDir } : {}),
    };

    const args = [
      vinaScript,
      '--receptor', receptorPath,
      '--ligand', ligandPath,
      '--reference', referencePath,
      '--output_dir', path.dirname(outputSdfGz),
      '--autobox_add', String(options?.autoboxAdd ?? 4),
      '--cpu', String(options?.cpu ?? 1),
      '--score_only',
      '--score_only_output_sdf', outputSdfGz,
    ];
    if ((options?.seed ?? 0) > 0) {
      args.push('--seed', String(options!.seed));
    }

    const { stdout, stderr, code } = await spawnPythonScript(args, {
      env,
      onStdout: options?.onStdout,
      onStderr: options?.onStderr,
    });
    if (code !== 0) {
      return Err({
        type: 'DOCKING_FAILED',
        message: stderr || `Vina score_only failed with exit code ${code}`,
      });
    }

    const match = stdout.match(/SCORE_ONLY:[^:]+:([-\d.]+)/);
    if (!match) {
      return Err({
        type: 'PARSE_FAILED',
        message: `Failed to parse Vina score_only output: ${stdout || stderr}`,
      });
    }

    return Ok(parseFloat(match[1]));
  } finally {
    try {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    } catch {
      // ignore temp cleanup failures
    }
  }
};

function scoreReferencePoseVina(
  receptor: string,
  referenceLigand: string,
  outputSdfGz: string,
  config: VinaDockConfig
): Promise<number> {
  return new Promise((resolve, reject) => {
    runVinaScoreOnly(receptor, referenceLigand, referenceLigand, {
      outputSdfGz,
      autoboxAdd: config.autoboxAdd,
      cpu: 1,
      seed: config.seed > 0 ? config.seed : undefined,
    }).then((result) => {
      if (result.ok) {
        resolve(result.value);
      } else {
        reject(new Error(result.error.message));
      }
    }).catch(reject);
  });
}

function parseSdfProperties(sdfPath: string): Promise<{
  success: boolean;
  error?: string;
  smiles?: string;
  vinaAffinity: number | null;
  vinaScoreOnlyAffinity?: number;
  refinementEnergy?: number;
  isReferencePose?: boolean;
  qed: number;
  mw: number;
  logp: number;
  thumbnail?: string;
  coreRmsd?: number;
}> {
  return new Promise((resolve) => {
    if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
      resolve({
        success: false,
        error: 'Python not found',
        vinaAffinity: null,
        qed: 0,
        mw: 0,
        logp: 0,
      });
      return;
    }

    const scriptPath = path.join(appState.fraggenRoot, 'parse_sdf_properties.py');
    if (!fs.existsSync(scriptPath)) {
      resolve({
        success: false,
        error: 'parse_sdf_properties.py not found',
        vinaAffinity: null,
        qed: 0,
        mw: 0,
        logp: 0,
      });
      return;
    }

    const python = spawn(appState.condaPythonPath, [scriptPath, '--sdf_file', sdfPath]);
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
            vinaAffinity: result.vinaAffinity ?? result.minimizedAffinity ?? null,
            vinaScoreOnlyAffinity: result.vinaScoreOnlyAffinity ?? undefined,
            refinementEnergy: result.refinementEnergy ?? undefined,
            isReferencePose: result.isReferencePose === true,
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
            vinaAffinity: null,
            qed: 0,
            mw: 0,
            logp: 0,
          });
        }
      } else {
          resolve({
            success: false,
            error: stderr || 'Script failed',
            vinaAffinity: null,
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
          vinaAffinity: null,
          qed: 0,
          mw: 0,
          logp: 0,
        });
    });
  });
}

const readJsonIfExists = <T>(jsonPath: string): T | null => {
  try {
    if (!fs.existsSync(jsonPath)) return null;
    return JSON.parse(fs.readFileSync(jsonPath, 'utf-8')) as T;
  } catch {
    return null;
  }
};

const readClusteringResult = (directoryPath: string): ClusteringResult | null => {
  const resultsPath = path.join(directoryPath, 'clustering_results.json');
  return readJsonIfExists<ClusteringResult>(resultsPath);
};

const readClusterScoreRows = (directoryPath: string): ScoredClusterResult[] => {
  const resultsPath = path.join(directoryPath, 'cluster_scores.json');
  const scoreData = readJsonIfExists<{ clusters?: ScoredClusterResult[] }>(resultsPath);
  return Array.isArray(scoreData?.clusters) ? scoreData!.clusters : [];
};

const writeClusterScoreRows = (directoryPath: string, clusters: ScoredClusterResult[]): void => {
  const resultsPath = path.join(directoryPath, 'cluster_scores.json');
  fs.writeFileSync(resultsPath, JSON.stringify({ clusters }, null, 2));
};

const mergeClusterScoresWithCanonical = (
  clusteringResults: ClusteringResult,
  scoreClusters: Array<Partial<ScoredClusterResult> & { clusterId: number }>,
): ScoredClusterResult[] => {
  const scoreMap = new Map(scoreClusters.map((cluster) => [cluster.clusterId, cluster]));
  return clusteringResults.clusters.map((cluster) => {
    const scored = scoreMap.get(cluster.clusterId);
    return {
      clusterId: cluster.clusterId,
      frameCount: cluster.frameCount,
      population: cluster.population,
      centroidFrame: cluster.centroidFrame,
      centroidPdbPath: cluster.centroidPdbPath || scored?.centroidPdbPath || '',
      receptorPdbPath: scored?.receptorPdbPath,
      ligandSdfPath: scored?.ligandSdfPath,
      vinaRescore: scored?.vinaRescore,
      cordialExpectedPkd: scored?.cordialExpectedPkd,
      cordialPHighAffinity: scored?.cordialPHighAffinity,
      cordialPVeryHighAffinity: scored?.cordialPVeryHighAffinity,
    };
  });
};

const resolveCordialScriptPath = (): string | null => {
  let scriptPath = path.join(appState.fraggenRoot, 'score_cordial.py');
  if (fs.existsSync(scriptPath)) {
    return scriptPath;
  }
  const projectRoot = path.resolve(__dirname, '..', '..');
  scriptPath = path.join(projectRoot, 'scripts', 'score_cordial.py');
  return fs.existsSync(scriptPath) ? scriptPath : null;
};

const runCordialScoringJob = async (
  input: { dockDir?: string; pairCsv?: string },
  outputCsv: string,
  batchSize: number,
  options?: {
    cwd?: string;
    onStdout?: (text: string) => void;
    onStderr?: (text: string) => void;
  },
): Promise<Result<{ scoresFile: string; count: number }, AppError>> => {
  const cordialRoot = getCordialRoot();
  if (!cordialRoot) {
    return Err({
      type: 'CORDIAL_FAILED',
      message: 'CORDIAL not found. Set CORDIAL_ROOT environment variable or clone to ~/Desktop/CORDIAL',
    });
  }

  const pythonPath = appState.condaPythonPath;
  if (!pythonPath) {
    return Err({
      type: 'PYTHON_NOT_FOUND',
      message: 'Conda environment not found. Make sure the openmm-metal environment is set up.',
    });
  }

  const scriptPath = resolveCordialScriptPath();
  if (!scriptPath) {
    return Err({
      type: 'SCRIPT_NOT_FOUND',
      path: path.join(appState.fraggenRoot, 'score_cordial.py'),
      message: 'CORDIAL scoring script not found',
    });
  }

  const args = [
    scriptPath,
    '--cordial_root', cordialRoot,
    '--output', outputCsv,
    '--batch_size', String(batchSize),
  ];
  if (input.dockDir) {
    args.push('--dock_dir', input.dockDir);
  } else if (input.pairCsv) {
    args.push('--pair_csv', input.pairCsv);
  } else {
    return Err({ type: 'CORDIAL_FAILED', message: 'No CORDIAL input was provided' });
  }

  const proc = spawn(pythonPath, args, {
    cwd: options?.cwd || cordialRoot,
    env: {
      ...process.env,
      PYTHONPATH: cordialRoot,
      KMP_DUPLICATE_LIB_OK: 'TRUE',
      OMP_NUM_THREADS: process.env.OMP_NUM_THREADS || '1',
      MKL_NUM_THREADS: process.env.MKL_NUM_THREADS || '1',
      OPENBLAS_NUM_THREADS: process.env.OPENBLAS_NUM_THREADS || '1',
      NUMEXPR_NUM_THREADS: process.env.NUMEXPR_NUM_THREADS || '1',
      VECLIB_MAXIMUM_THREADS: process.env.VECLIB_MAXIMUM_THREADS || '1',
    },
  });

  childProcesses.add(proc);

  return await new Promise((resolve) => {
    let stderr = '';

    proc.stdout?.on('data', (data: Buffer) => {
      const text = data.toString();
      options?.onStdout?.(text);
    });

    proc.stderr?.on('data', (data: Buffer) => {
      const text = data.toString();
      stderr += text;
      options?.onStderr?.(text);
    });

    proc.on('close', (code) => {
      childProcesses.delete(proc);

      if (code === 0 && fs.existsSync(outputCsv)) {
        try {
          const content = fs.readFileSync(outputCsv, 'utf-8');
          const lines = content.trim().split('\n');
          resolve(Ok({ scoresFile: outputCsv, count: Math.max(0, lines.length - 1) }));
        } catch (err) {
          resolve(Err({
            type: 'CORDIAL_FAILED',
            message: `Error reading CORDIAL output: ${err}`,
          }));
        }
      } else {
        resolve(Err({
          type: 'CORDIAL_FAILED',
          message: stderr || `CORDIAL scoring failed with exit code ${code}`,
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
};

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

export function register(): void {

  // Cancel Vina docking -- kill all docking child processes
  ipcMain.handle(IpcChannels.CANCEL_VINA_DOCKING, async (): Promise<void> => {
    for (const proc of dockingProcesses) {
      if (!proc.killed) {
        proc.kill('SIGTERM');
      }
    }
    dockingProcesses.clear();
  });

  // Run Vina docking -- Node.js-managed parallel execution
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
      if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
        return Err({
          type: 'PYTHON_NOT_FOUND',
          message: 'Python not found. Please install miniconda and create the openmm-metal environment.',
        });
      }

      const scriptPath = path.join(appState.fraggenRoot, 'run_vina_docking.py');
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
      const receptorOutputPath = path.join(inputsDir, 'receptor.pdb');
      const referenceOutputPath = path.join(inputsDir, 'reference_ligand.sdf');
      fs.copyFileSync(receptorPdb, receptorOutputPath);
      fs.copyFileSync(referenceLigand, referenceOutputPath);

      // Write ligands list to inputs/ligands.json
      const ligandsJsonPath = path.join(inputsDir, 'ligands.json');
      fs.writeFileSync(ligandsJsonPath, JSON.stringify(ligandSdfPaths, null, 2));

      // Vina is CPU-only -- concurrency = CPU count (each process uses 1 CPU)
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
          100  // minimal stagger -- OBabel already initialized by first ligand
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

      // Post-processing: move docked files into results/poses/, score the prepared
      // X-ray reference with Vina score_only, then regenerate the pooled SDF from
      // the final pose files.
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

        if (referenceLigand && fs.existsSync(referenceLigand)) {
          try {
            const refName = path.basename(outputDir) + '_xray_reference_docked.sdf.gz';
            const refPosePath = path.join(posesDir, refName);
            const refScore = await scoreReferencePoseVina(
              receptorOutputPath,
              referenceOutputPath,
              refPosePath,
              config
            );
            event.sender.send(IpcChannels.DOCK_OUTPUT, {
              type: 'stdout',
              data: `Reference pose score_only complete: ${refScore.toFixed(2)} kcal/mol\n`,
            });
          } catch (e) {
            console.error('Failed to score X-ray reference pose:', e);
          }
        }

        rebuildDockingPool(resultsDir, posesDir);
      } catch (e) {
        console.error('Post-processing (pooling) failed:', e);
        // Non-fatal -- docking results are still available
      }

      return Ok(outputDir);
    }
  );

  // Parse Vina docking results -- reads *_docked.sdf.gz files from output dir
  ipcMain.handle(
    IpcChannels.PARSE_DOCK_RESULTS,
    async (_event, outputDir: string): Promise<Result<Array<{
      ligandName: string;
      smiles: string;
      qed: number;
      vinaAffinity: number | null;
      vinaScoreOnlyAffinity?: number;
      poseIndex: number;
      outputSdf: string;
      parentMolecule: string;
      protonationVariant: number | null;
      conformerIndex: number | null;
      isReferencePose: boolean;
      refinementEnergy?: number;
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
          const isReferencePose = props.isReferencePose === true || name.includes('xray_reference');
          return {
            ligandName: name,
            smiles: props.smiles || '',
            qed: props.qed,
            vinaAffinity: props.vinaAffinity,
            vinaScoreOnlyAffinity: props.vinaScoreOnlyAffinity,
            poseIndex: 0,
            outputSdf: sdfPath,
            parentMolecule: name,
            protonationVariant: null,
            conformerIndex: null,
            isReferencePose,
            refinementEnergy: props.refinementEnergy,
            coreRmsd: props.coreRmsd,
          };
        });
        const results: Array<any> = await Promise.all(parsePromises);

        loadAndMergeCordialScores(outputDir, results, 'ligandName');

        // Load xTB relative energies if available
        const xtbEnergyPath = path.join(outputDir, 'results', 'xtb_energy.json');
        if (fs.existsSync(xtbEnergyPath)) {
          try {
            const xtbData = JSON.parse(fs.readFileSync(xtbEnergyPath, 'utf-8'));
            for (const result of results) {
              const key = `${result.ligandName}_${result.poseIndex}`;
              if (key in xtbData) {
                result.xtbEnergyKcal = xtbData[key];
              }
            }
          } catch (e) {
            console.error('Failed to load xTB energy scores:', e);
          }
        }

        // Keep docked poses ranked first; append reference poses after the docked ranking.
        results.sort((a: any, b: any) => {
          if (a.isReferencePose !== b.isReferencePose) {
            return a.isReferencePose ? 1 : -1;
          }
          const aScore = a.vinaAffinity ?? a.vinaScoreOnlyAffinity ?? Number.POSITIVE_INFINITY;
          const bScore = b.vinaAffinity ?? b.vinaScoreOnlyAffinity ?? Number.POSITIVE_INFINITY;
          return aScore - bScore;
        });

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
    async (_event, pdbPath: string): Promise<Result<{
      ligands: Array<{
        id: string;
        resname: string;
        chain: string;
        resnum: string;
        num_atoms: number;
        centroid: { x: number; y: number; z: number };
      }>;
      structureInfo?: {
        totalAtoms: number;
        hydrogenCount: number;
        isPrepared: boolean;
      };
    }, AppError>> => {
      return new Promise((resolve) => {
        if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
          resolve(Err({
            type: 'PYTHON_NOT_FOUND',
            message: 'Python not found',
          }));
          return;
        }

        const scriptPath = path.join(appState.fraggenRoot, 'detect_pdb_ligands.py');
        if (!fs.existsSync(scriptPath)) {
          resolve(Err({
            type: 'SCRIPT_NOT_FOUND',
            path: scriptPath,
            message: `Ligand detection script not found: ${scriptPath}`,
          }));
          return;
        }

        const python = spawn(appState.condaPythonPath, [
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
              const parsed = JSON.parse(stdout);
              // Handle both old array format and new object format
              if (Array.isArray(parsed)) {
                resolve(Ok({ ligands: parsed }));
              } else {
                resolve(Ok(parsed));
              }
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
        if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
          resolve(Err({
            type: 'PYTHON_NOT_FOUND',
            message: 'Python not found',
          }));
          return;
        }

        const scriptPath = path.join(appState.fraggenRoot, 'detect_pdb_ligands.py');
        if (!fs.existsSync(scriptPath)) {
          resolve(Err({
            type: 'SCRIPT_NOT_FOUND',
            path: scriptPath,
            message: `Ligand extraction script not found: ${scriptPath}`,
          }));
          return;
        }

        fs.mkdirSync(path.dirname(outputPath), { recursive: true });

        const python = spawn(appState.condaPythonPath, [
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
    async (
      _event,
      pdbPath: string,
      ligandId: string,
      outputPath: string,
      waterDistance: number = 0,
      protonationPh: number = 7.4
    ): Promise<Result<string, AppError>> => {
      return new Promise((resolve) => {
        if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
          resolve(Err({
            type: 'PYTHON_NOT_FOUND',
            message: 'Python not found',
          }));
          return;
        }

        const scriptPath = path.join(appState.fraggenRoot, 'detect_pdb_ligands.py');
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
          '--ph', String(protonationPh),
        ];

        const python = spawn(appState.condaPythonPath, prepArgs);

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

  // Prepare docking complex
  ipcMain.handle(
    IpcChannels.PREPARE_DOCKING_COMPLEX,
    async (
      event,
      receptorPdb: string,
      xrayLigandSdf: string,
      outputDir: string,
      chargeMethod: 'gasteiger' | 'am1bcc' = 'am1bcc',
      phMin: number = 6.4,
      phMax: number = 8.4,
      protonateReference: boolean = true
    ): Promise<Result<PreparedComplexRunResult, AppError>> => {
      return new Promise((resolve) => {
        if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
          resolve(Err({
            type: 'PYTHON_NOT_FOUND',
            message: 'Python not found',
          }));
          return;
        }

        const scriptPath = path.join(appState.fraggenRoot, 'prepare_docking_complex.py');
        if (!fs.existsSync(scriptPath)) {
          resolve(Err({
            type: 'SCRIPT_NOT_FOUND',
            path: scriptPath,
            message: `Prepared complex script not found: ${scriptPath}`,
          }));
          return;
        }

        fs.mkdirSync(outputDir, { recursive: true });
        const args = [
          scriptPath,
          '--receptor_pdb', receptorPdb,
          '--xray_ligand_sdf', xrayLigandSdf,
          '--output_dir', outputDir,
          '--charge_method', chargeMethod,
          '--ph_min', String(phMin),
          '--ph_max', String(phMax),
          ...(protonateReference ? [] : ['--skip_reference_protonation']),
        ];

        event.sender.send(IpcChannels.DOCK_OUTPUT, {
          type: 'stdout',
          data: `=== Preparing Docking Complex ===\nReceptor: ${path.basename(receptorPdb)}\nReference ligand: ${path.basename(xrayLigandSdf)}\nReference protonation: ${protonateReference ? 'enabled' : 'disabled'}\nCharges: ${chargeMethod}\npH range: ${phMin}-${phMax}\n\n`,
        });

        const python = spawn(appState.condaPythonPath, args);
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
          if (text.includes('Warning') || text.includes('ERROR') || text.includes('Traceback')) {
            event.sender.send(IpcChannels.DOCK_OUTPUT, { type: 'stderr', data: text });
          }
        });

        python.on('close', (code: number | null) => {
          childProcesses.delete(python);
          if (code === 0) {
            try {
              const lines = stdout.trim().split('\n');
              const lastLine = lines[lines.length - 1];
              if (!lastLine || !lastLine.startsWith('{')) {
                throw new Error('Missing JSON result from prepared complex script');
              }
              const result = JSON.parse(lastLine);
              const manifestPath = result.manifest_path as string;
              const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8')) as PreparedComplexManifest;
              resolve(Ok({
                manifestPath,
                preparedReceptorPdb: result.prepared_receptor_pdb,
                preparedReferenceLigandSdf: result.prepared_reference_ligand_sdf,
                manifest,
              }));
            } catch (error) {
              resolve(Err({
                type: 'PARSE_FAILED',
                message: `Failed to parse prepared complex results: ${(error as Error).message}`,
              }));
            }
          } else {
            resolve(Err({
              type: 'PARSE_FAILED',
              message: stderr || 'prepare_docking_complex.py failed',
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

  // Export docking results to CSV -- done in Node.js (no Python needed)
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
        const header = ['ligand', 'is_reference_pose', 'vina_affinity', 'vina_score_only_affinity', 'refinement_energy', 'qed', 'smiles', 'sdf_path'];

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
            String(props.isReferencePose === true),
            props.vinaAffinity != null ? String(props.vinaAffinity) : '',
            props.vinaScoreOnlyAffinity != null ? String(props.vinaScoreOnlyAffinity) : '',
            props.refinementEnergy != null ? String(props.refinementEnergy) : '',
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
        if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
          resolve(Err({
            type: 'PYTHON_NOT_FOUND',
            message: 'Python not found',
          }));
          return;
        }

        const scriptPath = path.join(appState.fraggenRoot, 'export_complex_pdb.py');
        if (!fs.existsSync(scriptPath)) {
          resolve(Err({
            type: 'SCRIPT_NOT_FOUND',
            path: scriptPath,
            message: `Complex export script not found: ${scriptPath}`,
          }));
          return;
        }

        const python = spawn(appState.condaPythonPath, [
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

  // Post-docking pocket refinement (OpenMM + Sage 2.3.0 + OBC2)
  ipcMain.handle(
    'dock:refine-poses',
    async (
      event,
      receptorPdb: string,
      posesDir: string,
      outputDir: string,
      maxIterations: number,
      chargeMethod?: string
    ): Promise<Result<{ refinedCount: number; outputDir: string }, AppError>> => {
      return new Promise((resolve) => {
        if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
          resolve(Err({ type: 'PYTHON_NOT_FOUND', message: 'Python not found' }));
          return;
        }

        const scriptPath = path.join(appState.fraggenRoot, 'refine_poses.py');
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
          '--charge_method', chargeMethod || 'am1bcc',
        ];

        event.sender.send(IpcChannels.DOCK_OUTPUT, {
          type: 'stdout',
          data: `=== Pocket Refinement (Sage 2.3.0 + OBC2) ===\nReceptor: ${path.basename(receptorPdb)}\nPoses: ${posesDir}\nMax iterations: ${maxIterations}\n\n`
        });

        const envVars = { ...process.env };
        if (appState.condaEnvBin) {
          envVars.PATH = `${appState.condaEnvBin}:${envVars.PATH || ''}`;
        }

        const python = spawn(appState.condaPythonPath, args, { env: envVars });
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
          // Only show warnings/errors -- suppress Metal transform noise and OpenMM debug output
          if (text.includes('Warning') || text.includes('ERROR') || text.includes('Traceback')) {
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
                rebuildDockingPool(path.dirname(outputDir), outputDir);
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

  // Check if QupKake is installed
  ipcMain.handle(IpcChannels.CHECK_QUPKAKE_INSTALLED, async (): Promise<QupkakeCapabilityResult> => {
    if (appState.qupkakeCapabilityCache) {
      return appState.qupkakeCapabilityCache;
    }

    const pythonPath = appState.condaPythonPath;
    if (!pythonPath || !fs.existsSync(pythonPath)) {
      return {
        available: false,
        validated: false,
        message: 'Primary app Python environment not found, so the QupKake wrapper cannot run.',
      };
    }

    const scriptPath = path.join(appState.fraggenRoot, 'predict_ligand_pka.py');
    if (!fs.existsSync(scriptPath)) {
      return {
        available: false,
        validated: false,
        message: `Ligand pKa script not found: ${scriptPath}`,
      };
    }

    return await new Promise((resolve) => {
      const proc: ChildProcess = spawn(pythonPath, [scriptPath, '--check'], { env: getQupkakeSpawnEnv() });
      childProcesses.add(proc);

      let stdout = '';
      let stderr = '';

      proc.stdout?.on('data', (data: Buffer) => {
        stdout += data.toString();
      });

      proc.stderr?.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      proc.on('close', (code: number | null) => {
        childProcesses.delete(proc);
        if (code !== 0) {
          resolve({
            available: false,
            validated: false,
            message: stderr || `QupKake capability check failed with exit code ${code}`,
          });
          return;
        }

        try {
          const result = JSON.parse(stdout.trim()) as QupkakeCapabilityResult;
          appState.setQupkakeCapabilityCache(result);
          resolve(result);
        } catch {
          console.warn('[QupKake] Failed to parse availability output:', stderr || stdout);
          resolve({
            available: false,
            validated: false,
            message: stderr || stdout || 'Failed to parse QupKake capability output.',
          });
        }
      });

      proc.on('error', (error: Error) => {
        childProcesses.delete(proc);
        resolve({
          available: false,
          validated: false,
          message: `Failed to start QupKake capability check: ${error.message}`,
        });
      });
    });
  });

  // Predict ligand pKa via QupKake
  ipcMain.handle(
    IpcChannels.PREDICT_LIGAND_PKA,
    async (_event, ligandPath: string): Promise<Result<LigandPkaResult, AppError>> => {
      const pythonPath = appState.condaPythonPath;
      if (!pythonPath || !fs.existsSync(pythonPath)) {
        return Err({
          type: 'PYTHON_NOT_FOUND',
          message: 'Primary app Python environment not found, so the QupKake wrapper cannot run.',
        });
      }

      const scriptPath = path.join(appState.fraggenRoot, 'predict_ligand_pka.py');
      if (!fs.existsSync(scriptPath)) {
        return Err({
          type: 'SCRIPT_NOT_FOUND',
          path: scriptPath,
          message: `Ligand pKa script not found: ${scriptPath}`,
        });
      }

      return await new Promise((resolve) => {
        const proc: ChildProcess = spawn(pythonPath, [scriptPath, '--ligand', ligandPath], { env: getQupkakeSpawnEnv() });
        childProcesses.add(proc);

        let stdout = '';
        let stderr = '';

        proc.stdout?.on('data', (data: Buffer) => {
          stdout += data.toString();
        });

        proc.stderr?.on('data', (data: Buffer) => {
          stderr += data.toString();
        });

        proc.on('close', (code: number | null) => {
          childProcesses.delete(proc);

          if (code !== 0) {
            resolve(Err({
              type: 'QUPKAKE_FAILED',
              message: stderr || `QupKake prediction failed with exit code ${code}`,
            }));
            return;
          }

          try {
            const result = JSON.parse(stdout.trim()) as LigandPkaResult & { error?: string };
            if (result?.error) {
              resolve(Err({
                type: 'QUPKAKE_FAILED',
                message: result.error,
              }));
              return;
            }
            resolve(Ok(result));
          } catch {
            resolve(Err({
              type: 'PARSE_FAILED',
              message: `Failed to parse QupKake output: ${stderr || stdout}`,
            }));
          }
        });

        proc.on('error', (err: Error) => {
          childProcesses.delete(proc);
          resolve(Err({
            type: 'QUPKAKE_FAILED',
            message: `Failed to start QupKake prediction: ${err.message}`,
          }));
        });
      });
    }
  );

  // Score a single protein-ligand complex (viewer scoring)
  ipcMain.handle(
    IpcChannels.SCORE_COMPLEX,
    async (
      _event,
      pdbPath: string,
      ligandSdfPath?: string
    ): Promise<Result<{
      vinaRescore?: number;
      xtbStrainKcal?: number;
      cordialExpectedPkd?: number;
      cordialPHighAffinity?: number;
      cordialPVeryHighAffinity?: number;
    }, AppError>> => {
      console.log(`[Score] Scoring complex: ${pdbPath}${ligandSdfPath ? ` + ${ligandSdfPath}` : ''}`);

      if (!appState.condaPythonPath) {
        console.error('[Score] Python not found');
        return Err({ type: 'PYTHON_NOT_FOUND', message: 'Python not found' });
      }

      const result: {
        vinaRescore?: number;
        xtbStrainKcal?: number;
        cordialExpectedPkd?: number;
        cordialPHighAffinity?: number;
        cordialPVeryHighAffinity?: number;
      } = {};

      const xtbPath = getQupkakeXtbPath();
      const strainScript = path.join(appState.fraggenRoot, 'score_xtb_strain.py');
      const vinaScript = path.join(appState.fraggenRoot, 'run_vina_docking.py');

      const tasks: Promise<void>[] = [];

      if (ligandSdfPath && xtbPath && fs.existsSync(strainScript)) {
        tasks.push((async () => {
          try {
            console.log('[Score] Running xTB strain...');
            const { stdout, code } = await spawnPythonScript([
              strainScript, '--ligand', ligandSdfPath!, '--xtb_binary', xtbPath, '--mode', 'strain',
            ]);
            if (code !== 0) {
              console.error(`[Score] xTB strain failed (exit ${code})`);
              return;
            }
            const match = stdout.match(/XTB_STRAIN:([-\d.]+)/);
            if (match) {
              result.xtbStrainKcal = Math.round(parseFloat(match[1]) * 10) / 10;
              console.log(`[Score] xTB strain: ${result.xtbStrainKcal} kcal/mol`);
            }
          } catch (e) {
            console.error('[Score] xTB strain error:', e);
          }
        })());
      }

      if (ligandSdfPath && fs.existsSync(pdbPath) && fs.existsSync(vinaScript)) {
        tasks.push((async () => {
          try {
            console.log('[Score] Running Vina score_only...');
            const vinaResult = await runVinaScoreOnly(pdbPath, ligandSdfPath!, ligandSdfPath!, {
              autoboxAdd: 4,
              cpu: 1,
            });
            if (vinaResult.ok) {
              result.vinaRescore = vinaResult.value;
              console.log(`[Score] Vina rescore: ${result.vinaRescore} kcal/mol`);
            } else {
              console.error('[Score] Vina score_only failed:', vinaResult.error.message);
            }
          } catch (e) {
            console.error('[Score] Vina score_only error:', e);
          }
        })());
      }

      await Promise.all(tasks);
      console.log('[Score] Complete:', JSON.stringify(result));
      return Ok(result);
    }
  );

  // Pre-optimize docking ligands with xTB before Vina
  ipcMain.handle(
    IpcChannels.PREOPTIMIZE_DOCK_LIGANDS,
    async (
      event,
      ligandSdfPaths: string[],
      outputDir: string
    ): Promise<Result<{
      optimizedLigandPaths: string[];
      optimizedCount: number;
      failedCount: number;
    }, AppError>> => {
      if (ligandSdfPaths.length === 0) {
        return Ok({ optimizedLigandPaths: [], optimizedCount: 0, failedCount: 0 });
      }

      const xtbPath = getQupkakeXtbPath();
      if (!xtbPath) {
        return Err({ type: 'DOCKING_FAILED', message: 'xTB binary not found' });
      }

      if (!appState.condaPythonPath) {
        return Err({ type: 'PYTHON_NOT_FOUND', message: 'Python not found' });
      }

      const scriptPath = path.join(appState.fraggenRoot, 'score_xtb_strain.py');
      if (!fs.existsSync(scriptPath)) {
        return Err({ type: 'SCRIPT_NOT_FOUND', path: scriptPath, message: 'score_xtb_strain.py not found' });
      }

      fs.mkdirSync(outputDir, { recursive: true });

      event.sender.send(IpcChannels.DOCK_OUTPUT, {
        type: 'stdout',
        data: `=== xTB Ligand Pre-optimization ===\nLigands: ${ligandSdfPaths.length}\nOutput: ${outputDir}\n\n`,
      });

      const optimizedLigandPaths: string[] = [];
      let optimizedCount = 0;
      let failedCount = 0;

      for (const [index, ligandPath] of ligandSdfPaths.entries()) {
        const baseName = path.basename(ligandPath).replace(/(\.sdf\.gz|\.sdf)$/i, '');
        const optimizedPath = path.join(
          outputDir,
          `${String(index + 1).padStart(4, '0')}_${baseName}_xtbopt.sdf`,
        );

        event.sender.send(IpcChannels.DOCK_OUTPUT, {
          type: 'stdout',
          data: `XTB_PREOPT: ${index + 1}/${ligandSdfPaths.length} ${path.basename(ligandPath)}\n`,
        });

        try {
          const { stdout, stderr, code } = await spawnPythonScript([
            scriptPath,
            '--xtb_binary', xtbPath,
            '--mode', 'optimize',
            '--ligand', ligandPath,
            '--output_sdf', optimizedPath,
          ]);

          if (code === 0 && fs.existsSync(optimizedPath)) {
            optimizedLigandPaths.push(optimizedPath);
            optimizedCount += 1;
            continue;
          }

          failedCount += 1;
          optimizedLigandPaths.push(ligandPath);
          const detail = (stderr || stdout || `exit ${code}`).trim().slice(0, 240);
          event.sender.send(IpcChannels.DOCK_OUTPUT, {
            type: 'stderr',
            data: `  Warning: xTB pre-optimization failed for ${path.basename(ligandPath)}; using original geometry (${detail})\n`,
          });
        } catch (error) {
          failedCount += 1;
          optimizedLigandPaths.push(ligandPath);
          event.sender.send(IpcChannels.DOCK_OUTPUT, {
            type: 'stderr',
            data: `  Warning: xTB pre-optimization failed for ${path.basename(ligandPath)}; using original geometry (${(error as Error).message})\n`,
          });
        }
      }

      event.sender.send(IpcChannels.DOCK_OUTPUT, {
        type: 'stdout',
        data: `xTB pre-optimization finished: ${optimizedCount} optimized, ${failedCount} fallback\n`,
      });

      return Ok({ optimizedLigandPaths, optimizedCount, failedCount });
    }
  );

  // Score docked poses with xTB single-point energy (relative per compound)
  ipcMain.handle(
    IpcChannels.SCORE_DOCKING_XTB_ENERGY,
    async (
      event,
      dockOutputDir: string
    ): Promise<Result<{ count: number }, AppError>> => {
      const xtbPath = getQupkakeXtbPath();
      if (!xtbPath) {
        return Err({ type: 'DOCKING_FAILED', message: 'xTB binary not found' });
      }
      if (!appState.condaPythonPath) {
        return Err({ type: 'PYTHON_NOT_FOUND', message: 'Python not found' });
      }

      const scriptPath = path.join(appState.fraggenRoot, 'score_xtb_strain.py');
      if (!fs.existsSync(scriptPath)) {
        return Err({ type: 'SCRIPT_NOT_FOUND', path: scriptPath, message: 'score_xtb_strain.py not found' });
      }

      const posesDir = path.join(dockOutputDir, 'results', 'poses');
      if (!fs.existsSync(posesDir)) {
        return Err({ type: 'FILE_NOT_FOUND', path: posesDir, message: 'No poses directory found' });
      }

      const resultsDir = path.join(dockOutputDir, 'results');
      const outputJson = path.join(resultsDir, 'xtb_energy.json');

      event.sender.send(IpcChannels.DOCK_OUTPUT, {
        type: 'stdout', data: `=== xTB Energy Scoring ===\n`,
      });

      return new Promise((resolve) => {
        const args = [
          scriptPath,
          '--xtb_binary', xtbPath,
          '--mode', 'batch_energy',
          '--ligand_dir', posesDir,
          '--output_json', outputJson,
        ];

        const python = spawn(appState.condaPythonPath!, args);
        childProcesses.add(python);

        let stdout = '';
        python.stdout.on('data', (data: Buffer) => {
          const text = data.toString();
          stdout += text;
          event.sender.send(IpcChannels.DOCK_OUTPUT, { type: 'stdout', data: text });
        });
        python.stderr.on('data', (data: Buffer) => {
          event.sender.send(IpcChannels.DOCK_OUTPUT, { type: 'stderr', data: data.toString() });
        });

        python.on('close', (code: number | null) => {
          childProcesses.delete(python);
          if (code === 0) {
            const match = stdout.match(/BATCH_ENERGY_JSON:(.+)/);
            const jsonPath = match ? match[1].trim() : outputJson;
            try {
              const data = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
              resolve(Ok({ count: Object.keys(data).length }));
            } catch {
              resolve(Ok({ count: 0 }));
            }
          } else {
            resolve(Err({ type: 'DOCKING_FAILED', message: `xTB energy scoring failed (exit ${code})` }));
          }
        });

        python.on('error', (error: Error) => {
          childProcesses.delete(python);
          resolve(Err({ type: 'DOCKING_FAILED', message: error.message }));
        });
      });
    }
  );

  // Run CORDIAL scoring on docked poses
  ipcMain.handle(
    IpcChannels.RUN_CORDIAL_SCORING,
    async (
      event,
      dockOutputDir: string,
      batchSize: number = 32
    ): Promise<Result<{ scoresFile: string; count: number }, AppError>> => {
      const cordialResultsDir = path.join(dockOutputDir, 'results');
      fs.mkdirSync(cordialResultsDir, { recursive: true });
      const outputCsv = path.join(cordialResultsDir, 'cordial_scores.csv');
      const cordialRoot = getCordialRoot() || '(missing)';
      event.sender.send(IpcChannels.DOCK_OUTPUT, {
        type: 'stdout',
        data: `=== CORDIAL Rescoring ===\nCORDIAL Root: ${cordialRoot}\nDock Output: ${dockOutputDir}\nBatch Size: ${batchSize}\n\n`,
      });

      const result = await runCordialScoringJob(
        { dockDir: dockOutputDir },
        outputCsv,
        batchSize,
        {
          onStdout: (text) => {
            event.sender.send(IpcChannels.DOCK_OUTPUT, { type: 'stdout', data: text });
          },
          onStderr: (text) => {
            if (!text.toLowerCase().includes('error') && !text.toLowerCase().includes('traceback')) {
              event.sender.send(IpcChannels.DOCK_OUTPUT, { type: 'stdout', data: text });
            } else {
              event.sender.send(IpcChannels.DOCK_OUTPUT, { type: 'stderr', data: text });
            }
          },
        },
      );

      if (result.ok) {
        event.sender.send(IpcChannels.DOCK_OUTPUT, {
          type: 'stdout',
          data: `\n=== CORDIAL Scoring Complete ===\nScored ${result.value.count} poses\nOutput: ${outputCsv}\n`,
        });
      }

      return result;
    }
  );

  // Score MD clusters (cluster + Vina + xTB + CORDIAL pipeline)
  ipcMain.handle(
    IpcChannels.SCORE_MD_CLUSTERS,
    async (
      event,
      options: {
        topologyPath: string;
        trajectoryPath: string;
        outputDir: string;
        inputLigandSdf: string;
        inputReceptorPdb?: string;
        numClusters: number;
        enableVina: boolean;
        enableCordial: boolean;
      }
    ): Promise<Result<{
      clusters: ScoredClusterResult[];
      outputDir: string;
      clusteringResults: ClusteringResult;
    }, AppError>> => {
      if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
        return Err({
          type: 'PYTHON_NOT_FOUND',
          message: 'Python not found. Please install miniconda and create the openmm-metal environment.',
        });
      }

      const analysisDir = options.outputDir;
      const clusteringDir = path.join(analysisDir, 'clustering');
      const scoredClustersDir = path.join(analysisDir, 'scored_clusters');
      fs.mkdirSync(clusteringDir, { recursive: true });
      fs.mkdirSync(scoredClustersDir, { recursive: true });

      // --- Step 1: Create or reuse canonical clustering output ---
      let clusteringResults = readClusteringResult(clusteringDir);
      if (!clusteringResults || clusteringResults.clusters.length === 0) {
        const clusterScript = path.join(appState.fraggenRoot, 'cluster_trajectory.py');
        if (!fs.existsSync(clusterScript)) {
          return Err({
            type: 'SCRIPT_NOT_FOUND',
            path: clusterScript,
            message: `Clustering script not found: ${clusterScript}`,
          });
        }

        event.sender.send(IpcChannels.MD_OUTPUT, {
          type: 'stdout',
          data: `=== Clustering into ${options.numClusters} centroids ===\n`,
        });

        const clusterResult = await new Promise<Result<void, AppError>>((resolve) => {
          const args = [
            clusterScript,
            '--topology', options.topologyPath,
            '--trajectory', options.trajectoryPath,
            '--n_clusters', String(options.numClusters),
            '--method', 'kmeans',
            '--selection', 'ligand',
            '--strip_waters',
            '--output_dir', clusteringDir,
          ];

          const proc = spawn(appState.condaPythonPath!, args, { env: getSpawnEnv() });
          childProcesses.add(proc);

          proc.stdout?.on('data', (data: Buffer) => {
            const text = data.toString();
            event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stdout', data: text });
            const match = text.match(/Calculated (\d+)\/(\d+)/);
            if (match) {
              const pct = Math.round(100 * parseInt(match[1], 10) / parseInt(match[2], 10));
              event.sender.send(IpcChannels.MD_OUTPUT, {
                type: 'stdout', data: `PROGRESS:clustering:${pct}\n`,
              });
            }
          });

          proc.stderr?.on('data', (data: Buffer) => {
            event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stderr', data: data.toString() });
          });

          proc.on('close', (code: number | null) => {
            childProcesses.delete(proc);
            if (code === 0) {
              resolve(Ok(undefined));
            } else {
              resolve(Err({ type: 'CLUSTERING_FAILED', message: `Clustering failed with exit code ${code}` }));
            }
          });

          proc.on('error', (err: Error) => {
            childProcesses.delete(proc);
            resolve(Err({ type: 'CLUSTERING_FAILED', message: err.message }));
          });
        });

        if (!clusterResult.ok) {
          return Err(clusterResult.error);
        }

        clusteringResults = readClusteringResult(clusteringDir);
      } else {
        event.sender.send(IpcChannels.MD_OUTPUT, {
          type: 'stdout',
          data: `=== Reusing existing clustering (${clusteringResults.clusters.length} centroids) ===\nPROGRESS:clustering:100\n`,
        });
      }

      if (!clusteringResults || clusteringResults.clusters.length === 0) {
        return Err({
          type: 'CLUSTERING_FAILED',
          message: 'Canonical clustering results were not found after clustering completed',
        });
      }

      // --- Step 2: Prepare centroid receptor/ligand pairs ---
      const scoreScript = path.join(appState.fraggenRoot, 'score_cluster_centroids.py');
      if (!fs.existsSync(scoreScript)) {
        return Err({
          type: 'SCRIPT_NOT_FOUND',
          path: scoreScript,
          message: `Cluster preparation script not found: ${scoreScript}`,
        });
      }

      event.sender.send(IpcChannels.MD_OUTPUT, {
        type: 'stdout',
        data: '=== Preparing cluster centroids for rescoring ===\n',
      });

      const prepareResult = await new Promise<Result<void, AppError>>((resolve) => {
        const args = [
          scoreScript,
          '--clustering_dir', clusteringDir,
          '--input_ligand_sdf', options.inputLigandSdf,
          '--output_dir', scoredClustersDir,
        ];

        const proc = spawn(appState.condaPythonPath!, args, { env: getSpawnEnv() });
        childProcesses.add(proc);

        proc.stdout?.on('data', (data: Buffer) => {
          event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stdout', data: data.toString() });
        });

        proc.stderr?.on('data', (data: Buffer) => {
          event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stderr', data: data.toString() });
        });

        proc.on('close', (code: number | null) => {
          childProcesses.delete(proc);
          if (code === 0) {
            resolve(Ok(undefined));
          } else {
            resolve(Err({
              type: 'CLUSTER_SCORING_FAILED',
              message: `Cluster preparation failed with exit code ${code}`,
            }));
          }
        });

        proc.on('error', (err: Error) => {
          childProcesses.delete(proc);
          resolve(Err({ type: 'CLUSTER_SCORING_FAILED', message: err.message }));
        });
      });

      if (!prepareResult.ok) {
        return Err(prepareResult.error);
      }

      let scoredClusters = readClusterScoreRows(scoredClustersDir);
      if (scoredClusters.length !== clusteringResults.clusters.length) {
        return Err({
          type: 'CLUSTER_SCORING_FAILED',
          message: `Expected ${clusteringResults.clusters.length} prepared clusters, found ${scoredClusters.length}`,
        });
      }

      for (const cluster of scoredClusters) {
        if (!cluster.receptorPdbPath || !cluster.ligandSdfPath) {
          return Err({
            type: 'CLUSTER_SCORING_FAILED',
            message: `Cluster ${cluster.clusterId + 1} is missing prepared receptor/ligand files`,
          });
        }
      }

      // --- Step 3: Vina fixed-pose rescoring ---
      if (options.enableVina) {
        event.sender.send(IpcChannels.MD_OUTPUT, {
          type: 'stdout',
          data: '=== Vina rescoring cluster centroids ===\n',
        });

        for (let i = 0; i < scoredClusters.length; i++) {
          const cluster = scoredClusters[i];
          const vinaResult = await runVinaScoreOnly(
            cluster.receptorPdbPath!,
            cluster.ligandSdfPath!,
            cluster.ligandSdfPath!,
            {
              autoboxAdd: 4,
              cpu: 1,
              onStdout: (text) => {
                event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stdout', data: text });
              },
              onStderr: (text) => {
                event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stderr', data: text });
              },
            },
          );
          if (!vinaResult.ok) {
            return Err({
              type: 'CLUSTER_SCORING_FAILED',
              message: `Vina rescoring failed for cluster ${cluster.clusterId + 1}: ${vinaResult.error.message}`,
            });
          }
          cluster.vinaRescore = Math.round(vinaResult.value * 100) / 100;
          const pct = Math.round(100 * (i + 1) / scoredClusters.length);
          event.sender.send(IpcChannels.MD_OUTPUT, {
            type: 'stdout',
            data: `PROGRESS:scoring_vina:${pct}\n`,
          });
        }
        writeClusterScoreRows(scoredClustersDir, scoredClusters);
      } else {
        return Err({
          type: 'CLUSTER_SCORING_FAILED',
          message: 'Holo MD rescoring requires Vina, but Vina rescoring is disabled',
        });
      }

      // --- Step 3.5: xTB relative energy scoring ---
      const xtbPath = getQupkakeXtbPath();
      if (xtbPath) {
        event.sender.send(IpcChannels.MD_OUTPUT, {
          type: 'stdout',
          data: '=== xTB energy scoring cluster centroids ===\n',
        });

        const xtbScript = path.join(appState.fraggenRoot, 'score_xtb_strain.py');
        if (fs.existsSync(xtbScript)) {
          // Collect ligand SDF paths from prepared clusters
          const ligandSdfs = scoredClusters
            .filter((c) => c.ligandSdfPath && fs.existsSync(c.ligandSdfPath!))
            .map((c) => c.ligandSdfPath!);

          if (ligandSdfs.length > 0) {
            // Write temp ligand list for batch processing
            const xtbLigandDir = path.join(scoredClustersDir, '_xtb_ligands');
            fs.mkdirSync(xtbLigandDir, { recursive: true });
            for (const sdf of ligandSdfs) {
              const dest = path.join(xtbLigandDir, path.basename(sdf));
              if (!fs.existsSync(dest)) fs.copyFileSync(sdf, dest);
            }

            const xtbOutputJson = path.join(scoredClustersDir, 'xtb_energy.json');
            try {
              const { stdout, code } = await spawnPythonScript([
                xtbScript,
                '--xtb_binary', xtbPath,
                '--mode', 'batch_energy',
                '--ligand_dir', xtbLigandDir,
                '--output_json', xtbOutputJson,
              ]);
              if (code === 0 && fs.existsSync(xtbOutputJson)) {
                const xtbData = JSON.parse(fs.readFileSync(xtbOutputJson, 'utf-8'));
                for (const cluster of scoredClusters) {
                  if (!cluster.ligandSdfPath) continue;
                  const baseName = path.basename(cluster.ligandSdfPath).replace(/\.sdf(\.gz)?$/, '');
                  const key = `${baseName}_0`;
                  if (key in xtbData) {
                    cluster.xtbStrainKcal = xtbData[key];
                  }
                }
                writeClusterScoreRows(scoredClustersDir, scoredClusters);
                event.sender.send(IpcChannels.MD_OUTPUT, {
                  type: 'stdout',
                  data: `xTB energy scoring complete: ${Object.keys(xtbData).length} centroids\nPROGRESS:scoring_xtb:100\n`,
                });
              }
            } catch (e) {
              event.sender.send(IpcChannels.MD_OUTPUT, {
                type: 'stderr',
                data: `xTB scoring warning: ${(e as Error).message}\n`,
              });
            }
            // Clean up temp dir
            try { fs.rmSync(xtbLigandDir, { recursive: true }); } catch { /* */ }
          }
        }
      }

      // --- Step 4: CORDIAL rescoring ---
      if (options.enableCordial) {
        event.sender.send(IpcChannels.MD_OUTPUT, {
          type: 'stdout',
          data: '=== CORDIAL rescoring cluster centroids ===\n',
        });

        const pairCsvPath = path.join(scoredClustersDir, 'pairs.csv');
        const csvCell = (value: string) => `"${value.replace(/"/g, '""')}"`;
        const pairCsvRows = [
          'source_name,ligand_sdf,receptor_pdb,pose_index',
          ...scoredClusters.map((cluster) => [
            csvCell(`cluster_${cluster.clusterId}`),
            csvCell(cluster.ligandSdfPath!),
            csvCell(cluster.receptorPdbPath!),
            csvCell('0'),
          ].join(',')),
        ];
        fs.writeFileSync(pairCsvPath, `${pairCsvRows.join('\n')}\n`);

        const cordialResult = await runCordialScoringJob(
          { pairCsv: pairCsvPath },
          path.join(scoredClustersDir, 'cordial_scores.csv'),
          32,
          {
            onStdout: (text) => {
              event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stdout', data: text });
            },
            onStderr: (text) => {
              event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stderr', data: text });
            },
          },
        );
        if (!cordialResult.ok) {
          return Err({
            type: 'CORDIAL_FAILED',
            message: cordialResult.error.message,
          });
        }

        event.sender.send(IpcChannels.MD_OUTPUT, {
          type: 'stdout',
          data: 'PROGRESS:scoring_cordial:100\n',
        });
      } else {
        return Err({
          type: 'CORDIAL_FAILED',
          message: 'Holo MD rescoring requires CORDIAL, but CORDIAL rescoring is disabled',
        });
      }

      // --- Step 5: Read and validate all results ---
      const cordialJsonPath = path.join(scoredClustersDir, 'cordial_scores.json');
      if (!fs.existsSync(cordialJsonPath)) {
        return Err({
          type: 'CORDIAL_FAILED',
          message: `CORDIAL output JSON not found: ${cordialJsonPath}`,
        });
      }

      try {
        const cordialData = JSON.parse(fs.readFileSync(cordialJsonPath, 'utf-8'));
        const cordialByName = new Map<string, {
          expectedPkd: number;
          pHighAffinity: number;
          pVeryHighAffinity: number;
        }>();

        for (const entry of cordialData) {
          const name = entry.source_name;
          cordialByName.set(name, {
            expectedPkd: entry.cordial_expected_pkd,
            pHighAffinity: entry.cordial_p_high_affinity,
            pVeryHighAffinity: entry.cordial_p_very_high ?? entry.cordial_p_very_high_affinity ?? 0,
          });
        }

        for (const cluster of scoredClusters) {
          const cordialKey = `cluster_${cluster.clusterId}`;
          const scores = cordialByName.get(cordialKey);
          if (!scores) {
            return Err({
              type: 'CORDIAL_FAILED',
              message: `Missing CORDIAL scores for cluster ${cluster.clusterId + 1}`,
            });
          }
          cluster.cordialExpectedPkd = scores.expectedPkd;
          cluster.cordialPHighAffinity = scores.pHighAffinity;
          cluster.cordialPVeryHighAffinity = scores.pVeryHighAffinity;
        }
      } catch (err) {
        return Err({
          type: 'CORDIAL_FAILED',
          message: `Failed to parse CORDIAL output: ${err}`,
        });
      }

      for (const cluster of scoredClusters) {
        if (
          !cluster.receptorPdbPath ||
          !cluster.ligandSdfPath ||
          cluster.vinaRescore == null ||
          cluster.cordialExpectedPkd == null ||
          cluster.cordialPHighAffinity == null ||
          cluster.cordialPVeryHighAffinity == null
        ) {
          return Err({
            type: 'CLUSTER_SCORING_FAILED',
            message: `Cluster ${cluster.clusterId + 1} is missing required rescoring fields`,
          });
        }
      }

      writeClusterScoreRows(scoredClustersDir, scoredClusters);

      event.sender.send(IpcChannels.MD_OUTPUT, {
        type: 'stdout',
        data: `PROGRESS:scoring:100\n=== Cluster scoring complete ===\n`,
      });

      return Ok({
        clusters: mergeClusterScoresWithCanonical(clusteringResults, scoredClusters),
        outputDir: analysisDir,
        clusteringResults,
      });
    }
  );
}
