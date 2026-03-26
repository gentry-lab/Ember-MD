// Copyright (c) 2026 Ember Contributors. MIT License.
/**
 * Shared scoring utilities used by docking, viewer, and score IPC modules.
 *
 * Extracted from docking.ts/viewer.ts to avoid duplication.
 * Functions here close over appState — callers just import and call.
 */
import { spawn } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { Ok, Err, Result } from '../shared/types/result';
import { AppError } from '../shared/types/errors';
import * as appState from './app-state';
import { childProcesses, getSpawnEnv as _getSpawnEnv } from './spawn';
import { spawnPythonScript as _spawnPythonScriptRaw } from './spawn';
import { getCordialRoot, detectBabelDataDir } from './paths';

// ---------------------------------------------------------------------------
// Local convenience wrappers
// ---------------------------------------------------------------------------

function getSpawnEnv(): NodeJS.ProcessEnv {
  return _getSpawnEnv(appState.condaEnvBin);
}

function spawnPythonScript(
  args: string[],
  options?: {
    env?: NodeJS.ProcessEnv;
    cwd?: string;
    onStdout?: (text: string) => void;
    onStderr?: (text: string) => void;
  },
): Promise<{ stdout: string; stderr: string; code: number }> {
  return _spawnPythonScriptRaw(appState.condaPythonPath, appState.condaEnvBin, args, options);
}

// ---------------------------------------------------------------------------
// Vina score_only
// ---------------------------------------------------------------------------

const resolveVinaScriptPath = (): string =>
  path.join(appState.fraggenRoot, 'run_vina_docking.py');

export const runVinaScoreOnly = async (
  receptorPath: string,
  ligandPath: string,
  referencePath: string,
  options?: {
    outputSdfGz?: string;
    autoboxAdd?: number;
    cpu?: number;
    seed?: number;
    env?: NodeJS.ProcessEnv;
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
      ...options?.env,
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

// ---------------------------------------------------------------------------
// CORDIAL rescoring
// ---------------------------------------------------------------------------

const resolveCordialScriptPath = (): string | null => {
  const projectRoot = path.resolve(__dirname, '..');
  const candidates = [
    path.join(appState.fraggenRoot, 'score_cordial.py'),
    path.join(process.resourcesPath, 'scripts', 'score_cordial.py'),
    path.join(projectRoot, 'scripts', 'score_cordial.py'),
    path.join(process.cwd(), 'scripts', 'score_cordial.py'),
  ];

  for (const scriptPath of candidates) {
    if (fs.existsSync(scriptPath)) {
      return scriptPath;
    }
  }

  return null;
};

export const runCordialScoringJob = async (
  input: { dockDir?: string; pairCsv?: string },
  outputCsv: string,
  batchSize: number,
  options?: {
    env?: NodeJS.ProcessEnv;
    cwd?: string;
    onStdout?: (text: string) => void;
    onStderr?: (text: string) => void;
  },
): Promise<Result<{ scoresFile: string; count: number }, AppError>> => {
  const cordialRoot = getCordialRoot();
  if (!cordialRoot) {
    return Err({
      type: 'CORDIAL_FAILED',
      message: 'CORDIAL not found. Add the macOS-patched fork at ./CORDIAL or set CORDIAL_ROOT.',
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
    '--device', 'cpu',
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
      ...getSpawnEnv(),
      PYTHONPATH: cordialRoot,
      KMP_DUPLICATE_LIB_OK: 'TRUE',
      OMP_NUM_THREADS: process.env.OMP_NUM_THREADS || '1',
      MKL_NUM_THREADS: process.env.MKL_NUM_THREADS || '1',
      OPENBLAS_NUM_THREADS: process.env.OPENBLAS_NUM_THREADS || '1',
      NUMEXPR_NUM_THREADS: process.env.NUMEXPR_NUM_THREADS || '1',
      VECLIB_MAXIMUM_THREADS: process.env.VECLIB_MAXIMUM_THREADS || '1',
      ...options?.env,
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
// SDF property parsing (QED, MW, LogP)
// ---------------------------------------------------------------------------

export interface SdfPropertyResult {
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
  centroid?: { x: number; y: number; z: number } | null;
  rmsd?: number | null;
}

export function parseSdfProperties(sdfPath: string, referenceSdf?: string): Promise<SdfPropertyResult> {
  const failResult: SdfPropertyResult = {
    success: false,
    vinaAffinity: null,
    qed: 0,
    mw: 0,
    logp: 0,
  };

  return new Promise((resolve) => {
    if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
      resolve({ ...failResult, error: 'Python not found' });
      return;
    }

    const scriptPath = path.join(appState.fraggenRoot, 'parse_sdf_properties.py');
    if (!fs.existsSync(scriptPath)) {
      resolve({ ...failResult, error: 'parse_sdf_properties.py not found' });
      return;
    }

    const args = [scriptPath, '--sdf_file', sdfPath];
    if (referenceSdf) {
      args.push('--reference_sdf', referenceSdf);
    }
    const python = spawn(appState.condaPythonPath, args);
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
            centroid: result.centroid ?? null,
            rmsd: result.rmsd ?? null,
          });
        } catch {
          resolve({ ...failResult, error: 'Failed to parse JSON output' });
        }
      } else {
        resolve({ ...failResult, error: stderr || 'Script failed' });
      }
    });

    python.on('error', (err) => {
      resolve({ ...failResult, error: err.message });
    });
  });
}
