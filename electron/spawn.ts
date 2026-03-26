// Copyright (c) 2026 Ember Contributors. MIT License.
/**
 * Subprocess management for Ember.
 * Handles Python script spawning, process tracking, stderr filtering, and CORDIAL score merging.
 */
import * as path from 'path';
import * as fs from 'fs';
import { spawn, ChildProcess } from 'child_process';

// Track spawned processes for cleanup.
// Wrapped in a Proxy so caffeinate is updated automatically on any add/delete,
// regardless of which IPC module manages the process.
const _childProcesses = new Set<ChildProcess>();
export const childProcesses: Set<ChildProcess> = new Proxy(_childProcesses, {
  get(target, prop, receiver) {
    const val = Reflect.get(target, prop, receiver);
    if (prop === 'add') return (proc: ChildProcess) => { target.add(proc); updateCaffeinate(); return childProcesses; };
    if (prop === 'delete') return (proc: ChildProcess) => { const r = target.delete(proc); updateCaffeinate(); return r; };
    if (prop === 'clear') return () => { target.clear(); updateCaffeinate(); };
    return typeof val === 'function' ? val.bind(target) : val;
  },
});

// Prevent idle sleep while jobs are running (macOS caffeinate -i)
let caffeinateProc: ChildProcess | null = null;

function updateCaffeinate(): void {
  if (_childProcesses.size > 0 && !caffeinateProc) {
    caffeinateProc = spawn('caffeinate', ['-i'], { stdio: 'ignore', detached: true });
    caffeinateProc.unref();
    caffeinateProc.on('exit', () => { caffeinateProc = null; });
  } else if (_childProcesses.size === 0 && caffeinateProc) {
    caffeinateProc.kill();
    caffeinateProc = null;
  }
}

export function killAllChildProcesses(): void {
  for (const proc of childProcesses) {
    if (!proc.killed) {
      proc.kill('SIGTERM');
    }
  }
  childProcesses.clear();
}

/** Filter noisy stderr from Metal backend and Python deprecation warnings */
export function filterMdStderr(text: string): string {
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

/** Build spawn env with conda bin on PATH so child processes find sqm, obabel, etc. */
export function getSpawnEnv(condaEnvBin: string | null): NodeJS.ProcessEnv {
  if (!condaEnvBin) return { ...process.env };
  const currentPath = process.env.PATH || '';
  return { ...process.env, PATH: `${condaEnvBin}:${currentPath}` };
}

/**
 * Spawn a Python script and collect stdout/stderr. Handles childProcesses tracking.
 * Use for accumulate-then-parse patterns (not real-time streaming).
 */
export function spawnPythonScript(
  condaPythonPath: string | null,
  condaEnvBin: string | null,
  args: string[],
  options?: {
    env?: NodeJS.ProcessEnv;
    cwd?: string;
    onStdout?: (text: string) => void;
    onStderr?: (text: string) => void;
  }
): Promise<{ stdout: string; stderr: string; code: number }> {
  return new Promise((resolve) => {
    if (!condaPythonPath) {
      resolve({ stdout: '', stderr: 'Python not found', code: 1 });
      return;
    }
    const proc = spawn(condaPythonPath, args, {
      env: options?.env || getSpawnEnv(condaEnvBin),
      cwd: options?.cwd,
    });
    childProcesses.add(proc);

    let stdout = '';
    let stderr = '';

    proc.stdout?.on('data', (data: Buffer) => {
      const text = data.toString();
      stdout += text;
      options?.onStdout?.(text);
    });

    proc.stderr?.on('data', (data: Buffer) => {
      const text = data.toString();
      stderr += text;
      options?.onStderr?.(text);
    });

    proc.on('close', (code: number | null) => {
      childProcesses.delete(proc);
      resolve({ stdout, stderr, code: code ?? 1 });
    });

    proc.on('error', (err: Error) => {
      childProcesses.delete(proc);
      resolve({ stdout, stderr: err.message, code: 1 });
    });
  });
}

/**
 * Load and merge CORDIAL scores from a JSON file into an array of result objects.
 * Handles the results/ vs top-level path fallback and best-score deduplication.
 */
export function loadAndMergeCordialScores(
  baseDir: string,
  items: Array<Record<string, any>>,
  nameKey: string = 'ligandName'
): void {
  const newPath = path.join(baseDir, 'results', 'cordial_scores.json');
  const legacyPath = path.join(baseDir, 'cordial_scores.json');
  const cordialJsonPath = fs.existsSync(newPath) ? newPath : legacyPath;
  if (!fs.existsSync(cordialJsonPath)) return;

  try {
    const cordialData = JSON.parse(fs.readFileSync(cordialJsonPath, 'utf-8'));
    const cordialByName = new Map<string, {
      expectedPkd: number;
      pHighAffinity: number;
      pVeryHighAffinity: number;
    }>();

    for (const entry of cordialData) {
      const entryName = entry.source_name;
      const pHighAffinity = entry.cordial_p_high_affinity;
      const existing = cordialByName.get(entryName);
      if (!existing || pHighAffinity > existing.pHighAffinity) {
        cordialByName.set(entryName, {
          expectedPkd: entry.cordial_expected_pkd,
          pHighAffinity,
          pVeryHighAffinity: entry.cordial_p_very_high ?? entry.cordial_p_very_high_affinity ?? 0,
        });
      }
    }

    for (const item of items) {
      const scores = cordialByName.get(item[nameKey]);
      if (scores) {
        item.cordialExpectedPkd = scores.expectedPkd;
        item.cordialPHighAffinity = scores.pHighAffinity;
        item.cordialPVeryHighAffinity = scores.pVeryHighAffinity;
      }
    }
  } catch (e) {
    console.error('Failed to load CORDIAL scores:', e);
  }
}
