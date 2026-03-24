// Copyright (c) 2026 Ember Contributors. MIT License.
/**
 * Shared dependency interface for IPC handler modules.
 * Each module receives this context via its register() function.
 */
import type { BrowserWindow } from 'electron';
import type { ChildProcess } from 'child_process';
import type { ResolvedPaths } from './paths';

export interface IpcContext {
  /** Returns the main BrowserWindow (may be null if closed) */
  mainWindow: () => BrowserWindow | null;

  /** Resolved path configuration */
  paths: ResolvedPaths;

  /** Global child process tracker — all spawned processes are added here */
  childProcesses: Set<ChildProcess>;

  /** Spawn a Python script, track it, and collect output */
  spawnPythonScript: (
    args: string[],
    options?: {
      env?: NodeJS.ProcessEnv;
      cwd?: string;
      onStdout?: (text: string) => void;
      onStderr?: (text: string) => void;
    }
  ) => Promise<{ stdout: string; stderr: string; code: number }>;

  /** Build spawn env with conda bin on PATH */
  getSpawnEnv: () => NodeJS.ProcessEnv;

  /** Load and merge CORDIAL scores into a result array */
  loadAndMergeCordialScores: (
    baseDir: string,
    items: Array<Record<string, any>>,
    nameKey?: string
  ) => void;

  /** Filter noisy Metal/Python stderr */
  filterMdStderr: (text: string) => string;
}
