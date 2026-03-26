// Copyright (c) 2026 Ember Contributors. MIT License.
/**
 * Shared mutable application state.
 * IPC modules import from here instead of main.ts to avoid circular dependencies.
 * main.ts sets these values during createWindow() via initializeState().
 */
import type { BrowserWindow } from 'electron';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';

export let mainWindow: BrowserWindow | null = null;
export let fraggenRoot: string = '';
export let condaPythonPath: string | null = null;
export let condaEnvBin: string | null = null;
export let surfaceGenPythonPath: string | null = null;

// ---------------------------------------------------------------------------
// Ember home directory (configurable base for all projects)
// ---------------------------------------------------------------------------

const defaultEmberDir = path.join(os.homedir(), 'Ember');
const settingsPath = path.join(defaultEmberDir, '.ember-settings.json');

let cachedBaseDir: string | null = null;

type StoredSettings = {
  homeDir?: string;
  [key: string]: unknown;
};

const normalizeDir = (dir: string) => path.resolve(dir);

const readSettings = (): StoredSettings => {
  try {
    if (fs.existsSync(settingsPath)) {
      return JSON.parse(fs.readFileSync(settingsPath, 'utf-8')) as StoredSettings;
    }
  } catch { /* fall through */ }
  return {};
};

const writeSettings = (settings: StoredSettings): void => {
  fs.mkdirSync(defaultEmberDir, { recursive: true });
  fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2));
};

const resolveConfiguredHome = (): string => {
  const settings = readSettings();
  if (typeof settings.homeDir === 'string' && settings.homeDir.trim().length > 0) {
    return normalizeDir(settings.homeDir);
  }
  return defaultEmberDir;
};

const persistHomeDir = (newDir: string): void => {
  const normalized = normalizeDir(newDir);
  fs.mkdirSync(normalized, { recursive: true });
  const settings = readSettings();
  settings.homeDir = normalized;
  writeSettings(settings);
  cachedBaseDir = normalized;
};

/** Read the configured home directory, falling back to ~/Ember/. */
export function getEmberBaseDir(): string {
  if (cachedBaseDir) return cachedBaseDir;
  cachedBaseDir = resolveConfiguredHome();
  return cachedBaseDir;
}

/** Update the home directory setting. Returns the new path. */
export function setEmberBaseDir(newDir: string): void {
  persistHomeDir(newDir);
}

export function setMainWindow(w: BrowserWindow | null): void {
  mainWindow = w;
}

// ---------------------------------------------------------------------------
// Last-seen version (for changelog popup)
// ---------------------------------------------------------------------------

export function getLastSeenVersion(): string | null {
  const settings = readSettings();
  return (settings.lastSeenVersion as string) || null;
}

export function setLastSeenVersion(version: string): void {
  const settings = readSettings();
  settings.lastSeenVersion = version;
  writeSettings(settings);
}

export function initializeState(resolved: {
  fraggenRoot: string;
  condaPythonPath: string | null;
  condaEnvBin: string | null;
  surfaceGenPythonPath: string | null;
}): void {
  fraggenRoot = resolved.fraggenRoot;
  condaPythonPath = resolved.condaPythonPath;
  condaEnvBin = resolved.condaEnvBin;
  surfaceGenPythonPath = resolved.surfaceGenPythonPath;
}
