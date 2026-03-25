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

/** Read the configured home directory, falling back to ~/Ember/. */
export function getEmberBaseDir(): string {
  if (cachedBaseDir) return cachedBaseDir;
  try {
    if (fs.existsSync(settingsPath)) {
      const settings = JSON.parse(fs.readFileSync(settingsPath, 'utf-8'));
      if (settings.homeDir && fs.existsSync(settings.homeDir)) {
        cachedBaseDir = settings.homeDir as string;
        return settings.homeDir as string;
      }
    }
  } catch { /* fall through */ }
  cachedBaseDir = defaultEmberDir;
  return cachedBaseDir;
}

/** Update the home directory setting. Returns the new path. */
export function setEmberBaseDir(newDir: string): void {
  fs.mkdirSync(defaultEmberDir, { recursive: true });
  fs.mkdirSync(newDir, { recursive: true });

  // Read existing settings to preserve other fields
  let settings: Record<string, any> = {};
  try {
    if (fs.existsSync(settingsPath)) {
      settings = JSON.parse(fs.readFileSync(settingsPath, 'utf-8'));
    }
  } catch { /* start fresh */ }

  settings.homeDir = newDir;
  fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2));
  cachedBaseDir = newDir;

  // Add old ~/Ember/ projects to external manifest so they remain visible
  if (newDir !== defaultEmberDir) {
    const manifestPath = path.join(newDir, '.external-projects.json');
    let external: string[] = [];
    try {
      if (fs.existsSync(manifestPath)) {
        external = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
      }
    } catch { /* start fresh */ }

    // Scan old ~/Ember/ for projects and add them
    try {
      const oldEntries = fs.readdirSync(defaultEmberDir, { withFileTypes: true });
      for (const entry of oldEntries) {
        if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
        const oldProjectPath = path.join(defaultEmberDir, entry.name);
        if (fs.existsSync(path.join(oldProjectPath, '.ember-project')) && !external.includes(oldProjectPath)) {
          external.push(oldProjectPath);
        }
      }
    } catch { /* old dir may not exist */ }

    if (external.length > 0) {
      fs.writeFileSync(manifestPath, JSON.stringify(external, null, 2));
    }
  }
}

export function setMainWindow(w: BrowserWindow | null): void {
  mainWindow = w;
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
