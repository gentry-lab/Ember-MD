// Copyright (c) 2026 Ember Contributors. MIT License.
/**
 * Shared mutable application state.
 * IPC modules import from here instead of main.ts to avoid circular dependencies.
 * main.ts sets these values during createWindow() via initializeState().
 */
import type { BrowserWindow } from 'electron';

export let mainWindow: BrowserWindow | null = null;
export let fraggenRoot: string = '';
export let condaPythonPath: string | null = null;
export let condaEnvBin: string | null = null;
export let surfaceGenPythonPath: string | null = null;

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
