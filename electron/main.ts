// Copyright (c) 2026 Ember Contributors. MIT License.
import { app, BrowserWindow, ipcMain } from 'electron';
import * as path from 'path';
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

// --- Extracted modules ---
import * as paths from './paths';
import * as appState from './app-state';
import { childProcesses, killAllChildProcesses } from './spawn';

// --- IPC handler modules ---
import * as ipcDialogs from './ipc/dialogs';
import * as ipcStats from './ipc/stats';
import * as ipcGeneration from './ipc/generation';
import * as ipcDocking from './ipc/docking';
import * as ipcLigandSources from './ipc/ligand-sources';
import * as ipcConformers from './ipc/conformers';
import * as ipcSimulation from './ipc/simulation';
import * as ipcProjects from './ipc/projects';
import * as ipcViewer from './ipc/viewer';
import * as ipcMaps from './ipc/maps';
import * as ipcFep from './ipc/fep';
import * as ipcXray from './ipc/xray';
import * as ipcScoring from './ipc/scoring';

let mainWindow: BrowserWindow | null = null;

function createWindow(): void {
  // Initialize paths on startup
  const resolved = paths.initializePaths();
  appState.initializeState(resolved);

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
    ...(process.platform === 'darwin' ? { titleBarStyle: 'hiddenInset' as const } : {}),
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

  appState.setMainWindow(mainWindow);
  mainWindow.on('closed', () => {
    mainWindow = null;
    appState.setMainWindow(null);
  });
}

// --- App lifecycle ---

app.on('ready', () => {
  createWindow();

  // App version (read from package.json by Electron)
  ipcMain.handle('get-app-version', () => app.getVersion());

  // Register all IPC handler modules
  ipcDialogs.register();
  ipcStats.register();
  ipcGeneration.register();
  ipcDocking.register();
  ipcLigandSources.register();
  ipcConformers.register();
  ipcSimulation.register();
  ipcProjects.register();
  ipcViewer.register();
  ipcMaps.register();
  ipcFep.register();
  ipcXray.register();
  ipcScoring.register();
});

app.on('window-all-closed', () => {
  killAllChildProcesses();
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', async (event) => {
  if (childProcesses.size > 0) {
    event.preventDefault();
    const { dialog } = require('electron');
    const { response } = await dialog.showMessageBox({
      type: 'warning',
      buttons: ['Cancel', 'Quit Anyway'],
      defaultId: 0,
      cancelId: 0,
      title: 'Job Running',
      message: 'A simulation or docking job is still running.',
      detail: 'Quitting now will terminate the running job. Any unsaved progress will be lost.',
    });
    if (response === 1) {
      killAllChildProcesses();
      app.quit();
    }
  }
});

app.on('activate', () => {
  if (mainWindow === null) createWindow();
});
