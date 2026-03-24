// Copyright (c) 2026 Ember Contributors. MIT License.
/**
 * File/folder dialog handlers and simple file operations.
 */
import { ipcMain, dialog } from 'electron';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { Ok, Err, Result } from '../../shared/types/result';
import { AppError } from '../../shared/types/errors';
import { IpcChannels } from '../../shared/types/ipc';
import * as appState from '../app-state';

export function register(): void {
  // Select multiple protein structure files
  ipcMain.handle(IpcChannels.SELECT_PDB_FILES_MULTI, async (): Promise<string[]> => {
    if (!appState.mainWindow) return [];
    const result = await dialog.showOpenDialog(appState.mainWindow, {
      properties: ['openFile', 'multiSelections'],
      filters: [
        { name: 'Structure Files', extensions: ['pdb', 'cif'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    });
    return result.filePaths;
  });

  ipcMain.handle(IpcChannels.SELECT_STRUCTURE_FILES_MULTI, async (): Promise<string[]> => {
    if (!appState.mainWindow) return [];
    const result = await dialog.showOpenDialog(appState.mainWindow, {
      properties: ['openFile', 'multiSelections'],
      filters: [
        { name: 'View Files', extensions: ['pdb', 'cif', 'sdf', 'mol', 'mol2', 'gz', 'dcd'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    });
    return result.filePaths;
  });

  ipcMain.handle(IpcChannels.SELECT_PDB_FILE, async (_event, defaultPath?: string): Promise<string | null> => {
    if (!appState.mainWindow) return null;
    const result = await dialog.showOpenDialog(appState.mainWindow, {
      properties: ['openFile'],
      defaultPath: defaultPath || undefined,
      filters: [
        { name: 'Structure Files', extensions: ['pdb', 'cif'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    });
    return result.filePaths[0] || null;
  });

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
        devices.push('mps');
        devices.push('metal');
      } else if (process.platform === 'linux') {
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

  ipcMain.handle(IpcChannels.SELECT_OUTPUT_FOLDER, async (): Promise<string | null> => {
    if (!appState.mainWindow) return null;
    const result = await dialog.showOpenDialog(appState.mainWindow, {
      properties: ['openDirectory', 'createDirectory'],
      title: 'Select Output Folder',
    });
    return result.filePaths[0] || null;
  });

  // Select CSV file
  ipcMain.handle(IpcChannels.SELECT_CSV_FILE, async (): Promise<string | null> => {
    if (!appState.mainWindow) return null;
    const result = await dialog.showOpenDialog(appState.mainWindow, {
      properties: ['openFile'],
      filters: [
        { name: 'CSV Files', extensions: ['csv'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    });
    return result.filePaths[0] || null;
  });

  // Select a ligand structure file
  ipcMain.handle(IpcChannels.SELECT_SDF_FILE, async (): Promise<string | null> => {
    if (!appState.mainWindow) return null;
    const result = await dialog.showOpenDialog(appState.mainWindow, {
      properties: ['openFile'],
      title: 'Select Molecule Structure File',
      filters: [
        { name: 'Molecule Files', extensions: ['sdf', 'mol', 'mol2', 'gz'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    });
    return result.filePaths[0] || null;
  });

  // Save PDB file (for Viewer mode export)
  ipcMain.handle(
    IpcChannels.SAVE_PDB_FILE,
    async (_event, content: string, defaultName?: string): Promise<string | null> => {
      if (!appState.mainWindow) return null;
      const result = await dialog.showSaveDialog(appState.mainWindow, {
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

  // Get CPU count for parallel docking
  ipcMain.handle(IpcChannels.GET_CPU_COUNT, async (): Promise<number> => {
    return os.cpus().length;
  });

  // JSON file reading (for CORDIAL scores etc.)
  ipcMain.handle('read-json-file', async (_event, jsonPath: string): Promise<unknown | null> => {
    try {
      if (!fs.existsSync(jsonPath)) return null;
      const content = fs.readFileSync(jsonPath, 'utf-8');
      return JSON.parse(content);
    } catch {
      return null;
    }
  });

  // Text file writing (for logs)
  ipcMain.handle(IpcChannels.WRITE_TEXT_FILE, async (_event, filePath: string, content: string): Promise<Result<string, AppError>> => {
    try {
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

  // Parse FragGen results CSV (legacy)
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
        let sdfDirectory = '';
        const possibleSdfDirs = [
          path.join(csvDir, 'SDF'),
          path.join(csvDir, '..', 'SDF'),
          path.join(csvDir, 'ligand', 'SDF'),
        ];

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
}
