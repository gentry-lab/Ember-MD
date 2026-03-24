// Copyright (c) 2026 Ember Contributors. MIT License.
/**
 * Multi-input ligand source handlers.
 * File selection, SDF scanning, SMILES/CSV import, X-ray extraction,
 * protonation/stereoisomer enumeration, and pre-docking conformer generation.
 */
import { ipcMain, dialog } from 'electron';
import { spawn } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import * as zlib from 'zlib';
import { Ok, Err, Result } from '../../shared/types/result';
import { AppError } from '../../shared/types/errors';
import { IpcChannels } from '../../shared/types/ipc';
import * as appState from '../app-state';
import { childProcesses } from '../spawn';
import { getSpawnEnv as _getSpawnEnv } from '../spawn';
import { getXtbPath } from '../paths';

function getSpawnEnv(): NodeJS.ProcessEnv {
  return _getSpawnEnv(appState.condaEnvBin);
}

function stripStructureExtension(filePath: string): string {
  return path.basename(filePath).replace(/(\.sdf\.gz|\.sdf|\.mol2|\.mol)$/i, '');
}

function convertMolFileToSdf(
  inputPath: string,
  outputDir: string,
  name?: string
): Promise<Result<{
  sdfPath: string;
  smiles: string;
  name: string;
  qed: number;
  mw: number;
  thumbnail: string;
}, AppError>> {
  return new Promise((resolve) => {
    if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
      resolve(Err({ type: 'PYTHON_NOT_FOUND', message: 'Python not found' }));
      return;
    }

    const scriptPath = path.join(appState.fraggenRoot, 'smiles_to_sdf.py');
    if (!fs.existsSync(scriptPath)) {
      resolve(Err({ type: 'SCRIPT_NOT_FOUND', path: scriptPath, message: `Script not found: ${scriptPath}` }));
      return;
    }

    fs.mkdirSync(outputDir, { recursive: true });

    const args = [scriptPath, '--output_dir', outputDir, '--mol_file', inputPath];
    if (name) args.push('--name', name);

    const python = spawn(appState.condaPythonPath, args);
    childProcesses.add(python);

    let stdout = '';
    let stderr = '';

    python.stdout.on('data', (data: Buffer) => { stdout += data.toString(); });
    python.stderr.on('data', (data: Buffer) => { stderr += data.toString(); });

    python.on('close', (code: number | null) => {
      childProcesses.delete(python);
      if (code === 0) {
        try {
          const result = JSON.parse(stdout.trim());
          if (result.error) {
            resolve(Err({ type: 'PARSE_FAILED', message: result.error }));
          } else {
            resolve(Ok(result));
          }
        } catch {
          resolve(Err({ type: 'PARSE_FAILED', message: `Failed to parse output: ${stdout}` }));
        }
      } else {
        resolve(Err({ type: 'PARSE_FAILED', message: stderr || 'Conversion failed' }));
      }
    });

    python.on('error', (error: Error) => {
      childProcesses.delete(python);
      resolve(Err({ type: 'PARSE_FAILED', message: error.message }));
    });
  });
}

export function register(): void {
  ipcMain.handle(IpcChannels.SELECT_MOLECULE_FILES_MULTI, async (): Promise<string[]> => {
    if (!appState.mainWindow) return [];
    const result = await dialog.showOpenDialog(appState.mainWindow, {
      properties: ['openFile', 'multiSelections'],
      title: 'Select Molecule Files',
      filters: [
        { name: 'Molecule Files', extensions: ['sdf', 'mol', 'mol2', 'gz'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    });
    return result.filePaths;
  });

  ipcMain.handle(
    IpcChannels.IMPORT_MOLECULE_FILES,
    async (_event, filePaths: string[], outputDir: string): Promise<Result<Array<{
      filename: string;
      smiles: string;
      qed: number;
      sdfPath: string;
    }>, AppError>> => {
      try {
        if (!filePaths || filePaths.length === 0) {
          return Ok([]);
        }

        fs.mkdirSync(outputDir, { recursive: true });
        const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ember-mol-import-'));
        const molecules: Array<{ filename: string; smiles: string; qed: number; sdfPath: string }> = [];

        try {
          for (const filePath of filePaths) {
            const baseName = stripStructureExtension(filePath) || 'molecule';
            let inputPath = filePath;

            if (/\.sdf\.gz$/i.test(filePath)) {
              const uncompressedPath = path.join(tempDir, `${baseName}.sdf`);
              fs.writeFileSync(uncompressedPath, zlib.gunzipSync(fs.readFileSync(filePath)));
              inputPath = uncompressedPath;
            }

            const converted = await convertMolFileToSdf(inputPath, outputDir, baseName);
            if (!converted.ok) {
              return Err(converted.error);
            }

            molecules.push({
              filename: baseName,
              smiles: converted.value.smiles,
              qed: converted.value.qed,
              sdfPath: converted.value.sdfPath,
            });
          }
        } finally {
          fs.rmSync(tempDir, { recursive: true, force: true });
        }

        return Ok(molecules);
      } catch (err) {
        return Err({
          type: 'PARSE_FAILED',
          message: `Failed to import molecule files: ${(err as Error).message}`,
        });
      }
    }
  );

  // Select folder dialog
  ipcMain.handle(IpcChannels.SELECT_FOLDER, async (): Promise<string | null> => {
    if (!appState.mainWindow) return null;
    const result = await dialog.showOpenDialog(appState.mainWindow, {
      properties: ['openDirectory'],
      title: 'Select Folder',
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
        if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
          resolve(Err({
            type: 'PYTHON_NOT_FOUND',
            message: 'Python not found. Please install miniconda and create fraggen environment.',
          }));
          return;
        }

        const scriptPath = path.join(appState.fraggenRoot, 'scan_sdf_directory.py');
        if (!fs.existsSync(scriptPath)) {
          resolve(Err({
            type: 'SCRIPT_NOT_FOUND',
            path: scriptPath,
            message: `SDF directory scanner script not found: ${scriptPath}`,
          }));
          return;
        }

        const python = spawn(appState.condaPythonPath, [
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

  // Parse structure CSV and generate 3D structures
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
        if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
          resolve(Err({
            type: 'PYTHON_NOT_FOUND',
            message: 'Python not found. Please install miniconda and create fraggen environment.',
          }));
          return;
        }

        const scriptPath = path.join(appState.fraggenRoot, 'smiles_to_3d.py');
        if (!fs.existsSync(scriptPath)) {
          resolve(Err({
            type: 'SCRIPT_NOT_FOUND',
            path: scriptPath,
            message: `CSV structure converter script not found: ${scriptPath}`,
          }));
          return;
        }

        fs.mkdirSync(outputDir, { recursive: true });

        const python = spawn(appState.condaPythonPath, [
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
                  message: 'No molecule data returned from CSV import',
                }));
              }
            } catch (e) {
              resolve(Err({
                type: 'PARSE_FAILED',
                message: `Failed to parse CSV import results: ${(e as Error).message}`,
              }));
            }
          } else {
            resolve(Err({
              type: 'PARSE_FAILED',
              message: `CSV import failed: ${stderr}`,
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

  // Convert a list of SMILES strings to 3D SDF files
  ipcMain.handle(
    IpcChannels.CONVERT_SMILES_LIST,
    async (_event, smilesList: string[], outputDir: string): Promise<Result<Array<{
      filename: string;
      smiles: string;
      qed: number;
      saScore: number;
      sdfPath: string;
    }>, AppError>> => {
      if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
        return Err({ type: 'PYTHON_NOT_FOUND', message: 'Python not found' });
      }

      const scriptPath = path.join(appState.fraggenRoot, 'smiles_to_3d.py');
      if (!fs.existsSync(scriptPath)) {
        return Err({ type: 'SCRIPT_NOT_FOUND', path: scriptPath, message: 'smiles_to_3d.py not found' });
      }

      fs.mkdirSync(outputDir, { recursive: true });

      // Write a temp CSV with smiles and name columns
      const tmpCsv = path.join(outputDir, '_smiles_input.csv');
      const csvLines = ['smiles,name'];
      for (let i = 0; i < smilesList.length; i++) {
        const smi = smilesList[i].trim();
        if (!smi) continue;
        // Use molecule index as name; escape commas in SMILES (rare but possible)
        const name = `mol_${i + 1}`;
        csvLines.push(`${smi},${name}`);
      }
      fs.writeFileSync(tmpCsv, csvLines.join('\n'));

      return new Promise((resolve) => {
        const python = spawn(appState.condaPythonPath!, [
          scriptPath, '--input_csv', tmpCsv, '--output_dir', outputDir,
        ]);
        childProcesses.add(python);

        let stdout = '';
        let stderr = '';

        python.stdout.on('data', (data: Buffer) => { stdout += data.toString(); });
        python.stderr.on('data', (data: Buffer) => { stderr += data.toString(); });

        python.on('close', (code: number | null) => {
          childProcesses.delete(python);
          try { fs.unlinkSync(tmpCsv); } catch { /* */ }

          if (code === 0) {
            try {
              const lines = stdout.trim().split('\n');
              const lastLine = lines[lines.length - 1];
              if (lastLine && (lastLine.startsWith('[') || lastLine.startsWith('{'))) {
                resolve(Ok(JSON.parse(lastLine)));
              } else {
                resolve(Err({ type: 'PARSE_FAILED', message: 'No molecule data returned' }));
              }
            } catch (e) {
              resolve(Err({ type: 'PARSE_FAILED', message: `Failed to parse results: ${(e as Error).message}` }));
            }
          } else {
            resolve(Err({ type: 'PARSE_FAILED', message: `SMILES conversion failed: ${stderr}` }));
          }
        });

        python.on('error', (error: Error) => {
          childProcesses.delete(python);
          resolve(Err({ type: 'PARSE_FAILED', message: error.message }));
        });
      });
    }
  );

  // Convert single SMILES or MOL file to 3D SDF + thumbnail
  ipcMain.handle(
    IpcChannels.CONVERT_SINGLE_MOLECULE,
    async (_event, input: string, outputDir: string, inputType: 'smiles' | 'mol_file'): Promise<Result<any, AppError>> => {
      return new Promise((resolve) => {
        if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
          resolve(Err({ type: 'PYTHON_NOT_FOUND', message: 'Python not found' }));
          return;
        }

        const scriptPath = path.join(appState.fraggenRoot, 'smiles_to_sdf.py');
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

        const python = spawn(appState.condaPythonPath, args);
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
        if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
          resolve(Err({ type: 'PYTHON_NOT_FOUND', message: 'Python not found' }));
          return;
        }

        const scriptPath = path.join(appState.fraggenRoot, 'extract_xray_ligand.py');
        if (!fs.existsSync(scriptPath)) {
          resolve(Err({ type: 'SCRIPT_NOT_FOUND', path: scriptPath, message: `Script not found: ${scriptPath}` }));
          return;
        }

        fs.mkdirSync(outputDir, { recursive: true });

        const args = [scriptPath, '--pdb', pdbPath, '--ligand_id', ligandId, '--output_dir', outputDir];
        if (smiles) {
          args.push('--smiles', smiles);
        }

        const python = spawn(appState.condaPythonPath, args);
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
        if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
          resolve(Err({
            type: 'PYTHON_NOT_FOUND',
            message: 'Python not found. Please install miniconda and create fraggen environment.',
          }));
          return;
        }

        const scriptPath = path.join(appState.fraggenRoot, 'enumerate_protonation.py');
        if (!fs.existsSync(scriptPath)) {
          resolve(Err({
            type: 'SCRIPT_NOT_FOUND',
            path: scriptPath,
            message: `Protonation script not found: ${scriptPath}`,
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
          data: `  Protonation enumeration — pH ${phMin}–${phMax}, ${ligandSdfPaths.length} molecule(s)\n`
        });

        console.log(`[Dock:Protonation] pH ${phMin}-${phMax}, ${ligandSdfPaths.length} molecules`);

        const python = spawn(appState.condaPythonPath, args);
        childProcesses.add(python);

        let stdout = '';
        let stderr = '';

        python.stdout.on('data', (data: Buffer) => {
          const text = data.toString();
          stdout += text;
          for (const line of text.split('\n')) {
            if (line.trim()) console.log(`[Dock:Protonation] ${line}`);
          }
        });

        python.stderr.on('data', (data: Buffer) => {
          const text = data.toString();
          stderr += text;
          for (const line of text.split('\n')) {
            if (line.trim()) console.log(`[Dock:Protonation:err] ${line}`);
          }
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
                const protonatedPaths = result.protonated_paths || [];
                if (protonatedPaths.length === 0) {
                  resolve(Err({
                    type: 'PROTONATION_FAILED',
                    message: 'Protonation completed without producing any protonated ligands.',
                  }));
                  return;
                }
                resolve(Ok({
                  protonatedPaths,
                  parentMapping: result.parent_mapping || {},
                }));
              } else {
                resolve(Err({
                  type: 'PROTONATION_FAILED',
                  message: 'Protonation did not return a result payload.',
                }));
              }
            } catch (e) {
              resolve(Err({
                type: 'PARSE_FAILED',
                message: `Failed to parse protonation results: ${(e as Error).message}`,
              }));
            }
          } else {
            if (stderr.includes('molscrub') || stderr.includes('Molscrub')) {
              resolve(Err({
                type: 'PROTONATION_FAILED',
                message: 'Ligand protonation requires Molscrub in the active Python environment. Install `molscrub` or disable protonation.',
              }));
            } else {
              resolve(Err({
                type: 'PROTONATION_FAILED',
                message: `Protonation failed: ${(stderr || stdout).slice(0, 200)}`,
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
        if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
          resolve(Err({
            type: 'PYTHON_NOT_FOUND',
            message: 'Python not found.',
          }));
          return;
        }

        const scriptPath = path.join(appState.fraggenRoot, 'enumerate_stereoisomers.py');
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
          data: `  Stereoisomer enumeration — ${maxStereoisomers} max per molecule, ${ligandSdfPaths.length} input(s)\n`
        });

        console.log(`[Dock:Stereo] Max ${maxStereoisomers} per molecule, ${ligandSdfPaths.length} molecules`);

        const python = spawn(appState.condaPythonPath, args);
        childProcesses.add(python);

        let stdout = '';
        let stderr = '';

        python.stdout.on('data', (data: Buffer) => {
          const text = data.toString();
          stdout += text;
          for (const line of text.split('\n')) {
            if (line.trim()) console.log(`[Dock:Stereo] ${line}`);
          }
        });

        python.stderr.on('data', (data: Buffer) => {
          const text = data.toString();
          stderr += text;
          for (const line of text.split('\n')) {
            if (line.trim()) console.log(`[Dock:Stereo:err] ${line}`);
          }
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
      mcmmOptions?: { steps: number; temperature: number; sampleAmides: boolean; xtbRerank?: boolean }
    ): Promise<Result<{
      conformerPaths: string[];
      parentMapping: Record<string, string>;
    }, AppError>> => {
      return new Promise((resolve) => {
        if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
          resolve(Err({
            type: 'PYTHON_NOT_FOUND',
            message: 'Python not found. Please install miniconda and create fraggen environment.',
          }));
          return;
        }

        const scriptPath = path.join(appState.fraggenRoot, 'generate_conformers.py');
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

        // xTB reranking support for docking conformer generation
        const xtbPathDock = getXtbPath();
        const shouldRerank = mcmmOptions?.xtbRerank ?? true;
        if (xtbPathDock) {
          args.push('--xtb_binary', xtbPathDock);
          if (shouldRerank && effectiveMethod !== 'crest') {
            args.push('--xtb_rerank');
          }
        }

        const methodLabel = effectiveMethod.toUpperCase();
        const xtbStatus =
          effectiveMethod === 'crest'
            ? 'xTB energy: intrinsic (CREST)'
            : xtbPathDock
              ? (shouldRerank ? 'xTB reranking: enabled' : 'xTB reranking: disabled')
              : 'xTB reranking: unavailable';
        event.sender.send(IpcChannels.DOCK_OUTPUT, {
          type: 'stdout',
          data: `  ${methodLabel} — ${maxConformers} max, ${rmsdCutoff} Å cutoff, ${xtbStatus}\n`
        });

        console.log(`[Dock:Conform] Starting ${methodLabel} conformer generation for ${ligandSdfPaths.length} molecules`);
        console.log(`[Dock:Conform] Args: ${args.slice(1).join(' ')}`);

        const python = spawn(appState.condaPythonPath, args);
        childProcesses.add(python);

        let stdout = '';
        let stderr = '';

        python.stdout.on('data', (data: Buffer) => {
          const text = data.toString();
          stdout += text;
          for (const line of text.split('\n')) {
            if (line.trim()) console.log(`[Dock:Conform] ${line}`);
          }
        });

        python.stderr.on('data', (data: Buffer) => {
          const text = data.toString();
          stderr += text;
          for (const line of text.split('\n')) {
            if (line.trim()) console.log(`[Dock:Conform:err] ${line}`);
          }
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
                  conformerEnergies: result.conformer_energies || {},
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
                  conformerEnergies: {},
                }));
              }
            } catch (e) {
              resolve(Err({
                type: 'PARSE_FAILED',
                message: `Failed to parse conformer results: ${(e as Error).message}`,
              }));
            }
          } else {
            const errMsg = stderr.slice(0, 200).trim() || `exit code ${code}`;
            console.error(`[Dock:Conform] CONFORMER_FAILED: ${errMsg}`);
            event.sender.send(IpcChannels.DOCK_OUTPUT, {
              type: 'stderr',
              data: `  Error [CONFORMER_FAILED]: ${errMsg}\n`
            });
            resolve(Err({
              type: 'CONFORMER_FAILED',
              message: `Conformer generation failed: ${errMsg}`,
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
}
