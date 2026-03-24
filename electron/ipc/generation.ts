// Copyright (c) 2026 Ember Contributors. MIT License.
/**
 * Legacy FragGen generation handlers.
 * PREP_PDB, GENERATE_SURFACE, RUN_GENERATION, GENERATE_THUMBNAIL,
 * GENERATE_RESULTS_CSV, VALIDATE_ANCHOR_SDF.
 */
import { ipcMain } from 'electron';
import { spawn } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import { Ok, Err, Result } from '../../shared/types/result';
import { AppError } from '../../shared/types/errors';
import {
  IpcChannels,
  GenerationOptions,
  GenerationResult,
  RunParameters,
  PrepPdbOptions,
  PrepPdbResult,
  SurfaceResult,
} from '../../shared/types/ipc';
import * as appState from '../app-state';
import { childProcesses } from '../spawn';
import {
  getFragGenScript,
  getFragBase,
  getBaseConfigs,
  generateRuntimeConfig,
  saveRunParameters,
} from '../paths';

// Local wrappers that pass appState.fraggenRoot
function fragGenScript(): string {
  return getFragGenScript(appState.fraggenRoot);
}

function fragBase(): string {
  return getFragBase(appState.fraggenRoot);
}

function baseConfigs(): Record<string, string> {
  return getBaseConfigs(appState.fraggenRoot);
}

export function register(): void {
  // Prep PDB (extract pocket + ligand from complex)
  ipcMain.handle(
    IpcChannels.PREP_PDB,
    async (
      event,
      pdbPath: string,
      outputDir: string,
      options?: PrepPdbOptions
    ): Promise<Result<PrepPdbResult, AppError>> => {
      return new Promise((resolve) => {
        if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
          resolve(Err({
            type: 'PYTHON_NOT_FOUND',
            message: 'Python not found. Please install miniconda and create fraggen environment.',
          }));
          return;
        }

        const scriptPath = path.join(appState.fraggenRoot, 'prep_pdb_gui.py');
        if (!fs.existsSync(scriptPath)) {
          resolve(Err({
            type: 'SCRIPT_NOT_FOUND',
            path: scriptPath,
            message: `Prep script not found: ${scriptPath}`,
          }));
          return;
        }

        const args = [scriptPath, '--input_pdb', pdbPath, '--output_dir', outputDir];
        if (options?.ligandName) args.push('--ligand_name', options.ligandName);
        if (options?.pocketRadius) args.push('--pocket_radius', String(options.pocketRadius));

        const python = spawn(appState.condaPythonPath, args);
        childProcesses.add(python);

        python.stdout.on('data', (data: Buffer) => {
          event.sender.send(IpcChannels.PREP_OUTPUT, { type: 'stdout', data: data.toString() });
        });

        python.stderr.on('data', (data: Buffer) => {
          event.sender.send(IpcChannels.PREP_OUTPUT, { type: 'stderr', data: data.toString() });
        });

        python.on('close', (code: number | null) => {
          childProcesses.delete(python);
          if (code === 0) {
            resolve(Ok({
              pocketPdb: path.join(outputDir, 'pocket.pdb'),
              ligandPdb: path.join(outputDir, 'ligand.pdb'),
              outputDir,
            }));
          } else {
            resolve(Err({
              type: 'PREP_FAILED',
              message: `Process exited with code ${code}`,
            }));
          }
        });

        python.on('error', (error: Error) => {
          childProcesses.delete(python);
          resolve(Err({
            type: 'PREP_FAILED',
            message: error.message,
          }));
        });
      });
    }
  );

  // Generate surface PLY file
  ipcMain.handle(
    IpcChannels.GENERATE_SURFACE,
    async (
      event,
      pocketPdb: string,
      ligandPdb: string,
      outputPly: string
    ): Promise<Result<SurfaceResult, AppError>> => {
      return new Promise((resolve) => {
        // Surface generation requires Python 3.6 with pymesh2 (surface_gen env)
        if (!appState.surfaceGenPythonPath || !fs.existsSync(appState.surfaceGenPythonPath)) {
          resolve(Err({
            type: 'PYTHON_NOT_FOUND',
            message: 'Surface generation requires surface_gen conda environment (Python 3.6 with pymesh2). Run: conda create -n surface_gen python=3.6 && conda activate surface_gen && conda install -c conda-forge pymesh2',
          }));
          return;
        }

        const scriptPath = path.join(appState.fraggenRoot, 'generate_pocket_surface.py');
        if (!fs.existsSync(scriptPath)) {
          resolve(Err({
            type: 'SCRIPT_NOT_FOUND',
            path: scriptPath,
            message: `Surface script not found: ${scriptPath}`,
          }));
          return;
        }

        if (!ligandPdb || !fs.existsSync(ligandPdb)) {
          resolve(Err({
            type: 'FILE_NOT_FOUND',
            path: ligandPdb || 'undefined',
            message: `Ligand file required for surface generation but not found: ${ligandPdb}`,
          }));
          return;
        }

        const args = [
          scriptPath,
          '--pdb_file', pocketPdb,
          '--ligand_file', ligandPdb,
          '--output', outputPly,
        ];

        console.log('Surface generation args:', args);
        console.log('Using Python:', appState.surfaceGenPythonPath);

        const python = spawn(appState.surfaceGenPythonPath, args);
        childProcesses.add(python);

        python.stdout.on('data', (data: Buffer) => {
          event.sender.send(IpcChannels.SURFACE_OUTPUT, { type: 'stdout', data: data.toString() });
        });

        python.stderr.on('data', (data: Buffer) => {
          event.sender.send(IpcChannels.SURFACE_OUTPUT, { type: 'stderr', data: data.toString() });
        });

        python.on('close', (code: number | null) => {
          childProcesses.delete(python);
          if (code === 0) {
            resolve(Ok({ surfaceFile: outputPly }));
          } else {
            resolve(Err({
              type: 'SURFACE_FAILED',
              message: `Process exited with code ${code}`,
            }));
          }
        });

        python.on('error', (error: Error) => {
          childProcesses.delete(python);
          resolve(Err({
            type: 'SURFACE_FAILED',
            message: error.message,
          }));
        });
      });
    }
  );

  // Run FragGen generation
  ipcMain.handle(
    IpcChannels.RUN_GENERATION,
    async (event, options: GenerationOptions): Promise<Result<GenerationResult, AppError>> => {
      const {
        surfacePly, pocketPdb, ligandPdb, outputDir, modelVariant, device, sampling, pocketRadius,
        generationMode = 'denovo', anchorSdfPath = null
      } = options;

      return new Promise((resolve) => {
        console.log('=== RUN FRAGGEN GENERATION ===');
        console.log('Surface PLY:', surfacePly);
        console.log('Pocket PDB:', pocketPdb);
        console.log('Ligand PDB:', ligandPdb);
        console.log('Output dir:', outputDir);
        console.log('Model variant:', modelVariant);
        console.log('Device:', device);
        console.log('Generation mode:', generationMode);
        console.log('Anchor SDF path:', anchorSdfPath);
        console.log('Sampling config:', JSON.stringify(sampling, null, 2));

        if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
          resolve(Err({
            type: 'PYTHON_NOT_FOUND',
            message: 'Python not found. Please install miniconda and create fraggen environment.',
          }));
          return;
        }

        const script = fragGenScript();
        if (!fs.existsSync(script)) {
          resolve(Err({
            type: 'SCRIPT_NOT_FOUND',
            path: script,
            message: `FragGen script not found: ${script}`,
          }));
          return;
        }

        const configs = baseConfigs();
        const baseConfigPath = configs[modelVariant] || configs.dihedral;
        if (!fs.existsSync(baseConfigPath)) {
          resolve(Err({
            type: 'FILE_NOT_FOUND',
            path: baseConfigPath,
            message: `Base config not found: ${baseConfigPath}`,
          }));
          return;
        }

        const fragBaseDir = fragBase();
        if (!fs.existsSync(fragBaseDir)) {
          resolve(Err({
            type: 'FILE_NOT_FOUND',
            path: fragBaseDir,
            message: `Fragment database not found: ${fragBaseDir}`,
          }));
          return;
        }

        // Ensure output directory exists
        fs.mkdirSync(outputDir, { recursive: true });

        // Generate runtime config with custom sampling parameters
        const runtimeConfigPath = generateRuntimeConfig(baseConfigPath, sampling, outputDir);
        console.log('Generated runtime config:', runtimeConfigPath);

        // Save run parameters log
        const runParams: RunParameters = {
          timestamp: new Date().toISOString(),
          inputPdb: pocketPdb,
          modelVariant,
          device,
          pocketRadius,
          sampling,
          outputDir,
          pocketPdb,
          ligandPdb,
          surfacePly,
        };
        const paramsFile = saveRunParameters(runParams, outputDir);
        console.log('Saved run parameters:', paramsFile);

        const args = [
          script,
          '--config', runtimeConfigPath,
          '--device', device || 'cpu',
          '--surf_file', surfacePly,
          '--pdb_file', pocketPdb,
          '--frag_base', fragBaseDir,
          '--save_dir', outputDir,
        ];

        // Add anchor mode parameter
        if (generationMode === 'grow') {
          args.push('--anchor_mode', 'grow');

          // Use custom anchor SDF if provided, otherwise use extracted ligand
          if (anchorSdfPath && fs.existsSync(anchorSdfPath)) {
            args.push('--sdf_file', anchorSdfPath);
          } else if (ligandPdb && fs.existsSync(ligandPdb)) {
            // For extracted ligand, the ligandPdb is the anchor
            args.push('--sdf_file', ligandPdb);
          }
        } else {
          args.push('--anchor_mode', 'denovo');

          // Add ligand PDB if provided (for pocket center reference)
          if (ligandPdb && fs.existsSync(ligandPdb)) {
            args.push('--sdf_file', ligandPdb);
          }
        }

        console.log('Running command:', appState.condaPythonPath, args.join(' '));

        const python = spawn(appState.condaPythonPath, args, { cwd: appState.fraggenRoot });
        childProcesses.add(python);
        let stderrOutput = '';

        python.stdout.on('data', (data: Buffer) => {
          const text = data.toString();
          console.log('stdout:', text);
          event.sender.send(IpcChannels.GENERATION_OUTPUT, { type: 'stdout', data: text });
        });

        python.stderr.on('data', (data: Buffer) => {
          const text = data.toString();
          console.error('stderr:', text);
          stderrOutput += text;
          event.sender.send(IpcChannels.GENERATION_OUTPUT, { type: 'stderr', data: text });
        });

        python.on('close', (code: number | null) => {
          childProcesses.delete(python);
          console.log('Generation finished with code:', code);
          if (code === 0) {
            // Determine SDF directory path
            const ligandBaseName = ligandPdb
              ? path.basename(ligandPdb, path.extname(ligandPdb))
              : path.basename(pocketPdb, '.pdb');
            const sdfDir = path.join(outputDir, ligandBaseName, 'SDF');

            resolve(Ok({ outputDir, sdfDir, paramsFile }));
          } else {
            // Include relevant part of stderr in error message
            const lastLines = stderrOutput.split('\n').slice(-5).join('\n');
            resolve(Err({
              type: 'GENERATION_FAILED',
              message: `Process exited with code ${code}: ${lastLines}`,
            }));
          }
        });

        python.on('error', (error: Error) => {
          childProcesses.delete(python);
          console.error('Python process error:', error);
          resolve(Err({
            type: 'GENERATION_FAILED',
            message: error.message,
          }));
        });
      });
    }
  );

  // Generate 2D thumbnail from SDF
  ipcMain.handle(
    IpcChannels.GENERATE_THUMBNAIL,
    async (_event, sdfPath: string): Promise<string | null> => {
      return new Promise((resolve) => {
        console.log('[Thumbnail] Generating for:', sdfPath);

        if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
          console.log('[Thumbnail] Python not found');
          resolve(null);
          return;
        }

        const scriptPath = path.join(appState.fraggenRoot, 'generate_2d_thumbnail.py');
        if (!fs.existsSync(scriptPath)) {
          console.log('[Thumbnail] Script not found:', scriptPath);
          resolve(null);
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

        python.on('close', (code: number | null) => {
          if (code === 0 && stdout.includes('data:image')) {
            console.log('[Thumbnail] Success');
            resolve(stdout.trim());
          } else {
            console.log('[Thumbnail] Failed - code:', code, 'stderr:', stderr);
            resolve(null);
          }
        });

        python.on('error', (err) => {
          console.log('[Thumbnail] Spawn error:', err);
          resolve(null);
        });
      });
    }
  );

  // Generate results CSV with SMILES and properties
  ipcMain.handle(
    IpcChannels.GENERATE_RESULTS_CSV,
    async (event, sdfDir: string, outputCsv: string): Promise<Result<string, AppError>> => {
      return new Promise((resolve) => {
        if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
          resolve(Err({
            type: 'PYTHON_NOT_FOUND',
            message: 'Python not found',
          }));
          return;
        }

        const scriptPath = path.join(appState.fraggenRoot, 'generate_results_csv.py');
        if (!fs.existsSync(scriptPath)) {
          resolve(Err({
            type: 'SCRIPT_NOT_FOUND',
            path: scriptPath,
            message: `CSV generator script not found: ${scriptPath}`,
          }));
          return;
        }

        const args = [scriptPath, '--sdf_dir', sdfDir, '--output', outputCsv];
        const python = spawn(appState.condaPythonPath, args);
        childProcesses.add(python);

        python.stdout.on('data', (data: Buffer) => {
          event.sender.send(IpcChannels.GENERATION_OUTPUT, { type: 'stdout', data: data.toString() });
        });

        python.stderr.on('data', (data: Buffer) => {
          event.sender.send(IpcChannels.GENERATION_OUTPUT, { type: 'stderr', data: data.toString() });
        });

        python.on('close', (code: number | null) => {
          childProcesses.delete(python);
          if (code === 0) {
            resolve(Ok(outputCsv));
          } else {
            resolve(Err({
              type: 'GENERATION_FAILED',
              message: `CSV generation failed with code ${code}`,
            }));
          }
        });

        python.on('error', (error: Error) => {
          childProcesses.delete(python);
          resolve(Err({
            type: 'GENERATION_FAILED',
            message: error.message,
          }));
        });
      });
    }
  );

  // Validate anchor SDF for fragment growing mode
  ipcMain.handle(
    IpcChannels.VALIDATE_ANCHOR_SDF,
    async (_event, sdfPath: string): Promise<Result<{
      valid: boolean;
      atomCount: number;
      has3DCoords: boolean;
    }, AppError>> => {
      return new Promise((resolve) => {
        if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
          resolve(Err({
            type: 'PYTHON_NOT_FOUND',
            message: 'Python not found. Please install miniconda and create fraggen environment.',
          }));
          return;
        }

        if (!fs.existsSync(sdfPath)) {
          resolve(Err({
            type: 'FILE_NOT_FOUND',
            path: sdfPath,
            message: `Anchor file not found: ${sdfPath}`,
          }));
          return;
        }

        // Use inline Python to validate the SDF or PDB
        const pythonCode = `
import sys
import json

try:
    from rdkit import Chem
    import numpy as np

    file_path = '${sdfPath.replace(/'/g, "\\'")}'
    ext = file_path.lower().split('.')[-1]
    mol = None

    # Load based on file extension
    if ext == 'pdb':
        mol = Chem.MolFromPDBFile(file_path, removeHs=False)
    elif ext in ('sdf', 'mol'):
        mol = Chem.SDMolSupplier(file_path, removeHs=False)[0]
    else:
        # Try SDF first, then PDB
        mol = Chem.SDMolSupplier(file_path, removeHs=False)[0]
        if mol is None:
            mol = Chem.MolFromPDBFile(file_path, removeHs=False)

    if mol is None:
        print(json.dumps({"valid": False, "atomCount": 0, "has3DCoords": False, "error": "Could not parse file"}))
        sys.exit(0)

    atom_count = mol.GetNumAtoms()

    # Check for 3D coordinates
    has_3d = False
    try:
        conf = mol.GetConformer()
        positions = conf.GetPositions()
        # Check if Z coordinates are not all zero (2D structure check)
        z_coords = positions[:, 2]
        has_3d = not np.allclose(z_coords, 0, atol=0.01)
    except:
        has_3d = False

    print(json.dumps({"valid": True, "atomCount": atom_count, "has3DCoords": has_3d}))
except Exception as e:
    print(json.dumps({"valid": False, "atomCount": 0, "has3DCoords": False, "error": str(e)}))
`;

        const python = spawn(appState.condaPythonPath, ['-c', pythonCode]);
        childProcesses.add(python);

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
          if (code === 0 && stdout.trim()) {
            try {
              const result = JSON.parse(stdout.trim());
              if (result.error) {
                resolve(Err({
                  type: 'VALIDATION_FAILED',
                  message: result.error,
                }));
              } else {
                resolve(Ok({
                  valid: result.valid,
                  atomCount: result.atomCount,
                  has3DCoords: result.has3DCoords,
                }));
              }
            } catch (e) {
              resolve(Err({
                type: 'PARSE_FAILED',
                message: `Failed to parse validation result: ${stdout}`,
              }));
            }
          } else {
            resolve(Err({
              type: 'VALIDATION_FAILED',
              message: stderr || 'Validation failed',
            }));
          }
        });

        python.on('error', (error: Error) => {
          childProcesses.delete(python);
          resolve(Err({
            type: 'VALIDATION_FAILED',
            message: error.message,
          }));
        });
      });
    }
  );
}
