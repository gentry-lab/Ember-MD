/**
 * Pocket map handlers (static / solvation / probe dispatch).
 */
import { ipcMain } from 'electron';
import { spawn } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import { Ok, Err, Result } from '../../shared/types/result';
import { AppError } from '../../shared/types/errors';
import { IpcChannels } from '../../shared/types/ipc';
import * as appState from '../app-state';
import { childProcesses } from '../spawn';

type MapJobMetadata = {
  method?: 'static' | 'solvation' | 'probe';
  sourcePdbPath?: string;
  sourceTrajectoryPath?: string;
  ligandResname?: string;
  ligandResnum?: number;
  computedAt?: string;
};

function getBindingSiteResultFile(outputDir: string, projectName?: string): string | null {
  const prefixedPath = projectName ? path.join(outputDir, `${projectName}_binding_site_results.json`) : null;
  if (prefixedPath && fs.existsSync(prefixedPath)) return prefixedPath;
  const legacyPath = path.join(outputDir, 'binding_site_results.json');
  if (fs.existsSync(legacyPath)) return legacyPath;
  return null;
}

function writeMapMetadata(outputDir: string, metadata: MapJobMetadata): void {
  fs.mkdirSync(outputDir, { recursive: true });
  const metadataPath = path.join(outputDir, 'map_metadata.json');
  fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));
}

export function register(): void {
  ipcMain.handle(
    IpcChannels.COMPUTE_POCKET_MAP,
    async (
      event,
      options: {
        method: 'static' | 'solvation' | 'probe';
        pdbPath: string;
        ligandResname: string;
        ligandResnum: number;
        outputDir: string;
        trajectoryPath?: string;
        sourcePdbPath?: string;
        sourceTrajectoryPath?: string;
        boxPadding?: number;
        gridSpacing?: number;
      }
    ): Promise<Result<{
      hydrophobicDx: string;
      hbondDonorDx: string;
      hbondAcceptorDx: string;
      hotspots: Array<{ type: string; position: number[]; direction: number[]; score: number }>;
      gridDimensions: number[];
      ligandCom: number[];
      method?: string;
    }, AppError>> => {
      if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
        return Err({
          type: 'PYTHON_NOT_FOUND',
          message: 'Python not found. Please install miniconda and create the openmm-metal environment.',
        });
      }

      let scriptName: string;
      let scriptArgs: string[];
      const projectName = path.basename(path.resolve(options.outputDir, '../..'));

      switch (options.method) {
        case 'static': {
          scriptName = 'map_binding_site.py';
          scriptArgs = [
            '--pdb_path', options.pdbPath,
            '--ligand_resname', options.ligandResname,
            '--ligand_resnum', String(options.ligandResnum),
            '--output_dir', options.outputDir,
            '--project_name', projectName,
            '--scoring', 'energy',
          ];
          if (options.boxPadding !== undefined) {
            scriptArgs.push('--box_padding', String(options.boxPadding));
          }
          if (options.gridSpacing !== undefined) {
            scriptArgs.push('--grid_spacing', String(options.gridSpacing));
          }
          break;
        }
        case 'solvation': {
          if (!options.trajectoryPath || !fs.existsSync(options.trajectoryPath)) {
            return Err({
              type: 'POCKET_MAP_FAILED',
              message: 'Solvation method requires an MD trajectory. Run a simulation first.',
            });
          }
          scriptName = 'analyze_gist.py';
          scriptArgs = [
            '--pdb_path', options.pdbPath,
            '--trajectory_path', options.trajectoryPath,
            '--ligand_resname', options.ligandResname,
            '--ligand_resnum', String(options.ligandResnum),
            '--output_dir', options.outputDir,
            '--project_name', projectName,
          ];
          if (options.boxPadding !== undefined) {
            scriptArgs.push('--box_padding', String(options.boxPadding));
          }
          if (options.gridSpacing !== undefined) {
            scriptArgs.push('--grid_spacing', String(options.gridSpacing));
          }
          break;
        }
        case 'probe': {
          scriptName = 'run_probe_md.py';
          scriptArgs = [
            '--pdb_path', options.pdbPath,
            '--ligand_resname', options.ligandResname,
            '--ligand_resnum', String(options.ligandResnum),
            '--output_dir', options.outputDir,
            '--project_name', projectName,
          ];
          break;
        }
        default:
          return Err({
            type: 'POCKET_MAP_FAILED',
            message: `Unknown pocket map method: ${options.method}`,
          });
      }

      const scriptPath = path.join(appState.fraggenRoot, scriptName);
      if (!fs.existsSync(scriptPath)) {
        return Err({
          type: 'SCRIPT_NOT_FOUND',
          path: scriptPath,
          message: `Pocket map script not found: ${scriptPath}`,
        });
      }

      if (!fs.existsSync(options.outputDir)) {
        fs.mkdirSync(options.outputDir, { recursive: true });
      }

      return new Promise((resolve) => {
        event.sender.send(IpcChannels.MD_OUTPUT, {
          type: 'stdout',
          data: `=== Pocket Map: ${options.method} ===\n`,
        });

        const proc = spawn(appState.condaPythonPath!, [scriptPath, ...scriptArgs]);
        childProcesses.add(proc);

        proc.stdout?.on('data', (data: Buffer) => {
          const text = data.toString();
          event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stdout', data: text });
        });

        proc.stderr?.on('data', (data: Buffer) => {
          const text = data.toString();
          event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stderr', data: text });
        });

        proc.on('close', (code: number | null) => {
          childProcesses.delete(proc);

          const resultFile = getBindingSiteResultFile(options.outputDir, projectName);

          if (code === 0 && resultFile && fs.existsSync(resultFile)) {
            try {
              const content = fs.readFileSync(resultFile, 'utf-8');
              const result = JSON.parse(content);
              result.method = options.method;
              writeMapMetadata(options.outputDir, {
                method: options.method,
                sourcePdbPath: options.sourcePdbPath || options.pdbPath,
                sourceTrajectoryPath: options.sourceTrajectoryPath || options.trajectoryPath,
                ligandResname: options.ligandResname,
                ligandResnum: options.ligandResnum,
                computedAt: new Date().toISOString(),
              });
              resolve(Ok(result));
            } catch (err) {
              resolve(Err({
                type: 'POCKET_MAP_FAILED',
                message: `Error reading pocket map results: ${err}`,
              }));
            }
          } else {
            resolve(Err({
              type: 'POCKET_MAP_FAILED',
              message: `Pocket map (${options.method}) failed with exit code ${code}`,
            }));
          }
        });

        proc.on('error', (err: Error) => {
          childProcesses.delete(proc);
          resolve(Err({
            type: 'POCKET_MAP_FAILED',
            message: `Failed to start pocket map: ${err.message}`,
          }));
        });
      });
    }
  );
}
