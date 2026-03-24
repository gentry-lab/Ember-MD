// Copyright (c) 2026 Ember Contributors. MIT License.
/**
 * FEP (ABFE free energy) scoring handlers.
 * Removed from UI but handlers preserved for future use.
 */
import { ipcMain } from 'electron';
import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import { Ok, Err, Result } from '../../shared/types/result';
import { AppError } from '../../shared/types/errors';
import { IpcChannels } from '../../shared/types/ipc';
import * as appState from '../app-state';
import { childProcesses, filterMdStderr } from '../spawn';

let currentFepProcess: ChildProcess | null = null;

export function register(): void {
  ipcMain.handle(
    IpcChannels.RUN_FEP_SCORING,
    async (
      event,
      options: {
        topologyPath: string;
        trajectoryPath: string;
        startNs: number;
        endNs: number;
        numSnapshots: number;
        speedPreset: 'fast' | 'accurate';
        outputDir: string;
        forceFieldPreset: string;
        ligandSdf?: string;
      }
    ): Promise<Result<{
      snapshots: Array<{
        snapshotIndex: number;
        frameIndex: number;
        timeNs: number;
        deltaG_complex: number;
        deltaG_solvent: number;
        deltaG_bind: number;
        uncertainty: number;
      }>;
      meanDeltaG: number;
      sem: number;
      outputDir: string;
    }, AppError>> => {
      if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
        return Err({
          type: 'PYTHON_NOT_FOUND',
          message: 'Python not found. Please install miniconda and create the openmm-metal environment.',
        });
      }

      const scriptPath = path.join(appState.fraggenRoot, 'run_abfe.py');
      if (!fs.existsSync(scriptPath)) {
        return Err({
          type: 'SCRIPT_NOT_FOUND',
          path: scriptPath,
          message: `ABFE FEP script not found: ${scriptPath}`,
        });
      }

      if (!fs.existsSync(options.outputDir)) {
        fs.mkdirSync(options.outputDir, { recursive: true });
      }

      return new Promise((resolve) => {
        event.sender.send(IpcChannels.MD_OUTPUT, {
          type: 'stdout',
          data: `=== ABFE Free Energy Perturbation ===\nPreset: ${options.speedPreset}\nSnapshots: ${options.numSnapshots}\nRange: ${options.startNs.toFixed(1)} - ${options.endNs.toFixed(1)} ns\n\n`,
        });

        const args = [
          scriptPath,
          '--topology', options.topologyPath,
          '--trajectory', options.trajectoryPath,
          '--start_ns', String(options.startNs),
          '--end_ns', String(options.endNs),
          '--num_snapshots', String(options.numSnapshots),
          '--speed_preset', options.speedPreset,
          '--output_dir', options.outputDir,
          '--force_field_preset', options.forceFieldPreset,
        ];

        if (options.ligandSdf) {
          args.push('--ligand_sdf', options.ligandSdf);
        }

        const proc = spawn(appState.condaPythonPath!, args);
        currentFepProcess = proc;
        childProcesses.add(proc);

        let stderrOutput = '';

        proc.stdout?.on('data', (data: Buffer) => {
          const text = data.toString();
          try { event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stdout', data: text }); } catch {}
        });

        proc.stderr?.on('data', (data: Buffer) => {
          const text = data.toString();
          stderrOutput = (stderrOutput + text).slice(-2048);
          const filtered = filterMdStderr(text);
          if (filtered.trim()) {
            try { event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stderr', data: filtered }); } catch {}
          }
        });

        proc.on('close', (code: number | null) => {
          childProcesses.delete(proc);
          if (currentFepProcess === proc) currentFepProcess = null;

          const resultFile = path.join(options.outputDir, 'fep_results.json');
          if (code === 0 && fs.existsSync(resultFile)) {
            try {
              const content = fs.readFileSync(resultFile, 'utf-8');
              const result = JSON.parse(content);
              resolve(Ok(result));
            } catch (err) {
              resolve(Err({
                type: 'FEP_SCORING_FAILED',
                message: `Error reading FEP results: ${err}`,
              }));
            }
          } else if (code === null || code === 137 || code === 143) {
            resolve(Err({
              type: 'FEP_SCORING_CANCELLED',
              message: 'FEP scoring was cancelled.',
            }));
          } else {
            resolve(Err({
              type: 'FEP_SCORING_FAILED',
              message: stderrOutput.slice(-500) || `FEP scoring failed with exit code ${code}`,
            }));
          }
        });

        proc.on('error', (err: Error) => {
          childProcesses.delete(proc);
          if (currentFepProcess === proc) currentFepProcess = null;
          resolve(Err({
            type: 'FEP_SCORING_FAILED',
            message: err.message,
          }));
        });
      });
    }
  );

  ipcMain.handle(IpcChannels.CANCEL_FEP_SCORING, async () => {
    if (currentFepProcess && !currentFepProcess.killed) {
      currentFepProcess.kill('SIGTERM');
      currentFepProcess = null;
    }
  });
}
