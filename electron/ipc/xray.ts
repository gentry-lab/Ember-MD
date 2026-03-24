import { ipcMain } from 'electron';
import { spawn, ChildProcess } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { Ok, Err, Result } from '../../shared/types/result';
import { AppError } from '../../shared/types/errors';
import { IpcChannels, XrayAnalysisResult } from '../../shared/types/ipc';
import * as appState from '../app-state';
import { childProcesses, getSpawnEnv as _getSpawnEnv } from '../spawn';

const xrayProcesses = new Set<ChildProcess>();

function getSpawnEnv(): NodeJS.ProcessEnv {
  return _getSpawnEnv(appState.condaEnvBin);
}

function listGeneratedPdfs(outputDir: string): string[] {
  if (!fs.existsSync(outputDir)) return [];
  return fs.readdirSync(outputDir)
    .filter((name) => /^xray_analysis_.*\.pdf$/i.test(name))
    .map((name) => path.join(outputDir, name))
    .sort((a, b) => a.localeCompare(b));
}

export function register(): void {
  ipcMain.handle(
    IpcChannels.RUN_XRAY_ANALYSIS,
    async (event, inputDir: string, outputDir: string): Promise<Result<XrayAnalysisResult, AppError>> => {
      return new Promise((resolve) => {
        if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
          resolve(Err({
            type: 'PYTHON_NOT_FOUND',
            message: 'Python not found. Please install miniconda and create the openmm-metal environment.',
          }));
          return;
        }

        if (!fs.existsSync(inputDir)) {
          resolve(Err({
            type: 'DIRECTORY_NOT_FOUND',
            message: `Input directory not found: ${inputDir}`,
          }));
          return;
        }

        const scriptPath = path.join(appState.fraggenRoot, 'xray_analyzer.py');
        if (!fs.existsSync(scriptPath)) {
          resolve(Err({
            type: 'SCRIPT_NOT_FOUND',
            path: scriptPath,
            message: 'xray_analyzer.py not found in bundled scripts directory.',
          }));
          return;
        }

        fs.mkdirSync(outputDir, { recursive: true });

        const outputPrefix = path.join(outputDir, 'xray_analysis');
        const args = [scriptPath, inputDir, '-o', outputPrefix];

        event.sender.send(IpcChannels.XRAY_OUTPUT, {
          type: 'stdout',
          data:
            `=== X-ray Pose Scoring ===\n` +
            `Input folder: ${inputDir}\n` +
            `Output folder: ${outputDir}\n\n`,
        });

        const proc = spawn(appState.condaPythonPath, args, {
          cwd: outputDir,
          env: getSpawnEnv(),
        });
        childProcesses.add(proc);
        xrayProcesses.add(proc);

        let stdout = '';
        let stderr = '';

        proc.stdout.on('data', (data: Buffer) => {
          const text = data.toString();
          stdout += text;
          event.sender.send(IpcChannels.XRAY_OUTPUT, { type: 'stdout', data: text });
        });

        proc.stderr.on('data', (data: Buffer) => {
          const text = data.toString();
          stderr += text;
          event.sender.send(IpcChannels.XRAY_OUTPUT, { type: 'stderr', data: text });
        });

        proc.on('close', (code: number | null) => {
          childProcesses.delete(proc);
          xrayProcesses.delete(proc);

          if (code === 0) {
            resolve(Ok({
              inputDir,
              outputDir,
              pdfPaths: listGeneratedPdfs(outputDir),
            }));
            return;
          }

          const errorText = stderr.trim() || stdout.trim() || 'Unknown error';
          resolve(Err({
            type: 'ANALYSIS_FAILED',
            message: `X-ray pose scoring failed: ${errorText.slice(0, 400)}`,
          }));
        });

        proc.on('error', (error: Error) => {
          childProcesses.delete(proc);
          xrayProcesses.delete(proc);
          resolve(Err({
            type: 'ANALYSIS_FAILED',
            message: error.message,
          }));
        });
      });
    }
  );
}
