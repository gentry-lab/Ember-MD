// Copyright (c) 2026 Ember Contributors. MIT License.
import { ipcMain } from 'electron';
import { spawn, ChildProcess } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { Ok, Err, Result } from '../../shared/types/result';
import { AppError } from '../../shared/types/errors';
import { IpcChannels, XrayAnalysisResult, XrayDirectoryScanResult } from '../../shared/types/ipc';
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

function normalizeStem(stem: string): string {
  let value = stem.toLowerCase();
  for (const suffix of ['_map', '_final', '_refine', '_001']) {
    if (value.endsWith(suffix)) {
      value = value.slice(0, -suffix.length);
    }
  }
  return value;
}

function extractCompoundId(stem: string): string | null {
  const upper = stem.toUpperCase();
  const withVu = [...upper.matchAll(/VU?0*(\d{5,})/g)].map((match) => match[1]);
  if (withVu.length > 0) return withVu[0];
  const plain = [...upper.matchAll(/(\d{5,})/g)].map((match) => match[1]);
  return plain.length > 0 ? plain[0] : null;
}

function scanDirectory(dirPath: string): XrayDirectoryScanResult {
  const entries = fs.existsSync(dirPath) ? fs.readdirSync(dirPath) : [];
  const pdbFiles = entries.filter((name) => /\.(pdb|cif|mmcif)$/i.test(name));
  const mtzFiles = entries.filter((name) => /\.mtz$/i.test(name));

  const mtzByStem = new Map<string, string>();
  const mtzByNorm = new Map<string, string>();
  const mtzById = new Map<string, string>();

  for (const fileName of mtzFiles) {
    const stem = path.parse(fileName).name;
    mtzByStem.set(stem, fileName);
    mtzByNorm.set(normalizeStem(stem), fileName);
    const compoundId = extractCompoundId(stem);
    if (compoundId) mtzById.set(compoundId, fileName);
  }

  let pairedCount = 0;
  for (const fileName of pdbFiles) {
    const stem = path.parse(fileName).name;
    const normalized = normalizeStem(stem);
    const compoundId = extractCompoundId(stem);
    const matched =
      mtzByStem.has(stem) ||
      mtzByNorm.has(normalized) ||
      (compoundId ? mtzById.has(compoundId) : false);
    if (matched) pairedCount += 1;
  }

  return {
    pdbCount: pdbFiles.length,
    mtzCount: mtzFiles.length,
    pairedCount,
    unpairedPdbCount: Math.max(0, pdbFiles.length - pairedCount),
  };
}

export function register(): void {
  ipcMain.handle(
    IpcChannels.SCAN_XRAY_DIRECTORY,
    async (_event, dirPath: string): Promise<Result<XrayDirectoryScanResult, AppError>> => {
      try {
        if (!fs.existsSync(dirPath)) {
          return Err({
            type: 'DIRECTORY_NOT_FOUND',
            message: `Input directory not found: ${dirPath}`,
          });
        }
        return Ok(scanDirectory(dirPath));
      } catch (error) {
        return Err({
          type: 'SCAN_FAILED',
          message: `Failed to scan X-ray directory: ${(error as Error).message}`,
        });
      }
    }
  );

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
        const runRoot = path.dirname(outputDir);
        const stagedInputDir = path.join(runRoot, 'inputs');
        fs.rmSync(stagedInputDir, { recursive: true, force: true });
        fs.cpSync(inputDir, stagedInputDir, { recursive: true });

        const outputPrefix = path.join(outputDir, 'xray_analysis');
        const args = [scriptPath, stagedInputDir, '-o', outputPrefix];

        event.sender.send(IpcChannels.XRAY_OUTPUT, {
          type: 'stdout',
          data:
            `=== X-ray Pose Scoring ===\n` +
            `Input folder: ${stagedInputDir}\n` +
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
          for (const line of text.split('\n')) {
            if (line.trim()) console.log(`[Xray] ${line}`);
          }
        });

        proc.stderr.on('data', (data: Buffer) => {
          const text = data.toString();
          stderr += text;
          for (const line of text.split('\n')) {
            if (line.trim()) console.log(`[Xray:err] ${line}`);
          }
          if (text.includes('Warning') || text.includes('ERROR') || text.includes('Error')) {
            event.sender.send(IpcChannels.XRAY_OUTPUT, { type: 'stderr', data: text });
          }
        });

        proc.on('close', (code: number | null) => {
          childProcesses.delete(proc);
          xrayProcesses.delete(proc);

          if (code === 0) {
            resolve(Ok({
              inputDir: stagedInputDir,
              outputDir,
              pdfPaths: listGeneratedPdfs(outputDir),
            }));
            return;
          }

          const errorText = stderr.trim() || stdout.trim() || 'Unknown error';
          console.error(`[Xray] ANALYSIS_FAILED: ${errorText.slice(0, 400)}`);
          event.sender.send(IpcChannels.XRAY_OUTPUT, {
            type: 'stderr',
            data: `Error [ANALYSIS_FAILED]: ${errorText.slice(0, 200)}\n`
          });
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
