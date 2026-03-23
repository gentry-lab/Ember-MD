/**
 * Standalone conformer generation handler (MCMM/ETKDG/CREST).
 */
import { ipcMain } from 'electron';
import { spawn } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import { Ok, Err, Result } from '../../shared/types/result';
import { AppError } from '../../shared/types/errors';
import * as appState from '../app-state';
import { childProcesses } from '../spawn';
import { getQupkakeXtbPath } from '../paths';

export function register(): void {
  ipcMain.handle(
    'conform:generate',
    async (
      event,
      ligandSdfPath: string,
      outputDir: string,
      maxConformers: number,
      rmsdCutoff: number,
      energyWindow: number,
      method: string,
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
          event.sender.send('conform:output', {
            type: 'stdout',
            data: 'Warning: generate_conformers.py not found, skipping conformer generation\n'
          });
          const name = path.basename(ligandSdfPath, '.sdf');
          resolve(Ok({
            conformerPaths: [ligandSdfPath],
            parentMapping: { [name]: name },
          }));
          return;
        }

        fs.mkdirSync(outputDir, { recursive: true });

        const ligandSdfPaths = [ligandSdfPath];
        const ligandListPath = path.join(outputDir, 'ligand_list_for_conformers.json');
        fs.writeFileSync(ligandListPath, JSON.stringify(ligandSdfPaths, null, 2));

        const effectiveMethod = method || 'etkdg';
        const args = [
          scriptPath,
          '--ligand_list', ligandListPath,
          '--output_dir', outputDir,
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

        // xTB reranking (default on for ETKDG/MCMM) and CREST support
        const xtbPath = getQupkakeXtbPath();
        if (xtbPath) {
          args.push('--xtb_binary', xtbPath);
          const shouldRerank = mcmmOptions?.xtbRerank ?? true;
          if (shouldRerank && effectiveMethod !== 'crest') {
            args.push('--xtb_rerank');
          }
        }

        if (effectiveMethod === 'crest') {
          const crestCandidates = [
            appState.condaEnvBin ? path.join(appState.condaEnvBin, 'crest') : '',
            '/usr/local/bin/crest',
          ].filter(Boolean);
          const crestPath = crestCandidates.find(p => fs.existsSync(p));
          if (crestPath) {
            args.push('--crest_binary', crestPath);
          }
        }

        const methodLabel = effectiveMethod.toUpperCase();
        event.sender.send('conform:output', {
          type: 'stdout',
          data: `=== Conformer Generation (${methodLabel}) ===\nMax conformers: ${maxConformers}\nRMSD cutoff: ${rmsdCutoff} A\nEnergy window: ${energyWindow} kcal/mol\nInput: ${path.basename(ligandSdfPath)}\n\n`
        });

        const python = spawn(appState.condaPythonPath, args);
        childProcesses.add(python);

        let stdout = '';
        let stderr = '';

        python.stdout.on('data', (data: Buffer) => {
          const text = data.toString();
          stdout += text;
          event.sender.send('conform:output', { type: 'stdout', data: text });
        });

        python.stderr.on('data', (data: Buffer) => {
          const text = data.toString();
          stderr += text;
          if (text.includes('Warning') || text.includes('ERROR')) {
            event.sender.send('conform:output', { type: 'stderr', data: text });
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
                  conformerPaths: result.conformer_paths || [],
                  parentMapping: result.parent_mapping || {},
                  conformerEnergies: result.conformer_energies || {},
                }));
              } else {
                event.sender.send('conform:output', {
                  type: 'stdout',
                  data: 'Warning: No conformer output, using original molecule\n'
                });
                const name = path.basename(ligandSdfPath, '.sdf');
                resolve(Ok({
                  conformerPaths: [ligandSdfPath],
                  parentMapping: { [name]: name },
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
            resolve(Err({
              type: 'CONFORMER_FAILED',
              message: `Conformer generation failed: ${stderr.slice(0, 200)}`,
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
