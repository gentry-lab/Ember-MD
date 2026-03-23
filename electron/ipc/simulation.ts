/**
 * MD simulation IPC handlers.
 * Covers MD benchmark, simulation run/cancel/pause/resume, dock output loading,
 * viewing preparation, and Ember job folder selection.
 */
import { ipcMain, app, dialog } from 'electron';
import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import { Ok, Err, Result } from '../../shared/types/result';
import { AppError } from '../../shared/types/errors';
import { IpcChannels } from '../../shared/types/ipc';
import * as appState from '../app-state';
import { childProcesses, loadAndMergeCordialScores, filterMdStderr } from '../spawn';
import {
  readJsonIfExists,
  extractVinaAffinity,
  getSimulationClusterArtifacts,
  getBindingSiteResultFile,
} from './projects';

// ---------------------------------------------------------------------------
// Module-level state
// ---------------------------------------------------------------------------

let currentBenchmarkProcess: ChildProcess | null = null;
let currentMdProcess: ChildProcess | null = null;

// ---------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------

interface MDConfig {
  productionNs: number;
  forceFieldPreset: string;
  compoundId?: string;
  temperatureK?: number;
  saltConcentrationM?: number;
  paddingNm?: number;
  restrainLigandNs?: number;
  seed?: number;
}

interface MDBenchmarkResult {
  nsPerDay: number;
  systemInfo: {
    atomCount: number;
    boxVolumeA3: number;
  };
}

interface MDLoadedLigand {
  name: string;
  sdfPath: string;
  smiles: string;
  vinaAffinity: number;
  qed: number;
  mw?: number;
  logp?: number;
  cordialPHighAffinity?: number;
  cordialExpectedPkd?: number;
  thumbnail?: string;
}

interface MDDockOutput {
  receptorPdb: string;
  ligands: MDLoadedLigand[];
}

// ---------------------------------------------------------------------------
// Local types (not in shared/types)
// ---------------------------------------------------------------------------

type MapJobMetadata = {
  method?: 'static' | 'solvation' | 'probe';
  sourcePdbPath?: string;
  sourceTrajectoryPath?: string;
  ligandResname?: string;
  ligandResnum?: number;
  computedAt?: string;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Helper function to parse SDF properties using Python script
function parseSdfProperties(sdfPath: string): Promise<{
  success: boolean;
  error?: string;
  smiles?: string;
  vinaAffinity: number | null;
  vinaScoreOnlyAffinity?: number;
  refinementEnergy?: number;
  isReferencePose?: boolean;
  qed: number;
  mw: number;
  logp: number;
  thumbnail?: string;
  coreRmsd?: number;
}> {
  return new Promise((resolve) => {
    if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
      resolve({
        success: false,
        error: 'Python not found',
        vinaAffinity: null,
        qed: 0,
        mw: 0,
        logp: 0,
      });
      return;
    }

    const scriptPath = path.join(appState.fraggenRoot, 'parse_sdf_properties.py');
    if (!fs.existsSync(scriptPath)) {
      resolve({
        success: false,
        error: 'parse_sdf_properties.py not found',
        vinaAffinity: null,
        qed: 0,
        mw: 0,
        logp: 0,
      });
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

    python.on('close', (code) => {
      if (code === 0 && stdout.trim()) {
        try {
          const result = JSON.parse(stdout.trim());
          resolve({
            success: result.success,
            error: result.error,
            smiles: result.smiles,
            vinaAffinity: result.vinaAffinity ?? result.minimizedAffinity ?? null,
            vinaScoreOnlyAffinity: result.vinaScoreOnlyAffinity ?? undefined,
            refinementEnergy: result.refinementEnergy ?? undefined,
            isReferencePose: result.isReferencePose === true,
            qed: result.qed || 0,
            mw: result.mw || 0,
            logp: result.logp || 0,
            thumbnail: result.thumbnail,
            coreRmsd: result.coreRMSD != null ? parseFloat(result.coreRMSD) : undefined,
          });
        } catch (e) {
          resolve({
            success: false,
            error: 'Failed to parse JSON output',
            vinaAffinity: null,
            qed: 0,
            mw: 0,
            logp: 0,
          });
        }
      } else {
          resolve({
            success: false,
            error: stderr || 'Script failed',
            vinaAffinity: null,
            qed: 0,
            mw: 0,
            logp: 0,
          });
      }
    });

    python.on('error', (err) => {
        resolve({
          success: false,
          error: err.message,
          vinaAffinity: null,
          qed: 0,
          mw: 0,
          logp: 0,
        });
    });
  });
}

// ---------------------------------------------------------------------------
// IPC handler registration
// ---------------------------------------------------------------------------

export function register(): void {

  // Load docking output directory for MD
  ipcMain.handle(
    IpcChannels.LOAD_DOCK_OUTPUT_FOR_MD,
    async (_event, dirPath: string): Promise<Result<MDDockOutput, AppError>> => {
      try {
        // Check if directory exists
        if (!fs.existsSync(dirPath)) {
          return Err({
            type: 'DIRECTORY_ERROR',
            path: dirPath,
            message: 'Directory does not exist',
          });
        }

        // Look for receptor: inputs/receptor.pdb (new), *_receptor_prepared.pdb (legacy)
        const files0 = fs.readdirSync(dirPath);
        const prefixedReceptor = files0.find((f) => f.endsWith('_receptor_prepared.pdb'));

        const receptorCandidates = [
          // New layout: inputs/receptor.pdb
          path.join(dirPath, 'inputs', 'receptor.pdb'),
          // Legacy: {projectName}_receptor_prepared.pdb in docking dir
          ...(prefixedReceptor ? [path.join(dirPath, prefixedReceptor)] : []),
          // Very old legacy: receptor_prepared.pdb in docking dir or parent dirs
          path.join(dirPath, 'receptor_prepared.pdb'),
          path.join(path.dirname(dirPath), 'receptor_prepared.pdb'),
          path.join(path.dirname(path.dirname(dirPath)), 'receptor_prepared.pdb'),
        ];

        let receptorPdb = '';
        for (const candidate of receptorCandidates) {
          if (fs.existsSync(candidate)) {
            receptorPdb = candidate;
            break;
          }
        }

        if (!receptorPdb) {
          return Err({
            type: 'FILE_NOT_FOUND',
            path: path.join(dirPath, 'receptor_prepared.pdb'),
            message: 'receptor_prepared.pdb not found in docking output directory or parent directories. Please ensure you ran docking with a prepared receptor.',
          });
        }

        // Find all *_docked.sdf.gz or *.sdf files (check results/poses/ first, then poses/, then top-level)
        const newPosesSubDir = path.join(dirPath, 'results', 'poses');
        const legacyPosesSubDir = path.join(dirPath, 'poses');
        const sdfSearchDir = fs.existsSync(newPosesSubDir) ? newPosesSubDir : fs.existsSync(legacyPosesSubDir) ? legacyPosesSubDir : dirPath;
        const files = fs.readdirSync(sdfSearchDir);
        const dockedFiles = files.filter((f) => f.endsWith('_docked.sdf.gz'));
        // Also check top-level for regular SDF files (legacy or manual)
        const topFiles = sdfSearchDir !== dirPath ? fs.readdirSync(dirPath) : files;
        const regularSdfFiles = topFiles.filter((f) => f.endsWith('.sdf') && !f.endsWith('.sdf.gz') && !f.includes('_all_docked'));

        // Use docked files if available, otherwise fall back to regular SDF files
        const sdfFiles = dockedFiles.length > 0 ? dockedFiles : regularSdfFiles;

        if (sdfFiles.length === 0) {
          return Err({
            type: 'FILE_NOT_FOUND',
            path: dirPath,
            message: 'No SDF files found in directory. Expected *_docked.sdf.gz or *.sdf files.',
          });
        }

        // Look for parent FragGen directory for legacy thumbnails
        let thumbnailDir = '';
        const parentDir = path.dirname(dirPath);
        const grandparentDir = path.dirname(parentDir);

        const thumbnailCandidates = [
          path.join(parentDir, 'thumbnails'),
          path.join(grandparentDir, 'thumbnails'),
          path.join(parentDir, '..', 'thumbnails'),
          path.join(grandparentDir, '..', 'thumbnails'),
        ];

        for (const candidate of thumbnailCandidates) {
          if (fs.existsSync(candidate)) {
            thumbnailDir = candidate;
            break;
          }
        }

        // Parse each SDF file directly for properties
        // Process in parallel for better performance
        const ligands: MDLoadedLigand[] = [];
        const parsePromises: Promise<MDLoadedLigand | null>[] = sdfFiles.map(async (sdfFile) => {
          const isDockedFile = sdfFile.endsWith('_docked.sdf.gz');
          const name = isDockedFile
            ? sdfFile.replace('_docked.sdf.gz', '')
            : sdfFile.replace('.sdf', '');
          const sdfPath = path.join(isDockedFile ? sdfSearchDir : dirPath, sdfFile);

          // Parse SDF for all properties (SMILES, scores, QED, thumbnail)
          const props = await parseSdfProperties(sdfPath);

          // Check for legacy thumbnail file
          let thumbnailPath: string | undefined;
          if (thumbnailDir) {
            const thumbFile = path.join(thumbnailDir, `${name}.png`);
            if (fs.existsSync(thumbFile)) {
              thumbnailPath = thumbFile;
            }
          }

          if (props.isReferencePose) {
            return null;
          }

          return {
            name,
            sdfPath,
            smiles: props.smiles || '',
            vinaAffinity: props.vinaAffinity ?? props.vinaScoreOnlyAffinity ?? 0,
            qed: props.qed,
            mw: props.mw,
            logp: props.logp,
            thumbnail: props.thumbnail,
          };
        });

        const parsedLigands = await Promise.all(parsePromises);
        ligands.push(...parsedLigands.filter((ligand): ligand is MDLoadedLigand => ligand !== null));

        loadAndMergeCordialScores(dirPath, ligands, 'name');

        // Sort by vinaAffinity ascending (most negative = best)
        ligands.sort((a, b) => a.vinaAffinity - b.vinaAffinity);

        return Ok({
          receptorPdb,
          ligands,
        });
      } catch (error) {
        return Err({
          type: 'PARSE_FAILED',
          message: (error as Error).message,
        });
      }
    }
  );

  // Run MD benchmark to estimate performance
  ipcMain.handle(
    IpcChannels.RUN_MD_BENCHMARK,
    async (
      event,
      receptorPdb: string | null,
      ligandSdf: string,
      outputDir: string,
      forceFieldPreset: string = 'ff19sb-opc',
      ligandOnly: boolean = false
    ): Promise<Result<MDBenchmarkResult, AppError>> => {
      return new Promise((resolve) => {
        if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
          resolve(Err({
            type: 'PYTHON_NOT_FOUND',
            message: 'Python not found. Please install miniconda and create fraggen environment.',
          }));
          return;
        }

        const scriptPath = path.join(appState.fraggenRoot, 'run_md_simulation.py');
        if (!fs.existsSync(scriptPath)) {
          resolve(Err({
            type: 'SCRIPT_NOT_FOUND',
            path: scriptPath,
            message: `MD simulation script not found: ${scriptPath}`,
          }));
          return;
        }

        fs.mkdirSync(outputDir, { recursive: true });

        const args = [scriptPath];

        if (!ligandOnly && receptorPdb) {
          args.push('--receptor', receptorPdb);
        }

        args.push(
          '--ligand', ligandSdf,
          '--output_dir', outputDir,
          '--force_field_preset', forceFieldPreset || 'ff19sb-opc',
          '--benchmark_only',
        );

        if (ligandOnly) {
          args.push('--ligand_only');
        }

        // Kill any previous benchmark still running
        if (currentBenchmarkProcess && !currentBenchmarkProcess.killed) {
          currentBenchmarkProcess.kill('SIGTERM');
        }

        const python = spawn(appState.condaPythonPath, args);
        childProcesses.add(python);
        currentBenchmarkProcess = python;

        let stdout = '';
        let systemInfo = { atomCount: 0, boxVolumeA3: 0 };
        let nsPerDay = 0;

        python.stdout.on('data', (data: Buffer) => {
          const text = data.toString();
          stdout += text;
          event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stdout', data: text });

          // Parse SYSTEM_INFO:atomCount:volume
          const systemMatch = text.match(/SYSTEM_INFO:(\d+):(\d+)/);
          if (systemMatch) {
            systemInfo.atomCount = parseInt(systemMatch[1]);
            systemInfo.boxVolumeA3 = parseInt(systemMatch[2]);
          }

          // Parse BENCHMARK:nsPerDay
          const benchmarkMatch = text.match(/BENCHMARK:([\d.]+)/);
          if (benchmarkMatch) {
            nsPerDay = parseFloat(benchmarkMatch[1]);
          }
        });

        python.stderr.on('data', (data: Buffer) => {
          const filtered = filterMdStderr(data.toString());
          if (filtered.trim()) {
            event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stderr', data: filtered });
          }
        });

        python.on('close', (code: number | null) => {
          childProcesses.delete(python);
          if (currentBenchmarkProcess === python) currentBenchmarkProcess = null;
          if (code === 0 && nsPerDay > 0) {
            resolve(Ok({
              nsPerDay,
              systemInfo,
            }));
          } else {
            resolve(Err({
              type: 'BENCHMARK_FAILED',
              message: `Benchmark failed with code ${code}`,
            }));
          }
        });

        python.on('error', (error: Error) => {
          childProcesses.delete(python);
          if (currentBenchmarkProcess === python) currentBenchmarkProcess = null;
          resolve(Err({
            type: 'BENCHMARK_FAILED',
            message: error.message,
          }));
        });
      });
    }
  );

  // Run full MD simulation
  ipcMain.handle(
    IpcChannels.RUN_MD_SIMULATION,
    async (
      event,
      receptorPdb: string | null,
      ligandSdf: string,
      outputDir: string,
      config: MDConfig,
      ligandOnly: boolean = false,
      apo: boolean = false
    ): Promise<Result<string, AppError>> => {
      return new Promise((resolve) => {
        if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
          resolve(Err({
            type: 'PYTHON_NOT_FOUND',
            message: 'Python not found. Please install miniconda and create fraggen environment.',
          }));
          return;
        }

        const scriptPath = path.join(appState.fraggenRoot, 'run_md_simulation.py');
        if (!fs.existsSync(scriptPath)) {
          resolve(Err({
            type: 'SCRIPT_NOT_FOUND',
            path: scriptPath,
            message: `MD simulation script not found: ${scriptPath}`,
          }));
          return;
        }

        fs.mkdirSync(outputDir, { recursive: true });

        const args = [
          scriptPath,
          '--output_dir', outputDir,
          '--production_ns', String(config.productionNs),
          '--force_field_preset', config.forceFieldPreset || 'ff19sb-opc',
          '--temperature', String(config.temperatureK || 300),
          '--salt_concentration', String(config.saltConcentrationM || 0.15),
          '--padding', String(config.paddingNm || 1.2),
        ];

        if (!apo && ligandSdf) {
          args.push('--ligand', ligandSdf);
        }

        if (config.restrainLigandNs && config.restrainLigandNs > 0) {
          args.push('--restrain_ligand_ns', String(config.restrainLigandNs));
        }

        if (config.seed && config.seed > 0) {
          args.push('--seed', String(config.seed));
        }

        if (apo) {
          args.push('--apo');
          if (receptorPdb) args.push('--receptor', receptorPdb);
        } else if (ligandOnly) {
          args.push('--ligand_only');
        } else if (receptorPdb) {
          args.push('--receptor', receptorPdb);
        }

        // Kill any running benchmark before starting simulation
        if (currentBenchmarkProcess && !currentBenchmarkProcess.killed) {
          currentBenchmarkProcess.kill('SIGTERM');
          currentBenchmarkProcess = null;
        }

        console.log('Running MD simulation:', appState.condaPythonPath, args.join(' '));

        const python = spawn(appState.condaPythonPath, args);
        childProcesses.add(python);
        currentMdProcess = python;

        let trajectoryPath = '';

        python.stdout.on('data', (data: Buffer) => {
          const text = data.toString();
          event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stdout', data: text });

          // Parse SUCCESS:path
          const successMatch = text.match(/SUCCESS:(.+)/);
          if (successMatch) {
            trajectoryPath = successMatch[1].trim();
          }
        });

        python.stderr.on('data', (data: Buffer) => {
          const filtered = filterMdStderr(data.toString());
          if (filtered.trim()) {
            event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stderr', data: filtered });
          }
        });

        python.on('close', (code: number | null) => {
          childProcesses.delete(python);
          currentMdProcess = null;
          if (code === 0 && trajectoryPath) {
            resolve(Ok(trajectoryPath));
          } else {
            resolve(Err({
              type: 'SIMULATION_FAILED',
              message: `Simulation failed with code ${code}`,
            }));
          }
        });

        python.on('error', (error: Error) => {
          childProcesses.delete(python);
          currentMdProcess = null;
          resolve(Err({
            type: 'SIMULATION_FAILED',
            message: error.message,
          }));
        });
      });
    }
  );

  // Cancel running benchmark
  ipcMain.handle(IpcChannels.CANCEL_MD_BENCHMARK, async (): Promise<void> => {
    if (currentBenchmarkProcess && !currentBenchmarkProcess.killed) {
      currentBenchmarkProcess.kill('SIGTERM');
      currentBenchmarkProcess = null;
    }
  });

  // Cancel running MD simulation
  ipcMain.handle(IpcChannels.CANCEL_MD_SIMULATION, async (): Promise<void> => {
    if (currentMdProcess && !currentMdProcess.killed) {
      currentMdProcess.kill('SIGTERM');
      currentMdProcess = null;
    }
  });

  // Pause running MD simulation (SIGSTOP)
  ipcMain.handle(IpcChannels.PAUSE_MD_SIMULATION, async (): Promise<void> => {
    if (currentMdProcess && !currentMdProcess.killed) {
      currentMdProcess.kill('SIGSTOP');
    }
  });

  // Resume paused MD simulation (SIGCONT)
  ipcMain.handle(IpcChannels.RESUME_MD_SIMULATION, async (): Promise<void> => {
    if (currentMdProcess && !currentMdProcess.killed) {
      currentMdProcess.kill('SIGCONT');
    }
  });

  // Prepare a PDB for viewing: add missing hydrogens via PDBFixer
  ipcMain.handle(
    IpcChannels.PREPARE_FOR_VIEWING,
    async (_event, rawPdbPath: string, preparedPath: string): Promise<Result<string, AppError>> => {
      return new Promise((resolve) => {
        if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
          // No Python — just use raw file
          resolve(Ok(rawPdbPath));
          return;
        }

        const scriptPath = path.join(appState.fraggenRoot, 'detect_pdb_ligands.py');
        if (!fs.existsSync(scriptPath)) {
          resolve(Ok(rawPdbPath));
          return;
        }

        fs.mkdirSync(path.dirname(preparedPath), { recursive: true });

        const python = spawn(appState.condaPythonPath, [
          scriptPath,
          '--pdb', rawPdbPath,
          '--mode', 'add_hydrogens',
          '--output', preparedPath,
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
          if (code === 0 && fs.existsSync(preparedPath)) {
            resolve(Ok(preparedPath));
          } else {
            // Fall back to raw file on any failure
            console.error('[prepare-for-viewing] Failed:', stderr);
            resolve(Ok(rawPdbPath));
          }
        });

        python.on('error', () => {
          resolve(Ok(rawPdbPath));
        });
      });
    }
  );

  // Select an Ember job folder via dialog, validate, and return as ProjectJob
  ipcMain.handle(
    IpcChannels.SELECT_EMBER_JOB_FOLDER,
    async (): Promise<any | null> => {
      const emberDir = path.join(app.getPath('home'), 'Ember');
      const result = await dialog.showOpenDialog({
        title: 'Select Ember Job Folder',
        defaultPath: emberDir,
        properties: ['openDirectory'],
      });
      if (result.canceled || result.filePaths.length === 0) return null;

      const folderPath = result.filePaths[0];
      const folderName = path.basename(folderPath);
      const folderFiles = fs.readdirSync(folderPath);
      const folderStat = fs.statSync(folderPath);

      // Check for docking job: inputs/receptor.pdb + results/poses/*_docked.sdf.gz
      const newReceptor = path.join(folderPath, 'inputs', 'receptor.pdb');
      const legacyReceptor = folderFiles.find((f) => f.includes('_receptor_prepared') && f.endsWith('.pdb'));
      const receptorPdb = fs.existsSync(newReceptor) ? newReceptor : (legacyReceptor ? path.join(folderPath, legacyReceptor) : undefined);

      const newPosesDir = path.join(folderPath, 'results', 'poses');
      const legacyPosesDir = path.join(folderPath, 'poses');
      const posesSearchDir = fs.existsSync(newPosesDir) ? newPosesDir : fs.existsSync(legacyPosesDir) ? legacyPosesDir : folderPath;

      let poseFiles: string[] = [];
      try {
        poseFiles = fs.readdirSync(posesSearchDir).filter((f) => f.endsWith('_docked.sdf.gz'));
      } catch { /* ignore */ }

      if (receptorPdb && poseFiles.length > 0) {
        const poses = poseFiles.map((f) => ({
          name: f.replace('_docked.sdf.gz', ''),
          path: path.join(posesSearchDir, f),
          affinity: extractVinaAffinity(path.join(posesSearchDir, f)),
        }));
        poses.sort((a, b) => (a.affinity ?? 0) - (b.affinity ?? 0));
        return {
          id: `dock:${folderName}`,
          type: 'docking',
          folder: folderName,
          label: `${folderName} (${poses.length} poses)`,
          path: folderPath,
          lastModified: folderStat.mtimeMs,
          receptorPdb,
          poses,
        };
      }

      // Check for simulation job: system.pdb + trajectory.dcd (or results/ subdir)
      const resultsDir = path.join(folderPath, 'results');
      const resultsFiles = fs.existsSync(resultsDir) ? fs.readdirSync(resultsDir) : [];
      const runFiles = folderFiles;

      let systemPdb: string | undefined;
      let trajectoryDcd: string | undefined;

      if (resultsFiles.includes('system.pdb')) systemPdb = path.join(resultsDir, 'system.pdb');
      if (resultsFiles.includes('trajectory.dcd')) trajectoryDcd = path.join(resultsDir, 'trajectory.dcd');
      if (!systemPdb) {
        const legacy = runFiles.find((f) => f.endsWith('_system.pdb') || f === 'system.pdb');
        if (legacy) systemPdb = path.join(folderPath, legacy);
      }
      if (!trajectoryDcd) {
        const legacy = runFiles.find((f) => f.endsWith('_trajectory.dcd') || f === 'trajectory.dcd');
        if (legacy) trajectoryDcd = path.join(folderPath, legacy);
      }

	    if (systemPdb) {
	      const {
	        clusterCount,
	        clusterDirPath,
	        clusteringResultsPath,
	      } = getSimulationClusterArtifacts(folderPath);
	      return {
	        id: `sim:${folderName}`,
	        type: 'simulation',
	        folder: folderName,
	        label: folderName,
        path: folderPath,
        lastModified: folderStat.mtimeMs,
	        systemPdb,
	        trajectoryDcd,
	        hasTrajectory: !!trajectoryDcd,
	        clusterCount,
	        clusterDir: clusterDirPath,
	        clusteringResultsPath,
	      };
	    }

      const conformerFiles = folderFiles
        .filter((f) => /\.(sdf|sdf\.gz|mol|mol2)$/i.test(f))
        .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
      if (conformerFiles.length > 0) {
        return {
          id: `conformer:${folderName}`,
          type: 'conformer',
          folder: folderName,
          label: `${folderName} (${conformerFiles.length} conformers)`,
          path: folderPath,
          lastModified: folderStat.mtimeMs,
          conformerPaths: conformerFiles.map((f) => path.join(folderPath, f)),
          conformerCount: conformerFiles.length,
        };
      }

      const projectDir = path.dirname(path.dirname(folderPath));
      const projectName = path.basename(projectDir);
      const mapResultJson = getBindingSiteResultFile(folderPath, projectName);
      if (mapResultJson) {
        const metadata = readJsonIfExists<MapJobMetadata>(path.join(folderPath, 'map_metadata.json'));
        const mapResult = readJsonIfExists<{ hotspots?: unknown[] }>(mapResultJson);
        const inferredMethod: 'solvation' = 'solvation';
        return {
          id: `map:${folderName}`,
          type: 'map',
          folder: folderName,
          label: folderName,
          path: folderPath,
          lastModified: folderStat.mtimeMs,
          mapMethod: inferredMethod,
          mapResultJson,
          mapPdb: metadata?.sourcePdbPath,
          mapTrajectoryDcd: metadata?.sourceTrajectoryPath,
          hotspotCount: Array.isArray(mapResult?.hotspots) ? mapResult.hotspots.length : 0,
        };
      }

      return null;
    }
  );
}
