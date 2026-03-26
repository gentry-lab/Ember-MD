// Copyright (c) 2026 Ember Contributors. MIT License.
import { ipcMain, dialog } from 'electron';
import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import * as crypto from 'crypto';
import { Ok, Err, Result } from '../../shared/types/result';
import { AppError } from '../../shared/types/errors';
import { IpcChannels } from '../../shared/types/ipc';
import type { ClusteringResult, ScoredClusterResult, MdTorsionAnalysis } from '../../shared/types/ipc';
import * as appState from '../app-state';
import { childProcesses, loadAndMergeCordialScores, filterMdStderr } from '../spawn';
import { getSpawnEnv as _getSpawnEnv } from '../spawn';
import { spawnPythonScript as _spawnPythonScriptRaw } from '../spawn';
import { getXtbPath, getCordialRoot, detectBabelDataDir } from '../paths';
import {
  runVinaScoreOnly,
  runCordialScoringJob,
} from '../scoring-utils';

function getSpawnEnv(): NodeJS.ProcessEnv {
  return _getSpawnEnv(appState.condaEnvBin);
}

function spawnPythonScript(
  args: string[],
  options?: {
    env?: NodeJS.ProcessEnv;
    cwd?: string;
    onStdout?: (text: string) => void;
    onStderr?: (text: string) => void;
  }
): Promise<{ stdout: string; stderr: string; code: number }> {
  return _spawnPythonScriptRaw(appState.condaPythonPath, appState.condaEnvBin, args, options);
}

// --- Helper types ---

type MapJobMetadata = {
  method?: 'static' | 'solvation' | 'probe';
  sourcePdbPath?: string;
  sourceTrajectoryPath?: string;
  ligandResname?: string;
  ligandResnum?: number;
  computedAt?: string;
};

// --- Helper functions ---

const readJsonIfExists = <T>(jsonPath: string): T | null => {
  try {
    if (!fs.existsSync(jsonPath)) return null;
    return JSON.parse(fs.readFileSync(jsonPath, 'utf-8')) as T;
  } catch {
    return null;
  }
};

const getCanonicalAnalysisRoot = (runPath: string) => path.join(runPath, 'analysis');

const normalizeAnalysisDir = (inputPath: string) =>
  path.basename(inputPath) === 'analysis' ? inputPath : getCanonicalAnalysisRoot(inputPath);

const readClusteringResult = (directoryPath: string): ClusteringResult | null => {
  const resultsPath = path.join(directoryPath, 'clustering_results.json');
  return readJsonIfExists<ClusteringResult>(resultsPath);
};

const readClusterScoreRows = (directoryPath: string): ScoredClusterResult[] => {
  const resultsPath = path.join(directoryPath, 'cluster_scores.json');
  const scoreData = readJsonIfExists<{ clusters?: ScoredClusterResult[] }>(resultsPath);
  return Array.isArray(scoreData?.clusters) ? scoreData!.clusters : [];
};

const writeClusterScoreRows = (directoryPath: string, clusters: ScoredClusterResult[]): void => {
  const resultsPath = path.join(directoryPath, 'cluster_scores.json');
  fs.writeFileSync(resultsPath, JSON.stringify({ clusters }, null, 2));
};

const mergeClusterScoresWithCanonical = (
  clusteringResults: ClusteringResult,
  scoreClusters: Array<Partial<ScoredClusterResult> & { clusterId: number }>,
): ScoredClusterResult[] => {
  const scoreMap = new Map(scoreClusters.map((cluster) => [cluster.clusterId, cluster]));
  return clusteringResults.clusters.map((cluster) => {
    const scored = scoreMap.get(cluster.clusterId);
    return {
      clusterId: cluster.clusterId,
      frameCount: cluster.frameCount,
      population: cluster.population,
      centroidFrame: cluster.centroidFrame,
      centroidPdbPath: cluster.centroidPdbPath || scored?.centroidPdbPath || '',
      receptorPdbPath: scored?.receptorPdbPath,
      ligandSdfPath: scored?.ligandSdfPath,
      vinaRescore: scored?.vinaRescore,
      cordialExpectedPkd: scored?.cordialExpectedPkd,
      cordialPHighAffinity: scored?.cordialPHighAffinity,
      cordialPVeryHighAffinity: scored?.cordialPVeryHighAffinity,
    };
  });
};

// runVinaScoreOnly, runCordialScoringJob — imported from ../scoring-utils

const getBindingSiteResultFile = (outputDir: string, projectName?: string): string | null => {
  const prefixedPath = projectName ? path.join(outputDir, `${projectName}_binding_site_results.json`) : null;
  if (prefixedPath && fs.existsSync(prefixedPath)) return prefixedPath;

  const legacyPath = path.join(outputDir, 'binding_site_results.json');
  if (fs.existsSync(legacyPath)) return legacyPath;

  return null;
};

const writeMapMetadata = (
  outputDir: string,
  metadata: MapJobMetadata,
) => {
  fs.mkdirSync(outputDir, { recursive: true });
  const metadataPath = path.join(outputDir, 'map_metadata.json');
  fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));
};

const normalizeMdTorsionAnalysis = (raw: any): MdTorsionAnalysis | null => {
  if (!raw || raw.type !== 'torsions') return null;
  const torsions = Array.isArray(raw?.data?.torsions)
    ? raw.data.torsions.map((row: any) => ({
        torsionId: String(row?.torsionId ?? ''),
        bondId: String(row?.bondId ?? ''),
        bondIndex: Number(row?.bondIndex ?? 0),
        centralBondAtomIndices: Array.isArray(row?.centralBondAtomIndices) ? row.centralBondAtomIndices.map(Number) : [],
        quartetAtomIndices: Array.isArray(row?.quartetAtomIndices) ? row.quartetAtomIndices.map(Number) : [],
        atomNames: Array.isArray(row?.atomNames) ? row.atomNames.map(String) : [],
        label: String(row?.label ?? ''),
        circularMean: Number(row?.circularMean ?? 0),
        circularStd: Number(row?.circularStd ?? 0),
        min: Number(row?.min ?? 0),
        max: Number(row?.max ?? 0),
        median: Number(row?.median ?? 0),
        nFrames: Number(row?.nFrames ?? 0),
        trajectoryAngles: Array.isArray(row?.trajectoryAngles) ? row.trajectoryAngles.map(Number) : [],
        clusterValues: Array.isArray(row?.clusterValues)
          ? row.clusterValues.map((value: any) => ({
              clusterId: Number(value?.clusterId ?? 0),
              centroidFrame: Number(value?.centroidFrame ?? 0),
              population: Number(value?.population ?? 0),
              angle: Number(value?.angle ?? 0),
            }))
          : [],
      }))
    : [];

  return {
    type: 'torsions',
    pdfPath: raw?.pdfPath ?? null,
    csvPath: raw?.csvPath ?? null,
    ligandPresent: Boolean(raw?.ligandPresent),
    ligandSdfPath: raw?.ligandSdfPath ?? null,
    nFrames: Number(raw?.nFrames ?? 0),
    nSampledFrames: Number(raw?.nSampledFrames ?? 0),
    stride: Number(raw?.stride ?? 1),
    sampledFrameIndices: Array.isArray(raw?.sampledFrameIndices) ? raw.sampledFrameIndices.map(Number) : [],
    nRotatableBonds: Number(raw?.nRotatableBonds ?? torsions.length),
    depiction: raw?.depiction ?? null,
    data: {
      torsions,
    },
  };
};

// === Registration ===

export function register(): void {

// Select DCD file dialog
ipcMain.handle(IpcChannels.SELECT_DCD_FILE, async (): Promise<string | null> => {
  const result = await dialog.showOpenDialog(appState.mainWindow!, {
    title: 'Select Trajectory File',
    filters: [
      { name: 'DCD Trajectory', extensions: ['dcd'] },
      { name: 'All Files', extensions: ['*'] },
    ],
    properties: ['openFile'],
  });

  if (result.canceled || result.filePaths.length === 0) {
    return null;
  }

  return result.filePaths[0];
});

// List PDB files in a directory (for topology auto-detection)
ipcMain.handle(
  IpcChannels.LIST_PDB_IN_DIRECTORY,
  async (_event, dirPath: string): Promise<string[]> => {
    try {
      if (!fs.existsSync(dirPath)) {
        return [];
      }
      const files = fs.readdirSync(dirPath)
        .filter((f) => f.toLowerCase().endsWith('.pdb'))
        .sort();
      return files;
    } catch (error) {
      console.error('Error listing PDB files:', error);
      return [];
    }
  }
);

// Get trajectory info (frame count, timestep, duration)
ipcMain.handle(
  IpcChannels.GET_TRAJECTORY_INFO,
  async (
    _event,
    topologyPath: string,
    trajectoryPath: string
  ): Promise<Result<{
    frameCount: number;
    timestepPs: number;
    totalTimeNs: number;
  }, AppError>> => {
    if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Python not found. Please install miniconda and create fraggen environment.',
      });
    }

    if (!fs.existsSync(topologyPath)) {
      return Err({
        type: 'FILE_NOT_FOUND',
        path: topologyPath,
        message: `Topology file not found: ${topologyPath}`,
      });
    }

    if (!fs.existsSync(trajectoryPath)) {
      return Err({
        type: 'FILE_NOT_FOUND',
        path: trajectoryPath,
        message: `Trajectory file not found: ${trajectoryPath}`,
      });
    }

    const pythonCode = `
import sys
import json

try:
    import MDAnalysis as mda

    topology_path = '${topologyPath.replace(/'/g, "\\'")}'
    trajectory_path = '${trajectoryPath.replace(/'/g, "\\'")}'

    u = mda.Universe(topology_path, trajectory_path)

    frame_count = len(u.trajectory)

    # Get timestep in picoseconds (MDAnalysis stores in ps)
    timestep_ps = 0
    if frame_count > 1:
        u.trajectory[0]
        t0 = u.trajectory.time
        u.trajectory[1]
        t1 = u.trajectory.time
        timestep_ps = t1 - t0
    elif hasattr(u.trajectory, 'dt'):
        timestep_ps = u.trajectory.dt
    else:
        # Default to 10 ps if cannot determine
        timestep_ps = 10.0

    total_time_ns = (frame_count * timestep_ps) / 1000.0

    result = {
        "frameCount": frame_count,
        "timestepPs": timestep_ps,
        "totalTimeNs": total_time_ns
    }

    print(json.dumps(result))
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
`;

    return new Promise((resolve) => {
      const python = spawn(appState.condaPythonPath!, ['-c', pythonCode]);
      childProcesses.add(python);

      let stdout = '';
      let stderr = '';

      python.stdout?.on('data', (data: Buffer) => {
        stdout += data.toString();
      });

      python.stderr?.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      python.on('close', (code: number | null) => {
        childProcesses.delete(python);
        if (code === 0 && stdout.trim()) {
          try {
            const result = JSON.parse(stdout.trim());
            if (result.error) {
              resolve(Err({
                type: 'TRAJECTORY_READ_FAILED',
                message: result.error,
              }));
            } else {
              resolve(Ok({
                frameCount: result.frameCount,
                timestepPs: result.timestepPs,
                totalTimeNs: result.totalTimeNs,
              }));
            }
          } catch (e) {
            resolve(Err({
              type: 'PARSE_FAILED',
              message: `Failed to parse trajectory info: ${stdout}`,
            }));
          }
        } else {
          resolve(Err({
            type: 'TRAJECTORY_READ_FAILED',
            message: stderr || 'Failed to read trajectory info',
          }));
        }
      });

      python.on('error', (error: Error) => {
        childProcesses.delete(python);
        resolve(Err({
          type: 'TRAJECTORY_READ_FAILED',
          message: error.message,
        }));
      });
    });
  }
);

// Get a specific trajectory frame as PDB string
ipcMain.handle(
  IpcChannels.GET_TRAJECTORY_FRAME,
  async (
    _event,
    topologyPath: string,
    trajectoryPath: string,
    frameIndex: number
  ): Promise<Result<{ pdbString: string }, AppError>> => {
    if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Python not found.',
      });
    }

    const pythonCode = `
import sys
import json
import warnings
import tempfile
import os
warnings.filterwarnings('ignore')

try:
    import MDAnalysis as mda

    topology_path = '${topologyPath.replace(/'/g, "\\'")}'
    trajectory_path = '${trajectoryPath.replace(/'/g, "\\'")}'
    frame_index = ${frameIndex}

    u = mda.Universe(topology_path, trajectory_path)

    # Apply PBC transformations to center system in view
    try:
        from MDAnalysis import transformations as trans

        protein = u.select_atoms('protein')
        # Try to get ligand for centering together
        lig_sele = 'not protein and not resname WAT HOH TIP3 TIP4 NA CL SOL K MG and not element H'
        ligand = u.select_atoms(lig_sele)
        if len(ligand) == 0:
            ligand = u.select_atoms('(resname LIG UNL UNK MOL) and not element H')

        if len(protein) > 0:
            if len(ligand) > 0:
                complex_group = protein + ligand
            else:
                complex_group = protein

            workflow = [
                trans.unwrap(complex_group),
                trans.center_in_box(protein, center='mass'),
                trans.wrap(complex_group, compound='fragments'),
            ]
            u.trajectory.add_transformations(*workflow)
        elif len(ligand) > 0:
            # Ligand-only system: center on ligand
            workflow = [
                trans.unwrap(ligand),
                trans.center_in_box(ligand, center='mass'),
                trans.wrap(ligand, compound='fragments'),
            ]
            u.trajectory.add_transformations(*workflow)
    except Exception:
        pass  # If PBC transforms fail, continue without them

    u.trajectory[frame_index]

    # Write frame to temporary PDB file then read it back
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        temp_path = f.name

    u.atoms.write(temp_path)

    with open(temp_path, 'r') as f:
        pdb_string = f.read()

    os.unlink(temp_path)

    # Output as base64 to avoid JSON issues with special characters
    import base64
    encoded = base64.b64encode(pdb_string.encode()).decode()
    print(json.dumps({"pdbBase64": encoded}))
except Exception as e:
    import traceback
    print(json.dumps({"error": str(e) + "\\n" + traceback.format_exc()}))
    sys.exit(1)
`;

    return new Promise((resolve) => {
      const python = spawn(appState.condaPythonPath!, ['-c', pythonCode]);
      childProcesses.add(python);

      let stdout = '';
      let stderr = '';

      python.stdout?.on('data', (data: Buffer) => {
        stdout += data.toString();
      });

      python.stderr?.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      python.on('close', (code: number | null) => {
        childProcesses.delete(python);
        if (code === 0 && stdout.trim()) {
          try {
            const result = JSON.parse(stdout.trim());
            if (result.error) {
              resolve(Err({
                type: 'TRAJECTORY_READ_FAILED',
                message: result.error,
              }));
            } else {
              // Decode base64 PDB string
              const pdbString = Buffer.from(result.pdbBase64, 'base64').toString('utf-8');
              resolve(Ok({ pdbString }));
            }
          } catch (e) {
            resolve(Err({
              type: 'PARSE_FAILED',
              message: `Failed to parse frame data: ${stdout.substring(0, 200)}`,
            }));
          }
        } else {
          resolve(Err({
            type: 'TRAJECTORY_READ_FAILED',
            message: stderr || 'Failed to read trajectory frame',
          }));
        }
      });
    });
  }
);

// Get trajectory frame coordinates only (no PDB parsing — for atomStore updates)
ipcMain.handle(
  IpcChannels.GET_TRAJECTORY_COORDS,
  async (
    _event,
    topologyPath: string,
    trajectoryPath: string,
    frameIndex: number
  ): Promise<Result<{ coordsBase64: string; atomCount: number }, AppError>> => {
    if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Python not found.',
      });
    }

    const pythonCode = `
import sys
import json
import warnings
import base64
import numpy as np
warnings.filterwarnings('ignore')

try:
    import MDAnalysis as mda

    topology_path = '${topologyPath.replace(/'/g, "\\'")}'
    trajectory_path = '${trajectoryPath.replace(/'/g, "\\'")}'
    frame_index = ${frameIndex}

    u = mda.Universe(topology_path, trajectory_path)

    # Apply PBC transformations (same as GET_TRAJECTORY_FRAME)
    try:
        from MDAnalysis import transformations as trans

        protein = u.select_atoms('protein')
        lig_sele = 'not protein and not resname WAT HOH TIP3 TIP4 NA CL SOL K MG and not element H'
        ligand = u.select_atoms(lig_sele)
        if len(ligand) == 0:
            ligand = u.select_atoms('(resname LIG UNL UNK MOL) and not element H')

        if len(protein) > 0:
            if len(ligand) > 0:
                complex_group = protein + ligand
            else:
                complex_group = protein

            workflow = [
                trans.unwrap(complex_group),
                trans.center_in_box(protein, center='mass'),
                trans.wrap(complex_group, compound='fragments'),
            ]
            u.trajectory.add_transformations(*workflow)
        elif len(ligand) > 0:
            workflow = [
                trans.unwrap(ligand),
                trans.center_in_box(ligand, center='mass'),
                trans.wrap(ligand, compound='fragments'),
            ]
            u.trajectory.add_transformations(*workflow)
    except Exception:
        pass

    u.trajectory[frame_index]

    # Flatten positions to [x0, y0, z0, x1, y1, z1, ...] as float32
    coords = u.atoms.positions.astype(np.float32).flatten()
    encoded = base64.b64encode(coords.tobytes()).decode()
    print(json.dumps({"coordsBase64": encoded, "atomCount": len(u.atoms)}))
except Exception as e:
    import traceback
    print(json.dumps({"error": str(e) + "\\n" + traceback.format_exc()}))
    sys.exit(1)
`;

    return new Promise((resolve) => {
      const python = spawn(appState.condaPythonPath!, ['-c', pythonCode]);
      childProcesses.add(python);

      let stdout = '';
      let stderr = '';

      python.stdout?.on('data', (data: Buffer) => {
        stdout += data.toString();
      });

      python.stderr?.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      python.on('close', (code: number | null) => {
        childProcesses.delete(python);
        if (code === 0 && stdout.trim()) {
          try {
            const result = JSON.parse(stdout.trim());
            if (result.error) {
              resolve(Err({
                type: 'TRAJECTORY_READ_FAILED',
                message: result.error,
              }));
            } else {
              resolve(Ok({
                coordsBase64: result.coordsBase64,
                atomCount: result.atomCount,
              }));
            }
          } catch (e) {
            resolve(Err({
              type: 'PARSE_FAILED',
              message: `Failed to parse coords data: ${stdout.substring(0, 200)}`,
            }));
          }
        } else {
          resolve(Err({
            type: 'TRAJECTORY_READ_FAILED',
            message: stderr || 'Failed to read trajectory coordinates',
          }));
        }
      });
    });
  }
);

// Cluster trajectory using MDAnalysis
ipcMain.handle(
  IpcChannels.CLUSTER_TRAJECTORY,
  async (
    event,
    options: {
      topologyPath: string;
      trajectoryPath: string;
      numClusters: number;
      method: 'kmeans' | 'dbscan' | 'hierarchical';
      rmsdSelection: 'ligand' | 'backbone' | 'all';
      stripWaters: boolean;
      outputDir: string;
    }
  ): Promise<Result<ClusteringResult, AppError>> => {
    if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Python not found. Please install miniconda and create fraggen environment.',
      });
    }

    const scriptPath = path.join(appState.fraggenRoot, 'cluster_trajectory.py');
    if (!fs.existsSync(scriptPath)) {
      return Err({
        type: 'SCRIPT_NOT_FOUND',
        path: scriptPath,
        message: `Clustering script not found: ${scriptPath}`,
      });
    }

    // Ensure output directory exists
    if (!fs.existsSync(options.outputDir)) {
      fs.mkdirSync(options.outputDir, { recursive: true });
    }

    return new Promise((resolve) => {
      event.sender.send(IpcChannels.MD_OUTPUT, {
        type: 'stdout',
        data: `=== Trajectory Clustering ===\nMethod: ${options.method}\nClusters: ${options.numClusters}\nRMSD Selection: ${options.rmsdSelection}\n\n`,
      });

      const args = [
        scriptPath,
        '--topology', options.topologyPath,
        '--trajectory', options.trajectoryPath,
        '--n_clusters', String(options.numClusters),
        '--method', options.method,
        '--selection', options.rmsdSelection,
        '--output_dir', options.outputDir,
      ];

      if (options.stripWaters) {
        args.push('--strip_waters');
      }

      const proc = spawn(appState.condaPythonPath!, args);
      childProcesses.add(proc);

      let stderrOutput = '';

      proc.stdout?.on('data', (data: Buffer) => {
        const text = data.toString();
        const match = text.match(/Calculated (\d+)\/(\d+)/);
        if (match) {
          const pct = Math.round(100 * parseInt(match[1], 10) / parseInt(match[2], 10));
          try { event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stdout', data: `PROGRESS:clustering:${pct}\n` }); } catch {}
        }
        try { event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stdout', data: text }); } catch {}
      });

      proc.stderr?.on('data', (data: Buffer) => {
        const text = data.toString();
        stderrOutput += text;
        try { event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stderr', data: text }); } catch {}
      });

      proc.on('close', (code: number | null) => {
        childProcesses.delete(proc);

        const resultFile = path.join(options.outputDir, 'clustering_results.json');
        if (code === 0 && resultFile && fs.existsSync(resultFile)) {
          try {
            const content = fs.readFileSync(resultFile, 'utf-8');
            const result = JSON.parse(content);
            resolve(Ok({
              clusters: result.clusters,
              frameAssignments: result.frameAssignments,
              outputDir: options.outputDir,
              requestedClusters: result.requestedClusters,
              actualClusters: result.actualClusters,
            }));
          } catch (err) {
            resolve(Err({
              type: 'CLUSTERING_FAILED',
              message: `Error reading clustering results: ${err}`,
            }));
          }
        } else {
          // Include stderr in error message for debugging
          const stderrTail = stderrOutput.trim().split('\n').slice(-10).join('\n');
          resolve(Err({
            type: 'CLUSTERING_FAILED',
            message: `Clustering failed (exit code ${code}):\n${stderrTail}`,
          }));
        }
      });

      proc.on('error', (err: Error) => {
        childProcesses.delete(proc);
        resolve(Err({
          type: 'CLUSTERING_FAILED',
          message: `Failed to start clustering: ${err.message}`,
        }));
      });
    });
  }
);

// Scan a cluster directory for centroid PDB files
ipcMain.handle(
  IpcChannels.SCAN_CLUSTER_DIRECTORY,
  async (
    _event,
    directoryPath: string
  ): Promise<Result<{
    clusters: Array<{
      clusterId: number;
      pdbPath: string;
      population: number;
    }>;
    clusteringResultsPath?: string;
  }, AppError>> => {
    if (!fs.existsSync(directoryPath)) {
      return Err({
        type: 'DIRECTORY_NOT_FOUND',
        message: `Directory not found: ${directoryPath}`,
      });
    }

    try {
      // Find cluster centroid PDB files (pattern: cluster_N_centroid.pdb)
      const files = fs.readdirSync(directoryPath);
      const clusterPattern = /^cluster_(\d+)_centroid\.pdb$/;

      const clusterFiles: Array<{ clusterId: number; pdbPath: string }> = [];
      for (const file of files) {
        const match = file.match(clusterPattern);
        if (match) {
          clusterFiles.push({
            clusterId: parseInt(match[1], 10),
            pdbPath: path.join(directoryPath, file),
          });
        }
      }

      if (clusterFiles.length === 0) {
        return Err({
          type: 'NO_CLUSTERS_FOUND',
          message: 'No cluster centroid PDB files found (expected pattern: cluster_N_centroid.pdb)',
        });
      }

      // Sort by cluster ID
      clusterFiles.sort((a, b) => a.clusterId - b.clusterId);

      // Try to read clustering_results.json for population info
      const resultsPath = path.join(directoryPath, 'clustering_results.json');
      let populationMap = new Map<number, number>();
      let clusteringResultsPath: string | undefined;

      if (fs.existsSync(resultsPath)) {
        try {
          const content = fs.readFileSync(resultsPath, 'utf-8');
          const results = JSON.parse(content);
          clusteringResultsPath = resultsPath;

          if (results.clusters && Array.isArray(results.clusters)) {
            for (const cluster of results.clusters) {
              if (typeof cluster.clusterId === 'number' && typeof cluster.population === 'number') {
                populationMap.set(cluster.clusterId, cluster.population);
              }
            }
          }
        } catch {
          // Ignore JSON parsing errors, just use default populations
        }
      }

      // Build result with populations
      const clusters = clusterFiles.map(cf => ({
        clusterId: cf.clusterId,
        pdbPath: cf.pdbPath,
        population: populationMap.get(cf.clusterId) ?? 0,
      }));

      return Ok({ clusters, clusteringResultsPath });
    } catch (err) {
      return Err({
        type: 'SCAN_FAILED',
        message: `Failed to scan cluster directory: ${err}`,
      });
    }
  }
);

// Load and align cluster PDB files
ipcMain.handle(
  IpcChannels.LOAD_ALIGNED_CLUSTERS,
  async (
    _event,
    directoryPath: string,
    clusterIds: number[]
  ): Promise<Result<{
    clusters: Array<{
      clusterId: number;
      pdbPath: string;
      alignedPath: string;
      population: number;
    }>;
  }, AppError>> => {
    if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Python not found. Please install miniconda and create fraggen environment.',
      });
    }

    const scriptPath = path.join(appState.fraggenRoot, 'align_clusters.py');
    if (!fs.existsSync(scriptPath)) {
      return Err({
        type: 'SCRIPT_NOT_FOUND',
        path: scriptPath,
        message: `Alignment script not found: ${scriptPath}`,
      });
    }

    // Get cluster files and populations from directory
    const resultsPath = path.join(directoryPath, 'clustering_results.json');
    let populationMap = new Map<number, number>();

    if (fs.existsSync(resultsPath)) {
      try {
        const content = fs.readFileSync(resultsPath, 'utf-8');
        const results = JSON.parse(content);
        if (results.clusters && Array.isArray(results.clusters)) {
          for (const cluster of results.clusters) {
            if (typeof cluster.clusterId === 'number' && typeof cluster.population === 'number') {
              populationMap.set(cluster.clusterId, cluster.population);
            }
          }
        }
      } catch {
        // Ignore errors
      }
    }

    // Build list of PDB paths to align
    const pdbPaths: string[] = [];
    for (const clusterId of clusterIds) {
      const pdbPath = path.join(directoryPath, `cluster_${clusterId}_centroid.pdb`);
      if (fs.existsSync(pdbPath)) {
        pdbPaths.push(pdbPath);
      }
    }

    if (pdbPaths.length === 0) {
      return Err({
        type: 'NO_CLUSTERS_FOUND',
        message: 'No cluster PDB files found for the specified cluster IDs',
      });
    }

    // Create output directory for aligned PDBs
    const alignedDir = path.join(directoryPath, 'aligned');

    return new Promise((resolve) => {
      const args = [
        scriptPath,
        '--pdb_files', ...pdbPaths,
        '--output_dir', alignedDir,
      ];

      const proc = spawn(appState.condaPythonPath!, args);
      childProcesses.add(proc);

      let stdout = '';
      let stderr = '';

      proc.stdout?.on('data', (data: Buffer) => {
        stdout += data.toString();
      });

      proc.stderr?.on('data', (data: Buffer) => {
        stderr += data.toString();
        // Log alignment progress
        console.log(data.toString().trim());
      });

      proc.on('close', (code: number | null) => {
        childProcesses.delete(proc);

        if (code === 0) {
          try {
            const result = JSON.parse(stdout);
            // Map the aligned PDB paths back to clusters
            const clusters = result.alignedPdbs.map((item: { originalPath: string; alignedPath: string }) => {
              const filename = path.basename(item.originalPath);
              const match = filename.match(/cluster_(\d+)_centroid\.pdb/);
              const clusterId = match ? parseInt(match[1], 10) : 0;
              return {
                clusterId,
                pdbPath: item.originalPath,
                alignedPath: item.alignedPath,
                population: populationMap.get(clusterId) ?? 0,
              };
            });
            resolve(Ok({ clusters }));
          } catch (err) {
            resolve(Err({
              type: 'ALIGNMENT_FAILED',
              message: `Failed to parse alignment results: ${err}`,
            }));
          }
        } else {
          resolve(Err({
            type: 'ALIGNMENT_FAILED',
            message: stderr || `Alignment failed with exit code ${code}`,
          }));
        }
      });

      proc.on('error', (err: Error) => {
        childProcesses.delete(proc);
        resolve(Err({
          type: 'ALIGNMENT_FAILED',
          message: `Failed to start alignment: ${err.message}`,
        }));
      });
    });
  }
);

// Export a single frame from trajectory as PDB
ipcMain.handle(
  IpcChannels.EXPORT_TRAJECTORY_FRAME,
  async (
    _event,
    options: {
      topologyPath: string;
      trajectoryPath: string;
      frameIndex: number;
      outputPath: string;
      stripWaters?: boolean;
    }
  ): Promise<Result<{ pdbPath: string }, AppError>> => {
    if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Python not found. Please install miniconda and create fraggen environment.',
      });
    }

    const scriptPath = path.join(appState.fraggenRoot, 'export_frame.py');
    if (!fs.existsSync(scriptPath)) {
      return Err({
        type: 'SCRIPT_NOT_FOUND',
        path: scriptPath,
        message: `Export frame script not found: ${scriptPath}`,
      });
    }

    return new Promise((resolve) => {
      const args = [
        scriptPath,
        '--topology', options.topologyPath,
        '--trajectory', options.trajectoryPath,
        '--frame', String(options.frameIndex),
        '--output', options.outputPath,
      ];

      if (options.stripWaters) {
        args.push('--strip_waters');
      }

      const proc = spawn(appState.condaPythonPath!, args);
      childProcesses.add(proc);

      let stderr = '';

      proc.stderr?.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      proc.on('close', (code: number | null) => {
        childProcesses.delete(proc);

        if (code === 0 && fs.existsSync(options.outputPath)) {
          resolve(Ok({ pdbPath: options.outputPath }));
        } else {
          resolve(Err({
            type: 'EXPORT_FAILED',
            message: stderr || `Failed to export frame with exit code ${code}`,
          }));
        }
      });

      proc.on('error', (err: Error) => {
        childProcesses.delete(proc);
        resolve(Err({
          type: 'EXPORT_FAILED',
          message: `Failed to start export: ${err.message}`,
        }));
      });
    });
  }
);

// Analyze trajectory (RMSD, RMSF, H-bonds, contacts)
ipcMain.handle(
  IpcChannels.ANALYZE_TRAJECTORY,
  async (
    event,
    options: {
      topologyPath: string;
      trajectoryPath: string;
      analysisType: 'rmsd' | 'rmsf' | 'hbonds' | 'contacts';
      outputDir: string;
      ligandSelection?: string;
    }
  ): Promise<Result<{
    type: string;
    plotPath: string;
    csvPath?: string;
    data: unknown;
  }, AppError>> => {
    if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Python not found. Please install miniconda and create fraggen environment.',
      });
    }

    const scriptMap: Record<string, string> = {
      rmsd: 'analyze_rmsd.py',
      rmsf: 'analyze_rmsf.py',
      hbonds: 'analyze_hbonds.py',
      contacts: 'analyze_contacts.py',
    };

    const scriptPath = path.join(appState.fraggenRoot, scriptMap[options.analysisType]);
    if (!fs.existsSync(scriptPath)) {
      return Err({
        type: 'SCRIPT_NOT_FOUND',
        path: scriptPath,
        message: `Analysis script not found: ${scriptPath}`,
      });
    }

    // Ensure output directory exists
    if (!fs.existsSync(options.outputDir)) {
      fs.mkdirSync(options.outputDir, { recursive: true });
    }

    return new Promise((resolve) => {
      event.sender.send(IpcChannels.MD_OUTPUT, {
        type: 'stdout',
        data: `=== ${options.analysisType.toUpperCase()} Analysis ===\n`,
      });

      const args = [
        scriptPath,
        '--topology', options.topologyPath,
        '--trajectory', options.trajectoryPath,
        '--output_dir', options.outputDir,
      ];

      if (options.ligandSelection) {
        args.push('--ligand_selection', options.ligandSelection);
      }

      const proc = spawn(appState.condaPythonPath!, args);
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

        const resultFile = path.join(options.outputDir, `${options.analysisType}_results.json`);
        if (code === 0 && resultFile && fs.existsSync(resultFile)) {
          try {
            const content = fs.readFileSync(resultFile, 'utf-8');
            const result = JSON.parse(content);
            resolve(Ok(result));
          } catch (err) {
            resolve(Err({
              type: 'ANALYSIS_FAILED',
              message: `Error reading analysis results: ${err}`,
            }));
          }
        } else {
          resolve(Err({
            type: 'ANALYSIS_FAILED',
            message: `Analysis failed with exit code ${code}`,
          }));
        }
      });

      proc.on('error', (err: Error) => {
        childProcesses.delete(proc);
        resolve(Err({
          type: 'ANALYSIS_FAILED',
          message: `Failed to start analysis: ${err.message}`,
        }));
      });
    });
  }
);

// Generate comprehensive MD analysis report (PDF pipeline)
ipcMain.handle(
  IpcChannels.GENERATE_MD_REPORT,
  async (
    event,
    options: {
      topologyPath: string;
      trajectoryPath: string;
      outputDir: string;
      ligandSelection?: string;
      ligandSdf?: string;
      simInfo?: Record<string, string>;
    }
  ): Promise<Result<{
    reportPath: string;
    analysisDir: string;
    sectionPdfs: string[];
    clusteringResults?: Array<{
      clusterId: number;
      frameCount: number;
      population: number;
      centroidFrame: number;
      centroidPdbPath?: string;
    }>;
  }, AppError>> => {
    if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Python not found. Please install miniconda and create fraggen environment.',
      });
    }

    const scriptPath = path.join(appState.fraggenRoot, 'generate_md_report.py');
    if (!fs.existsSync(scriptPath)) {
      return Err({
        type: 'SCRIPT_NOT_FOUND',
        path: scriptPath,
        message: `Report generation script not found: ${scriptPath}`,
      });
    }

    // Ensure output directory exists
    if (!fs.existsSync(options.outputDir)) {
      fs.mkdirSync(options.outputDir, { recursive: true });
    }

    return new Promise((resolve) => {
      event.sender.send(IpcChannels.MD_OUTPUT, {
        type: 'stdout',
        data: `=== Generating MD Analysis Report ===\nOutput: ${options.outputDir}\n\n`,
      });

      const args = [
        scriptPath,
        '--topology', options.topologyPath,
        '--trajectory', options.trajectoryPath,
        '--output_dir', options.outputDir,
      ];

      if (options.ligandSelection) {
        args.push('--ligand_selection', options.ligandSelection);
      }

      if (options.ligandSdf) {
        args.push('--ligand_sdf', options.ligandSdf);
      }

      if (options.simInfo) {
        args.push('--sim_info', JSON.stringify(options.simInfo));
      }

      const proc = spawn(appState.condaPythonPath!, args);
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

        const reportPath = path.join(options.outputDir, 'full_report.pdf');
        if (code === 0 && fs.existsSync(reportPath)) {
          // Collect section PDFs
          const sectionPdfs: string[] = [];
          const subdirs = ['contacts', 'rmsd', 'rmsf', 'sse', 'hbonds', 'ligand_props', 'torsions'];
          for (const subdir of subdirs) {
            const dirPath = path.join(options.outputDir, subdir);
            if (fs.existsSync(dirPath)) {
              const files = fs.readdirSync(dirPath).filter((f: string) => f.endsWith('.pdf'));
              for (const f of files) {
                sectionPdfs.push(path.join(dirPath, f));
              }
            }
          }

          // Read clustering results if available
          let clusteringResults;
          const clusteringFile = path.join(options.outputDir, 'clustering', 'clustering_results.json');
          if (fs.existsSync(clusteringFile)) {
            try {
              const clusterData = JSON.parse(fs.readFileSync(clusteringFile, 'utf-8'));
              clusteringResults = clusterData.clusters;
            } catch {
              // ignore parse errors
            }
          }

          resolve(Ok({
            reportPath,
            analysisDir: options.outputDir,
            sectionPdfs,
            clusteringResults,
          }));
        } else {
          resolve(Err({
            type: 'REPORT_FAILED',
            message: `Report generation failed with exit code ${code}`,
          }));
        }
      });

      proc.on('error', (err: Error) => {
        childProcesses.delete(proc);
        resolve(Err({
          type: 'REPORT_FAILED',
          message: `Failed to start report generation: ${err.message}`,
        }));
      });
    });
  }
);

ipcMain.handle(
  IpcChannels.LOAD_MD_TORSION_ANALYSIS,
  async (
    _event,
    options: { analysisDir: string }
  ): Promise<Result<MdTorsionAnalysis | null, AppError>> => {
    try {
      const analysisDir = normalizeAnalysisDir(options.analysisDir);
      const torsionPath = path.join(analysisDir, 'torsions', 'torsions_results.json');
      if (!fs.existsSync(torsionPath)) {
        return Ok(null);
      }
      const content = JSON.parse(fs.readFileSync(torsionPath, 'utf-8'));
      return Ok(normalizeMdTorsionAnalysis(content));
    } catch (err) {
      return Err({
        type: 'ANALYSIS_FAILED',
        message: `Failed to load MD torsion analysis: ${(err as Error).message}`,
      });
    }
  }
);

// Score holo MD cluster centroids with shared Vina score_only + CORDIAL rescoring
ipcMain.handle(
  IpcChannels.SCORE_MD_CLUSTERS,
  async (
    event,
    options: {
      topologyPath: string;
      trajectoryPath: string;
      outputDir: string;
      inputLigandSdf: string;
      inputReceptorPdb?: string;
      numClusters: number;
      enableVina: boolean;
      enableCordial: boolean;
    }
  ): Promise<Result<{
    clusters: ScoredClusterResult[];
    outputDir: string;
    clusteringResults: ClusteringResult;
  }, AppError>> => {
    if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Python not found. Please install miniconda and create the openmm-metal environment.',
      });
    }

    const analysisDir = options.outputDir;
    const clusteringDir = path.join(analysisDir, 'clustering');
    const scoredClustersDir = path.join(analysisDir, 'scored_clusters');
    fs.mkdirSync(clusteringDir, { recursive: true });
    fs.mkdirSync(scoredClustersDir, { recursive: true });

    // --- Step 1: Create or reuse canonical clustering output ---
    let clusteringResults = readClusteringResult(clusteringDir);
    if (!clusteringResults || clusteringResults.clusters.length === 0) {
      const clusterScript = path.join(appState.fraggenRoot, 'cluster_trajectory.py');
      if (!fs.existsSync(clusterScript)) {
        return Err({
          type: 'SCRIPT_NOT_FOUND',
          path: clusterScript,
          message: `Clustering script not found: ${clusterScript}`,
        });
      }

      event.sender.send(IpcChannels.MD_OUTPUT, {
        type: 'stdout',
        data: `=== Clustering into ${options.numClusters} centroids ===\n`,
      });

      const clusterResult = await new Promise<Result<void, AppError>>((resolve) => {
        const args = [
          clusterScript,
          '--topology', options.topologyPath,
          '--trajectory', options.trajectoryPath,
          '--n_clusters', String(options.numClusters),
          '--method', 'kmeans',
          '--selection', 'ligand',
          '--strip_waters',
          '--output_dir', clusteringDir,
        ];

        const proc = spawn(appState.condaPythonPath!, args, { env: getSpawnEnv() });
        childProcesses.add(proc);

        proc.stdout?.on('data', (data: Buffer) => {
          const text = data.toString();
          event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stdout', data: text });
          const match = text.match(/Calculated (\d+)\/(\d+)/);
          if (match) {
            const pct = Math.round(100 * parseInt(match[1], 10) / parseInt(match[2], 10));
            event.sender.send(IpcChannels.MD_OUTPUT, {
              type: 'stdout', data: `PROGRESS:clustering:${pct}\n`,
            });
          }
        });

        proc.stderr?.on('data', (data: Buffer) => {
          event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stderr', data: data.toString() });
        });

        proc.on('close', (code: number | null) => {
          childProcesses.delete(proc);
          if (code === 0) {
            resolve(Ok(undefined));
          } else {
            resolve(Err({ type: 'CLUSTERING_FAILED', message: `Clustering failed with exit code ${code}` }));
          }
        });

        proc.on('error', (err: Error) => {
          childProcesses.delete(proc);
          resolve(Err({ type: 'CLUSTERING_FAILED', message: err.message }));
        });
      });

      if (!clusterResult.ok) {
        return Err(clusterResult.error);
      }

      clusteringResults = readClusteringResult(clusteringDir);
    } else {
      event.sender.send(IpcChannels.MD_OUTPUT, {
        type: 'stdout',
        data: `=== Reusing existing clustering (${clusteringResults.clusters.length} centroids) ===\nPROGRESS:clustering:100\n`,
      });
    }

    if (!clusteringResults || clusteringResults.clusters.length === 0) {
      return Err({
        type: 'CLUSTERING_FAILED',
        message: 'Canonical clustering results were not found after clustering completed',
      });
    }

    // --- Step 2: Prepare centroid receptor/ligand pairs ---
    const scoreScript = path.join(appState.fraggenRoot, 'score_cluster_centroids.py');
    if (!fs.existsSync(scoreScript)) {
      return Err({
        type: 'SCRIPT_NOT_FOUND',
        path: scoreScript,
        message: `Cluster preparation script not found: ${scoreScript}`,
      });
    }

    event.sender.send(IpcChannels.MD_OUTPUT, {
      type: 'stdout',
      data: '=== Preparing cluster centroids for rescoring ===\n',
    });

    const prepareResult = await new Promise<Result<void, AppError>>((resolve) => {
      const args = [
        scoreScript,
        '--clustering_dir', clusteringDir,
        '--input_ligand_sdf', options.inputLigandSdf,
        '--output_dir', scoredClustersDir,
      ];

      const proc = spawn(appState.condaPythonPath!, args, { env: getSpawnEnv() });
      childProcesses.add(proc);

      proc.stdout?.on('data', (data: Buffer) => {
        event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stdout', data: data.toString() });
      });

      proc.stderr?.on('data', (data: Buffer) => {
        event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stderr', data: data.toString() });
      });

      proc.on('close', (code: number | null) => {
        childProcesses.delete(proc);
        if (code === 0) {
          resolve(Ok(undefined));
        } else {
          resolve(Err({
            type: 'CLUSTER_SCORING_FAILED',
            message: `Cluster preparation failed with exit code ${code}`,
          }));
        }
      });

      proc.on('error', (err: Error) => {
        childProcesses.delete(proc);
        resolve(Err({ type: 'CLUSTER_SCORING_FAILED', message: err.message }));
      });
    });

    if (!prepareResult.ok) {
      return Err(prepareResult.error);
    }

    let scoredClusters = readClusterScoreRows(scoredClustersDir);
    if (scoredClusters.length !== clusteringResults.clusters.length) {
      return Err({
        type: 'CLUSTER_SCORING_FAILED',
        message: `Expected ${clusteringResults.clusters.length} prepared clusters, found ${scoredClusters.length}`,
      });
    }

    for (const cluster of scoredClusters) {
      if (!cluster.receptorPdbPath || !cluster.ligandSdfPath) {
        return Err({
          type: 'CLUSTER_SCORING_FAILED',
          message: `Cluster ${cluster.clusterId + 1} is missing prepared receptor/ligand files`,
        });
      }
    }

    // --- Step 3: Vina fixed-pose rescoring ---
    if (options.enableVina) {
      event.sender.send(IpcChannels.MD_OUTPUT, {
        type: 'stdout',
        data: '=== Vina rescoring cluster centroids ===\n',
      });

      for (let i = 0; i < scoredClusters.length; i++) {
        const cluster = scoredClusters[i];
        const vinaResult = await runVinaScoreOnly(
          cluster.receptorPdbPath!,
          cluster.ligandSdfPath!,
          cluster.ligandSdfPath!,
          {
            autoboxAdd: 4,
            cpu: 1,
            onStdout: (text) => {
              event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stdout', data: text });
            },
            onStderr: (text) => {
              event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stderr', data: text });
            },
          },
        );
        if (!vinaResult.ok) {
          return Err({
            type: 'CLUSTER_SCORING_FAILED',
            message: `Vina rescoring failed for cluster ${cluster.clusterId + 1}: ${vinaResult.error.message}`,
          });
        }
        cluster.vinaRescore = Math.round(vinaResult.value * 100) / 100;
        const pct = Math.round(100 * (i + 1) / scoredClusters.length);
        event.sender.send(IpcChannels.MD_OUTPUT, {
          type: 'stdout',
          data: `PROGRESS:scoring_vina:${pct}\n`,
        });
      }
      writeClusterScoreRows(scoredClustersDir, scoredClusters);
    } else {
      return Err({
        type: 'CLUSTER_SCORING_FAILED',
        message: 'Holo MD rescoring requires Vina, but Vina rescoring is disabled',
      });
    }

    // --- Step 3.5: xTB relative energy scoring ---
    const xtbPath = getXtbPath();
    if (xtbPath) {
      event.sender.send(IpcChannels.MD_OUTPUT, {
        type: 'stdout',
        data: '=== xTB energy scoring cluster centroids ===\n',
      });

      const xtbScript = path.join(appState.fraggenRoot, 'score_xtb_strain.py');
      if (fs.existsSync(xtbScript)) {
        // Collect ligand SDF paths from prepared clusters
        const ligandSdfs = scoredClusters
          .filter((c) => c.ligandSdfPath && fs.existsSync(c.ligandSdfPath!))
          .map((c) => c.ligandSdfPath!);

        if (ligandSdfs.length > 0) {
          // Write temp ligand list for batch processing
          const xtbLigandDir = path.join(scoredClustersDir, '_xtb_ligands');
          fs.mkdirSync(xtbLigandDir, { recursive: true });
          for (const sdf of ligandSdfs) {
            const dest = path.join(xtbLigandDir, path.basename(sdf));
            if (!fs.existsSync(dest)) fs.copyFileSync(sdf, dest);
          }

          const xtbOutputJson = path.join(scoredClustersDir, 'xtb_energy.json');
          try {
            const { stdout, code } = await spawnPythonScript([
              xtbScript,
              '--xtb_binary', xtbPath,
              '--mode', 'batch_energy',
              '--ligand_dir', xtbLigandDir,
              '--output_json', xtbOutputJson,
            ]);
            if (code === 0 && fs.existsSync(xtbOutputJson)) {
              const xtbData = JSON.parse(fs.readFileSync(xtbOutputJson, 'utf-8'));
              for (const cluster of scoredClusters) {
                if (!cluster.ligandSdfPath) continue;
                const baseName = path.basename(cluster.ligandSdfPath).replace(/\.sdf(\.gz)?$/, '');
                const key = `${baseName}_0`;
                if (key in xtbData) {
                  cluster.xtbStrainKcal = xtbData[key];
                }
              }
              writeClusterScoreRows(scoredClustersDir, scoredClusters);
              event.sender.send(IpcChannels.MD_OUTPUT, {
                type: 'stdout',
                data: `xTB energy scoring complete: ${Object.keys(xtbData).length} centroids\nPROGRESS:scoring_xtb:100\n`,
              });
            }
          } catch (e) {
            event.sender.send(IpcChannels.MD_OUTPUT, {
              type: 'stderr',
              data: `xTB scoring warning: ${(e as Error).message}\n`,
            });
          }
          // Clean up temp dir
          try { fs.rmSync(xtbLigandDir, { recursive: true }); } catch { /* */ }
        }
      }
    }

    // --- Step 4: CORDIAL rescoring ---
    if (options.enableCordial) {
      event.sender.send(IpcChannels.MD_OUTPUT, {
        type: 'stdout',
        data: '=== CORDIAL rescoring cluster centroids ===\n',
      });

      const pairCsvPath = path.join(scoredClustersDir, 'pairs.csv');
      const csvCell = (value: string) => `"${value.replace(/"/g, '""')}"`;
      const pairCsvRows = [
        'source_name,ligand_sdf,receptor_pdb,pose_index',
        ...scoredClusters.map((cluster) => [
          csvCell(`cluster_${cluster.clusterId}`),
          csvCell(cluster.ligandSdfPath!),
          csvCell(cluster.receptorPdbPath!),
          csvCell('0'),
        ].join(',')),
      ];
      fs.writeFileSync(pairCsvPath, `${pairCsvRows.join('\n')}\n`);

      const cordialResult = await runCordialScoringJob(
        { pairCsv: pairCsvPath },
        path.join(scoredClustersDir, 'cordial_scores.csv'),
        32,
        {
          onStdout: (text) => {
            event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stdout', data: text });
          },
          onStderr: (text) => {
            event.sender.send(IpcChannels.MD_OUTPUT, { type: 'stderr', data: text });
          },
        },
      );
      if (!cordialResult.ok) {
        return Err({
          type: 'CORDIAL_FAILED',
          message: cordialResult.error.message,
        });
      }

      event.sender.send(IpcChannels.MD_OUTPUT, {
        type: 'stdout',
        data: 'PROGRESS:scoring_cordial:100\n',
      });
    } else {
      return Err({
        type: 'CORDIAL_FAILED',
        message: 'Holo MD rescoring requires CORDIAL, but CORDIAL rescoring is disabled',
      });
    }

    // --- Step 5: Read and validate all results ---
    const cordialJsonPath = path.join(scoredClustersDir, 'cordial_scores.json');
    if (!fs.existsSync(cordialJsonPath)) {
      return Err({
        type: 'CORDIAL_FAILED',
        message: `CORDIAL output JSON not found: ${cordialJsonPath}`,
      });
    }

    try {
      const cordialData = JSON.parse(fs.readFileSync(cordialJsonPath, 'utf-8'));
      const cordialByName = new Map<string, {
        expectedPkd: number;
        pHighAffinity: number;
        pVeryHighAffinity: number;
      }>();

      for (const entry of cordialData) {
        const name = entry.source_name;
        cordialByName.set(name, {
          expectedPkd: entry.cordial_expected_pkd,
          pHighAffinity: entry.cordial_p_high_affinity,
          pVeryHighAffinity: entry.cordial_p_very_high ?? entry.cordial_p_very_high_affinity ?? 0,
        });
      }

      for (const cluster of scoredClusters) {
        const cordialKey = `cluster_${cluster.clusterId}`;
        const scores = cordialByName.get(cordialKey);
        if (!scores) {
          return Err({
            type: 'CORDIAL_FAILED',
            message: `Missing CORDIAL scores for cluster ${cluster.clusterId + 1}`,
          });
        }
        cluster.cordialExpectedPkd = scores.expectedPkd;
        cluster.cordialPHighAffinity = scores.pHighAffinity;
        cluster.cordialPVeryHighAffinity = scores.pVeryHighAffinity;
      }
    } catch (err) {
      return Err({
        type: 'CORDIAL_FAILED',
        message: `Failed to parse CORDIAL output: ${err}`,
      });
    }

    for (const cluster of scoredClusters) {
      if (
        !cluster.receptorPdbPath ||
        !cluster.ligandSdfPath ||
        cluster.vinaRescore == null ||
        cluster.cordialExpectedPkd == null ||
        cluster.cordialPHighAffinity == null ||
        cluster.cordialPVeryHighAffinity == null
      ) {
        return Err({
          type: 'CLUSTER_SCORING_FAILED',
          message: `Cluster ${cluster.clusterId + 1} is missing required rescoring fields`,
        });
      }
    }

    writeClusterScoreRows(scoredClustersDir, scoredClusters);

    event.sender.send(IpcChannels.MD_OUTPUT, {
      type: 'stdout',
      data: `PROGRESS:scoring:100\n=== Cluster scoring complete ===\n`,
    });

    return Ok({
      clusters: mergeClusterScoresWithCanonical(clusteringResults, scoredClusters),
      outputDir: analysisDir,
      clusteringResults,
    });
  }
);

// Map binding site interaction potentials around a ligand
ipcMain.handle(
  IpcChannels.MAP_BINDING_SITE,
  async (
    event,
    options: {
      pdbPath: string;
      ligandResname: string;
      ligandResnum: number;
      outputDir: string;
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
  }, AppError>> => {
    if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Python not found. Please install miniconda and create the openmm-metal environment.',
      });
    }

    const scriptPath = path.join(appState.fraggenRoot, 'map_binding_site.py');
    if (!fs.existsSync(scriptPath)) {
      return Err({
        type: 'SCRIPT_NOT_FOUND',
        path: scriptPath,
        message: `Binding site mapping script not found: ${scriptPath}`,
      });
    }

    if (!fs.existsSync(options.outputDir)) {
      fs.mkdirSync(options.outputDir, { recursive: true });
    }

    return new Promise((resolve) => {
      event.sender.send(IpcChannels.MD_OUTPUT, {
        type: 'stdout',
        data: '=== Binding Site Interaction Map ===\n',
      });

      // Derive project name from output path: .../surfaces/binding_site_map
      const bsmProjectName = path.basename(path.resolve(options.outputDir, '../..'));

      const args = [
        scriptPath,
        '--pdb_path', options.pdbPath,
        '--ligand_resname', options.ligandResname,
        '--ligand_resnum', String(options.ligandResnum),
        '--output_dir', options.outputDir,
        '--project_name', bsmProjectName,
      ];

      if (options.boxPadding !== undefined) {
        args.push('--box_padding', String(options.boxPadding));
      }
      if (options.gridSpacing !== undefined) {
        args.push('--grid_spacing', String(options.gridSpacing));
      }

      const proc = spawn(appState.condaPythonPath!, args);
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

        // Try prefixed results first, fall back to unprefixed
        const resultFile = getBindingSiteResultFile(options.outputDir, bsmProjectName);
        if (code === 0 && resultFile && fs.existsSync(resultFile)) {
          try {
            const content = fs.readFileSync(resultFile, 'utf-8');
            const result = JSON.parse(content);
            writeMapMetadata(options.outputDir, {
              method: 'static',
              sourcePdbPath: options.sourcePdbPath || options.pdbPath,
              sourceTrajectoryPath: options.sourceTrajectoryPath,
              ligandResname: options.ligandResname,
              ligandResnum: options.ligandResnum,
              computedAt: new Date().toISOString(),
            });
            resolve(Ok(result));
          } catch (err) {
            resolve(Err({
              type: 'BINDING_SITE_MAP_FAILED',
              message: `Error reading binding site results: ${err}`,
            }));
          }
        } else {
          resolve(Err({
            type: 'BINDING_SITE_MAP_FAILED',
            message: `Binding site mapping failed with exit code ${code}`,
          }));
        }
      });

      proc.on('error', (err: Error) => {
        childProcesses.delete(proc);
        resolve(Err({
          type: 'BINDING_SITE_MAP_FAILED',
          message: `Failed to start binding site mapping: ${err.message}`,
        }));
      });
    });
  }
);

// Compute surface properties (hydrophobic, electrostatic) for a PDB
ipcMain.handle(
  IpcChannels.COMPUTE_SURFACE_PROPS,
  async (
    _event,
    pdbPath: string,
    outputDir: string
  ): Promise<Result<{
    atomCount: number;
    hydrophobic: number[];
    electrostatic: number[];
    cachedPath: string;
  }, AppError>> => {
    if (!appState.condaPythonPath || !fs.existsSync(appState.condaPythonPath)) {
      return Err({
        type: 'PYTHON_NOT_FOUND',
        message: 'Python not found.',
      });
    }

    const scriptPath = path.join(appState.fraggenRoot, 'compute_surface_props.py');
    if (!fs.existsSync(scriptPath)) {
      return Err({
        type: 'SCRIPT_NOT_FOUND',
        path: scriptPath,
        message: `Surface properties script not found: ${scriptPath}`,
      });
    }

    const sourceHash = crypto
      .createHash('sha1')
      .update(fs.readFileSync(pdbPath))
      .digest('hex')
      .slice(0, 12);

    // Cache path is keyed to the actual source file contents so surfaces from
    // different structures cannot silently overwrite each other inside one
    // project-level surfaces/ directory.
    fs.mkdirSync(outputDir, { recursive: true });
    const cachePath = path.join(outputDir, `surface_properties_${sourceHash}.json`);

    // Return cached if exists
    if (fs.existsSync(cachePath)) {
      try {
        const cached = JSON.parse(fs.readFileSync(cachePath, 'utf-8'));
        return Ok({ ...cached, cachedPath: cachePath });
      } catch {
        // Corrupted cache, recompute
      }
    }

    return new Promise((resolve) => {
      const proc = spawn(appState.condaPythonPath!, [
        scriptPath,
        '--pdb_path', pdbPath,
        '--output_path', cachePath,
      ]);
      childProcesses.add(proc);

      let stderr = '';

      proc.stderr?.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      proc.on('close', (code: number | null) => {
        childProcesses.delete(proc);
        if (code === 0 && fs.existsSync(cachePath)) {
          try {
            const result = JSON.parse(fs.readFileSync(cachePath, 'utf-8'));
            resolve(Ok({ ...result, cachedPath: cachePath }));
          } catch (err) {
            resolve(Err({
              type: 'SURFACE_PROPS_FAILED',
              message: `Failed to parse surface properties: ${err}`,
            }));
          }
        } else {
          resolve(Err({
            type: 'SURFACE_PROPS_FAILED',
            message: `Surface property computation failed (exit ${code}): ${stderr.slice(0, 300)}`,
          }));
        }
      });

      proc.on('error', (err: Error) => {
        childProcesses.delete(proc);
        resolve(Err({
          type: 'SURFACE_PROPS_FAILED',
          message: err.message,
        }));
      });
    });
  }
);

// --- Molecule alignment ---

ipcMain.handle(
  IpcChannels.ALIGN_MOLECULES_MCS,
  async (_event, refSdf: string, mobileSdf: string, outPath: string) => {
    try {
      const scriptPath = path.join(appState.fraggenRoot, 'align_molecules.py');
      const { stdout, stderr, code } = await spawnPythonScript(
        [scriptPath, '--mode', 'mcs', '--ref', refSdf, '--mobile', mobileSdf, '--out', outPath],
        { env: getSpawnEnv() }
      );
      if (code !== 0) return Err({ type: 'ALIGNMENT_FAILED', message: stderr.slice(0, 500) });
      const result = JSON.parse(stdout);
      return Ok(result);
    } catch (err) {
      return Err({ type: 'ALIGNMENT_FAILED', message: (err as Error).message });
    }
  }
);

ipcMain.handle(
  IpcChannels.ALIGN_DETECT_SCAFFOLDS,
  async (_event, refSdf: string, mobileSdf: string) => {
    try {
      const scriptPath = path.join(appState.fraggenRoot, 'align_molecules.py');
      const { stdout, stderr, code } = await spawnPythonScript(
        [scriptPath, '--mode', 'scaffolds', '--ref', refSdf, '--mobile', mobileSdf],
        { env: getSpawnEnv() }
      );
      if (code !== 0) return Err({ type: 'ALIGNMENT_FAILED', message: stderr.slice(0, 500) });
      const result = JSON.parse(stdout);
      return Ok(result);
    } catch (err) {
      return Err({ type: 'ALIGNMENT_FAILED', message: (err as Error).message });
    }
  }
);

ipcMain.handle(
  IpcChannels.ALIGN_BY_SCAFFOLD,
  async (_event, refSdf: string, mobileSdf: string, scaffoldIndex: number, outPath: string) => {
    try {
      const scriptPath = path.join(appState.fraggenRoot, 'align_molecules.py');
      const { stdout, stderr, code } = await spawnPythonScript(
        [scriptPath, '--mode', 'align_scaffold', '--ref', refSdf, '--mobile', mobileSdf,
         '--scaffold-index', String(scaffoldIndex), '--out', outPath],
        { env: getSpawnEnv() }
      );
      if (code !== 0) return Err({ type: 'ALIGNMENT_FAILED', message: stderr.slice(0, 500) });
      const result = JSON.parse(stdout);
      return Ok(result);
    } catch (err) {
      return Err({ type: 'ALIGNMENT_FAILED', message: (err as Error).message });
    }
  }
);

} // end register()
