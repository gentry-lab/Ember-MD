/**
 * Project scanning and management handlers.
 * Covers project creation, scanning, renaming, deletion, structure import,
 * and artifact discovery (docking poses, simulation runs, conformers, maps).
 */
import { app, ipcMain } from 'electron';
import * as path from 'path';
import * as fs from 'fs';
import * as zlib from 'zlib';
import { Ok, Err, Result } from '../../shared/types/result';
import { AppError } from '../../shared/types/errors';
import { IpcChannels } from '../../shared/types/ipc';
import type {
  ClusteringResult,
  ProjectJob,
  ProjectJobPose,
  ScoredClusterResult,
} from '../../shared/types/ipc';

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
// Shared helpers (also used by other IPC modules — importable from here)
// ---------------------------------------------------------------------------

export const readJsonIfExists = <T>(jsonPath: string): T | null => {
  try {
    if (!fs.existsSync(jsonPath)) return null;
    return JSON.parse(fs.readFileSync(jsonPath, 'utf-8')) as T;
  } catch {
    return null;
  }
};

export const getCanonicalAnalysisRoot = (runPath: string) =>
  path.join(runPath, 'analysis');

export const getCanonicalClusteringDir = (runPath: string) =>
  path.join(getCanonicalAnalysisRoot(runPath), 'clustering');

export const getCanonicalScoredClustersDir = (runPath: string) =>
  path.join(getCanonicalAnalysisRoot(runPath), 'scored_clusters');

export const getSimulationClusterArtifacts = (runPath: string): {
  clusterDirPath?: string;
  clusteringResultsPath?: string;
  clusterCount: number;
} => {
  const clusterDirCandidates = [
    getCanonicalClusteringDir(runPath),
    path.join(runPath, 'results', 'analysis', 'clustering'),
    getCanonicalScoredClustersDir(runPath),
    path.join(runPath, 'results', 'analysis', 'scored_clusters'),
    path.join(runPath, 'clustering'),
  ];

  for (const candidateDir of clusterDirCandidates) {
    if (!fs.existsSync(candidateDir)) continue;
    const clusterFiles = fs.readdirSync(candidateDir).filter((f: string) => f.match(/cluster_\d+_centroid\.pdb/));
    if (clusterFiles.length === 0) continue;
    const clusteringResultsPath = path.join(candidateDir, 'clustering_results.json');
    return {
      clusterDirPath: candidateDir,
      clusteringResultsPath: fs.existsSync(clusteringResultsPath) ? clusteringResultsPath : undefined,
      clusterCount: clusterFiles.length,
    };
  }

  return { clusterCount: 0 };
};

/** Extract Vina affinity from first model of an .sdf.gz file */
export const extractVinaAffinity = (sdfGzPath: string): number | undefined => {
  try {
    const gzData = fs.readFileSync(sdfGzPath);
    const text = zlib.gunzipSync(gzData).toString('utf-8');
    const dockedMatch = text.match(/>  <minimizedAffinity>\s*\n([\-\d.]+)/);
    if (dockedMatch) return parseFloat(dockedMatch[1]);
    const scoreOnlyMatch = text.match(/>  <vinaScoreOnlyAffinity>\s*\n([\-\d.]+)/);
    if (scoreOnlyMatch) return parseFloat(scoreOnlyMatch[1]);
  } catch { /* ignore */ }
  return undefined;
};

export const getBindingSiteResultFile = (outputDir: string, projectName?: string): string | null => {
  const prefixedPath = projectName ? path.join(outputDir, `${projectName}_binding_site_results.json`) : null;
  if (prefixedPath && fs.existsSync(prefixedPath)) return prefixedPath;

  const legacyPath = path.join(outputDir, 'binding_site_results.json');
  if (fs.existsSync(legacyPath)) return legacyPath;

  return null;
};

export const writeMapMetadata = (
  outputDir: string,
  metadata: MapJobMetadata,
) => {
  fs.mkdirSync(outputDir, { recursive: true });
  const metadataPath = path.join(outputDir, 'map_metadata.json');
  fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));
};

export const findDockingRunJobs = (
  runPath: string,
  runName: string,
  stripPrefix: (name: string) => string,
): ProjectJob[] => {
  const stat = fs.statSync(runPath);
  const runFiles = fs.readdirSync(runPath);
  const newReceptor = path.join(runPath, 'inputs', 'receptor.pdb');
  const legacyReceptorFile = runFiles.find((f) => f.includes('_receptor_prepared') && f.endsWith('.pdb'));
  const receptorPdb = fs.existsSync(newReceptor)
    ? newReceptor
    : legacyReceptorFile
      ? path.join(runPath, legacyReceptorFile)
      : undefined;

  const newPosesDir = path.join(runPath, 'results', 'poses');
  const legacyPosesDir = path.join(runPath, 'poses');
  const posesSearchDir = fs.existsSync(newPosesDir)
    ? newPosesDir
    : fs.existsSync(legacyPosesDir)
      ? legacyPosesDir
      : runPath;
  const poseFiles = fs.existsSync(posesSearchDir)
    ? fs.readdirSync(posesSearchDir).filter((f) => f.endsWith('_docked.sdf.gz'))
    : [];

  const manifest = readJsonIfExists<{
    prepared_reference_ligand_sdf?: string;
  }>(path.join(runPath, 'prep', 'prepared_complex_manifest.json'));
  const referenceLigandCandidates = [
    manifest?.prepared_reference_ligand_sdf,
    path.join(runPath, 'inputs', 'reference_ligand.sdf'),
  ].filter((candidate): candidate is string => typeof candidate === 'string' && fs.existsSync(candidate));

  const preparedLigandCandidates: string[] = [];
  const inputsLigandsDir = path.join(runPath, 'inputs', 'ligands');
  if (fs.existsSync(inputsLigandsDir)) {
    preparedLigandCandidates.push(
      ...fs.readdirSync(inputsLigandsDir)
        .filter((fileName) => /\.(sdf|sdf\.gz|mol|mol2)$/i.test(fileName))
        .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))
        .map((fileName) => path.join(inputsLigandsDir, fileName)),
    );
  }
  const preparedLigandPath = preparedLigandCandidates[0];
  const referenceLigandPath = referenceLigandCandidates[0];

  if (!receptorPdb || poseFiles.length === 0) return [];

  const poses: ProjectJobPose[] = poseFiles.map((f) => {
    const posePath = path.join(posesSearchDir, f);
    const name = f.replace('_docked.sdf.gz', '');
    return {
      name: stripPrefix(name),
      path: posePath,
      affinity: extractVinaAffinity(posePath),
    };
  });

  poses.sort((a, b) => (a.affinity ?? 0) - (b.affinity ?? 0));

  const groupId = `dock:${runName}`;
  const allJob: ProjectJob = {
    id: groupId,
    type: 'docking',
    folder: runName,
    label: `${runName} (${poses.length} poses)`,
    path: runPath,
    lastModified: stat.mtimeMs,
    parentId: groupId,
    parentLabel: runName,
    sortKey: 0,
    receptorPdb,
    poses,
    preparedLigandPath,
    referenceLigandPath,
  };

  const poseJobs: ProjectJob[] = poses.map((pose, index) => ({
    id: `${groupId}:pose:${index}`,
    type: 'docking-pose',
    folder: runName,
    label: `${pose.name}${pose.affinity != null ? ` (${pose.affinity.toFixed(1)})` : ''}`,
    path: runPath,
    lastModified: stat.mtimeMs,
    parentId: groupId,
    parentLabel: runName,
    sortKey: index + 1,
    receptorPdb,
    poses,
    ligandPath: pose.path,
    poseIndex: index,
    preparedLigandPath,
    referenceLigandPath,
  }));

  return [allJob, ...poseJobs];
};

export const findSimulationJob = (runPath: string, runName: string): ProjectJob | null => {
  const stat = fs.statSync(runPath);
  const runFiles = fs.readdirSync(runPath);
  const resultsDir = path.join(runPath, 'results');
  const resultsFiles = fs.existsSync(resultsDir) ? fs.readdirSync(resultsDir) : [];

  let finalPdbPath: string | undefined;
  if (resultsFiles.includes('final.pdb')) {
    finalPdbPath = path.join(resultsDir, 'final.pdb');
  } else if (runFiles.includes('final.pdb')) {
    finalPdbPath = path.join(runPath, 'final.pdb');
  } else {
    const legacyFinal = runFiles.find((f) => f.endsWith('_final.pdb'));
    if (legacyFinal) finalPdbPath = path.join(runPath, legacyFinal);
  }

  let systemPdbPath: string | undefined;
  let trajectoryDcdPath: string | undefined;

  if (resultsFiles.includes('system.pdb')) systemPdbPath = path.join(resultsDir, 'system.pdb');
  if (resultsFiles.includes('trajectory.dcd')) trajectoryDcdPath = path.join(resultsDir, 'trajectory.dcd');

  // Top-level unprefixed files (current MD runner writes directly to run root)
  if (!systemPdbPath && runFiles.includes('system.pdb')) {
    systemPdbPath = path.join(runPath, 'system.pdb');
  }
  if (!trajectoryDcdPath && runFiles.includes('trajectory.dcd')) {
    trajectoryDcdPath = path.join(runPath, 'trajectory.dcd');
  }

  if (!systemPdbPath) {
    const legacySys = runFiles.find((f) => f.endsWith('_system.pdb'));
    if (legacySys) systemPdbPath = path.join(runPath, legacySys);
  }
  if (!trajectoryDcdPath) {
    const legacyTraj = runFiles.find((f) => f.endsWith('_trajectory.dcd'));
    if (legacyTraj) trajectoryDcdPath = path.join(runPath, legacyTraj);
  }

  if (!finalPdbPath && !(systemPdbPath && trajectoryDcdPath)) return null;

  const {
    clusterCount,
    clusterDirPath,
    clusteringResultsPath,
  } = getSimulationClusterArtifacts(runPath);

  const parts = [];
  if (trajectoryDcdPath) parts.push('trajectory');
  if (clusterCount > 0) parts.push(`${clusterCount} clusters`);
  const suffix = parts.length > 0 ? ` (${parts.join(', ')})` : '';

  return {
    id: `sim:${runName}`,
    type: 'simulation',
    folder: runName,
    label: `${runName}${suffix}`,
    path: runPath,
    lastModified: stat.mtimeMs,
    systemPdb: systemPdbPath,
    trajectoryDcd: trajectoryDcdPath,
    finalPdb: finalPdbPath,
    hasTrajectory: !!trajectoryDcdPath,
    clusterCount,
    clusterDir: clusterDirPath,
    clusteringResultsPath,
  };
};

export const findConformerJob = (runPath: string, runName: string): ProjectJob | null => {
  if (!fs.existsSync(runPath)) return null;
  const stat = fs.statSync(runPath);

  const conformerPaths = fs.readdirSync(runPath)
    .filter((f) => /\.(sdf|sdf\.gz|mol|mol2)$/i.test(f))
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))
    .map((f) => path.join(runPath, f));

  if (conformerPaths.length === 0) return null;

  return {
    id: `conformer:${runName}`,
    type: 'conformer',
    folder: runName,
    label: `${runName} (${conformerPaths.length} conformers)`,
    path: runPath,
    lastModified: stat.mtimeMs,
    conformerPaths,
    conformerCount: conformerPaths.length,
  };
};

export const inferPreferredMapSources = (projectDir: string): { pdbPath?: string; trajectoryPath?: string } => {
  const simsDir = path.join(projectDir, 'simulations');
  if (fs.existsSync(simsDir)) {
    const simRuns = fs.readdirSync(simsDir, { withFileTypes: true })
      .filter((entry) => entry.isDirectory() && !entry.name.startsWith('.'))
      .map((entry) => {
        const runPath = path.join(simsDir, entry.name);
        return {
          job: findSimulationJob(runPath, entry.name),
          mtime: fs.statSync(runPath).mtimeMs,
        };
      })
      .filter((entry): entry is { job: ProjectJob; mtime: number } => entry.job !== null)
      .sort((a, b) => b.mtime - a.mtime);

    const latestSimulation = simRuns[0]?.job;
    if (latestSimulation) {
      return {
        pdbPath: latestSimulation.finalPdb || latestSimulation.systemPdb,
        trajectoryPath: latestSimulation.trajectoryDcd,
      };
    }
  }

  const dockingDir = path.join(projectDir, 'docking');
  if (fs.existsSync(dockingDir)) {
    const dockRuns = fs.readdirSync(dockingDir, { withFileTypes: true })
      .filter((entry) => entry.isDirectory() && !entry.name.startsWith('.'))
      .map((entry) => {
        const runPath = path.join(dockingDir, entry.name);
        const jobs = findDockingRunJobs(runPath, entry.name, (name) => name);
        return {
          receptorPdb: jobs.find((job) => job.type === 'docking')?.receptorPdb,
          mtime: fs.statSync(runPath).mtimeMs,
        };
      })
      .filter((entry) => !!entry.receptorPdb)
      .sort((a, b) => b.mtime - a.mtime);

    if (dockRuns[0]?.receptorPdb) {
      return { pdbPath: dockRuns[0].receptorPdb };
    }
  }

  const structuresDir = path.join(projectDir, 'structures');
  if (fs.existsSync(structuresDir)) {
    const structure = fs.readdirSync(structuresDir)
      .filter((fileName) => /\.(pdb|cif)$/i.test(fileName))
      .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))[0];
    if (structure) {
      return { pdbPath: path.join(structuresDir, structure) };
    }
  }

  return {};
};

const findMapJob = (projectDir: string, runPath: string, runName: string): ProjectJob | null => {
  if (!fs.existsSync(runPath)) return null;

  const stat = fs.statSync(runPath);
  const projectName = path.basename(projectDir);
  const metadata = readJsonIfExists<MapJobMetadata>(path.join(runPath, 'map_metadata.json'));
  const resultJson = getBindingSiteResultFile(runPath, projectName);
  const result = resultJson
    ? readJsonIfExists<{
        hydrophobicDx?: string;
        hbondDonorDx?: string;
        hbondAcceptorDx?: string;
        hotspots?: unknown[];
      }>(resultJson)
    : null;

  if (!resultJson || !result?.hydrophobicDx || !result.hbondDonorDx || !result.hbondAcceptorDx) {
    return null;
  }

  const inferredMethod: 'solvation' = 'solvation';

  const inferredSources = inferPreferredMapSources(projectDir);
  const methodLabel = 'Water Map (GIST)';
  const hotspotCount = Array.isArray(result.hotspots) ? result.hotspots.length : 0;

  return {
    id: `map:${runName}`,
    type: 'map',
    folder: runName,
    label: `${methodLabel}${hotspotCount > 0 ? ` (${hotspotCount} hotspots)` : ''}`,
    path: runPath,
    lastModified: stat.mtimeMs,
    mapMethod: inferredMethod,
    mapResultJson: resultJson,
    mapPdb: metadata?.sourcePdbPath || inferredSources.pdbPath,
    mapTrajectoryDcd: metadata?.sourceTrajectoryPath || inferredSources.trajectoryPath,
    hotspotCount,
  };
};

// ---------------------------------------------------------------------------
// IPC handler registration
// ---------------------------------------------------------------------------

export function register(): void {
  // Return the default output directory (~/Ember)
  ipcMain.handle('get-default-output-dir', async () => {
    return path.join(app.getPath('home'), 'Ember');
  });

  // Ensure a project directory exists with a .ember-project ID file
  ipcMain.handle(
    IpcChannels.ENSURE_PROJECT,
    async (_event: any, projectName: string): Promise<Result<string, AppError>> => {
      try {
        const emberDir = path.join(app.getPath('home'), 'Ember');
        const projectDir = path.join(emberDir, projectName);
        const idFile = path.join(projectDir, '.ember-project');

        fs.mkdirSync(projectDir, { recursive: true });

        if (!fs.existsSync(idFile)) {
          fs.writeFileSync(idFile, JSON.stringify({
            name: projectName,
            created: new Date().toISOString(),
          }, null, 2));
        }

        return Ok(projectDir);
      } catch (err: any) {
        return Err({
          type: 'DIRECTORY_ERROR',
          path: path.join(app.getPath('home'), 'Ember', projectName),
          message: `Failed to create project: ${err.message}`,
        });
      }
    }
  );

  // Import a structure file into a project's structures/ directory
  ipcMain.handle(
    IpcChannels.IMPORT_STRUCTURE,
    async (_event, sourcePath: string, projectDir: string): Promise<Result<string, AppError>> => {
      try {
        const structuresDir = path.join(projectDir, 'structures');
        fs.mkdirSync(structuresDir, { recursive: true });
        const filename = path.basename(sourcePath);
        const destPath = path.join(structuresDir, filename);
        // Don't re-copy if already in the project
        if (path.resolve(sourcePath) !== path.resolve(destPath)) {
          fs.copyFileSync(sourcePath, destPath);
        }
        return Ok(destPath);
      } catch (error) {
        return Err({
          type: 'IMPORT_FAILED',
          message: `Failed to import structure: ${(error as Error).message}`,
        });
      }
    }
  );

  // Fetch a structure from RCSB PDB by ID (e.g. "8TCE") → save to project structures/
  ipcMain.handle(
    IpcChannels.FETCH_PDB,
    async (_event, pdbId: string, projectDir: string): Promise<Result<string, AppError>> => {
      try {
        const id = pdbId.trim().toUpperCase();
        if (!/^[A-Z0-9]{4}$/.test(id)) {
          return Err({ type: 'VALIDATION_FAILED', message: `Invalid PDB ID: "${pdbId}". Must be 4 alphanumeric characters.` });
        }

        const structuresDir = path.join(projectDir, 'structures');
        fs.mkdirSync(structuresDir, { recursive: true });
        const destPath = path.join(structuresDir, `${id}.cif`);

        // Skip download if already fetched
        if (fs.existsSync(destPath) && fs.statSync(destPath).size > 0) {
          return Ok(destPath);
        }

        const url = `https://files.rcsb.org/download/${id}.cif`;
        const response = await fetch(url);
        if (!response.ok) {
          return Err({ type: 'DOWNLOAD_FAILED', message: `RCSB returned ${response.status} for PDB ID "${id}". Check the ID and try again.` });
        }
        const text = await response.text();
        fs.writeFileSync(destPath, text);
        console.log(`[Project] Fetched ${id}.cif from RCSB (${(text.length / 1024).toFixed(0)} KB)`);
        return Ok(destPath);
      } catch (err: any) {
        return Err({ type: 'DOWNLOAD_FAILED', message: `Failed to fetch PDB: ${err.message}` });
      }
    }
  );

  // Project browser: scan ~/Ember for projects and runs
  // Primary: .ember-project ID file -> project detected regardless of run state
  // Secondary: scan simulations/ + docking/ for runs
  // Tertiary: legacy fallback for old directories without .ember-project
  ipcMain.handle(IpcChannels.SCAN_PROJECTS, async (): Promise<any[]> => {
    const emberDir = path.join(app.getPath('home'), 'Ember');
    if (!fs.existsSync(emberDir)) return [];

    const projects: any[] = [];
    const legacyPattern = /^(.+?)_(ff14sb-TIP3P|ff19sb-OPC|ff19sb-OPC3|charmm36-mTIP3P)_MD-/;

    try {
      const entries = fs.readdirSync(emberDir, { withFileTypes: true });
      const legacyGroups: Record<string, any[]> = {};

      // Helper: check a directory for simulation or docking output files and return run info
      const scanRunDir = (runPath: string, folderName: string): any | null => {
        try {
          const runFiles = fs.readdirSync(runPath);
          // Check results/ subdir for new layout
          const resultsPath = path.join(runPath, 'results');
          const resultsFiles = fs.existsSync(resultsPath) ? fs.readdirSync(resultsPath) : [];
          // New layout: unprefixed files in results/ or top-level; Legacy: prefixed files
          const hasSimOutput = runFiles.some((f: string) => f.endsWith('_system.pdb') || f.endsWith('_trajectory.dcd') || f === 'simulation.log' || f === 'system.pdb' || f === 'trajectory.dcd')
            || resultsFiles.some((f: string) => f === 'system.pdb' || f === 'trajectory.dcd');
          // New layout: docked files in results/poses/; Legacy: in poses/ or top-level
          const newPosesPath = path.join(runPath, 'results', 'poses');
          const legacyPosesPath = path.join(runPath, 'poses');
          const hasDockOutput = runFiles.some((f: string) => f.endsWith('_docked.sdf.gz') || f.endsWith('_docked.sdf'))
            || (fs.existsSync(newPosesPath) && fs.readdirSync(newPosesPath).some((f: string) => f.endsWith('_docked.sdf.gz')))
            || (fs.existsSync(legacyPosesPath) && fs.readdirSync(legacyPosesPath).some((f: string) => f.endsWith('_docked.sdf.gz')));
          // New layout: inputs/ dir with receptor.pdb indicates a docking run even without results yet
          const hasInputs = fs.existsSync(path.join(runPath, 'inputs', 'receptor.pdb'));
          const hasConformerOutput = runFiles.some((f: string) => /\.(sdf|sdf\.gz|mol|mol2)$/i.test(f));
          const hasMapOutput = runFiles.some((f: string) => f === 'binding_site_results.json' || f.endsWith('_binding_site_results.json'));
          if (!hasSimOutput && !hasDockOutput && !hasInputs && !hasConformerOutput && !hasMapOutput) return null;
          const stat = fs.statSync(runPath);
          const type = hasMapOutput
            ? 'map'
            : hasConformerOutput
              ? 'conformer'
              : hasSimOutput
                ? 'simulation'
                : 'docking';
          return {
            folderName,
            path: runPath,
            lastModified: stat.mtimeMs,
            type,
            hasTrajectory: runFiles.some((f: string) => f.endsWith('_trajectory.dcd') || f === 'trajectory.dcd')
              || resultsFiles.some((f: string) => f === 'trajectory.dcd'),
            hasFinalPdb: runFiles.some((f: string) => f.endsWith('_final.pdb') || f === 'final.pdb')
              || resultsFiles.some((f: string) => f === 'final.pdb'),
          };
        } catch { return null; }
      };

      // Scan runs inside a project directory (simulations/, docking/, or legacy direct children)
      const scanProjectRuns = (entryPath: string): any[] => {
        const runs: any[] = [];
        try {
          const subEntries = fs.readdirSync(entryPath, { withFileTypes: true });
          for (const sub of subEntries) {
            if (!sub.isDirectory() || sub.name.startsWith('.')) continue;

            // New layout: runs live under simulations/ and docking/ subdirectories
            if (sub.name === 'simulations' || sub.name === 'docking' || sub.name === 'conformers') {
              const typeDir = path.join(entryPath, sub.name);
              try {
                const typeEntries = fs.readdirSync(typeDir, { withFileTypes: true });
                for (const runEntry of typeEntries) {
                  if (!runEntry.isDirectory() || runEntry.name.startsWith('.')) continue;
                  const runInfo = scanRunDir(path.join(typeDir, runEntry.name), `${sub.name}/${runEntry.name}`);
                  if (runInfo) runs.push(runInfo);
                }
              } catch { /* skip unreadable */ }
              continue;
            }

            if (sub.name === 'surfaces') {
              const surfaceDir = path.join(entryPath, sub.name);
              try {
                const typeEntries = fs.readdirSync(surfaceDir, { withFileTypes: true });
                for (const runEntry of typeEntries) {
                  if (!runEntry.isDirectory() || runEntry.name.startsWith('.')) continue;
                  if (runEntry.name !== 'binding_site_map' && !runEntry.name.startsWith('pocket_map_')) continue;
                  const runInfo = scanRunDir(path.join(surfaceDir, runEntry.name), `${sub.name}/${runEntry.name}`);
                  if (runInfo) runs.push(runInfo);
                }
              } catch { /* skip unreadable */ }
              continue;
            }

            if (sub.name === 'structures' || sub.name === 'fep' || sub.name === 'raw' || sub.name === 'prepared' || sub.name === 'ligands') {
              continue;
            }

            // Legacy layout: runs directly under project dir
            const runInfo = scanRunDir(path.join(entryPath, sub.name), sub.name);
            if (runInfo) runs.push(runInfo);
          }
        } catch { /* skip unreadable */ }
        return runs;
      };

      for (const entry of entries) {
        if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
        const entryPath = path.join(emberDir, entry.name);

        // Primary: .ember-project ID file -> project always detected
        const idFile = path.join(entryPath, '.ember-project');
        if (fs.existsSync(idFile)) {
          const runs = scanProjectRuns(entryPath);
          runs.sort((a: any, b: any) => b.lastModified - a.lastModified);
          const stat = fs.statSync(entryPath);
          projects.push({
            name: entry.name,
            path: entryPath,
            runs,
            lastModified: runs.length > 0 ? runs[0].lastModified : stat.mtimeMs,
          });
          continue;
        }

        // Fallback: detect by run output (for projects created before .ember-project)
        const runs = scanProjectRuns(entryPath);
        if (runs.length > 0) {
          runs.sort((a: any, b: any) => b.lastModified - a.lastModified);
          projects.push({
            name: entry.name,
            path: entryPath,
            runs,
            lastModified: runs[0].lastModified,
          });
        } else {
          // Legacy grouped layout: folder name matches ff pattern
          const match = entry.name.match(legacyPattern);
          if (match) {
            const projectName = match[1];
            const files = fs.readdirSync(entryPath);
            const hasSimOutput = files.some((f: string) => f.endsWith('_system.pdb') || f.endsWith('_trajectory.dcd') || f === 'simulation.log' || f === 'system.pdb' || f === 'trajectory.dcd');
            if (hasSimOutput) {
              if (!legacyGroups[projectName]) legacyGroups[projectName] = [];
              const stat = fs.statSync(entryPath);
              legacyGroups[projectName].push({
                folderName: entry.name,
                path: entryPath,
                lastModified: stat.mtimeMs,
                type: 'simulation',
                hasTrajectory: files.some((f: string) => f.endsWith('_trajectory.dcd') || f === 'trajectory.dcd'),
                hasFinalPdb: files.some((f: string) => f.endsWith('_final.pdb') || f === 'final.pdb'),
              });
            }
          }
        }
      }

      for (const [name, runs] of Object.entries(legacyGroups)) {
        runs.sort((a: any, b: any) => b.lastModified - a.lastModified);
        projects.push({
          name,
          path: path.join(emberDir, name),
          runs,
          lastModified: runs[0].lastModified,
        });
      }

      projects.sort((a, b) => b.lastModified - a.lastModified);
    } catch (err) {
      console.error('Error scanning projects:', err);
    }
    return projects;
  });

  // Project browser: scan a run directory for output files
  ipcMain.handle(IpcChannels.SCAN_RUN_FILES, async (_event: any, runDir: string): Promise<any> => {
    const result: any = {
      systemPdb: null,
      trajectory: null,
      finalPdb: null,
      equilibratedPdb: null,
      energyCsv: null,
    };
    try {
      if (!fs.existsSync(runDir)) return result;
      const files = fs.readdirSync(runDir);

      // Check results/ subdir first (new layout)
      const resultsDir = path.join(runDir, 'results');
      const resultsFiles = fs.existsSync(resultsDir) ? fs.readdirSync(resultsDir) : [];

      // New layout: unprefixed files in results/
      for (const f of resultsFiles) {
        if (f === 'system.pdb' && !result.systemPdb) result.systemPdb = path.join(resultsDir, f);
        else if (f === 'trajectory.dcd' && !result.trajectory) result.trajectory = path.join(resultsDir, f);
        else if (f === 'final.pdb' && !result.finalPdb) result.finalPdb = path.join(resultsDir, f);
        else if (f === 'equilibrated.pdb' && !result.equilibratedPdb) result.equilibratedPdb = path.join(resultsDir, f);
        else if (f === 'energy.csv' && !result.energyCsv) result.energyCsv = path.join(resultsDir, f);
      }

      // Legacy: prefixed files at top level; also check unprefixed at top level
      for (const f of files) {
        if ((f.endsWith('_system.pdb') || f === 'system.pdb') && !result.systemPdb) result.systemPdb = path.join(runDir, f);
        else if ((f.endsWith('_trajectory.dcd') || f === 'trajectory.dcd') && !result.trajectory) result.trajectory = path.join(runDir, f);
        else if ((f.endsWith('_final.pdb') || f === 'final.pdb') && !result.finalPdb) result.finalPdb = path.join(runDir, f);
        else if ((f.endsWith('_equilibrated.pdb') || f === 'equilibrated.pdb') && !result.equilibratedPdb) result.equilibratedPdb = path.join(runDir, f);
        else if ((f.endsWith('_energy.csv') || f === 'energy.csv') && !result.energyCsv) result.energyCsv = path.join(runDir, f);
      }
    } catch (err) {
      console.error('Error scanning run files:', err);
    }
    return result;
  });

  // Get file count and total size for a project (for delete confirmation)
  ipcMain.handle(IpcChannels.GET_PROJECT_FILE_COUNT, async (_event: any, projectName: string): Promise<{ fileCount: number; totalSizeMb: number }> => {
    const emberDir = path.join(app.getPath('home'), 'Ember');
    const projectDir = path.join(emberDir, projectName);
    if (!fs.existsSync(projectDir)) return { fileCount: 0, totalSizeMb: 0 };

    let fileCount = 0;
    let totalSize = 0;
    const walk = (dir: string) => {
      try {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
          const full = path.join(dir, entry.name);
          if (entry.isDirectory()) {
            walk(full);
          } else {
            fileCount++;
            try { totalSize += fs.statSync(full).size; } catch { /* skip */ }
          }
        }
      } catch { /* skip unreadable */ }
    };
    walk(projectDir);
    return { fileCount, totalSizeMb: Math.round(totalSize / (1024 * 1024) * 10) / 10 };
  });

  // Rename a project directory
  ipcMain.handle(IpcChannels.RENAME_PROJECT, async (_event: any, oldName: string, newName: string): Promise<any> => {
    const emberDir = path.join(app.getPath('home'), 'Ember');
    const oldDir = path.join(emberDir, oldName);
    const newDir = path.join(emberDir, newName);

    if (!fs.existsSync(oldDir)) {
      return { ok: false, error: { type: 'NOT_FOUND', message: `Project "${oldName}" not found` } };
    }
    if (fs.existsSync(newDir)) {
      return { ok: false, error: { type: 'ALREADY_EXISTS', message: `Project "${newName}" already exists` } };
    }

    try {
      // Rename the project directory
      fs.renameSync(oldDir, newDir);

      // Rename prefixed files inside the project (e.g., projectName_system.pdb -> newName_system.pdb)
      const renamePrefix = (dir: string) => {
        try {
          const entries = fs.readdirSync(dir, { withFileTypes: true });
          for (const entry of entries) {
            const full = path.join(dir, entry.name);
            if (entry.isDirectory()) {
              renamePrefix(full);
            } else if (entry.name.startsWith(oldName + '_')) {
              const newFileName = newName + entry.name.substring(oldName.length);
              fs.renameSync(full, path.join(dir, newFileName));
            }
          }
        } catch { /* skip unreadable */ }
      };
      renamePrefix(newDir);

      // Update .ember-project ID file with new name
      const idFile = path.join(newDir, '.ember-project');
      if (fs.existsSync(idFile)) {
        try {
          const idData = JSON.parse(fs.readFileSync(idFile, 'utf-8'));
          idData.name = newName;
          fs.writeFileSync(idFile, JSON.stringify(idData, null, 2));
        } catch { /* non-critical -- project still works without updated ID */ }
      }

      return { ok: true, value: undefined };
    } catch (err: any) {
      return { ok: false, error: { type: 'RENAME_FAILED', message: err.message || 'Failed to rename project' } };
    }
  });

  // Delete a project directory entirely
  ipcMain.handle(IpcChannels.DELETE_PROJECT, async (_event: any, projectName: string): Promise<any> => {
    const emberDir = path.join(app.getPath('home'), 'Ember');
    const projectDir = path.join(emberDir, projectName);

    if (!fs.existsSync(projectDir)) {
      return { ok: false, error: { type: 'NOT_FOUND', message: `Project "${projectName}" not found` } };
    }

    try {
      fs.rmSync(projectDir, { recursive: true, force: true });
      return { ok: true, value: undefined };
    } catch (err: any) {
      return { ok: false, error: { type: 'DELETE_FAILED', message: err.message || 'Failed to delete project' } };
    }
  });

  // Scan project artifacts as ProjectJob[] -- docking runs/poses, simulation runs, MCMM runs, and maps
  ipcMain.handle(
    IpcChannels.SCAN_PROJECT_ARTIFACTS,
    async (_event: any, projectName: string): Promise<ProjectJob[]> => {
      const emberDir = path.join(app.getPath('home'), 'Ember');
      const projectDir = path.join(emberDir, projectName);
      if (!fs.existsSync(projectDir)) return [];

      const jobs: ProjectJob[] = [];

      // Strip project name prefix from filenames for cleaner labels
      const stripPrefix = (name: string) => {
        if (name.startsWith(projectName + '_')) return name.substring(projectName.length + 1);
        return name;
      };

      // 1. Scan docking/ for docking run folders and per-pose entries
      const dockingDir = path.join(projectDir, 'docking');
      if (fs.existsSync(dockingDir)) {
        try {
          const dockRuns = fs.readdirSync(dockingDir, { withFileTypes: true });
          for (const run of dockRuns) {
            if (!run.isDirectory() || run.name.startsWith('.')) continue;
            const runPath = path.join(dockingDir, run.name);
            jobs.push(...findDockingRunJobs(runPath, run.name, stripPrefix));
          }
        } catch { /* skip */ }
      }

      // 2. Scan simulations/ for MD run folders
      const simsDir = path.join(projectDir, 'simulations');
      if (fs.existsSync(simsDir)) {
        try {
          const simRuns = fs.readdirSync(simsDir, { withFileTypes: true });
          for (const run of simRuns) {
            if (!run.isDirectory() || run.name.startsWith('.')) continue;
            const runPath = path.join(simsDir, run.name);
            const job = findSimulationJob(runPath, run.name);
            if (job) jobs.push(job);
          }
        } catch { /* skip */ }
      }

      // 3. Scan conformers/ for MCMM runs
      const conformersDir = path.join(projectDir, 'conformers');
      if (fs.existsSync(conformersDir)) {
        try {
          const conformerRuns = fs.readdirSync(conformersDir, { withFileTypes: true });
          for (const run of conformerRuns) {
            if (!run.isDirectory() || run.name.startsWith('.') || run.name.startsWith('_')) continue;
            const runPath = path.join(conformersDir, run.name);
            const job = findConformerJob(runPath, run.name);
            if (job) jobs.push(job);
          }
        } catch { /* skip */ }
      }

      // 4. Scan surfaces/ for map runs
      const surfacesDir = path.join(projectDir, 'surfaces');
      if (fs.existsSync(surfacesDir)) {
        try {
          const surfaceRuns = fs.readdirSync(surfacesDir, { withFileTypes: true });
          for (const run of surfaceRuns) {
            if (!run.isDirectory() || run.name.startsWith('.')) continue;
            if (run.name !== 'binding_site_map' && !run.name.startsWith('pocket_map_')) continue;
            const runPath = path.join(surfacesDir, run.name);
            const job = findMapJob(projectDir, runPath, run.name);
            if (job) jobs.push(job);
          }
        } catch { /* skip */ }
      }

      return jobs.sort((a, b) => (b.lastModified ?? 0) - (a.lastModified ?? 0));
    }
  );

  // Read image file and return as data URL (for thumbnail display in Electron)
  ipcMain.handle(
    'read-image-as-data-url',
    async (_event, imagePath: string): Promise<string | null> => {
      try {
        if (!fs.existsSync(imagePath)) return null;
        const data = fs.readFileSync(imagePath);
        const ext = path.extname(imagePath).toLowerCase();
        const mimeType = ext === '.png' ? 'image/png' : 'image/jpeg';
        return `data:${mimeType};base64,${data.toString('base64')}`;
      } catch {
        return null;
      }
    }
  );
}
