// Copyright (c) 2026 Ember Contributors. MIT License.
/**
 * Score tab IPC handlers.
 *
 * Batch-scores protein-ligand complex PDBs with Vina score_only,
 * CORDIAL neural-network rescoring, and RDKit QED.
 *
 * Reuses existing Python scripts for ligand detection, extraction,
 * receptor preparation, and scoring. No new scoring logic.
 */
import { ipcMain } from 'electron';
import * as path from 'path';
import * as fs from 'fs';
import { Ok, Err } from '../../shared/types/result';
import { IpcChannels } from '../../shared/types/ipc';
import type { BatchScoreRequest, BatchScoreEntryResult, BatchScoreResult, ScoreTrajectoryRequest } from '../../shared/types/ipc';
import type { ScoredClusterResult } from '../../shared/types/ipc';
import * as appState from '../app-state';
import { spawnPythonScript as _spawnPythonScriptRaw } from '../spawn';
import { getCordialRoot } from '../paths';
import { runVinaScoreOnly, runCordialScoringJob, parseSdfProperties } from '../scoring-utils';

// ---------------------------------------------------------------------------
// Local convenience wrappers
// ---------------------------------------------------------------------------

function spawnPythonScript(
  args: string[],
  options?: {
    env?: NodeJS.ProcessEnv;
    cwd?: string;
    onStdout?: (text: string) => void;
    onStderr?: (text: string) => void;
  },
): Promise<{ stdout: string; stderr: string; code: number }> {
  return _spawnPythonScriptRaw(appState.condaPythonPath, appState.condaEnvBin, args, options);
}

let cancelRequested = false;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function emit(event: Electron.IpcMainInvokeEvent, text: string): void {
  event.sender.send(IpcChannels.SCORE_OUTPUT, { type: 'stdout', data: text });
}

async function extractLigandFromPdb(
  pdbPath: string,
  ligandId: string,
  outputSdfPath: string,
  event: Electron.IpcMainInvokeEvent,
): Promise<{ ok: boolean; sdfPath?: string; error?: string }> {
  const scriptPath = path.join(appState.fraggenRoot, 'detect_pdb_ligands.py');
  if (!fs.existsSync(scriptPath)) {
    return { ok: false, error: 'detect_pdb_ligands.py not found' };
  }

  fs.mkdirSync(path.dirname(outputSdfPath), { recursive: true });
  const args = [
    scriptPath,
    '--mode', 'extract',
    '--pdb', pdbPath,
    '--ligand_id', ligandId,
    '--output', outputSdfPath,
  ];

  const { code, stderr } = await spawnPythonScript(args, {
    onStderr: (text) => emit(event, text),
  });

  if (code !== 0 || !fs.existsSync(outputSdfPath)) {
    return { ok: false, error: stderr || 'Ligand extraction failed' };
  }
  return { ok: true, sdfPath: outputSdfPath };
}

async function prepareReceptorForScoring(
  pdbPath: string,
  ligandId: string,
  outputPath: string,
  event: Electron.IpcMainInvokeEvent,
): Promise<{ ok: boolean; preparedPath?: string; error?: string }> {
  const scriptPath = path.join(appState.fraggenRoot, 'detect_pdb_ligands.py');
  if (!fs.existsSync(scriptPath)) {
    return { ok: false, error: 'detect_pdb_ligands.py not found' };
  }

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  const args = [
    scriptPath,
    '--mode', 'prepare_receptor',
    '--pdb', pdbPath,
    '--ligand_id', ligandId,
    '--output', outputPath,
    '--water_distance', '0',
    '--ph', '7.4',
  ];

  const { code, stderr } = await spawnPythonScript(args, {
    onStdout: (text) => emit(event, text),
    onStderr: (text) => {
      // Relay PROGRESS: lines
      for (const line of text.split('\n')) {
        if (line.startsWith('PROGRESS: ')) {
          emit(event, `  ${line.replace('PROGRESS: ', '')}\n`);
        }
      }
    },
  });

  if (code !== 0 || !fs.existsSync(outputPath)) {
    return { ok: false, error: stderr || 'Receptor preparation failed' };
  }
  return { ok: true, preparedPath: outputPath };
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

export function register(): void {

  // -------------------------------------------------------------------------
  // SCORE_BATCH — Main scoring pipeline
  // -------------------------------------------------------------------------
  ipcMain.handle(IpcChannels.SCORE_BATCH, async (event, request: BatchScoreRequest) => {
    cancelRequested = false;
    const { entries, outputDir } = request;
    const results: BatchScoreEntryResult[] = [];

    fs.mkdirSync(outputDir, { recursive: true });
    const runRoot = path.dirname(outputDir);
    const inputsDir = path.join(runRoot, 'inputs');
    fs.mkdirSync(inputsDir, { recursive: true });

    const cordialRoot = getCordialRoot();
    const cordialAvailable = !!cordialRoot;

    for (let i = 0; i < entries.length; i++) {
      if (cancelRequested) break;

      const entry = entries[i];
      const name = entry.name || path.basename(entry.pdbPath).replace(/\.(pdb|cif)$/i, '');
      const id = entry.id;
      const entryDir = path.join(outputDir, name);
      const stagedPdbPath = path.join(inputsDir, `${String(i + 1).padStart(3, '0')}_${path.basename(entry.pdbPath)}`);
      if (path.resolve(stagedPdbPath) !== path.resolve(entry.pdbPath)) {
        fs.copyFileSync(entry.pdbPath, stagedPdbPath);
      }
      fs.mkdirSync(entryDir, { recursive: true });

      emit(event, `\n[Score] (${i + 1}/${entries.length}) ${name}\n`);

      const result: BatchScoreEntryResult = {
        id,
        pdbPath: stagedPdbPath,
        name,
        ligandId: entry.ligandId,
        isPrepared: entry.isPrepared,
        preparedReceptorPath: null,
        extractedLigandSdfPath: null,
        vinaScore: null,
        cordialExpectedPkd: null,
        cordialPHighAffinity: null,
        qed: null,
        status: 'done',
        errorMessage: null,
      };

      try {
        // Detect ligand if not specified
        let ligandId = entry.ligandId;
        if (!ligandId) {
          emit(event, `  Detecting ligands...\n`);
          const detectScript = path.join(appState.fraggenRoot, 'detect_pdb_ligands.py');
          const { stdout, code } = await spawnPythonScript([
            detectScript, '--mode', 'detect', '--pdb', stagedPdbPath,
          ]);
          if (code === 0 && stdout.trim()) {
            try {
              const parsed = JSON.parse(stdout.trim());
              if (parsed.ligands && parsed.ligands.length > 0) {
                // Pick the largest ligand by atom count
                const sorted = [...parsed.ligands].sort((a: any, b: any) => (b.num_atoms || 0) - (a.num_atoms || 0));
                ligandId = sorted[0].id;
                result.ligandId = ligandId;
                emit(event, `  Found ligand: ${sorted[0].resname || ligandId} (${sorted[0].num_atoms} atoms)\n`);
              }
            } catch { /* ignore parse errors */ }
          }
          if (!ligandId) {
            result.status = 'error';
            result.errorMessage = 'No ligand detected in PDB';
            results.push(result);
            emit(event, `  Error: No ligand detected\n`);
            continue;
          }
        }

        // Extract ligand to SDF
        emit(event, `  Extracting ligand...\n`);
        const ligandSdfPath = path.join(entryDir, 'ligand.sdf');
        const extractResult = await extractLigandFromPdb(stagedPdbPath, ligandId, ligandSdfPath, event);
        if (!extractResult.ok) {
          result.status = 'error';
          result.errorMessage = extractResult.error || 'Ligand extraction failed';
          results.push(result);
          emit(event, `  Error: ${result.errorMessage}\n`);
          continue;
        }
        result.extractedLigandSdfPath = extractResult.sdfPath!;

        // Prepare receptor if needed
        let receptorPath = stagedPdbPath;
        if (!entry.isPrepared) {
          emit(event, `  Preparing receptor (auto-protonation)...\n`);
          const prepPath = path.join(entryDir, 'receptor_prepared.pdb');
          const prepResult = await prepareReceptorForScoring(stagedPdbPath, ligandId, prepPath, event);
          if (prepResult.ok) {
            receptorPath = prepResult.preparedPath!;
            result.preparedReceptorPath = receptorPath;
            emit(event, `  Receptor prepared\n`);
          } else {
            // Non-fatal: continue with unprepared receptor
            emit(event, `  Warning: Receptor preparation failed, scoring with raw PDB\n`);
          }
        } else {
          result.preparedReceptorPath = stagedPdbPath;
        }

        if (cancelRequested) break;

        // Score with Vina score_only
        emit(event, `  Vina score_only...\n`);
        const vinaResult = await runVinaScoreOnly(receptorPath, ligandSdfPath, ligandSdfPath, {
          autoboxAdd: 4,
          cpu: 1,
          onStderr: (text) => emit(event, text),
        });
        if (vinaResult.ok) {
          result.vinaScore = Math.round(vinaResult.value * 10) / 10;
          emit(event, `  Vina: ${result.vinaScore} kcal/mol\n`);
        } else {
          emit(event, `  Vina failed: ${vinaResult.error.message}\n`);
        }

        // Compute QED via parse_sdf_properties
        emit(event, `  Computing QED...\n`);
        const sdfProps = await parseSdfProperties(ligandSdfPath);
        if (sdfProps.success) {
          result.qed = sdfProps.qed;
          emit(event, `  QED: ${result.qed}\n`);
        } else {
          emit(event, `  QED failed: ${sdfProps.error}\n`);
        }

        // Emit progress for live UI updates
        event.sender.send(IpcChannels.SCORE_OUTPUT, {
          type: 'stdout',
          data: `SCORE_ENTRY_RESULT:${id}:${JSON.stringify(result)}\n`,
        });

      } catch (err) {
        result.status = 'error';
        result.errorMessage = (err as Error).message;
        emit(event, `  Error: ${result.errorMessage}\n`);
      }

      results.push(result);
    }

    // Batch CORDIAL scoring at the end
    if (cordialAvailable && !cancelRequested) {
      const scorableEntries = results.filter(
        (r) => r.status === 'done' && r.extractedLigandSdfPath && r.preparedReceptorPath,
      );

      if (scorableEntries.length > 0) {
        emit(event, `\n[Score] Running CORDIAL on ${scorableEntries.length} complexes...\n`);

        // Write pair CSV for CORDIAL
        const pairCsvPath = path.join(outputDir, 'cordial_pairs.csv');
        const csvLines = ['source_name,ligand_sdf,receptor_pdb,pose_index'];
        for (const entry of scorableEntries) {
          csvLines.push(`${entry.name},${entry.extractedLigandSdfPath},${entry.preparedReceptorPath},0`);
        }
        fs.writeFileSync(pairCsvPath, csvLines.join('\n'));

        const cordialOutputCsv = path.join(outputDir, 'cordial_scores.csv');
        const cordialResult = await runCordialScoringJob(
          { pairCsv: pairCsvPath },
          cordialOutputCsv,
          8,
          {
            onStdout: (text) => emit(event, text),
            onStderr: (text) => emit(event, text),
          },
        );

        if (cordialResult.ok) {
          emit(event, `  CORDIAL scored ${cordialResult.value.count} complexes\n`);
          // Parse CORDIAL JSON output and merge
          const jsonPath = cordialOutputCsv.replace(/\.csv$/, '.json');
          if (fs.existsSync(jsonPath)) {
            try {
              const cordialData = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
              const scoresByName = new Map<string, any>();
              for (const row of cordialData) {
                scoresByName.set(row.source_name, row);
              }
              for (const entry of results) {
                const score = scoresByName.get(entry.name);
                if (score) {
                  entry.cordialExpectedPkd = score.cordial_expected_pkd ?? null;
                  entry.cordialPHighAffinity = score.cordial_p_high_affinity ?? null;
                }
              }
            } catch (err) {
              emit(event, `  Warning: Failed to parse CORDIAL JSON: ${err}\n`);
            }
          }
        } else {
          emit(event, `  CORDIAL failed: ${cordialResult.error.message}\n`);
        }
      }
    }

    // Write results JSON
    const resultsJsonPath = path.join(outputDir, 'score_results.json');
    fs.writeFileSync(resultsJsonPath, JSON.stringify({ entries: results }, null, 2));
    const doneCount = results.filter((entry) => entry.status === 'done').length;
    const errorCount = results.length - doneCount;
    emit(event, `\n[Score] Batch summary: ${doneCount} done, ${errorCount} error, ${results.length} total\n`);
    emit(event, `\n[Score] Results saved to ${resultsJsonPath}\n`);

    return Ok({ entries: results, outputDir, cordialAvailable } as BatchScoreResult);
  });

  // -------------------------------------------------------------------------
  // CANCEL_SCORE_BATCH
  // -------------------------------------------------------------------------
  ipcMain.handle(IpcChannels.CANCEL_SCORE_BATCH, async () => {
    cancelRequested = true;
  });

  // -------------------------------------------------------------------------
  // EXPORT_SCORE_CSV
  // -------------------------------------------------------------------------
  ipcMain.handle(IpcChannels.EXPORT_SCORE_CSV, async (_event, entriesJson: string, csvPath: string) => {
    try {
      const entries: BatchScoreEntryResult[] = JSON.parse(entriesJson);
      const lines = ['Name,Vina (kcal/mol),CORDIAL pKd,CORDIAL P(high),QED,Prepared,Status'];
      for (const e of entries) {
        lines.push([
          e.name,
          e.vinaScore ?? '',
          e.cordialExpectedPkd ?? '',
          e.cordialPHighAffinity ?? '',
          e.qed ?? '',
          e.isPrepared ? 'Yes' : 'Auto',
          e.status,
        ].join(','));
      }
      fs.mkdirSync(path.dirname(csvPath), { recursive: true });
      fs.writeFileSync(csvPath, lines.join('\n'));
      return Ok(csvPath);
    } catch (err) {
      return Err({ type: 'WRITE_FAILED', message: (err as Error).message });
    }
  });

  // -------------------------------------------------------------------------
  // GET_MOLECULE_DETAILS — lazy on-demand details for detail panels
  // -------------------------------------------------------------------------
  ipcMain.handle(IpcChannels.GET_MOLECULE_DETAILS, async (_event, sdfPath: string, referenceSdfPath?: string) => {
    try {
      const result = await parseSdfProperties(sdfPath, referenceSdfPath);
      if (!result.success) {
        return Err({ type: 'PARSE_FAILED', message: result.error || 'Failed to parse SDF' });
      }
      return Ok({
        thumbnail: result.thumbnail || null,
        centroid: result.centroid || null,
        rmsd: result.rmsd ?? null,
        qed: result.qed,
        mw: result.mw,
        logp: result.logp,
        smiles: result.smiles || null,
      });
    } catch (err) {
      return Err({ type: 'PARSE_FAILED', message: (err as Error).message });
    }
  });

  // -------------------------------------------------------------------------
  // SCORE_TRAJECTORY — cluster a DCD trajectory then score centroids
  // -------------------------------------------------------------------------
  ipcMain.handle(IpcChannels.SCORE_TRAJECTORY, async (event, request: ScoreTrajectoryRequest) => {
    cancelRequested = false;
    const { trajectoryPath, topologyPath, ligandSdfPath, numClusters, outputDir } = request;

    fs.mkdirSync(outputDir, { recursive: true });
    const clusteringDir = path.join(outputDir, 'clustering');
    const scoredDir = path.join(outputDir, 'scored_clusters');
    fs.mkdirSync(clusteringDir, { recursive: true });
    fs.mkdirSync(scoredDir, { recursive: true });

    const cordialRoot = getCordialRoot();
    const cordialAvailable = !!cordialRoot;

    emit(event, `[Score] Trajectory scoring: ${path.basename(trajectoryPath)}\n`);
    emit(event, `  Topology: ${path.basename(topologyPath)}\n`);
    emit(event, `  Ligand: ${path.basename(ligandSdfPath)}\n`);
    emit(event, `  Clusters: ${numClusters}\n\n`);

    // --- Step 1: Cluster (or reuse existing) ---
    const existingClustering = path.join(clusteringDir, 'clustering_results.json');
    let clusteringData: any = null;
    if (fs.existsSync(existingClustering)) {
      try {
        clusteringData = JSON.parse(fs.readFileSync(existingClustering, 'utf-8'));
        if (clusteringData?.clusters?.length > 0) {
          emit(event, `  Reusing existing clustering (${clusteringData.clusters.length} centroids)\n`);
        } else {
          clusteringData = null;
        }
      } catch { clusteringData = null; }
    }

    if (!clusteringData) {
      emit(event, `  Clustering trajectory into ${numClusters} centroids...\n`);
      const clusterScript = path.join(appState.fraggenRoot, 'cluster_trajectory.py');
      if (!fs.existsSync(clusterScript)) {
        return Err({ type: 'SCRIPT_NOT_FOUND', message: 'cluster_trajectory.py not found' });
      }
      const { code, stderr } = await spawnPythonScript([
        clusterScript,
        '--topology', topologyPath,
        '--trajectory', trajectoryPath,
        '--n_clusters', String(numClusters),
        '--method', 'kmeans',
        '--selection', 'ligand',
        '--strip_waters',
        '--output_dir', clusteringDir,
      ], {
        onStdout: (text) => emit(event, text),
        onStderr: (text) => emit(event, text),
      });
      if (code !== 0) {
        return Err({ type: 'CLUSTERING_FAILED', message: stderr || 'Clustering failed' });
      }
      try {
        clusteringData = JSON.parse(fs.readFileSync(existingClustering, 'utf-8'));
      } catch {
        return Err({ type: 'CLUSTERING_FAILED', message: 'Failed to read clustering results' });
      }
    }

    if (!clusteringData?.clusters?.length) {
      return Err({ type: 'CLUSTERING_FAILED', message: 'No clusters produced' });
    }

    // --- Step 2: Prepare centroid receptor/ligand pairs ---
    emit(event, `\n  Preparing ${clusteringData.clusters.length} centroid pairs...\n`);
    const scoreScript = path.join(appState.fraggenRoot, 'score_cluster_centroids.py');
    if (fs.existsSync(scoreScript)) {
      const { code } = await spawnPythonScript([
        scoreScript,
        '--clustering_dir', clusteringDir,
        '--input_ligand_sdf', ligandSdfPath,
        '--output_dir', scoredDir,
      ], {
        onStdout: (text) => emit(event, text),
        onStderr: (text) => emit(event, text),
      });
      if (code !== 0) {
        emit(event, `  Warning: Centroid preparation failed, scoring raw centroids\n`);
      }
    }

    // Read scored cluster scaffold
    let clusterScores: ScoredClusterResult[] = [];
    const clusterScoresPath = path.join(scoredDir, 'cluster_scores.json');
    if (fs.existsSync(clusterScoresPath)) {
      try {
        const data = JSON.parse(fs.readFileSync(clusterScoresPath, 'utf-8'));
        clusterScores = data.clusters || [];
      } catch { /* ignore */ }
    }

    // --- Step 3: Score each centroid with Vina + QED ---
    const results: BatchScoreEntryResult[] = [];
    for (let i = 0; i < clusteringData.clusters.length; i++) {
      if (cancelRequested) break;
      const cluster = clusteringData.clusters[i];
      const scored = clusterScores.find((c: ScoredClusterResult) => c.clusterId === cluster.clusterId);
      const centroidPdb = cluster.centroidPdbPath || scored?.centroidPdbPath;
      const receptorPdb = scored?.receptorPdbPath;
      const ligandSdf = scored?.ligandSdfPath;
      const name = `Cluster ${cluster.clusterId + 1} (${cluster.population?.toFixed(1) || '?'}%)`;

      emit(event, `\n[Score] (${i + 1}/${clusteringData.clusters.length}) ${name}\n`);

      const result: BatchScoreEntryResult = {
        id: String(i),
        pdbPath: centroidPdb || '',
        name,
        ligandId: null,
        isPrepared: true,
        preparedReceptorPath: receptorPdb || null,
        extractedLigandSdfPath: ligandSdf || null,
        vinaScore: null,
        cordialExpectedPkd: null,
        cordialPHighAffinity: null,
        qed: null,
        status: 'done',
        errorMessage: null,
      };

      if (receptorPdb && ligandSdf) {
        // Vina score_only
        emit(event, `  Vina score_only...\n`);
        const vinaResult = await runVinaScoreOnly(receptorPdb, ligandSdf, ligandSdf, {
          autoboxAdd: 4,
          cpu: 1,
          onStderr: (text) => emit(event, text),
        });
        if (vinaResult.ok) {
          result.vinaScore = Math.round(vinaResult.value * 10) / 10;
          emit(event, `  Vina: ${result.vinaScore} kcal/mol\n`);
        } else {
          emit(event, `  Vina failed: ${vinaResult.error.message}\n`);
        }

        // QED
        const sdfProps = await parseSdfProperties(ligandSdf);
        if (sdfProps.success) {
          result.qed = sdfProps.qed;
          emit(event, `  QED: ${result.qed}\n`);
        }
      } else {
        emit(event, `  Skipping scoring (no prepared receptor/ligand pair)\n`);
      }

      results.push(result);
    }

    // --- Step 4: Batch CORDIAL ---
    if (cordialAvailable && !cancelRequested) {
      const scorableEntries = results.filter(
        (r) => r.status === 'done' && r.extractedLigandSdfPath && r.preparedReceptorPath,
      );
      if (scorableEntries.length > 0) {
        emit(event, `\n[Score] Running CORDIAL on ${scorableEntries.length} centroids...\n`);
        const pairCsvPath = path.join(outputDir, 'cordial_pairs.csv');
        const csvLines = ['source_name,ligand_sdf,receptor_pdb,pose_index'];
        for (const entry of scorableEntries) {
          csvLines.push(`${entry.name},${entry.extractedLigandSdfPath},${entry.preparedReceptorPath},0`);
        }
        fs.writeFileSync(pairCsvPath, csvLines.join('\n'));

        const cordialOutputCsv = path.join(outputDir, 'cordial_scores.csv');
        const cordialResult = await runCordialScoringJob(
          { pairCsv: pairCsvPath },
          cordialOutputCsv,
          8,
          {
            onStdout: (text) => emit(event, text),
            onStderr: (text) => emit(event, text),
          },
        );
        if (cordialResult.ok) {
          const jsonPath = cordialOutputCsv.replace(/\.csv$/, '.json');
          if (fs.existsSync(jsonPath)) {
            try {
              const cordialData = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
              const scoresByName = new Map<string, any>();
              for (const row of cordialData) {
                scoresByName.set(row.source_name, row);
              }
              for (const entry of results) {
                const score = scoresByName.get(entry.name);
                if (score) {
                  entry.cordialExpectedPkd = score.cordial_expected_pkd ?? null;
                  entry.cordialPHighAffinity = score.cordial_p_high_affinity ?? null;
                }
              }
            } catch { /* ignore */ }
          }
        }
      }
    }

    // Write results
    const resultsJsonPath = path.join(outputDir, 'score_results.json');
    fs.writeFileSync(resultsJsonPath, JSON.stringify({ entries: results }, null, 2));
    emit(event, `\n[Score] Trajectory scoring complete. ${results.length} centroids scored.\n`);

    return Ok({ entries: results, outputDir, cordialAvailable } as BatchScoreResult);
  });
}
