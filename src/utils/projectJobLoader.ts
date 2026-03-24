// Copyright (c) 2026 Ember Contributors. MIT License.
import type { ElectronAPI } from '../../shared/types/electron-api';
import type { ClusteringResult, ProjectJob, ScoredClusterResult } from '../../shared/types/ipc';
import { workflowStore } from '../stores/workflow';
import { buildDockingProjectTable, buildDockingViewerQueue } from './viewerQueue';

export async function loadProjectJob(job: ProjectJob, api: ElectronAPI): Promise<void> {
  const {
    clearViewerSession,
    openViewerSession,
    addViewerProjectFamily,
    setMode,
    setDockOutputDir,
    setDockResults,
    setDockReceptorPdbPath,
    setDockStep,
    setDockCordialScored,
    setMdResult,
    setMdOutputDir,
    setMdStep,
    setMdClusteringResultsFromIpc,
    setMdClusterScores,
    setMdTorsionAnalysis,
    setConformOutputDir,
    setConformPaths,
    setConformOutputName,
    setConformLigandName,
    setConformStep,
    setMapMethod,
    setMapStep,
    setMapPdbPath,
    setMapDetectedLigands,
    setMapSelectedLigandId,
    setMapIsDetecting,
    setMapResult,
  } = workflowStore;

  const loadClusterScores = async () => {
    const candidatePaths = [
      `${job.path}/results/analysis/scored_clusters/cluster_scores.json`,
      `${job.path}/analysis/scored_clusters/cluster_scores.json`,
    ];
    for (const candidatePath of candidatePaths) {
      const scoreData = await api.readJsonFile(candidatePath) as { clusters?: unknown[] } | null;
      if (scoreData && Array.isArray(scoreData.clusters)) {
        return scoreData.clusters as any[];
      }
    }
    return [];
  };

  const loadClusteringResults = async () => {
    const candidatePaths = [
      job.clusteringResultsPath,
      `${job.path}/results/analysis/clustering/clustering_results.json`,
      `${job.path}/analysis/clustering/clustering_results.json`,
      `${job.path}/clustering/clustering_results.json`,
    ].filter((candidate): candidate is string => Boolean(candidate));

    for (const candidatePath of candidatePaths) {
      const clusteringData = await api.readJsonFile(candidatePath) as {
        clusters?: unknown[];
        frameAssignments?: unknown[];
        requestedClusters?: unknown;
        actualClusters?: unknown;
      } | null;
      if (!clusteringData || !Array.isArray(clusteringData.clusters) || !Array.isArray(clusteringData.frameAssignments)) {
        continue;
      }
      return {
        clusters: clusteringData.clusters as ClusteringResult['clusters'],
        frameAssignments: clusteringData.frameAssignments as number[],
        outputDir: candidatePath.replace(/\/clustering_results\.json$/, ''),
        requestedClusters: typeof clusteringData.requestedClusters === 'number' ? clusteringData.requestedClusters : undefined,
        actualClusters: typeof clusteringData.actualClusters === 'number' ? clusteringData.actualClusters : undefined,
      };
    }

    return null;
  };

  const loadMdTorsionAnalysis = async () => {
    return api.loadMdTorsionAnalysis({ analysisDir: `${job.path}/analysis` });
  };

  const loadMapResult = async () => {
    const pathParts = job.path.split('/');
    const projectName = pathParts.length >= 3 ? pathParts[pathParts.length - 3] : '';
    const candidatePaths = [
      job.mapResultJson,
      projectName ? `${job.path}/${projectName}_binding_site_results.json` : null,
      `${job.path}/binding_site_results.json`,
    ].filter((candidate): candidate is string => Boolean(candidate));

    for (const candidatePath of candidatePaths) {
      const result = await api.readJsonFile(candidatePath) as {
        hydrophobicDx?: string;
        hbondDonorDx?: string;
        hbondAcceptorDx?: string;
        hotspots?: Array<{ type: string; position: number[]; direction: number[]; score: number }>;
        method?: string;
      } | null;
      if (!result?.hydrophobicDx || !result.hbondDonorDx || !result.hbondAcceptorDx) continue;
      return {
        hydrophobic: { visible: true, isolevel: 0.3, opacity: 0.7 },
        hbondDonor: { visible: true, isolevel: 0.3, opacity: 0.7 },
        hbondAcceptor: { visible: true, isolevel: 0.3, opacity: 0.7 },
        hydrophobicDx: result.hydrophobicDx,
        hbondDonorDx: result.hbondDonorDx,
        hbondAcceptorDx: result.hbondAcceptorDx,
        hotspots: result.hotspots || [],
        method: (job.mapMethod || result.method || 'solvation') as 'solvation',
        pdbPath: job.mapPdb || '',
        outputDir: job.path,
        trajectoryPath: job.mapTrajectoryDcd || null,
      };
    }
    return null;
  };

  if (job.type === 'docking-pose') {
    if (!job.receptorPdb || !job.poses || job.poses.length === 0) return;

    const queue = buildDockingViewerQueue(job.receptorPdb, job.poses);
    const selectedPoseIndex = Math.min(Math.max(job.poseIndex ?? 0, 0), queue.length - 1);
    const selectedPose = job.poses[selectedPoseIndex];
    const projectTable = buildDockingProjectTable({
      familyId: job.parentId || `dock:${job.folder}`,
      title: job.parentLabel || job.folder,
      receptorPdb: job.receptorPdb,
      holoPdb: job.holoPdb,
      preparedLigandPath: job.preparedLigandPath,
      referenceLigandPath: job.referenceLigandPath,
      poses: queue.map((item, index) => ({
        ligandName: job.poses![index].name,
        outputSdf: item.ligandPath!,
        vinaAffinity: job.poses![index].affinity ?? null,
      })) as any,
      poseQueue: queue,
      selectedQueueIndex: selectedPoseIndex,
    });

    openViewerSession({
      pdbPath: job.receptorPdb,
      ligandPath: selectedPose?.path || job.ligandPath || null,
      pdbQueue: queue,
      pdbQueueIndex: selectedPoseIndex,
    });
    // Accumulate into existing project table (not replace)
    addViewerProjectFamily(projectTable.families[0], projectTable.rows);
    return;
  }

  if (job.type === 'docking') {
    try {
      const parseResult = await api.parseDockResults(job.path);
      if (parseResult.ok) {
        setDockOutputDir(job.path);
        setDockResults(parseResult.value);
        if (job.receptorPdb) setDockReceptorPdbPath(job.receptorPdb);
        setDockCordialScored(parseResult.value.some((r: any) => r.cordialExpectedPkd != null));
        setMode('dock');
        setDockStep('dock-results');
      } else if (job.poses && job.poses.length > 0 && job.receptorPdb) {
        const queue = buildDockingViewerQueue(job.receptorPdb, job.poses);
        openViewerSession({
          pdbPath: queue[0].pdbPath,
          ligandPath: queue[0].ligandPath || null,
          pdbQueue: queue,
          pdbQueueIndex: 0,
        });
      } else {
        clearViewerSession();
        setMode('viewer');
      }
    } catch {
      clearViewerSession();
      setMode('viewer');
    }
    return;
  }

  if (job.type === 'simulation') {
    const systemPdb = job.systemPdb || '';
    const trajectoryDcd = job.trajectoryDcd || '';
    const finalPdb = job.finalPdb || systemPdb;
    const trajectoryName = trajectoryDcd.split('/').pop() || '';
    const energyCsvPath = trajectoryName.endsWith('_trajectory.dcd')
      ? trajectoryDcd.replace(/_trajectory\.dcd$/, '_energy.csv')
      : trajectoryDcd.replace(/trajectory\.dcd$/, 'energy.csv');

    if (systemPdb && trajectoryDcd) {
      const [clusteringResults, clusterScores, torsionAnalysis] = await Promise.all([
        loadClusteringResults(),
        loadClusterScores(),
        loadMdTorsionAnalysis(),
      ]);
      setMdClusteringResultsFromIpc(clusteringResults);
      setMdClusterScores(clusterScores as ScoredClusterResult[]);
      setMdTorsionAnalysis(torsionAnalysis.ok ? torsionAnalysis.value : null);
      setMdResult({
        systemPdbPath: systemPdb,
        trajectoryPath: trajectoryDcd,
        equilibratedPdbPath: systemPdb,
        finalPdbPath: finalPdb,
        energyCsvPath,
      });
      setMdOutputDir(job.path);
      setMode('md');
      setMdStep('md-results');
    } else {
      openViewerSession({
        pdbPath: finalPdb || null,
      });
    }
    return;
  }

  if (job.type === 'conformer') {
    const conformerPaths = job.conformerPaths || [];
    setConformOutputDir(job.path);
    setConformPaths(conformerPaths);
    setConformOutputName(job.folder);
    setConformLigandName(job.folder);
    setMode('conform');
    setConformStep('conform-results');
    return;
  }

  if (job.type === 'map') {
    const mapResult = await loadMapResult();
    if (!mapResult) {
      if (job.mapPdb) {
        openViewerSession({
          pdbPath: job.mapPdb,
          trajectoryPath: job.mapTrajectoryDcd || null,
        });
      }
      return;
    }

    setMapMethod(mapResult.method);
    setMapPdbPath(mapResult.pdbPath || null);
    if (mapResult.pdbPath) {
      try {
        const detected = await api.detectPdbLigands(mapResult.pdbPath);
        if (detected.ok) {
          const ligands = Array.isArray(detected.value) ? detected.value : detected.value.ligands;
          setMapDetectedLigands(ligands);
          setMapSelectedLigandId(ligands[0]?.id || null);
        } else {
          setMapDetectedLigands([]);
          setMapSelectedLigandId(null);
        }
      } catch {
        setMapDetectedLigands([]);
        setMapSelectedLigandId(null);
      }
    } else {
      setMapDetectedLigands([]);
      setMapSelectedLigandId(null);
    }
    setMapIsDetecting(false);
    setMapResult(mapResult);
    setMode('map');
    setMapStep('map-results');
  }
}
