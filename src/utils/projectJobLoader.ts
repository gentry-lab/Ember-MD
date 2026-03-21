import type { ElectronAPI } from '../../shared/types/electron-api';
import type { ProjectJob } from '../../shared/types/ipc';
import { workflowStore } from '../stores/workflow';
import { buildDockingViewerQueue } from './viewerQueue';

export async function loadProjectJob(job: ProjectJob, api: ElectronAPI): Promise<void> {
  const {
    clearViewerSession,
    openViewerSession,
    setMode,
    setDockOutputDir,
    setDockResults,
    setDockReceptorPdbPath,
    setDockStep,
    setDockCordialScored,
    setMdResult,
    setMdOutputDir,
    setMdStep,
    setMdClusterScores,
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

    openViewerSession({
      pdbPath: job.receptorPdb,
      ligandPath: selectedPose?.path || job.ligandPath || null,
      pdbQueue: queue,
      pdbQueueIndex: selectedPoseIndex,
    });
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
      setMdClusterScores(await loadClusterScores());
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
          setMapDetectedLigands(detected.value);
          setMapSelectedLigandId(detected.value[0]?.id || null);
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
