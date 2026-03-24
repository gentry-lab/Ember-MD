import type { ProjectJobPose } from '../../shared/types/ipc';
import type { DockResult } from '../../shared/types/dock';
import type {
  ViewerProjectColumn,
  ViewerProjectFamily,
  ViewerProjectRow,
  ViewerProjectTableState,
  ViewerQueueItem,
} from '../stores/workflow';

export function formatDockingQueueLabel(pose: Pick<ProjectJobPose, 'name' | 'affinity'>): string {
  return `${pose.name}${pose.affinity != null ? ` (${pose.affinity.toFixed(1)} kcal/mol)` : ''}`;
}

export function buildDockingViewerQueue(
  receptorPdb: string,
  poses: Array<Pick<ProjectJobPose, 'name' | 'path' | 'affinity'>>,
): ViewerQueueItem[] {
  return poses.map((pose) => ({
    pdbPath: receptorPdb,
    ligandPath: pose.path,
    label: formatDockingQueueLabel(pose),
  }));
}

export function buildConformerViewerQueue(paths: string[]): ViewerQueueItem[] {
  return paths.map((sdfPath, index) => ({
    pdbPath: sdfPath,
    label: `Conformer ${index + 1}`,
    type: 'conformer' as const,
  }));
}

const percent = (value: number | null | undefined): string | number | null =>
  value == null ? null : value;

export function buildDockingProjectTable(options: {
  familyId: string;
  title: string;
  receptorPdb: string;
  holoPdb?: string | null;
  preparedLigandPath?: string | null;
  referenceLigandPath?: string | null;
  poses: DockResult[];
  poseQueue: ViewerQueueItem[];
  selectedQueueIndex: number;
}): ViewerProjectTableState {
  const {
    familyId,
    title,
    receptorPdb,
    holoPdb,
    preparedLigandPath,
    referenceLigandPath,
    poses,
    poseQueue,
    selectedQueueIndex,
  } = options;
  const rows: ViewerProjectRow[] = [
    {
      id: `${familyId}:apo`,
      familyId,
      label: 'Apo receptor',
      rowKind: 'apo',
      jobType: 'docking',
      item: { pdbPath: receptorPdb, label: 'Apo receptor' },
      loadKind: 'structure',
      metrics: {},
      pocketLigandPath: referenceLigandPath ?? null,
      pocketSourcePdbPath: holoPdb ?? null,
    },
  ];

  if (holoPdb) {
    rows.push({
      id: `${familyId}:holo`,
      familyId,
      label: 'Holo reference complex',
      rowKind: 'holo',
      jobType: 'docking',
      item: { pdbPath: holoPdb, label: 'Holo reference complex' },
      loadKind: 'structure',
      metrics: {},
    });
  }

  if (preparedLigandPath) {
    rows.push({
      id: `${familyId}:prepared-ligand`,
      familyId,
      label: 'Prepared ligand',
      rowKind: 'prepared-ligand',
      jobType: 'docking',
      item: { pdbPath: preparedLigandPath, label: 'Prepared ligand', type: 'ligand' },
      loadKind: 'standalone-ligand',
      metrics: {},
    });
  }

  const poseRows = poses.map<ViewerProjectRow>((pose, index) => ({
    id: `${familyId}:pose:${index}`,
    familyId,
    label: formatDockingQueueLabel({
      name: pose.ligandName,
      affinity: pose.vinaAffinity ?? pose.vinaScoreOnlyAffinity ?? undefined,
    }),
    rowKind: 'pose',
    jobType: 'docking',
    item: poseQueue[index],
    loadKind: 'queue',
    queueIndex: index,
    metrics: {
      vinaAffinity: pose.vinaAffinity ?? pose.vinaScoreOnlyAffinity ?? null,
      cordialPHighAffinity: percent(pose.cordialPHighAffinity),
      qed: pose.qed,
      xtbEnergyKcal: pose.xtbEnergyKcal ?? null,
    },
  }));

  rows.push(...poseRows);

  const columns: ViewerProjectColumn[] = [];
  if (poseRows.some((row) => row.metrics.vinaAffinity != null)) {
    columns.push({ key: 'vinaAffinity', label: 'Vina', kind: 'number', priority: 0, minPanelWidth: 0 });
  }
  if (poseRows.some((row) => row.metrics.cordialPHighAffinity != null)) {
    columns.push({ key: 'cordialPHighAffinity', label: 'P(<1uM)', kind: 'percent', priority: 1, minPanelWidth: 360 });
  }
  if (poseRows.some((row) => row.metrics.qed != null)) {
    columns.push({ key: 'qed', label: 'QED', kind: 'number', priority: 2, minPanelWidth: 420 });
  }
  if (poseRows.some((row) => row.metrics.xtbEnergyKcal != null)) {
    columns.push({ key: 'xtbEnergyKcal', label: 'xTB', kind: 'number', priority: 3, minPanelWidth: 500 });
  }

  const families: ViewerProjectFamily[] = [
    {
      id: familyId,
      title,
      jobType: 'docking',
      collapsed: false,
      rowIds: rows.map((row) => row.id),
      columns,
      sortKey: poseRows.some((row) => row.metrics.cordialPHighAffinity != null) ? 'cordialPHighAffinity' : 'vinaAffinity',
      sortDirection: poseRows.some((row) => row.metrics.cordialPHighAffinity != null) ? 'desc' : 'asc',
    },
  ];

  return {
    families,
    rows,
    activeRowId: poseRows[selectedQueueIndex]?.id ?? rows[0]?.id ?? null,
  };
}

export function buildConformerProjectTable(options: {
  familyId: string;
  title: string;
  inputPath?: string | null;
  conformerPaths: string[];
  conformerEnergies: Record<string, number>;
}): ViewerProjectTableState {
  const { familyId, title, inputPath, conformerPaths, conformerEnergies } = options;
  const rows: ViewerProjectRow[] = [];

  if (inputPath) {
    rows.push({
      id: `${familyId}:input`,
      familyId,
      label: 'Input molecule',
      rowKind: 'input',
      jobType: 'conformer',
      item: { pdbPath: inputPath, label: 'Input molecule', type: 'ligand' },
      loadKind: 'standalone-ligand',
      metrics: {},
    });
  }

  const conformerRows = conformerPaths.map<ViewerProjectRow>((sdfPath, index) => {
    let energy: number | null = null;
    if (sdfPath in conformerEnergies) {
      energy = conformerEnergies[sdfPath];
    } else {
      const match = Object.entries(conformerEnergies).find(([key]) => key.endsWith(`/${sdfPath.split('/').pop()}`));
      energy = match ? match[1] : null;
    }

    return {
      id: `${familyId}:conformer:${index}`,
      familyId,
      label: `Conformer ${index + 1}`,
      rowKind: 'conformer',
      jobType: 'conformer',
      item: { pdbPath: sdfPath, label: `Conformer ${index + 1}`, type: 'ligand' },
      loadKind: 'standalone-ligand',
      queueIndex: index,
      metrics: {
        relativeEnergy: energy,
      },
    };
  });

  rows.push(...conformerRows);

  return {
    families: [{
      id: familyId,
      title,
      jobType: 'conformer',
      collapsed: false,
      rowIds: rows.map((row) => row.id),
      columns: [{ key: 'relativeEnergy', label: 'Rel E', kind: 'number', priority: 0, minPanelWidth: 0 }],
      sortKey: 'relativeEnergy',
      sortDirection: 'asc',
    }],
    rows,
    activeRowId: conformerRows[0]?.id ?? rows[0]?.id ?? null,
  };
}

export function buildMdProjectTable(options: {
  familyId: string;
  title: string;
  systemPdb: string;
  trajectoryPath?: string | null;
  queueBackedClusters?: boolean;
  clusters: Array<{
    clusterId: number;
    population: number;
    centroidPdbPath?: string;
    vinaRescore?: number;
    cordialPHighAffinity?: number;
  }>;
}): ViewerProjectTableState {
  const { familyId, title, systemPdb, trajectoryPath, clusters, queueBackedClusters = true } = options;
  const rows: ViewerProjectRow[] = [
    {
      id: `${familyId}:initial-complex`,
      familyId,
      label: 'Initial complex',
      rowKind: 'initial-complex',
      jobType: 'simulation',
      item: { pdbPath: systemPdb, label: 'Initial complex' },
      loadKind: 'structure',
      metrics: {},
      trajectoryPath: trajectoryPath ?? null,
    },
  ];

  rows.push(...clusters
    .filter((cluster) => cluster.centroidPdbPath)
    .map<ViewerProjectRow>((cluster, index) => ({
      id: `${familyId}:cluster:${cluster.clusterId}`,
      familyId,
      label: `Cluster ${cluster.clusterId + 1}`,
      rowKind: 'cluster',
      jobType: 'simulation',
      item: {
        pdbPath: cluster.centroidPdbPath!,
        label: `Cluster ${cluster.clusterId + 1} (${cluster.population.toFixed(0)}%)`,
      },
      loadKind: queueBackedClusters ? 'queue' : 'structure',
      queueIndex: queueBackedClusters ? index : undefined,
      metrics: {
        population: cluster.population / 100,
        vinaRescore: cluster.vinaRescore ?? null,
        cordialPHighAffinity: percent(cluster.cordialPHighAffinity),
      },
    })));

  return {
    families: [{
      id: familyId,
      title,
      jobType: 'simulation',
      collapsed: false,
      rowIds: rows.map((row) => row.id),
      columns: [
        { key: 'population', label: 'Pop%', kind: 'percent', priority: 0, minPanelWidth: 0 },
        ...(rows.some((row) => row.metrics.vinaRescore != null)
          ? [{ key: 'vinaRescore', label: 'Vina', kind: 'number', priority: 1, minPanelWidth: 360 } satisfies ViewerProjectColumn]
          : []),
        ...(rows.some((row) => row.metrics.cordialPHighAffinity != null)
          ? [{ key: 'cordialPHighAffinity', label: 'P(<1uM)', kind: 'percent', priority: 2, minPanelWidth: 420 } satisfies ViewerProjectColumn]
          : []),
      ],
      sortKey: 'population',
      sortDirection: 'desc',
      trajectoryPath: trajectoryPath ?? null,
    }],
    rows,
    activeRowId: rows[0]?.id ?? null,
  };
}
