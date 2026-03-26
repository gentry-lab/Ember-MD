// Copyright (c) 2026 Ember Contributors. MIT License.
import { test, expect, createTestProject } from './fixtures';
import type { Page } from '@playwright/test';
import * as path from 'path';
import {
  buildConformerProjectTable,
  buildDockingProjectTable,
  buildDockingViewerQueue,
  buildMdProjectTable,
} from '../../src/utils/viewerQueue';

const FIXTURES = path.resolve(__dirname, '../fixtures');
const ALANINE_PDB = path.join(FIXTURES, 'alanine_dipeptide.pdb');
const BENZENE_SDF = path.join(FIXTURES, 'benzene.sdf');
const DOCK_RECEPTOR_CIF = path.resolve(__dirname, '../../ember-test-protein/8tce.cif');
const DOCK_REFERENCE_LIGAND = path.resolve(__dirname, '../../ember-test-protein/kiv/kiv.sdf');

async function setupViewer(window: Page): Promise<void> {
  await createTestProject(window, '__e2e_project_table__');
  const viewTab = window.locator('.tab.tab-sm', { hasText: 'View' });
  await expect(viewTab).toHaveClass(/tab-active/);
  await window.waitForFunction(() => !!(window as any).__nglStage, null, { timeout: 10_000 });
}

async function getViewerState(window: Page): Promise<Record<string, any>> {
  return window.evaluate(() => {
    const store = (window as any).__emberStore;
    return store ? { ...store.state().viewer } : {};
  });
}

async function countReprOfType(window: Page, type: string): Promise<number> {
  return window.evaluate((reprType: string) => {
    const stage = (window as any).__nglStage;
    if (!stage) return 0;
    let count = 0;
    for (const comp of stage.compList) {
      for (const repr of comp.reprList) {
        if (repr.repr?.type === reprType) count++;
      }
    }
    return count;
  }, type);
}

test.describe('Project table', () => {
  test.beforeEach(async ({ window }) => {
    await setupViewer(window);
  });

  test('responsive columns and resize update the NGL viewport without disturbing camera rotation', async ({ window }) => {
    test.setTimeout(60_000);

    const poses = [
      {
        ligandName: 'Pose A',
        vinaAffinity: -7.5,
        cordialPHighAffinity: 0.64,
        qed: 0.71,
        outputSdf: BENZENE_SDF,
      },
      {
        ligandName: 'Pose B',
        vinaAffinity: -6.8,
        cordialPHighAffinity: 0.42,
        qed: 0.66,
        outputSdf: BENZENE_SDF,
      },
    ] as any[];

    const queue = buildDockingViewerQueue(ALANINE_PDB, poses.map((pose) => ({
      name: pose.ligandName,
      path: pose.outputSdf,
      affinity: pose.vinaAffinity,
    })));

    const projectTable = buildDockingProjectTable({
      familyId: 'dock:test',
      title: 'Docking job',
      receptorPdb: ALANINE_PDB,
      poses: poses as any,
      poseQueue: queue,
      selectedQueueIndex: 0,
    });

    await window.evaluate((args: { pdbPath: string; ligandPath: string; queue: any[]; projectTable: any }) => {
      const store = (window as any).__emberStore;
      store.openViewerSession({
        pdbPath: args.pdbPath,
        ligandPath: args.ligandPath,
        pdbQueue: args.queue,
        pdbQueueIndex: 0,
        projectTable: args.projectTable,
      });
    }, {
      pdbPath: ALANINE_PDB,
      ligandPath: BENZENE_SDF,
      queue,
      projectTable,
    });

    await expect(window.locator('[data-testid="project-table"]')).toBeVisible({ timeout: 10_000 });
    await expect(window.locator('div[data-testid="project-family-dock:test"]')).toBeVisible();
    await expect(window.locator('[data-testid="project-table-nav-prev"]')).toBeVisible();
    await expect(window.locator('[data-testid="project-table-nav-next"]')).toBeVisible();
    await expect(window.locator('text=/\\(1\\/2\\)/')).toHaveCount(0);

    await expect(window.locator('th', { hasText: 'Vina' })).toBeVisible();
    await expect(window.locator('th', { hasText: 'P(<1uM)' })).not.toBeVisible();
    await expect(window.locator('th', { hasText: 'QED' })).not.toBeVisible();

    const before = await window.evaluate(() => {
      const stage = (window as any).__nglStage;
      return {
        width: stage?.viewer?.renderer?.domElement?.width ?? 0,
        rotation: stage?.viewerControls?.rotation?.toArray?.() ?? null,
      };
    });
    expect(before.width).toBeGreaterThan(0);
    expect(before.rotation).not.toBeNull();

    const handle = window.locator('[data-testid="project-table-resize-handle"]');
    const box = await handle.boundingBox();
    expect(box).not.toBeNull();
    if (!box) return;

    const startX = box.x + box.width / 2;
    const startY = box.y + box.height / 2;
    await window.mouse.move(startX, startY);
    await window.mouse.down();
    await window.mouse.move(startX - 180, startY, { steps: 12 });
    await window.waitForTimeout(300);

    const duringDrag = await window.evaluate(() => {
      const stage = (window as any).__nglStage;
      return {
        width: stage?.viewer?.renderer?.domElement?.width ?? 0,
        rotation: stage?.viewerControls?.rotation?.toArray?.() ?? null,
      };
    });

    expect(duringDrag.width).toBeLessThan(before.width);
    expect(duringDrag.rotation).toEqual(before.rotation);

    await window.mouse.up();
    await window.waitForTimeout(500);

    await expect(window.locator('th', { hasText: 'P(<1uM)' })).toBeVisible();
    await expect(window.locator('th', { hasText: 'QED' })).toBeVisible();
  });

  test('header arrows and keyboard navigation follow visible table order and restore queue-backed rows', async ({ window }) => {
    test.setTimeout(45_000);

    const poses = [
      {
        ligandName: 'Pose A',
        vinaAffinity: -7.5,
        outputSdf: BENZENE_SDF,
      },
      {
        ligandName: 'Pose B',
        vinaAffinity: -6.8,
        outputSdf: BENZENE_SDF,
      },
    ] as any[];

    const queue = buildDockingViewerQueue(ALANINE_PDB, poses.map((pose) => ({
      name: pose.ligandName,
      path: pose.outputSdf,
      affinity: pose.vinaAffinity,
    })));

    const projectTable = buildDockingProjectTable({
      familyId: 'dock:state',
      title: 'Docking state job',
      receptorPdb: ALANINE_PDB,
      preparedLigandPath: BENZENE_SDF,
      referenceLigandPath: BENZENE_SDF,
      poses: poses as any,
      poseQueue: queue,
      selectedQueueIndex: 0,
    });

    await window.evaluate((args: { pdbPath: string; ligandPath: string; queue: any[]; projectTable: any }) => {
      const store = (window as any).__emberStore;
      store.openViewerSession({
        pdbPath: args.pdbPath,
        ligandPath: args.ligandPath,
        pdbQueue: args.queue,
        pdbQueueIndex: 0,
        projectTable: args.projectTable,
      });
    }, {
      pdbPath: ALANINE_PDB,
      ligandPath: BENZENE_SDF,
      queue,
      projectTable,
    });

    await expect(window.locator('[data-testid="project-row-dock:state:apo"]')).toBeVisible({ timeout: 10_000 });
    await expect(window.locator('[data-testid="project-row-dock:state:prepared-ligand"]')).toBeVisible();

    await window.locator('[data-testid="project-table-nav-prev"]').click();
    await window.waitForFunction(() => {
      const s = (window as any).__emberStore.state();
      return s.viewer.projectTable?.activeRowId === 'dock:state:prepared-ligand' && s.viewer.pdbQueue.length === 0;
    }, null, { timeout: 10_000 });

    const afterPreparedLigand = await window.evaluate(() => {
      const s = (window as any).__emberStore.state();
      return {
        activeRowId: s.viewer.projectTable?.activeRowId,
        queueLength: s.viewer.pdbQueue.length,
        trajectoryPath: s.viewer.trajectoryPath,
      };
    });
    expect(afterPreparedLigand.activeRowId).toBe('dock:state:prepared-ligand');
    expect(afterPreparedLigand.queueLength).toBe(0);
    expect(afterPreparedLigand.trajectoryPath).toBeFalsy();

    await window.keyboard.press('ArrowUp');
    await window.waitForFunction(() => {
      const s = (window as any).__emberStore.state();
      return s.viewer.projectTable?.activeRowId === 'dock:state:apo' && s.viewer.pdbQueue.length === 0;
    }, null, { timeout: 10_000 });

    await window.keyboard.press('ArrowRight');
    await window.keyboard.press('ArrowDown');
    await window.keyboard.press('ArrowDown');
    await window.waitForFunction(() => {
      const s = (window as any).__emberStore.state();
      return s.viewer.projectTable?.activeRowId === 'dock:state:pose:1'
        && s.viewer.pdbQueue.length === 2
        && s.viewer.pdbQueueIndex === 1;
    }, null, { timeout: 10_000 });

    const afterPose = await window.evaluate(() => {
      const s = (window as any).__emberStore.state();
      return {
        activeRowId: s.viewer.projectTable?.activeRowId,
        queueLength: s.viewer.pdbQueue.length,
        queueIndex: s.viewer.pdbQueueIndex,
      };
    });
    expect(afterPose.activeRowId).toBe('dock:state:pose:1');
    expect(afterPose.queueLength).toBe(2);
    expect(afterPose.queueIndex).toBe(1);
  });

  test('footer actions and close button are relocated into the new project-table layout', async ({ window }) => {
    test.setTimeout(45_000);

    const poses = [
      {
        ligandName: 'Pose A',
        vinaAffinity: -7.5,
        outputSdf: BENZENE_SDF,
      },
      {
        ligandName: 'Pose B',
        vinaAffinity: -6.8,
        outputSdf: BENZENE_SDF,
      },
    ] as any[];

    const queue = buildDockingViewerQueue(ALANINE_PDB, poses.map((pose) => ({
      name: pose.ligandName,
      path: pose.outputSdf,
      affinity: pose.vinaAffinity,
    })));

    const projectTable = buildDockingProjectTable({
      familyId: 'dock:layout',
      title: 'Docking layout job',
      receptorPdb: ALANINE_PDB,
      preparedLigandPath: BENZENE_SDF,
      referenceLigandPath: BENZENE_SDF,
      poses: poses as any,
      poseQueue: queue,
      selectedQueueIndex: 0,
    });

    await window.evaluate((args: { pdbPath: string; ligandPath: string; queue: any[]; projectTable: any }) => {
      const store = (window as any).__emberStore;
      store.openViewerSession({
        pdbPath: args.pdbPath,
        ligandPath: args.ligandPath,
        pdbQueue: args.queue,
        pdbQueueIndex: 0,
        projectTable: args.projectTable,
      });
    }, {
      pdbPath: ALANINE_PDB,
      ligandPath: BENZENE_SDF,
      queue,
      projectTable,
    });

    await expect(window.locator('[data-testid="viewer-close-button"]')).toBeVisible({ timeout: 10_000 });
    await expect(window.locator('[data-testid="project-table-transfer"]')).toBeVisible();
    await expect(window.locator('[data-testid="project-table-export"]')).toBeVisible();
    await expect(window.locator('[title=\"Simulate — run MD on this structure\"]')).toHaveCount(0);
    await expect(window.locator('[title=\"Export as PDB\"]')).toHaveCount(0);

    await expect(window.locator('[data-testid="project-table-transfer"]')).toBeEnabled();
    await expect(window.locator('[data-testid="project-table-export"]')).toBeEnabled();

    await window.locator('[data-testid="project-row-dock:layout:prepared-ligand"]').click();
    await window.waitForFunction(() => {
      const s = (window as any).__emberStore.state();
      return s.viewer.projectTable?.activeRowId === 'dock:layout:prepared-ligand';
    }, null, { timeout: 10_000 });

    await expect(window.locator('[data-testid="project-table-transfer"]')).toBeEnabled();
    await expect(window.locator('[data-testid="project-table-export"]')).toBeEnabled();
  });

  test('apo row keeps pocket residues available and prepared ligand uses the same ligand styling as docked poses', async ({ window }) => {
    test.setTimeout(90_000);

    const preparedReceptor = await window.evaluate(async ({ receptorCif, projectName }: { receptorCif: string; projectName: string }) => {
      const api = (window as any).electronAPI;
      const project = await api.ensureProject(projectName);
      if (!project.ok) return null;
      const imported = await api.importStructure(receptorCif, project.value);
      const importedPath = imported.ok ? imported.value : receptorCif;
      const detected = await api.detectPdbLigands(importedPath);
      if (!detected.ok) return null;
      const ligands = Array.isArray(detected.value) ? detected.value : detected.value.ligands;
      const ligandId = ligands[0]?.id;
      if (!ligandId) return null;
      const outputPath = `${project.value}/structures/apo_receptor.pdb`;
      const prepared = await api.prepareReceptor(importedPath, ligandId, outputPath);
      return prepared.ok ? prepared.value : null;
    }, { receptorCif: DOCK_RECEPTOR_CIF, projectName: '__e2e_project_table__' });

    expect(preparedReceptor).not.toBeNull();
    if (!preparedReceptor) return;

    const poses = [
      {
        ligandName: 'Reference pose',
        vinaAffinity: -8.2,
        outputSdf: DOCK_REFERENCE_LIGAND,
      },
    ] as any[];

    const queue = buildDockingViewerQueue(preparedReceptor, poses.map((pose) => ({
      name: pose.ligandName,
      path: pose.outputSdf,
      affinity: pose.vinaAffinity,
    })));

    const projectTable = buildDockingProjectTable({
      familyId: 'dock:apo',
      title: 'Docking apo job',
      receptorPdb: preparedReceptor,
      preparedLigandPath: DOCK_REFERENCE_LIGAND,
      referenceLigandPath: DOCK_REFERENCE_LIGAND,
      poses: poses as any,
      poseQueue: queue,
      selectedQueueIndex: 0,
    });

    await window.evaluate((args: { pdbPath: string; ligandPath: string; queue: any[]; projectTable: any }) => {
      const store = (window as any).__emberStore;
      store.openViewerSession({
        pdbPath: args.pdbPath,
        ligandPath: args.ligandPath,
        pdbQueue: args.queue,
        pdbQueueIndex: 0,
        projectTable: args.projectTable,
      });
    }, {
      pdbPath: preparedReceptor,
      ligandPath: DOCK_REFERENCE_LIGAND,
      queue,
      projectTable,
    });

    await window.locator('[data-testid="project-row-dock:apo:apo"]').click();
    await window.waitForFunction(() => {
      const s = (window as any).__emberStore.state();
      return s.viewer.projectTable?.activeRowId === 'dock:apo:apo' && s.viewer.showPocketResidues === true;
    }, null, { timeout: 10_000 });

    await window.waitForTimeout(1_500);
    expect(await countReprOfType(window, 'licorice')).toBeGreaterThan(0);

    await window.evaluate(() => {
      const store = (window as any).__emberStore;
      store.setViewerLigandRep('spacefill');
    });
    await window.locator('[data-testid="project-row-dock:apo:prepared-ligand"]').click();
    await window.waitForFunction(() => {
      const s = (window as any).__emberStore.state();
      return s.viewer.projectTable?.activeRowId === 'dock:apo:prepared-ligand'
        && s.viewer.ligandRep === 'spacefill'
        && s.viewer.detectedLigands.length === 0;
    }, null, { timeout: 10_000 });

    await window.waitForTimeout(1_000);
    const viewerState = await getViewerState(window);
    expect(viewerState.ligandPath).toContain('kiv');
    expect(await countReprOfType(window, 'spacefill')).toBeGreaterThan(0);
  });

  test('result columns are data-driven and pinned rows stay at the top', async ({ window }) => {
    test.setTimeout(45_000);

    const conformerTable = buildConformerProjectTable({
      familyId: 'conform:test',
      title: 'Conformer job',
      inputPath: BENZENE_SDF,
      conformerPaths: [BENZENE_SDF, BENZENE_SDF],
      conformerEnergies: {
        [BENZENE_SDF]: 0.5,
      },
    });

    await window.evaluate((args: { pdbPath: string; projectTable: any }) => {
      const store = (window as any).__emberStore;
      store.openViewerSession({
        pdbPath: args.pdbPath,
        pdbQueue: [
          { pdbPath: args.pdbPath, label: 'Conformer 1', type: 'conformer' },
          { pdbPath: args.pdbPath, label: 'Conformer 2', type: 'conformer' },
        ],
        pdbQueueIndex: 0,
        projectTable: args.projectTable,
      });
    }, { pdbPath: BENZENE_SDF, projectTable: conformerTable });

    await expect(window.locator('th', { hasText: 'Rel E' })).toBeVisible({ timeout: 10_000 });
    await expect(window.locator('[data-testid="project-row-conform:test:input"]')).toBeVisible();

    const firstConformerRow = await window.locator('[data-testid="project-table"] tbody tr').first().textContent();
    expect(firstConformerRow).toContain('Input molecule');

    const mdTable = buildMdProjectTable({
      familyId: 'md:test',
      title: 'MD job',
      systemPdb: ALANINE_PDB,
      clusters: [{
        clusterId: 0,
        population: 100,
        centroidPdbPath: ALANINE_PDB,
      }],
      queueBackedClusters: false,
    });

    await window.evaluate((args: { pdbPath: string; projectTable: any }) => {
      const store = (window as any).__emberStore;
      store.openViewerSession({
        pdbPath: args.pdbPath,
        projectTable: args.projectTable,
      });
    }, { pdbPath: ALANINE_PDB, projectTable: mdTable });

    await expect(window.locator('th', { hasText: 'Pop%' })).toBeVisible({ timeout: 10_000 });
    await expect(window.locator('th', { hasText: 'Vina' })).not.toBeVisible();
    await expect(window.locator('th', { hasText: 'P(<1uM)' })).not.toBeVisible();
  });

  test('cmd+click multi-selects rows and updates selectedRowIds', async ({ window }) => {
    test.setTimeout(30_000);

    const poses = [
      { ligandName: 'Pose A', vinaAffinity: -7.5, outputSdf: BENZENE_SDF },
      { ligandName: 'Pose B', vinaAffinity: -6.8, outputSdf: BENZENE_SDF },
      { ligandName: 'Pose C', vinaAffinity: -5.2, outputSdf: BENZENE_SDF },
    ] as any[];

    const queue = buildDockingViewerQueue(ALANINE_PDB, poses.map((p) => ({
      name: p.ligandName, path: p.outputSdf, affinity: p.vinaAffinity,
    })));

    const projectTable = buildDockingProjectTable({
      familyId: 'dock:multi',
      title: 'Multi-select test',
      receptorPdb: ALANINE_PDB,
      poses: poses as any,
      poseQueue: queue,
      selectedQueueIndex: 0,
    });

    await window.evaluate((args: any) => {
      const store = (window as any).__emberStore;
      store.openViewerSession({
        pdbPath: args.pdbPath,
        ligandPath: args.ligandPath,
        pdbQueue: args.queue,
        pdbQueueIndex: 0,
        projectTable: args.projectTable,
      });
    }, { pdbPath: ALANINE_PDB, ligandPath: BENZENE_SDF, queue, projectTable });

    await expect(window.locator('[data-testid="project-table"]')).toBeVisible({ timeout: 10_000 });

    // Initially, one row should be selected
    const initialState = await window.evaluate(() => {
      const s = (window as any).__emberStore.state();
      return s.viewer.projectTable?.selectedRowIds ?? [];
    });
    expect(initialState.length).toBe(1);

    // Cmd+click on Pose B row
    const poseBRow = window.locator('[data-testid="project-row-dock:multi:pose:1"]');
    await poseBRow.click({ modifiers: ['Meta'] });
    await window.waitForTimeout(300);

    const afterCmdClick = await window.evaluate(() => {
      const s = (window as any).__emberStore.state();
      return {
        selectedRowIds: s.viewer.projectTable?.selectedRowIds ?? [],
        activeRowId: s.viewer.projectTable?.activeRowId,
      };
    });
    expect(afterCmdClick.selectedRowIds.length).toBe(2);
    expect(afterCmdClick.selectedRowIds).toContain('dock:multi:pose:1');

    // Cmd+click on Pose C to add a third
    const poseCRow = window.locator('[data-testid="project-row-dock:multi:pose:2"]');
    await poseCRow.click({ modifiers: ['Meta'] });
    await window.waitForTimeout(300);

    const afterSecondCmd = await window.evaluate(() => {
      const s = (window as any).__emberStore.state();
      return s.viewer.projectTable?.selectedRowIds ?? [];
    });
    expect(afterSecondCmd.length).toBe(3);

    // Regular click resets to single selection
    await poseBRow.click();
    await window.waitForTimeout(300);

    const afterRegularClick = await window.evaluate(() => {
      const s = (window as any).__emberStore.state();
      return {
        selectedRowIds: s.viewer.projectTable?.selectedRowIds ?? [],
        activeRowId: s.viewer.projectTable?.activeRowId,
      };
    });
    expect(afterRegularClick.selectedRowIds.length).toBe(1);
    expect(afterRegularClick.activeRowId).toBe('dock:multi:pose:1');
  });

  test('alignment toolbar appears when multiple rows selected and P button aligns proteins', async ({ window }) => {
    test.setTimeout(30_000);

    const projectTable = buildMdProjectTable({
      familyId: 'md:align',
      title: 'Align test',
      systemPdb: ALANINE_PDB,
      clusters: [
        { clusterId: 0, population: 60, centroidPdbPath: ALANINE_PDB },
        { clusterId: 1, population: 40, centroidPdbPath: ALANINE_PDB },
      ],
    });

    const queue = [
      { pdbPath: ALANINE_PDB, label: 'Cluster 1 (60%)' },
      { pdbPath: ALANINE_PDB, label: 'Cluster 2 (40%)' },
    ];

    await window.evaluate((args: any) => {
      const store = (window as any).__emberStore;
      store.openViewerSession({
        pdbPath: args.pdbPath,
        pdbQueue: args.queue,
        pdbQueueIndex: 0,
        projectTable: args.projectTable,
      });
    }, { pdbPath: ALANINE_PDB, queue, projectTable });

    await expect(window.locator('[data-testid="project-table"]')).toBeVisible({ timeout: 10_000 });

    // Initially P button should not be visible (only 1 row selected)
    const alignPBtn = window.locator('[data-testid="project-table-align-protein"]');

    // Cmd+click second protein row to multi-select
    const initialRow = window.locator('[data-testid="project-row-md:align:initial-complex"]');
    await initialRow.click();
    await window.waitForTimeout(300);

    const clusterRow = window.locator('[data-testid="project-row-md:align:cluster:0"]');
    await clusterRow.click({ modifiers: ['Meta'] });
    await window.waitForTimeout(300);

    // Both rows selected → alignment toolbar should appear, P button enabled
    await expect(alignPBtn).toBeVisible({ timeout: 5_000 });
    await expect(alignPBtn).toBeEnabled();

    // L and SS should be disabled (these are proteins, not ligands)
    const alignLBtn = window.locator('[data-testid="project-table-align-ligand"]');
    await expect(alignLBtn).toBeDisabled();
  });

  test('cumulative project table: import creates a family, remove button deletes it', async ({ window }) => {
    test.setTimeout(30_000);

    // Open a viewer session first, then add an import family
    await window.evaluate((args: any) => {
      const store = (window as any).__emberStore;
      store.openViewerSession({ pdbPath: args.pdbPath });
      store.addViewerProjectFamily(
        {
          id: 'import:test',
          title: 'test_structure.pdb',
          jobType: 'import',
          collapsed: false,
          rowIds: ['import:test:0'],
          columns: [],
        },
        [{
          id: 'import:test:0',
          familyId: 'import:test',
          label: 'test_structure.pdb',
          rowKind: 'apo',
          jobType: 'import',
          item: { pdbPath: args.pdbPath, label: 'test_structure.pdb' },
          loadKind: 'structure',
          metrics: {},
        }],
      );
    }, { pdbPath: ALANINE_PDB });

    // Project table should show the import family
    await expect(window.locator('[data-testid="project-family-import:test"]')).toBeVisible({ timeout: 5_000 });
    await expect(window.locator('[data-testid="project-row-import:test:0"]')).toBeVisible();

    // Click the remove button
    const removeBtn = window.locator('[data-testid="project-family-remove-import:test"]');
    await expect(removeBtn).toBeVisible();
    await removeBtn.click();
    await window.waitForTimeout(300);

    // Family should be gone
    const tableState = await window.evaluate(() => {
      const s = (window as any).__emberStore.state();
      return s.viewer.projectTable;
    });
    expect(tableState).toBeNull();
  });

  test('alignment toolbar is always visible with buttons disabled until selection conditions met', async ({ window }) => {
    test.setTimeout(30_000);

    // Set up a docking job with apo + ligand rows
    const poses = [
      { ligandName: 'Pose A', vinaAffinity: -7.5, outputSdf: BENZENE_SDF },
    ] as any[];
    const queue = buildDockingViewerQueue(ALANINE_PDB, poses.map((p) => ({
      name: p.ligandName, path: p.outputSdf, affinity: p.vinaAffinity,
    })));
    const projectTable = buildDockingProjectTable({
      familyId: 'dock:align-test',
      title: 'Align visibility test',
      receptorPdb: ALANINE_PDB,
      poses: poses as any,
      poseQueue: queue,
      selectedQueueIndex: 0,
    });

    await window.evaluate((args: any) => {
      const store = (window as any).__emberStore;
      store.openViewerSession({
        pdbPath: args.pdbPath,
        ligandPath: args.ligandPath,
        pdbQueue: args.queue,
        pdbQueueIndex: 0,
        projectTable: args.projectTable,
      });
    }, { pdbPath: ALANINE_PDB, ligandPath: BENZENE_SDF, queue, projectTable });

    // Alignment toolbar always visible
    await expect(window.locator('[data-testid="project-table-align-protein"]')).toBeVisible({ timeout: 10_000 });
    await expect(window.locator('[data-testid="project-table-align-ligand"]')).toBeVisible();
    await expect(window.locator('[data-testid="project-table-align-substructure"]')).toBeVisible();

    // All disabled with single selection
    await expect(window.locator('[data-testid="project-table-align-protein"]')).toBeDisabled();
    await expect(window.locator('[data-testid="project-table-align-ligand"]')).toBeDisabled();
    await expect(window.locator('[data-testid="project-table-align-substructure"]')).toBeDisabled();
  });

  test('multiple imports accumulate in one Imports family', async ({ window }) => {
    test.setTimeout(30_000);

    await window.evaluate((args: any) => {
      const store = (window as any).__emberStore;
      store.openViewerSession({ pdbPath: args.pdbPath });

      // First import
      store.addViewerProjectFamily(
        { id: 'imports', title: 'Imports', jobType: 'import', collapsed: false, rowIds: ['imports:first'], columns: [] },
        [{ id: 'imports:first', familyId: 'imports', label: 'first.pdb', rowKind: 'apo', jobType: 'import',
           item: { pdbPath: args.pdbPath, label: 'first.pdb' }, loadKind: 'structure', metrics: {} }],
      );

      // Second import — should merge into existing Imports family
      store.addViewerProjectFamily(
        { id: 'imports', title: 'Imports', jobType: 'import', collapsed: false, rowIds: ['imports:second'], columns: [] },
        [{ id: 'imports:second', familyId: 'imports', label: 'second.sdf', rowKind: 'ligand', jobType: 'import',
           item: { pdbPath: args.ligandPath, label: 'second.sdf', type: 'ligand' }, loadKind: 'standalone-ligand', metrics: {} }],
      );
    }, { pdbPath: ALANINE_PDB, ligandPath: BENZENE_SDF });

    // Should be one family with 2 rows
    const tableState = await window.evaluate(() => {
      const s = (window as any).__emberStore.state();
      return {
        familyCount: s.viewer.projectTable?.families.length ?? 0,
        rowCount: s.viewer.projectTable?.rows.length ?? 0,
        familyTitle: s.viewer.projectTable?.families[0]?.title,
        rowIds: s.viewer.projectTable?.families[0]?.rowIds,
      };
    });
    expect(tableState.familyCount).toBe(1);
    expect(tableState.rowCount).toBe(2);
    expect(tableState.familyTitle).toBe('Imports');
    expect(tableState.rowIds).toContain('imports:first');
    expect(tableState.rowIds).toContain('imports:second');

    // Both rows visible in the table
    await expect(window.locator('[data-testid="project-row-imports:first"]')).toBeVisible({ timeout: 5_000 });
    await expect(window.locator('[data-testid="project-row-imports:second"]')).toBeVisible();
  });

  test('transfer dropdown routes ligand to MCMM with single selection', async ({ window }) => {
    test.setTimeout(30_000);

    await window.evaluate((args: any) => {
      const store = (window as any).__emberStore;
      store.openViewerSession({ pdbPath: args.ligandPath });
      store.addViewerProjectFamily(
        { id: 'imports', title: 'Imports', jobType: 'import', collapsed: false, rowIds: ['imports:lig'], columns: [] },
        [{ id: 'imports:lig', familyId: 'imports', label: 'benzene.sdf', rowKind: 'ligand', jobType: 'import',
           item: { pdbPath: args.ligandPath, label: 'benzene.sdf', type: 'ligand' }, loadKind: 'standalone-ligand', metrics: {} }],
      );
    }, { ligandPath: BENZENE_SDF });

    await expect(window.locator('[data-testid="project-table"]')).toBeVisible({ timeout: 10_000 });

    // Transfer should be enabled with single selection
    const transferBtn = window.locator('[data-testid="project-table-transfer"]');
    await expect(transferBtn).toBeVisible();

    // Click Transfer dropdown and select MCMM
    await transferBtn.click();
    await window.waitForTimeout(300);
    await expect(window.locator('[data-testid="project-table-transfer-dock"]')).toBeEnabled();
    await expect(window.locator('[data-testid="project-table-transfer-mcmm"]')).toBeEnabled();
    await expect(window.locator('[data-testid="project-table-transfer-simulate"]')).toBeDisabled();
    const mcmmOption = window.locator('[data-testid="project-table-transfer-mcmm"]');
    await expect(mcmmOption).toBeVisible({ timeout: 3_000 });
    await mcmmOption.click();
    await window.waitForTimeout(500);

    // Should switch to MCMM mode
    const mode = await window.evaluate(() => {
      const s = (window as any).__emberStore.state();
      return s.mode;
    });
    expect(mode).toBe('conform');
  });

  test('transfer dropdown disables MCMM for protein and protein-ligand complex rows', async ({ window }) => {
    test.setTimeout(30_000);

    const projectTable = buildMdProjectTable({
      familyId: 'md:test',
      title: 'MD job',
      systemPdb: ALANINE_PDB,
      clusters: [],
    });

    await window.evaluate((args: { pdbPath: string; projectTable: any }) => {
      const store = (window as any).__emberStore;
      store.openViewerSession({
        pdbPath: args.pdbPath,
        projectTable: args.projectTable,
      });
    }, {
      pdbPath: ALANINE_PDB,
      projectTable,
    });

    await expect(window.locator('[data-testid="project-table"]')).toBeVisible({ timeout: 10_000 });

    const transferBtn = window.locator('[data-testid="project-table-transfer"]');
    await transferBtn.click();
    await window.waitForTimeout(300);

    await expect(window.locator('[data-testid="project-table-transfer-dock"]')).toBeEnabled();
    await expect(window.locator('[data-testid="project-table-transfer-mcmm"]')).toBeDisabled();
    await expect(window.locator('[data-testid="project-table-transfer-simulate"]')).toBeEnabled();

    await window.locator('[data-testid="project-table-transfer-mcmm"]').click({ force: true });
    await window.waitForTimeout(300);

    const mode = await window.evaluate(() => {
      const s = (window as any).__emberStore.state();
      return s.mode;
    });
    expect(mode).toBe('viewer');
  });
});
