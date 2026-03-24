/**
 * MD simulation pipeline test.
 * Uses PDB ID fetch for receptor + SMILES for ligand (no file dialogs).
 */
import { test, expect, createTestProject } from './fixtures';
import type { Page } from '@playwright/test';

/** Navigate to MD configure page via PDB ID fetch → Continue */
async function navigateToConfigure(window: Page): Promise<void> {
  await fetchReceptor(window);
  const continueBtn = window.locator('.btn.btn-primary:visible', { hasText: /Continue/i });
  await expect(continueBtn).toBeEnabled({ timeout: 5_000 });
  await continueBtn.click();
  await window.waitForTimeout(1_000);
  await expect(window.locator('main').locator('text=/Configure/i').first()).toBeVisible({ timeout: 5_000 });
}

/** Fetch 8TCE via PDB ID, verifying no errors and structure loads */
async function fetchReceptor(window: Page): Promise<void> {
  const pdbInput = window.locator('input[placeholder*="8TCE"]:visible');
  await pdbInput.fill('8TCE');
  await window.locator('.btn.btn-primary.btn-sm:visible', { hasText: 'Fetch' }).click();

  // Verify no error after fetch
  await window.waitForTimeout(2_000);
  const errorAlert = window.locator('.alert.alert-error');
  if (await errorAlert.isVisible()) {
    const errorText = await errorAlert.textContent();
    throw new Error(`Unexpected error after PDB fetch: ${errorText}`);
  }

  // Wait for structure to load — "8TCE.cif" appears in main content area
  await expect(
    window.locator('main').locator('text=8TCE.cif')
  ).toBeVisible({ timeout: 30_000 });
}

async function getViewerTrajectoryRenderState(window: Page) {
  return window.evaluate(() => {
    const s = (window as any).__emberStore.state();
    const telemetry = (window as any).__viewerTestState ?? {};
    return {
      currentFrame: s.viewer.currentFrame,
      isPlaying: s.viewer.isPlaying,
      playbackSpeed: s.viewer.playbackSpeed,
      centerTarget: s.viewer.centerTarget,
      renderedFrameIndex: telemetry.renderedFrameIndex ?? null,
      coordinateSignature: telemetry.coordinateSignature ?? null,
    };
  });
}

test.describe('MD simulation pipeline', () => {
  test.beforeEach(async ({ window }) => {
    await createTestProject(window, '__e2e_md__');
    await window.locator('.tab.tab-sm', { hasText: 'Simulate' }).click();
    await window.waitForTimeout(500);
  });

  test('load PDB via PDB ID fetch', async ({ window }) => {
    test.setTimeout(60_000);

    await fetchReceptor(window);

    // Should show Protein + Ligand mode badge and Continue should be enabled
    await expect(window.locator('main .badge', { hasText: 'Protein + Ligand' })).toBeVisible();
    await expect(window.locator('.btn.btn-primary:visible', { hasText: /Continue/i })).toBeEnabled();
  });

  test('SMILES input sets ligand-only mode', async ({ window }) => {
    test.setTimeout(30_000);

    const textarea = window.locator('textarea:visible');
    await expect(textarea).toBeVisible();
    await textarea.fill('c1ccccc1');

    const enterBtn = window.locator('.btn.btn-primary.btn-sm:visible', { hasText: /Enter SMILES/i });
    await expect(enterBtn).toBeVisible();
    await enterBtn.click();

    // Should show "Ligand Only" mode indicator
    await expect(
      window.locator('text=/Ligand Only/i').first()
    ).toBeVisible({ timeout: 15_000 });
  });

  test('configure page shows force field preset', async ({ window }) => {
    test.setTimeout(90_000);

    // Fetch receptor — auto-detects ligand, Continue becomes enabled
    await fetchReceptor(window);
    const continueBtn = window.locator('.btn.btn-primary:visible', { hasText: /Continue/i });
    await expect(continueBtn).toBeEnabled({ timeout: 5_000 });
    await continueBtn.click();
    await window.waitForTimeout(1_000);

    // Should see Configure heading
    await expect(window.locator('main').locator('text=/Configure/i').first()).toBeVisible({ timeout: 5_000 });

    // Force field preset dropdown with ff19sb option
    const ffSelect = window.locator('select:visible').filter({
      has: window.locator('option', { hasText: /ff19sb/i }),
    });
    await expect(ffSelect).toBeVisible();
    const ffOptions = await ffSelect.locator('option').allTextContents();
    expect(ffOptions.some(o => o.toLowerCase().includes('ff19sb'))).toBe(true);
  });

  test('configure: production duration dial adjusts value', async ({ window }) => {
    test.setTimeout(90_000);
    await navigateToConfigure(window);

    // Read initial productionNs
    const initial = await window.evaluate(() => {
      const store = (window as any).__emberStore;
      return store.state().md.config.productionNs;
    });
    expect(initial).toBeGreaterThan(0);

    // Set a specific value via store API — tests reactive binding to SVG display
    await window.evaluate(() => {
      const store = (window as any).__emberStore;
      store.setMdConfig({ productionNs: 0.5 });
    });
    await window.waitForTimeout(300);

    // Verify store updated
    const updated = await window.evaluate(() => {
      return (window as any).__emberStore.state().md.config.productionNs;
    });
    expect(updated).toBe(0.5);

    // Verify SVG text shows "0.5" (the formatValue output for 0.5ns)
    const svgText = window.locator('svg[viewBox="-15 -15 170 170"] text').first();
    await expect(svgText).toBeVisible();
    const svgContent = await window.locator('svg[viewBox="-15 -15 170 170"]').textContent();
    expect(svgContent).toContain('0.5');

    // Also test edit mode: click center of SVG → type value → press Enter
    // The center text (invisible hit rect) is at ~center of 160x160 rendered SVG
    const svg = window.locator('svg[viewBox="-15 -15 170 170"]');
    await svg.click({ position: { x: 80, y: 70 } });
    await window.waitForTimeout(300);

    // Edit input should appear
    const editInput = window.locator('input[type="number"][min="0.1"][max="10000"]:visible');
    await expect(editInput).toBeVisible({ timeout: 2_000 });
    await editInput.fill('1');
    await editInput.press('Enter');
    await window.waitForTimeout(300);

    const afterEdit = await window.evaluate(() => {
      return (window as any).__emberStore.state().md.config.productionNs;
    });
    expect(afterEdit).toBe(1);
  });

  test('configure: temperature input works', async ({ window }) => {
    test.setTimeout(90_000);
    await navigateToConfigure(window);

    // Temperature input has min=200, max=500, step=10 — unique on this page
    const tempInput = window.locator('input[type="number"][min="200"][max="500"]:visible');
    await expect(tempInput).toBeVisible({ timeout: 5_000 });

    // Verify current value is 300 (default)
    const defaultVal = await tempInput.inputValue();
    expect(Number(defaultVal)).toBe(300);

    // Change to 310
    await tempInput.fill('310');
    await tempInput.dispatchEvent('input');
    await window.waitForTimeout(300);

    const stored = await window.evaluate(() => {
      return (window as any).__emberStore.state().md.config.temperatureK;
    });
    expect(stored).toBe(310);

    // Change back to 300
    await tempInput.fill('300');
    await tempInput.dispatchEvent('input');
    await window.waitForTimeout(300);
    const restored = await window.evaluate(() => {
      return (window as any).__emberStore.state().md.config.temperatureK;
    });
    expect(restored).toBe(300);
  });

  test('run simulation (0.01 ns ligand-only ibuprofen): progress → completion', async ({ window }) => {
    // Uses ligand-only mode to avoid _patched_createSystem ArgTracker bug
    // (build_ligand_only_system does NOT call _patch_forcefield_for_chain_breaks)
    // Ibuprofen (33 atoms) instead of benzene (12 atoms) — benzene solvation box
    // is too small for the 1.0 nm nonbonded cutoff with dodecahedron geometry
    test.setTimeout(300_000);

    // Ligand-only via SMILES — ibuprofen
    const textarea = window.locator('textarea:visible');
    await expect(textarea).toBeVisible();
    await textarea.fill('CC(C)Cc1ccc(cc1)C(C)C(=O)O');
    await window.locator('.btn.btn-primary.btn-sm:visible', { hasText: /Enter SMILES/i }).click();
    await expect(window.locator('text=/Ligand Only/i').first()).toBeVisible({ timeout: 15_000 });

    await window.locator('.btn.btn-primary:visible', { hasText: /Continue/i }).click();
    await window.waitForTimeout(1_000);
    await expect(window.locator('main').locator('text=/Configure/i').first()).toBeVisible({ timeout: 5_000 });

    // Set minimal duration (0.01 ns) and increase padding (1.5 nm) so the
    // dodecahedron box is large enough for the 1.0 nm nonbonded cutoff
    await window.evaluate(() => {
      const store = (window as any).__emberStore;
      store.setMdConfig({ productionNs: 0.01, paddingNm: 1.5 });
    });
    await window.waitForTimeout(300);

    // Start simulation
    await window.locator('.btn.btn-primary', { hasText: /Start Simulation/i }).click();
    await window.waitForTimeout(1_000);

    // Progress page should be visible
    await expect(window.locator('text=/Running MD Simulation/i')).toBeVisible({ timeout: 10_000 });

    // Wait for completion — "View Results" appears
    const viewResultsBtn = window.locator('.btn.btn-primary', { hasText: /View Results/i });
    await expect(viewResultsBtn).toBeVisible({ timeout: 250_000 });

    // Verify completion state, no error
    await expect(window.locator('text=/Simulation Complete/i')).toBeVisible();
    await expect(window.locator('.alert.alert-error')).not.toBeVisible();

    // Store: phase should be 'complete'
    const phase = await window.evaluate(() => {
      return (window as any).__emberStore.state().currentPhase;
    });
    expect(phase).toBe('complete');

    // --- Output file verification ---
    const outputFiles = await window.evaluate(async () => {
      const s = (window as any).__emberStore.state();
      const result = s.md.result;
      if (!result) return null;
      const api = (window as any).electronAPI;
      const trajDir = result.trajectoryPath.replace(/\/[^/]+$/, '');
      const runRoot = trajDir.endsWith('/results')
        ? trajDir.replace(/\/results$/, '')
        : trajDir;

      return {
        systemPdb: await api.fileExists(result.systemPdbPath),
        trajectory: await api.fileExists(result.trajectoryPath),
        energyCsv: await api.fileExists(`${runRoot}/energy.csv`),
        seedTxt: await api.fileExists(`${runRoot}/seed.txt`),
        finalPdb: await api.fileExists(`${runRoot}/final.pdb`),
        clusteringDir: await api.fileExists(`${runRoot}/analysis/clustering`),
        reportPdf: await api.fileExists(`${runRoot}/analysis/full_report.pdf`),
        rmsdDir: await api.fileExists(`${runRoot}/analysis/rmsd`),
        rmsfDir: await api.fileExists(`${runRoot}/analysis/rmsf`),
        hbondsDir: await api.fileExists(`${runRoot}/analysis/hbonds`),
        contactsDir: await api.fileExists(`${runRoot}/analysis/contacts`),
      };
    });
    expect(outputFiles).not.toBeNull();
    expect(outputFiles!.systemPdb).toBe(true);
    expect(outputFiles!.trajectory).toBe(true);
    expect(outputFiles!.energyCsv).toBe(true);
    expect(outputFiles!.seedTxt).toBe(true);
    expect(outputFiles!.finalPdb).toBe(true);
    expect(outputFiles!.clusteringDir).toBe(true);
    expect(outputFiles!.reportPdf).toBe(true);
    expect(outputFiles!.rmsdDir).toBe(true);
    expect(outputFiles!.rmsfDir).toBe(true);
    expect(outputFiles!.hbondsDir).toBe(true);
    expect(outputFiles!.contactsDir).toBe(true);

    // Navigate to results page
    await viewResultsBtn.click();
    await window.waitForTimeout(1_000);

    // Verify results page: "Simulation Complete" heading
    await expect(window.locator('h2', { hasText: /Simulation Complete/i })).toBeVisible({ timeout: 5_000 });

    // Clustering table should be present with Pop% column
    // (use .first() — torsion table may also be present)
    const table = window.locator('table.table').filter({
      has: window.locator('th', { hasText: 'Pop%' }),
    });
    await expect(table).toBeVisible({ timeout: 10_000 });
    await expect(table.locator('th', { hasText: 'Cluster' })).toBeVisible();

    // Table should have at least 1 data row with a population percentage
    const rows = table.locator('tbody tr');
    const rowCount = await rows.count();
    expect(rowCount).toBeGreaterThan(0);

    // First row should show a population percentage (N.N%)
    const firstRowPop = await rows.first().locator('td').nth(1).textContent();
    expect(firstRowPop).toMatch(/\d+\.\d+%/);

    // Populations should sum to ~100%
    const populations: number[] = [];
    for (let i = 0; i < rowCount; i++) {
      const text = await rows.nth(i).locator('td').nth(1).textContent();
      const val = parseFloat(text || '0');
      populations.push(val);
    }
    const totalPop = populations.reduce((a, b) => a + b, 0);
    expect(totalPop).toBeGreaterThan(95);
    expect(totalPop).toBeLessThanOrEqual(101);

    // Ligand-only mode: no Vina column expected
    await expect(table.locator('th', { hasText: 'Vina' })).not.toBeVisible();

    // "View All Clusters" button should exist
    await expect(window.locator('.btn', { hasText: /View All Clusters/i })).toBeVisible();

    // "Play Trajectory" button should exist
    await expect(window.locator('.btn', { hasText: /Play Trajectory/i })).toBeVisible();

    // --- Cluster row selection ---
    // Click first row → selected cluster detail panel should appear
    // Use dispatchEvent since parent card-body can intercept pointer events
    await rows.first().dispatchEvent('click');
    await window.waitForTimeout(500);

    // Selected cluster detail panel appears with "View 3D" button
    const view3dBtn = window.locator('.btn.btn-primary.btn-xs', { hasText: /View 3D/i });
    await expect(view3dBtn).toBeVisible({ timeout: 3_000 });

    // Verify selected cluster info text
    await expect(window.locator('text=/% of trajectory/')).toBeVisible();
    await expect(window.locator('text=/Cluster \\d+/').first()).toBeVisible();

    // Click "View 3D" → mode switches to viewer with pdbQueue
    await view3dBtn.dispatchEvent('click');
    await window.waitForTimeout(1_000);
    const viewerState = await window.evaluate(() => {
      const s = (window as any).__emberStore.state();
      return {
        mode: s.mode,
        pdbPath: s.viewer.pdbPath,
        pdbQueueLen: s.viewer.pdbQueue.length,
        trajectoryPath: s.viewer.trajectoryPath,
      };
    });
    expect(viewerState.mode).toBe('viewer');
    expect(viewerState.pdbPath).toBeTruthy();
    expect(viewerState.pdbQueueLen).toBeGreaterThan(0);
    // Cluster view should NOT have trajectory loaded
    expect(viewerState.trajectoryPath).toBeFalsy();

    // --- Report PDF verification ---
    // Navigate back to MD results to check report
    await window.locator('.tab.tab-sm', { hasText: 'Simulate' }).click();
    await window.waitForTimeout(500);

    // "Open Report" button should be visible
    await expect(window.locator('.btn', { hasText: /Open Report/i })).toBeVisible();

    // Verify full_report.pdf actually exists on disk
    const reportExists = await window.evaluate(async () => {
      const s = (window as any).__emberStore.state();
      const result = s.md.result;
      if (!result) return false;
      const trajDir = result.trajectoryPath.replace(/\/[^/]+$/, '');
      const runRoot = trajDir.endsWith('/results')
        ? trajDir.replace(/\/results$/, '')
        : trajDir;
      const reportPath = `${runRoot}/analysis/full_report.pdf`;
      return await (window as any).electronAPI.fileExists(reportPath);
    });
    expect(reportExists).toBe(true);

    // --- Torsion analysis panel ---
    // MDTorsionPanel should render on results page (ligand with rotatable bonds)
    const torsionTable = window.locator('table.table').filter({
      has: window.locator('th', { hasText: 'Torsion' }),
    });
    await expect(torsionTable).toBeVisible({ timeout: 3_000 });

    // Click "Play Trajectory" → mode switches to viewer with trajectory
    const playTrajBtn = window.locator('.btn', { hasText: /Play Trajectory/i });
    await playTrajBtn.click({ force: true });
    await window.waitForTimeout(1_000);
    const trajState = await window.evaluate(() => {
      const s = (window as any).__emberStore.state();
      return {
        mode: s.mode,
        pdbPath: s.viewer.pdbPath,
        trajectoryPath: s.viewer.trajectoryPath,
      };
    });
    expect(trajState.mode).toBe('viewer');
    expect(trajState.pdbPath).toBeTruthy();
    expect(trajState.trajectoryPath).toBeTruthy();

    // --- Trajectory controls verification ---
    await window.waitForFunction(() => {
      const s = (window as any).__emberStore.state();
      return !!s.viewer.trajectoryInfo && s.viewer.trajectoryInfo.frameCount > 0;
    }, null, { timeout: 20_000 });

    await expect(window.locator('[data-testid="trajectory-controls"]')).toBeVisible({ timeout: 10_000 });
    await window.waitForFunction(() => {
      const telemetry = (window as any).__viewerTestState;
      return telemetry?.renderedFrameIndex === 0 && Array.isArray(telemetry?.coordinateSignature);
    }, null, { timeout: 45_000 });

    const initialTrajInfo = await window.evaluate(() => {
      const s = (window as any).__emberStore.state();
      return {
        trajectoryPath: s.viewer.trajectoryPath,
        frameCount: s.viewer.trajectoryInfo?.frameCount ?? 0,
      };
    });
    expect(initialTrajInfo.trajectoryPath).toBeTruthy();
    expect(initialTrajInfo.frameCount).toBeGreaterThan(0);

    const initialRenderState = await getViewerTrajectoryRenderState(window);
    expect(initialRenderState.renderedFrameIndex).toBe(0);
    expect(initialRenderState.coordinateSignature).not.toBeNull();

    const playBtn = window.locator('[data-testid="trajectory-play"]');
    await playBtn.click();

    await window.waitForFunction(() => {
      const s = (window as any).__emberStore.state();
      const telemetry = (window as any).__viewerTestState;
      return s.viewer.isPlaying === true && telemetry?.renderedFrameIndex === s.viewer.currentFrame;
    }, null, { timeout: 20_000 });

    if (initialTrajInfo.frameCount > 1) {
      await window.waitForFunction(() => {
        const s = (window as any).__emberStore.state();
        const telemetry = (window as any).__viewerTestState;
        return s.viewer.currentFrame > 0 && telemetry?.renderedFrameIndex === s.viewer.currentFrame;
      }, null, { timeout: 20_000 });
    }

    const playedState = await getViewerTrajectoryRenderState(window);
    expect(playedState.isPlaying).toBe(true);
    if (initialTrajInfo.frameCount > 1) {
      expect(playedState.currentFrame).toBeGreaterThan(0);
      expect(playedState.renderedFrameIndex).toBe(playedState.currentFrame);
      expect(playedState.coordinateSignature).not.toEqual(initialRenderState.coordinateSignature);
    }

    await playBtn.click();
    await window.waitForFunction(() => !(window as any).__emberStore.state().viewer.isPlaying, null, { timeout: 10_000 });

    const pausedState = await getViewerTrajectoryRenderState(window);
    const pausedFrame = pausedState.currentFrame;

    if (initialTrajInfo.frameCount > 1) {
      await window.locator('[data-testid="trajectory-next"]').click();
      await window.waitForFunction((frame: number) => {
        const s = (window as any).__emberStore.state();
        const telemetry = (window as any).__viewerTestState;
        return s.viewer.currentFrame > frame && telemetry?.renderedFrameIndex === s.viewer.currentFrame;
      }, pausedFrame, { timeout: 10_000 });
      const steppedForward = await getViewerTrajectoryRenderState(window);
      const steppedForwardFrame = steppedForward.currentFrame;
      expect(steppedForwardFrame).toBeGreaterThan(pausedFrame);
      expect(steppedForward.coordinateSignature).not.toEqual(pausedState.coordinateSignature);

      await window.locator('[data-testid="trajectory-prev"]').click();
      await window.waitForFunction((frame: number) => {
        const s = (window as any).__emberStore.state();
        const telemetry = (window as any).__viewerTestState;
        return s.viewer.currentFrame < frame && telemetry?.renderedFrameIndex === s.viewer.currentFrame;
      }, steppedForwardFrame, { timeout: 10_000 });
      const steppedBackward = await getViewerTrajectoryRenderState(window);
      const steppedBackwardFrame = steppedBackward.currentFrame;
      expect(steppedBackwardFrame).toBeLessThan(steppedForwardFrame);
      expect(steppedBackward.coordinateSignature).not.toEqual(steppedForward.coordinateSignature);
    }

    await window.locator('[data-testid="trajectory-speed"]').selectOption('2');
    await window.waitForTimeout(300);
    const speedState = await getViewerTrajectoryRenderState(window);
    expect(speedState.playbackSpeed).toBe(2);

    await window.locator('[data-testid="trajectory-center-protein"]').click();
    await window.waitForTimeout(300);
    const centerTargetState = await getViewerTrajectoryRenderState(window);
    expect(centerTargetState.centerTarget).toBe('protein');

    await expect(window.locator('div[data-testid^="project-family-"]')).toBeVisible();
    await expect(window.locator('[data-testid^="project-row-"]', { hasText: /Initial complex/i })).toBeVisible();
    await expect(window.locator('.btn.btn-outline', { hasText: 'RMSD' })).toBeVisible();
    await expect(window.locator('.btn.btn-outline', { hasText: 'RMSF' })).toBeVisible();
    await expect(window.locator('.btn.btn-outline', { hasText: 'H-bonds' })).toBeVisible();
    await expect(window.locator('.btn.btn-outline', { hasText: 'Contacts' })).toBeVisible();

    await window.locator('[data-testid^="project-row-"]', { hasText: /Cluster 1/i }).click();
    await window.waitForTimeout(1_000);
    const projectTableNavState = await window.evaluate(() => {
      const s = (window as any).__emberStore.state();
      return {
        pdbPath: s.viewer.pdbPath,
        trajectoryPath: s.viewer.trajectoryPath,
      };
    });
    expect(projectTableNavState.pdbPath).toMatch(/cluster_0_centroid/i);
    expect(projectTableNavState.trajectoryPath).toBeFalsy();
  });

  test('configure: Estimate Runtime returns throughput and estimated runtime', async ({ window }) => {
    test.setTimeout(240_000);
    await navigateToConfigure(window);

    const benchmarkBtn = window.locator('.btn.btn-secondary:visible', { hasText: /Estimate Runtime/i });
    await expect(benchmarkBtn).toBeVisible({ timeout: 5_000 });
    await expect(benchmarkBtn).toBeEnabled();

    await benchmarkBtn.click();
    await expect(window.locator('[data-testid="benchmark-status"]')).toBeVisible({ timeout: 10_000 });

    const benchmarkOutcomeHandle = await window.waitForFunction(() => {
      const result = (window as any).__emberStore.state().md.benchmarkResult;
      const statusEl = document.querySelector('[data-testid="benchmark-status"]');
      const statusText = statusEl?.textContent ?? '';
      return !!result || /^Error:/i.test(statusText);
    }, null, { timeout: 220_000 });
    expect(await benchmarkOutcomeHandle.jsonValue()).toBe(true);

    const benchmarkOutcome = await window.evaluate(() => {
      const result = (window as any).__emberStore.state().md.benchmarkResult;
      const statusEl = document.querySelector('[data-testid="benchmark-status"]');
      return {
        result,
        statusText: statusEl?.textContent ?? '',
      };
    }) as {
      result: { nsPerDay: number; estimatedHours: number; systemInfo: { atomCount: number; boxVolumeA3: number } } | null;
      statusText: string;
    };

    expect(benchmarkOutcome.statusText).not.toMatch(/^Error:/i);
    expect(benchmarkOutcome.result).not.toBeNull();
    expect(benchmarkOutcome.result!.nsPerDay).toBeGreaterThan(0);
    expect(benchmarkOutcome.result!.estimatedHours).toBeGreaterThan(0);
    expect(benchmarkOutcome.result!.systemInfo.atomCount).toBeGreaterThan(0);
    expect(benchmarkOutcome.result!.systemInfo.boxVolumeA3).toBeGreaterThan(0);

    const benchmarkPanel = window.locator('[data-testid="benchmark-results"]');
    await expect(benchmarkPanel).toBeVisible({ timeout: 10_000 });
    await expect(benchmarkPanel).toContainText('Benchmark Results');
    await expect(benchmarkPanel).toContainText(/ns\/day/);
    await expect(benchmarkPanel).toContainText('Atom Count');
    await expect(benchmarkPanel).toContainText('Box Volume');
  });
});
