/**
 * Viewer UI controls tests.
 * Loads a holo complex (8TCE — protein + ligand), then exercises each viewer
 * control and verifies the NGL stage state changes accordingly.
 * NO file dialogs — uses direct store manipulation for loading.
 */
import { test, expect, createTestProject } from './fixtures';
import type { Page } from '@playwright/test';
import * as path from 'path';

const RECEPTOR_CIF = path.resolve(__dirname, '../../ember-test-protein/8tce.cif');

async function setupAndLoadComplex(window: Page): Promise<void> {
  await createTestProject(window, '__e2e_viewer_ctrl__');
  // Wait for NGL stage
  await window.waitForFunction(() => !!(window as any).__nglStage, null, { timeout: 10_000 });

  // Load 8TCE via store
  await window.evaluate(async (cif: string) => {
    const api = (window as any).electronAPI;
    const proj = await api.ensureProject('__e2e_viewer_ctrl__');
    const projDir = proj.ok ? proj.value : '';
    const imp = await api.importStructure(cif, projDir);
    const imported = imp.ok ? imp.value : cif;
    const store = (window as any).__emberStore;
    if (store) {
      store.resetViewer();
      store.setViewerPdbPath(imported);
      store.setMode('viewer');
    }
  }, RECEPTOR_CIF);

  // Wait for structure to load in NGL
  await window.waitForFunction(
    () => ((window as any).__nglStage?.compList?.length ?? 0) > 0,
    null,
    { timeout: 15_000 }
  );
  await window.waitForTimeout(1_000);
}

/** Get viewer store state */
async function getViewerState(window: Page): Promise<Record<string, any>> {
  return window.evaluate(() => {
    const store = (window as any).__emberStore;
    if (!store) return {};
    const s = store.state();
    return { ...s.viewer };
  });
}

/** Get NGL stage parameters */
async function getStageParams(window: Page): Promise<Record<string, any>> {
  return window.evaluate(() => {
    const stage = (window as any).__nglStage;
    if (!stage) return {};
    const p = stage.getParameters();
    return { clipDist: p.clipDist, fogNear: p.fogNear, fogFar: p.fogFar };
  });
}

/** Get all repr types across all components */
async function getAllReprTypes(window: Page): Promise<string[]> {
  return window.evaluate(() => {
    const stage = (window as any).__nglStage;
    if (!stage) return [];
    const types: string[] = [];
    for (const comp of stage.compList) {
      for (const r of comp.reprList) {
        types.push(r.repr?.type || 'unknown');
      }
    }
    return types;
  });
}

/** Count representations of a given type */
async function countReprOfType(window: Page, type: string): Promise<number> {
  return window.evaluate((t: string) => {
    const stage = (window as any).__nglStage;
    if (!stage) return 0;
    let count = 0;
    for (const comp of stage.compList) {
      for (const r of comp.reprList) {
        if (r.repr?.type === t) count++;
      }
    }
    return count;
  }, type);
}

test.describe('Viewer UI controls', () => {
  test.beforeEach(async ({ window }) => {
    test.setTimeout(60_000);
    await setupAndLoadComplex(window);
  });

  test('protein representation dropdown changes repr type', async ({ window }) => {
    // Default is cartoon — verify
    const vs = await getViewerState(window);
    expect(vs.proteinRep).toBe('cartoon');

    // Change to ribbon via the Protein dropdown
    const proteinSelect = window.locator('select').filter({ has: window.locator('option', { hasText: 'Cartoon' }) }).first();
    await proteinSelect.selectOption('ribbon');
    await window.waitForTimeout(1_000);

    // Store state should reflect change
    const vs2 = await getViewerState(window);
    expect(vs2.proteinRep).toBe('ribbon');

    // Change to spacefill
    await proteinSelect.selectOption('spacefill');
    await window.waitForTimeout(1_000);
    const vs3 = await getViewerState(window);
    expect(vs3.proteinRep).toBe('spacefill');
  });

  test('ligand representation dropdown changes repr type', async ({ window }) => {
    // Find ligand dropdown (has Ball+Stick option)
    const ligandSelect = window.locator('select').filter({ has: window.locator('option', { hasText: 'Ball+Stick' }) });
    await expect(ligandSelect).toBeVisible();

    await ligandSelect.selectOption('stick');
    await window.waitForTimeout(500);
    const vs = await getViewerState(window);
    expect(vs.ligandRep).toBe('stick');

    await ligandSelect.selectOption('spacefill');
    await window.waitForTimeout(500);
    const vs2 = await getViewerState(window);
    expect(vs2.ligandRep).toBe('spacefill');
  });

  test('protein surface toggle adds/removes surface representation', async ({ window }) => {
    // Surface should be off by default
    const vs = await getViewerState(window);
    expect(vs.proteinSurface).toBe(false);

    const surfaceBefore = await countReprOfType(window, 'surface');

    // Toggle surface on — find the Protein row's Surface checkbox
    const surfaceCheckbox = window.locator('label', { hasText: 'Surface' }).first().locator('input[type="checkbox"]');
    await surfaceCheckbox.check();
    await window.waitForTimeout(2_000);

    const vs2 = await getViewerState(window);
    expect(vs2.proteinSurface).toBe(true);

    const surfaceAfter = await countReprOfType(window, 'surface');
    expect(surfaceAfter).toBeGreaterThan(surfaceBefore);

    // Toggle off
    await surfaceCheckbox.uncheck();
    await window.waitForTimeout(1_000);
    const vs3 = await getViewerState(window);
    expect(vs3.proteinSurface).toBe(false);
  });

  test('electrostatic surface computation returns nonzero values', async ({ window }) => {
    const surfaceCheckbox = window.locator('label', { hasText: 'Surface' }).first().locator('input[type="checkbox"]');
    await surfaceCheckbox.check();
    await window.waitForTimeout(1_000);

    const surfaceSchemeSelect = window.locator('select').filter({
      has: window.locator('option', { hasText: 'Electrostatic' }),
    }).first();
    await surfaceSchemeSelect.selectOption('electrostatic');

    await window.waitForFunction(() => {
      const stage = (window as any).__nglStage;
      return Array.from(stage?.compList || []).some((comp: any) =>
        Array.from(comp.reprList || []).some((repr: any) => repr.repr?.type === 'surface')
      );
    }, null, { timeout: 15_000 });

    const surfaceProps = await window.evaluate(async () => {
      const api = (window as any).electronAPI;
      const pdbPath = (window as any).__emberStore.state().viewer.pdbPath;
      const outDir = `${(window as any).__emberStore.state().projectDir || '/tmp'}/surface-props-test`;
      const result = await api.computeSurfaceProps(pdbPath, outDir);
      if (!result.ok) return null;
      return {
        atomCount: result.value.atomCount,
        electrostaticNonZero: result.value.electrostatic.some((value: number) => Math.abs(value) > 1e-6),
      };
    });

    expect(surfaceProps).not.toBeNull();
    expect(surfaceProps!.atomCount).toBeGreaterThan(0);
    expect(surfaceProps!.electrostaticNonZero).toBe(true);
  });

  test('clipping plane slider changes NGL clipDist parameter', async ({ window }) => {
    const paramsBefore = await getStageParams(window);

    // Find the clipping slider (range input near "Clip" label)
    const clipSlider = window.locator('input[type="range"]');
    await expect(clipSlider).toBeVisible();

    // Set to a high value
    await clipSlider.fill('80');
    await window.waitForTimeout(500);

    const paramsAfter = await getStageParams(window);
    expect(paramsAfter.clipDist).not.toBe(paramsBefore.clipDist);
    expect(paramsAfter.clipDist).toBe(80);
  });

  test('pocket residues toggle changes store state', async ({ window }) => {
    // Pocket residues checkbox (next to "Show Pocket" text)
    const pocketCheckbox = window.locator('label', { hasText: 'Show Pocket' }).locator('input[type="checkbox"]');

    // Default: checked
    const vs = await getViewerState(window);
    const wasChecked = vs.showPocketResidues;

    // Toggle
    if (wasChecked) {
      await pocketCheckbox.uncheck();
    } else {
      await pocketCheckbox.check();
    }
    await window.waitForTimeout(1_000);

    const vs2 = await getViewerState(window);
    expect(vs2.showPocketResidues).toBe(!wasChecked);
  });

  test('hide waters/ions toggle changes store state', async ({ window }) => {
    const waterCheckbox = window.locator('label', { hasText: /Hide H₂O/i }).locator('input[type="checkbox"]');

    const vs = await getViewerState(window);
    const wasChecked = vs.hideWaterIons;

    // Toggle
    if (wasChecked) {
      await waterCheckbox.uncheck();
    } else {
      await waterCheckbox.check();
    }
    await window.waitForTimeout(500);

    const vs2 = await getViewerState(window);
    expect(vs2.hideWaterIons).toBe(!wasChecked);
  });

  test('interactions toggle changes store state', async ({ window }) => {
    const interCheckbox = window.locator('label', { hasText: 'Interactions' }).locator('input[type="checkbox"]');

    const vs = await getViewerState(window);
    const wasChecked = vs.showInteractions;

    if (wasChecked) {
      await interCheckbox.uncheck();
    } else {
      await interCheckbox.check();
    }
    await window.waitForTimeout(500);

    const vs2 = await getViewerState(window);
    expect(vs2.showInteractions).toBe(!wasChecked);
  });

  test('polar H toggle changes store state', async ({ window }) => {
    const polarCheckbox = window.locator('label', { hasText: 'Polar H' }).locator('input[type="checkbox"]');

    const vs = await getViewerState(window);
    const wasChecked = vs.ligandPolarHOnly;

    if (wasChecked) {
      await polarCheckbox.uncheck();
    } else {
      await polarCheckbox.check();
    }
    await window.waitForTimeout(500);

    const vs2 = await getViewerState(window);
    expect(vs2.ligandPolarHOnly).toBe(!wasChecked);
  });

  test('reset button resets clipping and fog parameters', async ({ window }) => {
    // Change clipping first
    const clipSlider = window.locator('input[type="range"]');
    await clipSlider.fill('90');
    await window.waitForTimeout(500);

    const paramsBefore = await getStageParams(window);
    expect(paramsBefore.clipDist).toBe(90);

    // Click Reset
    const resetBtn = window.locator('.btn', { hasText: 'Reset' });
    await resetBtn.click();
    await window.waitForTimeout(1_000);

    // clipDist should be restored (default is 10)
    const paramsAfter = await getStageParams(window);
    expect(paramsAfter.clipDist).not.toBe(90);
  });
});
