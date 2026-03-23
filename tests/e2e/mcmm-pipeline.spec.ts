/**
 * MCMM (conformer generation) full pipeline test.
 * Uses SMILES input (no file dialogs needed).
 */
import { test, expect, createTestProject } from './fixtures';
import type { Page } from '@playwright/test';

/** Load aspirin via SMILES and navigate to configure page */
async function loadAndConfigure(window: Page): Promise<void> {
  await window.locator('textarea').fill('CC(=O)Oc1ccccc1C(=O)O');
  await window.locator('.btn.btn-primary.btn-sm', { hasText: /Enter SMILES/i }).click();
  await expect(window.locator('.btn.btn-primary', { hasText: /Continue/i })).toBeEnabled({ timeout: 15_000 });
  await window.locator('.btn.btn-primary', { hasText: /Continue/i }).click();
  await window.waitForTimeout(500);
}

/** Locator for the conformer method dropdown (unique by its ETKDG option) */
function methodDropdown(window: Page) {
  return window.locator('select').filter({ has: window.locator('option', { hasText: 'ETKDG' }) });
}

test.describe('MCMM pipeline', () => {
  test.beforeEach(async ({ window }) => {
    await createTestProject(window, '__e2e_mcmm__');
    await window.locator('.tab.tab-sm', { hasText: 'MCMM' }).click();
    await window.waitForTimeout(500);
  });

  test('load ligand via SMILES', async ({ window }) => {
    test.setTimeout(30_000);

    await window.locator('textarea').fill('CC(=O)Oc1ccccc1C(=O)O');
    await window.locator('.btn.btn-primary.btn-sm', { hasText: /Enter SMILES/i }).click();
    await expect(window.locator('.btn.btn-primary', { hasText: /Continue/i })).toBeEnabled({ timeout: 15_000 });
  });

  test('configure shows method dropdown with ETKDG, MCMM, CREST', async ({ window }) => {
    test.setTimeout(30_000);
    await loadAndConfigure(window);

    const dd = methodDropdown(window);
    await expect(dd).toBeVisible();

    const options = await dd.locator('option').allTextContents();
    const lower = options.map(o => o.toLowerCase());
    expect(lower).toContain('etkdg');
    expect(lower).toContain('mcmm');
    expect(lower).toContain('crest');
  });

  test('MCMM-specific controls appear when method=mcmm', async ({ window }) => {
    test.setTimeout(30_000);
    await loadAndConfigure(window);

    // Default method is mcmm — verify MCMM-specific controls are shown
    const dd = methodDropdown(window);
    await dd.selectOption('mcmm');
    await window.waitForTimeout(300);

    // MCMM-specific: Search steps, Temperature, amide toggle
    await expect(window.locator('text=Search steps')).toBeVisible();
    await expect(window.locator('text=Temperature (K)')).toBeVisible();
    await expect(window.locator('text=Sample amide cis/trans')).toBeVisible();
    await expect(window.locator('text=MCMM uses OpenFF Sage')).toBeVisible();

    // Switch to ETKDG — MCMM controls should disappear
    await dd.selectOption('etkdg');
    await window.waitForTimeout(300);
    await expect(window.locator('text=Search steps')).not.toBeVisible();
    await expect(window.locator('text=Temperature (K)')).not.toBeVisible();
  });

  test('CREST-specific info appears when method=crest', async ({ window }) => {
    test.setTimeout(30_000);
    await loadAndConfigure(window);

    const dd = methodDropdown(window);
    await dd.selectOption('crest');
    await window.waitForTimeout(300);

    await expect(window.locator('text=CREST uses GFN2-xTB metadynamics')).toBeVisible();
    // MCMM-specific controls should not appear
    await expect(window.locator('text=Search steps')).not.toBeVisible();
  });

  test('run ETKDG: results table with energies and View 3D', async ({ window }) => {
    test.setTimeout(120_000);
    await loadAndConfigure(window);

    // Select ETKDG (fastest)
    await methodDropdown(window).selectOption('etkdg');
    await window.waitForTimeout(200);

    // Start search
    await window.locator('.btn.btn-primary', { hasText: /Start/i }).click();

    // Wait for completion, then navigate to results
    const viewResultsBtn = window.locator('.btn.btn-primary', { hasText: /View Results/i });
    await expect(viewResultsBtn).toBeVisible({ timeout: 60_000 });
    await viewResultsBtn.click();
    await window.waitForTimeout(500);

    // Verify results page
    await expect(window.locator('text=Conformer Results')).toBeVisible({ timeout: 5_000 });

    const table = window.locator('table');
    await expect(table).toBeVisible();

    // Table has conformer rows
    const rows = table.locator('tbody tr');
    expect(await rows.count()).toBeGreaterThan(0);

    // Energy column header present
    const headerText = await table.locator('thead').textContent();
    expect(headerText?.toLowerCase()).toContain('energy');

    // First row energy should be 0.0 (min)
    const firstRowEnergy = await rows.first().locator('td').last().textContent();
    expect(firstRowEnergy).toContain('0.0');

    // View 3D button visible
    await expect(window.locator('.btn.btn-primary', { hasText: /View 3D/i })).toBeVisible();
  });

  test('run MCMM method: completes with energies', async ({ window }) => {
    test.setTimeout(180_000);

    // Use benzene (small, fast MCMM)
    await window.locator('textarea').fill('c1ccccc1');
    await window.locator('.btn.btn-primary.btn-sm', { hasText: /Enter SMILES/i }).click();
    await expect(window.locator('.btn.btn-primary', { hasText: /Continue/i })).toBeEnabled({ timeout: 15_000 });
    await window.locator('.btn.btn-primary', { hasText: /Continue/i }).click();
    await window.waitForTimeout(500);

    // Select MCMM method
    await methodDropdown(window).selectOption('mcmm');
    await window.waitForTimeout(300);

    // Reduce steps for speed (find Search steps input)
    const stepsInput = window.locator('input[type="number"]').nth(2); // 3rd number input = steps
    await stepsInput.fill('100');
    await window.waitForTimeout(200);

    // Start search
    await window.locator('.btn.btn-primary', { hasText: /Start/i }).click();

    // Wait for completion
    const viewResultsBtn = window.locator('.btn.btn-primary', { hasText: /View Results/i });
    await expect(viewResultsBtn).toBeVisible({ timeout: 120_000 });
    await viewResultsBtn.click();
    await window.waitForTimeout(500);

    // Verify results
    await expect(window.locator('text=Conformer Results')).toBeVisible({ timeout: 5_000 });
    const table = window.locator('table');
    await expect(table).toBeVisible();
    expect(await table.locator('tbody tr').count()).toBeGreaterThan(0);

    // Should show MCMM method label
    await expect(window.locator('text=/MCMM.*Sage/i')).toBeVisible();
  });

  test('View 3D transitions to viewer with conformer queue', async ({ window }) => {
    test.setTimeout(120_000);
    await loadAndConfigure(window);

    await methodDropdown(window).selectOption('etkdg');
    await window.waitForTimeout(200);
    await window.locator('.btn.btn-primary', { hasText: /Start/i }).click();

    const viewResultsBtn = window.locator('.btn.btn-primary', { hasText: /View Results/i });
    await expect(viewResultsBtn).toBeVisible({ timeout: 60_000 });
    await viewResultsBtn.click();
    await window.waitForTimeout(500);

    // Click View 3D
    await window.locator('.btn.btn-primary', { hasText: /View 3D/i }).click();
    await window.waitForTimeout(1000);

    // Should switch to View mode — check that the View tab is active
    const viewTab = window.locator('.tab.tab-sm', { hasText: 'View' });
    await expect(viewTab).toHaveClass(/tab-active/, { timeout: 5_000 });
  });

  test('conformer queue navigation: prev/next buttons and index', async ({ window }) => {
    test.setTimeout(120_000);

    // Use 2-phenylethanol — phenyl ring + flexible chain → diverse conformers
    await window.locator('textarea').fill('OCCc1ccccc1');
    await window.locator('.btn.btn-primary.btn-sm', { hasText: /Enter SMILES/i }).click();
    await expect(window.locator('.btn.btn-primary', { hasText: /Continue/i })).toBeEnabled({ timeout: 15_000 });
    await window.locator('.btn.btn-primary', { hasText: /Continue/i }).click();
    await window.waitForTimeout(500);

    // Select ETKDG and lower RMSD cutoff to ensure multiple conformers survive
    await methodDropdown(window).selectOption('etkdg');
    await window.waitForTimeout(200);

    // Set RMSD cutoff to 0.3 Å (second number input in Parameters section)
    const rmsdInput = window.locator('input[type="number"]').nth(1);
    await rmsdInput.fill('0.3');
    await window.waitForTimeout(200);

    await window.locator('.btn.btn-primary', { hasText: /Start/i }).click();

    const viewResultsBtn = window.locator('.btn.btn-primary', { hasText: /View Results/i });
    await expect(viewResultsBtn).toBeVisible({ timeout: 60_000 });
    await viewResultsBtn.click();
    await window.waitForTimeout(500);

    // Click View 3D
    await window.locator('.btn.btn-primary', { hasText: /View 3D/i }).click();
    await window.waitForTimeout(1500);

    // Verify queue was populated in store
    const queueInfo = await window.evaluate(() => {
      const store = (window as any).__emberStore;
      const s = store.state();
      return {
        length: s.viewer.pdbQueue.length,
        index: s.viewer.pdbQueueIndex,
        type: s.viewer.pdbQueue[0]?.type,
        paths: s.conform.conformerPaths,
        queueLabels: s.viewer.pdbQueue.map((q: any) => q.label),
      };
    });
    expect(queueInfo.length).toBeGreaterThan(1);
    expect(queueInfo.index).toBe(0);
    expect(queueInfo.type).toBe('conformer');

    // Queue nav controls should be visible (prev/next buttons)
    const prevBtn = window.locator('button[title="Previous"]');
    const nextBtn = window.locator('button[title="Next"]');
    await expect(prevBtn).toBeVisible({ timeout: 3_000 });
    await expect(nextBtn).toBeVisible();

    // Previous should be disabled at index 0
    await expect(prevBtn).toBeDisabled();
    // Next should be enabled (multiple conformers)
    await expect(nextBtn).toBeEnabled();

    // Counter text shows "1/N"
    const counterText = await window.locator('button[title="Previous"]').locator('..').locator('..').textContent();
    expect(counterText).toContain(`1/${queueInfo.length}`);

    // Click Next → index advances to 1
    await nextBtn.click();
    await window.waitForTimeout(500);

    const afterNext = await window.evaluate(() => {
      const store = (window as any).__emberStore;
      return store.state().viewer.pdbQueueIndex;
    });
    expect(afterNext).toBe(1);

    // Counter now shows "2/N"
    const counterAfter = await window.locator('button[title="Previous"]').locator('..').locator('..').textContent();
    expect(counterAfter).toContain(`2/${queueInfo.length}`);

    // Previous should now be enabled
    await expect(prevBtn).toBeEnabled();

    // Click Previous → back to 0
    await prevBtn.click();
    await window.waitForTimeout(500);

    const afterPrev = await window.evaluate(() => {
      const store = (window as any).__emberStore;
      return store.state().viewer.pdbQueueIndex;
    });
    expect(afterPrev).toBe(0);

    // Previous disabled again at 0
    await expect(prevBtn).toBeDisabled();

    // NGL stage should have a component loaded (conformer)
    const nglState = await window.evaluate(() => {
      const stage = (window as any).__nglStage;
      return {
        compCount: stage?.compList?.length ?? 0,
      };
    });
    expect(nglState.compCount).toBeGreaterThan(0);
  });
});
