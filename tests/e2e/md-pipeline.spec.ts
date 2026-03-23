/**
 * MD simulation pipeline test.
 * Uses PDB ID fetch for receptor + SMILES for ligand (no file dialogs).
 */
import { test, expect, createTestProject } from './fixtures';
import type { Page } from '@playwright/test';

/** Navigate to MD configure page via PDB ID fetch → Continue */
async function navigateToConfigure(window: Page): Promise<void> {
  await fetchReceptor(window);
  const continueBtn = window.locator('.btn.btn-primary', { hasText: /Continue/i });
  await expect(continueBtn).toBeEnabled({ timeout: 5_000 });
  await continueBtn.click();
  await window.waitForTimeout(1_000);
  await expect(window.locator('main').locator('text=/Configure/i').first()).toBeVisible({ timeout: 5_000 });
}

/** Fetch 8TCE via PDB ID, verifying no errors and structure loads */
async function fetchReceptor(window: Page): Promise<void> {
  const pdbInput = window.locator('input[placeholder*="8TCE"]');
  await pdbInput.fill('8TCE');
  await window.locator('.btn.btn-primary.btn-sm', { hasText: 'Fetch' }).click();

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
    await expect(window.locator('.btn.btn-primary', { hasText: /Continue/i })).toBeEnabled();
  });

  test('SMILES input sets ligand-only mode', async ({ window }) => {
    test.setTimeout(30_000);

    const textarea = window.locator('textarea');
    await expect(textarea).toBeVisible();
    await textarea.fill('c1ccccc1');

    const enterBtn = window.locator('.btn.btn-primary.btn-sm', { hasText: /Enter SMILES/i });
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
    const continueBtn = window.locator('.btn.btn-primary', { hasText: /Continue/i });
    await expect(continueBtn).toBeEnabled({ timeout: 5_000 });
    await continueBtn.click();
    await window.waitForTimeout(1_000);

    // Should see Configure heading
    await expect(window.locator('main').locator('text=/Configure/i').first()).toBeVisible({ timeout: 5_000 });

    // Force field preset dropdown with ff19sb option
    const ffSelect = window.locator('select').filter({
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
    const editInput = window.locator('input[type="number"][min="0.1"][max="10000"]');
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
    const tempInput = window.locator('input[type="number"][min="200"][max="500"]');
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

  test('configure: Estimate Runtime button is visible and clickable', async ({ window }) => {
    // NOTE: Full benchmark test is blocked by a Python bug in run_md_simulation.py:
    // _patched_createSystem conflicts with OpenMM ArgTracker — flexibleConstraints
    // default arg is considered "unused" by ArgTracker when fn.__name__ is the patched wrapper.
    // This causes BENCHMARK_FAILED(code 1). Do not fix in staging scripts.
    test.setTimeout(60_000);
    await navigateToConfigure(window);

    const benchmarkBtn = window.locator('.btn.btn-secondary', { hasText: /Estimate Runtime/i });
    await expect(benchmarkBtn).toBeVisible({ timeout: 5_000 });
    await expect(benchmarkBtn).toBeEnabled();

    // Click — should show some status change (Cancel button or status text)
    await benchmarkBtn.click();
    await window.waitForTimeout(2_000);

    // Either showing Cancel (benchmarking) or benchmark already errored
    // The button should have changed state — either Cancel or back to Estimate Runtime
    // We just verify no crash: app still responsive
    const simulateTab = window.locator('.tab.tab-sm', { hasText: 'Simulate' });
    await expect(simulateTab).toBeVisible();
  });
});
