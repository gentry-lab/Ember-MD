/**
 * MCMM (conformer generation) full pipeline test.
 * Uses KIV ligand from ember-test-protein/kiv/kiv.sdf
 */
import { test, expect, createTestProject } from './fixtures';
import * as path from 'path';

const KIV_SDF = path.resolve(__dirname, '../../ember-test-protein/kiv/kiv.sdf');

test.describe('MCMM pipeline', () => {
  test.beforeEach(async ({ window }) => {
    // Create project through UI so tabs become enabled
    await createTestProject(window, '__e2e_mcmm__');

    // Navigate to MCMM tab
    await window.locator('.tab.tab-sm', { hasText: 'MCMM' }).click();
    await window.waitForTimeout(500);
  });

  test('load ligand via mocked file dialog', async ({ window }) => {
    test.setTimeout(30_000);

    // Mock the SDF file dialog
    await window.evaluate((sdfPath) => {
      (window as any).electronAPI.selectSdfFile = async () => sdfPath;
    }, KIV_SDF);

    // Click import button
    const importBtn = window.locator('.btn.btn-outline', { hasText: /\.sdf.*\.mol2/ });
    await importBtn.click();
    await window.waitForTimeout(2000);

    // Ligand name should appear
    const content = await window.locator('.card').first().textContent();
    expect(content?.toLowerCase()).toContain('kiv');
  });

  test('load ligand via SMILES', async ({ window }) => {
    test.setTimeout(30_000);

    // Enter benzene SMILES
    const textarea = window.locator('textarea');
    await textarea.fill('c1ccccc1');

    // Click Enter SMILES
    const enterBtn = window.locator('.btn.btn-primary.btn-sm', { hasText: /Enter SMILES/i });
    await enterBtn.click();
    await window.waitForTimeout(3000);

    // Should show a loaded molecule
    const continueBtn = window.locator('.btn.btn-primary', { hasText: /Continue/i });
    // If SMILES conversion succeeded, continue should be available
    const isVisible = await continueBtn.isVisible().catch(() => false);
    expect(isVisible).toBe(true);
  });

  test('configure shows method dropdown', async ({ window }) => {
    test.setTimeout(30_000);

    // Load ligand first
    await window.evaluate((sdfPath) => {
      (window as any).electronAPI.selectSdfFile = async () => sdfPath;
    }, KIV_SDF);
    const importBtn = window.locator('.btn.btn-outline', { hasText: /\.sdf.*\.mol2/ });
    await importBtn.click();
    await window.waitForTimeout(2000);

    // Click Continue
    const continueBtn = window.locator('.btn.btn-primary', { hasText: /Continue/i });
    await continueBtn.click();
    await window.waitForTimeout(500);

    // Should see method dropdown
    const methodSelect = window.locator('select.select-bordered');
    await expect(methodSelect.first()).toBeVisible();

    // Check options
    const options = await methodSelect.first().locator('option').allTextContents();
    const optionsLower = options.map(o => o.toLowerCase());
    expect(optionsLower.some(o => o.includes('etkdg'))).toBe(true);
    expect(optionsLower.some(o => o.includes('mcmm'))).toBe(true);
  });

  test('run ETKDG conformer search and verify results', async ({ window }) => {
    test.setTimeout(120_000);

    // Load ligand
    await window.evaluate((sdfPath) => {
      (window as any).electronAPI.selectSdfFile = async () => sdfPath;
    }, KIV_SDF);
    const importBtn = window.locator('.btn.btn-outline', { hasText: /\.sdf.*\.mol2/ });
    await importBtn.click();
    await window.waitForTimeout(2000);

    // Continue to configure
    await window.locator('.btn.btn-primary', { hasText: /Continue/i }).click();
    await window.waitForTimeout(500);

    // Ensure method is ETKDG (fastest)
    const methodSelect = window.locator('select.select-bordered').first();
    await methodSelect.selectOption('etkdg');
    await window.waitForTimeout(200);

    // Start search
    await window.locator('.btn.btn-primary', { hasText: /Start/i }).click();

    // Wait for results (ETKDG should complete in <30s for a small molecule)
    const resultsTitle = window.locator('text=Conformer Results');
    await expect(resultsTitle).toBeVisible({ timeout: 60_000 });

    // Check results table exists
    const table = window.locator('table');
    await expect(table).toBeVisible();

    // Should have at least 1 row
    const rows = table.locator('tbody tr');
    const rowCount = await rows.count();
    expect(rowCount).toBeGreaterThan(0);

    // Check for energy column
    const headerText = await table.locator('thead').textContent();
    expect(headerText?.toLowerCase()).toContain('energy');

    // View 3D button should be visible
    const view3dBtn = window.locator('.btn.btn-primary', { hasText: /View 3D/i });
    await expect(view3dBtn).toBeVisible();
  });
});
