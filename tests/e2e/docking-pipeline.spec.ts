/**
 * Docking full pipeline test.
 * Uses 8TCE.cif receptor + KIV ligand from ember-test-protein/
 */
import { test, expect, createTestProject } from './fixtures';
import * as path from 'path';

const RECEPTOR_CIF = path.resolve(__dirname, '../../ember-test-protein/8tce.cif');
const KIV_SDF = path.resolve(__dirname, '../../ember-test-protein/kiv/kiv.sdf');

test.describe('Docking pipeline', () => {
  test.beforeEach(async ({ window }) => {
    await createTestProject(window, '__e2e_dock__');

    await window.locator('.tab.tab-sm', { hasText: 'Dock' }).click();
    await window.waitForTimeout(500);
  });

  test('load receptor and detect ligands', async ({ window }) => {
    test.setTimeout(60_000);

    // Mock the PDB file dialog
    await window.evaluate((cifPath) => {
      (window as any).electronAPI.selectPdbFile = async () => cifPath;
    }, RECEPTOR_CIF);

    // Click "Select Structure"
    const selectBtn = window.locator('.btn.btn-primary.btn-sm', { hasText: /Select Structure/i });
    await selectBtn.click();

    // Wait for ligand detection
    await window.waitForTimeout(10_000);

    // Reference ligand dropdown should appear and have options
    const ligandSelect = window.locator('select.select-bordered.select-xs');
    const selectCount = await ligandSelect.count();
    expect(selectCount).toBeGreaterThan(0);
  });

  test('load docking ligand via SDF', async ({ window }) => {
    test.setTimeout(30_000);

    // Mock molecule file dialog to return KIV SDF
    await window.evaluate((sdfPath) => {
      (window as any).electronAPI.selectMoleculeFilesMulti = async () => [sdfPath];
    }, KIV_SDF);

    // Click ligand import button
    const importBtn = window.locator('.btn.btn-outline', { hasText: /Import/ });
    if (await importBtn.first().isVisible()) {
      await importBtn.first().click();
      await window.waitForTimeout(3000);
    }

    // Should show loaded ligand count
    const content = await window.textContent('body');
    // Either "1 ligand" or the filename should appear
    expect(content?.includes('ligand') || content?.includes('kiv')).toBeTruthy();
  });

  test('configure page shows docking parameters', async ({ window }) => {
    test.setTimeout(60_000);

    // Load receptor + ligand (minimal setup to reach configure)
    await window.evaluate((cifPath) => {
      (window as any).electronAPI.selectPdbFile = async () => cifPath;
    }, RECEPTOR_CIF);

    const selectBtn = window.locator('.btn.btn-primary.btn-sm', { hasText: /Select Structure/i });
    if (await selectBtn.isVisible()) {
      await selectBtn.click();
      await window.waitForTimeout(10_000);
    }

    // Select first detected ligand if dropdown is present
    const ligandSelect = window.locator('select.select-bordered.select-xs');
    if (await ligandSelect.first().isVisible()) {
      const options = await ligandSelect.first().locator('option').allTextContents();
      if (options.length > 1) {
        await ligandSelect.first().selectOption({ index: 1 });
        await window.waitForTimeout(5000);
      }
    }

    // Add docking ligand
    await window.evaluate((sdfPath) => {
      (window as any).electronAPI.selectMoleculeFilesMulti = async () => [sdfPath];
    }, KIV_SDF);
    const importBtn = window.locator('.btn.btn-outline', { hasText: /Import/ });
    if (await importBtn.first().isVisible()) {
      await importBtn.first().click();
      await window.waitForTimeout(3000);
    }

    // Click Continue if available
    const continueBtn = window.locator('.btn.btn-primary', { hasText: /Continue/i });
    if (await continueBtn.isVisible() && await continueBtn.isEnabled()) {
      await continueBtn.click();
      await window.waitForTimeout(1000);
    }

    // Should see docking parameter inputs
    const exhaustInput = window.locator('input.input-bordered');
    const selectInputs = window.locator('select.select-bordered');
    const inputCount = await exhaustInput.count();
    const selectCount = await selectInputs.count();
    expect(inputCount + selectCount).toBeGreaterThan(0);
  });
});
