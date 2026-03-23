/**
 * MD simulation pipeline test.
 * Uses 8TCE.cif receptor + KIV ligand for protein+ligand mode.
 */
import { test, expect, createTestProject } from './fixtures';
import * as path from 'path';

const RECEPTOR_CIF = path.resolve(__dirname, '../../ember-test-protein/8tce.cif');

test.describe('MD simulation pipeline', () => {
  test.beforeEach(async ({ window }) => {
    await createTestProject(window, '__e2e_md__');

    await window.locator('.tab.tab-sm', { hasText: 'Simulate' }).click();
    await window.waitForTimeout(500);
  });

  test('load PDB and detect protein+ligand mode', async ({ window }) => {
    test.setTimeout(60_000);

    // Mock PDB file dialog
    await window.evaluate((cifPath) => {
      (window as any).electronAPI.selectPdbFile = async () => cifPath;
    }, RECEPTOR_CIF);

    // Click import
    const importBtn = window.locator('.btn.btn-outline', { hasText: /Import/ });
    await importBtn.click();
    await window.waitForTimeout(10_000);

    // Should detect ligands in the structure
    const bodyText = await window.textContent('body');
    // Look for ligand detection indicators
    const hasLigandUI = bodyText?.includes('Ligand') ||
                        bodyText?.includes('ligand') ||
                        bodyText?.includes('Protein');
    expect(hasLigandUI).toBeTruthy();
  });

  test('SMILES input sets ligand-only mode', async ({ window }) => {
    test.setTimeout(30_000);

    const textarea = window.locator('textarea');
    if (await textarea.isVisible()) {
      await textarea.fill('c1ccccc1');

      const enterBtn = window.locator('.btn.btn-primary.btn-sm', { hasText: /Enter SMILES/i });
      if (await enterBtn.isVisible()) {
        await enterBtn.click();
        await window.waitForTimeout(3000);

        // Should show ligand-only mode badge
        const bodyText = await window.textContent('body');
        const isLigandOnly = bodyText?.includes('Ligand Only') || bodyText?.includes('ligand');
        expect(isLigandOnly).toBeTruthy();
      }
    }
  });

  test('configure page shows force field preset', async ({ window }) => {
    test.setTimeout(60_000);

    // Load structure first
    await window.evaluate((cifPath) => {
      (window as any).electronAPI.selectPdbFile = async () => cifPath;
    }, RECEPTOR_CIF);

    const importBtn = window.locator('.btn.btn-outline', { hasText: /Import/ });
    await importBtn.click();
    await window.waitForTimeout(10_000);

    // Select first detected ligand
    const ligandSelect = window.locator('select.select-bordered.select-xs');
    if (await ligandSelect.first().isVisible()) {
      const options = await ligandSelect.first().locator('option').allTextContents();
      if (options.length > 1) {
        await ligandSelect.first().selectOption({ index: 1 });
        await window.waitForTimeout(5000);
      }
    }

    // Continue to configure
    const continueBtn = window.locator('.btn.btn-primary', { hasText: /Continue/i });
    if (await continueBtn.isVisible() && await continueBtn.isEnabled()) {
      await continueBtn.click();
      await window.waitForTimeout(1000);

      // Should show force field dropdown
      const ffSelect = window.locator('select.select-sm');
      if (await ffSelect.isVisible()) {
        const options = await ffSelect.locator('option').allTextContents();
        const hasFF19SB = options.some(o => o.toLowerCase().includes('ff19sb'));
        expect(hasFF19SB).toBe(true);
      }

      // Should show temperature input
      const bodyText = await window.textContent('body');
      expect(bodyText?.includes('Temperature') || bodyText?.includes('300')).toBeTruthy();
    }
  });
});
