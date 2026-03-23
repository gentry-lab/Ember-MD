/**
 * Computational output verification tests.
 * Verifies that pipelines produce real output files with valid data.
 */
import { test, expect, createTestProject } from './fixtures';
import type { Page } from '@playwright/test';
import * as path from 'path';

/** Helper: load via SMILES, configure ETKDG, run, wait for completion */
async function runEtkdg(window: Page, smiles: string): Promise<void> {
  // Enter SMILES
  await window.locator('textarea').fill(smiles);
  await window.locator('.btn.btn-primary.btn-sm', { hasText: /Enter SMILES/i }).click();
  await expect(window.locator('.btn.btn-primary', { hasText: /Continue/i })).toBeEnabled({ timeout: 15_000 });
  await window.locator('.btn.btn-primary', { hasText: /Continue/i }).click();
  await window.waitForTimeout(500);

  // Select ETKDG
  const methodSelect = window.locator('select').filter({
    has: window.locator('option', { hasText: 'ETKDG' }),
  });
  await methodSelect.selectOption('etkdg');
  await window.waitForTimeout(200);

  // Start
  await window.locator('.btn.btn-primary', { hasText: /Start/i }).click();

  // Wait for completion
  const viewResults = window.locator('.btn.btn-primary', { hasText: /View Results/i });
  await expect(viewResults).toBeVisible({ timeout: 60_000 });
  await viewResults.click();
  await window.waitForTimeout(500);
}

test.describe('Receptor preparation verification', () => {
  test('raw input CIF preserved in structures/ after import', async ({ window }) => {
    test.setTimeout(30_000);

    const RECEPTOR_CIF = path.resolve(__dirname, '../../ember-test-protein/8tce.cif');
    await createTestProject(window, '__e2e_output_receptor__');

    // Import structure via IPC
    const importResult = await window.evaluate(async (cifPath: string) => {
      const api = (window as any).electronAPI;
      const projResult = await api.ensureProject('__e2e_output_receptor__');
      const projDir = projResult.ok ? projResult.value : '';
      const result = await api.importStructure(cifPath, projDir);
      return {
        ok: result.ok,
        importedPath: result.ok ? result.value : null,
        projDir,
      };
    }, RECEPTOR_CIF);

    expect(importResult.ok).toBe(true);
    expect(importResult.importedPath).toBeTruthy();

    // Imported file should be in structures/ subdir
    expect(importResult.importedPath).toContain('/structures/');

    // File should actually exist on disk
    const exists = await window.evaluate(async (p: string) => {
      return await (window as any).electronAPI.fileExists(p);
    }, importResult.importedPath!);
    expect(exists).toBe(true);
  });
});

test.describe('Computational output verification', () => {
  test.describe('MCMM/Conformer outputs', () => {
    test.beforeEach(async ({ window }) => {
      await createTestProject(window, '__e2e_output__');
      await window.locator('.tab.tab-sm', { hasText: 'MCMM' }).click();
      await window.waitForTimeout(500);
    });

    test('ETKDG: output SDF exists, has conformers, energies are numeric', async ({ window }) => {
      test.setTimeout(120_000);
      await runEtkdg(window, 'CC(=O)Oc1ccccc1C(=O)O'); // aspirin

      // Get conformer data from store
      const conformData = await window.evaluate(() => {
        const s = (window as any).__emberStore.state();
        return {
          paths: s.conform.conformerPaths,
          energies: s.conform.conformerEnergies,
          outputDir: s.conform.outputDir,
        };
      });

      // Should have at least 1 conformer path
      expect(conformData.paths.length).toBeGreaterThan(0);

      // All paths should be real files
      for (const p of conformData.paths) {
        const exists = await window.evaluate(async (filePath: string) => {
          return await (window as any).electronAPI.fileExists(filePath);
        }, p);
        expect(exists).toBe(true);
      }

      // Energies should be numeric for each path
      const energyValues = Object.values(conformData.energies) as number[];
      expect(energyValues.length).toBeGreaterThan(0);
      for (const e of energyValues) {
        expect(typeof e).toBe('number');
        expect(Number.isFinite(e)).toBe(true);
      }

      // Minimum energy should be 0.0 (relative)
      const minEnergy = Math.min(...energyValues);
      expect(minEnergy).toBeCloseTo(0.0, 1);

      // Output directory should match expected pattern (conformers/{run}/)
      expect(conformData.outputDir).toMatch(/conformers\//);
    });
  });
});
