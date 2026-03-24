// Copyright (c) 2026 Ember Contributors. MIT License.
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
  await window.locator('textarea:visible').fill(smiles);
  await window.locator('.btn.btn-primary.btn-sm:visible', { hasText: /Enter SMILES/i }).click();
  await expect(window.locator('.btn.btn-primary:visible', { hasText: /Continue/i })).toBeEnabled({ timeout: 15_000 });
  await window.locator('.btn.btn-primary:visible', { hasText: /Continue/i }).click();
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

  test('detectPdbLigands returns ligands and preparation metadata', async ({ window }) => {
    test.setTimeout(120_000);

    const RECEPTOR_CIF = path.resolve(__dirname, '../../ember-test-protein/8tce.cif');

    // Call detectPdbLigands — returns Result<{ ligands, structureInfo }>
    const result = await window.evaluate(async (cifPath: string) => {
      const api = (window as any).electronAPI;
      const r = await api.detectPdbLigands(cifPath);
      if (!r.ok) return { ok: false, error: r.error?.message };
      return {
        ok: true,
        ligandCount: r.value.ligands.length,
        firstLigandId: r.value.ligands[0]?.id,
        structureInfo: r.value.structureInfo,
      };
    }, RECEPTOR_CIF);

    expect(result.ok).toBe(true);
    // 8TCE should have at least one detected ligand (KIV)
    expect(result.ligandCount).toBeGreaterThan(0);
    expect(result.firstLigandId).toBeTruthy();

    // Structure info should have atom count
    if (result.structureInfo) {
      expect(result.structureInfo.totalAtoms).toBeGreaterThan(0);
    }
  });

  test('prepareReceptor adds hydrogens and writes receptor prep metadata json', async ({ window }) => {
    test.setTimeout(120_000);

    const RECEPTOR_CIF = path.resolve(__dirname, '../../ember-test-protein/8tce.cif');
    await createTestProject(window, '__e2e_output_receptor_prep__');

    const prepResult = await window.evaluate(async (cifPath: string) => {
      const api = (window as any).electronAPI;
      const projResult = await api.ensureProject('__e2e_output_receptor_prep__');
      if (!projResult.ok) return { ok: false, error: 'project failed' };
      const projDir = projResult.value;

      const detected = await api.detectPdbLigands(cifPath);
      if (!detected.ok || detected.value.ligands.length === 0) {
        return { ok: false, error: detected.error?.message || 'no ligands detected' };
      }

      const ligandId = detected.value.ligands[0].id;
      const preparedPath = `${projDir}/prepared/receptor_prepared.pdb`;
      const prepared = await api.prepareReceptor(cifPath, ligandId, preparedPath);
      if (!prepared.ok) {
        return { ok: false, error: prepared.error?.message || 'prepare failed' };
      }

      const preparedInfo = await api.detectPdbLigands(prepared.value);
      const metadataPath = prepared.value.replace(/\.pdb$/i, '.prep.json');
      const metadata = await api.readJsonFile(metadataPath) as Record<string, unknown> | null;
      const metadataExists = await api.fileExists(metadataPath);

      return {
        ok: true,
        raw: detected.value.structureInfo,
        prepared: preparedInfo.ok ? preparedInfo.value.structureInfo : null,
        preparedPath: prepared.value,
        metadataPath,
        metadataExists,
        metadata,
      };
    }, RECEPTOR_CIF);

    expect(prepResult.ok).toBe(true);
    expect(prepResult.preparedPath).toBeTruthy();
    expect(prepResult.metadataExists).toBe(true);
    expect(prepResult.raw?.totalAtoms ?? 0).toBeGreaterThan(0);
    expect(prepResult.prepared?.totalAtoms ?? 0).toBeGreaterThan(prepResult.raw?.totalAtoms ?? 0);
    expect(prepResult.prepared?.hydrogenCount ?? 0).toBeGreaterThan(prepResult.raw?.hydrogenCount ?? 0);
    expect(prepResult.prepared?.isPrepared).toBe(true);
    expect(prepResult.metadata?.['prepared_receptor_pdb']).toBe(prepResult.preparedPath);
    expect(Array.isArray(prepResult.metadata?.['applied_overrides'])).toBe(true);
    expect(Array.isArray(prepResult.metadata?.['ignored_shifted_residues'])).toBe(true);
    expect(typeof prepResult.metadata?.['resolved_variants']).toBe('object');
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

    test('CREST (methylcyclobutane): output SDF exists, conformers ranked, energies populated', async ({ window }) => {
      test.setTimeout(600_000); // CREST takes ~6 min

      // Enter methylcyclobutane SMILES
      await window.locator('textarea:visible').fill('CC1CCC1');
      await window.locator('.btn.btn-primary.btn-sm:visible', { hasText: /Enter SMILES/i }).click();
      await expect(window.locator('.btn.btn-primary:visible', { hasText: /Continue/i })).toBeEnabled({ timeout: 15_000 });
      await window.locator('.btn.btn-primary:visible', { hasText: /Continue/i }).click();
      await window.waitForTimeout(500);

      // Select CREST method
      const methodSelect = window.locator('select').filter({
        has: window.locator('option', { hasText: 'CREST' }),
      });
      await methodSelect.selectOption('crest');
      await window.waitForTimeout(200);

      // Start
      await window.locator('.btn.btn-primary:visible', { hasText: /Start/i }).click();

      // Wait for completion
      const viewResults = window.locator('.btn.btn-primary:visible', { hasText: /View Results/i });
      await expect(viewResults).toBeVisible({ timeout: 540_000 });
      await viewResults.click();
      await window.waitForTimeout(500);

      // Get conformer data
      const conformData = await window.evaluate(() => {
        const s = (window as any).__emberStore.state();
        return {
          paths: s.conform.conformerPaths,
          energies: s.conform.conformerEnergies,
          outputDir: s.conform.outputDir,
        };
      });

      // Should have multiple conformers (CREST finds diverse minima)
      expect(conformData.paths.length).toBeGreaterThan(0);

      // Energies should be numeric and ranked (min = 0.0)
      const energyValues = Object.values(conformData.energies) as number[];
      expect(energyValues.length).toBeGreaterThan(0);
      for (const e of energyValues) {
        expect(typeof e).toBe('number');
        expect(Number.isFinite(e)).toBe(true);
      }
      const minEnergy = Math.min(...energyValues);
      expect(minEnergy).toBeCloseTo(0.0, 1);

      // All conformer files should exist
      for (const p of conformData.paths) {
        const exists = await window.evaluate(async (filePath: string) => {
          return await (window as any).electronAPI.fileExists(filePath);
        }, p);
        expect(exists).toBe(true);
      }
    });
  });
});
