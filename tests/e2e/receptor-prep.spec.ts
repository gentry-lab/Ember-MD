/**
 * Receptor preparation E2E tests.
 * Tests water retention UI toggle and HIS tautomer energy scoring metadata.
 */
import { test, expect, createTestProject } from './fixtures';
import type { Page } from '@playwright/test';
import path from 'path';

const RECEPTOR_CIF = path.resolve(__dirname, '../../ember-test-protein/8tce.cif');
const LIGAND_SDF = path.resolve(__dirname, '../../ember-test-protein/kiv/kiv.sdf');

/** Navigate to Dock tab configure step via PDB ID fetch + SMILES */
async function navigateToDockConfigure(window: Page): Promise<void> {
  // Fetch receptor via PDB ID
  const pdbInput = window.locator('input[placeholder*="8TCE"]:visible');
  await pdbInput.fill('8TCE');
  await window.locator('.btn.btn-primary.btn-sm:visible', { hasText: 'Fetch' }).click();
  await window.waitForTimeout(2_000);

  // Wait for ligand detection
  await expect(
    window.locator('text=/Detected ligands|Reference ligand|8TCE|receptor/i').first()
  ).toBeVisible({ timeout: 30_000 });

  // Select first ligand from dropdown
  const ligandDropdown = window.locator('select').filter({
    has: window.locator('option', { hasText: /Select|ligand/i }),
  });
  await expect(ligandDropdown.first()).toBeVisible({ timeout: 10_000 });
  const options = await ligandDropdown.first().locator('option').allTextContents();
  if (options.length > 1) {
    await ligandDropdown.first().selectOption({ index: 1 });
    await window.waitForTimeout(3_000);
  }

  // Enter SMILES for a docking ligand
  const textarea = window.locator('textarea:visible');
  await textarea.fill('CC(=O)Oc1ccccc1C(=O)O');
  await window.locator('.btn.btn-primary.btn-sm:visible', { hasText: /Enter SMILES/i }).click();
  await window.waitForTimeout(5_000);

  // Continue to configure
  const continueBtn = window.locator('.btn.btn-primary:visible', { hasText: /Continue/i });
  await expect(continueBtn).toBeEnabled({ timeout: 30_000 });
  await continueBtn.click();
  await window.waitForTimeout(1_000);
}

test.describe('Receptor preparation features', () => {
  test.describe('Water retention UI toggle', () => {
    test.beforeEach(async ({ window }) => {
      await createTestProject(window, '__e2e_water__');
      await window.locator('.tab.tab-sm', { hasText: 'Dock' }).click();
      await window.waitForTimeout(500);
    });

    test('water retention checkbox visible on configure page', async ({ window }) => {
      test.setTimeout(90_000);
      await navigateToDockConfigure(window);

      await expect(window.locator('text=Configure Docking')).toBeVisible({ timeout: 5_000 });
      const waterCheckbox = window.locator('label', { hasText: /crystallographic waters/i }).locator('input[type="checkbox"]');
      await expect(waterCheckbox).toBeVisible();
      // Default: enabled
      expect(await waterCheckbox.isChecked()).toBe(true);
    });

    test('water retention toggle shows/hides distance input', async ({ window }) => {
      test.setTimeout(90_000);
      await navigateToDockConfigure(window);

      const waterCheckbox = window.locator('label', { hasText: /crystallographic waters/i }).locator('input[type="checkbox"]');
      await expect(waterCheckbox).toBeVisible();

      // Distance input should be visible when enabled
      const distanceInput = window.locator('input[type="number"]').filter({ has: window.locator('..') }).locator('..').filter({ hasText: /of ligand/i }).locator('input[type="number"]');
      // Use a simpler selector: the text "of ligand" should be visible
      await expect(window.locator('text=/of ligand/i')).toBeVisible();

      // Uncheck → distance input disappears
      await waterCheckbox.uncheck();
      await window.waitForTimeout(300);
      await expect(window.locator('text=/of ligand/i')).not.toBeVisible();

      // Re-check → reappears
      await waterCheckbox.check();
      await window.waitForTimeout(300);
      await expect(window.locator('text=/of ligand/i')).toBeVisible();
    });

    test('water retention config persists in store', async ({ window }) => {
      test.setTimeout(90_000);
      await navigateToDockConfigure(window);

      // Check default store state
      const defaultConfig = await window.evaluate(() => {
        const s = (window as any).__emberStore.state();
        return s.dock.waterRetentionConfig;
      });
      expect(defaultConfig.enabled).toBe(true);
      expect(defaultConfig.distance).toBe(3.5);

      // Toggle off via store
      await window.evaluate(() => {
        (window as any).__emberStore.setDockWaterRetentionConfig({ enabled: false });
      });
      await window.waitForTimeout(300);

      const afterDisable = await window.evaluate(() => {
        const s = (window as any).__emberStore.state();
        return s.dock.waterRetentionConfig;
      });
      expect(afterDisable.enabled).toBe(false);

      // Change distance via store
      await window.evaluate(() => {
        (window as any).__emberStore.setDockWaterRetentionConfig({ enabled: true, distance: 5.0 });
      });
      await window.waitForTimeout(300);

      const afterChange = await window.evaluate(() => {
        const s = (window as any).__emberStore.state();
        return s.dock.waterRetentionConfig;
      });
      expect(afterChange.enabled).toBe(true);
      expect(afterChange.distance).toBe(5.0);
    });
  });

  test.describe('HIS tautomer energy scoring', () => {
    test('receptor prep metadata includes his_tautomer_scoring', async ({ window }) => {
      test.setTimeout(180_000);
      await createTestProject(window, '__e2e_his__');

      // Step 1: Detect ligands
      const detectResult = await window.evaluate(async (receptorPath) => {
        const api = (window as any).electronAPI;
        return await api.detectPdbLigands(receptorPath);
      }, RECEPTOR_CIF);

      expect(detectResult?.ok).toBe(true);
      const ligands = detectResult.value.ligands || detectResult.value;
      expect(ligands.length).toBeGreaterThan(0);
      const ligandId = ligands[0].id;

      const outputPath = path.join(process.env.HOME || '/tmp', 'Ember', '__e2e_his__', 'docking', 'test', 'inputs', 'receptor.pdb');

      // Step 2: Prepare receptor (triggers HIS scoring + H-minimization)
      const prepResult = await window.evaluate(async (args) => {
        const api = (window as any).electronAPI;
        return await api.prepareReceptor(
          args.receptorPath, args.ligandId, args.outputPath, 3.5, 7.4,
        );
      }, { receptorPath: RECEPTOR_CIF, ligandId, outputPath });

      expect(prepResult?.ok).toBe(true);
      const prepPath = prepResult.value;

      // Step 3: Read the metadata sidecar JSON via IPC
      const metadataPath = prepPath.replace('.pdb', '.prep.json');
      const metadataExists = await window.evaluate(async (p) => {
        return await (window as any).electronAPI.fileExists(p);
      }, metadataPath);
      expect(metadataExists).toBe(true);

      const metadata = await window.evaluate(async (p) => {
        return await (window as any).electronAPI.readJsonFile(p);
      }, metadataPath);

      const result = {
        prepPath,
        hasHisScoring: 'his_tautomer_scoring' in metadata,
        hisScoring: metadata.his_tautomer_scoring,
        hasHMinimization: 'hydrogen_minimization' in metadata,
        hMinimization: metadata.hydrogen_minimization,
        resolvedVariants: metadata.resolved_variants,
      };

      // Validate results
      expect(result.prepPath).toBeTruthy();

      // HIS tautomer scoring should be present in metadata
      expect(result.hasHisScoring).toBe(true);
      expect(result.hisScoring).toBeDefined();
      expect(typeof result.hisScoring.scored_count).toBe('number');

      // 8TCE has histidines — scored_count should be > 0
      if (result.hisScoring.scored_count > 0) {
        expect(result.hisScoring.per_residue).toBeDefined();
        expect(result.hisScoring.per_residue.length).toBe(result.hisScoring.scored_count);
        expect(typeof result.hisScoring.wall_time_s).toBe('number');

        // Each per-residue entry should have expected fields
        for (const entry of result.hisScoring.per_residue) {
          expect(entry.residue_key).toBeTruthy();
          expect(['HID', 'HIE']).toContain(entry.geometric_pick);
          expect(['HID', 'HIE']).toContain(entry.final_pick);
          expect(typeof entry.changed).toBe('boolean');
          // Energies should be numeric (or null if evaluation failed)
          if (entry.hid_energy_kjmol !== null) {
            expect(typeof entry.hid_energy_kjmol).toBe('number');
          }
          if (entry.hie_energy_kjmol !== null) {
            expect(typeof entry.hie_energy_kjmol).toBe('number');
          }
        }
      }

      // H-minimization should also be present and applied
      expect(result.hasHMinimization).toBe(true);
      expect(result.hMinimization.applied).toBe(true);

      // Resolved variants should include HIS entries
      const hisVariants = Object.entries(result.resolvedVariants || {})
        .filter(([_, v]) => v === 'HID' || v === 'HIE' || v === 'HIP');
      expect(hisVariants.length).toBeGreaterThan(0);
    });
  });
});
