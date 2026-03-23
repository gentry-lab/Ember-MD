/**
 * Docking full pipeline test.
 * Uses PDB ID fetch for receptor + SMILES for ligand (no file dialogs).
 */
import { test, expect, createTestProject } from './fixtures';
import type { Page } from '@playwright/test';

/** Fetch receptor via PDB ID and wait for ligand detection */
async function fetchReceptor(window: Page): Promise<void> {
  const pdbInput = window.locator('input[placeholder*="8TCE"]');
  await pdbInput.fill('8TCE');
  await window.locator('.btn.btn-primary.btn-sm', { hasText: 'Fetch' }).click();

  // Wait for receptor to load — status text changes from "Fetching..."
  // and "No project selected" error should NOT appear
  await window.waitForTimeout(2_000);
  const errorAlert = window.locator('.alert.alert-error');
  const hasError = await errorAlert.isVisible();
  if (hasError) {
    const errorText = await errorAlert.textContent();
    throw new Error(`Unexpected error after PDB fetch: ${errorText}`);
  }

  // Wait for ligand detection to complete (receptor name or ligand dropdown appears)
  await expect(
    window.locator('text=/Detected ligands|Reference ligand|8TCE|receptor/i').first()
  ).toBeVisible({ timeout: 30_000 });
}

test.describe('Docking pipeline', () => {
  test.beforeEach(async ({ window }) => {
    await createTestProject(window, '__e2e_dock__');
    await window.locator('.tab.tab-sm', { hasText: 'Dock' }).click();
    await window.waitForTimeout(500);
  });

  test('load receptor via PDB ID and detect ligands', async ({ window }) => {
    test.setTimeout(60_000);

    await fetchReceptor(window);

    // Reference ligand dropdown should appear with detected ligands
    const ligandDropdown = window.locator('select').filter({
      has: window.locator('option', { hasText: /Select|ligand/i }),
    });
    await expect(ligandDropdown.first()).toBeVisible({ timeout: 10_000 });
    const optionCount = await ligandDropdown.first().locator('option').count();
    expect(optionCount).toBeGreaterThan(1); // placeholder + at least one ligand
  });

  test('load docking ligand via SMILES', async ({ window }) => {
    test.setTimeout(30_000);

    // Enter SMILES for a docking ligand (aspirin)
    const textarea = window.locator('textarea');
    await expect(textarea).toBeVisible();
    await textarea.fill('CC(=O)Oc1ccccc1C(=O)O');

    const enterBtn = window.locator('.btn.btn-primary.btn-sm', { hasText: /Enter SMILES/i });
    await expect(enterBtn).toBeVisible();
    await enterBtn.click();

    // Should show ligand was loaded (molecule count or loaded indicator)
    await expect(
      window.locator('text=/1 molecule|mol_1|loaded/i').first()
    ).toBeVisible({ timeout: 15_000 });
  });

  test('configure page shows docking parameters', async ({ window }) => {
    test.setTimeout(90_000);

    // Fetch receptor
    await fetchReceptor(window);

    // Select first detected ligand
    const ligandDropdown = window.locator('select').filter({
      has: window.locator('option', { hasText: /Select|ligand/i }),
    });
    await expect(ligandDropdown.first()).toBeVisible({ timeout: 10_000 });
    const options = await ligandDropdown.first().locator('option').allTextContents();
    if (options.length > 1) {
      await ligandDropdown.first().selectOption({ index: 1 });
      await window.waitForTimeout(3_000);
    }

    // Add docking ligand via SMILES
    const textarea = window.locator('textarea');
    await textarea.fill('CC(=O)Oc1ccccc1C(=O)O');
    await window.locator('.btn.btn-primary.btn-sm', { hasText: /Enter SMILES/i }).click();
    await window.waitForTimeout(5_000);

    // Click Continue
    const continueBtn = window.locator('.btn.btn-primary', { hasText: /Continue/i });
    await expect(continueBtn).toBeEnabled({ timeout: 10_000 });
    await continueBtn.click();
    await window.waitForTimeout(1_000);

    // Should see Configure Docking heading and parameter inputs
    await expect(window.locator('text=Configure Docking')).toBeVisible({ timeout: 5_000 });

    // Exhaustiveness and poses inputs
    await expect(window.locator('text=/exhaustiveness/i')).toBeVisible();
    await expect(window.locator('text=/poses/i').first()).toBeVisible();
  });
});
