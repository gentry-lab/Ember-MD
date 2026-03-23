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

/** Full setup: fetch receptor, select ligand, add SMILES, navigate to configure */
async function navigateToConfigure(window: Page): Promise<void> {
  await fetchReceptor(window);

  const ligandDropdown = window.locator('select').filter({
    has: window.locator('option', { hasText: /Select|ligand/i }),
  });
  await expect(ligandDropdown.first()).toBeVisible({ timeout: 10_000 });
  const options = await ligandDropdown.first().locator('option').allTextContents();
  if (options.length > 1) {
    await ligandDropdown.first().selectOption({ index: 1 });
    await window.waitForTimeout(3_000);
  }

  const textarea = window.locator('textarea');
  await textarea.fill('CC(=O)Oc1ccccc1C(=O)O');
  await window.locator('.btn.btn-primary.btn-sm', { hasText: /Enter SMILES/i }).click();
  await window.waitForTimeout(5_000);

  const continueBtn = window.locator('.btn.btn-primary', { hasText: /Continue/i });
  await expect(continueBtn).toBeEnabled({ timeout: 10_000 });
  await continueBtn.click();
  await window.waitForTimeout(1_000);
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
    await navigateToConfigure(window);

    // Should see Configure Docking heading and parameter inputs
    await expect(window.locator('text=Configure Docking')).toBeVisible({ timeout: 5_000 });
    await expect(window.locator('text=/exhaustiveness/i')).toBeVisible();
    await expect(window.locator('text=/poses/i').first()).toBeVisible();
  });

  test('configure: protonation toggle enables pH inputs', async ({ window }) => {
    test.setTimeout(90_000);
    await navigateToConfigure(window);

    // Find protonation checkbox
    const protonationCheckbox = window.locator('label', { hasText: /Protonation/i }).locator('input[type="checkbox"]');
    await expect(protonationCheckbox).toBeVisible();

    // Should be checked by default; pH inputs should be visible
    const phMin = window.locator('text=/pH min/i');
    await expect(phMin).toBeVisible();

    // Uncheck protonation → pH inputs should disappear
    await protonationCheckbox.uncheck();
    await window.waitForTimeout(500);
    await expect(phMin).not.toBeVisible();

    // Re-check → pH inputs reappear
    await protonationCheckbox.check();
    await window.waitForTimeout(500);
    await expect(phMin).toBeVisible();
  });

  test('configure: stereoisomer toggle works', async ({ window }) => {
    test.setTimeout(90_000);
    await navigateToConfigure(window);

    const stereoCheckbox = window.locator('label', { hasText: /enantiomer/i }).locator('input[type="checkbox"]');
    await expect(stereoCheckbox).toBeVisible();

    // Toggle on and off
    const wasChecked = await stereoCheckbox.isChecked();
    if (wasChecked) {
      await stereoCheckbox.uncheck();
      expect(await stereoCheckbox.isChecked()).toBe(false);
    } else {
      await stereoCheckbox.check();
      expect(await stereoCheckbox.isChecked()).toBe(true);
    }
  });

  test('configure: conformer method dropdown works', async ({ window }) => {
    test.setTimeout(90_000);
    await navigateToConfigure(window);

    // Find conformer method dropdown (has Simple/ETKDG/MCMM options)
    const conformerSelect = window.locator('select').filter({
      has: window.locator('option', { hasText: 'Simple' }),
    });
    await expect(conformerSelect).toBeVisible();

    const options = await conformerSelect.locator('option').allTextContents();
    expect(options.some(o => o.includes('Simple'))).toBe(true);
    expect(options.some(o => o.includes('ETKDG'))).toBe(true);
    expect(options.some(o => o.includes('MCMM'))).toBe(true);

    // Select ETKDG
    await conformerSelect.selectOption({ label: 'ETKDG' });
    await window.waitForTimeout(300);
  });

  test('configure: pocket refinement toggle works', async ({ window }) => {
    test.setTimeout(90_000);
    await navigateToConfigure(window);

    const refinementCheckbox = window.locator('label', { hasText: /Pocket refinement/i }).locator('input[type="checkbox"]');
    await expect(refinementCheckbox).toBeVisible();

    const wasChecked = await refinementCheckbox.isChecked();
    if (wasChecked) {
      await refinementCheckbox.uncheck();
      expect(await refinementCheckbox.isChecked()).toBe(false);
    } else {
      await refinementCheckbox.check();
      expect(await refinementCheckbox.isChecked()).toBe(true);
    }
  });

  test('run docking (exhaust=1, poses=1): progress → results with Vina affinity', async ({ window }) => {
    test.setTimeout(180_000);
    await navigateToConfigure(window);

    // Set minimal parameters for speed — use store directly for reliability
    await window.evaluate(() => {
      const store = (window as any).__emberStore;
      if (store) {
        store.setDockConfig({ exhaustiveness: 1, poses: 1 });
      }
    });
    await window.waitForTimeout(300);

    // Disable pocket refinement for speed
    const refinementCheckbox = window.locator('label', { hasText: /Pocket refinement/i }).locator('input[type="checkbox"]');
    if (await refinementCheckbox.isChecked()) {
      await refinementCheckbox.uncheck();
    }

    // Disable CORDIAL for speed
    const cordialCheckbox = window.locator('label', { hasText: /CORDIAL/i }).locator('input[type="checkbox"]');
    if (await cordialCheckbox.isVisible() && await cordialCheckbox.isChecked()) {
      await cordialCheckbox.uncheck();
    }

    // Use Simple conformer method (fastest)
    const conformerSelect = window.locator('select').filter({
      has: window.locator('option', { hasText: 'Simple' }),
    });
    await conformerSelect.selectOption({ label: 'Simple' });

    // Start docking
    await window.locator('.btn.btn-primary', { hasText: /Start Docking/i }).click();

    // Wait for docking to complete (progress page shows "View Results")
    const viewResultsBtn = window.locator('.btn.btn-primary', { hasText: /View Results/i });
    await expect(viewResultsBtn).toBeVisible({ timeout: 120_000 });
    await viewResultsBtn.click();
    await window.waitForTimeout(1_000);

    // Now on results page — verify table with Vina column
    const table = window.locator('table');
    await expect(table).toBeVisible({ timeout: 5_000 });

    const headerText = await table.locator('thead').textContent();
    expect(headerText?.toLowerCase()).toContain('vina');

    // At least one result row
    const rows = table.locator('tbody tr');
    expect(await rows.count()).toBeGreaterThan(0);

    // First row should have a numeric Vina score (negative kcal/mol typically)
    const firstRowText = await rows.first().textContent();
    expect(firstRowText).toMatch(/-?\d+\.\d/); // e.g., -6.3

    // View 3D button should be visible
    await expect(window.locator('.btn', { hasText: /View 3D/i })).toBeVisible();

    // --- Sorting: click Vina column header to sort ---
    const vinaHeader = table.locator('th', { hasText: /Vina/i });
    await vinaHeader.click();
    await window.waitForTimeout(500);
    // Sort indicator should appear (▲ or ▼)
    const headerAfterSort = await vinaHeader.textContent();
    expect(headerAfterSort).toMatch(/[▲▼]/);

    // --- View 3D: click → viewer loads with NGL component ---
    await window.locator('.btn', { hasText: /View 3D/i }).click();
    await window.waitForTimeout(2_000);

    // Should switch to viewer mode
    const viewTab = window.locator('.tab.tab-sm', { hasText: 'View' });
    await expect(viewTab).toHaveClass(/tab-active/, { timeout: 5_000 });

    // NGL stage should have loaded a component
    const compCount = await window.evaluate(() => {
      const stage = (window as any).__nglStage;
      return stage ? stage.compList.length : 0;
    });
    expect(compCount).toBeGreaterThan(0);
  });

  test('results: Simulate navigates to MD configure with docked pose', async ({ window }) => {
    test.setTimeout(180_000);
    await navigateToConfigure(window);

    // Minimal params for speed
    await window.evaluate(() => {
      const store = (window as any).__emberStore;
      if (store) store.setDockConfig({ exhaustiveness: 1, poses: 1 });
    });
    await window.waitForTimeout(300);

    const refinementCheckbox = window.locator('label', { hasText: /Pocket refinement/i }).locator('input[type="checkbox"]');
    if (await refinementCheckbox.isChecked()) await refinementCheckbox.uncheck();

    const cordialCheckbox = window.locator('label', { hasText: /CORDIAL/i }).locator('input[type="checkbox"]');
    if (await cordialCheckbox.isVisible() && await cordialCheckbox.isChecked()) await cordialCheckbox.uncheck();

    const conformerSelect = window.locator('select').filter({ has: window.locator('option', { hasText: 'Simple' }) });
    await conformerSelect.selectOption({ label: 'Simple' });

    await window.locator('.btn.btn-primary', { hasText: /Start Docking/i }).click();

    const viewResultsBtn = window.locator('.btn.btn-primary', { hasText: /View Results/i });
    await expect(viewResultsBtn).toBeVisible({ timeout: 120_000 });
    await viewResultsBtn.click();
    await window.waitForTimeout(1_000);

    // First pose is auto-selected (index 0) — Simulate button should be visible
    const simulateBtn = window.locator('.btn.btn-outline', { hasText: /^Simulate$/ });
    await expect(simulateBtn).toBeVisible({ timeout: 5_000 });

    // Click Simulate
    await simulateBtn.click();
    await window.waitForTimeout(1_000);

    // Should switch to Simulate tab
    const simulateTab = window.locator('.tab.tab-sm', { hasText: 'Simulate' });
    await expect(simulateTab).toHaveClass(/tab-active/, { timeout: 5_000 });

    // Store state: mode='md', step='md-configure'
    const mdState = await window.evaluate(() => {
      const store = (window as any).__emberStore;
      const s = store.state();
      return {
        mode: s.mode,
        mdStep: s.mdStep,
        receptorPdb: s.md.receptorPdb,
        ligandSdf: s.md.ligandSdf,
      };
    });
    expect(mdState.mode).toBe('md');
    expect(mdState.mdStep).toBe('md-configure');
    expect(mdState.receptorPdb).toBeTruthy();
    expect(mdState.ligandSdf).toBeTruthy();
    expect(mdState.ligandSdf).toContain('.sdf');
  });
});
