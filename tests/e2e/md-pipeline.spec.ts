/**
 * MD simulation pipeline test.
 * Uses PDB ID fetch for receptor + SMILES for ligand (no file dialogs).
 */
import { test, expect, createTestProject } from './fixtures';
import type { Page } from '@playwright/test';

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
});
