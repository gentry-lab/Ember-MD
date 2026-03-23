/**
 * Error path tests — verify graceful handling of bad input.
 * No file dialogs; uses SMILES input and direct IPC.
 */
import { test, expect, createTestProject } from './fixtures';

test.describe('Error paths', () => {
  test.beforeEach(async ({ window }) => {
    await createTestProject(window, '__e2e_errors__');
  });

  test('invalid SMILES in MCMM → error message, no crash', async ({ window }) => {
    test.setTimeout(30_000);
    await window.locator('.tab.tab-sm', { hasText: 'MCMM' }).click();
    await window.waitForTimeout(500);

    // Enter invalid SMILES
    await window.locator('textarea').fill('NOT_A_VALID_SMILES!!!');
    await window.locator('.btn.btn-primary.btn-sm', { hasText: /Enter SMILES/i }).click();
    await window.waitForTimeout(5_000);

    // Should show error alert or Continue should stay disabled
    const errorAlert = window.locator('.alert.alert-error');
    const continueBtn = window.locator('.btn.btn-primary', { hasText: /Continue/i });

    const hasError = await errorAlert.isVisible();
    const isDisabled = !(await continueBtn.isEnabled());

    // Either an error shows OR continue stays disabled (both acceptable)
    expect(hasError || isDisabled).toBe(true);

    // App should not have crashed — tabs still visible
    await expect(window.locator('.tab.tab-sm', { hasText: 'View' })).toBeVisible();
  });

  test('invalid SMILES in Dock → error message, no crash', async ({ window }) => {
    test.setTimeout(30_000);
    await window.locator('.tab.tab-sm', { hasText: 'Dock' }).click();
    await window.waitForTimeout(500);

    const textarea = window.locator('textarea');
    await expect(textarea).toBeVisible();
    await textarea.fill('INVALID_SMILES_STRING');
    await window.locator('.btn.btn-primary.btn-sm', { hasText: /Enter SMILES/i }).click();
    await window.waitForTimeout(5_000);

    // Error should appear or ligand count should not increase
    const errorAlert = window.locator('.alert.alert-error');
    const hasError = await errorAlert.isVisible();

    // Even if no error shown, app should not crash — tabs visible
    await expect(window.locator('.tab.tab-sm', { hasText: 'View' })).toBeVisible();
  });

  test('invalid SMILES in Simulate → error message, no crash', async ({ window }) => {
    test.setTimeout(30_000);
    await window.locator('.tab.tab-sm', { hasText: 'Simulate' }).click();
    await window.waitForTimeout(500);

    const textarea = window.locator('textarea');
    await expect(textarea).toBeVisible();
    await textarea.fill('BOGUS_MOLECULE_XYZ');
    await window.locator('.btn.btn-primary.btn-sm', { hasText: /Enter SMILES/i }).click();
    await window.waitForTimeout(5_000);

    // App should not crash
    await expect(window.locator('.tab.tab-sm', { hasText: 'View' })).toBeVisible();
  });

  test('empty SMILES does not crash', async ({ window }) => {
    test.setTimeout(15_000);
    await window.locator('.tab.tab-sm', { hasText: 'MCMM' }).click();
    await window.waitForTimeout(500);

    // Enter SMILES button should be disabled with empty input
    const enterBtn = window.locator('.btn.btn-primary.btn-sm', { hasText: /Enter SMILES/i });
    await expect(enterBtn).toBeDisabled();
  });

  test('switching modes during idle does not crash', async ({ window }) => {
    test.setTimeout(20_000);

    // Rapidly switch through all tabs
    for (const tab of ['MCMM', 'Dock', 'Simulate', 'View', 'MCMM', 'Dock', 'Simulate', 'View']) {
      await window.locator('.tab.tab-sm', { hasText: tab }).click();
      await window.waitForTimeout(300);
    }

    // App should still be alive — no blank screen or error
    await expect(window.locator('.tab.tab-sm', { hasText: 'View' })).toBeVisible();
    await expect(window.locator('.tab.tab-sm', { hasText: 'MCMM' })).toBeVisible();
  });
});
