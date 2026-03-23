/**
 * Cross-mode integration tests.
 * Verifies state consistency across mode switches and tab navigation.
 */
import { test, expect, createTestProject } from './fixtures';

test.describe('Cross-mode integration', () => {
  test.beforeEach(async ({ window }) => {
    await createTestProject(window, '__e2e_integration__');
  });

  test('switch all tabs and return to View — no crash, state preserved', async ({ window }) => {
    test.setTimeout(30_000);

    // Start in View mode (default)
    const viewTab = window.locator('.tab.tab-sm', { hasText: 'View' });
    const mcmmTab = window.locator('.tab.tab-sm', { hasText: 'MCMM' });
    const dockTab = window.locator('.tab.tab-sm', { hasText: 'Dock' });
    const simTab = window.locator('.tab.tab-sm', { hasText: 'Simulate' });

    // Verify all tabs are enabled
    await expect(viewTab).toBeVisible();
    await expect(mcmmTab).toBeVisible();
    await expect(dockTab).toBeVisible();
    await expect(simTab).toBeVisible();

    // Record initial mode
    const initialMode = await window.evaluate(() => {
      return (window as any).__emberStore.state().mode;
    });
    expect(initialMode).toBe('viewer');

    // Switch to MCMM
    await mcmmTab.click();
    await window.waitForTimeout(500);
    let mode = await window.evaluate(() => (window as any).__emberStore.state().mode);
    expect(mode).toBe('conform');

    // Switch to Dock
    await dockTab.click();
    await window.waitForTimeout(500);
    mode = await window.evaluate(() => (window as any).__emberStore.state().mode);
    expect(mode).toBe('dock');

    // Switch to Simulate
    await simTab.click();
    await window.waitForTimeout(500);
    mode = await window.evaluate(() => (window as any).__emberStore.state().mode);
    expect(mode).toBe('md');

    // Return to View
    await viewTab.click();
    await window.waitForTimeout(500);
    mode = await window.evaluate(() => (window as any).__emberStore.state().mode);
    expect(mode).toBe('viewer');

    // App should be responsive — no error alerts
    await expect(window.locator('.alert.alert-error')).not.toBeVisible();

    // NGL stage should still exist
    const nglExists = await window.evaluate(() => {
      return (window as any).__nglStage != null;
    });
    expect(nglExists).toBe(true);
  });
});
