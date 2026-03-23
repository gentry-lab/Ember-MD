import { test, expect } from './fixtures';

test.describe('Docking mode', () => {
  test('dock tab exists and is visible', async ({ window }) => {
    const dockTab = window.locator('.tab.tab-sm', { hasText: 'Dock' });
    await expect(dockTab).toBeVisible();
  });

  test('dock tab is disabled without project', async ({ window }) => {
    const dockTab = window.locator('.tab.tab-sm', { hasText: 'Dock' });
    await expect(dockTab).toBeDisabled();
  });
});
