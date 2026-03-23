import { test, expect } from './fixtures';

test.describe('Simulate mode', () => {
  test('Simulate tab exists and is visible', async ({ window }) => {
    const simTab = window.locator('.tab.tab-sm', { hasText: 'Simulate' });
    await expect(simTab).toBeVisible();
  });

  test('Simulate tab is disabled without project', async ({ window }) => {
    const simTab = window.locator('.tab.tab-sm', { hasText: 'Simulate' });
    await expect(simTab).toBeDisabled();
  });
});
