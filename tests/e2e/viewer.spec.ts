// Copyright (c) 2026 Ember Contributors. MIT License.
import { test, expect } from './fixtures';

test.describe('Viewer mode', () => {
  test('shows import/recent segmented control when empty', async ({ window }) => {
    await expect(window.locator('.tab.tab-sm', { hasText: 'View' })).toHaveClass(/tab-active/);
    // Should show import or recent jobs UI
    const importTab = window.locator('.tab', { hasText: 'Import' });
    const recentTab = window.locator('.tab', { hasText: 'Recent Jobs' });
    const importVisible = await importTab.isVisible().catch(() => false);
    const recentVisible = await recentTab.isVisible().catch(() => false);
    expect(importVisible || recentVisible).toBe(true);
  });

  test('viewer mode is the default active mode', async ({ window }) => {
    const viewTab = window.locator('.tab.tab-sm', { hasText: 'View' });
    await expect(viewTab).toHaveClass(/tab-active/);
  });
});
