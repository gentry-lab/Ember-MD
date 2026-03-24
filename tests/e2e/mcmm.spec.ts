// Copyright (c) 2026 Ember Contributors. MIT License.
import { test, expect } from './fixtures';

test.describe('MCMM mode', () => {
  test('MCMM tab exists and is visible', async ({ window }) => {
    const mcmmTab = window.locator('.tab.tab-sm', { hasText: 'MCMM' });
    await expect(mcmmTab).toBeVisible();
  });

  test('MCMM tab is disabled without project', async ({ window }) => {
    const mcmmTab = window.locator('.tab.tab-sm', { hasText: 'MCMM' });
    await expect(mcmmTab).toBeDisabled();
  });
});
