// Copyright (c) 2026 Ember Contributors. MIT License.
import { test, expect } from './fixtures';

test.describe('Score X-ray Pose mode', () => {
  test('Analyze X-ray tab exists and is visible', async ({ window }) => {
    const scoreTab = window.locator('.tab.tab-sm', { hasText: 'Analyze X-ray' });
    await expect(scoreTab).toBeVisible();
  });

  test('Analyze X-ray tab is disabled without project', async ({ window }) => {
    const scoreTab = window.locator('.tab.tab-sm', { hasText: 'Analyze X-ray' });
    await expect(scoreTab).toBeDisabled();
  });
});
