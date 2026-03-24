import { test, expect } from './fixtures';

test.describe('Score X-ray Pose mode', () => {
  test('Score X-ray Pose tab exists and is visible', async ({ window }) => {
    const scoreTab = window.locator('.tab.tab-sm', { hasText: 'Score X-ray Pose' });
    await expect(scoreTab).toBeVisible();
  });

  test('Score X-ray Pose tab is disabled without project', async ({ window }) => {
    const scoreTab = window.locator('.tab.tab-sm', { hasText: 'Score X-ray Pose' });
    await expect(scoreTab).toBeDisabled();
  });
});
