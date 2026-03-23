import { test, expect } from './fixtures';

test.describe('App boot', () => {
  test('launches without crash', async ({ window }) => {
    const title = await window.title();
    expect(title).toBeTruthy();
  });

  test('header shows all four mode tabs', async ({ window }) => {
    const tabs = window.locator('.tabs-boxed .tab.tab-sm');
    await expect(tabs).toHaveCount(4);

    const labels = await tabs.allTextContents();
    expect(labels).toEqual(['View', 'MCMM', 'Dock', 'Simulate']);
  });

  test('View tab is active by default', async ({ window }) => {
    const viewTab = window.locator('.tab', { hasText: 'View' });
    await expect(viewTab).toHaveClass(/tab-active/);
  });

  test('no console errors on boot', async ({ app }) => {
    const errors: string[] = [];
    const window = await app.firstWindow();
    window.on('console', msg => {
      if (msg.type() === 'error') errors.push(msg.text());
    });
    // Give the app a moment to settle
    await window.waitForTimeout(2000);
    // Filter out known noise (NGL WebGL warnings, etc.)
    const real = errors.filter(e =>
      !e.includes('WebGL') &&
      !e.includes('THREE') &&
      !e.includes('deprecated')
    );
    expect(real).toEqual([]);
  });
});
