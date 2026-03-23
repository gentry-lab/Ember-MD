/**
 * Shared Playwright fixtures for Ember Electron app testing.
 */
import { test as base, _electron as electron, ElectronApplication, Page } from '@playwright/test';

export const test = base.extend<{ app: ElectronApplication; window: Page }>({
  app: async ({}, use) => {
    const app = await electron.launch({
      args: ['.'],
      env: { ...process.env, NODE_ENV: 'test' },
    });
    await use(app);
    await app.close();
  },
  window: async ({ app }, use) => {
    const window = await app.firstWindow();
    await window.waitForLoadState('domcontentloaded');
    // Wait for SolidJS to mount
    await window.waitForTimeout(1000);
    await use(window);
  },
});

export { expect } from '@playwright/test';

/** Create a project through the UI so tabs become enabled */
export async function createTestProject(window: Page, name: string = '__e2e_test__'): Promise<void> {
  // Find the "New Project" input in the project selector overlay
  const projectInput = window.locator('input[placeholder="project-name"]');
  await projectInput.waitFor({ state: 'visible', timeout: 10_000 });
  await projectInput.fill(name);
  // Click "Create" button
  const createBtn = window.locator('button', { hasText: 'Create' });
  await createBtn.click();
  // Wait for project to be set and tabs to become enabled
  await window.waitForTimeout(1000);
}
