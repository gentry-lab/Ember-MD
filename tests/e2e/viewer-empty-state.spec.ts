// Copyright (c) 2026 Ember Contributors. MIT License.
import { test, expect, createTestProject } from './fixtures';

test.describe('Viewer empty state', () => {
  test('fresh project shows viewer import UI when switching to View', async ({ window }) => {
    await createTestProject(window, '__e2e_viewer_empty__');

    await window.getByRole('button', { name: 'View', exact: true }).click();

    await expect(window.locator('.tabs').getByRole('button', { name: 'Import', exact: true }).first()).toBeVisible();
    await expect(window.getByRole('button', { name: 'Recent Jobs', exact: true })).toBeVisible();
    await expect(window.getByTestId('project-table-import')).toBeVisible();
  });

  test('fresh project shows empty project table shell in View', async ({ window }) => {
    await createTestProject(window, '__e2e_viewer_table__');

    await window.getByRole('button', { name: 'View', exact: true }).click();

    await expect(window.getByTestId('project-table')).toBeVisible();
    await expect(window.getByText('No project rows yet')).toBeVisible();
    await expect(window.getByTestId('project-table-import')).toBeVisible();
    await expect(window.getByTestId('project-table-export')).toBeVisible();
    await expect(window.getByTestId('project-table-transfer')).toBeVisible();
  });
});
