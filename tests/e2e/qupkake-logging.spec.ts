import { test, expect } from './fixtures';
import * as fs from 'fs/promises';
import * as os from 'os';
import * as path from 'path';
import type { QupkakeCapabilityResult } from '../../shared/types/ipc';

async function getNewestEmberLogPath(): Promise<string> {
  const logsDir = path.join(os.homedir(), 'Ember', 'logs');
  const entries = await fs.readdir(logsDir);
  const logEntries = await Promise.all(
    entries
      .filter((entry) => entry.startsWith('ember-') && entry.endsWith('.log'))
      .map(async (entry) => {
        const fullPath = path.join(logsDir, entry);
        const stats = await fs.stat(fullPath);
        return { fullPath, mtimeMs: stats.mtimeMs };
      })
  );
  logEntries.sort((a, b) => b.mtimeMs - a.mtimeMs);
  if (logEntries.length === 0) {
    throw new Error(`No Ember logs found in ${logsDir}`);
  }
  return logEntries[0].fullPath;
}

test.describe('QupKake logging', () => {
  test('viewer startup enables validated QupKake runtimes without logging validation failures', async ({ window }) => {
    test.setTimeout(45_000);

    await expect(window.locator('.tab.tab-sm', { hasText: 'View' })).toHaveClass(/tab-active/);

    await expect.poll(async () => {
      const capability = await window.evaluate(() => window.electronAPI.checkQupkakeInstalled());
      return capability.validated;
    }, {
      timeout: 30_000,
      message: 'Expected QupKake capability check to complete successfully',
    }).toBe(true);

    const result = await window.evaluate(() => window.electronAPI.checkQupkakeInstalled()) as QupkakeCapabilityResult;
    expect(result.available).toBe(true);
    expect(result.validated).toBe(true);
    expect(result.runtimeFingerprint?.fukuiCompatibilityMode).toBe('flip_f0');
    expect(result.validationReport?.validated).toBe(true);

    const logPath = await getNewestEmberLogPath();

    await expect.poll(async () => {
      const logText = await fs.readFile(logPath, 'utf8');
      return {
        hasInvalidRuntimeWarning: logText.includes('[QupKake] capability check reported unavailable or unvalidated runtime'),
        hasProcessFailure: logText.includes('[QupKake] capability check process failed'),
        hasParseFailure: logText.includes('[QupKake] capability check output parse failed'),
      };
    }, {
      timeout: 30_000,
      message: `Expected QupKake startup to avoid capability failures in ${logPath}`,
    }).toEqual({
      hasInvalidRuntimeWarning: false,
      hasProcessFailure: false,
      hasParseFailure: false,
    });

    const logText = await fs.readFile(logPath, 'utf8');
    expect(logText).not.toContain('[QupKake] capability check reported unavailable or unvalidated runtime');
    expect(logText).not.toContain('[QupKake] capability check process failed');
    expect(logText).not.toContain('[QupKake] capability check output parse failed');
  });
});
