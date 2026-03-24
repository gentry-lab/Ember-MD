import { test, expect, createTestProject } from './fixtures';
import type { Page } from '@playwright/test';

const MULTI_FRAME_SYSTEM_PDB = '/Users/controller/Ember/rustic-ligand-lemur/simulations/ff19sb-OPC_MD-300K-1ns/system.pdb';
const MULTI_FRAME_TRAJECTORY_DCD = '/Users/controller/Ember/rustic-ligand-lemur/simulations/ff19sb-OPC_MD-300K-1ns/trajectory.dcd';

async function setupViewer(window: Page): Promise<void> {
  await createTestProject(window, '__e2e_viewer_traj__');
  const viewTab = window.locator('.tab.tab-sm', { hasText: 'View' });
  await expect(viewTab).toHaveClass(/tab-active/);
  await window.waitForFunction(() => !!(window as any).__nglStage, null, { timeout: 10_000 });
}

async function getTrajectoryRenderState(window: Page) {
  return window.evaluate(() => {
    const s = (window as any).__emberStore.state();
    const telemetry = (window as any).__viewerTestState ?? {};
    return {
      frameCount: s.viewer.trajectoryInfo?.frameCount ?? 0,
      currentFrame: s.viewer.currentFrame,
      isPlaying: s.viewer.isPlaying,
      playbackSpeed: s.viewer.playbackSpeed,
      centerTarget: s.viewer.centerTarget,
      renderedFrameIndex: telemetry.renderedFrameIndex ?? null,
      coordinateSignature: telemetry.coordinateSignature ?? null,
    };
  });
}

test.describe('Viewer trajectory playback', () => {
  test.beforeEach(async ({ window }) => {
    await setupViewer(window);
  });

  test('multi-frame trajectory playback advances frames and supports controls', async ({ window }) => {
    test.setTimeout(180_000);

    const fixtureAvailable = await window.evaluate(async (paths: { pdb: string; dcd: string }) => {
      const api = (window as any).electronAPI;
      return await api.fileExists(paths.pdb) && await api.fileExists(paths.dcd);
    }, { pdb: MULTI_FRAME_SYSTEM_PDB, dcd: MULTI_FRAME_TRAJECTORY_DCD });

    test.skip(!fixtureAvailable, 'Multi-frame trajectory fixture not available on this machine');

    await window.evaluate((paths: { pdb: string; dcd: string }) => {
      const store = (window as any).__emberStore;
      store.openViewerSession({
        pdbPath: paths.pdb,
        trajectoryPath: paths.dcd,
      });
    }, { pdb: MULTI_FRAME_SYSTEM_PDB, dcd: MULTI_FRAME_TRAJECTORY_DCD });

    await window.waitForFunction(() => {
      const s = (window as any).__emberStore.state();
      return !!s.viewer.trajectoryInfo && s.viewer.trajectoryInfo.frameCount > 1;
    }, null, { timeout: 45_000 });

    await expect(window.locator('[data-testid="trajectory-controls"]')).toBeVisible({ timeout: 10_000 });
    await window.waitForFunction(() => {
      const telemetry = (window as any).__viewerTestState;
      return telemetry?.renderedFrameIndex === 0 && Array.isArray(telemetry?.coordinateSignature);
    }, null, { timeout: 45_000 });

    const initialState = await getTrajectoryRenderState(window);
    expect(initialState.frameCount).toBeGreaterThan(1);
    expect(initialState.currentFrame).toBe(0);
    expect(initialState.renderedFrameIndex).toBe(0);
    expect(initialState.coordinateSignature).not.toBeNull();

    const playBtn = window.locator('[data-testid="trajectory-play"]');
    await playBtn.click();

    await window.waitForFunction(() => {
      const s = (window as any).__emberStore.state();
      const telemetry = (window as any).__viewerTestState;
      return s.viewer.isPlaying === true && s.viewer.currentFrame > 0 && telemetry?.renderedFrameIndex === s.viewer.currentFrame;
    }, null, { timeout: 20_000 });

    await playBtn.click();
    await window.waitForFunction(() => !(window as any).__emberStore.state().viewer.isPlaying, null, { timeout: 10_000 });

    const playedState = await getTrajectoryRenderState(window);
    const pausedFrame = playedState.currentFrame;
    expect(pausedFrame).toBeGreaterThan(0);
    expect(playedState.isPlaying).toBe(false);
    expect(playedState.renderedFrameIndex).toBe(playedState.currentFrame);
    expect(playedState.coordinateSignature).not.toEqual(initialState.coordinateSignature);

    await window.locator('[data-testid="trajectory-next"]').click();
    await window.waitForFunction((frame: number) => {
      const s = (window as any).__emberStore.state();
      const telemetry = (window as any).__viewerTestState;
      return s.viewer.currentFrame > frame && telemetry?.renderedFrameIndex === s.viewer.currentFrame;
    }, pausedFrame, { timeout: 10_000 });
    const steppedForward = await getTrajectoryRenderState(window);
    const steppedForwardFrame = steppedForward.currentFrame;
    expect(steppedForwardFrame).toBeGreaterThan(pausedFrame);
    expect(steppedForward.renderedFrameIndex).toBe(steppedForward.currentFrame);
    expect(steppedForward.coordinateSignature).not.toEqual(playedState.coordinateSignature);

    await window.locator('[data-testid="trajectory-prev"]').click();
    await window.waitForFunction((frame: number) => {
      const s = (window as any).__emberStore.state();
      const telemetry = (window as any).__viewerTestState;
      return s.viewer.currentFrame < frame && telemetry?.renderedFrameIndex === s.viewer.currentFrame;
    }, steppedForwardFrame, { timeout: 10_000 });
    const steppedBackward = await getTrajectoryRenderState(window);
    const steppedBackwardFrame = steppedBackward.currentFrame;
    expect(steppedBackwardFrame).toBeLessThan(steppedForwardFrame);
    expect(steppedBackward.renderedFrameIndex).toBe(steppedBackward.currentFrame);
    expect(steppedBackward.coordinateSignature).not.toEqual(steppedForward.coordinateSignature);

    await window.locator('[data-testid="trajectory-speed"]').selectOption('2');
    await window.waitForTimeout(300);
    const speedState = await getTrajectoryRenderState(window);
    expect(speedState.playbackSpeed).toBe(2);

    await window.locator('[data-testid="trajectory-center-ligand"]').click();
    await window.waitForTimeout(300);
    const centeredState = await getTrajectoryRenderState(window);
    expect(centeredState.centerTarget).toBe('ligand');
  });
});
