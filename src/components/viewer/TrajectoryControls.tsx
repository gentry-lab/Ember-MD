import { Component, Show, createEffect, onCleanup } from 'solid-js';
import { workflowStore } from '../../stores/workflow';
import type { PlaybackSpeed, CenterTarget } from '../../stores/workflow';

interface TrajectoryControlsProps {
  onSeek: (frame: number) => void;
  onPlay: () => void;
  onPause: () => void;
  onStepForward: () => void;
  onStepBackward: () => void;
  onFirstFrame: () => void;
  onLastFrame: () => void;
}

const TrajectoryControls: Component<TrajectoryControlsProps> = (props) => {
  const {
    state,
    setViewerPlaybackSpeed,
    setViewerSmoothing,
    setViewerLoopPlayback,
    setViewerCenterTarget,
    setViewerCurrentFrame,
  } = workflowStore;

  const trajectoryInfo = () => state().viewer.trajectoryInfo;
  const currentFrame = () => state().viewer.currentFrame;
  const isPlaying = () => state().viewer.isPlaying;
  const playbackSpeed = () => state().viewer.playbackSpeed;
  const smoothing = () => state().viewer.smoothing;
  const loopPlayback = () => state().viewer.loopPlayback;
  const centerTarget = () => state().viewer.centerTarget;

  // Format time from frame index
  const formatTime = (frame: number): string => {
    const info = trajectoryInfo();
    if (!info) return '0.00 ns';
    const timeNs = (frame * info.timestepPs) / 1000;
    return `${timeNs.toFixed(2)} ns`;
  };

  // Handle slider change
  const handleSliderChange = (e: Event) => {
    const target = e.target as HTMLInputElement;
    const frame = parseInt(target.value, 10);
    setViewerCurrentFrame(frame);
    props.onSeek(frame);
  };

  // Handle speed change
  const handleSpeedChange = (e: Event) => {
    const target = e.target as HTMLSelectElement;
    setViewerPlaybackSpeed(parseFloat(target.value) as PlaybackSpeed);
  };

  // Handle smoothing change
  const handleSmoothingChange = (e: Event) => {
    const target = e.target as HTMLInputElement;
    setViewerSmoothing(parseInt(target.value, 10));
  };

  // Handle center target change
  const handleCenterTargetChange = (target: CenterTarget) => {
    setViewerCenterTarget(target);
  };

  // Keyboard shortcuts
  const handleKeyDown = (e: KeyboardEvent) => {
    // Only handle if no input is focused
    if ((e.target as HTMLElement).tagName === 'INPUT') return;

    switch (e.key) {
      case ' ':
        e.preventDefault();
        if (isPlaying()) {
          props.onPause();
        } else {
          props.onPlay();
        }
        break;
      case 'ArrowLeft':
        e.preventDefault();
        props.onStepBackward();
        break;
      case 'ArrowRight':
        e.preventDefault();
        props.onStepForward();
        break;
      case 'Home':
        e.preventDefault();
        props.onFirstFrame();
        break;
      case 'End':
        e.preventDefault();
        props.onLastFrame();
        break;
    }
  };

  // Add keyboard listener
  createEffect(() => {
    if (trajectoryInfo()) {
      window.addEventListener('keydown', handleKeyDown);
    }

    onCleanup(() => {
      window.removeEventListener('keydown', handleKeyDown);
    });
  });

  return (
    <Show when={trajectoryInfo()}>
      <div class="card bg-base-200 p-2">
        <div class="flex flex-col gap-2">
          {/* Playback controls row */}
          <div class="flex items-center gap-2">
            {/* Transport buttons */}
            <div class="btn-group">
              <button
                class="btn btn-xs btn-ghost"
                onClick={() => props.onFirstFrame()}
                title="First frame (Home)"
              >
                <svg xmlns="http://www.w3.org/2000/svg" class="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                  <path d="M10 18a8 8 0 100-16 8 8 0 000 16zm1.5-13h-4a.5.5 0 00-.5.5v9a.5.5 0 00.5.5h4a.5.5 0 00.5-.5v-9a.5.5 0 00-.5-.5z" />
                  <path fill-rule="evenodd" d="M15.854 5.646a.5.5 0 010 .708L11.707 10l4.147 4.146a.5.5 0 01-.708.708l-4.5-4.5a.5.5 0 010-.708l4.5-4.5a.5.5 0 01.708 0z" clip-rule="evenodd" />
                </svg>
              </button>
              <button
                class="btn btn-xs btn-ghost"
                onClick={() => props.onStepBackward()}
                title="Previous frame (Left arrow)"
              >
                <svg xmlns="http://www.w3.org/2000/svg" class="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clip-rule="evenodd" />
                </svg>
              </button>
              <button
                class={`btn btn-xs ${isPlaying() ? 'btn-primary' : 'btn-ghost'}`}
                onClick={() => isPlaying() ? props.onPause() : props.onPlay()}
                title={isPlaying() ? 'Pause (Space)' : 'Play (Space)'}
              >
                <Show when={isPlaying()} fallback={
                  <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clip-rule="evenodd" />
                  </svg>
                }>
                  <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z" clip-rule="evenodd" />
                  </svg>
                </Show>
              </button>
              <button
                class="btn btn-xs btn-ghost"
                onClick={() => props.onStepForward()}
                title="Next frame (Right arrow)"
              >
                <svg xmlns="http://www.w3.org/2000/svg" class="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clip-rule="evenodd" />
                </svg>
              </button>
              <button
                class="btn btn-xs btn-ghost"
                onClick={() => props.onLastFrame()}
                title="Last frame (End)"
              >
                <svg xmlns="http://www.w3.org/2000/svg" class="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                  <path d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.5 5h4a.5.5 0 01.5.5v9a.5.5 0 01-.5.5h-4a.5.5 0 01-.5-.5v-9a.5.5 0 01.5-.5z" />
                  <path fill-rule="evenodd" d="M4.146 5.646a.5.5 0 000 .708L8.293 10l-4.147 4.146a.5.5 0 00.708.708l4.5-4.5a.5.5 0 000-.708l-4.5-4.5a.5.5 0 00-.708 0z" clip-rule="evenodd" />
                </svg>
              </button>
            </div>

            {/* Frame counter */}
            <div class="text-xs font-mono">
              <span class="text-primary">{currentFrame() + 1}</span>
              <span class="text-base-content/80">/{trajectoryInfo()?.frameCount || 0}</span>
            </div>

            {/* Time display */}
            <div class="text-xs text-base-content/90">
              {formatTime(currentFrame())}
            </div>

            <div class="flex-1" />

            {/* Speed selector */}
            <div class="flex items-center gap-1">
              <span class="text-xs text-base-content/90">Speed:</span>
              <select
                class="select select-xs select-bordered w-16"
                value={playbackSpeed()}
                onChange={handleSpeedChange}
              >
                <option value="0.25">0.25x</option>
                <option value="0.5">0.5x</option>
                <option value="1">1x</option>
                <option value="2">2x</option>
                <option value="4">4x</option>
              </select>
            </div>

            {/* Loop toggle */}
            <label class="label cursor-pointer gap-1 p-0">
              <input
                type="checkbox"
                class="checkbox checkbox-xs checkbox-primary"
                checked={loopPlayback()}
                onChange={() => setViewerLoopPlayback(!loopPlayback())}
              />
              <span class="text-xs">Loop</span>
            </label>
          </div>

          {/* Timeline slider */}
          <div class="flex items-center gap-2">
            <input
              type="range"
              class="range range-xs range-primary flex-1"
              min="0"
              max={(trajectoryInfo()?.frameCount || 1) - 1}
              value={currentFrame()}
              onInput={handleSliderChange}
            />
          </div>

          {/* Settings row */}
          <div class="flex items-center gap-3 flex-wrap">
            {/* Smoothing */}
            <div class="flex items-center gap-1">
              <span class="text-xs text-base-content/90">Smoothing:</span>
              <input
                type="range"
                class="range range-xs w-16"
                min="1"
                max="10"
                value={smoothing()}
                onInput={handleSmoothingChange}
              />
              <span class="text-xs font-mono w-4">{smoothing()}</span>
            </div>

            {/* Center on */}
            <div class="flex items-center gap-1">
              <span class="text-xs text-base-content/90">Center:</span>
              <div class="btn-group">
                <button
                  class={`btn btn-xs ${centerTarget() === 'ligand' ? 'btn-primary' : 'btn-ghost'}`}
                  onClick={() => handleCenterTargetChange('ligand')}
                >
                  Ligand
                </button>
                <button
                  class={`btn btn-xs ${centerTarget() === 'protein' ? 'btn-primary' : 'btn-ghost'}`}
                  onClick={() => handleCenterTargetChange('protein')}
                >
                  Protein
                </button>
                <button
                  class={`btn btn-xs ${centerTarget() === 'none' ? 'btn-primary' : 'btn-ghost'}`}
                  onClick={() => handleCenterTargetChange('none')}
                >
                  None
                </button>
              </div>
            </div>

            {/* Total duration */}
            <div class="text-xs text-base-content/80 ml-auto">
              Total: {trajectoryInfo()?.totalTimeNs.toFixed(2)} ns
            </div>
          </div>
        </div>
      </div>
    </Show>
  );
};

export default TrajectoryControls;
