import { onCleanup, onMount } from 'solid-js';
import type { OutputData } from '../../shared/types/ipc';

/**
 * Hook to subscribe to dock output events
 */
export function useDockOutput(callback: (data: OutputData) => void) {
  onMount(() => {
    const cleanup = window.electronAPI.onDockOutput(callback);
    onCleanup(cleanup);
  });
}

/**
 * Hook to subscribe to MD output events
 */
export function useMdOutput(callback: (data: OutputData) => void) {
  onMount(() => {
    const cleanup = window.electronAPI.onMdOutput(callback);
    onCleanup(cleanup);
  });
}
