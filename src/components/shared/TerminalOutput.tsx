import { Component, createSignal, createEffect } from 'solid-js';

interface TerminalOutputProps {
  title: string;
  logs: string;
}

const TerminalOutput: Component<TerminalOutputProps> = (props) => {
  const [copied, setCopied] = createSignal(false);
  // eslint-disable-next-line no-unassigned-vars -- SolidJS ref pattern
  let outputRef: HTMLPreElement | undefined;

  const handleCopyLogs = async () => {
    try {
      await navigator.clipboard.writeText(props.logs);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  createEffect(() => {
    const _ = props.logs;
    if (outputRef) outputRef.scrollTop = outputRef.scrollHeight;
  });

  return (
    <div class="flex-1 card bg-base-300 overflow-hidden relative">
      <div class="flex items-center justify-between px-3 py-1.5 bg-base-200 border-b border-base-100">
        <span class="font-mono text-xs text-base-content/90">{props.title}</span>
        <div class="flex gap-1">
          <div class="w-2.5 h-2.5 rounded-full bg-error/60" />
          <div class="w-2.5 h-2.5 rounded-full bg-warning/60" />
          <div class="w-2.5 h-2.5 rounded-full bg-success/60" />
        </div>
      </div>
      <button
        class={`absolute top-10 right-2 btn btn-xs ${copied() ? 'btn-success' : 'btn-ghost'} gap-1`}
        onClick={handleCopyLogs}
        title="Copy logs"
      >
        {copied() ? (
          <>
            <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
            </svg>
            Copied
          </>
        ) : (
          <>
            <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            Copy
          </>
        )}
      </button>
      <pre
        ref={outputRef}
        class="terminal-output flex-1 p-3 overflow-auto text-info whitespace-pre-wrap"
      >
        {props.logs || 'Waiting for output...'}
      </pre>
    </div>
  );
};

export default TerminalOutput;
