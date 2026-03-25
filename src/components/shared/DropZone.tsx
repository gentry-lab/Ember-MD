// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, JSXElement, createSignal } from 'solid-js';

interface DropZoneProps {
  /** Called with resolved filesystem paths of dropped files */
  onFiles: (paths: string[]) => void;
  /** Allowed lowercase extensions (e.g., ['.pdb', '.cif']). Empty = accept all. */
  accept?: string[];
  /** If true, also accept dropped folders (resolves contained files) */
  acceptFolders?: boolean;
  /** Custom label shown during drag hover */
  hoverLabel?: string;
  /** Whether drop zone is disabled */
  disabled?: boolean;
  children: JSXElement;
}

const DropZone: Component<DropZoneProps> = (props) => {
  const [isDragOver, setIsDragOver] = createSignal(false);
  let dragCounter = 0;

  const matchesExtension = (filename: string): boolean => {
    if (!props.accept || props.accept.length === 0) return true;
    const lower = filename.toLowerCase();
    return props.accept.some(ext => lower.endsWith(ext));
  };

  const handleDragEnter = (e: DragEvent) => {
    if (props.disabled) return;
    e.preventDefault();
    e.stopPropagation();
    dragCounter++;
    if (dragCounter === 1) setIsDragOver(true);
  };

  const handleDragLeave = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter--;
    if (dragCounter === 0) setIsDragOver(false);
  };

  const handleDragOver = (e: DragEvent) => {
    if (props.disabled) return;
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter = 0;
    setIsDragOver(false);

    if (props.disabled || !e.dataTransfer?.files.length) return;

    const api = window.electronAPI;
    const paths: string[] = [];

    for (let i = 0; i < e.dataTransfer.files.length; i++) {
      const file = e.dataTransfer.files[i];
      const filePath = api.getPathForFile(file);
      if (!filePath) continue;

      if (props.acceptFolders || matchesExtension(file.name)) {
        paths.push(filePath);
      }
    }

    if (paths.length > 0) {
      props.onFiles(paths);
    }
  };

  return (
    <div
      class="relative"
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {props.children}
      {isDragOver() && !props.disabled && (
        <div class="absolute inset-0 z-50 flex items-center justify-center rounded-lg border-2 border-dashed border-primary bg-primary/10 backdrop-blur-[1px] pointer-events-none">
          <span class="text-sm font-medium text-primary">
            {props.hoverLabel || 'Drop files here'}
          </span>
        </div>
      )}
    </div>
  );
};

export default DropZone;
