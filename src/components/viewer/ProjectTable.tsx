// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, For, Show, createSignal, onCleanup, onMount } from 'solid-js';
import type { ViewerProjectColumn, ViewerProjectFamily, ViewerProjectTableState } from '../../stores/workflow';
import { getSortedRowsForFamily, getVisibleColumnsForFamily } from '../../utils/projectTable';
import ImportInputPanel from '../shared/ImportInputPanel';

interface ProjectTableProps {
  projectTable: ViewerProjectTableState;
  panelWidth: number;
  onSelectRow: (rowId: string) => void;
  onToggleRowSelection: (rowId: string) => void;
  onToggleFamilyCollapsed: (familyId: string) => void;
  onSortFamily: (familyId: string, columnKey: string) => void;
  onPlayTrajectory: (familyId: string) => void;
  onRemoveFamily: (familyId: string) => void;
  onRemoveRow: (rowId: string) => void;
  onRenameRow: (rowId: string, newLabel: string) => void;
  canNavigatePrevious: boolean;
  canNavigateNext: boolean;
  onNavigatePrevious: () => void;
  onNavigateNext: () => void;
  canTransfer: boolean;
  transferTooltip?: string;
  canExport: boolean;
  canTransferDock: boolean;
  transferDockTooltip?: string;
  canTransferMcmm: boolean;
  transferMcmmTooltip?: string;
  canTransferSimulate: boolean;
  transferSimulateTooltip?: string;
  onTransferDock: () => void;
  onTransferMcmm: () => void;
  onTransferSimulate: () => void;
  onExport: () => void;
  onBrowseImport: () => void;
  importPdbIdValue: string;
  onImportPdbIdInput: (value: string) => void;
  onFetchImportPdb: () => void;
  importPdbFetchDisabled: boolean;
  importPdbFetchLoading: boolean;
  importSmilesValue: string;
  onImportSmilesInput: (value: string) => void;
  onSubmitImportSmiles: () => void;
  importSmilesDisabled: boolean;
  importSmilesLoading: boolean;
  importSmilesCount: number;
  importDisabled: boolean;
  importLoading: boolean;
  canAlignProtein: boolean;
  canAlignLigand: boolean;
  canAlignSubstructure: boolean;
  onAlignProtein: () => void;
  onAlignLigand: () => void;
  onAlignSubstructure: () => void;
  alignSubstructureLabel: string | null;
  hasAlignment: boolean;
  onResetAlignment: () => void;
  onViewResults?: () => void;
  viewResultsDisabled?: boolean;
  viewResultsTooltip?: string;
}

const ChevronRight: Component = () => (
  <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
  </svg>
);

const ChevronLeft: Component = () => (
  <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
  </svg>
);

const ChevronDown: Component = () => (
  <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
  </svg>
);

const formatMetric = (value: string | number | null | undefined, column: ViewerProjectColumn): string => {
  if (value == null) return '-';
  if (typeof value === 'string') return value;
  if (column.kind === 'percent') return `${(value * 100).toFixed(0)}%`;
  return Number.isInteger(value) ? `${value}` : value.toFixed(1);
};

const ProjectTable: Component<ProjectTableProps> = (props) => {
  const [editingRowId, setEditingRowId] = createSignal<string | null>(null);
  const [editText, setEditText] = createSignal('');
  const [showImportPopover, setShowImportPopover] = createSignal(false);
  let importButtonRef: HTMLButtonElement | undefined;
  let importPopoverRef: HTMLDivElement | undefined;

  onMount(() => {
    const handleClickAway = (event: MouseEvent) => {
      const target = event.target;
      if (!(target instanceof Node)) return;
      if (importPopoverRef?.contains(target)) return;
      if (importButtonRef?.contains(target)) return;
      setShowImportPopover(false);
    };
    document.addEventListener('mousedown', handleClickAway);
    onCleanup(() => document.removeEventListener('mousedown', handleClickAway));
  });

  const startRenameRow = (rowId: string, currentLabel: string) => {
    setEditingRowId(rowId);
    setEditText(currentLabel);
  };

  const commitRename = () => {
    const rowId = editingRowId();
    const text = editText().trim();
    if (rowId && text) {
      const current = props.projectTable.rows.find((r) => r.id === rowId);
      if (current && current.label !== text) {
        props.onRenameRow(rowId, text);
      }
    }
    setEditingRowId(null);
  };

  const cancelRename = () => {
    setEditingRowId(null);
  };

  const sortIndicator = (family: ViewerProjectFamily, columnKey: string) => {
    if (family.sortKey !== columnKey) return '';
    return family.sortDirection === 'asc' ? ' ▲' : ' ▼';
  };

  const isSelected = (rowId: string) =>
    (props.projectTable.selectedRowIds || []).includes(rowId);

  const isActive = (rowId: string) =>
    props.projectTable.activeRowId === rowId;

  const handleRowClick = (rowId: string, event: MouseEvent) => {
    if (event.metaKey || event.ctrlKey) {
      props.onToggleRowSelection(rowId);
    } else {
      props.onSelectRow(rowId);
    }
  };

  return (
    <div class="card bg-base-200 h-full overflow-hidden" data-testid="project-table">
      <div class="px-3 py-2 border-b border-base-300 flex items-center justify-between">
        <div>
          <div class="text-sm font-semibold">Project Table</div>
          <div class="text-[10px] text-base-content/60">Viewer session structures</div>
        </div>
        <div class="flex items-center gap-1">
          <button
            class="btn btn-ghost btn-xs btn-square"
            onClick={props.onNavigatePrevious}
            disabled={!props.canNavigatePrevious}
            title="Previous structure"
            data-testid="project-table-nav-prev"
          >
            <ChevronLeft />
          </button>
          <button
            class="btn btn-ghost btn-xs btn-square"
            onClick={props.onNavigateNext}
            disabled={!props.canNavigateNext}
            title="Next structure"
            data-testid="project-table-nav-next"
          >
            <ChevronRight />
          </button>
        </div>
      </div>

      <div class="flex-1 overflow-auto">
        <Show
          when={props.projectTable.families.length > 0}
          fallback={
            <div class="h-full flex items-center justify-center px-5 text-center">
              <div class="space-y-1.5">
                <p class="text-sm font-semibold">No project rows yet</p>
                <p class="text-xs text-base-content/60">
                  Import a structure or add the current view to start building the project table.
                </p>
              </div>
            </div>
          }
        >
          <For each={props.projectTable.families}>
            {(family) => (
              <div class="border-b border-base-300 last:border-b-0" data-testid={`project-family-${family.id}`}>
                <div class="px-2 py-2 flex items-center gap-2 bg-base-100/60">
                  <button
                    class="btn btn-ghost btn-xs btn-square"
                    onClick={() => props.onToggleFamilyCollapsed(family.id)}
                    title={family.collapsed ? 'Expand family' : 'Collapse family'}
                  >
                    {family.collapsed ? <ChevronRight /> : <ChevronDown />}
                  </button>
                  <div class="flex-1 min-w-0">
                    <div class="text-xs font-semibold truncate">{family.title}</div>
                    <div class="text-[10px] text-base-content/55 capitalize">{family.jobType}</div>
                  </div>
                  <Show when={family.trajectoryPath}>
                    <button
                      class="btn btn-primary btn-xs"
                      onClick={() => props.onPlayTrajectory(family.id)}
                      data-testid={`project-family-action-play-${family.id}`}
                    >
                      Play
                    </button>
                  </Show>
                  <button
                    class="btn btn-ghost btn-xs btn-square text-error/70 hover:text-error hover:bg-error/10"
                    onClick={(e) => { e.stopPropagation(); props.onRemoveFamily(family.id); }}
                    title="Remove from table"
                    data-testid={`project-family-remove-${family.id}`}
                  >
                    <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>

                <Show when={!family.collapsed}>
                  <div class="overflow-hidden">
                    <table class="table table-xs table-fixed w-full">
                      <thead>
                        <tr class="bg-base-200/70">
                          <th class="w-6" />
                          <th class="w-[11rem]">Structure</th>
                          <For each={getVisibleColumnsForFamily(family, props.panelWidth)}>
                            {(column) => (
                              <th
                                class="cursor-pointer select-none text-right"
                                onClick={() => props.onSortFamily(family.id, column.key)}
                              >
                                {column.label}{sortIndicator(family, column.key)}
                              </th>
                            )}
                          </For>
                          <th class="w-5" />
                        </tr>
                      </thead>
                      <tbody>
                        <For each={getSortedRowsForFamily(
                          family,
                          props.projectTable.rows,
                          getVisibleColumnsForFamily(family, props.panelWidth),
                        )}>
                          {(row) => (
                            <tr
                              class={`cursor-pointer hover:bg-base-200 group/row ${
                                isActive(row.id)
                                  ? 'bg-primary/10'
                                  : isSelected(row.id)
                                    ? 'bg-primary/5'
                                    : ''
                              }`}
                              onClick={(e) => handleRowClick(row.id, e)}
                              data-testid={`project-row-${row.id}`}
                            >
                              <td>
                                <div
                                  class={`w-2 h-2 rounded-full mx-auto ${
                                    isActive(row.id)
                                      ? 'bg-primary'
                                      : isSelected(row.id)
                                        ? 'bg-primary/40'
                                        : 'bg-base-300'
                                  }`}
                                />
                              </td>
                              <td class="font-medium truncate">
                                <Show when={editingRowId() === row.id} fallback={
                                  <span
                                    title={row.label}
                                    onDblClick={(e) => { e.stopPropagation(); startRenameRow(row.id, row.label); }}
                                  >
                                    {row.label}
                                  </span>
                                }>
                                  <input
                                    type="text"
                                    class="input input-xs input-bordered w-full font-medium"
                                    value={editText()}
                                    onInput={(e) => setEditText(e.currentTarget.value)}
                                    onKeyDown={(e) => {
                                      if (e.key === 'Enter') { e.preventDefault(); commitRename(); }
                                      else if (e.key === 'Escape') { e.preventDefault(); cancelRename(); }
                                    }}
                                    onBlur={commitRename}
                                    onClick={(e) => e.stopPropagation()}
                                    ref={(el) => requestAnimationFrame(() => { el.focus(); el.select(); })}
                                  />
                                </Show>
                              </td>
                              <For each={getVisibleColumnsForFamily(family, props.panelWidth)}>
                                {(column) => (
                                  <td class="text-right font-mono truncate">
                                    {formatMetric(row.metrics[column.key], column)}
                                  </td>
                                )}
                              </For>
                              <td class="w-5 p-0">
                                <button
                                  class="btn btn-ghost btn-xs btn-square text-error/65 hover:text-error hover:bg-error/10"
                                  onClick={(e) => { e.stopPropagation(); props.onRemoveRow(row.id); }}
                                  title="Remove row"
                                >
                                  <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                                  </svg>
                                </button>
                              </td>
                            </tr>
                          )}
                        </For>
                      </tbody>
                    </table>
                  </div>
                </Show>
              </div>
            )}
          </For>
        </Show>
      </div>

      {/* Alignment toolbar — always visible, buttons enabled per selection */}
        <div class="px-3 py-1.5 border-t border-base-300">
          <div class="flex items-center gap-1">
            <span class="text-[10px] text-base-content/55 mr-1">Align</span>
            <div class="btn-group">
              <button
                class={`btn btn-xs ${props.canAlignProtein ? 'btn-outline' : 'btn-disabled'}`}
                disabled={!props.canAlignProtein}
                onClick={props.onAlignProtein}
                title="Align proteins by backbone (C-alpha)"
                data-testid="project-table-align-protein"
              >
                P
              </button>
              <button
                class={`btn btn-xs ${props.canAlignLigand ? 'btn-outline' : 'btn-disabled'}`}
                disabled={!props.canAlignLigand}
                onClick={props.onAlignLigand}
                title="Align ligands by maximum common substructure"
                data-testid="project-table-align-ligand"
              >
                L
              </button>
              <button
                class={`btn btn-xs ${props.canAlignSubstructure ? 'btn-outline' : 'btn-disabled'}`}
                disabled={!props.canAlignSubstructure}
                onClick={props.onAlignSubstructure}
                title={props.alignSubstructureLabel
                  ? `Align by ${props.alignSubstructureLabel}`
                  : 'Align by shared rigid substructure'}
                data-testid="project-table-align-substructure"
              >
                SS
              </button>
            </div>
            <Show when={props.hasAlignment}>
              <button
                class="btn btn-ghost btn-xs btn-square"
                onClick={props.onResetAlignment}
                title="Reset alignment"
                data-testid="project-table-align-reset"
              >
                <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              </button>
            </Show>
            <Show when={props.alignSubstructureLabel}>
              <span class="text-[10px] text-base-content/55 ml-1">{props.alignSubstructureLabel}</span>
            </Show>
          </div>
        </div>

      <div class="relative px-3 py-2 border-t border-base-300 flex flex-col gap-1.5">
        <Show when={props.onViewResults}>
          <button
            class="btn btn-primary btn-sm w-full"
            onClick={props.onViewResults}
            disabled={props.viewResultsDisabled}
            title={props.viewResultsTooltip || undefined}
            data-testid="project-table-view-results"
          >
            View Results
          </button>
        </Show>
        <div class="flex items-center gap-1.5">
        <div class="dropdown dropdown-top dropdown-start">
          <button
            type="button"
            class={`btn btn-outline btn-sm whitespace-nowrap ${props.canTransfer ? '' : 'btn-disabled'}`}
            disabled={!props.canTransfer}
            title={props.transferTooltip || undefined}
            data-testid="project-table-transfer"
          >
            Add to
            <svg class="w-3 h-3 ml-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          <ul tabindex="0" class="dropdown-content menu menu-sm bg-base-100 rounded-box shadow-lg z-30 w-full mb-1">
            <li>
              <button
                onClick={props.onTransferDock}
                disabled={!props.canTransferDock}
                title={props.transferDockTooltip || undefined}
                data-testid="project-table-transfer-dock"
              >
                Dock
              </button>
            </li>
            <li>
              <button
                onClick={props.onTransferMcmm}
                disabled={!props.canTransferMcmm}
                title={props.transferMcmmTooltip || undefined}
                data-testid="project-table-transfer-mcmm"
              >
                MCMM
              </button>
            </li>
            <li>
              <button
                onClick={props.onTransferSimulate}
                disabled={!props.canTransferSimulate}
                title={props.transferSimulateTooltip || undefined}
                data-testid="project-table-transfer-simulate"
              >
                Dynamics
              </button>
            </li>
          </ul>
        </div>
        <button
          ref={importButtonRef}
          class={`btn btn-outline btn-sm flex-1 min-w-0 px-2 ${showImportPopover() ? 'btn-active' : ''}`}
          onClick={(event) => {
            event.stopPropagation();
            setShowImportPopover((open) => !open);
          }}
          data-testid="project-table-import"
        >
          Import
        </button>
        <button
          class="btn btn-outline btn-sm flex-1 min-w-0 px-2"
          onClick={props.onExport}
          disabled={!props.canExport}
          data-testid="project-table-export"
        >
          Export
        </button>
        </div>
        <Show when={showImportPopover()}>
          <div
            ref={importPopoverRef}
            class="absolute bottom-full left-3 right-3 mb-2 z-40 rounded-lg border border-base-300 bg-base-100 shadow-xl"
          >
            <div class="flex items-center justify-between border-b border-base-300 px-3 py-2">
              <div>
                <div class="text-xs font-semibold">Import Into View</div>
                <div class="text-[10px] text-base-content/55">Browse files, fetch a PDB, or load SMILES</div>
              </div>
              <button
                class="btn btn-ghost btn-xs btn-square"
                onClick={() => setShowImportPopover(false)}
                title="Close import popover"
              >
                <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div class="p-3">
              <ImportInputPanel
                compact
                importButtonLabel="Browse structures"
                onImport={props.onBrowseImport}
                importDisabled={props.importDisabled}
                importLoading={props.importLoading}
                showPdbFetch={true}
                pdbIdValue={props.importPdbIdValue}
                onPdbIdInput={props.onImportPdbIdInput}
                onFetchPdb={props.onFetchImportPdb}
                fetchDisabled={props.importPdbFetchDisabled}
                fetchLoading={props.importPdbFetchLoading}
                showSmiles={true}
                smilesValue={props.importSmilesValue}
                onSmilesInput={props.onImportSmilesInput}
                onSubmitSmiles={props.onSubmitImportSmiles}
                smilesDisabled={props.importSmilesDisabled}
                smilesLoading={props.importSmilesLoading}
                smilesCount={props.importSmilesCount}
              />
            </div>
          </div>
        </Show>
      </div>
    </div>
  );
};

export default ProjectTable;
