// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, For, Show } from 'solid-js';
import type { ViewerProjectColumn, ViewerProjectFamily, ViewerProjectTableState } from '../../stores/workflow';
import { getSortedRowsForFamily, getVisibleColumnsForFamily } from '../../utils/projectTable';

interface ProjectTableProps {
  projectTable: ViewerProjectTableState;
  panelWidth: number;
  onSelectRow: (rowId: string) => void;
  onToggleRowSelection: (rowId: string) => void;
  onToggleFamilyCollapsed: (familyId: string) => void;
  onSortFamily: (familyId: string, columnKey: string) => void;
  onPlayTrajectory: (familyId: string) => void;
  onRemoveFamily: (familyId: string) => void;
  canNavigatePrevious: boolean;
  canNavigateNext: boolean;
  onNavigatePrevious: () => void;
  onNavigateNext: () => void;
  canTransfer: boolean;
  canExport: boolean;
  onTransferDock: () => void;
  onTransferMcmm: () => void;
  onTransferSimulate: () => void;
  onExport: () => void;
  onImport: () => void;
  canAlignProtein: boolean;
  canAlignLigand: boolean;
  canAlignSubstructure: boolean;
  onAlignProtein: () => void;
  onAlignLigand: () => void;
  onAlignSubstructure: () => void;
  alignSubstructureLabel: string | null;
  hasAlignment: boolean;
  onResetAlignment: () => void;
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

  const anyAlignAvailable = () =>
    props.canAlignProtein || props.canAlignLigand || props.canAlignSubstructure;

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
                  class="btn btn-ghost btn-xs btn-square opacity-40 hover:opacity-100"
                  onClick={(e) => { e.stopPropagation(); props.onRemoveFamily(family.id); }}
                  title="Remove from table"
                  data-testid={`project-family-remove-${family.id}`}
                >
                  <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
                            class={`cursor-pointer hover:bg-base-200 ${
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
                            <td class="font-medium truncate" title={row.label}>{row.label}</td>
                            <For each={getVisibleColumnsForFamily(family, props.panelWidth)}>
                              {(column) => (
                                <td class="text-right font-mono truncate">
                                  {formatMetric(row.metrics[column.key], column)}
                                </td>
                              )}
                            </For>
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
      </div>

      {/* Alignment toolbar */}
      <Show when={anyAlignAvailable() || props.hasAlignment}>
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
      </Show>

      <div class="px-3 py-2 border-t border-base-300 flex items-center gap-1.5">
        <div class="dropdown dropdown-top flex-1">
          <label
            tabindex="0"
            class={`btn btn-outline btn-sm w-full ${props.canTransfer ? '' : 'btn-disabled'}`}
            data-testid="project-table-transfer"
          >
            Transfer
            <svg class="w-3 h-3 ml-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
            </svg>
          </label>
          <ul tabindex="0" class="dropdown-content menu menu-sm bg-base-100 rounded-box shadow-lg z-30 w-full mb-1">
            <li>
              <button onClick={props.onTransferDock} data-testid="project-table-transfer-dock">
                Dock
              </button>
            </li>
            <li>
              <button onClick={props.onTransferMcmm} data-testid="project-table-transfer-mcmm">
                MCMM
              </button>
            </li>
            <li>
              <button onClick={props.onTransferSimulate} data-testid="project-table-transfer-simulate">
                Simulate
              </button>
            </li>
          </ul>
        </div>
        <button
          class="btn btn-outline btn-sm flex-1"
          onClick={props.onImport}
          data-testid="project-table-import"
        >
          Import
        </button>
        <button
          class="btn btn-outline btn-sm flex-1"
          onClick={props.onExport}
          disabled={!props.canExport}
          data-testid="project-table-export"
        >
          Export
        </button>
      </div>
    </div>
  );
};

export default ProjectTable;
