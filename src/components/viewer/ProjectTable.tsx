import { Component, For, Show } from 'solid-js';
import type { ViewerProjectColumn, ViewerProjectFamily, ViewerProjectTableState } from '../../stores/workflow';
import { getSortedRowsForFamily, getVisibleColumnsForFamily } from '../../utils/projectTable';

interface ProjectTableProps {
  projectTable: ViewerProjectTableState;
  panelWidth: number;
  onSelectRow: (rowId: string) => void;
  onToggleFamilyCollapsed: (familyId: string) => void;
  onSortFamily: (familyId: string, columnKey: string) => void;
  onPlayTrajectory: (familyId: string) => void;
  canNavigatePrevious: boolean;
  canNavigateNext: boolean;
  onNavigatePrevious: () => void;
  onNavigateNext: () => void;
  canSimulate: boolean;
  canExport: boolean;
  onSimulate: () => void;
  onExport: () => void;
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
                              props.projectTable.activeRowId === row.id ? 'bg-primary/10' : ''
                            }`}
                            onClick={() => props.onSelectRow(row.id)}
                            data-testid={`project-row-${row.id}`}
                          >
                            <td>
                              <div
                                class={`w-2 h-2 rounded-full mx-auto ${
                                  props.projectTable.activeRowId === row.id ? 'bg-primary' : 'bg-base-300'
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

      <div class="px-3 py-2 border-t border-base-300 flex items-center gap-2">
        <button
          class="btn btn-outline btn-sm flex-1"
          onClick={props.onSimulate}
          disabled={!props.canSimulate}
          data-testid="project-table-simulate"
        >
          Simulate
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
