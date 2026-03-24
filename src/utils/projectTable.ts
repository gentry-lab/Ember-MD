import type { ViewerProjectColumn, ViewerProjectFamily, ViewerProjectRow, ViewerProjectTableState } from '../stores/workflow';

const isPinnedRow = (row: ViewerProjectRow): boolean =>
  row.rowKind === 'apo'
  || row.rowKind === 'holo'
  || row.rowKind === 'ligand'
  || row.rowKind === 'prepared-ligand'
  || row.rowKind === 'input'
  || row.rowKind === 'initial-complex';

export const getVisibleColumnsForFamily = (
  family: ViewerProjectFamily,
  panelWidth: number,
): ViewerProjectColumn[] =>
  [...family.columns]
    .sort((a, b) => a.priority - b.priority)
    .filter((column) => column.minPanelWidth == null || panelWidth >= column.minPanelWidth);

export const getSortedRowsForFamily = (
  family: ViewerProjectFamily,
  rows: ViewerProjectRow[],
  visibleColumns: ViewerProjectColumn[],
): ViewerProjectRow[] => {
  const familyRows = family.rowIds
    .map((rowId) => rows.find((row) => row.id === rowId))
    .filter((row): row is ViewerProjectRow => Boolean(row));

  const pinnedRows = familyRows.filter(isPinnedRow);
  const sortableRows = familyRows.filter((row) => !isPinnedRow(row));

  if (!family.sortKey || !visibleColumns.some((column) => column.key === family.sortKey)) {
    return [...pinnedRows, ...sortableRows];
  }

  const sortedRows = [...sortableRows].sort((a, b) => {
    const av = a.metrics[family.sortKey!];
    const bv = b.metrics[family.sortKey!];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    if (typeof av === 'string' || typeof bv === 'string') {
      const cmp = String(av).localeCompare(String(bv));
      return family.sortDirection === 'asc' ? cmp : -cmp;
    }
    return family.sortDirection === 'asc' ? av - bv : bv - av;
  });

  return [...pinnedRows, ...sortedRows];
};

export const getVisibleProjectRows = (
  projectTable: ViewerProjectTableState | null,
  panelWidth: number,
): ViewerProjectRow[] => {
  if (!projectTable) return [];

  return projectTable.families.flatMap((family) => {
    if (family.collapsed) return [];
    const visibleColumns = getVisibleColumnsForFamily(family, panelWidth);
    return getSortedRowsForFamily(family, projectTable.rows, visibleColumns);
  });
};
