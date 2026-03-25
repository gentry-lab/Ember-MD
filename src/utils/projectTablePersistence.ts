// Copyright (c) 2026 Ember Contributors. MIT License.
import type { ViewerProjectTableState } from '../stores/workflow';

const TABLE_FILENAME = 'project-table.json';

export function serializeProjectTable(table: ViewerProjectTableState): string {
  return JSON.stringify(table);
}

/**
 * Parse and validate a persisted project table. Checks that referenced files
 * still exist on disk — rows with dead paths are silently removed.
 * Returns null if nothing survives validation.
 */
export async function deserializeAndValidateProjectTable(
  raw: unknown,
  fileExists: (path: string) => Promise<boolean>,
): Promise<ViewerProjectTableState | null> {
  if (!raw || typeof raw !== 'object') return null;
  const data = raw as Record<string, unknown>;
  if (!Array.isArray(data.families) || !Array.isArray(data.rows)) return null;

  // Collect all unique paths to check in parallel
  const rows = data.rows as any[];
  const pathSet = new Map<string, boolean>();
  for (const row of rows) {
    if (!row?.id || !row?.item?.pdbPath) continue;
    pathSet.set(row.item.pdbPath, false);
    if (row.item.ligandPath) pathSet.set(row.item.ligandPath, false);
  }

  // Parallel file existence checks
  const paths = [...pathSet.keys()];
  const results = await Promise.all(paths.map((p) => fileExists(p)));
  for (let i = 0; i < paths.length; i++) {
    pathSet.set(paths[i], results[i]);
  }

  // Filter rows using precomputed existence map
  const validRows: ViewerProjectTableState['rows'] = [];
  for (const row of rows) {
    if (!row?.id || !row?.item?.pdbPath) continue;
    if (!pathSet.get(row.item.pdbPath)) continue;
    const ligandOk = !row.item.ligandPath || pathSet.get(row.item.ligandPath);
    validRows.push(ligandOk ? row : { ...row, item: { ...row.item, ligandPath: undefined } });
  }

  if (validRows.length === 0) return null;

  const validRowIds = new Set(validRows.map((r) => r.id));

  // Prune families — remove dead rowIds, drop empty families
  const validFamilies: ViewerProjectTableState['families'] = [];
  for (const fam of data.families as any[]) {
    if (!fam?.id || !Array.isArray(fam.rowIds)) continue;
    const liveRowIds = fam.rowIds.filter((id: string) => validRowIds.has(id));
    if (liveRowIds.length === 0) continue;
    validFamilies.push({ ...fam, rowIds: liveRowIds });
  }

  if (validFamilies.length === 0) return null;

  const activeRowId = validRowIds.has(data.activeRowId as string)
    ? (data.activeRowId as string)
    : null;

  const selectedRowIds = Array.isArray(data.selectedRowIds)
    ? (data.selectedRowIds as string[]).filter((id) => validRowIds.has(id))
    : [];

  return { families: validFamilies, rows: validRows, activeRowId, selectedRowIds };
}

/**
 * Create a debounced saver that writes the project table to disk.
 * Uses existing writeTextFile IPC — no new channels needed.
 */
export function createProjectTableSaver(
  getProjectDir: () => string | null,
  writeTextFile: (path: string, content: string) => Promise<unknown>,
  debounceMs = 500,
): (table: ViewerProjectTableState | null) => void {
  let timer: ReturnType<typeof setTimeout> | null = null;

  return (table: ViewerProjectTableState | null) => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(async () => {
      const dir = getProjectDir();
      if (!dir) return;
      const filePath = `${dir}/${TABLE_FILENAME}`;
      if (!table || table.families.length === 0) {
        await writeTextFile(filePath, '{}').catch(() => {});
        return;
      }
      await writeTextFile(filePath, serializeProjectTable(table)).catch(() => {});
    }, debounceMs);
  };
}
