# E2E Test Coverage Loop

You are running a Playwright E2E test coverage loop for Ember. Read your IDENTITY.md memory for authority scope and guardrails. Use `e2e-coverage.md` as the checkbox source of truth; read `BOUNTY_BOARD.md` and `VISION.md` for product context and stale-item reconciliation.

Your goal: make every test pass, fix failures, and add tests from `.claude/commands/e2e-coverage.md`.

## Loop Steps

### First invocation (cold start)
1. **Check git status** — `git log -1 --oneline` to see if the last commit was an E2E commit from this loop.
2. **Build**: `npm run build:electron && npm run build`. Fix build errors first.
3. **Run all tests**: `npx playwright test tests/e2e/ --reporter=list`
4. **If tests fail**: read failure output, fix test or app, re-run failing spec, then full suite.
5. **If all pass**: go to **Add new test** below.

### Subsequent invocations (warm — last commit was this loop)
1. **Skip full suite** — the last commit already confirmed all tests pass.
2. **Check for uncommitted changes** — if there are dirty files, build and run affected spec(s) only.
3. **Otherwise**: go straight to **Add new test**.

### Add new test
1. Read `e2e-coverage.md`, find the next unchecked `[ ]` item, and classify it before writing a test:
   - direct current-UI behavior to test now
   - dependency-gated / optional integration (`CREST`, `CORDIAL`, `xTB`)
   - stale or UI-inaccessible item that needs doc cleanup before testing
2. Write the test, run just that spec to get it passing.
3. **Targeted regression** — run only the related spec group (see mapping below), NOT the full suite.
4. **Commit**: `npm version patch --no-git-tag-version` then commit with `test(e2e): {what you added/fixed}`.
5. Mark the checklist item `[x]` in `e2e-coverage.md`.

### Spec group mapping (targeted regression)
When you add/fix a test, run the **new spec + its related specs** instead of the full suite:

| Area touched | Run these specs |
|---|---|
| `app-boot`, `navigation`, `error-paths` | `app-boot navigation error-paths` |
| `mcmm`, `mcmm-pipeline` | `mcmm mcmm-pipeline` |
| `docking`, `docking-pipeline` | `docking docking-pipeline` |
| `simulate`, `md-pipeline` | `simulate md-pipeline` |
| `viewer`, `viewer-ngl`, `viewer-controls`, `project-table`, `viewer-trajectory` | `viewer viewer-ngl viewer-controls project-table viewer-trajectory` |
| `electron/ipc/*`, fixtures, shared infra | Full suite (these affect everything) |

Example: after adding a viewer/project-table test, run:
```bash
npx playwright test tests/e2e/viewer.spec.ts tests/e2e/viewer-ngl.spec.ts tests/e2e/viewer-controls.spec.ts tests/e2e/project-table.spec.ts tests/e2e/viewer-trajectory.spec.ts --reporter=list
```

Only run the **full suite** on cold start or if you touched shared infrastructure (fixtures, electron IPC, App.tsx, workflow store).

### Stop conditions
- **Stop and report** if: >3 consecutive failures on same issue, or unclear test-bug vs app-bug.
- **Stop and report** if: no unchecked items remain in `e2e-coverage.md`.

## CRITICAL: No File Dialogs

**DO NOT mock file dialog functions** like `selectPdbFile`, `selectMoleculeFilesMulti`, etc. These open real OS dialogs that block headless testing.

Instead, call IPC methods directly via `window.evaluate`. All IPC methods accept absolute file paths:

```typescript
// Set up project (enables tabs)
await window.evaluate(async () => {
  await (window as any).electronAPI.ensureProject('__e2e_test__');
});
await window.waitForTimeout(1000);

// Import structure into project (copies file, no dialog)
await window.evaluate(async (args) => {
  return await (window as any).electronAPI.importStructure(args.src, args.projDir);
}, { src: RECEPTOR_CIF, projDir });

// Detect ligands in a PDB/CIF (no dialog)
const ligands = await window.evaluate(async (p) => {
  return await (window as any).electronAPI.detectPdbLigands(p);
}, RECEPTOR_CIF);

// Import molecule files (SDF/MOL, no dialog)
await window.evaluate(async (args) => {
  return await (window as any).electronAPI.importMoleculeFiles(args.paths, args.outDir);
}, { paths: [KIV_SDF], outDir });

// Convert SMILES to 3D (no dialog)
await window.evaluate(async (args) => {
  return await (window as any).electronAPI.convertSingleMolecule(args.smiles, args.outDir, 'smiles');
}, { smiles: 'c1ccccc1', outDir });

// Run conformer generation directly
await window.evaluate(async (args) => {
  return await (window as any).electronAPI.runConformGeneration(
    args.sdf, args.outDir, 50, 1.0, 5.0, 'etkdg'
  );
}, { sdf: KIV_SDF, outDir });
```

For tests that verify **UI interaction** (clicking buttons, filling forms), mock the dialog on the renderer side so the button click handler gets a path without opening the OS dialog:
```typescript
// This is the ONLY valid dialog mock — override BEFORE clicking
await window.evaluate((p) => {
  (window as any).electronAPI.selectPdbFile = async () => p;
}, RECEPTOR_CIF);
// Now click the UI button — it calls the mocked function
await window.locator('.btn', { hasText: /Select Structure/i }).click();
```

If a mock still opens an OS dialog, switch to the direct IPC pattern above.

## Key IPC Channels

| Channel | Use |
|---------|-----|
| `ensureProject(name)` | Create/ensure project directory |
| `importStructure(src, projDir)` | Copy structure file into project |
| `detectPdbLigands(pdbPath)` | Detect ligands in PDB/CIF |
| `importMoleculeFiles(paths[], outDir)` | Convert SDF/MOL files |
| `convertSingleMolecule(input, outDir, type)` | SMILES or MOL to 3D SDF |
| `convertSmilesList(smiles[], outDir)` | Batch SMILES conversion |
| `extractXrayLigand(pdb, ligId, outDir)` | Extract ligand from crystal |
| `runConformGeneration(sdf, outDir, ...)` | Standalone conformer gen |
| `runVinaDocking(rec, ref, ligs[], outDir, cfg)` | Run docking |
| `runMdSimulation(rec, lig, outDir, cfg, ...)` | Run MD |

## Test Data

- **Receptor**: `ember-test-protein/8tce.cif`
- **Ligand**: `ember-test-protein/kiv/kiv.sdf`
- **Minimal PDB**: `tests/fixtures/alanine_dipeptide.pdb`
- **Minimal SDF**: `tests/fixtures/benzene.sdf`

## Selectors

- Mode tabs: `.tab.tab-sm` with text `View`, `MCMM`, `Dock`, `Simulate`
- Primary CTA: `.btn.btn-primary`
- Outline buttons: `.btn.btn-outline`
- Dropdowns: `select.select-bordered`
- Inputs: `input.input-bordered`
- Results table: `table` / `.table.table-xs`
- SMILES textarea: `textarea`
- Error alerts: `.alert.alert-error`

## Rules

- Tests must complete in <3 minutes each (exhaust=1, poses=1, 0.01 ns for computational tests)
- Clean up test projects: `rm -rf ~/Ember/__e2e_*` in afterAll
- Fix root causes, don't skip failures
- If `BOUNTY_BOARD.md` conflicts with `e2e-coverage.md` or current app behavior, update docs before adding brittle tests
- Treat `CREST`, `CORDIAL`, and optional `xTB` paths as availability-gated; do not fail the base loop on missing external tools
- Don't modify `electron/ipc/` unless it's a real app bug
- Don't modify `deps/staging/scripts/` — flag issues, don't fix them
- Use sub-agents for selector discovery and IPC signature lookups when needed
- Never push to remote
