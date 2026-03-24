# Ember

Desktop app for GPU-accelerated molecular dynamics on Apple Silicon. User-facing tabs: **View** (NGL 3D viewer), **MCMM** (standalone conformer generation), **Dock** (AutoDock Vina), and **Simulate** (OpenMM AMBER MD). FEP scoring and GIST water map code exist in the codebase but are removed from the UI (recoverable from git v0.2.18).

**Repos**: `pseudo-control/Ember-MD` (this repo), `pseudo-control/Ember-Metal` (native Metal GPU backend, separate repo)
**Stale docs**: `README.md` and `deps/README.md` still reference FragGen/GNINA/Linux — use this file instead.
**Script duplication**: `run_md_simulation.py` exists at both repo root (reference copy) and `deps/staging/scripts/` (canonical bundled copy with plugin loading).

## Tech Stack
- **Frontend**: SolidJS + TypeScript + Tailwind/DaisyUI (wireframe/business themes)
- **Desktop**: Electron 27, Webpack
- **MD Engine**: OpenMM 8.1.2 (AMBER ff19SB/ff14SB + OPC/TIP3P + OpenFF Sage 2.3.0)
- **GPU**: OpenCL (cl2Metal, preferred in the app runtime for larger systems) → Metal (native MSL fallback, often faster on smaller systems) → CPU
- **Visualization**: NGL (WebGL), always mounted and CSS-hidden outside View mode to avoid stage re-creation/OOM issues
- **Cheminformatics**: RDKit, Meeko (PDBQT prep), Molscrub (protonation), PDBFixer, MDAnalysis, AmberTools
- **Scoring**: CORDIAL neural-network rescoring (optional, `~/Desktop/CORDIAL`), GFN2-xTB strain energy, Vina rescore
- **Semiempirical QM**: GFN2-xTB 6.7.1 (vendored at `vendor/xtb-env/`, arm64)
- **pKa Prediction**: QupKake (vendored fork at `vendor/QupKake/`, uses xTB features)

## Build
```bash
npm start               # Build + run dev mode
npm run build:electron  # TS → electron-dist/
npm run build           # Webpack → dist-webpack/
npm run dist:mac        # Bundle .dmg via scripts/bundle-mac.sh (conda-pack → electron-builder dir build → create-dmg/plain DMG)
                        # Auto-generates assets/dmg-background.png if missing
                        # Requires: brew install create-dmg, Finder automation permission
npm run test:e2e        # Build + run Playwright E2E tests (serial, 1 worker)
npm run test:e2e:headed # Same but with visible Electron window
```

## Project Structure
```
/electron/main.ts          — Thin orchestrator: window creation, app lifecycle, IPC module registration (~120 lines)
/electron/paths.ts         — Path resolution, env detection, initializePaths()
/electron/spawn.ts         — spawnPythonScript, childProcesses tracking, filterMdStderr, loadAndMergeCordialScores
/electron/app-state.ts     — Shared mutable state (mainWindow, fraggenRoot, condaPythonPath, etc.)
/electron/ipc/             — IPC handler modules (one per domain):
                             dialogs, stats, generation, docking, ligand-sources, conformers,
                             simulation, projects, viewer, maps, fep
/electron/preload.ts       — Context bridge (inlined channel names)
/src/App.tsx               — Root: ViewerMode always mounted (CSS-hidden); routes View, MCMM, Dock, MD
/src/stores/workflow.ts    — SolidJS signals (WorkflowState, MDState, DockState, ViewerState, MapState, ConformState)
/src/utils/projectPaths.ts — Project directory layout (DockingPaths, SimulationPaths, conformers path)
/src/utils/jobName.ts      — Job name generation + folder naming
/src/components/layout/WizardLayout.tsx — Header: View | MCMM | Dock | Simulate, project+job selector, step indicators
/src/components/steps/     — DockStep{Load,Configure,Progress,Results}, MDStep{Load,Configure,Progress,Results}, ConformStep{Load,Configure,Progress,Results}
/src/components/viewer/    — ViewerMode, TrajectoryControls, ClusteringModal, AnalysisPanel, FepScoringPanel, ScoringPanel, LayerPanel, BindingSiteMapPanel
/src/components/map/       — MapMode (GIST water map — solvation method only)
/shared/types/             — md.ts, dock.ts, ipc.ts, electron-api.ts, errors.ts, result.ts
/scripts/score_cordial.py  — CORDIAL rescoring (outside deps/staging/)
/deps/staging/scripts/     — Canonical Python scripts called by Electron
                             prepare_receptor.py, run_md_simulation.py, run_vina_docking.py,
                             run_abfe.py, detect_pdb_ligands.py, extract_xray_ligand.py,
                             enumerate_protonation.py, enumerate_stereoisomers.py,
                             generate_conformers.py, cluster_trajectory.py,
                             score_cluster_centroids.py, score_xtb_strain.py, predict_ligand_pka.py,
                             receptor_protonation.py, analyze_gist.py, analyze_*.py, utils.py, etc.
/vendor/xtb-env/           — GFN2-xTB 6.7.1 (conda env with ALPB solvation, arm64)
/vendor/QupKake/           — QupKake pKa prediction (forked, repo-local)
```

## Project Directory Layout
Each project lives under `~/Ember/{projectName}/`. Defined in `src/utils/projectPaths.ts`.
```text
{project}/
  .ember-project
  structures/                        — Imported PDB/CIF files
  surfaces/pocket_map_solvation/     — GIST water map OpenDX grids

  docking/Vina_{ligandId}/           — Self-contained docking job
    inputs/receptor.pdb, reference_ligand.sdf, ligands/*.sdf
    prep/                            — Extraction, protonation, stereoisomers, conformers
    results/all_docked.sdf, cordial_scores.json, xtb_strain.json, poses/*_docked.sdf.gz

  simulations/{run}/
    inputs/                          — receptor.pdb, ligand.sdf
    results/                         — system.pdb, trajectory.dcd, final.pdb, energy.csv, seed.txt
    results/analysis/                — clustering/, scored_clusters/, rmsd/, rmsf/, hbonds/, contacts/

  conformers/{run}/                  — Standalone conformer generation output
  fep/                               — FEP scoring output
```
**Legacy fallbacks**: scanners still check old layouts such as `*_receptor_prepared.pdb`, top-level `poses/`, prefixed `*_system.pdb`, and `raw/`.

**Path API**: `projectPaths(baseDir, name).docking(run)` returns `{ root, inputs, inputsLigands, prep, results, resultsPoses }`. `.simulations(run)` returns `{ root, inputs, results, analysis, analysisClustering }`. `.conformers(run)` returns the standalone conformer output directory.

## Header UI
Three-zone layout: mode tabs on the left, project name + job selector in the center, step indicators on the right. Tabs: `View`, `MCMM`, `Dock`, `Simulate`.

## Simulate Mode
**Equilibration**: AMBER-style restrained minimization → graduated minimization → NVT heating 5→100 K → NPT heating 100→300 K → restrained NPT → gradual restraint release → unrestrained NPT → production with HMR. Shared type metadata currently treats equilibration as roughly 270 ps.

**Random seed**: `MDConfig.seed` (`0 = auto`). Applied to velocity initialization and Langevin noise, then written to `seed.txt` for reproducibility.

**Presets**: `ff19sb-opc` (default), `ff14sb-tip3p`, `ff19sb-opc3`, `charmm36-mtip3p`. Ligands use OpenFF Sage 2.3.0.

**Key details**: protein FF path is `amber/protein.ff19SB.xml`. OpenCL production uses single precision. Restraints use `periodicdistance()^2` for PBC compatibility.

**Post-simulation pipeline**: After production completes, the app automatically clusters the trajectory into 10 centroids (K-means, ligand RMSD) and scores each with xTB strain energy, Vina rescore, and CORDIAL (if installed). Results are displayed as a sorted table in MDStepResults. The PDF analysis report (RMSD, RMSF, H-bonds, contacts, SSE, torsions) is also generated automatically. Output goes to `analysis/scored_clusters/` (scoring) and `analysis/` (report).

## Dock Mode
Pipeline: receptor prep (Meeko Polymer) → ligand PDBQT prep (Meeko) → autobox from reference ligand → Vina docking → optional xTB strain filter → optional post-dock pocket refinement (OpenMM, Sage 2.3.0 + OBC2, receptor-restrained) → optional CORDIAL rescoring → multi-pose SDF.gz output. Also supports protonation via Molscrub, stereoisomer enumeration via RDKit, and conformer generation (ETKDG/MCMM/CREST) before docking.

**Receptor prep**: removes the docking ligand and common crystallization artifacts, retains crystallographic waters within a configurable distance of the ligand (default 3.5 A, toggle in DockStepConfigure), metal ions, and relevant cofactors near the binding site. Metals are re-injected into PDBQT after Meeko processing with explicit AD4 atom types and charges.

**xTB strain filter**: Optional post-docking step (`xtbConfig.strainFilter`). Computes GFN2-xTB strain energy per pose via batch mode (`score_xtb_strain.py --mode batch_strain`). Uses per-molecule SMILES-keyed reference energies (each unique molecule gets its own optimized free minimum). Results in `results/xtb_strain.json`. Displayed as "Strain" column in DockStepResults (yellow >5 kcal/mol, red >8).

**xTB pre-optimization**: Optional pre-docking step (`xtbConfig.preOptimize`). Optimizes ligand geometry with GFN2-xTB before Meeko PDBQT conversion.

## Receptor Preparation (Unified)
Single core function: `receptor_protonation.py::prepare_receptor_with_propka()`. Accepts optional `fixer=` kwarg for pre-modified PDBFixer.

**Pipeline:** Reduce → PROPKA → PDBFixer (`findMissingResidues` → `findMissingAtoms` → `addMissingAtoms`, models missing loops) → variant plan (disulfides, PROPKA, HIS tautomers) → **HIS tautomer energy scoring** → `addHydrogens` → **hydrogen minimization** (ff19SB + OBC2, heavy atoms restrained at 50 kcal/mol/A^2, L-BFGS 500 iter, ~4-6s) → write PDB + metadata.

**HIS tautomer energy scoring**: For each neutral histidine (HID or HIE from geometric heuristic), evaluates both tautomers by building the full ff19SB+OBC2 system and running a short 50-iteration minimization. Picks the lower-energy tautomer if the delta exceeds 4 kJ/mol; otherwise keeps the geometric pick. Per-residue independent scoring (~1.6s per HIS). Gated on `minimize_hydrogens=True`. Results in `.prep.json` under `his_tautomer_scoring`.

**Hydrogen minimization**: On by default (`minimize_hydrogens=True`). Resolves template-placed H clashes and optimizes H-bond networks/hydroxyl orientations using gradient minimization with implicit solvent. Skips OBC2 for systems >15k atoms. Falls back to un-minimized positions on any failure.

**How it's called:**
- **Docking**: `detect_pdb_ligands.py` calls `prepare_receptor_with_propka()` directly (self-contained)
- **MD**: `prepare_receptor.py::prepare_receptor()` adds CIF conversion, then delegates to `prepare_receptor_with_propka(fixer=...)`
- **MD caller**: `run_md_simulation.py::_prepare_receptor_topology()` checks for existing `receptor_prepared.pdb`; if absent, calls `prepare_receptor()`

**Pocket refinement** (`prepare_docking_complex.py`): `createSystem` uses try/fallback — normal first, then `ignoreExternalBonds=True` + CYS/CYX resolution by atom content (HG present → CYS, absent → CYX).

## MCMM Mode
Standalone conformer generation exposed as **MCMM** tab. Internally the workflow state is `conform`. Three methods:

- **ETKDG**: Fast RDKit distance geometry (seconds)
- **MCMM**: Monte Carlo Multiple Minimum with Sage 2.3.0 + OBC2 implicit solvent (default: 50 max conformers, 1.0 A RMSD cutoff, 5.0 kcal/mol energy window, 1000 steps, 298 K, amide cis/trans sampling)
- **CREST**: GFN2-xTB metadynamics via `crest` 2.12 binary (conda-forge). Uses `--gfn2 --alpb water --chrg <formal_charge>`. Most thorough — biased sampling escapes local minima. ~6 min for a 21-atom molecule. CREST 3.x does NOT work (tblite crash on arm64); use 2.12.

Optional **xTB reranking** re-ranks ETKDG/MCMM conformers by GFN2-xTB single-point energy. Skipped for CREST (already xTB-ranked). Validated: methylcyclohexane equatorial/axial split = 1.68 kcal/mol (xTB rerank) vs 1.53 (CREST) vs ~1.8 (textbook).

**Energies**: All methods output relative kcal/mol (normalized to global minimum). Displayed in results table with method label. Energies flow: Python SDF property → JSON `conformer_energies` → store → UI.

## Removed Features (recoverable from git v0.2.18)
- **Map Mode** (GIST water map): `MapMode.tsx`, `analyze_gist.py`, `BindingSiteMapPanel.tsx`. GIST water thermodynamics via cpptraj.
- **FEP Panel**: `FepScoringPanel.tsx`, `run_abfe.py`. ABFE free energy scoring.
- **Viewer SCORE button**: `ScoringPanel.tsx`. Single-complex Vina + xTB scoring in the viewer.

## View Mode
NGL viewer with queue navigation for docking poses or cluster centroids. Trajectory playback updates coordinates in place after the first frame instead of reparsing PDBs. Supports clustering, RMSD/RMSF/H-bond/contact analysis, surface coloring, and multi-structure layer alignment. Import/Recent Jobs segmented control when no structure loaded.

## xTB (GFN2-xTB)
GFN2-xTB 6.7.1 is vendored at `vendor/xtb-env/bin/xtb` (conda env, arm64). Detected at runtime by `getQupkakeXtbPath()` in main.ts. Used for:
- **Conformer reranking**: re-rank ETKDG/MCMM conformers by xTB single-point energy
- **CREST**: xTB serves as the energy function for CREST metadynamics conformer search
- **MD cluster scoring**: xTB strain energy per cluster centroid
- **QupKake pKa**: GFN2-xTB features for neural-network pKa prediction

xTB docking strain filter and viewer SCORE button removed from UI (strain methodology unsound — uses local minimum, not CREST global minimum). `score_xtb_strain.py` still exists for future CREST-based strain.

**Note**: CREST must be 2.12 (3.x crashes on arm64).

## Logging
`~/Ember/logs/ember-<timestamp>.log` captures main-process and renderer console output. Tags commonly include `[Viewer]`, `[Dock]`, `[MD]`, `[FEP]`, `[Score]`, `[Nav]`, and `[Store]`.

## Key Patterns
```typescript
// SolidJS store update
setState(s => ({ ...s, md: { ...s.md, config: { ...s.md.config, ...config } } }));

// ViewerMode stays mounted and is only CSS-hidden outside View mode
<div style={{ display: state().mode === 'viewer' ? 'block' : 'none' }}>
  <ViewerMode />
</div>

// Load a job in the viewer
resetViewer();
setViewerPdbPath(pdb);
setViewerLigandPath(sdf);
setMode('viewer');

// NGL ligand selection: use resname, not chain
return `[${ligand.resname}] and ${ligand.resnum}`;

// Shared helpers in main.ts
spawnPythonScript(args, options)           // spawn + accumulate stdout/stderr + track in childProcesses
loadAndMergeCordialScores(dir, items, key) // read cordial_scores.json + merge into results array
```

## Path Resolution (main.ts)
Bundled app paths prefer `process.resourcesPath/scripts` and `process.resourcesPath/python/bin/python`. Dev mode falls back to `deps/staging/scripts/` and typical `openmm-metal`/`fraggen` conda env paths. Legacy env vars such as `FRAGGEN_ROOT` and `FRAGGEN_PYTHON` are still honored. xTB binary is resolved by `getXtbPath()` which searches `vendor/xtb-env/bin/xtb` first, then the bundled `xtb/` directory.

## GPU Platform Cascade
For the canonical bundled MD runner in `deps/staging/scripts/run_md_simulation.py`, the runtime platform order is `CUDA → OpenCL (cl2Metal) → Metal → CPU`. OpenCL is currently tried before Metal in both benchmarking and full simulation setup.

## Metal Backend
Native Metal backend work lives in the separate `pseudo-control/Ember-Metal` repo. Use that repo's instruction file for low-level backend architecture, profiling, and MSL porting notes rather than duplicating them here.

## E2E Testing (Playwright)
Playwright with native Electron support. Tests launch the real app, interact with the UI, and verify results.

**Config**: `playwright.config.ts` — 60s per test, 5 min global, serial (1 worker), html+list reporters.
**Fixtures**: `tests/e2e/fixtures.ts` — shared `app` (ElectronApplication) and `window` (Page) fixtures. App launched with `NODE_ENV=test`.
**Test data**: `ember-test-protein/` — 8TCE.cif (protein) + kiv/kiv.sdf (ligand) for full pipeline tests.
**Molecule fixtures**: `tests/fixtures/` — minimal PDB/SDF for unit-style E2E tests.

```bash
npx playwright test                          # Run all
npx playwright test tests/e2e/app-boot.spec.ts  # Run one spec
npx playwright test --headed                 # Watch the app
```

**Key patterns:**
- Tabs are disabled until a project is created (`canSwitchMode()` checks `projectReady`)
- File dialogs must be mocked via `window.evaluate` since Playwright can't interact with native OS dialogs
- IPC calls available via `window.electronAPI.*` in evaluate blocks
- DaisyUI selectors: `.tab.tab-sm` (mode tabs), `.btn.btn-primary` (CTAs), `.select.select-bordered` (dropdowns)
- The project selector overlay (`div.absolute.inset-0.z-30`) blocks interaction until a project is selected/created

**Test structure:**
```
tests/
  e2e/
    fixtures.ts          — Electron launch helper
    app-boot.spec.ts     — Launch, tabs, no console errors
    navigation.spec.ts   — Tab switching, project gating
    viewer.spec.ts       — Import/Recent, viewer default state
    mcmm.spec.ts         — MCMM tab presence, disabled state
    docking.spec.ts      — Dock tab presence, disabled state
    simulate.spec.ts     — Simulate tab presence, disabled state
  fixtures/
    alanine_dipeptide.pdb
    benzene.sdf
```

## Known Limitations
- macOS only in practice for the current app workflow
- App is unsigned, so launch from `/Applications` or Spotlight rather than Launchpad
- `run_md_simulation.py` exists both at repo root and in `deps/staging/scripts/`; the staging copy is canonical for the app
- CREST requires separate `crest` binary installation (`conda install -c conda-forge crest`)
- `prepare_docking_complex.py` uses a try/fallback for `createSystem` (ignoreExternalBonds + CYS/CYX atom-based resolution) for structures with unusual bond patterns

## License
MIT. GPL-licensed scientific tools are invoked as separate processes. Meeko and Molscrub are Apache-2.0.
