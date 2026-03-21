# Ember

Desktop app for GPU-accelerated molecular dynamics on Apple Silicon. Primary user-facing tabs: **View** (NGL 3D viewer), **MCMM** (standalone conformer generation), **Dock** (AutoDock Vina), **Map** (GIST water map), and **Simulate** (OpenMM AMBER MD). The codebase also contains an **FEP** scoring panel, but its header tab is currently disabled.

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
```

## Project Structure
```
/electron/main.ts          — IPC handlers, subprocess management, logging, path resolution
/electron/preload.ts       — Context bridge (inlined channel names)
/src/App.tsx               — Root: ViewerMode always mounted (CSS-hidden); routes View, MCMM, Dock, Map, MD, and FEP panel
/src/stores/workflow.ts    — SolidJS signals (WorkflowState, MDState, DockState, ViewerState, MapState, ConformState)
/src/utils/projectPaths.ts — Project directory layout (DockingPaths, SimulationPaths, conformers path)
/src/utils/jobName.ts      — Job name generation + folder naming
/src/components/layout/WizardLayout.tsx — Header: View | MCMM | Dock | Map | Simulate | FEP(disabled), project+job selector, step indicators
/src/components/steps/     — DockStep{Load,Configure,Progress,Results}, MDStep{Load,Configure,Progress,Results}, ConformStep{Load,Configure,Progress,Results}
/src/components/viewer/    — ViewerMode, TrajectoryControls, ClusteringModal, AnalysisPanel, FepScoringPanel, ScoringPanel, LayerPanel, BindingSiteMapPanel
/src/components/map/       — MapMode (GIST water map — solvation method only)
/shared/types/             — md.ts, dock.ts, ipc.ts, electron-api.ts, errors.ts, result.ts
/scripts/score_cordial.py  — CORDIAL rescoring (outside deps/staging/)
/deps/staging/scripts/     — Canonical Python scripts called by Electron
                             run_md_simulation.py, run_vina_docking.py, run_abfe.py,
                             detect_pdb_ligands.py, extract_xray_ligand.py, enumerate_protonation.py,
                             enumerate_stereoisomers.py, generate_conformers.py, cluster_trajectory.py,
                             score_cluster_centroids.py, score_xtb_strain.py, predict_ligand_pka.py,
                             analyze_gist.py, analyze_*.py, utils.py, etc.
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
Three-zone layout: mode tabs on the left, project name + job selector in the center, step indicators on the right. The visible tabs are `View`, `MCMM`, `Dock`, `Map`, `Simulate`, and a disabled `FEP` tab labeled "coming soon". The job selector currently groups jobs under docking and simulation.

## Simulate Mode
**Equilibration**: AMBER-style restrained minimization → graduated minimization → NVT heating 5→100 K → NPT heating 100→300 K → restrained NPT → gradual restraint release → unrestrained NPT → production with HMR. Shared type metadata currently treats equilibration as roughly 270 ps.

**Random seed**: `MDConfig.seed` (`0 = auto`). Applied to velocity initialization and Langevin noise, then written to `seed.txt` for reproducibility.

**Presets**: `ff19sb-opc` (default), `ff14sb-tip3p`, `ff19sb-opc3`, `charmm36-mtip3p`. Ligands use OpenFF Sage 2.3.0.

**Key details**: protein FF path is `amber/protein.ff19SB.xml`. OpenCL production uses single precision. Restraints use `periodicdistance()^2` for PBC compatibility.

**Post-simulation pipeline**: After production completes, the app automatically clusters the trajectory into 10 centroids (K-means, ligand RMSD) and scores each with xTB strain energy, Vina rescore, and CORDIAL (if installed). Results are displayed as a sorted table in MDStepResults. The PDF analysis report (RMSD, RMSF, H-bonds, contacts, SSE, torsions) is also generated automatically. Output goes to `analysis/scored_clusters/` (scoring) and `analysis/` (report).

## Dock Mode
Pipeline: receptor prep (Meeko Polymer) → ligand PDBQT prep (Meeko) → autobox from reference ligand → Vina docking → optional xTB strain filter → optional post-dock pocket refinement (OpenMM, Sage 2.3.0 + OBC2, receptor-restrained) → optional CORDIAL rescoring → multi-pose SDF.gz output. Also supports MCS core-constrained RMSD, protonation via Molscrub, stereoisomer enumeration via RDKit, and conformer generation (ETKDG/MCMM/CREST) before docking.

**Receptor prep**: removes the docking ligand and common crystallization artifacts, keeps nearby crystallographic waters, metal ions, and relevant cofactors near the binding site. Metals are re-injected into PDBQT after Meeko processing with explicit AD4 atom types and charges.

**xTB strain filter**: Optional post-docking step (`xtbConfig.strainFilter`). Computes GFN2-xTB strain energy per pose via batch mode (`score_xtb_strain.py --mode batch_strain`). Results in `results/xtb_strain.json`. Displayed as "Strain" column in DockStepResults (yellow >5 kcal/mol, red >8).

**xTB pre-optimization**: Optional pre-docking step (`xtbConfig.preOptimize`). Optimizes ligand geometry with GFN2-xTB before Meeko PDBQT conversion.

## MCMM Mode
Standalone conformer generation is exposed in the UI as **MCMM**. Internally the workflow state is still named `conform`, but user-facing docs and tabs should call this MCMM mode. Three conformer generation methods:

- **ETKDG**: Fast RDKit distance geometry
- **MCMM**: Monte Carlo Multiple Minimum with Sage 2.3.0 + OBC2 implicit solvent (default: 50 max conformers, 1.0 A RMSD cutoff, 5.0 kcal/mol energy window, 1000 steps, 298 K, amide cis/trans sampling)
- **CREST**: GFN2-xTB metadynamics via external `crest` binary (conda). Most thorough — uses biased sampling to escape local minima, ranks at semiempirical QM level with ALPB solvation

Optional **xTB reranking** toggle re-ranks ETKDG/MCMM conformers by GFN2-xTB single-point energy (parallelized via ThreadPoolExecutor). CREST conformers are already xTB-ranked so this is skipped for CREST.

## Map Mode
Water thermodynamics analysis using **GIST** (Grid Inhomogeneous Solvation Theory) via cpptraj from AmberTools. Computes solute-water energy, water-water energy, and translational/orientational entropy on a 3D grid around the binding site. Decomposes into three pharmacophore channels:
- **Hydrophobic** (green) — regions where displacing water gains free energy
- **H-bond donor** (blue) — ligand donor binding opportunities
- **H-bond acceptor** (red) — ligand acceptor binding opportunities

Output: 3 OpenDX files + hotspot JSON with clustered expansion vectors. Requires an MD trajectory (auto-launches a short 2-5 ns simulation if none loaded). Results saved to `surfaces/pocket_map_solvation/`.

## FEP Panel
`FepScoringPanel` and `run_abfe.py` are present in the codebase, and `state().mode === 'score'` renders the panel. In the current header UI the `FEP` tab is disabled, so this is implemented but not exposed as a normal selectable tab.

## View Mode
NGL viewer with queue navigation for docking poses or cluster centroids. Trajectory playback updates coordinates in place after the first frame instead of reparsing PDBs. Supports clustering, RMSD/RMSF/H-bond/contact analysis, binding-site maps, surface coloring, and multi-structure layer alignment.

**Scoring panel**: "SCORE" floating button appears when both protein and ligand are loaded. Runs Vina rescore + xTB strain energy in parallel via `SCORE_COMPLEX` IPC handler. Displays results in `ScoringPanel.tsx` overlay.

## xTB (GFN2-xTB)
GFN2-xTB 6.7.1 is vendored at `vendor/xtb-env/bin/xtb` (conda env, arm64). Detected at runtime by `getQupkakeXtbPath()` in main.ts. Used for:
- **MD cluster scoring**: xTB strain energy = E(pose) - E(free minimum) per cluster centroid
- **Docking strain filter**: batch strain scoring of all docked poses in a single Python invocation
- **Conformer reranking**: re-rank ETKDG/MCMM conformers by xTB single-point energy
- **CREST**: xTB serves as the energy function for CREST metadynamics conformer search
- **Ligand pre-optimization**: optional geometry optimization before docking
- **Viewer scoring**: single-complex strain energy from the SCORE button
- **QupKake pKa**: GFN2-xTB features for neural-network pKa prediction

`score_xtb_strain.py` is the standalone CLI utility. Modes: `single_point`, `optimize`, `strain`, `batch_strain`. Uses ALPB water solvation. Output parsed by main.ts via `XTB_SP_ENERGY:`, `XTB_OPT_ENERGY:`, `XTB_STRAIN:`, `BATCH_STRAIN_JSON:` stdout lines.

**Note**: xTB 6.4.1 (still at `vendor/xtb-6.4.1/`) has a bug where geometry optimization + ALPB solvation fails with SCF convergence errors. Always prefer the 6.7.1 binary.

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
Bundled app paths prefer `process.resourcesPath/scripts` and `process.resourcesPath/python/bin/python`. Dev mode falls back to `deps/staging/scripts/` and typical `openmm-metal`/`fraggen` conda env paths. Legacy env vars such as `FRAGGEN_ROOT` and `FRAGGEN_PYTHON` are still honored. xTB binary is resolved by `getQupkakeXtbPath()` which searches `vendor/xtb-env/bin/xtb` first, then `vendor/xtb-6.4.1/`, then conda envs.

## GPU Platform Cascade
For the canonical bundled MD runner in `deps/staging/scripts/run_md_simulation.py`, the runtime platform order is `CUDA → OpenCL (cl2Metal) → Metal → CPU`. OpenCL is currently tried before Metal in both benchmarking and full simulation setup.

## Metal Backend
Native Metal backend work lives in the separate `pseudo-control/Ember-Metal` repo. Use that repo's instruction file for low-level backend architecture, profiling, and MSL porting notes rather than duplicating them here.

## Known Limitations
- macOS only in practice for the current app workflow
- App is unsigned, so launch from `/Applications` or Spotlight rather than Launchpad
- `run_md_simulation.py` exists both at repo root and in `deps/staging/scripts/`; the staging copy is canonical for the app
- The header `FEP` tab is disabled even though the scoring panel exists in code
- CREST requires separate `crest` binary installation (`conda install -c conda-forge crest`)

## License
MIT. GPL-licensed scientific tools are invoked as separate processes. Meeko and Molscrub are Apache-2.0.
