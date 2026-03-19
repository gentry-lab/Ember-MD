# Ember

Desktop app for GPU-accelerated molecular dynamics on Apple Silicon. Five modes: **View** (NGL 3D viewer), **Dock** (AutoDock Vina), **Simulate** (OpenMM AMBER MD), **Score** (ABFE FEP), **Map** (binding site mapping).

**Repos**: `pseudo-control/Ember-MD` (this repo), `pseudo-control/Ember-Metal` (native Metal GPU backend, separate CLAUDE.md)
**Stale docs**: `README.md` and `deps/README.md` still reference FragGen/GNINA/Linux — use this file instead.

## Tech Stack
- **Frontend**: SolidJS + TypeScript + Tailwind/DaisyUI (wireframe/business themes)
- **Desktop**: Electron 27, Webpack
- **MD Engine**: OpenMM 8.1.2 (AMBER ff19SB/ff14SB + OPC/TIP3P + OpenFF Sage 2.3.0)
- **GPU**: OpenCL (cl2Metal, preferred >3K atoms) → Metal (native MSL, wins <3K) → CPU
- **Visualization**: NGL (WebGL), always-mounted CSS-hidden to avoid OOM on mode switch
- **Cheminformatics**: RDKit, Meeko (PDBQT prep), Molscrub (protonation), PDBFixer, MDAnalysis, AmberTools
- **Scoring**: CORDIAL neural network rescoring (optional, `~/Desktop/CORDIAL`)

## Build
```bash
npm start               # Build + run dev mode
npm run build:electron  # TS → electron-dist/
npm run build           # Webpack → dist-webpack/
npm run dist:mac        # Bundle .dmg via scripts/bundle-mac.sh (conda-pack → electron-builder → create-dmg)
                        # Auto-generates assets/dmg-background.png (gradient + text via Pillow)
                        # Requires: brew install create-dmg, Finder automation permission
```

## Project Structure
```
/electron/main.ts          — IPC handlers, subprocess management, path resolution
/electron/preload.ts       — Context bridge (inlined channel names)
/src/App.tsx               — Root: ViewerMode always mounted (CSS-hidden), Switch for other modes
/src/stores/workflow.ts    — SolidJS signals (WorkflowState, MDState, DockState, ViewerState)
/src/utils/projectPaths.ts — Project directory layout (DockingPaths, SimulationPaths)
/src/utils/jobName.ts      — Job name generation + folder naming
/src/components/layout/WizardLayout.tsx — Header: mode tabs | project+job selector | step indicators
/src/components/steps/     — DockStep{Load,Configure,Progress,Results}, MDStep{Load,Configure,Progress,Results}
/src/components/viewer/    — ViewerMode, TrajectoryControls, ClusteringModal, AnalysisPanel, FepScoringPanel, LayerPanel, BindingSiteMapPanel
/src/components/map/       — MapMode (binding site mapping)
/shared/types/             — md.ts, dock.ts, ipc.ts (ProjectJob, IpcChannels), electron-api.ts, errors.ts
/scripts/score_cordial.py  — CORDIAL scoring (outside deps/staging/)
/deps/staging/scripts/     — Python scripts (mypy-checked, TypedDicts in utils.py match shared/types/*.ts)
                             run_md_simulation.py, run_vina_docking.py, run_abfe.py,
                             detect_pdb_ligands.py, extract_xray_ligand.py, enumerate_protonation.py,
                             enumerate_stereoisomers.py, generate_conformers.py, cluster_trajectory.py,
                             analyze_*.py, utils.py, etc.
```

## Project Directory Layout
Each project under `~/Ember/{projectName}/`. Jobs are self-contained. Defined in `projectPaths.ts`.
```
{project}/
  .ember-project
  structures/                        — Imported PDB/CIF (CIF→PDB: 8tce.cif → 8tce.pdb)
  surfaces/binding_site_map/         — OpenDX interaction grids

  docking/Vina_{ligandId}/           — Self-contained docking job
    inputs/receptor.pdb, reference_ligand.sdf, ligands/*.sdf
    prep/                            — Intermediates (extraction, protonation, conformers)
    results/all_docked.sdf, cordial_scores.json, poses/*_docked.sdf.gz

  simulations/{ff}_MD-{temp}K-{ns}ns/ — Self-contained simulation job
    system.pdb, trajectory.dcd, final.pdb, energy.csv, seed.txt
    analysis/clustering/, rmsd/, rmsf/, hbonds/, contacts/

  fep/                               — FEP scoring results
```
**Legacy fallbacks** (all scanners check new → old): `inputs/receptor.pdb` → `*_receptor_prepared.pdb`; `results/poses/` → `poses/` → top-level; unprefixed `system.pdb` → `*_system.pdb`; `structures/` → `raw/`.

**Path API**: `projectPaths(baseDir, name).docking('Vina_HWF')` returns `{ root, inputs, inputsLigands, prep, results, resultsPoses }`. `.simulations(run)` returns `{ root, inputs, results, analysis, analysisClustering }`.

## Header UI
Three-zone layout: mode tabs (left) | project name + job selector dropdown (center, absolute) | step indicators (right, `ml-auto`). Job selector shows "N jobs" placeholder, grouped by docking/simulation. Selecting a job loads it in viewer.

## Simulate Mode
**Equilibration** (~360ps AMBER-style): restrained min → graduated min → NVT heating 5→100K → NPT heating 100→300K → restrained NPT → restraint release → unrestrained NPT → production (4fs HMR).

**Random seed**: `MDConfig.seed` (0=auto). Applied to `setVelocitiesToTemperature()` and `LangevinMiddleIntegrator.setRandomNumberSeed()`. Saved to `seed.txt`. Different seeds = independent replicates.

**Presets**: `ff19sb-opc` (default, 4-site OPC), `ff14sb-tip3p` (fast), `ff19sb-opc3`, `charmm36-mtip3p`. Ligand: OpenFF Sage 2.3.0.

**Key details**: FF path is `amber/protein.ff19SB.xml`. Production uses single precision on OpenCL (Apple doesn't support mixed). Restraints use `periodicdistance()^2` for PBC.

## Dock Mode
Vina pipeline: PDB→PDBQT receptor prep (Meeko) → SDF→PDBQT ligand prep (Meeko, preserves SMILES in REMARKs) → autobox from reference ligand → Vina docking → post-dock pocket refinement (OpenMM, Sage 2.3.0 + OBC2, receptor-restrained) → multi-pose SDF.gz. Optional CORDIAL rescoring, MCS core-constrained RMSD. Protonation via Molscrub (28 curated pKa rules, tautomer enumeration). Optional stereoisomer enumeration (RDKit `EnumerateStereoisomers`, strips 3D-inferred chirality, `onlyUnassigned=True`). MCMM conformer search (Sage 2.3.0 + OBC2 implicit solvent, ring pucker re-embedding). Pipeline: protonation → stereoisomers → conformers → docking → pocket refinement.

**Receptor prep**: removes docking ligand + crystallization artifacts (EDO, GOL, SO4...), retains crystallographic waters within 3.5 Å, metal ions (Zn/Mg/Mn/Ca/Fe — Vina AD4 types), and enzymatic cofactors (NAD/FAD/HEM/ATP...) within 5 Å of binding site. Metals are injected into PDBQT after Meeko processing with correct AD4 atom types and charges.

## Score Mode
ABFE via alchemical FEP (`run_abfe.py`): snapshot selection → complex leg (Boresch restraints) + solvent leg → lambda windows → MBAR/BAR → ΔG_bind. Fast (9 windows, 1ns) or Accurate (12 windows, 2ns).

## View Mode
NGL viewer with queue navigation (page-turn arrows for docking poses or cluster centroids). Trajectory playback uses `structure.updatePosition()` for in-place coordinate updates after first frame (no PDB re-parsing or component recreation). Clustering (K-means/DBSCAN/hierarchical), analysis panel (RMSD/RMSF/H-bonds/contacts), binding site maps (OpenDX), surface coloring (hydrophobic/electrostatic). Layer system for multi-structure alignment. Maestro-style interaction colors.

## Logging
`~/Ember/logs/ember-<timestamp>.log` — captures main process + renderer console output. Tags: `[Viewer]`, `[Dock]`, `[MD]`, `[FEP]`, `[Nav]`, `[Store]`.

## Key Patterns
```typescript
// SolidJS store update
setState(s => ({ ...s, md: { ...s.md, config: { ...s.md.config, ...config } } }));

// Load job in viewer (resetViewer clears NGL components but stage persists)
resetViewer(); setViewerPdbPath(pdb); setViewerLigandPath(sdf); setMode('viewer');

// NGL ligand selection: use resname, not chain
return `[${ligand.resname}] and ${ligand.resnum}`;
```

## Path Resolution (main.ts)
Bundled: `Resources/scripts/` + `Resources/python/bin/python`. Dev: `deps/staging/scripts/` + `~/miniconda3/envs/openmm-metal/bin/python`. `condaEnvBin` prepended to PATH for `sqm`.

## GPU Platform Cascade
`CUDA → OpenCL (cl2Metal) → Metal (native MSL) → CPU`. OpenCL preferred >3K atoms (cl2Metal IR optimizations). Metal wins <3K atoms (faster dispatch). macOS 26 blocks `__asm("air....")` — VENDOR_APPLE disabled, local-memory fallback.

## Metal Backend
Separate repo: `pseudo-control/Ember-Metal`. 46/47 tests pass, ~206 ns/day on M4 (22K atoms). See that repo's CLAUDE.md for architecture, profiling, MSL gotchas, and optimization history.

## Known Limitations
- macOS only (Apple Silicon). No CUDA/Linux.
- App unsigned — launch from /Applications or Spotlight, not Launchpad.
- `run_md_simulation.py` exists at both repo root (Metal testing) and `deps/staging/scripts/` (canonical).

## License
MIT. Bundles GPL-2.0 components (MDAnalysis) as separate processes. Meeko/Molscrub are Apache-2.0.
