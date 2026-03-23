# Full E2E Coverage Loop

Run comprehensive Playwright E2E tests against the Ember app. Build first, then write and run tests iteratively until all pass.

## Prerequisites
- `npm run build:electron && npm run build` must pass
- conda env `openmm-metal` active with all Python deps
- Test protein at `ember-test-protein/8tce.cif` + `ember-test-protein/kiv/kiv.sdf`

## Test Fixture Setup
Every test that needs a project must create one via IPC:
```typescript
await window.evaluate(async () => {
  await (window as any).electronAPI.ensureProject('e2e_test_run');
});
```

File dialogs must be mocked:
```typescript
await window.evaluate((pdbPath) => {
  (window as any).electronAPI.selectPdbFile = async () => pdbPath;
}, fixturePath);
```

## Test Coverage Checklist

### 1. App Boot & Navigation (tests/e2e/app-boot.spec.ts)
- [x] App launches without crash
- [x] Header shows 4 mode tabs (View, MCMM, Dock, Simulate)
- [x] View tab active by default
- [x] No console errors on boot
- [x] Tabs disabled without project
- [x] Tabs enabled after creating project

### 2. MCMM Pipeline (tests/e2e/mcmm-pipeline.spec.ts)
Test via SMILES input (no file dialogs):
- [x] Load ligand via SMILES input → "Enter SMILES" converts it, Continue enables
- [x] Configure: method dropdown has ETKDG/MCMM/CREST options
- [x] Configure: MCMM-specific controls appear when method=mcmm (steps, temp, amide toggle)
- [x] Configure: CREST-specific info appears when method=crest
- [x] Run ETKDG (fastest): progress shows, completes, results appear
- [x] Results: table shows conformers with energy column (kcal/mol)
- [x] Results: energies are numeric, min energy row shows 0.0
- [x] Results: "View 3D" button transitions to viewer with conformer queue
- [ ] Run MCMM: completes, results appear with energies
- [ ] Viewer after MCMM: conformer queue navigation works (prev/next)
- [ ] Viewer after MCMM: individual conformer selection and inspection

### 3. Docking Pipeline (tests/e2e/docking-pipeline.spec.ts)
Test with 8TCE via PDB ID fetch + SMILES for ligand:
- [x] Load receptor via PDB ID fetch → ligand detection runs → detected ligands dropdown populates
- [x] Select reference ligand from dropdown
- [x] Load docking ligand via SMILES
- [x] Configure: exhaustiveness input works (set to 1 for speed)
- [x] Configure: poses input works (set to 1 for speed)
- [x] Configure: protonation toggle enables/disables pH inputs
- [x] Configure: stereoisomer toggle works
- [x] Configure: conformer method dropdown works
- [x] Configure: pocket refinement toggle works
- [ ] Run docking (exhaustiveness=1, poses=1): progress, completion
- [ ] Results: table shows Vina affinity column
- [ ] Results: xTB strain column appears (if xTB available)
- [ ] Results: sorting by Vina column works
- [ ] Results: "View 3D" loads pose in viewer
- [ ] Results: "Simulate" navigates to MD configure with docked pose

### 4. MD Simulation Pipeline (tests/e2e/md-pipeline.spec.ts)
Test with 8TCE receptor via PDB ID fetch + SMILES for ligand:
- [x] Load PDB via PDB ID fetch → ligand detected → mode shows "Protein + Ligand"
- [x] Load via SMILES: mode shows "Ligand Only"
- [x] Configure: force field preset dropdown works (ff19sb-opc default)
- [ ] Configure: production duration dial adjusts value
- [ ] Configure: temperature input works
- [ ] Configure: "Estimate Runtime" runs benchmark, shows ns/day
- [ ] Run simulation (0.01 ns = 10 ps, minimal): progress bar updates
- [ ] Results: clustering table shows population percentages
- [ ] Results: xTB strain + Vina rescore columns appear (if protein+ligand)
- [ ] Results: "Play Trajectory" loads trajectory in viewer
- [ ] Results: cluster row click selects it, "View 3D" opens centroid
- [ ] Results: MD analysis report (PDF) generated
- [ ] Analysis: RMSD plot renders in AnalysisPanel
- [ ] Analysis: RMSF plot renders
- [ ] Analysis: H-bond count chart renders
- [ ] Analysis: contact map renders
- [ ] Analysis: torsion/dihedral analysis panel (MDTorsionPanel) loads and shows data

### 5. Viewer Mode — NGL State Tests (tests/e2e/viewer-ngl.spec.ts)
Expose `window.__nglStage` in test mode. Assert on NGL internal state, not screenshots.

**Import sources & structure loading:**
- [x] Import PDB (holo complex) → `stage.compList.length === 1`, repr includes cartoon
- [x] Import ligand-only SDF via SMILES → compList has ligand, repr is ball+stick
- [x] Import protein-only PDB (no ligand) → compList loaded, no ligand repr
- [ ] MCMM conformer output → queue loads, `compList` reflects current conformer
- [ ] Docking output → receptor retained, ligand component swaps on queue nav
- [ ] MD cluster centroid → centroid PDB loaded in compList

**Representations & rendering (tests/e2e/viewer-controls.spec.ts):**
- [x] Protein representation dropdown → changes repr type (cartoon→ribbon→spacefill)
- [x] Ligand representation dropdown → changes ligand repr (ball+stick→stick→spacefill)
- [x] Surface toggle → adds/removes surface repr from `reprList`
- [ ] Surface electrostatic → surface repr has colorScheme with computed values (not all zeros)
- [x] Pocket residues → toggle changes store state
- [x] Clipping plane slider → stage.parameters.clipDist changes
- [x] Hide waters/ions toggle → store state changes
- [x] Interactions toggle → store state changes
- [x] Polar H toggle → store state changes
- [x] Reset button → clipping/fog parameters restored

**Camera & interaction:**
- [ ] Camera centered on ligand after loading a complex (viewerControls.position near ligand centroid)
- [x] Rotation: simulate drag → `viewerControls.rotation` changes
- [ ] Auto-view on new structure load (camera encompasses bounds)

**Queue navigation:**
- [ ] Conformer queue: next/prev → compList reflects new conformer, old one replaced
- [ ] Docking pose queue: next/prev → ligand swaps, receptor stays
- [ ] Queue index display updates (e.g., "2 of 10")

**Layer management:**
- [ ] Import second structure → layer count increases
- [ ] Layer visibility toggle → component visibility changes in NGL
- [x] Close/clear → stage.compList becomes empty
- [ ] Align All → structures superposed (component positions changed)

**Trajectory playback:**
- [ ] Load DCD → trajectory controls appear, frame count > 0
- [ ] Play → frame index advances
- [ ] Step forward/backward → frame index ±1
- [ ] Speed control → playback rate changes
- [ ] Center tracking (ligand) → camera follows ligand through frames

### 6. Computational Output Verification (tests/e2e/output-verification.spec.ts)
Every job type must produce real output files. Verify files exist and contain valid data.

**MCMM/Conformer outputs:**
- [ ] ETKDG: output SDF exists, has ≥1 conformer, energies are numeric
- [ ] CREST (methylcyclobutane): output SDF exists, conformers ranked, energies populated
- [ ] Output directory matches expected path pattern (`conformers/{run}/`)

**Docking outputs:**
- [ ] `all_docked.sdf` exists after docking run
- [ ] Vina scores in SDF properties, range -12 to +2 kcal/mol
- [ ] Pose count matches requested (`poses=1` → 1 pose per ligand)
- [ ] `cordial_scores.json` exists if CORDIAL enabled
- [ ] `xtb_strain.json` exists if xTB strain filter enabled

**MD Simulation outputs:**
- [ ] `system.pdb` topology file exists
- [ ] `trajectory.dcd` exists with frames
- [ ] `energy.csv` exists, energy decreases during equilibration
- [ ] `seed.txt` written with random seed
- [ ] `final.pdb` exists (last frame)
- [ ] `analysis/clustering/` contains centroid PDB files
- [ ] `analysis/scored_clusters/` has scoring JSON
- [ ] `analysis/rmsd/`, `analysis/rmsf/`, `analysis/hbonds/` populated after analysis

**Receptor preparation intermediates:**
- [ ] Raw input PDB/CIF preserved in `structures/`
- [ ] `receptor_prepared.pdb` has hydrogens added (atom count > raw)
- [ ] Metadata JSON with PROPKA results, disulfides, HIS tautomers

### 7. Cross-Mode Integration (tests/e2e/integration.spec.ts)
- [ ] Dock → View 3D → viewer shows docked pose with ligand visible
- [ ] Dock → Simulate → MD load pre-populated with docked complex (receptor + ligand paths set)
- [ ] MCMM → View 3D → conformer queue in viewer with prev/next navigation
- [ ] MCMM → View 3D → individual conformer inspection (select from queue)
- [ ] MD Results → Play Trajectory → trajectory controls visible, frames advance
- [ ] MD Results → cluster row → View 3D → centroid structure in viewer
- [ ] MD cluster scoring: xTB strain + Vina rescore columns populated in results table
- [x] Switch modes during idle → no crashes, no stale state
- [ ] Create project → switch all tabs → back to View → state preserved
- [ ] PDB ID fetch works from both Dock and Simulate tabs (no file dialog needed)

## How to Run
```bash
# All tests
npm run test:e2e

# Specific pipeline
npx playwright test tests/e2e/mcmm-pipeline.spec.ts --headed

# With debug
PWDEBUG=1 npx playwright test tests/e2e/docking-pipeline.spec.ts
```

## Writing New Tests
1. Use `import { test, expect } from './fixtures'` for the shared app/window
2. Create project first if test needs tab switching: `await createTestProject(window, '__e2e_name__')`
3. **DO NOT mock file dialogs** — use PDB ID fetch for proteins, SMILES input for ligands, or direct IPC calls
4. PDB ID input: `input[placeholder*="8TCE"]` + Fetch button
5. SMILES input: `textarea` + "Enter SMILES" button
6. Use `.tab.tab-sm` for mode tabs, `.btn.btn-primary` for CTAs
7. ViewerMode is always mounted (CSS-hidden) — use specific selectors to avoid matching hidden elements
8. Computational tests need long timeouts (`test.setTimeout(120_000)`)
9. Check `state().dock.step` etc. via `window.evaluate` to verify state transitions
