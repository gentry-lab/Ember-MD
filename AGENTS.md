# Ember

Desktop app for GPU-accelerated molecular dynamics on Apple Silicon. Two modes: **Simulate** (AMBER MD) and **Viewer** (NGL molecular viewer with trajectory playback).

**Repo**: `pseudo-control/Ember-MD` (private, GitHub)

**Note**: `README.md` and `deps/README.md` are stale — they still reference FragGen/OpenSBDD, Linux, CUDA, and GNINA. Do not use them as a source of truth for architecture or packaging. Use this file instead.

**Script duplication**: `run_md_simulation.py` exists at both repo root (reference copy) and `deps/staging/scripts/` (bundled copy with plugin loading). The staging copy is canonical for the app; the root copy is for standalone Metal backend testing.

## Tech Stack
- **Frontend**: SolidJS + TypeScript + Tailwind/DaisyUI (wireframe theme, system fonts)
- **Desktop**: Electron 27, Webpack
- **MD Engine**: OpenMM 8.1.2 (AMBER ff19SB/ff14SB + OPC/TIP3P + OpenFF Sage 2.0)
- **GPU**: Apple OpenCL (cl2Metal) + openmm-metal plugin (HIP platform)
- **Visualization**: NGL (WebGL molecular viewer)
- **Cheminformatics**: RDKit, OpenBabel, PDBFixer, MDAnalysis, AmberTools

## GPU Platform Cascade
```
CUDA → HIP (openmm-metal plugin) → OpenCL (cl2Metal) → CPU
```
- **HIP/Metal**: Philip Turner's openmm-metal plugin, registers as "HIP" to bypass OpenMM's energy minimizer checks. Fixed for macOS 26 by disabling VENDOR_APPLE SIMD paths (2 line change in MetalContext.cpp + MetalNonbondedUtilities.cpp). All 34 tests pass.
- **OpenCL**: Apple's cl2Metal translates OpenCL to Metal GPU instructions. ~2.5% slower than HIP but zero maintenance. Works across all macOS versions (Ventura through Tahoe).
- **Performance**: ~210 ns/day on M4 for 22K atom protein-ligand system. ~51 ns/day for 92K atom ApoA1.
- **SIMD note**: macOS 26 blocks `__asm("air....")` inline assembly in cl2Metal. No subgroup extensions available. SIMD intrinsics only accessible via native Metal Shading Language (not implemented).

## Project Structure
```
/electron/
  main.ts               # IPC handlers, child process management, path resolution
  preload.ts            # Context bridge (inlined channel names)

/src/
  App.tsx               # Root + step routing (Simulate & Viewer modes)
  stores/workflow.ts    # SolidJS signals (WorkflowState, MDState, ViewerState)
  utils/jobName.ts      # Job name generation + folder naming
  components/layout/WizardLayout.tsx  # Mode tabs + step indicators + header
  components/AboutModal.tsx           # License info (MIT + dependencies)
  components/HelpModal.tsx            # Quick reference
  components/steps/
    MDStep{Load,Configure,Progress,Results}.tsx  # MD simulation steps
  components/viewer/
    ViewerMode.tsx            # NGL molecular viewer
    TrajectoryControls.tsx    # Playback UI
    ClusteringModal.tsx       # Trajectory clustering
    AnalysisPanel.tsx         # MD analysis tools

/shared/types/
  md.ts                 # MDConfig, MDStage, MDSystemInfo, MDBenchmarkResult
  ipc.ts                # IPC channels, trajectory types, clustering types
  electron-api.ts       # window.electronAPI typing
  gnina.ts              # Legacy types (kept for backend compatibility)

/deps/staging/scripts/  # Python scripts called by Electron via subprocess
  run_md_simulation.py  # Full MD pipeline (build, equilibrate, produce)
  extract_xray_ligand.py # Ligand extraction with OpenBabel bond orders
  cluster_trajectory.py, analyze_*.py, generate_md_report.py  # Analysis
```

## Build Commands
```bash
npm run build:electron  # TS → electron-dist/
npm run build           # Webpack → dist-webpack/
npm start               # Build + run dev mode
npm run dist:mac        # Bundle self-contained .dmg (includes Python env)
```

## DMG Bundling (npm run dist:mac)
1. Builds TypeScript + Webpack
2. Packs conda env `openmm-metal` via `conda-pack` (cached at bundle-mac/python-env.tar.gz)
3. Extracts to bundle-mac/extra-resources/{python,scripts}
4. Copies Metal plugin dylibs to python/lib/plugins/
5. Packages with electron-builder (extraResources in package.json)
6. Creates styled DMG with `create-dmg` (background, icon positioning, Applications link)

**Requires**: `create-dmg` (brew install), Finder automation permission for Terminal

## Path Resolution (main.ts)
Bundled app checks `process.resourcesPath` first, then falls back to dev paths:
- **Scripts**: `Resources/scripts/` → `deps/staging/scripts/` → env var FRAGGEN_ROOT
- **Python**: `Resources/python/bin/python` → `~/miniconda3/envs/openmm-metal/bin/python` → `fraggen` env
- **PATH**: `condaEnvBin` prepended to `process.env.PATH` in `initializePaths()` so child processes find `sqm`, `obabel`, etc.
- **OpenBabel**: `extract_xray_ligand.py` sets `BABEL_LIBDIR` and `DYLD_LIBRARY_PATH` relative to `sys.executable`
- **OpenMM plugins**: `run_md_simulation.py` loads from both default and bundled plugin directories

## MD Simulation
**Equilibration Protocol** (AMBER-style, ~270ps protein-ligand, ~170ps ligand-only):
1. Restrained minimization → graduated minimization
2. NVT heating 5K→100K → NPT heating 100K→300K (barostat added at 100K)
3. NPT equilibration with backbone restraints → gradual release
4. Unrestrained NPT equilibration → production with HMR (4fs)

**Force Field Presets**:
- Fast: ff14SB + TIP3P
- Accurate (default): ff19SB + OPC (4-site water)
- Ligand: OpenFF Sage 2.0 (AM1-BCC charges via AmberTools sqm)

**Key detail**: Force field path is `amber/protein.ff19SB.xml` (not `amber19/`). Production OpenCL uses single precision (not mixed — Apple doesn't support it).

**Critical**: Restraint forces use `periodicdistance(x,y,z,x0,y0,z0)^2` for PBC compatibility.

## Viewer
- NGL WebGL viewer with multi-PDB queue navigation
- Trajectory playback: DCD files, 10fps base rate, smoothing=1 (every frame)
- Auto-centers on ligand after detection
- Clustering: K-means by ligand RMSD, saves centroid PDBs
- Analysis: RMSD, RMSF, H-bonds, full HTML report

## Key Patterns
```typescript
// SolidJS store
const [state, setState] = createSignal<WorkflowState>({...});
setState(s => ({ ...s, md: { ...s.md, config: { ...s.md.config, ...config } } }));

// View PDB from results (no resetViewer — causes NGL race condition)
setViewerPdbPath(pdbPath);
setMode('viewer');

// NGL autoView: Stage only takes duration, StructureComponent takes selection
(proteinComponent as any).autoView(ligandSelection);  // centers on ligand
stage.autoView();  // centers on everything

// Ligand selection: use resname, not chain
return `[${ligand.resname}] and ${ligand.resnum}`;
```

## Output Structure
**Naming**: `{jobName}_{ff}_MD-{temp}K-{ns}ns` (e.g., `bold-pulse-shark_ff19sb-OPC_MD-300K-1ns`)

Files: `_system.pdb`, `_trajectory.dcd`, `_energy.csv`, `_checkpoint.chk`, `_final.pdb`, `_equilibrated.pdb`, `simulation.log`, `clustering/`

## Known Limitations
- macOS only (Apple Silicon). No CUDA, no Linux in this version.
- OpenCL deprecated by Apple but stable. HIP plugin as fallback insurance.
- Small systems (<3K atoms) don't saturate GPU — PME/FFT dominates.
- App unsigned — won't appear in Launchpad. Launch from /Applications or Spotlight.
- DMG styling requires Terminal→Finder automation permission in System Settings.

## OpenMM Metal Backend — Reference Files

Reference files for building a native Apple Metal GPU backend for OpenMM:

| File | Purpose |
|------|---------|
| `run_md_simulation.py` | Full OpenMM MD simulation script — defines every API feature the Metal backend must support |
| `test_opc_water_model.py` | OPC 4-site water model validation suite (762 lines) — adapt as Metal platform correctness tests |

### What the Metal Backend Must Support

From `run_md_simulation.py`:

- **Forces**: HarmonicBondForce, HarmonicAngleForce, PeriodicTorsionForce, NonbondedForce (PME), CustomExternalForce (restraints with `periodicdistance`)
- **Integrators**: LangevinMiddleIntegrator (2fs equilibration, 4fs production with HMR)
- **Constraints**: HBonds (SHAKE/SETTLE)
- **Barostat**: MonteCarloBarostat (NPT)
- **Force fields**: ff19SB + OPC (4-site), ff14SB + TIP3P (3-site), OpenFF Sage 2.0 (ligand)
- **PME**: 1.0 nm cutoff, 0.0005 Ewald error tolerance
- **HMR**: Hydrogen mass repartitioning for 4fs timestep
- **Checkpointing**: `saveCheckpoint()` / `loadCheckpoint()`

### Performance Targets

- M4 base (10-core GPU): ~40-60 ns/day for a ~30K atom protein-ligand system
- CPU Reference baseline: ~3-5 ns/day on the same system

### Verification

1. All 45 native Metal tests must pass (see **Test Suite Architecture** below for tiers and CMake flags)
2. Run `run_md_simulation.py` on Metal platform and compare forces/energies against CPU Reference
3. Max deviation: <0.01 kJ/mol per atom
4. Adapt `test_opc_water_model.py` to validate OPC water on Metal (density, geometry, SETTLE)

## Native Metal Backend (Active Project)

**Goal**: Replace cl2Metal-compiled OpenCL kernels with native Metal Shading Language (MSL) kernels in the openmm-metal plugin.

**Source location**: `~/openmm-metal-project/openmm-metal/`

**Why**: macOS 26 broke `__asm("air....")` inline assembly that cl2Metal used for SIMD intrinsics. Current fix disables VENDOR_APPLE entirely (slower local-memory reductions). Native MSL restores SIMD intrinsics and enables iOS cross-compilation.

**Architecture**: OpenCL C .metal files → native MSL .metal files. Host code migrated from OpenCL C++ API (`cl::Context`, `cl::Buffer`, etc.) to Metal Objective-C++ API (`id<MTLDevice>`, `id<MTLBuffer>`, etc.) in .mm files.

**Performance targets**:
- M4 Mac: ≥210 ns/day (22K atoms, ff19SB/OPC) — match or exceed current cl2Metal baseline
- iPhone 17 (A19): 100+ ns/day — first production MD on a phone
- SIMD intrinsics: `simd_sum()`, `simd_ballot()`, `ctz()` — native, no `__asm` needed

**Validation**: All 45 native Metal tests must pass (27 short + 12 long + 6 very-long). Force deviation vs CPU Reference < 0.01 kJ/mol/nm per atom.

**Current status (2026-03-16)**: 45/45 tests pass on the native Metal branch. All three tiers green: short 27/27, long 12/12, very-long 6/6.

**Resolved blockers**:
- `TestMetalDispersionPME` — fixed by switching from Abramowitz-Stegun 5-term erfc polynomial (max error 1.5e-7, systematic bias accumulated across PME grid) to Numerical Recipes 9-term Chebyshev rational approximation (max error ~1.2e-7, better bias distribution). Energy now within 5e-5 tolerance.
- `TestMetalLocalEnergyMinimizer` — fixed by clamping `realToFixedPoint()` in `common.metal` to prevent undefined behavior when converting `inf` to `long` (Metal produces 0 for `(long)inf`, unlike OpenCL which saturates). Clamped forces now correctly trigger the existing CPU fallback in L-BFGS for extreme overlapping-particle configurations.

### Test Suite Architecture

The native Metal backend (`.build-metal/`) has **45 tests** across 3 tiers, controlled by CMake flags. Turner's original HIP build (`.build/`) had 34 tests — the 7-test difference is AMOEBA/Drude/WCA plugin tests that don't apply to native Metal.

**Short tests (27)** — `platforms/metal/tests/` — built by default:
```
TestMetalCheckpoints          TestMetalCustomCVForce         TestMetalGayBerneForce
TestMetalCMAPTorsionForce     TestMetalCustomExternalForce   TestMetalGBSAOBCForce
TestMetalCMMotionRemover      TestMetalCustomGBForce         TestMetalHarmonicAngleForce
TestMetalCompoundIntegrator   TestMetalCustomHbondForce      TestMetalHarmonicBondForce
TestMetalCustomAngleForce     TestMetalCustomTorsionForce    TestMetalMultipleForces
TestMetalCustomBondForce      TestMetalDeviceQuery           TestMetalPeriodicTorsionForce
TestMetalCustomCentroidBondForce  TestMetalFFT               TestMetalRandom
TestMetalRBTorsionForce       TestMetalRMSDForce             TestMetalSettle
TestMetalSort                 TestMetalVariableVerletIntegrator
```
These validate bonded forces, constraints, checkpoints, utilities. Necessary but not sufficient for Ember.

**Long tests (12)** — `platforms/metal/long_tests/` — requires `-DOPENMM_BUILD_LONG_TESTS=ON`:
```
TestMetalAndersenThermostat        TestMetalEwald                    TestMetalLocalEnergyMinimizer
TestMetalCustomManyParticleForce   TestMetalLangevinIntegrator       TestMetalMonteCarloFlexibleBarostat
TestMetalCustomNonbondedForce *    TestMetalLangevinMiddleIntegrator *  TestMetalNonbondedForce *
TestMetalDispersionPME             TestMetalVerletIntegrator         TestMetalVirtualSites
```

**Very-long tests (6)** — `platforms/metal/very_long_tests/` — requires both long tests flags + `-DOPENMM_BUILD_VERY_LONG_TESTS=ON`:
```
TestMetalBrownianIntegrator                TestMetalMonteCarloBarostat *
TestMetalCustomIntegrator                  TestMetalNoseHooverIntegrator
TestMetalMonteCarloAnisotropicBarostat     TestMetalVariableLangevinIntegrator
```

**Ember-critical tests** (marked with * above):
| Test | Why |
|------|-----|
| `TestMetalNonbondedForce` | PME electrostatics — dominant force calculation (~70% of runtime) |
| `TestMetalLangevinMiddleIntegrator` | The integrator Ember uses for all equilibration and production |
| `TestMetalMonteCarloBarostat` | NPT ensemble — every Ember equilibration and production run |
| `TestMetalCustomNonbondedForce` | OpenFF Sage 2.0 ligand parameters |
| `TestMetalLocalEnergyMinimizer` | Used at the start of every Ember simulation |

**To build all tiers**:
```bash
cd ~/openmm-metal-project/openmm-metal/.build-metal
cmake .. -DOPENMM_BUILD_LONG_TESTS=ON -DOPENMM_BUILD_VERY_LONG_TESTS=ON
make -j8
```

**To run all tests**:
```bash
for test in .build-metal/TestMetal*; do
  printf "%-46s" "$(basename $test)"
  $test single -1 -1 >/dev/null 2>&1 && echo PASS || echo FAIL
done
```

### Metal Shading Language (MSL) Compatibility Gotchas

When porting OpenCL kernels to native MSL, these are non-obvious differences that cause silent compilation failures or wrong results:

**Missing math intrinsics:**
- `erf()` does not exist in MSL. `erfc()` does. Use `1.0f - erfc(x)` as a drop-in replacement for `erf(x)`.
- The source transformer in `MetalContext.mm` maps `ERF(x)` → `1.0f-erfc(x)` and `ERFC(x)` → `erfc(x)`.

**Reserved words that differ from OpenCL:**
- `thread` — MSL address space qualifier. OpenCL kernels that use `int thread = LOCAL_ID;` will compile, but downstream uses like `if (thread < 32)` can fail or misbehave. The transformer must rename these.
- `uint8` — In OpenCL, this is an 8-element vector type. In MSL, it's a scalar `uint8_t` alias. The transformer rewrites `uint8` vector types to avoid the collision.

**Dual kernel source paths:**
- Some kernels exist in **both** `platforms/metal/src/kernels/*.metal` (Metal-specific) and `platforms/common/src/kernels/*.cc` (shared). The `.cc` versions are compiled into `CommonKernelSources.cpp` at build time and may be what actually runs at runtime, depending on the code path.
- Example: `andersenThermostat` has both `.metal` and `.cc` copies. Patching only the `.metal` file won't fix tests that load the common source. **Always check and patch both.**

**Test file format:**
- `TestMetalNonbondedForce` was renamed from `.cpp` to `.mm` because it uses `#import <Metal/Metal.h>` for GPU memory queries. The `long_tests/CMakeLists.txt` glob includes `"*Test*.mm"` to pick it up.

### Debug Environment Variables

Set these when running tests or debugging the Metal backend:

| Env Var | Purpose |
|---------|---------|
| `OPENMM_METAL_PROFILE_KERNELS=1` | GPU timing per kernel dispatch (Chrome Tracing format) |
| `OPENMM_METAL_DUMP_SOURCE=1` | Writes each compiled MSL to `/tmp/metal_kernel_N.metal` (post-transformation source — what Metal actually sees) |
| `OPENMM_METAL_LOG_ARGS=1` | Logs every `executeKernel` call with kernel name, grid/threadgroup sizes, and argument bindings (buffer names+sizes, primitive byte sizes, threadgroup memory) |

**Usage**: `OPENMM_METAL_DUMP_SOURCE=1 OPENMM_METAL_LOG_ARGS=1 python test_script.py 2>metal_debug.log`

### Error Handling Architecture

All Metal API calls are fail-fast with `fprintf(stderr, "[Metal] ...")` + `throw OpenMMException(...)`:

**Compilation & Pipeline:**
- **MSL compilation** (`createProgram`): On failure, always writes full transformed source to `/tmp/metal_kernel_FAILED_N.metal` (plus first 600 lines to stderr with line numbers). On success, logs function count. Source dump to `/tmp/` when `OPENMM_METAL_DUMP_SOURCE=1`.
- **Pipeline creation** (`createKernel`, `createUtilityKernel`): Logs available function names on failure.
- **Source transformation**: STEP 1 logs each wrapped param with kernel name (`kernel 'X': wrapped 'int Y' → 'constant int& Y'`). STEP 1b logs each mutable param fix with kernel name. STEP 2 logs function names and edit count. Pre/post transform dumps for `generateRandomNumbers` kernels.

**GPU Execution:**
- **Kernel execution** (`executeKernel`): Checks `commandBuffer.status` after every dispatch. Logs argument bindings when `OPENMM_METAL_LOG_ARGS=1`.
- **FFT** (`execFFT` VkFFT path): Nil checks, VkFFT error code logging, commandBuffer status check. Pipeline creation checks NSError pointer.
- **Energy reduction** (`reduceEnergy`): Nil checks on commandBuffer/encoder, status check.

**Buffer Operations:**
- **clearBuffer / clearAutoclearBuffers / copyTo**: Nil checks on commandBuffer/encoder, status check after completion.
- **Buffer allocation** (`initialize`): Checks `[mtlBuffer contents]` for nil before memset.
- **Upload/download** (`uploadSubArray`, `download`): Validates `[mtlBuffer contents]` and destination pointer are non-nil before memcpy.
- **Pinned buffer** (`pinnedMemory`): Nil check on `[pinned contents]` during context init.

**Force Kernel Compilation (context on failure):**
- **Nonbonded**: Logs force group, exclusion/symmetry flags, parameter counts.
- **Bonded**: Logs force source count and energy derivative count.
- **Ewald**: Logs kmax values and alpha.
- **Coulomb PME**: Logs grid dimensions and alpha.
- **LJPME dispersion**: Logs dispersion grid dimensions and dispersion alpha.
- **Nonbonded parameters**: Logs define count.
- **CPU PME fallback**: Logs when CPU PME unavailable and falling back to GPU path (previously silent).
- **Sort**: Logs data type, key type, element count. Also logs when short-list sort fails and falls back.
- **Compact (prefix sum)**: Logs on compilation failure.

**Device Init:**
- Logs device name, max buffer length (MB), recommended working set (MB), max threads per threadgroup.

**Key pattern**: Every `[commandBuffer waitUntilCompleted]` is followed by a status check. Every `newLibraryWithSource`/`newComputePipelineStateWithFunction` checks for nil + NSError. Every `[mtlBuffer contents]` is nil-checked before use. No silent degradation paths.

### Debugging C++ in the Metal Backend

When debugging numerical issues or GPU failures:
1. Run with `OPENMM_METAL_DUMP_SOURCE=1` to inspect the MSL that Metal actually compiles (post-transformation, with `constant int&` rewriting applied)
2. Run with `OPENMM_METAL_LOG_ARGS=1` to verify correct buffer bindings — if a kernel produces garbage, check that the right arrays are bound at the right indices
3. All `[Metal]` prefixed stderr output can be captured with `2>metal_debug.log`
4. MSL compilation errors include line/column numbers against the dynamically-generated source — the source dump is essential since line numbers refer to the assembled source (defines + common.metal + kernel source), not the original .metal file

## License
MIT. See LICENSE file. Bundles GPL-2.0 components (OpenBabel, MDAnalysis) as separate processes.
