// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, onMount, onCleanup, createSignal, Show } from 'solid-js';
import path from 'path';
import { workflowStore } from '../../stores/workflow';
import { buildDockFolderName, buildDockConformRunFolderName } from '../../utils/jobName';
import { projectPaths } from '../../utils/projectPaths';
import TerminalOutput from '../shared/TerminalOutput';

const DockStepProgress: Component = () => {
  const {
    state,
    appendLog,
    setDockStep,
    setDockTotalLigands,
    setDockCompletedLigands,
    setDockResults,
    setDockOutputDir,
    setDockPreparedLigandPath,
    setDockReferenceLigandPath,
    setDockCordialScored,
    setIsRunning,
    setCurrentPhase,
    setError,
    clearLogs,
  } = workflowStore;

  const [hasStarted, setHasStarted] = createSignal(false);
  const [cordialRunning, setCordialRunning] = createSignal(false);

  const api = window.electronAPI;

  // Subscribe to dock output events
  onMount(() => {
    const cleanup = api.onDockOutput((data) => {
      appendLog(data.data);

      // Parse DOCKING: X/N lines for authoritative progress count
      const dockingMatch = data.data.match(/DOCKING:\s*(\d+)\s*\/\s*(\d+)/);
      if (dockingMatch) {
        setDockCompletedLigands(parseInt(dockingMatch[1]));
      }
    });
    onCleanup(cleanup);
  });

  const runDocking = async () => {
    const dock = state().dock;
    if (!dock.receptorPrepared) return;
    if (!dock.receptorPdbPath) return;
    if (!dock.referenceLigandId) return;
    if (!dock.referenceLigandPath) return;
    if (dock.ligandSdfPaths.length === 0) return;

    setIsRunning(true);
    setCurrentPhase('generation');
    clearLogs();
    setDockTotalLigands(dock.ligandSdfPaths.length);
    setDockCompletedLigands(0);
    setDockPreparedLigandPath(null);

    // Build output directory path
    const defaultDir = await api.getDefaultOutputDir();
    const baseOutputDir = state().customOutputDir || defaultDir;
    const jobName = state().jobName.trim();
    const paths = projectPaths(baseOutputDir, jobName);
    const dockFolder = buildDockFolderName({
      referenceLigandId: dock.referenceLigandId,
      numLigands: dock.ligandSdfPaths.length,
    });
    const dockPaths = paths.docking(dockFolder);
    const outputDir = dockPaths.root;
    setDockOutputDir(outputDir);

    try {
      let ligandPaths = [...dock.ligandSdfPaths];
      let dockingReceptorPath = dock.receptorPrepared;
      let dockingReferenceLigandPath = dock.referenceLigandPath;
      const receptorPh = (dock.protonationConfig.phMin + dock.protonationConfig.phMax) / 2;
      const waterDist = dock.waterRetentionConfig.enabled ? dock.waterRetentionConfig.distance : 0;

      appendLog('--- Preparing canonical receptor... ---\n');
      const canonicalReceptorPath = path.join(dockPaths.prep, 'canonical_receptor.pdb');
      const canonicalReceptorResult = await api.prepareReceptor(
        dock.receptorPdbPath,
        dock.referenceLigandId,
        canonicalReceptorPath,
        waterDist,
        receptorPh
      );
      if (!canonicalReceptorResult.ok) {
        if (state().currentPhase !== 'idle') {
          setError(canonicalReceptorResult.error.message);
          setCurrentPhase('error');
        }
        setIsRunning(false);
        return;
      }
      dockingReceptorPath = canonicalReceptorResult.value;
      appendLog(`  Canonical receptor: ${path.basename(dockingReceptorPath)} (uniform pH ${receptorPh.toFixed(1)})\n\n`);

      appendLog('--- Preparing canonical docking complex... ---\n');
      const complexDir = path.join(dockPaths.prep, 'complex');
      const preparedComplexResult = await api.prepareDockingComplex(
        dockingReceptorPath,
        dock.referenceLigandPath,
        complexDir,
        dock.refinementConfig.chargeMethod,
        dock.protonationConfig.phMin,
        dock.protonationConfig.phMax,
        dock.protonationConfig.enabled
      );
      if (!preparedComplexResult.ok) {
        if (state().currentPhase !== 'idle') {
          setError(preparedComplexResult.error.message);
          setCurrentPhase('error');
        }
        setIsRunning(false);
        return;
      }
      dockingReceptorPath = preparedComplexResult.value.preparedReceptorPdb;
      dockingReferenceLigandPath = preparedComplexResult.value.preparedReferenceLigandSdf;
      setDockReferenceLigandPath(dockingReferenceLigandPath);
      appendLog(`  Canonical receptor: ${path.basename(dockingReceptorPath)}\n`);
      appendLog(`  Canonical X-ray ligand: ${path.basename(dockingReferenceLigandPath)}\n\n`);

      // Preprocessing: protonation enumeration
      if (dock.protonationConfig.enabled && ligandPaths.length > 0) {
        appendLog('--- Enumerating protonation states... ---\n');
        const protonDir = path.join(dockPaths.prep, 'protonated');
        const protonResult = await api.enumerateProtonation(
          ligandPaths, protonDir,
          dock.protonationConfig.phMin, dock.protonationConfig.phMax
        );
        if (!protonResult.ok) {
          if (state().currentPhase !== 'idle') {
            setError(protonResult.error.message);
            setCurrentPhase('error');
          }
          setIsRunning(false);
          return;
        }
        ligandPaths = protonResult.value.protonatedPaths;
        appendLog(`  ${ligandPaths.length} protonation variants generated\n\n`);
      }

      // Preprocessing: stereoisomer enumeration
      if (dock.stereoisomerConfig.enabled && ligandPaths.length > 0) {
        appendLog('--- Enumerating stereoisomers... ---\n');
        const stereoDir = path.join(dockPaths.prep, 'stereoisomers');
        const stereoResult = await api.enumerateStereoisomers(
          ligandPaths, stereoDir,
          dock.stereoisomerConfig.maxStereoisomers
        );
        if (stereoResult.ok && stereoResult.value.stereoisomerPaths.length > 0) {
          const before = ligandPaths.length;
          ligandPaths = stereoResult.value.stereoisomerPaths;
          const added = ligandPaths.length - before;
          appendLog(added > 0
            ? `  ${ligandPaths.length} stereoisomers generated (+${added} enantiomers)\n\n`
            : `  No unspecified stereocenters found\n\n`
          );
        } else {
          appendLog('  Stereoisomer enumeration unchanged\n\n');
        }
      }

      // Preprocessing: conformer generation
      if (dock.conformerConfig.method !== 'none' && ligandPaths.length > 0) {
        const methodLabel = dock.conformerConfig.method.toUpperCase();
        appendLog(`--- Generating conformers (${methodLabel})... ---\n`);
        const conformerRunFolder = buildDockConformRunFolderName({
          referenceLigandId: dock.referenceLigandId,
          numLigands: dock.ligandMolecules.length,
          method: dock.conformerConfig.method,
          maxConformers: dock.conformerConfig.maxConformers,
          protonation: dock.protonationConfig,
        });
        const confDir = paths.conformers(conformerRunFolder);
        const mcmmOpts = dock.conformerConfig.method === 'mcmm' ? {
          steps: dock.conformerConfig.mcmmSteps,
          temperature: dock.conformerConfig.mcmmTemperature,
          sampleAmides: dock.conformerConfig.sampleAmides,
        } : undefined;
        const confResult = await api.generateConformers(
          ligandPaths, confDir,
          dock.conformerConfig.maxConformers,
          dock.conformerConfig.rmsdCutoff,
          dock.conformerConfig.energyWindow,
          dock.conformerConfig.method,
          mcmmOpts
        );
        if (confResult.ok && confResult.value.conformerPaths.length > 0) {
          ligandPaths = confResult.value.conformerPaths;
          appendLog(`  ${ligandPaths.length} conformers generated\n`);
          appendLog(`  Saved reusable conformer job: ${path.join(jobName, 'conformers', conformerRunFolder)}\n\n`);
        } else {
          appendLog('  Conformer generation skipped\n\n');
        }
      }

      if (dock.xtbConfig.preOptimize && ligandPaths.length > 0) {
        if (dock.conformerConfig.method === 'crest') {
          appendLog('--- Skipping xTB pre-optimization for CREST conformers ---\n');
          appendLog('  CREST already produced xTB-level geometries\n\n');
        } else {
          appendLog('--- Pre-optimizing ligands with xTB... ---\n');
          const preoptDir = path.join(dockPaths.prep, 'xtb_preopt');
          const preoptResult = await api.preOptimizeDockLigands(ligandPaths, preoptDir);
          if (preoptResult.ok) {
            ligandPaths = preoptResult.value.optimizedLigandPaths;
            appendLog(`  xTB pre-optimization complete: ${preoptResult.value.optimizedCount} ligands optimized\n`);
            if (preoptResult.value.failedCount > 0) {
              appendLog(`  ${preoptResult.value.failedCount} ligands kept their original geometry after xTB failure\n`);
            }
            appendLog(`  Saved optimized ligands: ${path.join(jobName, 'docking', dockFolder, 'prep', 'xtb_preopt')}\n\n`);
          } else {
            appendLog(`  xTB pre-optimization skipped: ${preoptResult.error.message}\n\n`);
          }
        }
      }

      // Update total count after preprocessing may have expanded the ligand set
      setDockTotalLigands(ligandPaths.length);
      setDockPreparedLigandPath(ligandPaths[0] || null);

      console.log(`[Dock] Starting Vina docking: ${ligandPaths.length} ligands → ${outputDir}`);
      const result = await api.runVinaDocking(
        dockingReceptorPath,
        dockingReferenceLigandPath,
        ligandPaths,
        outputDir,
        dock.config
      );

      if (!result.ok) {
        if (state().currentPhase !== 'idle') {
          setError(result.error.message);
          setCurrentPhase('error');
        }
        setIsRunning(false);
        return;
      }

      // Docking complete -- pocket refinement if enabled
      // Refines in-place: writes back to results/poses/ so downstream
      // (CORDIAL rescoring + result parsing) uses the refined geometries
      console.log('[Dock] Refinement config:', JSON.stringify(dock.refinementConfig));
      appendLog(`\n[debug] refinementConfig=${JSON.stringify(dock.refinementConfig)}\n`);
      if (dock.refinementConfig?.enabled) {
        appendLog('\n--- Refining poses in pocket (Sage 2.3.0 + OBC2)... ---\n');
        try {
          const posesDir = path.join(outputDir, 'results', 'poses');
          const refineResult = await api.refinePoses(
            dockingReceptorPath!,
            posesDir,
            posesDir,
            dock.refinementConfig.maxIterations,
            dock.refinementConfig.chargeMethod
          );
          if (refineResult.ok && refineResult.value.refinedCount > 0) {
            appendLog(`\nRefinement complete: ${refineResult.value.refinedCount} poses refined\n`);
          } else if (!refineResult.ok) {
            appendLog(`\nRefinement skipped: ${refineResult.error.message}\n`);
          }
        } catch (err) {
          appendLog(`\nRefinement error: ${(err as Error).message}\n`);
        }
      }

      // Run xTB energy scoring (relative per compound)
      appendLog('\n--- Computing xTB energies... ---\n');
      try {
        const xtbResult = await api.scoreDockingXtbEnergy(outputDir);
        if (xtbResult.ok) {
          appendLog(`\nxTB energy scoring complete: ${xtbResult.value.count} poses scored\n`);
        } else {
          appendLog(`\nxTB energy scoring skipped: ${xtbResult.error.message}\n`);
        }
      } catch (err) {
        appendLog(`\nxTB energy scoring error: ${(err as Error).message}\n`);
      }

      // Run CORDIAL rescoring if enabled and available
      if (dock.cordialConfig.enabled && dock.cordialAvailable) {
        setCordialRunning(true);
        appendLog('\n--- Rescoring with CORDIAL... ---\n');

        try {
          const cordialResult = await api.runCordialScoring(
            outputDir,
            dock.cordialConfig.batchSize
          );
          if (cordialResult.ok) {
            setDockCordialScored(true);
            appendLog(`\nCORDIAL rescoring complete: ${cordialResult.value.count} poses scored\n`);
          } else {
            appendLog(`\nCORDIAL rescoring failed: ${cordialResult.error.message}\n`);
          }
        } catch (err) {
          appendLog(`\nCORDIAL rescoring error: ${(err as Error).message}\n`);
        }
        setCordialRunning(false);
      }

      // Parse results
      const parseResult = await api.parseDockResults(outputDir);
      if (parseResult.ok) {
        setDockResults(parseResult.value);
      } else {
        appendLog(`\nWarning: Failed to parse results: ${parseResult.error.message}\n`);
      }

      setCurrentPhase('complete');
      setIsRunning(false);
    } catch (err) {
      if (state().currentPhase !== 'idle') {
        setError((err as Error).message);
        setCurrentPhase('error');
      }
      setIsRunning(false);
    }
  };


  onMount(() => {
    // Only start docking if we haven't already started or completed
    const phase = state().currentPhase;
    if (!state().isRunning && !hasStarted() && phase !== 'complete') {
      setHasStarted(true);
      runDocking();
    }
  });

  const handleBack = () => {
    if (!state().isRunning) {
      setCurrentPhase('idle');
      setError(null);
      clearLogs();
      setDockCompletedLigands(0);
      setDockResults([]);
      setDockStep('dock-configure');
    }
  };

  const totalLigandsDisplay = () => state().dock.totalLigands || state().dock.ligandSdfPaths.length;

  const dockProgress = () => {
    const total = totalLigandsDisplay();
    const completed = state().dock.completedLigands;
    if (total <= 0) return 0;
    return Math.min(100, (completed / total) * 100);
  };

  return (
    <div class="h-full flex flex-col">
      {/* Title + Status */}
      <div class="flex items-center justify-between mb-3">
        <div>
          <h2 class="text-xl font-bold">
            {state().currentPhase === 'complete'
              ? 'Docking Complete'
              : cordialRunning()
              ? 'Rescoring with CORDIAL...'
              : 'Running Docking'}
          </h2>
          <p class="text-sm text-base-content/90">
            {state().currentPhase === 'complete'
              ? `${state().dock.results.length} poses generated`
              : cordialRunning()
              ? 'Neural network rescoring in progress'
              : totalLigandsDisplay() > 0
              ? `Docking ${state().dock.completedLigands} / ${totalLigandsDisplay()} ligands`
              : 'Initializing...'}
          </p>
        </div>
        <div class="flex items-center gap-2">
          <Show when={state().isRunning}>
            <button
              class="btn btn-circle btn-xs btn-ghost text-error"
              title="Cancel"
              onClick={async () => {
                await api.cancelVinaDocking();
                setIsRunning(false);
                setCurrentPhase('idle');
                appendLog('\n--- Docking cancelled by user ---\n');
              }}
            >
              <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M6 6h12v12H6z" />
              </svg>
            </button>
            <span class="loading loading-spinner loading-sm text-primary" />
          </Show>
          <Show when={state().currentPhase === 'complete'}>
            <span class="badge badge-success badge-sm">Done</span>
          </Show>
          <Show when={state().currentPhase === 'error'}>
            <span class="badge badge-error badge-sm">Error</span>
          </Show>
        </div>
      </div>

      {/* Progress bar */}
      <div class="mb-3 p-3 bg-base-200 rounded-lg">
        <div class="flex justify-between text-xs text-base-content/80 mb-1">
          <span>
            {state().dock.completedLigands} / {totalLigandsDisplay()} ligands
          </span>
          <span>{dockProgress().toFixed(0)}%</span>
        </div>
        <progress
          class={`progress w-full h-2 ${
            state().currentPhase === 'error'
              ? 'progress-error'
              : state().currentPhase === 'complete'
              ? 'progress-success'
              : 'progress-primary'
          }`}
          value={dockProgress()}
          max="100"
         />
        {/* CORDIAL secondary indicator */}
        <Show when={cordialRunning()}>
          <div class="mt-2 flex items-center gap-2">
            <span class="loading loading-dots loading-xs text-secondary" />
            <span class="text-xs text-secondary">Rescoring with CORDIAL...</span>
          </div>
        </Show>
      </div>

      {/* Error message */}
      <Show when={state().errorMessage}>
        <div class="alert alert-error py-2 mb-2">
          <svg class="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span class="text-sm">{state().errorMessage}</span>
        </div>
      </Show>

      <TerminalOutput title="Vina Docking Output" logs={state().logs} />

      {/* Navigation */}
      <div class="flex justify-between mt-3">
        <button class="btn btn-ghost btn-sm" onClick={handleBack} disabled={state().isRunning}>
          <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 17l-5-5m0 0l5-5m-5 5h12" />
          </svg>
          Back
        </button>

        <Show when={state().currentPhase === 'complete'}>
          <button class="btn btn-primary" onClick={() => setDockStep('dock-results')}>
            View Results
            <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </button>
        </Show>

        <Show when={state().currentPhase === 'error'}>
          <button
            class="btn btn-error btn-sm"
            onClick={() => {
              setCurrentPhase('idle');
              clearLogs();
              setError(null);
              setHasStarted(false);
              runDocking();
            }}
          >
            Retry
          </button>
        </Show>
      </div>
    </div>
  );
};

export default DockStepProgress;
