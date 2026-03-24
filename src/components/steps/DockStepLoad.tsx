// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, Show, createMemo, createSignal, For } from 'solid-js';
import { workflowStore } from '../../stores/workflow';
import { DockMolecule, LigandSource } from '../../../shared/types/dock';
import { projectPaths, DockingPaths } from '../../utils/projectPaths';
import { buildDockFolderName } from '../../utils/jobName';
import ImportInputPanel from '../shared/ImportInputPanel';
import path from 'path';

const DockStepLoad: Component = () => {
  const {
    state,
    setDockStep,
    setDockReceptorPdbPath,
    setDockReceptorPrepared,
    setDockReferenceLigandId,
    setDockReferenceLigandPath,
    setDockDetectedLigands,
    setDockLigandSource,
    setDockLigandSdfPaths,
    setDockLigandMolecules,
    setError,
  } = workflowStore;
  const api = window.electronAPI;

  const [isLoading, setIsLoading] = createSignal(false);
  const [isLoadingLigands, setIsLoadingLigands] = createSignal(false);
  const [statusText, setStatusText] = createSignal<string | null>(null);
  const [pdbIdText, setPdbIdText] = createSignal('');
  const [receptorThumbnail, setReceptorThumbnail] = createSignal<string | null>(null);
  const [structureFilePaths, setStructureFilePaths] = createSignal<string[]>([]);
  const [csvFilePath, setCsvFilePath] = createSignal<string | null>(null);
  const [smilesText, setSmilesText] = createSignal('');

  const detectedSmiles = createMemo(() => {
    const lines = smilesText().split('\n').map(l => l.trim()).filter(l => l.length > 0);
    return lines;
  });

  const dock = () => state().dock;
  const receptorPdbPath = () => dock().receptorPdbPath;
  const receptorPrepared = () => dock().receptorPrepared;
  const referenceLigandId = () => dock().referenceLigandId;
  const referenceLigandPath = () => dock().referenceLigandPath;
  const detectedLigands = () => dock().detectedLigands;
  const ligandSource = () => dock().ligandSource;
  const ligandMolecules = () => dock().ligandMolecules;

  const getDockPaths = (paths: ReturnType<typeof projectPaths>): DockingPaths => {
    const dockFolder = buildDockFolderName({ referenceLigandId: referenceLigandId() });
    return paths.docking(dockFolder);
  };

  // Fetch structure from RCSB PDB by ID
  const handleFetchPdb = async () => {
    const id = pdbIdText().trim();
    if (!id) return;
    const projectDir = state().projectDir;
    if (!projectDir) { setError('No project selected'); return; }
    setIsLoading(true);
    setStatusText(`Fetching ${id.toUpperCase()} from RCSB...`);
    setError(null);
    try {
      const result = await api.fetchPdb(id, projectDir);
      if (result.ok) {
        setPdbIdText('');
        // Load the fetched CIF as receptor
        setDockReceptorPdbPath(result.value);
        setDockDetectedLigands([]);
        setDockReferenceLigandId(null);
        setDockReferenceLigandPath(null);
        setDockReceptorPrepared(null);
        setStatusText('Detecting ligands...');
        const detectResult = await api.detectPdbLigands(result.value);
        if (detectResult?.ok && detectResult.value?.ligands) {
          setDockDetectedLigands(detectResult.value.ligands);
          setStatusText(null);
        } else {
          setStatusText('No ligands detected');
        }
      } else {
        setError(result.error?.message || 'Failed to fetch PDB');
      }
    } catch (err: any) {
      setError(`PDB fetch error: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLoadReceptor = async () => {
    const filePath = await api.selectPdbFile();
    if (!filePath) return;

    setIsLoading(true);
    setDockReceptorPdbPath(filePath);
    setDockDetectedLigands([]);
    setDockReferenceLigandId(null);
    setDockReferenceLigandPath(null);
    setDockReceptorPrepared(null);
    setReceptorThumbnail(null);
    setStatusText('Detecting ligands...');
    setError(null);

    const result = await api.detectPdbLigands(filePath);
    setIsLoading(false);

    if (result.ok) {
      const { ligands } = result.value;
      setDockDetectedLigands(ligands);
      setStatusText(null);
      if (ligands.length === 1) {
        handleSelectReferenceLigand(ligands[0].id, filePath);
      } else if (ligands.length === 0) {
        setStatusText('No ligands detected. A bound ligand is required to define the docking box.');
      }
    } else {
      setError(result.error?.message || 'Failed to detect ligands');
      setStatusText(null);
    }
  };

  const handleSelectReferenceLigand = async (ligandId: string, pdbPathOverride?: string) => {
    const currentPdb = pdbPathOverride || receptorPdbPath();
    if (!currentPdb) return;

    setDockReferenceLigandId(ligandId);
    setIsLoading(true);
    setStatusText('Extracting reference ligand...');

    const defaultDir = await api.getDefaultOutputDir();
    const baseOutputDir = state().customOutputDir || defaultDir;
    const paths = projectPaths(baseOutputDir, state().jobName);
    const dockPaths = paths.docking(buildDockFolderName({ referenceLigandId: ligandId }));

    const extractResult = await api.extractXrayLigand(currentPdb, ligandId, dockPaths.prep);

    if (extractResult.ok) {
      const data = extractResult.value;
      setDockReferenceLigandPath(data.sdfPath);
      setReceptorThumbnail(data.thumbnail);

      setStatusText('Preparing receptor (adding hydrogens)...');
      const receptorPath = path.join(dockPaths.inputs, 'receptor.pdb');
      const receptorPh = (state().dock.protonationConfig.phMin + state().dock.protonationConfig.phMax) / 2;
      const wrc = state().dock.waterRetentionConfig;
      const receptorResult = await api.prepareReceptor(
        currentPdb,
        ligandId,
        receptorPath,
        wrc.enabled ? wrc.distance : 0,
        receptorPh
      );

      setIsLoading(false);
      setStatusText(null);

      if (receptorResult.ok) {
        setDockReceptorPrepared(receptorResult.value);
      } else {
        setError(`Receptor preparation failed: ${receptorResult.error?.message || 'Unknown error'}`);
      }
    } else {
      setIsLoading(false);
      setStatusText(null);
      setError(extractResult.error?.message || 'Reference ligand extraction failed');
    }
  };

  const handleClearReceptor = () => {
    setDockReceptorPdbPath(null);
    setDockReceptorPrepared(null);
    setDockReferenceLigandId(null);
    setDockReferenceLigandPath(null);
    setDockDetectedLigands([]);
    setReceptorThumbnail(null);
    setStatusText(null);
    setError(null);
  };

  const resetLigandInput = () => {
    setDockLigandSdfPaths([]);
    setDockLigandMolecules([]);
    setStructureFilePaths([]);
    setCsvFilePath(null);
    setSmilesText('');
    setError(null);
  };

  const handleConvertSmiles = async () => {
    const smiles = detectedSmiles();
    if (smiles.length === 0) return;

    setIsLoadingLigands(true);
    setError(null);

    const defaultDir = await api.getDefaultOutputDir();
    const baseOutputDir = state().customOutputDir || defaultDir;
    const paths = projectPaths(baseOutputDir, state().jobName);
    const dockPaths = getDockPaths(paths);

    const result = await api.convertSmilesList(smiles, dockPaths.inputsLigands);
    setIsLoadingLigands(false);

    if (result.ok) {
      setDockLigandMolecules(result.value);
      setDockLigandSdfPaths(result.value.map((m: DockMolecule) => m.sdfPath));
    } else {
      setError(result.error?.message || 'Failed to convert SMILES');
    }
  };

  const handleSourceChange = (source: LigandSource) => {
    setDockLigandSource(source);
    resetLigandInput();
  };

  const handleSelectStructureFiles = async () => {
    const filePaths = await api.selectMoleculeFilesMulti();
    if (!filePaths || filePaths.length === 0) return;

    setIsLoadingLigands(true);
    setError(null);

    const defaultDir = await api.getDefaultOutputDir();
    const baseOutputDir = state().customOutputDir || defaultDir;
    const paths = projectPaths(baseOutputDir, state().jobName);
    const dockPaths = getDockPaths(paths);

    const result = await api.importMoleculeFiles(filePaths, dockPaths.inputsLigands);
    setIsLoadingLigands(false);

    if (result.ok) {
      setStructureFilePaths(filePaths);
      setDockLigandMolecules(result.value);
      setDockLigandSdfPaths(result.value.map((m: DockMolecule) => m.sdfPath));
    } else {
      setError(result.error?.message || 'Failed to import molecule files');
    }
  };

  const handleSelectCsvFile = async () => {
    const filePath = await api.selectCsvFile();
    if (!filePath) return;

    setIsLoadingLigands(true);
    setCsvFilePath(filePath);
    setError(null);

    const defaultDir = await api.getDefaultOutputDir();
    const baseOutputDir = state().customOutputDir || defaultDir;
    const paths = projectPaths(baseOutputDir, state().jobName);
    const dockPaths = getDockPaths(paths);

    const result = await api.parseSmilesCsv(filePath, dockPaths.inputsLigands);
    setIsLoadingLigands(false);

    if (result.ok) {
      setDockLigandMolecules(result.value);
      setDockLigandSdfPaths(result.value.map((m: DockMolecule) => m.sdfPath));
    } else {
      setError(result.error?.message || 'Failed to parse CSV');
    }
  };

  const canContinue = createMemo(() => (
    receptorPrepared() !== null &&
    referenceLigandPath() !== null &&
    ligandMolecules().length > 0
  ));

  const importedStructureLabel = createMemo(() => {
    const files = structureFilePaths();
    if (files.length === 0) return '';
    if (files.length === 1) return path.basename(files[0]);
    return `${files.length} files selected`;
  });

  const handleContinue = () => {
    if (canContinue()) {
      setDockStep('dock-configure');
    }
  };

  return (
    <div class="h-full flex flex-col">
      <div class="text-center mb-3">
        <h2 class="text-xl font-bold">Load Molecules for Docking</h2>
        <p class="text-sm text-base-content/90">Load a receptor and choose ligands to dock</p>
      </div>

      <div class="flex-1 flex gap-4 overflow-auto">
        <div class="flex-1 flex flex-col">
          <div class="card bg-base-200 shadow-lg flex-1">
            <div class="card-body p-5">
              <h3 class="text-sm font-semibold mb-3">Receptor</h3>

              <Show
                when={receptorPrepared()}
                fallback={
                  <div class="text-center">
                    <div class="w-full">
                      <ImportInputPanel
                        importButtonLabel="Import (.pdb, .cif)"
                        onImport={handleLoadReceptor}
                        importDisabled={isLoading()}
                        importLoading={isLoading()}
                        showPdbFetch={true}
                        pdbIdValue={pdbIdText()}
                        onPdbIdInput={setPdbIdText}
                        onFetchPdb={handleFetchPdb}
                        fetchDisabled={isLoading() || pdbIdText().trim().length !== 4}
                        fetchLoading={isLoading()}
                        statusText={!detectedLigands().length || statusText() ? statusText() : null}
                        beforeInputs={
                          <Show when={detectedLigands().length > 1}>
                            <select
                              class="select select-bordered select-xs w-full"
                              onChange={(e) => handleSelectReferenceLigand(e.currentTarget.value)}
                              value={referenceLigandId() || ''}
                            >
                              <option value="" disabled>Select reference ligand...</option>
                              <For each={detectedLigands()}>
                                {(lig) => (
                                  <option value={lig.id}>{lig.resname} ({lig.chain}:{lig.resnum}) - {lig.num_atoms} atoms</option>
                                )}
                              </For>
                            </select>
                          </Show>
                        }
                        afterInputs={
                          <p class="text-[10px] text-base-content/70">
                            Accepted receptor formats: `.pdb`, `.cif`. Use a bound complex; the bound ligand defines the docking box.
                          </p>
                        }
                      />
                    </div>
                  </div>
                }
              >
                <div class="flex-1 flex flex-col items-center">
                  <Show when={receptorThumbnail()}>
                    <img
                      src={`data:image/png;base64,${receptorThumbnail()}`}
                      alt="Reference ligand"
                      class="rounded bg-base-100 p-1 mb-3"
                      style={{ 'max-width': '100%', 'max-height': '200px', 'object-fit': 'contain' }}
                    />
                  </Show>

                  <div class="w-full space-y-2 text-xs">
                    <div class="flex justify-between py-1 border-b border-base-300">
                      <span class="text-base-content/70">Structure</span>
                      <span class="font-mono truncate max-w-[180px]">{receptorPdbPath() ? path.basename(receptorPdbPath()!) : ''}</span>
                    </div>
                    <div class="flex justify-between py-1 border-b border-base-300">
                      <span class="text-base-content/70">Reference Ligand</span>
                      <span class="font-mono">{referenceLigandId()}</span>
                    </div>
                    <div class="flex justify-between py-1">
                      <span class="text-base-content/70">Receptor</span>
                      <span class="badge badge-success badge-xs">Prepared</span>
                    </div>
                  </div>

                  <button class="btn btn-ghost btn-xs mt-3" onClick={handleClearReceptor}>
                    Clear
                  </button>
                </div>
              </Show>
            </div>
          </div>
        </div>

        <div class="flex-1 flex flex-col">
          <div class="card bg-base-200 shadow-lg flex-1">
            <div class="card-body p-5">
              <div class="flex items-center justify-between mb-3">
                <h3 class="text-sm font-semibold">Ligands to Dock</h3>
                <Show when={ligandMolecules().length > 0}>
                  <span class="badge badge-primary badge-sm">{ligandMolecules().length} ligand{ligandMolecules().length !== 1 ? 's' : ''}</span>
                </Show>
              </div>

              <Show
                when={ligandMolecules().length > 0}
                fallback={
                  <div class="flex-1 flex flex-col w-full gap-3">
                    <button
                      class="btn btn-outline btn-sm w-full"
                      onClick={handleSelectStructureFiles}
                      disabled={isLoadingLigands()}
                    >
                      {isLoadingLigands() ? <span class="loading loading-spinner loading-xs" /> : 'Import (.sdf, .mol, .csv)'}
                    </button>

                    <div>
                      <div class="flex items-center justify-between mb-1">
                        <span class="text-[10px] text-base-content/50">or enter SMILES</span>
                        <Show when={detectedSmiles().length > 0}>
                          <span class="text-[10px] font-mono text-success">
                            {detectedSmiles().length} input{detectedSmiles().length !== 1 ? 's' : ''}
                          </span>
                        </Show>
                      </div>
                      <textarea
                        class="textarea textarea-bordered text-xs font-mono w-full resize-none leading-relaxed"
                        placeholder="Enter SMILES strings (one compound per line)"
                        value={smilesText()}
                        onInput={(e) => setSmilesText(e.currentTarget.value)}
                        rows={5}
                      />
                    </div>

                    <button
                      class="btn btn-primary btn-sm w-full"
                      onClick={handleConvertSmiles}
                      disabled={isLoadingLigands() || detectedSmiles().length === 0}
                    >
                      {isLoadingLigands() ? <span class="loading loading-spinner loading-xs" /> : 'Enter SMILES'}
                    </button>
                  </div>
                }
              >
                <div class="w-full space-y-2">
                  <div class="flex items-center gap-2 px-3 py-1.5 bg-base-300 rounded-lg">
                    <span class="text-xs flex-1">{ligandMolecules().length} ligand{ligandMolecules().length !== 1 ? 's' : ''} loaded</span>
                    <span class="badge badge-success badge-xs">ready</span>
                  </div>
                  <div class="bg-base-300/70 rounded-lg px-3 py-2 text-[10px] font-mono space-y-1 max-h-28 overflow-auto">
                    <For each={ligandMolecules().slice(0, 6)}>
                      {(mol) => <div class="truncate">{mol.filename}</div>}
                    </For>
                    <Show when={ligandMolecules().length > 6}>
                      <div class="text-base-content/70">+ {ligandMolecules().length - 6} more</div>
                    </Show>
                  </div>
                  <button class="btn btn-ghost btn-xs w-full" onClick={resetLigandInput}>
                    Clear
                  </button>
                </div>
              </Show>
            </div>
          </div>
        </div>
      </div>

      <Show when={state().errorMessage}>
        <div class="alert alert-error mt-3">
          <svg class="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span class="text-sm">{state().errorMessage}</span>
        </div>
      </Show>

      <div class="mt-4 flex items-center gap-3">
        <div class="flex-1 flex items-center gap-2 text-xs text-base-content/85 bg-base-200 rounded-lg px-3 py-2">
          <svg class="w-4 h-4 text-info flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>
            The docking box is centered on the reference ligand with autobox padding. Ligands are docked with AutoDock Vina.
          </span>
        </div>
        <button class="btn btn-primary" disabled={!canContinue()} onClick={handleContinue}>
          Continue
          <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
          </svg>
        </button>
      </div>
    </div>
  );
};

export default DockStepLoad;
