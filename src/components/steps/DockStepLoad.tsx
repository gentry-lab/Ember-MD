import { Component, Show, createMemo, createSignal, For } from 'solid-js';
import { workflowStore } from '../../stores/workflow';
import { DetectedLigand, DockMolecule, LigandSource } from '../../../shared/types/dock';
import { projectPaths } from '../../utils/projectPaths';
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
  const [receptorThumbnail, setReceptorThumbnail] = createSignal<string | null>(null);
  const [smilesInput, setSmilesInput] = createSignal('');
  const [sdfFolderPath, setSdfFolderPath] = createSignal<string | null>(null);
  const [csvFilePath, setCsvFilePath] = createSignal<string | null>(null);

  // Accessors for dock state
  const dock = () => state().dock;
  const receptorPdbPath = () => dock().receptorPdbPath;
  const receptorPrepared = () => dock().receptorPrepared;
  const referenceLigandId = () => dock().referenceLigandId;
  const referenceLigandPath = () => dock().referenceLigandPath;
  const detectedLigands = () => dock().detectedLigands;
  const ligandSource = () => dock().ligandSource;
  const ligandMolecules = () => dock().ligandMolecules;

  // ========== Receptor (Left Column) ==========

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
      setDockDetectedLigands(result.value);
      setStatusText(null);
      if (result.value.length === 1) {
        handleSelectReferenceLigand(result.value[0].id, filePath);
      } else if (result.value.length === 0) {
        setStatusText('No ligands detected. A reference ligand is required to define the docking box.');
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

    const extractResult = await api.extractXrayLigand(currentPdb, ligandId, paths.ligands.sdf);

    if (extractResult.ok) {
      const data = extractResult.value;
      setDockReferenceLigandPath(data.sdfPath);
      setReceptorThumbnail(data.thumbnail);

      setStatusText('Preparing receptor (adding hydrogens)...');
      const receptorPath = path.join(paths.prepared, `${state().jobName}_receptor_${ligandId}.pdb`);
      const receptorResult = await api.prepareReceptor(currentPdb, ligandId, receptorPath);

      setIsLoading(false);
      setStatusText(null);

      if (receptorResult.ok) {
        setDockReceptorPrepared(receptorResult.value);
        console.log('[Dock] Receptor prepared, reference ligand:', ligandId);
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

  // ========== Ligands (Right Column) ==========

  const handleSourceChange = (source: LigandSource) => {
    setDockLigandSource(source);
    setDockLigandSdfPaths([]);
    setDockLigandMolecules([]);
    setSdfFolderPath(null);
    setCsvFilePath(null);
    setSmilesInput('');
    setError(null);
  };

  // SDF Directory
  const handleSelectSdfFolder = async () => {
    const folderPath = await api.selectFolder();
    if (!folderPath) return;

    setIsLoadingLigands(true);
    setSdfFolderPath(folderPath);
    setError(null);

    const defaultDir = await api.getDefaultOutputDir();
    const baseOutputDir = state().customOutputDir || defaultDir;
    const paths = projectPaths(baseOutputDir, state().jobName);

    const result = await api.scanSdfDirectory(folderPath, paths.ligands.sdf);
    setIsLoadingLigands(false);

    if (result.ok) {
      setDockLigandMolecules(result.value);
      setDockLigandSdfPaths(result.value.map((m: DockMolecule) => m.sdfPath));
      console.log('[Dock] Loaded', result.value.length, 'SDFs from directory');
    } else {
      setError(result.error?.message || 'Failed to scan SDF directory');
    }
  };

  // SMILES CSV
  const handleSelectCsvFile = async () => {
    const filePath = await api.selectCsvFile();
    if (!filePath) return;

    setIsLoadingLigands(true);
    setCsvFilePath(filePath);
    setError(null);

    const defaultDir = await api.getDefaultOutputDir();
    const baseOutputDir = state().customOutputDir || defaultDir;
    const paths = projectPaths(baseOutputDir, state().jobName);

    const result = await api.parseSmilesCsv(filePath, paths.ligands.sdf);
    setIsLoadingLigands(false);

    if (result.ok) {
      setDockLigandMolecules(result.value);
      setDockLigandSdfPaths(result.value.map((m: DockMolecule) => m.sdfPath));
      console.log('[Dock] Parsed', result.value.length, 'molecules from CSV');
    } else {
      setError(result.error?.message || 'Failed to parse SMILES CSV');
    }
  };

  // Single Molecule — SMILES
  const handleSingleSmiles = async () => {
    const smiles = smilesInput().trim();
    if (!smiles) return;

    setIsLoadingLigands(true);
    setError(null);

    const defaultDir = await api.getDefaultOutputDir();
    const baseOutputDir = state().customOutputDir || defaultDir;
    const paths = projectPaths(baseOutputDir, state().jobName);

    const result = await api.convertSingleMolecule(smiles, paths.ligands.sdf, 'smiles');
    setIsLoadingLigands(false);

    if (result.ok) {
      const mol = result.value;
      const dockMol: DockMolecule = {
        filename: mol.name,
        smiles: mol.smiles,
        qed: mol.qed,
        sdfPath: mol.sdfPath,
      };
      setDockLigandMolecules([dockMol]);
      setDockLigandSdfPaths([mol.sdfPath]);
      console.log('[Dock] Single molecule from SMILES:', mol.name);
    } else {
      setError(result.error?.message || 'Failed to convert SMILES');
    }
  };

  // Single Molecule — SDF file
  const handleSingleSdfFile = async () => {
    const filePath = await api.selectSdfFile();
    if (!filePath) return;

    setIsLoadingLigands(true);
    setError(null);

    const defaultDir = await api.getDefaultOutputDir();
    const baseOutputDir = state().customOutputDir || defaultDir;
    const paths = projectPaths(baseOutputDir, state().jobName);

    const result = await api.convertSingleMolecule(filePath, paths.ligands.sdf, 'mol_file');
    setIsLoadingLigands(false);

    if (result.ok) {
      const mol = result.value;
      const dockMol: DockMolecule = {
        filename: path.basename(filePath),
        smiles: mol.smiles,
        qed: mol.qed,
        sdfPath: mol.sdfPath,
      };
      setDockLigandMolecules([dockMol]);
      setDockLigandSdfPaths([mol.sdfPath]);
      console.log('[Dock] Single molecule from file:', path.basename(filePath));
    } else {
      setError(result.error?.message || 'Failed to load SDF file');
    }
  };

  // ========== Navigation ==========

  const canContinue = createMemo(() => {
    return (
      receptorPrepared() !== null &&
      referenceLigandPath() !== null &&
      ligandMolecules().length > 0
    );
  });

  const handleContinue = () => {
    if (canContinue()) {
      setDockStep('dock-configure');
    }
  };

  return (
    <div class="h-full flex flex-col">
      {/* Title */}
      <div class="text-center mb-3">
        <h2 class="text-xl font-bold">Load Molecules for Docking</h2>
        <p class="text-sm text-base-content/90">
          Load a receptor structure and select ligands to dock
        </p>
      </div>

      {/* Two-column layout */}
      <div class="flex-1 flex gap-4 overflow-auto">

        {/* ========== Left Column: Receptor ========== */}
        <div class="flex-1 flex flex-col">
          <div class="card bg-base-200 shadow-lg flex-1">
            <div class="card-body p-5">
              <h3 class="text-sm font-semibold mb-3">Receptor</h3>

              <Show
                when={receptorPrepared()}
                fallback={
                  <div class="flex-1 flex flex-col items-center justify-center text-center">
                    <div class="w-14 h-14 rounded-full bg-primary/10 flex items-center justify-center mb-3">
                      <svg class="w-7 h-7 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                      </svg>
                    </div>

                    {/* Ligand selection dropdown for multi-ligand PDBs */}
                    <Show when={detectedLigands().length > 1}>
                      <div class="w-full mb-3">
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
                      </div>
                    </Show>

                    <Show when={statusText()}>
                      <p class="text-[10px] text-base-content/70 mb-2">{statusText()}</p>
                    </Show>

                    <Show when={!detectedLigands().length || statusText()}>
                      <button class="btn btn-primary btn-sm w-full" onClick={handleLoadReceptor} disabled={isLoading()}>
                        {isLoading() ? <span class="loading loading-spinner loading-xs"></span> : 'Browse PDB / CIF'}
                      </button>
                    </Show>

                    <p class="text-[10px] text-base-content/70 mt-3">
                      Load a protein structure with a bound ligand. The ligand defines the docking box center.
                    </p>
                  </div>
                }
              >
                {/* Receptor loaded summary */}
                <div class="flex-1 flex flex-col items-center">
                  <Show when={receptorThumbnail()}>
                    <img
                      src={`data:image/png;base64,${receptorThumbnail()}`}
                      alt="Reference ligand"
                      class="rounded bg-white p-1 mb-3"
                      style={{ "max-width": "100%", "max-height": "200px", "object-fit": "contain" }}
                    />
                  </Show>

                  <div class="w-full space-y-2 text-xs">
                    <div class="flex justify-between py-1 border-b border-base-300">
                      <span class="text-base-content/70">PDB</span>
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

        {/* ========== Right Column: Ligands to Dock ========== */}
        <div class="flex-1 flex flex-col">
          <div class="card bg-base-200 shadow-lg flex-1">
            <div class="card-body p-5">
              <div class="flex items-center justify-between mb-3">
                <h3 class="text-sm font-semibold">Ligands to Dock</h3>
                <Show when={ligandMolecules().length > 0}>
                  <span class="badge badge-primary badge-sm">{ligandMolecules().length} ligand{ligandMolecules().length !== 1 ? 's' : ''}</span>
                </Show>
              </div>

              {/* Source tabs */}
              <div class="tabs tabs-boxed bg-base-300 mb-3 w-full">
                <button
                  class={`tab tab-xs flex-1 ${ligandSource() === 'sdf_directory' ? 'tab-active' : ''}`}
                  onClick={() => handleSourceChange('sdf_directory')}
                >
                  SDF Folder
                </button>
                <button
                  class={`tab tab-xs flex-1 ${ligandSource() === 'smiles_csv' ? 'tab-active' : ''}`}
                  onClick={() => handleSourceChange('smiles_csv')}
                >
                  SMILES CSV
                </button>
                <button
                  class={`tab tab-xs flex-1 ${ligandSource() === 'single_molecule' ? 'tab-active' : ''}`}
                  onClick={() => handleSourceChange('single_molecule')}
                >
                  Single Molecule
                </button>
              </div>

              {/* ---- SDF Directory ---- */}
              <Show when={ligandSource() === 'sdf_directory'}>
                <div class="flex-1 flex flex-col items-center justify-center">
                  <Show
                    when={ligandMolecules().length > 0}
                    fallback={
                      <div class="text-center">
                        <button
                          class="btn btn-outline btn-sm w-full mb-2"
                          onClick={handleSelectSdfFolder}
                          disabled={isLoadingLigands()}
                        >
                          {isLoadingLigands() ? (
                            <span class="loading loading-spinner loading-xs"></span>
                          ) : (
                            <>
                              <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                              </svg>
                              Select SDF Folder
                            </>
                          )}
                        </button>
                        <p class="text-[10px] text-base-content/70">
                          Select a folder containing .sdf files, one molecule per file.
                        </p>
                      </div>
                    }
                  >
                    <div class="w-full">
                      <div class="flex items-center gap-2 px-3 py-1.5 bg-base-300 rounded-lg mb-2">
                        <svg class="w-4 h-4 text-base-content/70 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                        </svg>
                        <span class="text-xs truncate flex-1">{sdfFolderPath()}</span>
                        <span class="badge badge-success badge-xs">{ligandMolecules().length} files</span>
                      </div>
                      <button
                        class="btn btn-ghost btn-xs w-full"
                        onClick={() => { setDockLigandMolecules([]); setDockLigandSdfPaths([]); setSdfFolderPath(null); }}
                      >
                        Clear
                      </button>
                    </div>
                  </Show>
                </div>
              </Show>

              {/* ---- SMILES CSV ---- */}
              <Show when={ligandSource() === 'smiles_csv'}>
                <div class="flex-1 flex flex-col items-center justify-center">
                  <Show
                    when={ligandMolecules().length > 0}
                    fallback={
                      <div class="text-center">
                        <button
                          class="btn btn-outline btn-sm w-full mb-2"
                          onClick={handleSelectCsvFile}
                          disabled={isLoadingLigands()}
                        >
                          {isLoadingLigands() ? (
                            <span class="loading loading-spinner loading-xs"></span>
                          ) : (
                            <>
                              <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                              </svg>
                              Select SMILES CSV
                            </>
                          )}
                        </button>
                        <p class="text-[10px] text-base-content/70">
                          CSV with columns: smiles, name (optional). One molecule per row.
                        </p>
                      </div>
                    }
                  >
                    <div class="w-full">
                      <div class="flex items-center gap-2 px-3 py-1.5 bg-base-300 rounded-lg mb-2">
                        <svg class="w-4 h-4 text-base-content/70 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                        </svg>
                        <span class="text-xs truncate flex-1">{csvFilePath() ? path.basename(csvFilePath()!) : ''}</span>
                        <span class="badge badge-success badge-xs">{ligandMolecules().length} molecules</span>
                      </div>
                      <button
                        class="btn btn-ghost btn-xs w-full"
                        onClick={() => { setDockLigandMolecules([]); setDockLigandSdfPaths([]); setCsvFilePath(null); }}
                      >
                        Clear
                      </button>
                    </div>
                  </Show>
                </div>
              </Show>

              {/* ---- Single Molecule ---- */}
              <Show when={ligandSource() === 'single_molecule'}>
                <div class="flex-1 flex flex-col items-center justify-center">
                  <Show
                    when={ligandMolecules().length > 0}
                    fallback={
                      <div class="w-full">
                        {/* SMILES input */}
                        <div class="form-control mb-2">
                          <label class="label py-0.5">
                            <span class="label-text text-xs">Paste SMILES</span>
                          </label>
                          <div class="flex gap-1">
                            <input
                              type="text"
                              class="input input-bordered input-sm flex-1 font-mono text-xs"
                              placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O"
                              value={smilesInput()}
                              onInput={(e) => setSmilesInput(e.currentTarget.value)}
                              onKeyDown={(e) => e.key === 'Enter' && handleSingleSmiles()}
                            />
                            <button
                              class="btn btn-primary btn-sm"
                              onClick={handleSingleSmiles}
                              disabled={!smilesInput().trim() || isLoadingLigands()}
                            >
                              {isLoadingLigands() ? <span class="loading loading-spinner loading-xs"></span> : 'Go'}
                            </button>
                          </div>
                        </div>

                        <div class="divider my-1 text-[10px]">or</div>

                        {/* SDF file browser */}
                        <button
                          class="btn btn-outline btn-sm w-full"
                          onClick={handleSingleSdfFile}
                          disabled={isLoadingLigands()}
                        >
                          <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                          </svg>
                          Select SDF File
                        </button>
                      </div>
                    }
                  >
                    <div class="w-full">
                      <div class="flex items-center gap-2 px-3 py-1.5 bg-base-300 rounded-lg mb-2">
                        <span class="text-xs font-mono truncate flex-1">{ligandMolecules()[0]?.smiles || ligandMolecules()[0]?.filename}</span>
                        <span class="badge badge-success badge-xs">Ready</span>
                      </div>
                      <button
                        class="btn btn-ghost btn-xs w-full"
                        onClick={() => { setDockLigandMolecules([]); setDockLigandSdfPaths([]); setSmilesInput(''); }}
                      >
                        Clear
                      </button>
                    </div>
                  </Show>
                </div>
              </Show>
            </div>
          </div>
        </div>
      </div>

      {/* Error display */}
      <Show when={state().errorMessage}>
        <div class="alert alert-error mt-3">
          <svg class="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span class="text-sm">{state().errorMessage}</span>
        </div>
      </Show>

      {/* Info + Continue */}
      <div class="mt-4 flex items-center gap-3">
        <div class="flex-1 flex items-center gap-2 text-xs text-base-content/85 bg-base-200 rounded-lg px-3 py-2">
          <svg class="w-4 h-4 text-info flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>
            The docking box is centered on the reference ligand with an autobox padding. Ligands are docked with AutoDock Vina.
          </span>
        </div>
        <button
          class="btn btn-primary"
          disabled={!canContinue()}
          onClick={handleContinue}
        >
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
