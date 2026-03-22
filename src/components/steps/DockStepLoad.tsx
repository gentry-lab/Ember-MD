import { Component, Show, createMemo, createSignal, For } from 'solid-js';
import { workflowStore } from '../../stores/workflow';
import { DEFAULT_RECEPTOR_WATER_DISTANCE, DockMolecule, LigandSource } from '../../../shared/types/dock';
import { projectPaths, DockingPaths } from '../../utils/projectPaths';
import { buildDockFolderName } from '../../utils/jobName';
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
      const receptorResult = await api.prepareReceptor(
        currentPdb,
        ligandId,
        receptorPath,
        DEFAULT_RECEPTOR_WATER_DISTANCE,
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
                  <div class="flex-1 flex flex-col items-center justify-center text-center">
                    <div class="w-14 h-14 rounded-full bg-primary/10 flex items-center justify-center mb-3">
                      <svg class="w-7 h-7 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                      </svg>
                    </div>

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
                        {isLoading() ? <span class="loading loading-spinner loading-xs" /> : 'Select Structure'}
                      </button>
                    </Show>

                    <p class="text-[10px] text-base-content/70 mt-3">
                      Accepted receptor formats: `.pdb`, `.cif`. Use a bound complex; the bound ligand defines the docking box.
                    </p>
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

              <div class="tabs tabs-boxed bg-base-300 mb-3 w-full">
                <button
                  class={`tab tab-xs flex-1 ${ligandSource() === 'smiles_input' ? 'tab-active' : ''}`}
                  onClick={() => handleSourceChange('smiles_input')}
                >
                  SMILES
                </button>
                <button
                  class={`tab tab-xs flex-1 ${ligandSource() === 'structure_files' ? 'tab-active' : ''}`}
                  onClick={() => handleSourceChange('structure_files')}
                >
                  Files
                </button>
                <button
                  class={`tab tab-xs flex-1 ${ligandSource() === 'molecule_csv' ? 'tab-active' : ''}`}
                  onClick={() => handleSourceChange('molecule_csv')}
                >
                  CSV
                </button>
              </div>

              <Show when={ligandSource() === 'smiles_input'}>
                <div class="flex-1 flex flex-col">
                  <Show
                    when={ligandMolecules().length > 0}
                    fallback={
                      <div class="flex-1 flex flex-col w-full">
                        <div class="flex items-center justify-between mb-1">
                          <span class="text-[10px] text-base-content/70">One SMILES per line</span>
                          <span class={`text-[10px] font-mono ${detectedSmiles().length > 0 ? 'text-success' : 'text-base-content/50'}`}>
                            {detectedSmiles().length} molecule{detectedSmiles().length !== 1 ? 's' : ''} detected
                          </span>
                        </div>
                        <textarea
                          class="textarea textarea-bordered text-xs font-mono flex-1 w-full resize-none leading-relaxed"
                          placeholder="Enter SMILES strings (one compound per line)"
                          value={smilesText()}
                          onInput={(e) => setSmilesText(e.currentTarget.value)}
                          rows={6}
                        />
                        <button
                          class="btn btn-primary btn-sm w-full mt-2"
                          onClick={handleConvertSmiles}
                          disabled={isLoadingLigands() || detectedSmiles().length === 0}
                        >
                          {isLoadingLigands() ? <span class="loading loading-spinner loading-xs" /> : `Convert ${detectedSmiles().length} molecule${detectedSmiles().length !== 1 ? 's' : ''}`}
                        </button>
                      </div>
                    }
                  >
                    <div class="w-full space-y-2">
                      <div class="flex items-center gap-2 px-3 py-1.5 bg-base-300 rounded-lg">
                        <span class="text-xs flex-1">SMILES input</span>
                        <span class="badge badge-success badge-xs">{ligandMolecules().length} ready</span>
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
              </Show>

              <Show when={ligandSource() === 'structure_files'}>
                <div class="flex-1 flex flex-col items-center justify-center">
                  <Show
                    when={ligandMolecules().length > 0}
                    fallback={
                      <div class="text-center w-full">
                        <button
                          class="btn btn-outline btn-sm w-full mb-2"
                          onClick={handleSelectStructureFiles}
                          disabled={isLoadingLigands()}
                        >
                          {isLoadingLigands() ? <span class="loading loading-spinner loading-xs" /> : 'Select Structure'}
                        </button>
                        <p class="text-[10px] text-base-content/70">
                          Accepted ligand formats: `.sdf`, `.sdf.gz`, `.mol`, `.mol2`. Pick one file or many files at once.
                        </p>
                      </div>
                    }
                  >
                    <div class="w-full space-y-2">
                      <div class="flex items-center gap-2 px-3 py-1.5 bg-base-300 rounded-lg">
                        <svg class="w-4 h-4 text-base-content/70 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                        </svg>
                        <span class="text-xs truncate flex-1">{importedStructureLabel()}</span>
                        <span class="badge badge-success badge-xs">{ligandMolecules().length} ready</span>
                      </div>
                      <div class="bg-base-300/70 rounded-lg px-3 py-2 text-[10px] font-mono space-y-1 max-h-28 overflow-auto">
                        <For each={structureFilePaths().slice(0, 6)}>
                          {(filePath) => <div class="truncate">{path.basename(filePath)}</div>}
                        </For>
                        <Show when={structureFilePaths().length > 6}>
                          <div class="text-base-content/70">+ {structureFilePaths().length - 6} more</div>
                        </Show>
                      </div>
                      <button class="btn btn-ghost btn-xs w-full" onClick={resetLigandInput}>
                        Clear
                      </button>
                    </div>
                  </Show>
                </div>
              </Show>

              <Show when={ligandSource() === 'molecule_csv'}>
                <div class="flex-1 flex flex-col items-center justify-center">
                  <Show
                    when={ligandMolecules().length > 0}
                    fallback={
                      <div class="text-center w-full">
                        <button
                          class="btn btn-outline btn-sm w-full mb-2"
                          onClick={handleSelectCsvFile}
                          disabled={isLoadingLigands()}
                        >
                          {isLoadingLigands() ? <span class="loading loading-spinner loading-xs" /> : 'Select CSV'}
                        </button>
                        <p class="text-[10px] text-base-content/70">
                          Accepted CSV columns: `smiles` or `structure_file` / `mol_file` / `path` / `file`, plus optional `name`.
                        </p>
                      </div>
                    }
                  >
                    <div class="w-full space-y-2">
                      <div class="flex items-center gap-2 px-3 py-1.5 bg-base-300 rounded-lg">
                        <svg class="w-4 h-4 text-base-content/70 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                        </svg>
                        <span class="text-xs truncate flex-1">{csvFilePath() ? path.basename(csvFilePath()!) : ''}</span>
                        <span class="badge badge-success badge-xs">{ligandMolecules().length} ready</span>
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
