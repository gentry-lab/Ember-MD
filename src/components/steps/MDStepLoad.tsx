import { Component, Show, createMemo, createSignal, For } from 'solid-js';
import { workflowStore, MDInputMode } from '../../stores/workflow';
import { projectPaths } from '../../utils/projectPaths';
import path from 'path';

const MDStepLoad: Component = () => {
  const {
    state,
    setMdInputMode,
    setMdReceptorPdb,
    setMdLigandSdf,
    setMdLigandName,
    setMdPdbPath,
    setMdThumbnailDataUrl,
    setMdSingleMoleculeInput,
    setMdSingleMoleculeThumbnail,
    setMdStep,
    setError,
  } = workflowStore;
  const api = window.electronAPI;

  const [isLoading, setIsLoading] = createSignal(false);
  const [smilesInput, setSmilesInput] = createSignal('');

  // Accessors for store-persisted state (survive mode switches)
  const pdbPath = () => state().md.pdbPath;
  const setPdbPath = (v: string | null) => setMdPdbPath(v);
  const thumbnailDataUrl = () => state().md.thumbnailDataUrl;
  const setThumbnailDataUrl = (v: string | null) => setMdThumbnailDataUrl(v);
  const [detectedLigands, setDetectedLigands] = createSignal<Array<{ id: string; resname: string; chain: string; resnum: string; num_atoms: number }>>([]);
  const [selectedLigand, setSelectedLigand] = createSignal<string | null>(null);
  const [needsSmiles, setNeedsSmiles] = createSignal(false);
  const [smilesCorrection, setSmilesCorrection] = createSignal('');
  const [statusText, setStatusText] = createSignal<string | null>(null);

  // PDB complex handlers
  const handleLoadPdb = async () => {
    const filePath = await api.selectPdbFile();
    if (!filePath) return;

    setIsLoading(true);
    setPdbPath(filePath);
    setDetectedLigands([]);
    setSelectedLigand(null);
    setNeedsSmiles(false);
    setStatusText('Detecting ligands...');
    setError(null);

    const result = await api.detectPdbLigands(filePath);
    setIsLoading(false);

    if (result.ok) {
      setDetectedLigands(result.value);
      setStatusText(null);
      if (result.value.length === 1) {
        handleSelectLigand(result.value[0].id, filePath);
      } else if (result.value.length === 0) {
        setStatusText('No ligands detected. This PDB may be protein-only.');
      }
    } else {
      setError(result.error?.message || 'Failed to detect ligands');
      setStatusText(null);
    }
  };

  const handleSelectLigand = async (ligandId: string, pdbPathOverride?: string) => {
    const currentPdb = pdbPathOverride || pdbPath();
    if (!currentPdb) return;

    setSelectedLigand(ligandId);
    setIsLoading(true);
    setStatusText('Extracting ligand & preparing receptor...');
    setNeedsSmiles(false);

    const defaultDir = await api.getDefaultOutputDir();
    const baseOutputDir = state().customOutputDir || defaultDir;
    const paths = projectPaths(baseOutputDir, state().jobName);

    const extractResult = await api.extractXrayLigand(
      currentPdb, ligandId, paths.ligands.sdf, smilesCorrection() || undefined
    );

    if (extractResult.ok) {
      const data = extractResult.value;

      if (data.needsSmiles) {
        setNeedsSmiles(true);
        setStatusText('Could not determine bond orders. Please provide SMILES.');
        setIsLoading(false);
        return;
      }

      setStatusText('Preparing receptor (adding hydrogens)...');
      const receptorPath = path.join(paths.prepared, `${state().jobName}_receptor_${ligandId}.pdb`);
      const receptorResult = await api.prepareReceptor(currentPdb, ligandId, receptorPath);

      setIsLoading(false);
      setStatusText(null);

      if (receptorResult.ok) {
        setMdReceptorPdb(receptorResult.value);
        setMdLigandSdf(data.sdfPath);
        setMdLigandName(data.name);
        setThumbnailDataUrl(`data:image/png;base64,${data.thumbnail}`);
        console.log('[PDB] Ligand extracted via:', data.method, 'SMILES:', data.smiles);
      } else {
        setError(`Receptor preparation failed: ${receptorResult.error?.message || 'Unknown error'}`);
      }
    } else {
      setIsLoading(false);
      setStatusText(null);
      setError(extractResult.error?.message || 'Ligand extraction failed');
    }
  };

  const handleSmilesCorrection = () => {
    const ligandId = selectedLigand();
    if (!ligandId || !smilesCorrection().trim()) return;
    handleSelectLigand(ligandId);
  };

  const handleClearPdb = () => {
    setPdbPath(null);
    setDetectedLigands([]);
    setSelectedLigand(null);
    setNeedsSmiles(false);
    setSmilesCorrection('');
    setStatusText(null);
    setMdReceptorPdb(null);
    setMdLigandSdf(null);
    setMdLigandName(null);
    setThumbnailDataUrl(null);
  };

  // Single molecule handlers (ligand-only mode)
  const handleSmilesSubmit = async () => {
    const smiles = smilesInput().trim();
    if (!smiles) return;

    setIsLoading(true);
    setError(null);

    const defaultDir = await api.getDefaultOutputDir();
    const baseOutputDir = state().customOutputDir || defaultDir;
    const paths = projectPaths(baseOutputDir, state().jobName);

    const result = await api.convertSingleMolecule(smiles, paths.ligands.sdf, 'smiles');
    setIsLoading(false);

    if (result.ok) {
      const mol = result.value;
      setMdSingleMoleculeInput(smiles);
      setMdSingleMoleculeThumbnail(mol.thumbnail);
      setMdLigandSdf(mol.sdfPath);
      setMdLigandName(mol.name);
      setMdReceptorPdb(null);
      setThumbnailDataUrl(`data:image/png;base64,${mol.thumbnail}`);
    } else {
      setError(result.error?.message || 'Failed to convert molecule');
    }
  };

  const handleSelectMolFile = async () => {
    const filePath = await api.selectSdfFile();
    if (!filePath) return;

    setIsLoading(true);
    setError(null);

    const defaultDir = await api.getDefaultOutputDir();
    const baseOutputDir = state().customOutputDir || defaultDir;
    const paths = projectPaths(baseOutputDir, state().jobName);

    const result = await api.convertSingleMolecule(filePath, paths.ligands.sdf, 'mol_file');
    setIsLoading(false);

    if (result.ok) {
      const mol = result.value;
      setMdSingleMoleculeInput(path.basename(filePath));
      setMdSingleMoleculeThumbnail(mol.thumbnail);
      setMdLigandSdf(mol.sdfPath);
      setMdLigandName(mol.name);
      setMdReceptorPdb(null);
      setThumbnailDataUrl(`data:image/png;base64,${mol.thumbnail}`);
    } else {
      setError(result.error?.message || 'Failed to load molecule file');
    }
  };

  const handleClearSingleMolecule = () => {
    setMdSingleMoleculeInput(null);
    setMdSingleMoleculeThumbnail(null);
    setMdLigandSdf(null);
    setMdLigandName(null);
    setThumbnailDataUrl(null);
    setSmilesInput('');
  };

  const [isProtonating, setIsProtonating] = createSignal(false);

  const handleProtonateAtPh74 = async () => {
    if (!state().md.ligandSdf) return;

    setIsProtonating(true);
    setError(null);

    try {
      const defaultDir = await api.getDefaultOutputDir();
      const baseOutputDir = state().customOutputDir || defaultDir;
      const paths = projectPaths(baseOutputDir, state().jobName);
      const protonatedDir = path.join(paths.ligands.sdf, 'protonated');

      const result = await api.enumerateProtonation(
        [state().md.ligandSdf!],
        protonatedDir,
        7.2,
        7.6
      );

      if (result.ok && result.value.protonatedPaths?.length > 0) {
        const protonatedSdf = result.value.protonatedPaths[0];
        const convertResult = await api.convertSingleMolecule(protonatedSdf, protonatedDir, 'mol_file');

        if (convertResult.ok) {
          const mol = convertResult.value;
          setMdSingleMoleculeInput(mol.smiles);
          setMdSingleMoleculeThumbnail(mol.thumbnail);
          setMdLigandSdf(mol.sdfPath);
          setMdLigandName(mol.name);
          setThumbnailDataUrl(`data:image/png;base64,${mol.thumbnail}`);
        }
      } else {
        setError('Protonation unchanged (Dimorphite-DL may not be installed: pip install dimorphite_dl)');
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsProtonating(false);
    }
  };

  const handleModeChange = (mode: MDInputMode) => {
    setMdInputMode(mode);
    handleClearPdb();
    handleClearSingleMolecule();
    setError(null);
  };

  const handleContinue = () => {
    if (canContinue()) {
      setMdStep('md-configure');
    }
  };

  const canContinue = createMemo(() => {
    if (state().md.inputMode === 'ligand_only') {
      return state().md.ligandSdf !== null;
    }
    return (
      state().md.receptorPdb !== null &&
      state().md.ligandSdf !== null
    );
  });

  return (
    <div class="h-full flex flex-col">
      {/* Title */}
      <div class="text-center mb-3">
        <h2 class="text-xl font-bold">Load Molecules for MD</h2>
        <p class="text-sm text-base-content/90">
          {state().md.inputMode === 'ligand_only'
            ? 'Paste a SMILES or load a molecule file for ligand-only MD'
            : 'Load a PDB/CIF complex (X-ray or docked)'}
        </p>
      </div>

      {/* Input mode tabs */}
      <div class="tabs tabs-boxed bg-base-300 mb-3 w-fit mx-auto">
        <button
          class={`tab tab-sm ${state().md.inputMode === 'protein_ligand' ? 'tab-active' : ''}`}
          onClick={() => handleModeChange('protein_ligand')}
        >
          Protein + Ligand
        </button>
        <button
          class={`tab tab-sm ${state().md.inputMode === 'ligand_only' ? 'tab-active' : ''}`}
          onClick={() => handleModeChange('ligand_only')}
        >
          Ligand Only
        </button>
      </div>

      {/* ========== Ligand-only mode ========== */}
      <Show when={state().md.inputMode === 'ligand_only'}>
        <Show
          when={state().md.ligandSdf}
          fallback={
            <div class="flex-1 flex items-center justify-center">
              <div class="card bg-base-200 shadow-lg w-80">
                <div class="card-body p-5">
                  <h3 class="text-sm font-semibold mb-3">Single Molecule Input</h3>

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
                        onKeyDown={(e) => e.key === 'Enter' && handleSmilesSubmit()}
                      />
                      <button
                        class="btn btn-primary btn-sm"
                        onClick={handleSmilesSubmit}
                        disabled={!smilesInput().trim() || isLoading()}
                      >
                        {isLoading() ? <span class="loading loading-spinner loading-xs" /> : 'Go'}
                      </button>
                    </div>
                  </div>

                  <div class="divider my-1 text-[10px]">or</div>

                  {/* MOL/SDF file browser */}
                  <button class="btn btn-outline btn-sm w-full" onClick={handleSelectMolFile} disabled={isLoading()}>
                    <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                    </svg>
                    Select MOL / SDF File
                  </button>

                  <p class="text-[10px] text-base-content/60 mt-2">
                    The molecule will be solvated and simulated without a protein receptor.
                    3D coordinates are generated via ETKDG + MMFF minimization.
                  </p>
                </div>
              </div>
            </div>
          }
        >
          {/* Molecule loaded preview */}
          <div class="flex-1 flex items-center justify-center overflow-auto py-2">
            <div class="card bg-base-200 shadow-lg max-w-lg">
              <div class="card-body p-5 items-center">
                <Show when={thumbnailDataUrl()}>
                  <img
                    src={thumbnailDataUrl()!}
                    alt="Molecule structure"
                    class="rounded bg-base-100 p-2"
                    style={{ "max-width": "100%", "max-height": "280px", "object-fit": "contain" }}
                  />
                </Show>
                <p class="text-xs font-mono text-base-content/80 mt-2 truncate max-w-full">
                  {state().md.singleMoleculeInput}
                </p>
                <p class="text-xs text-base-content/60">
                  {state().md.ligandName}
                </p>
                <div class="flex gap-2 mt-2">
                  <button
                    class="btn btn-outline btn-xs"
                    onClick={handleProtonateAtPh74}
                    disabled={isProtonating()}
                  >
                    {isProtonating() ? (
                      <span class="loading loading-spinner loading-xs" />
                    ) : (
                      'Protonate pH 7.4'
                    )}
                  </button>
                  <button class="btn btn-ghost btn-xs" onClick={handleClearSingleMolecule}>
                    Clear
                  </button>
                </div>
                <p class="text-[9px] text-base-content/70 mt-1 text-center">
                  Molecule simulated as drawn. Click protonate to adjust for physiological pH.
                </p>
              </div>
            </div>
          </div>
        </Show>
      </Show>

      {/* ========== Protein + Ligand mode ========== */}
      <Show when={state().md.inputMode === 'protein_ligand'}>

        {/* Loaded PDB header bar */}
        <Show when={pdbPath() && (state().md.receptorPdb || detectedLigands().length > 0)}>
          <div class="flex items-center gap-3 mb-4">
            <div class="flex-1 flex items-center gap-2 px-3 py-1.5 bg-base-200 rounded-lg">
              <span class="badge badge-sm badge-primary">{pdbPath()!.toLowerCase().endsWith('.cif') ? 'CIF' : 'PDB'}</span>
              <span class="text-sm truncate flex-1">{path.basename(pdbPath()!)}</span>
              <Show when={state().md.ligandName}>
                <span class="badge badge-sm badge-ghost">{state().md.ligandName}</span>
              </Show>
              <button class="btn btn-ghost btn-xs text-error" onClick={handleClearPdb}>
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>
        </Show>

        <Show
          when={state().md.receptorPdb && state().md.ligandSdf}
          fallback={
            <div class="flex-1 flex items-center justify-center">
              <div class="card bg-base-200 shadow-lg w-80">
                <div class="card-body p-5 items-center text-center">
                  <div class="w-14 h-14 rounded-full bg-primary/10 flex items-center justify-center mb-2">
                    <svg class="w-7 h-7 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                    </svg>
                  </div>
                  <h3 class="text-sm font-semibold">Load Structure</h3>
                  <p class="text-[10px] text-base-content/60 mb-3">
                    X-ray structure (PDB or CIF), docked complex, or any structure with a bound ligand
                  </p>

                  {/* Ligand selection for multi-ligand PDBs */}
                  <Show when={detectedLigands().length > 1}>
                    <div class="w-full mb-2">
                      <select
                        class="select select-bordered select-xs w-full"
                        onChange={(e) => handleSelectLigand(e.currentTarget.value)}
                        value={selectedLigand() || ''}
                      >
                        <option value="" disabled>Select ligand...</option>
                        <For each={detectedLigands()}>
                          {(lig) => (
                            <option value={lig.id}>{lig.resname} ({lig.chain}:{lig.resnum}) - {lig.num_atoms} atoms</option>
                          )}
                        </For>
                      </select>
                    </div>
                  </Show>

                  {/* SMILES input for bond order help */}
                  <Show when={needsSmiles()}>
                    <div class="w-full mb-2">
                      <p class="text-[10px] text-warning mb-1">Provide SMILES for bond orders:</p>
                      <div class="flex gap-1">
                        <input
                          type="text"
                          class="input input-bordered input-xs flex-1 font-mono text-[10px]"
                          placeholder="e.g. c1ccc(cc1)O"
                          value={smilesCorrection()}
                          onInput={(e) => setSmilesCorrection(e.currentTarget.value)}
                          onKeyDown={(e) => e.key === 'Enter' && handleSmilesCorrection()}
                        />
                        <button class="btn btn-primary btn-xs" onClick={handleSmilesCorrection}>Go</button>
                      </div>
                    </div>
                  </Show>

                  <Show when={statusText()}>
                    <p class="text-[10px] text-base-content/60 mb-2">{statusText()}</p>
                  </Show>

                  <Show when={!detectedLigands().length || statusText()}>
                    <button class="btn btn-primary btn-sm w-full" onClick={handleLoadPdb} disabled={isLoading()}>
                      {isLoading() ? <span class="loading loading-spinner loading-xs" /> : 'Browse PDB / CIF'}
                    </button>
                  </Show>
                </div>
              </div>
            </div>
          }
        >
          {/* PDB loaded summary */}
          <div class="flex-1 flex items-center justify-center overflow-auto py-2">
            <div class="card bg-base-200 shadow-lg max-w-lg">
              <div class="card-body p-5 items-center">
                <h3 class="text-sm font-semibold mb-3 text-center">Complex Ready</h3>
                <Show when={thumbnailDataUrl()}>
                  <img
                    src={thumbnailDataUrl()!}
                    alt="Ligand structure"
                    class="rounded bg-base-100 p-1"
                    style={{ "max-width": "100%", "max-height": "280px", "object-fit": "contain" }}
                  />
                </Show>
                <div class="w-full space-y-2 text-xs mt-3">
                  <div class="flex justify-between py-1 border-b border-base-300">
                    <span class="text-base-content/70">PDB</span>
                    <span class="font-mono truncate max-w-[200px]">{path.basename(pdbPath() || state().md.receptorPdb || '')}</span>
                  </div>
                  <div class="flex justify-between py-1 border-b border-base-300">
                    <span class="text-base-content/70">Ligand</span>
                    <span class="font-mono">{state().md.ligandName}</span>
                  </div>
                  <Show when={state().md.receptorPdb}>
                    <div class="flex justify-between py-1 border-b border-base-300">
                      <span class="text-base-content/70">Receptor</span>
                      <span class="badge badge-success badge-xs">Prepared</span>
                    </div>
                  </Show>
                  <div class="flex justify-between py-1">
                    <span class="text-base-content/70">Ligand SDF</span>
                    <span class="badge badge-success badge-xs">Extracted</span>
                  </div>
                </div>
                <p class="text-[10px] text-base-content/70 mt-3 text-center">
                  Receptor protonated at pH 7.4.
                </p>
              </div>
            </div>
          </div>
        </Show>
      </Show>

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
            The system will be solvated in a rhombic dodecahedron box with 150 mM NaCl. Water model depends on force field preset (OPC for accurate, TIP3P for fast).
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

export default MDStepLoad;
