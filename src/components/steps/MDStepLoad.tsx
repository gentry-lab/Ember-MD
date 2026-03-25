// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, Show, createMemo, createSignal, For } from 'solid-js';
import { workflowStore } from '../../stores/workflow';
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
  const [smilesText, setSmilesText] = createSignal('');
  const [pdbIdText, setPdbIdText] = createSignal('');
  const [statusText, setStatusText] = createSignal<string | null>(null);
  const [detectedLigands, setDetectedLigands] = createSignal<Array<{ id: string; resname: string; chain: string; resnum: string; num_atoms: number }>>([]);
  const [selectedLigand, setSelectedLigand] = createSignal<string | null>(null);
  const [needsSmiles, setNeedsSmiles] = createSignal(false);
  const [smilesCorrection, setSmilesCorrection] = createSignal('');

  const detectedSmiles = createMemo(() =>
    smilesText().split('\n').map(l => l.trim()).filter(l => l.length > 0)
  );

  const pdbPath = () => state().md.pdbPath;
  const thumbnailDataUrl = () => state().md.thumbnailDataUrl;
  const setThumbnailDataUrl = (v: string | null) => setMdThumbnailDataUrl(v);
  const isLoaded = () => state().md.ligandSdf !== null || state().md.inputMode === 'apo';

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
        await handleLoadComplex(result.value);
      } else {
        setError(result.error?.message || 'Failed to fetch PDB');
      }
    } catch (err: any) {
      setError(`PDB fetch error: ${err.message}`);
    } finally {
      setIsLoading(false);
      setStatusText(null);
    }
  };

  // Unified import: auto-detect PDB/CIF (complex) vs SDF/MOL (ligand-only)
  const handleImport = async () => {
    const filePath = await api.selectPdbFile(); // accepts PDB, CIF, SDF, MOL
    if (!filePath) return;

    const ext = filePath.toLowerCase().split('.').pop() || '';
    if (ext === 'pdb' || ext === 'cif') {
      await handleLoadComplex(filePath);
    } else {
      await handleLoadLigandFile(filePath);
    }
  };

  const handleLoadComplex = async (filePath: string) => {
    setIsLoading(true);
    setMdPdbPath(filePath);
    setMdInputMode('holo');
    setDetectedLigands([]);
    setSelectedLigand(null);
    setNeedsSmiles(false);
    setStatusText('Detecting ligands...');
    setError(null);

    const result = await api.detectPdbLigands(filePath);
    setIsLoading(false);

    if (result.ok) {
      const { ligands, structureInfo } = result.value;
      setDetectedLigands(ligands);
      setStatusText(null);
      if (structureInfo?.isPrepared) {
        setStatusText('Structure appears pre-prepared (has hydrogens)');
      }
      if (ligands.length >= 1) {
        handleSelectLigand(ligands[0].id, filePath);
      } else if (ligands.length === 0) {
        // Apo protein — allow continuing without a ligand
        setMdInputMode('apo');
        setMdReceptorPdb(filePath);
        setMdLigandSdf(null);
        setMdLigandName(null);
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

  const handleLoadLigandFile = async (filePath: string) => {
    setIsLoading(true);
    setMdInputMode('ligand_only');
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

  const handleConvertSmiles = async () => {
    const smiles = detectedSmiles();
    if (smiles.length === 0) return;

    setIsLoading(true);
    setMdInputMode('ligand_only');
    setError(null);

    const defaultDir = await api.getDefaultOutputDir();
    const baseOutputDir = state().customOutputDir || defaultDir;
    const paths = projectPaths(baseOutputDir, state().jobName);

    const result = await api.convertSmilesList(smiles, paths.ligands.sdf);
    setIsLoading(false);

    if (result.ok && result.value.length > 0) {
      const first = result.value[0];
      setMdSingleMoleculeInput(smiles[0]);
      setMdSingleMoleculeThumbnail(null);
      setMdLigandSdf(first.sdfPath);
      setMdLigandName(first.filename);
      setMdReceptorPdb(null);
      setThumbnailDataUrl(null);
    } else {
      setError(result.ok ? 'No molecules converted' : (result.error?.message || 'SMILES conversion failed'));
    }
  };

  const handleClear = () => {
    setMdPdbPath(null);
    setDetectedLigands([]);
    setSelectedLigand(null);
    setNeedsSmiles(false);
    setSmilesCorrection('');
    setStatusText(null);
    setMdReceptorPdb(null);
    setMdLigandSdf(null);
    setMdLigandName(null);
    setMdSingleMoleculeInput(null);
    setMdSingleMoleculeThumbnail(null);
    setThumbnailDataUrl(null);
    setSmilesText('');
    setError(null);
  };

  const canContinue = createMemo(() => state().md.ligandSdf !== null || state().md.inputMode === 'apo');

  return (
    <div class="h-full flex flex-col">
      <div class="text-center mb-3">
        <h2 class="text-xl font-bold">Load Molecules for Simulation</h2>
        <p class="text-sm text-base-content/90">Import a protein complex, ligand file, or enter SMILES</p>
      </div>

      <div class="flex-1 min-h-0 overflow-auto flex flex-col items-center gap-4">
        <div class="card bg-base-200 shadow-lg w-full max-w-md">
          <div class="card-body p-4">
            <Show
              when={!isLoaded()}
              fallback={
                <div class="space-y-3">
                  <Show when={thumbnailDataUrl()}>
                    <div class="flex justify-center">
                      <img
                        src={thumbnailDataUrl()!}
                        alt="Structure"
                        class="rounded bg-base-100 p-1"
                        style={{ "max-width": "100%", "max-height": "200px", "object-fit": "contain" }}
                      />
                    </div>
                  </Show>
                  <div class="space-y-1.5 text-xs">
                    <Show when={pdbPath()}>
                      <div class="flex justify-between py-1 border-b border-base-300">
                        <span class="text-base-content/70">Structure</span>
                        <span class="font-mono truncate max-w-[200px]">{path.basename(pdbPath()!)}</span>
                      </div>
                    </Show>
                    <Show when={state().md.ligandName}>
                      <div class="flex justify-between py-1 border-b border-base-300">
                        <span class="text-base-content/70">Ligand</span>
                        <span class="font-mono">{state().md.ligandName}</span>
                      </div>
                    </Show>
                    <div class="flex justify-between py-1">
                      <span class="text-base-content/70">Mode</span>
                      <span class="badge badge-sm">
                        {state().md.inputMode === 'apo' ? 'Apo Protein' : state().md.receptorPdb ? 'Protein + Ligand' : 'Ligand Only'}
                      </span>
                    </div>
                  </div>
                  <Show when={state().md.inputMode === 'apo'}>
                    <div class="bg-warning/10 border border-warning rounded-lg p-2 text-xs text-warning">
                      No ligand detected — confirm this is an apo (protein-only) simulation before continuing.
                    </div>
                  </Show>
                  <button class="btn btn-ghost btn-xs w-full" onClick={handleClear}>Clear</button>
                </div>
              }
            >
              <div class="space-y-3">
                {/* Ligand selection for multi-ligand PDBs */}
                <Show when={detectedLigands().length > 1}>
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
                </Show>

                {/* SMILES correction for bond order issues */}
                <Show when={needsSmiles()}>
                  <div>
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
                  <p class="text-[10px] text-base-content/60 text-center">{statusText()}</p>
                </Show>

                <Show when={!detectedLigands().length || statusText()}>
                  <button class="btn btn-outline btn-sm w-full" onClick={handleImport} disabled={isLoading()}>
                    {isLoading() ? <span class="loading loading-spinner loading-xs" /> : 'Import (.pdb, .cif, .sdf, .mol)'}
                  </button>

                  <div>
                    <span class="text-[10px] text-base-content/50">or enter PDB ID</span>
                    <div class="flex gap-1 mt-1">
                      <input
                        type="text"
                        class="input input-bordered input-sm flex-1 font-mono uppercase"
                        placeholder="e.g. 8TCE"
                        value={pdbIdText()}
                        onInput={(e) => setPdbIdText(e.currentTarget.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleFetchPdb()}
                        maxLength={4}
                      />
                      <button class="btn btn-primary btn-sm" onClick={handleFetchPdb} disabled={isLoading() || pdbIdText().trim().length !== 4}>
                        {isLoading() ? <span class="loading loading-spinner loading-xs" /> : 'Fetch'}
                      </button>
                    </div>
                  </div>

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
                      rows={4}
                    />
                  </div>

                  <button
                    class="btn btn-primary btn-sm w-full"
                    onClick={handleConvertSmiles}
                    disabled={isLoading() || detectedSmiles().length === 0}
                  >
                    {isLoading() ? <span class="loading loading-spinner loading-xs" /> : 'Enter SMILES'}
                  </button>
                </Show>
              </div>
            </Show>
          </div>
        </div>

        <Show when={state().errorMessage}>
          <div class="alert alert-error py-2 w-full max-w-md">
            <span class="text-sm">{state().errorMessage}</span>
          </div>
        </Show>
      </div>

      <div class="mt-4 flex items-center gap-3">
        <div class="flex-1 flex items-center gap-2 text-xs text-base-content/85 bg-base-200 rounded-lg px-3 py-2">
          <svg class="w-4 h-4 text-info flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>AMBER force fields and solvent models. Mac GPUs supported via native Metal and OpenCL.</span>
        </div>
        <button class="btn btn-primary" disabled={!canContinue()} onClick={() => setMdStep('md-configure')}>
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
