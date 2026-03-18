import { Component, onMount, onCleanup, createSignal, createEffect, Show } from 'solid-js';
import * as NGL from 'ngl';
import {
  workflowStore,
  DetectedLigand,
  ProteinRepresentation,
  LigandRepresentation,
  SurfaceColorScheme,
  PlaybackSpeed,
  CenterTarget,
  BindingSiteMapState,
} from '../../stores/workflow';
import TrajectoryControls from './TrajectoryControls';
import ClusteringModal from './ClusteringModal';
import AnalysisPanel from './AnalysisPanel';
import BindingSiteMapPanel from './BindingSiteMapPanel';
import FepScoringPanel from './FepScoringPanel';

// NGL representation name mapping (shared across all style update functions)
const LIGAND_REP_MAP: Record<string, string> = {
  'ball+stick': 'ball+stick',
  stick: 'licorice',
  spacefill: 'spacefill',
};

const ViewerMode: Component = () => {
  const {
    state,
    setViewerPdbPath,
    setViewerLigandPath,
    setViewerDetectedLigands,
    setViewerSelectedLigandId,
    setViewerProteinRep,
    setViewerProteinSurface,
    setViewerProteinSurfaceOpacity,
    setViewerSurfaceColorScheme,
    setViewerProteinCarbonColor,
    setViewerShowPocketResidues,
    setViewerShowPocketLabels,
    setViewerHideWaterIons,
    setViewerLigandVisible,
    setViewerLigandRep,
    setViewerLigandSurface,
    setViewerLigandSurfaceOpacity,
    setViewerLigandCarbonColor,
    setViewerLigandPolarHOnly,
    setViewerShowInteractions,
    setViewerTrajectoryPath,
    setViewerTrajectoryInfo,
    setViewerCurrentFrame,
    setViewerIsPlaying,
    setViewerPdbQueue,
    setViewerPdbQueueIndex,
    setViewerBindingSiteMap,
    setViewerIsComputingBindingSiteMap,
    resetViewer,
  } = workflowStore;

  let containerRef: HTMLDivElement | undefined;
  let stage: NGL.Stage | null = null;
  let proteinComponent: NGL.Component | null = null;
  let ligandComponent: NGL.Component | null = null;
  let interactionShapeComponent: NGL.Component | null = null;
  let playbackTimer: ReturnType<typeof setTimeout> | null = null;
  let isFrameLoading = false;
  let pendingFrameIndex: number | null = null;
  let playbackGeneration = 0;
  let alignedComponents: Map<number, NGL.Component> = new Map();  // For multi-PDB alignment
  let volumeComponents: Map<string, NGL.Component> = new Map();  // For binding site isosurfaces

  const [isLoading, setIsLoading] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [surfacePropsLoading, setSurfacePropsLoading] = createSignal(false);
  const [surfacePropsData, setSurfacePropsData] = createSignal<{
    hydrophobic: number[];
    electrostatic: number[];
  } | null>(null);
  // Track which PDB the cached surface props are for
  let surfacePropsPdbPath: string | null = null;

  const api = window.electronAPI;

  // Helper to get ligand selection string for detected ligand
  // Uses resname + resnum (not chain ID) because MDAnalysis may rewrite chain IDs
  // when generating trajectory frame PDBs
  const getLigandSelection = (): string | null => {
    const selectedId = state().viewer.selectedLigandId;
    if (!selectedId) return null;
    const ligand = state().viewer.detectedLigands.find((l) => l.id === selectedId);
    if (!ligand) return null;
    return `[${ligand.resname}] and ${ligand.resnum}`;
  };

  // Counter for unique scheme names + cleanup tracker
  let colorSchemeCounter = 0;
  const registeredSchemes: string[] = [];
  const trackScheme = (id: string) => { registeredSchemes.push(id); return id; };
  const clearOldSchemes = () => {
    while (registeredSchemes.length > 10) {
      const old = registeredSchemes.shift()!;
      try { NGL.ColormakerRegistry.removeScheme(old); } catch {}
    }
  };

  // 3-letter to 1-letter amino acid code mapping
  const aa3to1: Record<string, string> = {
    ALA: 'A', ARG: 'R', ASN: 'N', ASP: 'D', CYS: 'C',
    GLN: 'Q', GLU: 'E', GLY: 'G', HIS: 'H', ILE: 'I',
    LEU: 'L', LYS: 'K', MET: 'M', PHE: 'F', PRO: 'P',
    SER: 'S', THR: 'T', TRP: 'W', TYR: 'Y', VAL: 'V',
    // Non-standard
    MSE: 'M', HSD: 'H', HSE: 'H', HSP: 'H',
  };

  // Interpolate between two hex colors (0xRRGGBB) by t in [0,1]
  const lerpColor = (c1: number, c2: number, t: number): number => {
    const r = ((c1 >> 16) & 0xff) + t * (((c2 >> 16) & 0xff) - ((c1 >> 16) & 0xff));
    const g = ((c1 >> 8) & 0xff) + t * (((c2 >> 8) & 0xff) - ((c1 >> 8) & 0xff));
    const b = (c1 & 0xff) + t * ((c2 & 0xff) - (c1 & 0xff));
    return (Math.round(r) << 16) | (Math.round(g) << 8) | Math.round(b);
  };

  // Create a surface color scheme from computed per-atom values
  // Uses precomputed data from compute_surface_props.py (Gaussian-smoothed, not per-residue)
  const createComputedScheme = (
    values: number[],
    lowColor: number,
    midColor: number,
    highColor: number
  ): string => {
    clearOldSchemes();
    const id = `computed-${Date.now()}-${colorSchemeCounter++}`;
    const FALLBACK = 0x888888;
    return trackScheme(NGL.ColormakerRegistry.addScheme(function (this: any) {
      this.atomColor = function (atom: any) {
        const v = values[atom.index];
        if (v === undefined) return FALLBACK;
        // v is in [-1, 1] — map to color ramp
        const t = (v + 1) / 2; // [0, 1]
        if (t < 0.5) return lerpColor(lowColor, midColor, t * 2);
        return lerpColor(midColor, highColor, (t - 0.5) * 2);
      };
    }, id));
  };

  // Hydrophobic: teal (hydrophilic) → white → goldenrod (hydrophobic)
  const createHydrophobicScheme = (): string | null => {
    const data = surfacePropsData();
    if (!data) return null;
    return createComputedScheme(data.hydrophobic, 0x3BA8A0, 0xFAFAFA, 0xB8860B);
  };

  // Coulombic electrostatic: red (negative/anionic) → white → blue (positive/cationic)
  // APBS/Maestro convention: blue = positive potential, red = negative potential
  const createElectrostaticScheme = (): string | null => {
    const data = surfacePropsData();
    if (!data) return null;
    return createComputedScheme(data.electrostatic, 0xD32F2F, 0xFAFAFA, 0x2979FF);
  };

  // Create a selection-based color scheme with custom carbon color
  // Uses addSelectionScheme which is simpler and more reliable
  const createCarbonColorScheme = (carbonColorHex: string): string => {
    clearOldSchemes();
    const id = `cpk-${Date.now()}-${colorSchemeCounter++}`;

    // Selection scheme: list of [color, selection] pairs
    // Last entry is the fallback/default
    const schemeData: [string, string][] = [
      [carbonColorHex, '_C'],      // Carbon - custom color
      ['#3050F8', '_N'],           // Nitrogen - blue
      ['#FF0D0D', '_O'],           // Oxygen - red
      ['#FFFF30', '_S'],           // Sulfur - yellow
      ['#FF8000', '_P'],           // Phosphorus - orange
      ['#FFFFFF', '_H'],           // Hydrogen - white
      ['#1FF01F', '_CL'],          // Chlorine - green
      ['#90E050', '_F'],           // Fluorine - light green
      ['#A62929', '_BR'],          // Bromine - dark red
      ['#940094', '_I'],           // Iodine - purple
      ['#808080', '*'],            // Default - gray
    ];

    return trackScheme(NGL.ColormakerRegistry.addSelectionScheme(schemeData, id));
  };

  // Compute polar hydrogen atom indices for a structure within a selection
  const getPolarHydrogenIndices = (structure: any, selection: string): number[] => {
    const polarHAtoms: number[] = [];
    try {
      structure.eachAtom((atom: any) => {
        if (atom.element === 'H') {
          let isPolar = false;
          atom.eachBondedAtom((bonded: any) => {
            const el = bonded.element.toUpperCase();
            if (el === 'N' || el === 'O' || el === 'S') {
              isPolar = true;
            }
          });
          if (isPolar) {
            polarHAtoms.push(atom.index);
          }
        }
      }, new NGL.Selection(selection));
    } catch (err) {
      console.warn('Failed to compute polar H:', err);
    }
    return polarHAtoms;
  };

  const [stageReady, setStageReady] = createSignal(false);

  onMount(() => {
    if (containerRef) {
      stage = new NGL.Stage(containerRef, {
        backgroundColor: '#ffffff',
      });

      // Handle window resize
      const handleResize = () => {
        if (stage) {
          stage.handleResize();
        }
      };
      window.addEventListener('resize', handleResize);
      setStageReady(true);

      onCleanup(() => {
        window.removeEventListener('resize', handleResize);
        if (stage) {
          stage.dispose();
          stage = null;
        }
      });
    }
  });


  const updateAllStyles = updateProteinStyle;

  // Update protein representation when state changes
  const updateProteinStyle = () => {
    if (!proteinComponent) return;

    proteinComponent.removeAllRepresentations();

    const rep = state().viewer.proteinRep;
    const hideWI = state().viewer.hideWaterIons;

    // Base selection for protein (exclude water, ions, and ligand if present)
    const ligandSele = getLigandSelection();
    const proteinBaseSele = ligandSele
      ? `protein and not water and not ion and not (${ligandSele})`
      : 'protein and not water and not ion';

    // Main protein representation
    if (rep === 'spacefill') {
      // For spacefill: use custom colormaker with carbon color
      const proteinColorSchemeId = createCarbonColorScheme(state().viewer.proteinCarbonColor);

      // Always show only polar hydrogens for protein spacefill
      let proteinFullSele = proteinBaseSele;
      const structure = (proteinComponent as any).structure;
      if (structure) {
        const polarHIndices = getPolarHydrogenIndices(structure, proteinBaseSele);
        if (polarHIndices.length > 0) {
          proteinFullSele = `(${proteinBaseSele} and not _H) or @${polarHIndices.join(',')}`;
        } else {
          proteinFullSele = `${proteinBaseSele} and not _H`;
        }
      }

      proteinComponent.addRepresentation('spacefill', {
        sele: proteinFullSele,
        color: proteinColorSchemeId,  // Custom scheme uses "color:"
      });
    } else {
      // Cartoon/ribbon - use chainid coloring (different color per chain)
      // Built-in schemes use "colorScheme:"
      proteinComponent.addRepresentation(rep, {
        sele: proteinBaseSele,
        colorScheme: 'chainid',
      });
    }

    // Water molecules - show as small spheres unless hidden
    if (!hideWI) {
      proteinComponent.addRepresentation('ball+stick', {
        sele: 'water',
        colorScheme: 'element',
        scale: 0.3,
      });
      // Ions - show as spheres unless hidden
      proteinComponent.addRepresentation('spacefill', {
        sele: 'ion',
        colorScheme: 'element',
        scale: 0.5,
      });
    }

    // Protein surface
    if (state().viewer.proteinSurface) {
      const scheme = state().viewer.surfaceColorScheme;
      const opacity = state().viewer.proteinSurfaceOpacity;

      if (scheme === 'uniform-grey') {
        proteinComponent.addRepresentation('surface', {
          sele: 'protein', opacity, color: 0x888888,
        });
      } else {
        // Computed schemes — requires precomputed data from Python
        let colorSchemeId: string | null = null;
        if (scheme === 'hydrophobic') colorSchemeId = createHydrophobicScheme();
        else if (scheme === 'electrostatic') colorSchemeId = createElectrostaticScheme();

        if (colorSchemeId) {
          proteinComponent.addRepresentation('surface', {
            sele: 'protein', opacity, color: colorSchemeId,
          });
        } else {
          // Data not loaded yet — show grey as placeholder while computing
          proteinComponent.addRepresentation('surface', {
            sele: 'protein', opacity, color: 0x888888,
          });
        }
      }
    }

    // If there's an auto-detected ligand selected, handle ligand-related visualizations
    if (ligandSele) {
      // Only render ligand representation if visible
      if (state().viewer.ligandVisible) {
        const ligandRep = state().viewer.ligandRep;
        const ligandPolarH = state().viewer.ligandPolarHOnly;

        // Create colormaker with custom carbon color
        const ligandColorSchemeId = createCarbonColorScheme(state().viewer.ligandCarbonColor);

        // Determine which hydrogens to show
        let ligandFullSele = ligandSele;
        if (ligandPolarH) {
          const structure = (proteinComponent as any).structure;
          if (structure) {
            const polarHIndices = getPolarHydrogenIndices(structure, ligandSele);
            if (polarHIndices.length > 0) {
              ligandFullSele = `(${ligandSele} and not _H) or @${polarHIndices.join(',')}`;
            } else {
              ligandFullSele = `${ligandSele} and not _H`;
            }
          }
        }

        // Single representation for entire ligand (preserves bonds)
        // Custom schemes use "color:", built-in schemes use "colorScheme:"
        proteinComponent.addRepresentation(LIGAND_REP_MAP[ligandRep], {
          sele: ligandFullSele,
          color: ligandColorSchemeId,
        });

        // Ligand surface
        if (state().viewer.ligandSurface) {
          proteinComponent.addRepresentation('surface', {
            sele: ligandSele,
            opacity: state().viewer.ligandSurfaceOpacity,
            colorScheme: 'element',
          });
        }
      }

      // Pocket residues (protein sidechains within 5A of ligand) - show even if ligand hidden
      // NGL does NOT support "around" in selection strings - must use JavaScript API
      if (state().viewer.showPocketResidues) {
        const structure = (proteinComponent as any).structure;
        if (structure) {
          try {
            // Create selection for ligand atoms
            const ligandSelection = new NGL.Selection(ligandSele);

            // Get all atoms within 5 angstroms of ligand
            const nearbyAtoms = structure.getAtomSetWithinSelection(ligandSelection, 5);

            // Expand to complete residues
            const nearbyResidues = structure.getAtomSetWithinGroup(nearbyAtoms);

            // Convert to selection string
            const nearbyResString = nearbyResidues.toSeleString();

            if (nearbyResString) {
              // Show sidechains of nearby residues, excluding the ligand itself
              let pocketSele = `sidechainAttached and (${nearbyResString}) and not (${ligandSele})`;

              // Always apply polar-H filter for pocket residues
              const pocketPolarH = getPolarHydrogenIndices(structure, pocketSele);
              if (pocketPolarH.length > 0) {
                pocketSele = `(${pocketSele} and not _H) or @${pocketPolarH.join(',')}`;
              } else {
                pocketSele = `${pocketSele} and not _H`;
              }

              proteinComponent.addRepresentation('licorice', {
                sele: pocketSele,
                colorScheme: 'element',
              });

              // Add pocket residue labels if enabled (single-letter codes, no water/ions)
              if (state().viewer.showPocketLabels) {
                // Build custom label text for each CA atom (excludes water/ions)
                const labelText: Record<number, string> = {};
                const labelSele = `(${nearbyResString}) and .CA and protein and not (${ligandSele})`;

                structure.eachAtom((atom: any) => {
                  const resname = atom.resname?.toUpperCase() || '';
                  const resno = atom.resno || '';
                  const oneLetterCode = aa3to1[resname] || resname.charAt(0);
                  labelText[atom.index] = `${oneLetterCode}${resno}`;
                }, new NGL.Selection(labelSele));

                if (Object.keys(labelText).length > 0) {
                  proteinComponent.addRepresentation('label', {
                    sele: labelSele,
                    labelType: 'text',
                    labelText: labelText,
                    labelGrouping: 'residue',
                    color: 'white',
                    fontWeight: 'bold',
                    xOffset: 0,
                    yOffset: 0,
                    zOffset: 2.0,
                    fixedSize: true,
                    radiusType: 'size',
                    radiusSize: 1.5,
                    showBackground: true,
                    backgroundColor: 'black',
                    backgroundOpacity: 0.7,
                    depthWrite: false, // Render on top of other objects
                  });
                }
              }
            }
          } catch (err) {
            console.warn('Failed to compute pocket residues:', err);
          }
        }
      }

      // Protein-ligand interactions (contacts)
      if (state().viewer.showInteractions) {
        const structure = (proteinComponent as any).structure;
        if (structure) {
          try {
            const ligandSelection = new NGL.Selection(ligandSele);
            const nearbyAtoms = structure.getAtomSetWithinSelection(ligandSelection, 5);
            const nearbyResidues = structure.getAtomSetWithinGroup(nearbyAtoms);
            const nearbyResString = nearbyResidues.toSeleString();

            if (nearbyResString) {
              const contactSele = `(${ligandSele}) or (protein and (${nearbyResString}))`;

              proteinComponent.addRepresentation('contact', {
                sele: contactSele,
                filterSele: ligandSele,
                hydrogenBond: true,
                hydrophobic: false,
                halogenBond: true,
                ionicInteraction: true,
                metalCoordination: true,
                cationPi: true,
                piStacking: true,
                weakHydrogenBond: false,
                waterHydrogenBond: false,
                backboneHydrogenBond: true,
                radiusSize: 0.07,
              });
            }
          } catch (err) {
            console.warn('Failed to compute interactions:', err);
          }
        }
      }
    }

    // Fallback: if no ligand detected yet, render non-protein/non-water/non-ion atoms
    // This ensures ligand-only PDBs (e.g. cluster centroids) are always visible,
    // even before async detection completes. For protein PDBs this selection is
    // typically empty so it adds nothing visible.
    if (!ligandSele && !ligandComponent) {
      proteinComponent.addRepresentation('ball+stick', {
        sele: 'not protein and not water and not ion',
        colorScheme: 'element',
      });
    }

    // Handle pocket residues and interactions for EXTERNAL ligands (loaded from SDF)
    // This is separate because external ligands are in a different NGL component
    console.log('[Viewer] External ligand check:', {
      ligandSele,
      hasLigandComponent: !!ligandComponent,
      showPocketResidues: state().viewer.showPocketResidues,
      showInteractions: state().viewer.showInteractions,
    });

    if (!ligandSele && ligandComponent && (state().viewer.showPocketResidues || state().viewer.showInteractions)) {
      const proteinStructure = (proteinComponent as any).structure;
      const ligandStructure = (ligandComponent as any).structure;

      console.log('[Viewer] Structures:', { hasProtein: !!proteinStructure, hasLigand: !!ligandStructure });

      if (proteinStructure && ligandStructure) {
        try {
          // Get all ligand atom positions
          const ligandPositions: { x: number; y: number; z: number }[] = [];
          ligandStructure.eachAtom((atom: any) => {
            ligandPositions.push({ x: atom.x, y: atom.y, z: atom.z });
          });
          console.log('[Viewer] Ligand positions:', ligandPositions.length);

          if (ligandPositions.length > 0) {
            // Find protein residues within 5Å of any ligand atom
            const nearbyResidueIndices = new Set<number>();
            const cutoffSq = 5 * 5; // 5 Angstroms squared

            proteinStructure.eachAtom((atom: any) => {
              if (atom.residueIndex !== undefined) {
                for (const ligPos of ligandPositions) {
                  const dx = atom.x - ligPos.x;
                  const dy = atom.y - ligPos.y;
                  const dz = atom.z - ligPos.z;
                  const distSq = dx * dx + dy * dy + dz * dz;
                  if (distSq <= cutoffSq) {
                    nearbyResidueIndices.add(atom.residueIndex);
                    break; // Found a nearby ligand atom, no need to check more
                  }
                }
              }
            }, new NGL.Selection('protein'));

            console.log('[Viewer] Nearby residue indices:', nearbyResidueIndices.size);

            if (nearbyResidueIndices.size > 0) {
              // Convert residue indices to selection string
              const residueIndicesArray = Array.from(nearbyResidueIndices);
              // NGL uses @residueIndex for residue-based selection
              const nearbyResString = `@${residueIndicesArray.map(i => `[${i}]`).join(',')}`.replace(/@\[/g, '').replace(/\]/g, '');

              // Actually, let's build a proper selection by getting residue info
              const residueSelections: string[] = [];
              const seenResidues = new Set<string>();
              proteinStructure.eachResidue((residue: any) => {
                if (nearbyResidueIndices.has(residue.index)) {
                  const key = `${residue.resno}:${residue.chainname}`;
                  if (!seenResidues.has(key)) {
                    seenResidues.add(key);
                    residueSelections.push(`(${residue.resno}:${residue.chainname})`);
                  }
                }
              });

              console.log('[Viewer] Residue selections:', residueSelections.length, residueSelections.slice(0, 5));

              if (residueSelections.length > 0) {
                const pocketResiduesSele = residueSelections.join(' or ');

                // Show pocket residues
                if (state().viewer.showPocketResidues) {
                  let pocketSele = `sidechainAttached and (${pocketResiduesSele})`;

                  // Apply polar-H filter
                  const pocketPolarH = getPolarHydrogenIndices(proteinStructure, pocketSele);
                  if (pocketPolarH.length > 0) {
                    pocketSele = `(${pocketSele} and not _H) or @${pocketPolarH.join(',')}`;
                  } else {
                    pocketSele = `${pocketSele} and not _H`;
                  }

                  proteinComponent.addRepresentation('licorice', {
                    sele: pocketSele,
                    colorScheme: 'element',
                  });

                  // Add pocket residue labels if enabled
                  if (state().viewer.showPocketLabels) {
                    const labelText: Record<number, string> = {};
                    const labelSele = `(${pocketResiduesSele}) and .CA and protein`;

                    proteinStructure.eachAtom((atom: any) => {
                      const resname = atom.resname?.toUpperCase() || '';
                      const resno = atom.resno || '';
                      const oneLetterCode = aa3to1[resname] || resname.charAt(0);
                      labelText[atom.index] = `${oneLetterCode}${resno}`;
                    }, new NGL.Selection(labelSele));

                    if (Object.keys(labelText).length > 0) {
                      proteinComponent.addRepresentation('label', {
                        sele: labelSele,
                        labelType: 'text',
                        labelText: labelText,
                        labelGrouping: 'residue',
                        color: 'white',
                        fontWeight: 'bold',
                        xOffset: 0,
                        yOffset: 0,
                        zOffset: 2.0,
                        fixedSize: true,
                        radiusType: 'size',
                        radiusSize: 1.5,
                        showBackground: true,
                        backgroundColor: 'black',
                        backgroundOpacity: 0.7,
                        depthWrite: false, // Render on top of other objects
                      });
                    }
                  }
                }

                // For interactions with external ligand, compute H-bonds manually
                // NGL's contact representation only works within a single structure
                if (state().viewer.showInteractions && ligandComponent) {
                  // Clean up previous interaction shape
                  if (interactionShapeComponent && stage) {
                    stage.removeComponent(interactionShapeComponent);
                    interactionShapeComponent = null;
                  }

                  try {
                    const ligandStructure = (ligandComponent as any).structure;
                    if (ligandStructure) {
                      // Collect potential H-bond donors/acceptors from ligand
                      const ligandAtoms: { x: number; y: number; z: number; element: string; index: number }[] = [];
                      ligandStructure.eachAtom((atom: any) => {
                        const el = atom.element.toUpperCase();
                        // H-bond donors/acceptors: N, O, and H attached to them
                        if (el === 'N' || el === 'O' || el === 'H' || el === 'F') {
                          ligandAtoms.push({ x: atom.x, y: atom.y, z: atom.z, element: el, index: atom.index });
                        }
                      });

                      // Collect potential H-bond partners from protein pocket
                      const proteinAtoms: { x: number; y: number; z: number; element: string }[] = [];
                      proteinStructure.eachAtom((atom: any) => {
                        const el = atom.element.toUpperCase();
                        if (el === 'N' || el === 'O' || el === 'H') {
                          proteinAtoms.push({ x: atom.x, y: atom.y, z: atom.z, element: el });
                        }
                      }, new NGL.Selection(pocketResiduesSele));

                      // Find H-bonds (N/O...H-N/O or N/O...N/O within ~3.5Å)
                      const hbonds: { from: { x: number; y: number; z: number }; to: { x: number; y: number; z: number } }[] = [];
                      const hbondCutoffSq = 3.5 * 3.5;

                      for (const ligAtom of ligandAtoms) {
                        if (ligAtom.element === 'H') continue; // Skip H for now, check heavy atoms
                        for (const protAtom of proteinAtoms) {
                          if (protAtom.element === 'H') continue;
                          const dx = ligAtom.x - protAtom.x;
                          const dy = ligAtom.y - protAtom.y;
                          const dz = ligAtom.z - protAtom.z;
                          const distSq = dx * dx + dy * dy + dz * dz;
                          if (distSq <= hbondCutoffSq && distSq > 1.0) { // Avoid covalent bonds
                            hbonds.push({
                              from: { x: ligAtom.x, y: ligAtom.y, z: ligAtom.z },
                              to: { x: protAtom.x, y: protAtom.y, z: protAtom.z }
                            });
                          }
                        }
                      }

                      // Draw H-bonds as cylinders using Shape
                      if (hbonds.length > 0 && stage) {
                        const shape = new NGL.Shape('hbonds');
                        for (const hb of hbonds) {
                          shape.addCylinder(
                            [hb.from.x, hb.from.y, hb.from.z],
                            [hb.to.x, hb.to.y, hb.to.z],
                            [0.2, 0.6, 1.0], // Blue color for H-bonds
                            0.05 // Radius
                          );
                        }
                        interactionShapeComponent = stage.addComponentFromObject(shape);
                        interactionShapeComponent.addRepresentation('buffer');
                      }
                    }
                  } catch (err) {
                    console.warn('Failed to compute interactions for external ligand:', err);
                  }
                } else if (!state().viewer.showInteractions && interactionShapeComponent && stage) {
                  // Clean up interaction shape when interactions are disabled
                  stage.removeComponent(interactionShapeComponent);
                  interactionShapeComponent = null;
                }
              }
            }
          }
        } catch (err) {
          console.warn('Failed to compute pocket for external ligand:', err);
        }
      }
    }
  };

  // Update external ligand (SDF) style
  const updateExternalLigandStyle = () => {
    if (!ligandComponent) return;

    ligandComponent.removeAllRepresentations();

    const ligandRep = state().viewer.ligandRep;
    const ligandPolarH = state().viewer.ligandPolarHOnly;

    // Create colormaker with custom carbon color
    const ligandColorSchemeId = createCarbonColorScheme(state().viewer.ligandCarbonColor);

    // Determine selection based on polar-H setting
    let sele = '*';  // All atoms by default
    if (ligandPolarH) {
      // Show only polar hydrogens - compute using Structure API
      const structure = (ligandComponent as any).structure;
      if (structure) {
        const polarHIndices = getPolarHydrogenIndices(structure, '*');
        if (polarHIndices.length > 0) {
          sele = `not _H or @${polarHIndices.join(',')}`;
        } else {
          sele = 'not _H';
        }
      } else {
        sele = 'not _H';
      }
    }

    // Single representation (preserves bonds)
    // Custom schemes use "color:", built-in schemes use "colorScheme:"
    ligandComponent.addRepresentation(LIGAND_REP_MAP[ligandRep], {
      sele: sele,
      color: ligandColorSchemeId,
    });

    if (state().viewer.ligandSurface) {
      ligandComponent.addRepresentation('surface', {
        opacity: state().viewer.ligandSurfaceOpacity,
        color: ligandColorSchemeId,
      });
    }
  };

  // Load PDB file into viewer (used by both browse and auto-load)
  // preserveExternalLigand: if true, don't clear ligand state (for auto-load with external SDF)
  const handleLoadPdb = async (pdbPath: string, preserveExternalLigand: boolean = false) => {
    setIsLoading(true);
    setError(null);

    try {
      // Clear binding site volumes from previous PDB
      clearBindingSiteVolumes();

      // Reset ligand state only if not preserving external ligand
      if (!preserveExternalLigand) {
        setViewerLigandPath(null);
        setViewerDetectedLigands([]);
        setViewerSelectedLigandId(null);
      }

      // Always clear the component reference (will be reloaded if needed)
      if (ligandComponent && stage) {
        stage.removeComponent(ligandComponent);
      }
      ligandComponent = null;

      // Load PDB into NGL (disable default representations to control styling)
      if (stage) {
        // Remove only the protein component if preserving ligand, otherwise remove all
        if (proteinComponent) {
          stage.removeComponent(proteinComponent);
        }
        if (!preserveExternalLigand) {
          stage.removeAllComponents();
        }

        proteinComponent = await stage.loadFile(pdbPath, { defaultRepresentation: false });

        // Clear cached surface props if PDB changed
        if (surfacePropsPdbPath !== pdbPath) {
          setSurfacePropsData(null);
          surfacePropsPdbPath = null;
        }

        updateProteinStyle();

        // Detect ligands in the PDB in the background (don't block the viewer)
        if (!preserveExternalLigand) {
          api.detectPdbLigands(pdbPath).then((result) => {
            // Only apply if this PDB is still the current one (user may have navigated away)
            if (state().viewer.pdbPath === pdbPath && result.ok && result.value.length > 0) {
              setViewerDetectedLigands(result.value);
              setViewerSelectedLigandId(result.value[0].id);
              updateAllStyles();
              // Center on ligand after detection
              if (proteinComponent) {
                const ligandSele = getLigandSelection();
                if (ligandSele) {
                  (proteinComponent as any).autoView(ligandSele);
                } else {
                  proteinComponent.autoView();
                }
              }
              // Auto-load existing binding site map if available
              tryAutoLoadBindingSiteMap(pdbPath);
            } else {
              // No ligand found — center on whole structure
              if (proteinComponent) proteinComponent.autoView();
            }
          });
        } else {
          // External ligand mode — just center
          stage.autoView();
        }
      }

      setViewerPdbPath(pdbPath);
    } catch (err) {
      console.error('[Viewer] Failed to load PDB:', err);
      setError(`Failed to load PDB: ${(err as Error).message}`);
      // Clear the pdb path to prevent auto-load from retrying infinitely
      setViewerPdbPath(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleBrowsePdb = async () => {
    // Multi-select: if user picks multiple PDBs, load them as a browseable queue
    const paths = await api.selectPdbFilesMulti();
    if (!paths || paths.length === 0) return;

    if (paths.length === 1) {
      // Single file — load directly, clear queue
      setViewerPdbQueue([]);
      await handleLoadPdb(paths[0], false);
    } else {
      // Multiple files — set up queue with page-turn navigation
      const queue = paths.map(p => ({
        pdbPath: p,
        label: p.split('/').pop()?.replace('.pdb', '') || p,
      }));
      setViewerPdbQueue(queue);
      // Don't call setViewerPdbPath here — handleLoadPdb sets it at the end.
      // Calling it early triggers the auto-load effect, causing a double-load.
      await handleLoadPdb(queue[0].pdbPath, false);
    }
  };

  const handleBrowseLigand = async () => {
    const path = await api.selectSdfFile();
    if (!path) return;
    await handleLoadExternalLigand(path);
  };

  // Load external ligand from path (supports .sdf and .sdf.gz)
  const handleLoadExternalLigand = async (filePath: string) => {
    setIsLoading(true);
    setError(null);

    try {
      // Load SDF into NGL
      if (stage) {
        // Remove previous external ligand component if any
        if (ligandComponent) {
          stage.removeComponent(ligandComponent);
        }

        // NGL supports gzipped files if we specify the correct ext
        // Use firstModelOnly to load only the selected pose (not all conformers)
        const loadOptions: any = {
          firstModelOnly: true,
        };
        if (filePath.endsWith('.sdf.gz')) {
          loadOptions.ext = 'sdf';
        }

        console.log('[Viewer] Loading ligand file:', filePath, 'options:', loadOptions);
        ligandComponent = await stage.loadFile(filePath, loadOptions);

        // Wait for structure to be parsed (NGL parses asynchronously)
        await new Promise(resolve => setTimeout(resolve, 100));

        updateExternalLigandStyle();

        // Update protein style to recalculate pocket residues and interactions with new ligand
        if (proteinComponent) {
          console.log('[Viewer] Updating protein style after ligand load');
          updateProteinStyle();
        }

        stage.autoView();
      }

      setViewerLigandPath(filePath);
    } catch (err) {
      console.error('[Viewer] Failed to load ligand:', err);
      setError(`Failed to load ligand: ${(err as Error).message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // React to PDB queue navigation (page-turn arrows)
  let lastQueueIndex = -1;
  createEffect(async () => {
    const queue = state().viewer.pdbQueue;
    const idx = state().viewer.pdbQueueIndex;
    if (queue.length > 1 && idx !== lastQueueIndex && lastQueueIndex >= 0 && stageReady()) {
      lastQueueIndex = idx;

      if (isAligned() && alignedComponents.has(idx)) {
        // Aligned mode: toggle visibility instead of reloading
        for (const [i, comp] of alignedComponents.entries()) {
          comp.setVisibility(i === idx);
        }
        // Update proteinComponent reference to the visible one
        proteinComponent = alignedComponents.get(idx) || null;
      } else {
        // Normal mode: load the PDB
        const item = queue[idx];
        if (item) {
          console.log('[Viewer] Queue navigation to:', item.label, item.pdbPath);
          if (isAligned()) clearAlignment();
          await handleLoadPdb(item.pdbPath, false);
        }
      }
    }
    lastQueueIndex = idx;
  });

  // Auto-load files from store when component mounts with pre-set paths
  createEffect(async () => {
    if (!stageReady()) return;

    const pdbPath = state().viewer.pdbPath;
    const ligandPath = state().viewer.ligandPath;

    // Debug: only log on actual changes, not every reactive re-evaluation

    // Only auto-load if paths are set and components aren't already loaded
    if (pdbPath && !proteinComponent) {
      // If we have an external ligand path, preserve it during PDB load
      const hasExternalLigand = !!ligandPath;

      console.log('[Viewer] Loading PDB:', pdbPath, 'preserveExternalLigand:', hasExternalLigand);
      await handleLoadPdb(pdbPath, hasExternalLigand);

      // Load external ligand after PDB is loaded (if specified)
      if (ligandPath && !ligandComponent) {
        console.log('[Viewer] Loading ligand:', ligandPath);
        await handleLoadExternalLigand(ligandPath);
      }
    }
  });

  const handleResetView = () => {
    if (stage) {
      // Reset clipping planes, fog, and camera
      stage.setParameters({
        clipNear: 0,
        clipFar: 100,
        clipDist: 10,
        fogNear: 50,
        fogFar: 100,
      });
      stage.autoView();
    }
  };

  const handleProteinRepChange = (rep: ProteinRepresentation) => {
    setViewerProteinRep(rep);
    updateAllStyles();
  };

  const handleProteinSurfaceToggle = () => {
    setViewerProteinSurface(!state().viewer.proteinSurface);
    updateAllStyles();
  };

  const handleProteinSurfaceOpacityChange = (opacity: number) => {
    setViewerProteinSurfaceOpacity(opacity);
    updateAllStyles();
  };

  const handleSurfaceColorChange = async (scheme: SurfaceColorScheme) => {
    setViewerSurfaceColorScheme(scheme);

    // For computed schemes, ensure property data is loaded
    if (scheme !== 'uniform-grey' && !surfacePropsData()) {
      const pdbPath = state().viewer.pdbPath;
      if (!pdbPath) return;

      setSurfacePropsLoading(true);
      try {
        // Store in a surface_props subdir next to the PDB, or in the project dir
        const pdbDir = pdbPath.substring(0, pdbPath.lastIndexOf('/'));
        const outputDir = state().customOutputDir
          ? `${state().customOutputDir}/${state().jobName}/surface_props`
          : `${pdbDir}/surface_props`;

        const result = await api.computeSurfaceProps(pdbPath, outputDir);
        if (result.ok) {
          surfacePropsPdbPath = pdbPath;
          setSurfacePropsData({
            hydrophobic: result.value.hydrophobic,
            electrostatic: result.value.electrostatic,
          });
        } else {
          console.error('Surface props computation failed:', result.error?.message);
        }
      } finally {
        setSurfacePropsLoading(false);
      }
    }

    updateAllStyles();
  };

  const handleProteinCarbonColorChange = (color: string) => {
    setViewerProteinCarbonColor(color);
    updateAllStyles();
  };

  const handlePocketResiduesToggle = () => {
    setViewerShowPocketResidues(!state().viewer.showPocketResidues);
    updateAllStyles();
  };

  const handlePocketLabelsToggle = () => {
    setViewerShowPocketLabels(!state().viewer.showPocketLabels);
    updateAllStyles();
  };

  const handleHideWaterIonsToggle = () => {
    setViewerHideWaterIons(!state().viewer.hideWaterIons);
    updateAllStyles();
  };

  const handleLigandVisibleToggle = () => {
    setViewerLigandVisible(!state().viewer.ligandVisible);
    updateAllStyles();
  };

  const handleLigandPolarHToggle = () => {
    setViewerLigandPolarHOnly(!state().viewer.ligandPolarHOnly);
    if (state().viewer.selectedLigandId) {
      updateAllStyles();
    } else if (ligandComponent) {
      updateExternalLigandStyle();
    }
  };

  const handleExportPdb = async () => {
    if (!proteinComponent) return;

    try {
      // Get the structure from NGL and export as PDB
      const structure = proteinComponent.structure;
      if (!structure) return;

      // Use NGL's built-in PDB writer
      const pdbWriter = new NGL.PdbWriter(structure);
      const pdbString = pdbWriter.getString();

      // Generate a default filename
      const pdbPath = state().viewer.pdbPath;
      const baseName = pdbPath ? pdbPath.split('/').pop()?.replace('.pdb', '') : 'complex';
      const defaultName = `${baseName}_export.pdb`;

      // Save via IPC
      const savedPath = await api.savePdbFile(pdbString, defaultName);
      if (savedPath) {
        console.log('Saved PDB to:', savedPath);
      }
    } catch (err) {
      setError(`Failed to export PDB: ${(err as Error).message}`);
    }
  };

  const handleLigandRepChange = (rep: LigandRepresentation) => {
    setViewerLigandRep(rep);
    if (state().viewer.selectedLigandId) {
      updateAllStyles();
    } else if (ligandComponent) {
      updateExternalLigandStyle();
    }
  };

  const handleLigandSurfaceToggle = () => {
    setViewerLigandSurface(!state().viewer.ligandSurface);
    if (state().viewer.selectedLigandId) {
      updateAllStyles();
    } else if (ligandComponent) {
      updateExternalLigandStyle();
    }
  };

  const handleLigandSurfaceOpacityChange = (opacity: number) => {
    setViewerLigandSurfaceOpacity(opacity);
    if (state().viewer.selectedLigandId) {
      updateAllStyles();
    } else if (ligandComponent) {
      updateExternalLigandStyle();
    }
  };

  const handleLigandCarbonColorChange = (color: string) => {
    setViewerLigandCarbonColor(color);
    if (state().viewer.selectedLigandId) {
      updateAllStyles();
    } else if (ligandComponent) {
      updateExternalLigandStyle();
    }
  };

  const handleInteractionsToggle = () => {
    setViewerShowInteractions(!state().viewer.showInteractions);
    updateAllStyles();
  };

  // === Trajectory functions ===

  const handleBrowseTrajectory = async () => {
    const dcdPath = await api.selectDcdFile();
    if (!dcdPath) return;

    // If no PDB loaded, auto-detect or prompt for topology file
    let pdbPath = state().viewer.pdbPath;
    if (!pdbPath) {
      // Get the directory containing the DCD file
      const dcdDir = dcdPath.substring(0, dcdPath.lastIndexOf('/'));
      const dcdFilename = dcdPath.substring(dcdPath.lastIndexOf('/') + 1);

      // Check if this is an MD job trajectory (*_trajectory.dcd)
      // If so, look for the matching *_system.pdb topology
      if (dcdFilename.endsWith('_trajectory.dcd')) {
        const baseName = dcdFilename.replace('_trajectory.dcd', '');
        const systemPdb = `${baseName}_system.pdb`;
        const systemPdbPath = `${dcdDir}/${systemPdb}`;

        // Check if the system.pdb exists
        const exists = await api.fileExists(systemPdbPath);
        if (exists) {
          pdbPath = systemPdbPath;
        }
      }

      // If no matching system.pdb found, fall back to general detection
      if (!pdbPath) {
        const pdbFiles = await api.listPdbInDirectory(dcdDir);

        if (pdbFiles.length === 1) {
          // Auto-use the single PDB file found
          pdbPath = `${dcdDir}/${pdbFiles[0]}`;
        } else {
          // Multiple or no PDB files - prompt user to select, starting from DCD directory
          pdbPath = await api.selectPdbFile(dcdDir);
          if (!pdbPath) {
            // User cancelled PDB selection
            return;
          }
        }
      }

      // Load the PDB first
      await handleLoadPdb(pdbPath, false);
    }

    await handleLoadTrajectory(dcdPath);
  };

  const handleLoadTrajectory = async (dcdPath: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const pdbPath = state().viewer.pdbPath;
      if (!pdbPath) {
        setError('No topology PDB loaded');
        return;
      }

      // Get trajectory info from backend
      const infoResult = await api.getTrajectoryInfo(pdbPath, dcdPath);
      if (!infoResult.ok) {
        setError(infoResult.error.message);
        return;
      }

      setViewerTrajectoryInfo(infoResult.value);
      setViewerTrajectoryPath(dcdPath);
      setViewerCurrentFrame(0);

      console.log('[Viewer] Trajectory info loaded:', dcdPath, 'frames:', infoResult.value.frameCount);

      // Load the first frame
      await loadTrajectoryFrame(0);
    } catch (err) {
      console.error('[Viewer] Failed to load trajectory:', err);
      setError(`Failed to load trajectory: ${(err as Error).message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Track if this is the first frame load (for initial centering)
  let isFirstFrameLoad = true;

  // Helper: get the center-of-mass of a selection within the current proteinComponent
  const getSelectionCenter = (sele: string): any | null => {
    if (!proteinComponent) return null;
    try {
      const structure = (proteinComponent as any).structure;
      if (!structure) return null;
      const selection = new NGL.Selection(sele);
      return structure.atomCenter(selection);
    } catch {
      return null;
    }
  };

  // Helper: get the center to track based on centerTarget setting
  const getTrackingCenter = (): any | null => {
    const centerTarget = state().viewer.centerTarget;
    if (centerTarget === 'none') return null;

    if (centerTarget === 'ligand' && hasAnyLigand()) {
      const ligandSele = getLigandSelection();
      if (ligandSele) {
        return getSelectionCenter(ligandSele);
      }
    } else if (centerTarget === 'protein') {
      return getSelectionCenter('protein');
    }
    // Fallback: center of all atoms
    if (!proteinComponent) return null;
    try {
      const structure = (proteinComponent as any).structure;
      return structure ? structure.atomCenter() : null;
    } catch {
      return null;
    }
  };

  // Internal: load a specific frame from the backend as PDB and reload in NGL
  const loadTrajectoryFrameInternal = async (frameIndex: number) => {
    const pdbPath = state().viewer.pdbPath;
    const dcdPath = state().viewer.trajectoryPath;
    if (!pdbPath || !dcdPath || !stage) {
      console.error('[Viewer] Missing requirements for frame load:', { pdbPath, dcdPath, stage: !!stage });
      return;
    }

    try {
      const frameResult = await api.getTrajectoryFrame(pdbPath, dcdPath, frameIndex);
      if (!frameResult.ok) {
        console.error('[Viewer] Failed to load frame:', frameResult.error.message);
        setError(`Failed to load frame ${frameIndex}: ${frameResult.error.message}`);
        return;
      }

      const pdbString = frameResult.value.pdbString;
      if (!pdbString || pdbString.length < 100) {
        console.error('[Viewer] PDB string too short:', pdbString?.length);
        setError('Received empty or invalid frame data');
        return;
      }

      console.log(`[Viewer] Loading frame ${frameIndex}, PDB size: ${pdbString.length} bytes`);

      // Save camera orientation before destroying old component
      // This preserves the user's rotation and zoom across frame changes
      const savedOrientation = !isFirstFrameLoad
        ? stage.viewerControls.getOrientation()
        : null;

      // Create a Blob from the PDB string and load it into NGL
      const pdbBlob = new Blob([pdbString], { type: 'text/plain' });

      // Remove old protein component
      if (proteinComponent) {
        stage.removeComponent(proteinComponent);
        proteinComponent = null;
      }

      // Load new frame
      proteinComponent = await stage.loadFile(pdbBlob, {
        ext: 'pdb',
        defaultRepresentation: false
      });

      if (!proteinComponent) {
        console.error('[Viewer] Failed to create protein component');
        setError('Failed to load structure into viewer');
        return;
      }

      // Re-apply styling
      updateProteinStyle();

      // Update current frame in state
      setViewerCurrentFrame(frameIndex);

      // Center view on target
      // First frame: use autoView on target to establish initial rotation + zoom + center
      // Subsequent frames: restore saved rotation/zoom, update only the center position
      //   This prevents tumbling while keeping the molecule tracked
      // NOTE: Stage.autoView() only takes duration; StructureComponent.autoView(sele) takes selection
      if (isFirstFrameLoad) {
        // Establish the initial view centered on the tracking target
        const centerTarget = state().viewer.centerTarget;
        if (centerTarget === 'ligand' && hasAnyLigand()) {
          const ligandSele = getLigandSelection();
          if (ligandSele) {
            (proteinComponent as any).autoView(ligandSele);
          } else {
            proteinComponent.autoView();
          }
        } else if (centerTarget === 'protein') {
          (proteinComponent as any).autoView('protein');
        } else {
          proteinComponent.autoView();
        }
        isFirstFrameLoad = false;
      } else if (savedOrientation) {
        // Restore camera rotation + zoom from before frame change
        stage.viewerControls.orient(savedOrientation);

        // Now update only the center position to track the target
        // This keeps the molecule centered without changing the viewing angle
        const centerTarget = state().viewer.centerTarget;
        if (centerTarget !== 'none') {
          const newCenter = getTrackingCenter();
          if (newCenter) {
            stage.viewerControls.center(newCenter);
          }
        }
      }
    } catch (err) {
      console.error('[Viewer] Error loading frame:', err);
      setError(`Error loading frame: ${(err as Error).message}`);
    }
  };

  // Load a specific frame with mutex to prevent concurrent loads
  const loadTrajectoryFrame = async (frameIndex: number) => {
    if (isFrameLoading) {
      pendingFrameIndex = frameIndex;
      return;
    }

    isFrameLoading = true;
    pendingFrameIndex = null;

    try {
      await loadTrajectoryFrameInternal(frameIndex);
    } finally {
      isFrameLoading = false;

      if (pendingFrameIndex !== null) {
        const next = pendingFrameIndex;
        pendingFrameIndex = null;
        queueMicrotask(() => loadTrajectoryFrame(next));
      }
    }
  };

  // Playback control functions
  const handleTrajectorySeek = async (frame: number) => {
    await loadTrajectoryFrame(frame);
  };

  const handleTrajectoryPlay = () => {
    if (!state().viewer.trajectoryPath) return;

    setViewerIsPlaying(true);

    const info = state().viewer.trajectoryInfo;
    if (!info) return;

    // Increment generation to cancel any prior playback loop
    const currentGeneration = ++playbackGeneration;

    const runPlaybackLoop = async () => {
      while (playbackGeneration === currentGeneration) {
        const speed = state().viewer.playbackSpeed;
        const smoothing = state().viewer.smoothing;
        const currentFrame = state().viewer.currentFrame;
        const totalFrames = info.frameCount;

        let nextFrame = currentFrame + smoothing;

        if (nextFrame >= totalFrames) {
          if (state().viewer.loopPlayback) {
            nextFrame = 0;
          } else {
            handleTrajectoryPause();
            return;
          }
        }

        const frameStart = performance.now();
        await loadTrajectoryFrame(nextFrame);
        const elapsed = performance.now() - frameStart;

        // If cancelled during load, exit
        if (playbackGeneration !== currentGeneration) return;

        // Calculate desired interval, subtract time already spent loading
        // Base rate: 10 fps (each frame is a full IPC round-trip so this is ambitious)
        const baseIntervalMs = 1000 / 10;
        const intervalMs = baseIntervalMs / speed;
        const remainingDelay = Math.max(0, intervalMs - elapsed);

        if (remainingDelay > 0) {
          await new Promise<void>((resolve) => {
            playbackTimer = setTimeout(resolve, remainingDelay);
          });
          playbackTimer = null;
        }
      }
    };

    runPlaybackLoop();
  };

  const handleTrajectoryPause = () => {
    setViewerIsPlaying(false);
    playbackGeneration++;
    pendingFrameIndex = null;

    if (playbackTimer) {
      clearTimeout(playbackTimer);
      playbackTimer = null;
    }
  };

  const handleTrajectoryStepForward = async () => {
    if (!state().viewer.trajectoryPath) return;

    const currentFrame = state().viewer.currentFrame;
    const totalFrames = state().viewer.trajectoryInfo?.frameCount || 1;
    const smoothing = state().viewer.smoothing;

    const nextFrame = Math.min(currentFrame + smoothing, totalFrames - 1);
    await loadTrajectoryFrame(nextFrame);
  };

  const handleTrajectoryStepBackward = async () => {
    if (!state().viewer.trajectoryPath) return;

    const currentFrame = state().viewer.currentFrame;
    const smoothing = state().viewer.smoothing;

    const prevFrame = Math.max(currentFrame - smoothing, 0);
    await loadTrajectoryFrame(prevFrame);
  };

  const handleTrajectoryFirstFrame = async () => {
    if (!state().viewer.trajectoryPath) return;
    await loadTrajectoryFrame(0);
  };

  const handleTrajectoryLastFrame = async () => {
    if (!state().viewer.trajectoryPath) return;
    const totalFrames = state().viewer.trajectoryInfo?.frameCount || 1;
    await loadTrajectoryFrame(totalFrames - 1);
  };

  // Clean up playback on unmount
  onCleanup(() => {
    playbackGeneration++;
    if (playbackTimer) {
      clearTimeout(playbackTimer);
      playbackTimer = null;
    }
    isFrameLoading = false;
    pendingFrameIndex = null;
  });

  const hasTrajectory = () => state().viewer.trajectoryPath !== null;

  // FEP scoring overlay
  const [showFepPanel, setShowFepPanel] = createSignal(false);

  // Clustering modal (for trajectory analysis — separate from multi-PDB import)
  const [showClusteringModal, setShowClusteringModal] = createSignal(false);
  const handleOpenClustering = () => setShowClusteringModal(true);
  const handleCloseClustering = () => setShowClusteringModal(false);
  const handleViewCluster = async (centroidFrame: number) => {
    await loadTrajectoryFrame(centroidFrame);
  };

  // Multi-PDB alignment: load all queue PDBs, superpose onto current, toggle visibility
  const [isAligned, setIsAligned] = createSignal(false);

  const handleAlignAll = async () => {
    const queue = state().viewer.pdbQueue;
    if (queue.length < 2 || !stage || !proteinComponent) return;

    setIsLoading(true);
    setError(null);

    try {
      const refIdx = state().viewer.pdbQueueIndex;

      // Clear any previous alignment
      clearAlignment();

      // Store the reference component
      alignedComponents.set(refIdx, proteinComponent);

      // Load all other PDBs, superpose onto reference
      for (let i = 0; i < queue.length; i++) {
        if (i === refIdx) continue;

        const comp = await stage.loadFile(queue[i].pdbPath, {
          defaultRepresentation: false,
        });
        if (!comp) continue;

        // Superpose onto reference
        try {
          // Try backbone alignment first (works for proteins)
          (comp as any).superpose(proteinComponent, true, 'backbone', 'backbone');
        } catch {
          try {
            // Fallback: align without selection (uses all matching atoms)
            (comp as any).superpose(proteinComponent, false);
          } catch {
            console.warn(`Could not align ${queue[i].label}`);
          }
        }

        // Apply same styling as the reference
        updateComponentStyle(comp);

        // Hide all non-current
        comp.setVisibility(false);
        alignedComponents.set(i, comp);
      }

      setIsAligned(true);
      proteinComponent.autoView();
    } catch (err) {
      setError(`Alignment failed: ${(err as Error).message}`);
      clearAlignment();
    } finally {
      setIsLoading(false);
    }
  };

  // Apply protein + ligand styling to an arbitrary component (for aligned PDBs)
  const updateComponentStyle = (comp: NGL.Component) => {
    comp.removeAllRepresentations();

    const rep = state().viewer.proteinRep;
    const ligandSele = getLigandSelection();
    const proteinBaseSele = ligandSele
      ? `protein and not water and not ion and not (${ligandSele})`
      : 'protein and not water and not ion';

    // Protein representation
    if (rep === 'spacefill') {
      comp.addRepresentation('spacefill', { sele: proteinBaseSele, colorScheme: 'element' });
    } else {
      comp.addRepresentation(rep, { sele: proteinBaseSele, colorScheme: 'chainid' });
    }

    // Ligand representation
    if (ligandSele && state().viewer.ligandVisible) {
      comp.addRepresentation(LIGAND_REP_MAP[state().viewer.ligandRep] || 'ball+stick', {
        sele: ligandSele,
        colorScheme: 'element',
      });
    }

    // Fallback for ligand-only: show hetero atoms
    if (!ligandSele) {
      const hetSele = 'hetero and not water and not ion';
      comp.addRepresentation('ball+stick', { sele: hetSele, colorScheme: 'element' });
    }
  };

  const clearAlignment = () => {
    if (stage) {
      for (const [idx, comp] of alignedComponents.entries()) {
        // Don't remove the current proteinComponent
        if (comp !== proteinComponent) {
          stage.removeComponent(comp);
        }
      }
    }
    alignedComponents.clear();
    setIsAligned(false);
  };

  // === Binding site interaction maps ===

  const BS_CHANNELS = [
    { key: 'hydrophobic' as const, color: '#22c55e' },
    { key: 'hbondDonor' as const, color: '#3b82f6' },
    { key: 'hbondAcceptor' as const, color: '#ef4444' },
  ];

  const DEFAULT_BS_CHANNEL = { visible: true, isolevel: 0.3, opacity: 0.5 };

  const buildMapState = (data: { hydrophobicDx: string; hbondDonorDx: string; hbondAcceptorDx: string; hotspots: any[] }): BindingSiteMapState => ({
    hydrophobic: { ...DEFAULT_BS_CHANNEL },
    hbondDonor: { ...DEFAULT_BS_CHANNEL },
    hbondAcceptor: { ...DEFAULT_BS_CHANNEL },
    hydrophobicDx: data.hydrophobicDx,
    hbondDonorDx: data.hbondDonorDx,
    hbondAcceptorDx: data.hbondAcceptorDx,
    hotspots: data.hotspots || [],
  });

  const clearBindingSiteVolumes = () => {
    if (stage) {
      for (const comp of volumeComponents.values()) {
        stage.removeComponent(comp);
      }
    }
    volumeComponents.clear();
    setViewerBindingSiteMap(null);
  };

  const loadBindingSiteVolumes = async (mapState: BindingSiteMapState) => {
    if (!stage) return;

    // Clear any existing volumes
    for (const comp of volumeComponents.values()) {
      stage.removeComponent(comp);
    }
    volumeComponents.clear();

    const dxPaths: Record<string, string> = {
      hydrophobic: mapState.hydrophobicDx,
      hbondDonor: mapState.hbondDonorDx,
      hbondAcceptor: mapState.hbondAcceptorDx,
    };

    // Load all 3 DX files in parallel
    const results = await Promise.allSettled(
      BS_CHANNELS.map(async (ch) => {
        const comp = await stage.loadFile(dxPaths[ch.key], { defaultRepresentation: false });
        return { key: ch.key, color: ch.color, comp };
      })
    );

    for (const r of results) {
      if (r.status === 'fulfilled' && r.value.comp) {
        const { key, color, comp } = r.value;
        const chState = mapState[key];
        comp.addRepresentation('surface', {
          isolevel: chState.isolevel,
          color,
          opacity: chState.opacity,
          wireframe: false,
        });
        comp.setVisibility(chState.visible);
        volumeComponents.set(key, comp);
      } else if (r.status === 'rejected') {
        console.error('[Viewer] Failed to load DX:', r.reason);
      }
    }
  };

  const tryAutoLoadBindingSiteMap = async (pdbPath: string) => {
    // Check if binding_site_map/ exists next to the PDB
    const pdbDir = pdbPath.replace(/\/[^/]+$/, '');
    const mapDir = `${pdbDir}/binding_site_map`;
    const resultsPath = `${mapDir}/binding_site_results.json`;

    try {
      const data = await api.readJsonFile(resultsPath) as any;
      if (!data || !data.hydrophobicDx) return;

      const mapState = buildMapState(data);
      setViewerBindingSiteMap(mapState);
      await loadBindingSiteVolumes(mapState);
    } catch (err) {
      console.error('[Viewer] Failed to auto-load binding site map:', err);
    }
  };

  const handleComputeBindingSiteMap = async () => {
    const pdbPath = state().viewer.pdbPath;
    const selectedId = state().viewer.selectedLigandId;
    if (!pdbPath || !selectedId) return;

    const ligand = state().viewer.detectedLigands.find((l) => l.id === selectedId);
    if (!ligand) return;

    setViewerIsComputingBindingSiteMap(true);

    try {
      // Determine PDB to use: if trajectory loaded, export current frame; otherwise use static PDB
      let targetPdb = pdbPath;
      const trajectoryPath = state().viewer.trajectoryPath;
      if (trajectoryPath) {
        const tmpPath = `/tmp/ember_expand_frame_${Date.now()}.pdb`;
        const exportResult = await api.exportTrajectoryFrame({
          topologyPath: pdbPath,
          trajectoryPath: trajectoryPath,
          frameIndex: state().viewer.currentFrame,
          outputPath: tmpPath,
          stripWaters: true,
        });
        if (exportResult.ok) {
          targetPdb = exportResult.value.pdbPath;
        }
      }

      // Output dir: next to the PDB
      const pdbDir = pdbPath.replace(/\/[^/]+$/, '');
      const outputDir = `${pdbDir}/binding_site_map`;

      const result = await api.mapBindingSite({
        pdbPath: targetPdb,
        ligandResname: ligand.resname,
        ligandResnum: ligand.resnum,
        outputDir,
      });

      if (result.ok) {
        const mapState = buildMapState(result.value);
        setViewerBindingSiteMap(mapState);
        await loadBindingSiteVolumes(mapState);
      } else {
        setError(`Binding site map failed: ${(result as any).error?.message || 'Unknown error'}`);
      }
    } catch (err) {
      setError(`Binding site map error: ${(err as Error).message}`);
    } finally {
      setViewerIsComputingBindingSiteMap(false);
    }
  };

  // Reactive effect: update volume representations when channel settings change
  createEffect(() => {
    const bsMap = state().viewer.bindingSiteMap;
    if (!bsMap) return;

    for (const ch of BS_CHANNELS) {
      const comp = volumeComponents.get(ch.key);
      if (!comp) continue;

      const chState = bsMap[ch.key];
      comp.setVisibility(chState.visible);

      // Update representation parameters
      if (comp.reprList && comp.reprList.length > 0) {
        const repr = comp.reprList[0];
        repr.setParameters({
          isolevel: chState.isolevel,
          opacity: chState.opacity,
        });
      }
    }
  });

  const handleClear = () => {
    // Stop playback if running
    playbackGeneration++;
    isFrameLoading = false;
    pendingFrameIndex = null;
    if (playbackTimer) {
      clearTimeout(playbackTimer);
      playbackTimer = null;
    }

    // Clear binding site volumes
    clearBindingSiteVolumes();

    if (stage) {
      stage.removeAllComponents();
    }
    proteinComponent = null;
    ligandComponent = null;
    interactionShapeComponent = null;
    alignedComponents.clear();
    setIsAligned(false);
    resetViewer();
    setError(null);
  };

  // Get display info for detected ligand
  const getDetectedLigandInfo = () => {
    const ligands = state().viewer.detectedLigands;
    const selectedId = state().viewer.selectedLigandId;
    if (ligands.length === 0 || !selectedId) return null;

    const ligand = ligands.find((l) => l.id === selectedId);
    if (!ligand) return null;

    return `${ligand.resname} (Chain ${ligand.chain}, ${ligand.num_atoms} atoms)`;
  };

  const hasAutoDetectedLigand = () => state().viewer.detectedLigands.length > 0;
  const hasExternalLigand = () => state().viewer.ligandPath !== null;
  const hasAnyLigand = () => hasAutoDetectedLigand() || hasExternalLigand();

  return (
    <div class="h-full flex flex-col gap-2">
      {/* File selection controls */}
      <div class="card bg-base-200 p-2">
        <div class="flex flex-col gap-1">
          {/* PDB Row */}
          <div class="flex items-center gap-2">
            <button
              class="btn btn-xs btn-primary"
              onClick={handleBrowsePdb}
              disabled={isLoading()}
            >
              PDB
            </button>
            <span class="text-xs text-base-content/90 flex-1 truncate">
              {state().viewer.pdbPath || 'No PDB loaded'}
            </span>
            <Show when={state().viewer.pdbPath}>
              <button
                class="btn btn-xs btn-ghost btn-square"
                onClick={handleClear}
                title="Clear viewer"
              >
                <svg xmlns="http://www.w3.org/2000/svg" class="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd" />
                </svg>
              </button>
            </Show>
          </div>

          {/* PDB Queue Navigation (page-turn arrows) */}
          <Show when={state().viewer.pdbQueue.length > 1}>
            <div class="flex items-center gap-2 px-1">
              <button
                class="btn btn-xs btn-ghost btn-square"
                onClick={() => {
                  const idx = Math.max(0, state().viewer.pdbQueueIndex - 1);
                  setViewerPdbQueueIndex(idx);
                }}
                disabled={state().viewer.pdbQueueIndex === 0}
                title="Previous structure"
              >
                <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                </svg>
              </button>
              <span class="text-xs text-base-content/80 flex-1 text-center">
                {state().viewer.pdbQueue[state().viewer.pdbQueueIndex]?.label || ''}
                <span class="text-base-content/50 ml-1">
                  ({state().viewer.pdbQueueIndex + 1}/{state().viewer.pdbQueue.length})
                </span>
              </span>
              <button
                class="btn btn-xs btn-ghost btn-square"
                onClick={() => {
                  const idx = Math.min(state().viewer.pdbQueue.length - 1, state().viewer.pdbQueueIndex + 1);
                  setViewerPdbQueueIndex(idx);
                }}
                disabled={state().viewer.pdbQueueIndex === state().viewer.pdbQueue.length - 1}
                title="Next structure"
              >
                <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                </svg>
              </button>
            </div>
          </Show>

          {/* Ligand Row */}
          <div class="flex items-center gap-2">
            <Show when={hasAutoDetectedLigand()}>
              <div class="badge badge-success badge-sm gap-1" title="X-ray ligand detected in PDB">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-2 w-2" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
                </svg>
                Ligand
              </div>
            </Show>
            <span class="text-xs text-base-content/90 flex-1 truncate">
              <Show when={hasAutoDetectedLigand() && !hasExternalLigand()}>
                {getDetectedLigandInfo()}
              </Show>
              <Show when={hasExternalLigand()}>
                {state().viewer.ligandPath?.split('/').pop()}
              </Show>
            </span>
          </div>

          {/* Binding site map channel controls (shown after map is computed) */}
          <Show when={state().viewer.bindingSiteMap}>
            <BindingSiteMapPanel
              onCompute={handleComputeBindingSiteMap}
              onClear={clearBindingSiteVolumes}
            />
          </Show>

          {/* Trajectory Row */}
          <div class="flex items-center gap-2">
            <button
              class="btn btn-xs btn-accent"
              onClick={handleBrowseTrajectory}
              disabled={isLoading()}
              title="Load DCD trajectory (will prompt for topology PDB if needed)"
            >
              DCD
            </button>
            <span class="text-xs text-base-content/90 flex-1 truncate">
              <Show when={hasTrajectory()}>
                {state().viewer.trajectoryPath?.split('/').pop()}
                <span class="text-base-content/80 ml-1">
                  ({state().viewer.trajectoryInfo?.frameCount || 0} frames)
                </span>
              </Show>
              <Show when={!hasTrajectory()}>
                No trajectory loaded
              </Show>
            </span>
          </div>

          {/* (Cluster loading removed — use multi-PDB import instead) */}
        </div>
      </div>

      {/* Trajectory Controls */}
      <Show when={hasTrajectory()}>
        <div class="flex flex-col gap-2">
          <TrajectoryControls
            onSeek={handleTrajectorySeek}
            onPlay={handleTrajectoryPlay}
            onPause={handleTrajectoryPause}
            onStepForward={handleTrajectoryStepForward}
            onStepBackward={handleTrajectoryStepBackward}
            onFirstFrame={handleTrajectoryFirstFrame}
            onLastFrame={handleTrajectoryLastFrame}
          />
          <AnalysisPanel onOpenClustering={handleOpenClustering} />
        </div>
      </Show>

      {/* (Cluster overlay list removed) */}

      {/* Error display */}
      <Show when={error()}>
        <div class="alert alert-error text-xs py-1">
          {error()}
        </div>
      </Show>

      {/* NGL Viewer Canvas */}
      <div class="flex-1 relative rounded-lg overflow-hidden border border-base-300">
        <div
          ref={containerRef}
          class="absolute inset-0"
          style={{ width: '100%', height: '100%' }}
        />
        {/* Floating buttons */}
        <Show when={state().viewer.pdbPath}>
          <div class="absolute top-2 right-2 z-10 flex gap-1.5">
            {/* Experimental features — shown when trajectory + ligand loaded */}
            <Show when={hasTrajectory() && hasAutoDetectedLigand()}>
              <button
                class="btn btn-sm btn-ghost bg-base-300/80 hover:bg-base-300 gap-1"
                onClick={() => setShowFepPanel(true)}
                title="FEP binding free energy (experimental)"
              >
                <svg xmlns="http://www.w3.org/2000/svg" class="w-3.5 h-3.5 text-warning" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M9 3h6v5l3 3-3 3v7H9v-7l-3-3 3-3V3z" />
                </svg>
                Score
              </button>
              <button
                class="btn btn-sm btn-ghost bg-base-300/80 hover:bg-base-300 gap-1"
                onClick={handleComputeBindingSiteMap}
                title="Binding site hotspot map (experimental)"
              >
                <svg xmlns="http://www.w3.org/2000/svg" class="w-3.5 h-3.5 text-warning" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M9 3h6v5l3 3-3 3v7H9v-7l-3-3 3-3V3z" />
                </svg>
                Grow
              </button>
            </Show>
            {/* Export (always visible) */}
            <button
              class="btn btn-sm btn-ghost bg-base-300/80 hover:bg-base-300"
              onClick={handleExportPdb}
              title="Export as PDB"
            >
              <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
            </button>
          </div>
        </Show>
        {/* FEP scoring overlay */}
        <Show when={showFepPanel()}>
          <FepScoringPanel onBack={() => setShowFepPanel(false)} />
        </Show>
        <Show when={isLoading()}>
          <div class="absolute inset-0 bg-base-300/50 flex items-center justify-center">
            <span class="loading loading-spinner loading-lg text-primary" />
          </div>
        </Show>
        <Show when={!state().viewer.pdbPath && !isLoading()}>
          <div class="absolute inset-0 flex items-center justify-center text-base-content/80">
            <div class="text-center">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 mx-auto mb-2 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
              </svg>
              <p class="text-sm">Load a PDB file to visualize</p>
            </div>
          </div>
        </Show>
      </div>

      {/* Style controls - compact layout */}
      <div class="card bg-base-200 p-2">
        <div class="flex flex-col gap-1">
          {/* Protein row */}
          <div class="flex items-center gap-2 flex-wrap">
            <span class="text-xs font-semibold w-14">Protein</span>
            <select
              class="select select-xs select-bordered w-24"
              value={state().viewer.proteinRep}
              onChange={(e) => handleProteinRepChange(e.target.value as ProteinRepresentation)}
              disabled={!state().viewer.pdbPath}
            >
              <option value="cartoon">Cartoon</option>
              <option value="ribbon">Ribbon</option>
              <option value="spacefill">Spacefill</option>
            </select>
            <input
              type="color"
              class="w-6 h-6 cursor-pointer rounded border border-base-300"
              value={state().viewer.proteinCarbonColor}
              onChange={(e) => handleProteinCarbonColorChange(e.target.value)}
              disabled={!state().viewer.pdbPath}
              title="Carbon color"
            />
            <label class="label cursor-pointer gap-1 p-0">
              <input
                type="checkbox"
                class="checkbox checkbox-xs checkbox-primary"
                checked={state().viewer.proteinSurface}
                onChange={handleProteinSurfaceToggle}
                disabled={!state().viewer.pdbPath}
              />
              <span class="text-xs">Surface</span>
            </label>
            <Show when={state().viewer.proteinSurface}>
              <select
                class="select select-xs select-bordered w-32"
                value={state().viewer.surfaceColorScheme}
                onChange={(e) => handleSurfaceColorChange(e.target.value as SurfaceColorScheme)}
                disabled={!state().viewer.pdbPath}
              >
                <option value="uniform-grey">Solid</option>
                <option value="hydrophobic">Hydrophobic</option>
                <option value="electrostatic">Electrostatic</option>
              </select>
              <Show when={surfacePropsLoading()}>
                <span class="loading loading-spinner loading-xs text-primary" title="Computing surface properties..."></span>
              </Show>
              <select
                class="select select-xs select-bordered w-16"
                value={state().viewer.proteinSurfaceOpacity}
                onChange={(e) => handleProteinSurfaceOpacityChange(parseFloat(e.target.value))}
                disabled={!state().viewer.pdbPath}
                title="Surface opacity"
              >
                <option value="0.1">10%</option>
                <option value="0.2">20%</option>
                <option value="0.3">30%</option>
                <option value="0.4">40%</option>
                <option value="0.5">50%</option>
                <option value="0.6">60%</option>
                <option value="0.7">70%</option>
                <option value="0.8">80%</option>
                <option value="0.9">90%</option>
                <option value="1.0">100%</option>
              </select>
            </Show>
            <label class="label cursor-pointer gap-1 p-0" title="Show pocket residues within 5Å">
              <input
                type="checkbox"
                class="checkbox checkbox-xs checkbox-primary"
                checked={state().viewer.showPocketResidues}
                onChange={handlePocketResiduesToggle}
                disabled={!hasAnyLigand()}
              />
              <span class="text-xs">Show Pocket</span>
            </label>
            <label class="label cursor-pointer gap-1 p-0" title="Show amino acid labels on pocket residues (e.g., F2108)">
              <input
                type="checkbox"
                class="checkbox checkbox-xs checkbox-primary"
                checked={state().viewer.showPocketLabels}
                onChange={handlePocketLabelsToggle}
                disabled={!hasAnyLigand() || !state().viewer.showPocketResidues}
              />
              <span class="text-xs">Labels</span>
            </label>
            <label class="label cursor-pointer gap-1 p-0" title="Hide water molecules and ions">
              <input
                type="checkbox"
                class="checkbox checkbox-xs checkbox-primary"
                checked={state().viewer.hideWaterIons}
                onChange={handleHideWaterIonsToggle}
                disabled={!state().viewer.pdbPath}
              />
              <span class="text-xs">Hide H₂O/Ions</span>
            </label>
          </div>

          {/* Ligand row */}
          <div class="flex items-center gap-2 flex-wrap">
            <span class="text-xs font-semibold w-14">Ligand</span>
            <label class="label cursor-pointer gap-1 p-0" title="Hide/show ligand">
              <input
                type="checkbox"
                class="checkbox checkbox-xs checkbox-secondary"
                checked={state().viewer.ligandVisible}
                onChange={handleLigandVisibleToggle}
                disabled={!hasAnyLigand()}
              />
              <span class="text-xs">Show</span>
            </label>
            <select
              class="select select-xs select-bordered w-24"
              value={state().viewer.ligandRep}
              onChange={(e) => handleLigandRepChange(e.target.value as LigandRepresentation)}
              disabled={!hasAnyLigand()}
            >
              <option value="ball+stick">Ball+Stick</option>
              <option value="stick">Stick</option>
              <option value="spacefill">Spacefill</option>
            </select>
            <input
              type="color"
              class="w-6 h-6 cursor-pointer rounded border border-base-300"
              value={state().viewer.ligandCarbonColor}
              onChange={(e) => handleLigandCarbonColorChange(e.target.value)}
              disabled={!hasAnyLigand()}
              title="Carbon color"
            />
            <label class="label cursor-pointer gap-1 p-0">
              <input
                type="checkbox"
                class="checkbox checkbox-xs checkbox-secondary"
                checked={state().viewer.ligandSurface}
                onChange={handleLigandSurfaceToggle}
                disabled={!hasAnyLigand()}
              />
              <span class="text-xs">Surface</span>
            </label>
            <Show when={state().viewer.ligandSurface}>
              <select
                class="select select-xs select-bordered w-16"
                value={state().viewer.ligandSurfaceOpacity}
                onChange={(e) => handleLigandSurfaceOpacityChange(parseFloat(e.target.value))}
                disabled={!hasAnyLigand()}
                title="Surface opacity"
              >
                <option value="0.1">10%</option>
                <option value="0.2">20%</option>
                <option value="0.3">30%</option>
                <option value="0.4">40%</option>
                <option value="0.5">50%</option>
                <option value="0.6">60%</option>
                <option value="0.7">70%</option>
                <option value="0.8">80%</option>
                <option value="0.9">90%</option>
                <option value="1.0">100%</option>
              </select>
            </Show>
            <label class="label cursor-pointer gap-1 p-0" title="Show only polar hydrogens (H bonded to N/O/S)">
              <input
                type="checkbox"
                class="checkbox checkbox-xs checkbox-secondary"
                checked={state().viewer.ligandPolarHOnly}
                onChange={handleLigandPolarHToggle}
                disabled={!hasAnyLigand()}
              />
              <span class="text-xs">Polar H</span>
            </label>
            <label class="label cursor-pointer gap-1 p-0" title="Show H-bonds, hydrophobic, ionic contacts">
              <input
                type="checkbox"
                class="checkbox checkbox-xs checkbox-accent"
                checked={state().viewer.showInteractions}
                onChange={handleInteractionsToggle}
                disabled={!hasAnyLigand()}
              />
              <span class="text-xs">Interactions</span>
            </label>
            <div class="flex-1" />
            {/* Clipping distance */}
            <div class="flex items-center gap-1" title="Clipping distance (how deep into the structure to see)">
              <span class="text-[10px] text-base-content/60">Clip</span>
              <input
                type="range"
                min="0"
                max="100"
                step="1"
                value="10"
                class="range range-xs w-16"
                onInput={(e) => {
                  if (stage) {
                    const val = parseInt(e.currentTarget.value);
                    stage.setParameters({ clipDist: val });
                  }
                }}
              />
            </div>
            <button
              class="btn btn-xs btn-ghost"
              onClick={handleResetView}
              disabled={!state().viewer.pdbPath}
            >
              Reset
            </button>
          </div>
        </div>
      </div>

      {/* Clustering Modal (trajectory analysis) */}
      <ClusteringModal
        isOpen={showClusteringModal()}
        onClose={handleCloseClustering}
        onViewCluster={handleViewCluster}
      />
    </div>
  );
};

export default ViewerMode;
