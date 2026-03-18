import { Component, onMount, onCleanup, createSignal, createEffect, Show } from 'solid-js';
import * as NGL from 'ngl';
import type { Vector3 } from 'ngl';
import {
  workflowStore,
} from '../../stores/workflow';
import type {
  ProteinRepresentation,
  LigandRepresentation,
  SurfaceColorScheme,
  BindingSiteMapState,
} from '../../stores/workflow';
import TrajectoryControls from './TrajectoryControls';
import ClusteringModal from './ClusteringModal';
import AnalysisPanel from './AnalysisPanel';
import BindingSiteMapPanel from './BindingSiteMapPanel';
import { projectPaths } from '../../utils/projectPaths';
import type { AtomProxy, ResidueProxy, SelectionSchemeEntry, NglLoadOptions, BindingSiteResultsJson } from '../../types/ngl';

// NGL representation name mapping (shared across all style update functions)
const LIGAND_REP_MAP: Record<string, string> = {
  'ball+stick': 'ball+stick',
  stick: 'licorice',
  spacefill: 'spacefill',
};

// Schrödinger-inspired interaction color palette
const INTERACTION_COLORS = {
  hydrogenBond:         [0.61, 0.42, 0.87] as [number, number, number],  // #9C6ADE purple
  backboneHydrogenBond: [0.72, 0.58, 0.96] as [number, number, number],  // #B794F4 light purple
  ionicInteraction:     [0.91, 0.27, 0.48] as [number, number, number],  // #E8457A magenta
  halogenBond:          [0.22, 0.74, 0.97] as [number, number, number],  // #38BDF8 turquoise
  metalCoordination:    [0.58, 0.64, 0.72] as [number, number, number],  // #94A3B8 gray
  piStacking:           [0.13, 0.83, 0.93] as [number, number, number],  // #22D3EE cyan
  cationPi:             [0.98, 0.57, 0.24] as [number, number, number],  // #FB923C orange
  hydrophobic:          [0.42, 0.45, 0.50] as [number, number, number],  // #6B7280 gray-green
};
const INTERACTION_RADIUS = 0.12;

type InteractionType = keyof typeof INTERACTION_COLORS;

interface DetectedInteraction {
  from: [number, number, number];
  to: [number, number, number];
  type: InteractionType;
}

interface IxAtom {
  x: number; y: number; z: number;
  element: string;
  atomname: string;
  resname: string;
  resno: number;
  chainname: string;
  isBackbone: boolean;
  aromatic: boolean;
  hasPolarBond: boolean;
  bonded: { x: number; y: number; z: number; element: string }[];
}

interface RingInfo {
  centroid: [number, number, number];
  normal: [number, number, number];
}

// Salt bridge charged group atoms
const SALT_BRIDGE_POS: Record<string, string[]> = {
  LYS: ['NZ'], ARG: ['NH1', 'NH2', 'NE'],
};
const SALT_BRIDGE_NEG: Record<string, string[]> = {
  ASP: ['OD1', 'OD2'], GLU: ['OE1', 'OE2'],
};

// Protein aromatic ring definitions (atom names per ring)
const AROMATIC_RINGS: Record<string, string[][]> = {
  PHE: [['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']],
  TYR: [['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']],
  TRP: [['CG', 'CD1', 'NE1', 'CE2', 'CD2'], ['CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2']],
  HIS: [['CG', 'ND1', 'CD2', 'CE1', 'NE2']],
};

const METALS = new Set(['ZN', 'FE', 'MG', 'CA', 'MN', 'CU', 'CO', 'NI']);
const HALOGENS = new Set(['CL', 'BR', 'I']);

/** Collect atom data from an NGL structure within a selection */
const collectAtoms = (
  structure: any, // NGL.Structure
  sele: string,
  isProteinSide: boolean,
): IxAtom[] => {
  const atoms: IxAtom[] = [];
  structure.eachAtom((atom: AtomProxy) => {
    let hasPolarBond = false;
    const bonded: { x: number; y: number; z: number; element: string }[] = [];
    atom.eachBondedAtom((b: AtomProxy) => {
      const el = b.element.toUpperCase();
      if (el === 'N' || el === 'O' || el === 'S') hasPolarBond = true;
      bonded.push({ x: b.x, y: b.y, z: b.z, element: el });
    });
    atoms.push({
      x: atom.x, y: atom.y, z: atom.z,
      element: atom.element.toUpperCase(),
      atomname: atom.atomname || '',
      resname: (atom.resname || '').toUpperCase(),
      resno: atom.resno || 0,
      chainname: (atom as any).chainname || '',
      isBackbone: isProteinSide ? !!(atom as any).isBackbone : false,
      aromatic: !!(atom as any).aromatic,
      hasPolarBond,
      bonded,
    });
  }, new NGL.Selection(sele));
  return atoms;
};

/** Compute unit normal for a set of ring atom coordinates */
const ringNormal = (coords: [number, number, number][]): [number, number, number] => {
  if (coords.length < 3) return [0, 0, 1];
  // Use spread atoms (first, middle, last) for robustness on non-planar rings
  const a = coords[0], b = coords[Math.min(2, coords.length - 1)], c = coords[coords.length - 1];
  const v1x = b[0] - a[0], v1y = b[1] - a[1], v1z = b[2] - a[2];
  const v2x = c[0] - a[0], v2y = c[1] - a[1], v2z = c[2] - a[2];
  const nx = v1y * v2z - v1z * v2y;
  const ny = v1z * v2x - v1x * v2z;
  const nz = v1x * v2y - v1y * v2x;
  const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
  if (len < 1e-8) return [0, 0, 1];
  return [nx / len, ny / len, nz / len];
};

/** Compute ring info (centroid + normal) for protein residues using known ring atom names */
const computeProteinRings = (atoms: IxAtom[]): RingInfo[] => {
  const byResidue = new Map<string, Map<string, [number, number, number]>>();
  for (const a of atoms) {
    const key = `${a.resname}_${a.resno}_${a.chainname}`;
    if (!byResidue.has(key)) byResidue.set(key, new Map());
    byResidue.get(key)!.set(a.atomname, [a.x, a.y, a.z]);
  }
  const rings: RingInfo[] = [];
  for (const [key, atomMap] of byResidue) {
    const resname = key.split('_')[0];
    const ringDefs = AROMATIC_RINGS[resname];
    if (!ringDefs) continue;
    for (const ringNames of ringDefs) {
      const coords = ringNames.map(n => atomMap.get(n)).filter(Boolean) as [number, number, number][];
      if (coords.length >= ringNames.length - 1) {
        const n = coords.length;
        rings.push({
          centroid: [
            coords.reduce((s, c) => s + c[0], 0) / n,
            coords.reduce((s, c) => s + c[1], 0) / n,
            coords.reduce((s, c) => s + c[2], 0) / n,
          ],
          normal: ringNormal(coords),
        });
      }
    }
  }
  return rings;
};

/** Cluster ligand aromatic atoms into rings (centroid + normal) by proximity */
const computeLigandRings = (atoms: IxAtom[]): RingInfo[] => {
  const aromatic = atoms.filter(a => a.aromatic);
  if (aromatic.length === 0) return [];
  const rings: RingInfo[] = [];
  const assigned = new Set<number>();
  for (let i = 0; i < aromatic.length; i++) {
    if (assigned.has(i)) continue;
    const cluster: IxAtom[] = [aromatic[i]];
    assigned.add(i);
    let head = 0;
    while (head < cluster.length) {
      const curr = cluster[head++];
      for (let j = 0; j < aromatic.length; j++) {
        if (assigned.has(j)) continue;
        const dx = curr.x - aromatic[j].x;
        const dy = curr.y - aromatic[j].y;
        const dz = curr.z - aromatic[j].z;
        if (dx * dx + dy * dy + dz * dz < 2.5 * 2.5) {
          cluster.push(aromatic[j]);
          assigned.add(j);
        }
      }
    }
    if (cluster.length >= 5) {
      const coords: [number, number, number][] = cluster.map(a => [a.x, a.y, a.z]);
      const n = coords.length;
      rings.push({
        centroid: [
          coords.reduce((s, c) => s + c[0], 0) / n,
          coords.reduce((s, c) => s + c[1], 0) / n,
          coords.reduce((s, c) => s + c[2], 0) / n,
        ],
        normal: ringNormal(coords),
      });
    }
  }
  return rings;
};

/** Detect protein-ligand interactions using distance, angle, and chemical heuristics */
const detectInteractions = (
  pocketAtoms: IxAtom[],
  ligandAtoms: IxAtom[],
): DetectedInteraction[] => {
  const results: DetectedInteraction[] = [];
  const paired = new Set<string>();
  const proteinRings = computeProteinRings(pocketAtoms);
  const ligandRings = computeLigandRings(ligandAtoms);

  // --- Angle-checking helpers ---

  // H-bond: D-H...A angle > 120° at the hydrogen (linear = 180°)
  // Falls back to distance-only if no H atoms are available in the structure
  const hbondAngleOk = (donor: IxAtom, acceptor: IxAtom): boolean => {
    const hydrogens = donor.bonded.filter(b => b.element === 'H');
    if (hydrogens.length === 0) return true; // No H available — accept on distance
    for (const h of hydrogens) {
      // Vectors from H
      const hdx = donor.x - h.x, hdy = donor.y - h.y, hdz = donor.z - h.z;
      const hax = acceptor.x - h.x, hay = acceptor.y - h.y, haz = acceptor.z - h.z;
      const dot = hdx * hax + hdy * hay + hdz * haz;
      const lenSq1 = hdx * hdx + hdy * hdy + hdz * hdz;
      const lenSq2 = hax * hax + hay * hay + haz * haz;
      if (lenSq1 < 1e-8 || lenSq2 < 1e-8) continue;
      const cosA = dot / Math.sqrt(lenSq1 * lenSq2);
      // D-H...A angle > 120° ⟹ cos(angle) < cos(120°) = -0.5
      if (cosA < -0.5) return true;
    }
    return false;
  };

  // Halogen bond: C-X...A angle > 140° (sigma-hole directionality, linear = 180°)
  const halogenAngleOk = (halogen: IxAtom, acceptor: IxAtom): boolean => {
    const carbons = halogen.bonded.filter(b => b.element === 'C');
    if (carbons.length === 0) return true; // No C found — accept on distance
    const c = carbons[0];
    // Vectors from X (halogen)
    const xcx = c.x - halogen.x, xcy = c.y - halogen.y, xcz = c.z - halogen.z;
    const xax = acceptor.x - halogen.x, xay = acceptor.y - halogen.y, xaz = acceptor.z - halogen.z;
    const dot = xcx * xax + xcy * xay + xcz * xaz;
    const lenSq1 = xcx * xcx + xcy * xcy + xcz * xcz;
    const lenSq2 = xax * xax + xay * xay + xaz * xaz;
    if (lenSq1 < 1e-8 || lenSq2 < 1e-8) return true;
    const cosA = dot / Math.sqrt(lenSq1 * lenSq2);
    // C-X...A angle > 140° ⟹ cos(angle) < cos(140°) ≈ -0.766
    return cosA < -0.766;
  };

  // --- Atom-atom interactions (priority: metal > salt bridge > halogen > H-bond > hydrophobic) ---

  for (let p = 0; p < pocketAtoms.length; p++) {
    const pa = pocketAtoms[p];
    for (let l = 0; l < ligandAtoms.length; l++) {
      const la = ligandAtoms[l];
      const dx = pa.x - la.x, dy = pa.y - la.y, dz = pa.z - la.z;
      const distSq = dx * dx + dy * dy + dz * dz;
      if (distSq > 16.0 || distSq < 1.44) continue; // >4Å or <1.2Å (covalent)
      const dist = Math.sqrt(distSq);
      let type: InteractionType | null = null;

      // 1. Metal coordination (metal...N/O/S ≤ 2.8Å) — no angle criterion
      if (dist <= 2.8) {
        const isCoord = (m: string, a: string) =>
          METALS.has(m) && (a === 'N' || a === 'O' || a === 'S');
        if (isCoord(pa.element, la.element) || isCoord(la.element, pa.element)) {
          type = 'metalCoordination';
        }
      }

      // 2. Salt bridge (charged residue atom...complementary element ≤ 4.0Å)
      if (!type && dist <= 4.0) {
        if (SALT_BRIDGE_POS[pa.resname]?.includes(pa.atomname) && la.element === 'O') {
          type = 'ionicInteraction';
        } else if (SALT_BRIDGE_NEG[pa.resname]?.includes(pa.atomname) && la.element === 'N') {
          type = 'ionicInteraction';
        }
      }

      // 3. Halogen bond (ligand Cl/Br/I...protein N/O/S ≤ 3.5Å + C-X...A > 140°)
      if (!type && dist <= 3.5 && HALOGENS.has(la.element) &&
          (pa.element === 'N' || pa.element === 'O' || pa.element === 'S') &&
          halogenAngleOk(la, pa)) {
        type = 'halogenBond';
      }

      // 4. H-bond / Backbone H-bond (N/O/F...N/O/F ≤ 3.5Å + D-H...A > 120°)
      if (!type && dist <= 3.5) {
        const isHBE = (el: string) => el === 'N' || el === 'O' || el === 'F';
        if (isHBE(pa.element) && isHBE(la.element)) {
          // Either atom could be the donor — check both directions
          if (hbondAngleOk(pa, la) || hbondAngleOk(la, pa)) {
            type = pa.isBackbone ? 'backboneHydrogenBond' : 'hydrogenBond';
          }
        }
      }

      // 5. Hydrophobic (C...C, both non-polar, ≤ 4.0Å)
      if (!type && dist <= 4.0 &&
          pa.element === 'C' && la.element === 'C' &&
          !pa.hasPolarBond && !la.hasPolarBond) {
        type = 'hydrophobic';
      }

      if (type) {
        const key = `${p}-${l}`;
        if (!paired.has(key)) {
          paired.add(key);
          results.push({ from: [la.x, la.y, la.z], to: [pa.x, pa.y, pa.z], type });
        }
      }
    }
  }

  // --- Ring-based interactions ---

  // Cation-Pi: positive protein residue → ligand aromatic ring (≤ 6.0Å)
  // Cation must be roughly above/below ring plane (angle to normal < 60°)
  const posAtoms = pocketAtoms.filter(a => SALT_BRIDGE_POS[a.resname]?.includes(a.atomname));
  for (const pa of posAtoms) {
    for (const ring of ligandRings) {
      const dx = pa.x - ring.centroid[0], dy = pa.y - ring.centroid[1], dz = pa.z - ring.centroid[2];
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist > 6.0 || dist < 1e-8) continue;
      // Angle between centroid→cation vector and ring normal
      const cosOff = Math.abs(
        (dx * ring.normal[0] + dy * ring.normal[1] + dz * ring.normal[2]) / dist
      );
      // cos > cos(60°) = 0.5 means cation is within 60° of ring normal (above/below)
      if (cosOff > 0.5) {
        results.push({ from: [pa.x, pa.y, pa.z], to: ring.centroid, type: 'cationPi' });
      }
    }
  }

  // Pi-stacking: protein ring ↔ ligand ring
  // Face-to-face (parallel displaced): normals within 30°, centroid dist ≤ 4.4Å
  // Edge-to-face (T-shaped):           normals > 50° apart, centroid dist ≤ 5.5Å
  for (const pr of proteinRings) {
    for (const lr of ligandRings) {
      const dx = pr.centroid[0] - lr.centroid[0];
      const dy = pr.centroid[1] - lr.centroid[1];
      const dz = pr.centroid[2] - lr.centroid[2];
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist > 5.5 || dist < 1e-8) continue;
      // Angle between ring normals (abs handles normal-flip ambiguity)
      const nDot = Math.abs(
        pr.normal[0] * lr.normal[0] + pr.normal[1] * lr.normal[1] + pr.normal[2] * lr.normal[2]
      );
      // |cos| > 0.866 → angle < 30° (face-to-face / parallel displaced)
      // |cos| < 0.643 → angle > 50° (edge-to-face / T-shaped)
      const isFaceToFace = nDot > 0.866 && dist <= 4.4;
      const isEdgeToFace = nDot < 0.643 && dist <= 5.5;
      if (isFaceToFace || isEdgeToFace) {
        results.push({ from: lr.centroid, to: pr.centroid, type: 'piStacking' });
      }
    }
  }

  return results;
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

  // eslint-disable-next-line no-unassigned-vars -- SolidJS ref pattern
  let containerRef: HTMLDivElement | undefined;
  let stage: NGL.Stage | null = null;
  let proteinComponent: NGL.Component | null = null;
  let ligandComponent: NGL.Component | null = null;
  let interactionShapeComponent: NGL.Component | null = null;
  let playbackTimer: ReturnType<typeof setTimeout> | null = null;
  let isFrameLoading = false;
  let pendingFrameIndex: number | null = null;
  let playbackGeneration = 0;
  const alignedComponents: Map<number, NGL.Component> = new Map();  // For multi-PDB alignment
  const volumeComponents: Map<string, NGL.Component> = new Map();  // For binding site isosurfaces

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
      try { NGL.ColormakerRegistry.removeScheme(old); } catch { /* best-effort cleanup */ }
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
    return trackScheme(NGL.ColormakerRegistry.addScheme(function (this: NGL.Colormaker) {
      this.atomColor = function (atom: AtomProxy) {
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
    const schemeData: SelectionSchemeEntry[] = [
      [carbonColorHex, '_C', undefined],      // Carbon - custom color
      ['#3050F8', '_N', undefined],           // Nitrogen - blue
      ['#FF0D0D', '_O', undefined],           // Oxygen - red
      ['#FFFF30', '_S', undefined],           // Sulfur - yellow
      ['#FF8000', '_P', undefined],           // Phosphorus - orange
      ['#FFFFFF', '_H', undefined],           // Hydrogen - white
      ['#1FF01F', '_CL', undefined],          // Chlorine - green
      ['#90E050', '_F', undefined],           // Fluorine - light green
      ['#A62929', '_BR', undefined],          // Bromine - dark red
      ['#940094', '_I', undefined],           // Iodine - purple
      ['#808080', '*', undefined],            // Default - gray
    ];

    return trackScheme(NGL.ColormakerRegistry.addSelectionScheme(schemeData, id));
  };

  // Compute polar hydrogen atom indices for a structure within a selection
  const getPolarHydrogenIndices = (structure: NGL.Structure, selection: string): number[] => {
    const polarHAtoms: number[] = [];
    try {
      structure.eachAtom((atom: AtomProxy) => {
        if (atom.element === 'H') {
          let isPolar = false;
          atom.eachBondedAtom((bonded: AtomProxy) => {
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
      console.log('[Viewer] NGL Stage created');

      onCleanup(() => {
        console.log('[Viewer] NGL Stage disposing');
        window.removeEventListener('resize', handleResize);
        if (stage) {
          stage.dispose();
          stage = null;
        }
        proteinComponent = null;
        ligandComponent = null;
        interactionShapeComponent = null;
      });
    }
  });

  // When viewer mode becomes active again after being CSS-hidden, NGL needs a resize
  // to recalculate canvas dimensions (display:none → display:block changes layout)
  createEffect(() => {
    if (state().mode === 'viewer' && stage) {
      // Defer to next frame so the DOM has finished layout
      requestAnimationFrame(() => stage?.handleResize());
    }
  });


  const updateAllStyles = () => updateProteinStyle();

  // Render detected interactions as dashed cylinders on the stage
  const renderInteractionShape = (interactions: DetectedInteraction[]) => {
    if (interactions.length > 0 && stage) {
      const shape = new NGL.Shape('interactions');
      const DASH = 0.25; // dash length in Å
      const GAP = 0.15;  // gap length in Å
      const STEP = DASH + GAP;
      for (const ix of interactions) {
        const color = INTERACTION_COLORS[ix.type];
        const dx = ix.to[0] - ix.from[0];
        const dy = ix.to[1] - ix.from[1];
        const dz = ix.to[2] - ix.from[2];
        const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (len < 1e-4) continue;
        const ux = dx / len, uy = dy / len, uz = dz / len;
        let t = 0;
        while (t < len) {
          const tEnd = Math.min(t + DASH, len);
          const p1: [number, number, number] = [
            ix.from[0] + ux * t, ix.from[1] + uy * t, ix.from[2] + uz * t,
          ];
          const p2: [number, number, number] = [
            ix.from[0] + ux * tEnd, ix.from[1] + uy * tEnd, ix.from[2] + uz * tEnd,
          ];
          shape.addCylinder(p1, p2, color, INTERACTION_RADIUS, ix.type);
          t += STEP;
        }
      }
      interactionShapeComponent = stage.addComponentFromObject(shape) as NGL.Component || null;
      if (interactionShapeComponent) interactionShapeComponent.addRepresentation('buffer', {});
    }
  };

  // Update protein representation when state changes
  const updateProteinStyle = () => {
    if (!proteinComponent) return;

    // Remove previous interaction shape (separate stage component)
    if (interactionShapeComponent && stage) {
      stage.removeComponent(interactionShapeComponent);
      interactionShapeComponent = null;
    }

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
      const structure = (proteinComponent as NGL.StructureComponent).structure;
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
          const structure = (proteinComponent as NGL.StructureComponent).structure;
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
        const structure = (proteinComponent as NGL.StructureComponent).structure;
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

                structure.eachAtom((atom: AtomProxy) => {
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
        const structure = (proteinComponent as NGL.StructureComponent).structure;
        if (structure) {
          try {
            const ligandSelection = new NGL.Selection(ligandSele);
            const nearbyAtoms = structure.getAtomSetWithinSelection(ligandSelection, 5);
            const nearbyResidues = structure.getAtomSetWithinGroup(nearbyAtoms);
            const nearbyResString = nearbyResidues.toSeleString();

            if (nearbyResString) {
              const pocketSele = `protein and (${nearbyResString})`;
              const pAtoms = collectAtoms(structure, pocketSele, true);
              const lAtoms = collectAtoms(structure, ligandSele, false);
              const ixs = detectInteractions(pAtoms, lAtoms);
              renderInteractionShape(ixs);
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
    if (!ligandSele && ligandComponent && (state().viewer.showPocketResidues || state().viewer.showInteractions)) {
      const proteinStructure = (proteinComponent as NGL.StructureComponent).structure;
      const ligandStructure = (ligandComponent as NGL.StructureComponent).structure;

      if (proteinStructure && ligandStructure) {
        try {
          // Get all ligand atom positions
          const ligandPositions: { x: number; y: number; z: number }[] = [];
          ligandStructure.eachAtom((atom: AtomProxy) => {
            ligandPositions.push({ x: atom.x, y: atom.y, z: atom.z });
          });
          console.log('[Viewer] Ligand positions:', ligandPositions.length);

          if (ligandPositions.length > 0) {
            // Find protein residues within 5Å of any ligand atom
            const nearbyResidueIndices = new Set<number>();
            const cutoffSq = 5 * 5; // 5 Angstroms squared

            proteinStructure.eachAtom((atom: AtomProxy) => {
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
              // Build a proper selection by getting residue info
              const residueSelections: string[] = [];
              const seenResidues = new Set<string>();
              proteinStructure.eachResidue((residue: ResidueProxy) => {
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

                    proteinStructure.eachAtom((atom: AtomProxy) => {
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

                // Cross-structure interactions with external ligand
                if (state().viewer.showInteractions && ligandComponent && ligandStructure) {
                  try {
                    const pAtoms = collectAtoms(proteinStructure, pocketResiduesSele, true);
                    const lAtoms = collectAtoms(ligandStructure, '*', false);
                    const ixs = detectInteractions(pAtoms, lAtoms);
                    renderInteractionShape(ixs);
                  } catch (err) {
                    console.warn('Failed to compute interactions for external ligand:', err);
                  }
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
      const structure = (ligandComponent as NGL.StructureComponent).structure;
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
    console.log(`[Viewer] Loading PDB: ${pdbPath} (preserveLigand=${preserveExternalLigand})`);
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

        proteinComponent = await stage.loadFile(pdbPath, { defaultRepresentation: false }) as NGL.Component || null;

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
                  (proteinComponent as NGL.StructureComponent).autoView(ligandSele);
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

  // Import a PDB/CIF into the current project's raw/ directory
  // Loads the raw file directly — NGL handles bond orders natively (CIF bond tables / PDB CCD lookup)
  // and detects interactions via distance-based heuristics without needing explicit H atoms.
  const importToProject = async (sourcePath: string): Promise<string> => {
    const defaultDir = await api.getDefaultOutputDir();
    const baseOutputDir = state().customOutputDir || defaultDir;
    const paths = projectPaths(baseOutputDir, state().jobName);
    const result = await api.importStructure(sourcePath, paths.root);
    if (result.ok) return result.value;
    return sourcePath;
  };

  const handleBrowsePdb = async () => {
    // Multi-select: if user picks multiple PDBs, load them as a browseable queue
    const selected = await api.selectPdbFilesMulti();
    if (!selected || selected.length === 0) return;

    // Import all selected files into the project
    const imported = await Promise.all(selected.map(importToProject));

    if (imported.length === 1) {
      // Single file — load directly, clear queue
      setViewerPdbQueue([]);
      await handleLoadPdb(imported[0], false);
    } else {
      // Multiple files — set up queue with page-turn navigation
      const queue = imported.map(p => ({
        pdbPath: p,
        label: p.split('/').pop()?.replace('.pdb', '').replace('.cif', '') || p,
      }));
      setViewerPdbQueue(queue);
      await handleLoadPdb(queue[0].pdbPath, false);
    }
  };

  // eslint-disable-next-line @typescript-eslint/no-unused-vars -- will be wired to UI
  const handleBrowseLigand = async () => {
    const path = await api.selectSdfFile();
    if (!path) return;
    await handleLoadExternalLigand(path);
  };

  // Load external ligand from path (supports .sdf and .sdf.gz)
  const handleLoadExternalLigand = async (filePath: string) => {
    console.log('[Viewer] Loading external ligand:', filePath);
    setIsLoading(true);
    setError(null);

    try {
      // Load SDF into NGL
      if (stage) {
        // Remove previous external ligand component if any
        if (ligandComponent) {
          stage.removeComponent(ligandComponent);
        }

        // NGL auto-detects .sdf.gz (format=SDF, compression=gzip) — don't override ext
        // Use firstModelOnly to load only the selected pose (not all conformers)
        const loadOptions: NglLoadOptions = {
          firstModelOnly: true,
        };

        console.log('[Viewer] Loading ligand file:', filePath, 'options:', loadOptions);
        ligandComponent = await stage.loadFile(filePath, loadOptions) as NGL.Component || null;

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
  // eslint-disable-next-line solid/reactivity -- async load is intentional
  createEffect(async () => {
    const queue = state().viewer.pdbQueue;
    const idx = state().viewer.pdbQueueIndex;
    if (queue.length > 1 && idx !== lastQueueIndex && lastQueueIndex >= 0 && stageReady()) {
      lastQueueIndex = idx;
      const item = queue[idx];
      if (!item) { lastQueueIndex = idx; return; }

      if (item.ligandPath) {
        // Docking mode: keep receptor, swap ligand only
        console.log(`[Viewer] Queue nav (docking): ${item.label} — ligand=${item.ligandPath}`);
        const currentPdb = state().viewer.pdbPath;
        if (currentPdb !== item.pdbPath || !proteinComponent) {
          await handleLoadPdb(item.pdbPath, true);
        }
        await handleLoadExternalLigand(item.ligandPath);
      } else if (isAligned() && alignedComponents.has(idx)) {
        // Aligned mode: toggle visibility instead of reloading
        for (const [i, comp] of alignedComponents.entries()) {
          comp.setVisibility(i === idx);
        }
        // Update proteinComponent reference to the visible one
        proteinComponent = alignedComponents.get(idx) || null;
      } else {
        // Normal mode: load the PDB
        console.log('[Viewer] Queue navigation to:', item.label, item.pdbPath);
        if (isAligned()) clearAlignment();
        await handleLoadPdb(item.pdbPath, false);
      }
    }
    lastQueueIndex = idx;
  });

  // Auto-load files from store when component mounts with pre-set paths
  // eslint-disable-next-line solid/reactivity -- async load is intentional
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
        // Store in surfaces/ subdir in the project dir, or next to the PDB
        const pdbDir = pdbPath.substring(0, pdbPath.lastIndexOf('/'));
        const outputDir = state().customOutputDir
          ? `${state().customOutputDir}/${state().jobName}/surfaces`
          : `${pdbDir}/surfaces`;

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
      const structure = (proteinComponent as NGL.StructureComponent).structure;
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
  const getSelectionCenter = (sele: string): Vector3 | null => {
    if (!proteinComponent) return null;
    try {
      const structure = (proteinComponent as NGL.StructureComponent).structure;
      if (!structure) return null;
      const selection = new NGL.Selection(sele);
      return structure.atomCenter(selection);
    } catch {
      return null;
    }
  };

  // Helper: get the center to track based on centerTarget setting
  const getTrackingCenter = (): Vector3 | null => {
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
      const structure = (proteinComponent as NGL.StructureComponent).structure;
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
      }) as NGL.Component || null;

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
            (proteinComponent as NGL.StructureComponent).autoView(ligandSele);
          } else {
            proteinComponent.autoView();
          }
        } else if (centerTarget === 'protein') {
          (proteinComponent as NGL.StructureComponent).autoView('protein');
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

  // Clustering modal (for trajectory analysis — separate from multi-PDB import)
  const [showClusteringModal, setShowClusteringModal] = createSignal(false);
  const handleOpenClustering = () => setShowClusteringModal(true);
  const handleCloseClustering = () => setShowClusteringModal(false);
  const handleViewCluster = async (centroidFrame: number) => {
    await loadTrajectoryFrame(centroidFrame);
  };

  // Multi-PDB alignment: load all queue PDBs, superpose onto current, toggle visibility
  const [isAligned, setIsAligned] = createSignal(false);

  // eslint-disable-next-line @typescript-eslint/no-unused-vars -- will be wired to UI
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
          (comp as NGL.StructureComponent).superpose(proteinComponent as NGL.StructureComponent, true, 'backbone', 'backbone');
        } catch {
          try {
            // Fallback: align without selection (uses all matching atoms)
            (comp as NGL.StructureComponent).superpose(proteinComponent as NGL.StructureComponent, false, '', '');
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
      for (const [_idx, comp] of alignedComponents.entries()) {
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

  const buildMapState = (data: BindingSiteResultsJson): BindingSiteMapState => ({
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
    const stageRef = stage;  // capture for async closure

    // Clear any existing volumes
    for (const comp of volumeComponents.values()) {
      stageRef.removeComponent(comp);
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
        const comp = await stageRef.loadFile(dxPaths[ch.key], { defaultRepresentation: false });
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
    // Check if binding_site_map/ exists in surfaces/ of the project, or next to the PDB
    const pdbDir = pdbPath.replace(/\/[^/]+$/, '');
    // Walk up to project root (handles simulations/{run}/ or docking/{run}/ nesting)
    const projectDir = pdbDir.replace(/\/(simulations|docking)\/[^/]+$/, '');
    const mapDir = projectDir !== pdbDir
      ? `${projectDir}/surfaces/binding_site_map`
      : `${pdbDir}/binding_site_map`;
    // Try prefixed results file first, then legacy unprefixed
    const projName = projectDir !== pdbDir ? projectDir.split('/').pop() || '' : '';
    const prefixedPath = projName ? `${mapDir}/${projName}_binding_site_results.json` : null;
    const legacyPath = `${mapDir}/binding_site_results.json`;

    try {
      const data = (prefixedPath ? await api.readJsonFile(prefixedPath) as BindingSiteResultsJson | null : null)
        || await api.readJsonFile(legacyPath) as BindingSiteResultsJson | null;
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

      // Output dir: surfaces/binding_site_map in the project, or next to the PDB
      const pdbDir = pdbPath.replace(/\/[^/]+$/, '');
      const projectDir = pdbDir.replace(/\/(simulations|docking)\/[^/]+$/, '');
      const outputDir = projectDir !== pdbDir
        ? `${projectDir}/surfaces/binding_site_map`
        : `${pdbDir}/binding_site_map`;

      const result = await api.mapBindingSite({
        pdbPath: targetPdb,
        ligandResname: ligand.resname,
        ligandResnum: parseInt(ligand.resnum, 10),
        outputDir,
      });

      if (result.ok) {
        const mapState = buildMapState(result.value);
        setViewerBindingSiteMap(mapState);
        await loadBindingSiteVolumes(mapState);
      } else {
        setError(`Binding site map failed: ${result.error?.message || 'Unknown error'}`);
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
            {/* Grow — binding site expansion map */}
            <Show when={hasAutoDetectedLigand()}>
              <button
                class="btn btn-sm btn-ghost bg-base-300/80 hover:bg-base-300"
                onClick={handleComputeBindingSiteMap}
                disabled={state().viewer.isComputingBindingSiteMap}
                title="Grow — show where to expand the ligand"
              >
                {state().viewer.isComputingBindingSiteMap ? (
                  <span class="loading loading-spinner loading-xs" />
                ) : (
                  <span class="font-mono text-xs">MAP</span>
                )}
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
                <span class="loading loading-spinner loading-xs text-primary" title="Computing surface properties..." />
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
