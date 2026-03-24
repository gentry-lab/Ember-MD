// Copyright (c) 2026 Ember Contributors. MIT License.
import * as NGL from 'ngl';
import type { AtomProxy } from '../types/ngl';

// NGL representation name mapping (shared across all style update functions)
export const LIGAND_REP_MAP: Record<string, string> = {
  'ball+stick': 'ball+stick',
  stick: 'licorice',
  spacefill: 'spacefill',
};

export const NGL_LABEL_RADIUS_SIZE = 1.875;

// Maestro interaction colors
export const INTERACTION_COLORS = {
  hydrogenBond:         [0.93, 0.82, 0.00] as [number, number, number],  // #EDD100 yellow
  backboneHydrogenBond: [0.93, 0.82, 0.00] as [number, number, number],  // #EDD100 yellow (same)
  ionicInteraction:     [1.00, 0.20, 0.60] as [number, number, number],  // #FF3399 hot pink/magenta
  halogenBond:          [0.58, 0.30, 0.85] as [number, number, number],  // #944DD9 purple
  metalCoordination:    [0.45, 0.50, 0.55] as [number, number, number],  // #73808C slate gray
  piStacking:           [0.00, 0.70, 0.65] as [number, number, number],  // #00B3A6 teal
  cationPi:             [0.55, 0.85, 0.15] as [number, number, number],  // #8CD926 lime green
  hydrophobic:          [0.50, 0.50, 0.50] as [number, number, number],  // #808080 gray
};
export const INTERACTION_RADIUS = 0.06;

export type InteractionType = keyof typeof INTERACTION_COLORS;

export interface DetectedInteraction {
  from: [number, number, number];
  to: [number, number, number];
  type: InteractionType;
}

export interface IxAtom {
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

export interface RingInfo {
  centroid: [number, number, number];
  normal: [number, number, number];
}

export interface SurfacePropsCacheEntry {
  atomCount: number;
  hydrophobic: number[];
  electrostatic: number[];
  cachedPath: string;
}

// Salt bridge charged group atoms
export const SALT_BRIDGE_POS: Record<string, string[]> = {
  LYS: ['NZ'], ARG: ['NH1', 'NH2', 'NE'],
};
export const SALT_BRIDGE_NEG: Record<string, string[]> = {
  ASP: ['OD1', 'OD2'], GLU: ['OE1', 'OE2'],
};

// Protein aromatic ring definitions (atom names per ring)
export const AROMATIC_RINGS: Record<string, string[][]> = {
  PHE: [['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']],
  TYR: [['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']],
  TRP: [['CG', 'CD1', 'NE1', 'CE2', 'CD2'], ['CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2']],
  HIS: [['CG', 'ND1', 'CD2', 'CE1', 'NE2']],
};

export const METALS = new Set(['ZN', 'FE', 'MG', 'CA', 'MN', 'CU', 'CO', 'NI']);
export const HALOGENS = new Set(['CL', 'BR', 'I']);

/** Collect atom data from an NGL structure within a selection */
export const collectAtoms = (
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
export const detectInteractions = (
  pocketAtoms: IxAtom[],
  ligandAtoms: IxAtom[],
): DetectedInteraction[] => {
  const results: DetectedInteraction[] = [];
  const paired = new Set<string>();
  const proteinRings = computeProteinRings(pocketAtoms);
  const ligandRings = computeLigandRings(ligandAtoms);

  // --- Angle-checking helpers ---

  // H-bond: D-H...A angle > 120° at the hydrogen (linear = 180°)
  // Returns false (not true) when donor has no H — caller handles distance-only fallback
  const hbondAngleOk = (donor: IxAtom, acceptor: IxAtom): boolean => {
    const hydrogens = donor.bonded.filter(b => b.element === 'H');
    if (hydrogens.length === 0) return false;
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
      let isSaltBridge = false;
      if (!type && dist <= 4.0) {
        if (SALT_BRIDGE_POS[pa.resname]?.includes(pa.atomname) && la.element === 'O') {
          type = 'ionicInteraction';
          isSaltBridge = true;
        } else if (SALT_BRIDGE_NEG[pa.resname]?.includes(pa.atomname) && la.element === 'N') {
          type = 'ionicInteraction';
          isSaltBridge = true;
        }
      }

      // 3. Halogen bond (ligand Cl/Br/I...protein N/O/S ≤ 3.5Å + C-X...A > 140°)
      if (!type && dist <= 3.5 && HALOGENS.has(la.element) &&
          (pa.element === 'N' || pa.element === 'O' || pa.element === 'S') &&
          halogenAngleOk(la, pa)) {
        type = 'halogenBond';
      }

      // 4. H-bond / Backbone H-bond (N/O/F...N/O/F ≤ 3.5Å + D-H...A > 120°)
      //    Also emit alongside salt bridges — charged groups often form both
      if ((!type || isSaltBridge) && dist <= 3.5) {
        const isHBE = (el: string) => el === 'N' || el === 'O' || el === 'F';
        if (isHBE(pa.element) && isHBE(la.element)) {
          const paHasH = pa.bonded.some(b => b.element === 'H');
          const laHasH = la.bonded.some(b => b.element === 'H');
          // Distance-only fallback only when neither atom has H; otherwise require angle check
          if ((!paHasH && !laHasH) || hbondAngleOk(pa, la) || hbondAngleOk(la, pa)) {
            const hbType = pa.isBackbone ? 'backboneHydrogenBond' : 'hydrogenBond';
            if (isSaltBridge) {
              // Emit H-bond as a separate interaction alongside the salt bridge
              const hbKey = `${p}-${l}-hb`;
              if (!paired.has(hbKey)) {
                paired.add(hbKey);
                results.push({ from: [la.x, la.y, la.z], to: [pa.x, pa.y, pa.z], type: hbType });
              }
            } else {
              type = hbType;
            }
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
