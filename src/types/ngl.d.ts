// Copyright (c) 2026 Ember Contributors. MIT License.
/**
 * Augmented NGL type declarations for Ember's viewer.
 *
 * NGL exports StructureComponent and Structure but does not export
 * AtomProxy / ResidueProxy from its main entry point.  We re-export
 * them here so the viewer can reference concrete types instead of `any`.
 */

export type { default as AtomProxy } from 'ngl/dist/declarations/proxy/atom-proxy';
export type { default as ResidueProxy } from 'ngl/dist/declarations/proxy/residue-proxy';

import type AtomProxy from 'ngl/dist/declarations/proxy/atom-proxy';
import type { ColormakerParameters } from 'ngl/dist/declarations/color/colormaker';

/**
 * The `this` context inside a ColormakerRegistry.addScheme() definition function.
 * NGL assigns `atomColor` on `this` inside the callback.
 * Extends Colormaker so the function signature matches ColormakerDefinitionFunction.
 */
export interface ColormakerSchemeContext {
  atomColor?: (atom: AtomProxy) => number;
}

/**
 * NGL's SelectionSchemeData is declared as [string, string, ColormakerParameters | undefined].
 * Re-exported here so the viewer doesn't need to import from NGL's internal path.
 */
export type SelectionSchemeEntry = [string, string, ColormakerParameters | undefined];

/**
 * A PDB/CIF path that has been passed through prepareStructure().
 * Only stageLoadProtein() accepts this type — raw strings cause a compile
 * error, preventing accidental bypass of hydrogen addition.
 */
export type PreparedPath = string & { readonly __prepared: unique symbol };

/**
 * NGL load file options. Combines StageLoadFileParams with ParserParams.
 * Used for stage.loadFile() calls.
 */
export interface NglLoadOptions {
  defaultRepresentation?: boolean;
  assembly?: string;
  ext?: string;
  compressed?: string | false;
  binary?: boolean;
  name?: string;
  dir?: string;
  path?: string;
  protocol?: string;
  firstModelOnly?: boolean;
  asTrajectory?: boolean;
  cAlphaOnly?: boolean;
}

/**
 * Augment NGL's Structure with updatePosition (exists at runtime but missing from declarations).
 * Takes a flat coordinate array [x0,y0,z0,x1,y1,z1,...] and updates atom positions in place.
 */
declare module 'ngl' {
  interface Structure {
    updatePosition(position: Float32Array | number[], refresh?: boolean): void;
    refreshPosition(): void;
  }
}

/**
 * Binding site results JSON shape (read from disk).
 * Matches the output of map_binding_site.py.
 */
export interface BindingSiteResultsJson {
  hydrophobicDx: string;
  hbondDonorDx: string;
  hbondAcceptorDx: string;
  hotspots: Array<{ type: string; position: number[]; direction: number[]; score: number }>;
  gridDimensions?: [number, number, number];
  method?: 'static' | 'solvation' | 'probe';
}
