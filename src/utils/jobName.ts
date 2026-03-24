// Copyright (c) 2026 Ember Contributors. MIT License.
/**
 * Utilities for generating and formatting job names
 */

import type { ConformerMethod } from '../../shared/types/dock';
import { MDForceFieldPreset, MD_PRESET_PARAMS } from '../../shared/types/md';

// Word lists for random job name generation (chemistry/science themed)
const ADJECTIVES = [
  'amber', 'azure', 'bold', 'bright', 'calm', 'clear', 'cosmic', 'crisp',
  'cyan', 'dancing', 'daring', 'deep', 'eager', 'elegant', 'emerald', 'fancy',
  'fast', 'fierce', 'gentle', 'gleaming', 'golden', 'grand', 'happy', 'hidden',
  'ionic', 'jade', 'keen', 'lively', 'lucid', 'lunar', 'magic', 'misty',
  'noble', 'omega', 'polar', 'prime', 'quick', 'quiet', 'radiant', 'rapid',
  'ruby', 'rustic', 'serene', 'sharp', 'silent', 'silver', 'sleek', 'smooth',
  'solar', 'solid', 'sonic', 'stellar', 'swift', 'tidal', 'ultra', 'vivid',
  'warm', 'wild', 'wise', 'zesty', 'zinc', 'atomic', 'quantum', 'orbital',
];

const NOUNS = [
  'alpha', 'anchor', 'apex', 'arrow', 'aurora', 'beacon', 'bolt', 'bond',
  'bridge', 'carbon', 'cascade', 'catalyst', 'cipher', 'circuit', 'cluster',
  'comet', 'core', 'crystal', 'delta', 'domain', 'drift', 'echo', 'electron',
  'ember', 'enzyme', 'field', 'flame', 'flux', 'forge', 'frost', 'fusion',
  'gamma', 'garden', 'glacier', 'grove', 'harbor', 'helix', 'horizon', 'hydra',
  'island', 'kernel', 'lattice', 'ligand', 'maple', 'matrix', 'meadow', 'mesa',
  'meteor', 'neutron', 'nexus', 'nucleus', 'oasis', 'orbit', 'oxide', 'path',
  'peak', 'photon', 'pine', 'plasma', 'proton', 'pulse', 'quartz', 'radix',
  'reef', 'ridge', 'river', 'sage', 'sigma', 'spark', 'sphere', 'spiral',
  'spring', 'storm', 'stream', 'summit', 'terra', 'theta', 'tide', 'tower',
  'trail', 'valley', 'vector', 'vertex', 'vortex', 'wave', 'willow', 'zenith',
];

const ANIMALS = [
  'badger', 'beaver', 'bobcat', 'condor', 'coyote', 'crane', 'dolphin', 'eagle',
  'falcon', 'finch', 'fox', 'gazelle', 'gecko', 'gopher', 'hawk', 'heron',
  'jackal', 'jaguar', 'koala', 'lemur', 'leopard', 'lion', 'lizard', 'lynx',
  'macaw', 'marmot', 'marten', 'newt', 'ocelot', 'otter', 'owl', 'panda',
  'panther', 'parrot', 'pelican', 'penguin', 'puma', 'python', 'quail', 'raven',
  'salmon', 'seal', 'shark', 'shrew', 'sloth', 'snake', 'spider', 'squid',
  'swan', 'tapir', 'tern', 'tiger', 'toucan', 'turtle', 'viper', 'wallaby',
  'walrus', 'weasel', 'whale', 'wolf', 'wombat', 'zebra', 'dragon', 'phoenix',
];

function randomFrom<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

/**
 * Generate a random three-word job name like "fancy-carbon-wallaby"
 */
export function generateJobName(): string {
  const adj = randomFrom(ADJECTIVES);
  const noun = randomFrom(NOUNS);
  const animal = randomFrom(ANIMALS);
  return `${adj}-${noun}-${animal}`;
}

/**
 * Job type identifiers for folder naming
 */
export type JobType = 'MD' | 'Dock';

/**
 * Build output folder name for MD simulation jobs
 * Format: {jobName}_{forceField}_MD-{temp}K-{duration}ns
 */
export function buildMdFolderName(
  jobName: string,
  params: {
    forceFieldPreset: MDForceFieldPreset;
    temperatureK?: number;
    productionNs: number;
  }
): string {
  const ff = MD_PRESET_PARAMS[params.forceFieldPreset].folderSuffix;
  const temp = params.temperatureK || 300;
  const ns = params.productionNs;
  return `${jobName}_${ff}_MD-${temp}K-${ns}ns`;
}

/**
 * Build output folder name for docking runs
 * Format: Vina_{ligand} or Vina_{ligand}_{n}lig
 * Examples: Vina_VU9, Vina_ATP_5lig
 */
export function buildDockFolderName(params: {
  referenceLigandId?: string | null;
  numLigands?: number;
}): string {
  const resname = params.referenceLigandId?.split('_')[0] || 'dock';
  const ligCount = params.numLigands && params.numLigands > 1
    ? `_${params.numLigands}lig` : '';
  return `Vina_${resname}${ligCount}`;
}

/**
 * Sanitize a string for filesystem use (lowercase, alphanumeric + hyphens only)
 */
function sanitizeForFilesystem(name: string, maxLength: number): string {
  return name
    .toLowerCase()
    .trim()
    .replace(/\s+/g, '-')
    .replace(/[^a-z0-9-]/g, '')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, maxLength);
}

export function sanitizeJobName(name: string): string {
  return sanitizeForFilesystem(name, 50);
}

export function sanitizeCompoundId(name: string): string {
  return sanitizeForFilesystem(name, 40);
}

export function sanitizeConformOutputName(name: string): string {
  return sanitizeForFilesystem(name, 40);
}

export function buildXrayRunFolderName(inputFolderName: string): string {
  const descriptor = sanitizeForFilesystem(inputFolderName, 40) || 'xray';
  return `${descriptor}_xray_pose`;
}

/**
 * Estimate AM1-BCC charge computation time from ligand atom count
 */
export function estimateChargeTime(atoms: number): string {
  if (atoms <= 20) return '< 10s';
  if (atoms <= 30) return '~10-30s';
  if (atoms <= 40) return '~30s-2min';
  if (atoms <= 50) return '~1-4min';
  if (atoms <= 65) return '~2-6min';
  return '~5min+';
}

/**
 * Build run folder name for project-based directory structure
 * Format: {ff}_{compoundId}_MD-{temp}K-{duration}ns or {ff}_MD-{temp}K-{duration}ns
 */
export function buildMdRunFolderName(params: {
  forceFieldPreset: MDForceFieldPreset;
  temperatureK?: number;
  productionNs: number;
  compoundId?: string;
}): string {
  const ff = MD_PRESET_PARAMS[params.forceFieldPreset].folderSuffix;
  const temp = params.temperatureK || 300;
  const ns = params.productionNs;
  const compound = params.compoundId?.trim();
  if (compound) {
    return `${ff}_${compound}_MD-${temp}K-${ns}ns`;
  }
  return `${ff}_MD-${temp}K-${ns}ns`;
}

export function buildConformRunFolderName(params: {
  method: Exclude<ConformerMethod, 'none'>;
  maxConformers: number;
  outputName?: string | null;
  ligandName?: string | null;
  protonation?: {
    enabled: boolean;
    phMin: number;
    phMax: number;
  };
}): string {
  const descriptor = sanitizeConformOutputName(
    params.outputName?.trim() || params.ligandName?.trim() || 'molecule'
  ) || 'molecule';
  const protonationSuffix = !params.protonation?.enabled
    ? ''
    : `_prot-ph${formatPhToken(params.protonation.phMin)}-${formatPhToken(params.protonation.phMax)}`;
  return `${descriptor}_${params.method.toUpperCase()}-${params.maxConformers}conf${protonationSuffix}`;
}

export function buildDockConformRunFolderName(params: {
  referenceLigandId?: string | null;
  numLigands?: number;
  method: Exclude<ConformerMethod, 'none'>;
  maxConformers: number;
  protonation?: {
    enabled: boolean;
    phMin: number;
    phMax: number;
  };
}): string {
  const dockRunName = buildDockFolderName({
    referenceLigandId: params.referenceLigandId,
    numLigands: params.numLigands,
  }).replace(/_/g, '-');

  return buildConformRunFolderName({
    method: params.method,
    maxConformers: params.maxConformers,
    outputName: dockRunName,
    protonation: params.protonation,
  });
}

function formatPhToken(value: number): string {
  const normalized = Number.isFinite(value) ? value : 7.0;
  return normalized.toFixed(1).replace('.', 'p');
}

export function formatJobCountLabel(count: number): string {
  return `${count} ${count === 1 ? 'job' : 'jobs'}`;
}
