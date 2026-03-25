// Copyright (c) 2026 Ember Contributors. MIT License.
/**
 * Application error types
 */

export type AppError =
  | { type: 'PYTHON_NOT_FOUND'; message: string }
  | { type: 'SCRIPT_NOT_FOUND'; path: string; message: string }
  | { type: 'FILE_NOT_FOUND'; path: string; message: string }
  | { type: 'PREP_FAILED'; message: string; stderr?: string }
  | { type: 'SURFACE_FAILED'; message: string; stderr?: string }
  | { type: 'GENERATION_FAILED'; message: string; stderr?: string }
  | { type: 'DIRECTORY_ERROR'; path: string; message: string }
  | { type: 'PARSE_FAILED'; message: string }
  | { type: 'DOWNLOAD_FAILED'; message: string }
  | { type: 'DOCKING_FAILED'; message: string }
  | { type: 'PROTONATION_FAILED'; message: string }
  | { type: 'CONFORMER_FAILED'; message: string }
  | { type: 'VALIDATION_FAILED'; message: string }
  | { type: 'BENCHMARK_FAILED'; message: string }
  | { type: 'SIMULATION_FAILED'; message: string }
  | { type: 'CORDIAL_FAILED'; message: string }
  | { type: 'FILE_WRITE_ERROR'; path: string; message: string }
  | { type: 'TRAJECTORY_READ_FAILED'; message: string }
  | { type: 'CLUSTERING_FAILED'; message: string }
  | { type: 'EXPORT_FAILED'; message: string }
  | { type: 'ANALYSIS_FAILED'; message: string }
  | { type: 'REPORT_FAILED'; message: string }
  | { type: 'DIRECTORY_NOT_FOUND'; message: string }
  | { type: 'NO_CLUSTERS_FOUND'; message: string }
  | { type: 'SCAN_FAILED'; message: string }
  | { type: 'ALIGNMENT_FAILED'; message: string }
  | { type: 'BINDING_SITE_MAP_FAILED'; message: string }
  | { type: 'POCKET_MAP_FAILED'; message: string }
  | { type: 'SURFACE_PROPS_FAILED'; message: string }
  | { type: 'FEP_SCORING_FAILED'; message: string }
  | { type: 'IMPORT_FAILED'; message: string }
  | { type: 'FEP_SCORING_CANCELLED'; message: string }
  | { type: 'REFINEMENT_FAILED'; message: string }
  | { type: 'CLUSTER_SCORING_FAILED'; message: string }
  | { type: 'USER_CANCELLED'; message: string }
  | { type: 'MOVE_FAILED'; message: string }
  | { type: 'UNKNOWN'; message: string };

export function createError(
  type: AppError['type'],
  message: string,
  extra?: Partial<AppError>
): AppError {
  return { type, message, ...extra } as AppError;
}

export function formatError(error: AppError): string {
  switch (error.type) {
    case 'PYTHON_NOT_FOUND':
      return `Python environment not found: ${error.message}`;
    case 'SCRIPT_NOT_FOUND':
      return `Script not found at ${error.path}`;
    case 'FILE_NOT_FOUND':
      return `File not found: ${error.path}`;
    case 'PREP_FAILED':
      return `PDB preparation failed: ${error.message}`;
    case 'SURFACE_FAILED':
      return `Surface generation failed: ${error.message}`;
    case 'GENERATION_FAILED':
      return `Molecule generation failed: ${error.message}`;
    case 'DIRECTORY_ERROR':
      return `Directory error at ${error.path}: ${error.message}`;
    case 'PARSE_FAILED':
      return `Parse failed: ${error.message}`;
    case 'DOWNLOAD_FAILED':
      return `Download failed: ${error.message}`;
    case 'DOCKING_FAILED':
      return `Docking failed: ${error.message}`;
    case 'PROTONATION_FAILED':
      return `Protonation failed: ${error.message}`;
    case 'CONFORMER_FAILED':
      return `Conformer generation failed: ${error.message}`;
    case 'VALIDATION_FAILED':
      return `Validation failed: ${error.message}`;
    case 'BENCHMARK_FAILED':
      return `Benchmark failed: ${error.message}`;
    case 'SIMULATION_FAILED':
      return `Simulation failed: ${error.message}`;
    case 'CORDIAL_FAILED':
      return `CORDIAL scoring failed: ${error.message}`;
    case 'FILE_WRITE_ERROR':
      return `File write error at ${error.path}: ${error.message}`;
    case 'TRAJECTORY_READ_FAILED':
      return `Trajectory read failed: ${error.message}`;
    case 'CLUSTERING_FAILED':
      return `Clustering failed: ${error.message}`;
    case 'EXPORT_FAILED':
      return `Export failed: ${error.message}`;
    case 'ANALYSIS_FAILED':
      return `Analysis failed: ${error.message}`;
    case 'REPORT_FAILED':
      return `Report generation failed: ${error.message}`;
    case 'DIRECTORY_NOT_FOUND':
      return `Directory not found: ${error.message}`;
    case 'NO_CLUSTERS_FOUND':
      return `No clusters found: ${error.message}`;
    case 'SCAN_FAILED':
      return `Scan failed: ${error.message}`;
    case 'ALIGNMENT_FAILED':
      return `Alignment failed: ${error.message}`;
    case 'BINDING_SITE_MAP_FAILED':
      return `Binding site mapping failed: ${error.message}`;
    case 'POCKET_MAP_FAILED':
      return `Pocket map failed: ${error.message}`;
    case 'CLUSTER_SCORING_FAILED':
      return `Cluster scoring failed: ${error.message}`;
    default:
      return error.message;
  }
}
