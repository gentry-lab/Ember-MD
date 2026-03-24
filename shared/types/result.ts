// Copyright (c) 2026 Ember Contributors. MIT License.
/**
 * Result type for explicit error handling
 * Inspired by Rust's Result<T, E>
 */
export type Result<T, E = Error> =
  | { ok: true; value: T }
  | { ok: false; error: E };

/** Create a success result */
export const Ok = <T>(value: T): Result<T, never> => ({ ok: true, value });

/** Create an error result */
export const Err = <E>(error: E): Result<never, E> => ({ ok: false, error });

/** Unwrap a result or throw */
export function unwrap<T, E>(result: Result<T, E>): T {
  if (result.ok) return result.value;
  throw result.error;
}

/** Unwrap a result or return a default value */
export function unwrapOr<T, E>(result: Result<T, E>, defaultValue: T): T {
  return result.ok ? result.value : defaultValue;
}

/** Map over a successful result */
export function map<T, U, E>(
  result: Result<T, E>,
  fn: (value: T) => U
): Result<U, E> {
  if (result.ok) return Ok(fn(result.value));
  return result;
}

/** Map over an error result */
export function mapErr<T, E, F>(
  result: Result<T, E>,
  fn: (error: E) => F
): Result<T, F> {
  if (!result.ok) return Err(fn(result.error));
  return result;
}
