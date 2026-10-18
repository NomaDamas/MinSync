import type { EmbedderIdentity, EmbedderOptions } from './binding'

export type { EmbedderIdentity, EmbedderOptions }

/** `error.code` of a rejected call. */
export type MinSyncErrorCode =
  /** A forward pass exceeded its deadline. The stuck thread cannot be cancelled: restart the process. */
  | 'ERR_MINSYNC_TIMEOUT'
  | 'ERR_MINSYNC_CONFIG'
  | 'ERR_MINSYNC_EMBEDDING'
  | 'ERR_MINSYNC'

export interface CallOptions {
  /**
   * Abort the call. Forward passes not yet started are skipped and the promise
   * rejects with `signal.reason`; a pass already running finishes in the
   * background because it cannot be interrupted.
   */
  signal?: AbortSignal
}

/**
 * In-process native embedder: the engine and weights behind
 * `minsync sync` / `minsync query`. The model loads on first use and stays
 * resident until `unload()`, `dispose()` or `idleUnloadSeconds`.
 */
export class NativeEmbedder {
  constructor(options?: EmbedderOptions)
  /** Model id, dimension, dtype and device. Loads the model if it is not resident. */
  identity(): Promise<EmbedderIdentity>
  /** Embed passages (`passagePrefix` applied): unit-length vectors in input order. */
  embed(texts: string[], options?: CallOptions): Promise<Float32Array[]>
  /** Embed a search query (`queryPrefix` applied). */
  embedQuery(text: string, options?: CallOptions): Promise<Float32Array>
  /** True while the model weights are resident. */
  readonly isLoaded: boolean
  /** Release the model; the next call reloads it. Resolves to whether one was resident. */
  unload(): Promise<boolean>
  /** Release the model and reject every later call. */
  dispose(): Promise<void>
}
