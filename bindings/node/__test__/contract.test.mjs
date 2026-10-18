// Behaviour that needs no model weights.
import assert from 'node:assert/strict'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'
import { createRequire } from 'node:module'

// Never reach the real Hugging Face cache or network from this file.
const scratch = mkdtempSync(join(tmpdir(), 'minsync-native-contract-'))
process.env.HF_HOME = scratch
process.env.HF_ENDPOINT = 'http://127.0.0.1:9'
delete process.env.HF_TOKEN

const { NativeEmbedder } = createRequire(import.meta.url)('../index.js')

test('invalid settings throw synchronously with a config code', () => {
  for (const options of [
    { device: 'tpu' },
    { dtype: 'int8' },
    { attention: 'psychic' },
    { timeoutSeconds: 0 },
    { idleUnloadSeconds: 0 },
    { id: 'native:' },
  ]) {
    assert.throws(
      () => new NativeEmbedder(options),
      (error) => error.code === 'ERR_MINSYNC_CONFIG' && !error.message.startsWith('ERR_'),
      JSON.stringify(options),
    )
  }
})

test('construction is lazy: no model is loaded until a call needs it', () => {
  const embedder = new NativeEmbedder()
  assert.equal(embedder.isLoaded, false)
})

test('a failed load rejects with an embedding code and stays unloaded', async () => {
  const embedder = new NativeEmbedder({ device: 'cpu' })
  await assert.rejects(embedder.embedQuery('x'), (error) => {
    assert.equal(error.code, 'ERR_MINSYNC_EMBEDDING')
    assert.match(error.message, /127\.0\.0\.1:9|download|config/i)
    return true
  })
  assert.equal(embedder.isLoaded, false)
})

test('an already aborted signal rejects with its reason before any work starts', async () => {
  const embedder = new NativeEmbedder({ device: 'cpu' })
  const reason = new Error('stop')
  const controller = new AbortController()
  controller.abort(reason)
  await assert.rejects(embedder.embed(['x'], { signal: controller.signal }), (error) => error === reason)
  await assert.rejects(embedder.embedQuery('x', { signal: controller.signal }), (error) => error === reason)
})

test('dispose rejects every later call and is idempotent', async () => {
  const embedder = new NativeEmbedder({ device: 'cpu' })
  await embedder.dispose()
  await embedder.dispose()
  for (const call of [
    () => embedder.embed(['x']),
    () => embedder.embedQuery('x'),
    () => embedder.identity(),
  ]) {
    await assert.rejects(call(), /disposed/)
  }
  assert.equal(await embedder.unload(), false)
})

test('unload on a model that was never loaded resolves false', async () => {
  const embedder = new NativeEmbedder({ device: 'cpu' })
  assert.equal(await embedder.unload(), false)
})
