// A forward pass that outlives its deadline maps to ERR_MINSYNC_TIMEOUT, as in
// the CLI. The stuck thread cannot be cancelled, so the embedder then fails
// fast. Own file: the failure is process-wide and must not leak into others.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createRequire } from 'node:module'

const { NativeEmbedder } = createRequire(import.meta.url)('../index.js')

test('perBatchTimeoutSeconds bounds one forward pass', { skip: process.env.MINSYNC_LIVE !== '1' }, async (t) => {
  // CPU, so the pass is certain to outlast a 1 s deadline.
  const embedder = new NativeEmbedder({ device: 'cpu', perBatchTimeoutSeconds: 1 })
  t.after(() => embedder.dispose())
  const heavy = Array.from({ length: 24 }, (_, i) => `passage ${i} `.repeat(400))

  const started = performance.now()
  await assert.rejects(embedder.embed(heavy), (error) => {
    assert.equal(error.code, 'ERR_MINSYNC_TIMEOUT')
    assert.match(error.message, /exceeded/)
    return true
  })
  assert.ok(performance.now() - started < 30_000, 'the caller must not wait for the stuck pass')

  const retry = performance.now()
  await assert.rejects(embedder.embedQuery('after'), (error) => error.code === 'ERR_MINSYNC_TIMEOUT')
  assert.ok(performance.now() - retry < 1000, 'later calls fail fast')
})
