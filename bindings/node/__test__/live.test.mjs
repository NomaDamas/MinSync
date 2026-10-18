// Runs the real model. Opt in with MINSYNC_LIVE=1; the weights come from the
// same cache the CLI uses (HF_HOME, FASTEMBED_CACHE_DIR or
// ~/.cache/minsync/models) and are downloaded on first use.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createRequire } from 'node:module'

const { NativeEmbedder } = createRequire(import.meta.url)('../index.js')
const live = process.env.MINSYNC_LIVE === '1'
const device = process.env.MINSYNC_TEST_DEVICE
const options = (extra = {}) => ({ ...(device ? { device } : {}), ...extra })
const bits = (vector) => Array.from(new Uint32Array(vector.buffer, vector.byteOffset, vector.length))
const norm = (vector) => Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0))
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

const PREFIX = 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'

test('identity reports the embedding space', { skip: !live }, async (t) => {
  const embedder = new NativeEmbedder(options())
  t.after(() => embedder.dispose())
  assert.equal(embedder.isLoaded, false)

  const identity = await embedder.identity()

  assert.equal(identity.id, 'native:Qwen/Qwen3-Embedding-0.6B')
  assert.equal(identity.dimension, 1024)
  assert.ok(['f32', 'f16', 'bf16'].includes(identity.dtype), identity.dtype)
  assert.ok(['cpu', 'metal', 'cuda'].includes(identity.device), identity.device)
  assert.equal(identity.maxLength, 2048)
  if (device) assert.equal(identity.device, device)
  assert.equal(embedder.isLoaded, true)
})

test('embed returns unit Float32Array vectors in input order', { skip: !live }, async (t) => {
  const embedder = new NativeEmbedder(options())
  t.after(() => embedder.dispose())
  const texts = ['alpha', 'beta gamma', '한국어 문장', 'alpha']

  const vectors = await embedder.embed(texts)

  assert.equal(vectors.length, texts.length)
  for (const vector of vectors) {
    assert.ok(vector instanceof Float32Array)
    assert.equal(vector.length, 1024)
    assert.ok(Math.abs(norm(vector) - 1) < 1e-3, `norm ${norm(vector)}`)
  }
  const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0)
  assert.ok(dot(vectors[0], vectors[3]) > 0.999, 'identical text, near-identical vector')
  assert.ok(dot(vectors[0], vectors[1]) < dot(vectors[0], vectors[3]))
  assert.deepEqual(await embedder.embed([]), [])
})

test('embedQuery applies queryPrefix, embed applies passagePrefix', { skip: !live }, async (t) => {
  const query = new NativeEmbedder(options({ queryPrefix: PREFIX }))
  const asPassage = new NativeEmbedder(options({ passagePrefix: PREFIX }))
  const plain = new NativeEmbedder(options())
  t.after(() => Promise.all([query.dispose(), asPassage.dispose(), plain.dispose()]))

  const prefixedQuery = await query.embedQuery('hello world')
  const prefixedPassage = (await asPassage.embed(['hello world']))[0]
  const unprefixed = await plain.embedQuery('hello world')
  const unprefixedPassage = (await query.embed(['hello world']))[0]

  assert.deepEqual(bits(prefixedQuery), bits(prefixedPassage))
  assert.notDeepEqual(bits(prefixedQuery), bits(unprefixed))
  assert.deepEqual(bits(unprefixed), bits(unprefixedPassage), 'queryPrefix must not touch passages')
})

test('inference does not block the JS event loop', { skip: !live }, async (t) => {
  const embedder = new NativeEmbedder(options())
  t.after(() => embedder.dispose())
  await embedder.identity()

  let worst = 0
  let last = performance.now()
  const timer = setInterval(() => {
    const now = performance.now()
    worst = Math.max(worst, now - last)
    last = now
  }, 5)
  const texts = Array.from({ length: 48 }, (_, i) => `passage ${i} `.repeat(120))
  const started = performance.now()
  await embedder.embed(texts)
  const elapsed = performance.now() - started
  clearInterval(timer)

  t.diagnostic(`embed took ${elapsed.toFixed(0)} ms, worst event-loop gap ${worst.toFixed(1)} ms`)
  assert.ok(elapsed > 100, 'the workload must be long enough to prove anything')
  assert.ok(worst < Math.min(250, elapsed / 2), `event loop stalled for ${worst} ms`)
})

test('the model loads once; warm calls are fast', { skip: !live }, async (t) => {
  const embedder = new NativeEmbedder(options())
  t.after(() => embedder.dispose())

  const cold = performance.now()
  await embedder.embedQuery('warm up')
  const coldMs = performance.now() - cold

  const samples = []
  for (let i = 0; i < 10; i++) {
    const started = performance.now()
    await embedder.embedQuery(`how does query number ${i} behave`)
    samples.push(performance.now() - started)
  }
  samples.sort((a, b) => a - b)
  const median = samples[samples.length / 2]
  t.diagnostic(`cold ${coldMs.toFixed(0)} ms, warm median ${median.toFixed(1)} ms`)
  assert.ok(median < 380, `warm query ${median} ms is not below the CLI's 380 ms end-to-end`)
  assert.ok(median < coldMs / 3, 'warm calls must not pay the load again')
})

test('unload releases the model and the next call reloads identical vectors', { skip: !live }, async (t) => {
  const embedder = new NativeEmbedder(options())
  t.after(() => embedder.dispose())
  const before = await embedder.embedQuery('deterministic')
  assert.equal(embedder.isLoaded, true)

  assert.equal(await embedder.unload(), true)
  assert.equal(embedder.isLoaded, false)
  assert.equal(await embedder.unload(), false)

  const after = await embedder.embedQuery('deterministic')
  assert.equal(embedder.isLoaded, true)
  assert.deepEqual(bits(after), bits(before))
})

test('idleUnloadSeconds releases the model after quiet time, then reloads on demand', { skip: !live }, async (t) => {
  const embedder = new NativeEmbedder(options({ idleUnloadSeconds: 1 }))
  t.after(() => embedder.dispose())

  await embedder.embedQuery('first')
  assert.equal(embedder.isLoaded, true)
  await sleep(600)
  assert.equal(embedder.isLoaded, true, 'still inside the idle window')
  await embedder.embedQuery('activity restarts the window')
  await sleep(600)
  assert.equal(embedder.isLoaded, true, 'activity must restart the idle window')

  await sleep(1000)
  assert.equal(embedder.isLoaded, false)
  assert.equal((await embedder.embedQuery('reloaded')).length, 1024)
  assert.equal(embedder.isLoaded, true)
})

test('an AbortSignal stops remaining batches and leaves the embedder usable', { skip: !live }, async (t) => {
  const embedder = new NativeEmbedder(options({ batchSize: 1 }))
  t.after(() => embedder.dispose())
  await embedder.identity()
  const texts = Array.from({ length: 400 }, (_, i) => `passage ${i} `.repeat(60))

  const full = performance.now()
  await embedder.embed(texts.slice(0, 20))
  const perBatch = (performance.now() - full) / 20

  const controller = new AbortController()
  const started = performance.now()
  const pending = embedder.embed(texts, { signal: controller.signal })
  setTimeout(() => controller.abort(new Error('user cancelled')), 150)
  await assert.rejects(pending, /user cancelled/)
  const elapsed = performance.now() - started

  t.diagnostic(`aborted after ${elapsed.toFixed(0)} ms; the full run would take ~${(perBatch * 400).toFixed(0)} ms`)
  assert.ok(elapsed < 150 + 20 * perBatch + 500, `abort took ${elapsed} ms`)
  assert.equal((await embedder.embedQuery('still works')).length, 1024)
})

test('concurrent calls share one load and each gets its own vectors', { skip: !live }, async (t) => {
  const embedder = new NativeEmbedder(options())
  t.after(() => embedder.dispose())

  const [a, b, c] = await Promise.all([
    embedder.embedQuery('one'),
    embedder.embedQuery('two'),
    embedder.embedQuery('three'),
  ])

  assert.notDeepEqual(bits(a), bits(b))
  assert.notDeepEqual(bits(b), bits(c))
  assert.deepEqual(bits(a), bits(await embedder.embedQuery('one')))
})
