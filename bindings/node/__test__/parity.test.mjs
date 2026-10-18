// Bit-for-bit parity with the CLI's `native:` embedder. Opt in by pointing
// MINSYNC_REFERENCE_VECTORS at the output of
//   cargo run --release --example embed_reference -- __test__/fixtures/texts.json <out>
// produced with the same MINSYNC_TEST_DEVICE.
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { createRequire } from 'node:module'

const { NativeEmbedder } = createRequire(import.meta.url)('../index.js')
const referencePath = process.env.MINSYNC_REFERENCE_VECTORS
const fixture = JSON.parse(readFileSync(new URL('./fixtures/texts.json', import.meta.url), 'utf8'))
const bits = (vector) => Array.from(new Uint32Array(vector.buffer, vector.byteOffset, vector.length))

test('vectors equal the CLI embedder bit for bit', { skip: !referencePath }, async (t) => {
  const reference = JSON.parse(readFileSync(referencePath, 'utf8'))
  const device = process.env.MINSYNC_TEST_DEVICE
  const embedder = new NativeEmbedder({
    ...(device ? { device } : {}),
    queryPrefix: fixture.queryPrefix ?? undefined,
    passagePrefix: fixture.passagePrefix ?? undefined,
  })
  t.after(() => embedder.dispose())

  const passages = await embedder.embed(fixture.passages)
  assert.equal(passages.length, reference.passages.length)
  passages.forEach((vector, i) => assert.deepEqual(bits(vector), reference.passages[i], `passage ${i}`))

  for (const [i, query] of fixture.queries.entries()) {
    assert.deepEqual(bits(await embedder.embedQuery(query)), reference.queries[i], `query ${i}`)
  }
})
