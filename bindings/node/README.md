# @nomadamas/minsync-native

MinSync's native embedder (Qwen3 on [candle](https://github.com/huggingface/candle), Metal on macOS) as an in-process
[Node-API](https://nodejs.org/api/n-api.html) addon. Same engine, weights, cache and batch limits as the `minsync` CLI's
`native:` embedder, without a second process, a port, or a cold model load per call.

Node-API is ABI-stable: one binary per platform works on every Node/Electron version >= 18, with no per-version rebuild.

| Platform | Package | Device |
|---|---|---|
| macOS arm64 | `@nomadamas/minsync-native-darwin-arm64` | Metal (CPU fallback) |
| Windows x64 | `@nomadamas/minsync-native-win32-x64-msvc` | CPU |
| Linux x64 (glibc) | `@nomadamas/minsync-native-linux-x64-gnu` | CPU |

```ts
import { NativeEmbedder } from '@nomadamas/minsync-native'

const embedder = new NativeEmbedder({
  id: 'native:Qwen/Qwen3-Embedding-0.6B', // default
  queryPrefix: 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:',
  idleUnloadSeconds: 600, // optional: release ~2.7 GB after 10 idle minutes
})

const identity = await embedder.identity()
// { id: 'native:Qwen/Qwen3-Embedding-0.6B', dimension: 1024, dtype: 'f16', device: 'metal', maxLength: 2048 }

const passages = await embedder.embed(['first passage', 'second passage']) // Float32Array[]
const query = await embedder.embedQuery('which passage?') // Float32Array

await embedder.dispose() // releases the model; later calls reject
```

## API

`new NativeEmbedder(options?)` takes the `[embedder]` keys of `config.toml` in camelCase. Omitted keys take the CLI's
defaults. Invalid values throw synchronously (`error.code === 'ERR_MINSYNC_CONFIG'`).

| Option | Default | Meaning |
|---|---|---|
| `id` | `native:Qwen/Qwen3-Embedding-0.6B` | `native:<Hugging Face repo id>` |
| `device` | `auto` | `auto`, `cpu`, `metal` |
| `dtype` | `auto` | `auto` (f16 on Metal, f32 elsewhere), `f32`, `f16`, `bf16` |
| `attention` | `auto` | `auto`, `eager`, `tiled`, `fused` |
| `queryPrefix` / `passagePrefix` | none | prepended by `embedQuery` / `embed` |
| `batchSize` | 64 | most inputs per forward pass |
| `maxLength` | 2048 | tokens kept per input |
| `maxBatchTokens` | 4096 | cap on padded tokens per forward pass |
| `timeoutSeconds` / `perBatchTimeoutSeconds` | 60 (900 on CPU) | deadline of one forward pass |
| `modelCacheDir` | see below | model cache directory |
| `idleUnloadSeconds` | never | release the model after this much idle time; the next call reloads it |

| Method | |
|---|---|
| `identity()` | model id (as in `config.toml`), dimension, resolved dtype and device. Loads the model if it is not resident. |
| `embed(texts, { signal? })` | passages, `passagePrefix` applied; unit-length `Float32Array` per text, in input order |
| `embedQuery(text, { signal? })` | one query, `queryPrefix` applied |
| `isLoaded` | `true` while the weights are resident |
| `unload()` | release the weights now; resolves to whether a model was resident |
| `dispose()` | `unload()`, then reject every later call |

The model loads on first use, once per process, and stays resident (~2.8 GB RSS on Metal with f16) until `unload()`,
`dispose()` or `idleUnloadSeconds`. Concurrent first calls share one load.

### Same results as the CLI

`embed` and `embedQuery` run the CLI's code path: same batch planner and padded-token cap, same prefixes, same
deadline. For the same text, device, dtype and batch composition the vectors are bit-for-bit identical to the CLI's
(`__test__/parity.test.mjs`). Different batch compositions or devices can differ in the low bits; do not mix vectors from
two devices in one index.

### Threads, abort and deadlines

Inference runs on a native thread pool; the JS event loop is never blocked.

- `signal` (an `AbortSignal`) rejects the call with `signal.reason` and skips forward passes not yet started. A pass
  already running cannot be interrupted and finishes in the background; the embedder stays usable.
- Each forward pass has a deadline (`perBatchTimeoutSeconds`, else `timeoutSeconds`). When it expires the call rejects
  with `error.code === 'ERR_MINSYNC_TIMEOUT'`. A stuck native thread cannot be cancelled, so **every later call in the
  process fails fast with the same code; restart the process**. This is the CLI's behaviour too.
- Other codes: `ERR_MINSYNC_EMBEDDING` (load or inference failed, e.g. no network and no cached weights),
  `ERR_MINSYNC` (anything else).

### Cache and environment

The weights are resolved exactly as the CLI does, so both share one copy and can run side by side:
`modelCacheDir`, else `HF_HOME`, else `FASTEMBED_CACHE_DIR`, else `~/.cache/minsync/models` (`%LOCALAPPDATA%\minsync\models`
on Windows). `HF_ENDPOINT` and `HF_TOKEN` are honoured. They are read from the process environment when a load starts,
so set them (`process.env`) before the first call.

## Development

```bash
npm ci
npm run build                      # release build of the addon for this platform
node --test __test__/contract.test.mjs   # no model needed

# Real model (downloads ~1.2 GB into the cache on first run):
MINSYNC_LIVE=1 MINSYNC_TEST_DEVICE=cpu node --test --test-concurrency=1 __test__/

# Parity with the CLI embedder:
MINSYNC_TEST_DEVICE=cpu cargo run --release -p minsync --example embed_reference -- \
  __test__/fixtures/texts.json /tmp/reference.json
MINSYNC_TEST_DEVICE=cpu MINSYNC_REFERENCE_VECTORS=/tmp/reference.json node --test __test__/parity.test.mjs
```

`binding.js` and `binding.d.ts` are generated by the build; `index.js` / `index.d.ts` are the hand-written public
surface (they add `AbortSignal` support and `error.code`).
