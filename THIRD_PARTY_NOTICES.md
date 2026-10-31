# Third-party notices

MinSync is MIT licensed (see `LICENSE`). The files listed here contain code
derived from other projects under their own licenses, and the dependencies
below are linked into the `minsync` executable under their own licenses.

## fastembed-rs: Qwen3 model code

`src/embedder/qwen3/` is derived from `src/models/qwen3.rs` of
[fastembed-rs](https://github.com/Anush008/fastembed-rs) 7.0.1.

- Copyright 2026 Anush008 and the fastembed-rs contributors
- License: Apache License, Version 2.0 (<http://www.apache.org/licenses/LICENSE-2.0>)

Changes made in MinSync: the text-embedding path only (no vision-language
code); per-device attention backends; candle's fused RMSNorm; host-side padding
mask and rotary tables; pooling and normalization on the last token in f32.

## candle: ModernBERT sparse encoder

`src/sparse/modernbert.rs` implements the ModernBERT masked-LM with
`ShiftedTopK` SPLADE pooling that the default sparse checkpoint family ships.
It is written against that checkpoint family rather than reusing
`candle_transformers::models::modernbert`, which is not a dependency of this
crate, applies no padding mask to the sliding window of padded keys, and has no
MLM-logit chunking. Weight names and numerics follow the Hugging Face
reference; the golden test reproduces its output.

- Library: [candle](https://github.com/huggingface/candle) (`candle-core`,
  `candle-nn`)
- License: Apache License, Version 2.0 or MIT
  (<http://www.apache.org/licenses/LICENSE-2.0> or
  <https://opensource.org/licenses/MIT>)

## nlpai-lab/KURE-Sparse-nano: learned sparse (SPLADE) model

Sparse retrieval uses the
[KURE-Sparse-nano](https://huggingface.co/nlpai-lab/KURE-Sparse-nano) model as
its default. The weights are **not** part of MinSync: the `native:` encoder
downloads them from Hugging Face on first use, and a `tei:` endpoint or
`minsync serve --sparse` serves them separately.

- Model: `nlpai-lab/KURE-Sparse-nano` (0.1B parameters)
- License: Apache License, Version 2.0 (<http://www.apache.org/licenses/LICENSE-2.0>)
