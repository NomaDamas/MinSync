# Third-party notices

MinSync is MIT licensed (see `LICENSE`). The files listed here contain code
derived from other projects under their own licenses, and the dependencies
below are linked into the `minsync` executable under their own licenses.

## im-not-llm: model inference

All in-process model inference (the Qwen3-Embedding and EmbeddingGemma 2 text
towers and the ModernBERT SPLADE encoder, on candle) is the
[im-not-llm](https://github.com/NomaDamas/im-not-llm) crates `inl-core`,
`inl-text`, `inl-sparse` and `inl-server`, linked into the `minsync`
executable. MinSync contains no model code of its own.

- License: MIT OR Apache-2.0
- Their own third-party notices (including the fastembed-rs-derived Qwen3 code,
  Apache-2.0) are in that repository's `THIRD_PARTY_NOTICES.md`.
- Library: [candle](https://github.com/huggingface/candle) (`candle-core`,
  `candle-nn`), Apache License, Version 2.0 or MIT.

## google/embeddinggemma-2: EmbeddingGemma 2 text tower

The weights are **not** part of MinSync: the `native:` embedder downloads them
from Hugging Face on first use.

- Model: `google/embeddinggemma-2` (270M-parameter text tower), Apache
  License, Version 2.0 (<http://www.apache.org/licenses/LICENSE-2.0>)

## nlpai-lab/KURE-Sparse-nano: learned sparse (SPLADE) model

Sparse retrieval uses the
[KURE-Sparse-nano](https://huggingface.co/nlpai-lab/KURE-Sparse-nano) model as
its default. The weights are **not** part of MinSync: the `native:` encoder
downloads them from Hugging Face on first use, and a `tei:` endpoint or
`minsync serve --sparse` serves them separately.

- Model: `nlpai-lab/KURE-Sparse-nano` (0.1B parameters)
- License: Apache License, Version 2.0 (<http://www.apache.org/licenses/LICENSE-2.0>)

## Multilingual BM25 tokenizers (`lindera`, `jieba-rs`)

BM25 tokenization uses pure-Rust crates with embedded dictionaries (no external native shared libraries required at runtime):

- [`lindera`](https://github.com/lindera/lindera) 3.0.7 (`embed-ipadic`, `embed-ko-dic`): MIT License. Embedded IPADIC (Japanese) and mecab-ko-dic (Korean) dictionaries carry their respective open-source dictionary licenses (IPADIC / Apache-2.0).
- [`jieba-rs`](https://github.com/messense/jieba-rs) 0.10.x (Chinese word segmentation): MIT License.
