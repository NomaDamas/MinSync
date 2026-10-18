# Third-party notices

MinSync is MIT licensed (see `LICENSE`). The files listed here contain code
derived from other projects under their own licenses.

## fastembed-rs: Qwen3 model code

`src/embedder/qwen3/` is derived from `src/models/qwen3.rs` of
[fastembed-rs](https://github.com/Anush008/fastembed-rs) 7.0.1.

- Copyright 2026 Anush008 and the fastembed-rs contributors
- License: Apache License, Version 2.0 (<http://www.apache.org/licenses/LICENSE-2.0>)

Changes made in MinSync: the text-embedding path only (no vision-language
code); per-device attention backends; candle's fused RMSNorm; host-side padding
mask and rotary tables; pooling and normalization on the last token in f32.
