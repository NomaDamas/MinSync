//! Helpers shared by the CLI end-to-end tests.
//!
//! Every test that runs `minsync sync` or a sparse query must not reach for
//! the real `nlpai-lab/KURE-Sparse-nano` weights (a ~100 MB download): CI is
//! offline and the tests should stay fast. They point `[sparse]` at the tiny
//! checked-in ModernBERT SPLADE checkpoint instead, which goes through the
//! same native code path (tokenizer, model, pooling) with random weights.
#![allow(dead_code)]

use minsync::config::Config;
use std::path::{Path, PathBuf};

/// Directory of the tiny checkpoint (`tests/fixtures/tiny_splade`).
pub fn tiny_model_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_splade")
}

/// The `[sparse]` id of the tiny checkpoint.
pub fn tiny_sparse_id() -> String {
    format!("native:{}", tiny_model_dir().display())
}

/// Point `config`'s sparse encoder at the tiny checkpoint, on CPU in f32.
pub fn use_tiny_sparse(config: &mut Config) {
    config.sparse.id = tiny_sparse_id();
    config.sparse.device = Some("cpu".to_string());
    config.sparse.dtype = Some("f32".to_string());
}

/// Load `<root>/.minsync/config.toml`, point it at the tiny checkpoint, save.
pub fn use_tiny_sparse_in(root: &Path) {
    let path = root.join(".minsync/config.toml");
    let mut config = Config::load(&path).expect("load config");
    use_tiny_sparse(&mut config);
    config.save(&path).expect("save config");
}

/// Encode passages (or, with `query`, search queries) in-process with the
/// tiny checkpoint, exactly as `minsync sync` / `minsync query` would.
pub fn encode_with_tiny(texts: &[&str], query: bool) -> Vec<minsync::sparse::SparseVector> {
    use minsync::sparse::SparseEncoder;
    let mut config = Config::default_for("oracle");
    use_tiny_sparse(&mut config);
    let encoder = minsync::sparse::native::NativeSparseEncoder::from_config(&config.sparse)
        .expect("build tiny encoder");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    runtime.block_on(async {
        if query {
            let mut out = Vec::new();
            for text in texts {
                out.push(encoder.encode_query(text).await.expect("encode query"));
            }
            out
        } else {
            let owned: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
            encoder.encode(&owned).await.expect("encode passages")
        }
    })
}

/// Brute-force ranking: indices of `docs` by descending inner product with
/// `query`, ties toward the earlier index, documents sharing no term dropped.
pub fn brute_force_ranking(
    docs: &[minsync::sparse::SparseVector],
    query: &minsync::sparse::SparseVector,
) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = docs
        .iter()
        .enumerate()
        .map(|(i, doc)| (i, doc.dot(query)))
        .filter(|(_, score)| *score > 0.0)
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored.into_iter().map(|(i, _)| i).collect()
}

/// The model id the native encoder stores in rows and the cursor.
pub fn tiny_model_id() -> String {
    tiny_model_dir().display().to_string()
}
