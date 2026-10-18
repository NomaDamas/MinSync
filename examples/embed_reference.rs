//! Reference vectors for the Node binding's parity test.
//!
//! Embeds the passages and queries of a fixture through
//! `minsync::embedder::create_embedder` — the exact path `minsync sync` and
//! `minsync query` take for a `native:` embedder — and writes every vector as
//! raw `f32` bit patterns, so the JS side can assert bit equality.
//!
//! ```text
//! MINSYNC_TEST_DEVICE=cpu cargo run --release --example embed_reference -- \
//!     bindings/node/__test__/fixtures/texts.json /tmp/reference.json
//! ```

use minsync::config::{Config, DEFAULT_EMBEDDER_ID};
use minsync::embedder::create_embedder;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Fixture {
    query_prefix: Option<String>,
    passage_prefix: Option<String>,
    passages: Vec<String>,
    queries: Vec<String>,
}

fn bits(vector: &[f32]) -> Vec<u32> {
    vector.iter().map(|value| value.to_bits()).collect()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let (Some(fixture_path), Some(out_path)) = (args.next(), args.next()) else {
        anyhow::bail!("usage: embed_reference <fixture.json> <out.json>");
    };
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(fixture_path)?)?;

    let mut config = Config::default_for("reference");
    config.embedder.id = DEFAULT_EMBEDDER_ID.to_string();
    config.embedder.device = std::env::var("MINSYNC_TEST_DEVICE").ok();
    config.embedder.query_prefix = fixture.query_prefix;
    config.embedder.passage_prefix = fixture.passage_prefix;
    let embedder = create_embedder(&config)?;

    let passages = embedder.embed(&fixture.passages).await?;
    let mut queries = Vec::new();
    for query in &fixture.queries {
        queries.push(embedder.embed_query(query).await?);
    }

    let out = serde_json::json!({
        "passages": passages.iter().map(|v| bits(v)).collect::<Vec<_>>(),
        "queries": queries.iter().map(|v| bits(v)).collect::<Vec<_>>(),
    });
    std::fs::write(out_path, serde_json::to_vec(&out)?)?;
    Ok(())
}
