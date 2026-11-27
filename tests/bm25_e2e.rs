//! Live CLI end-to-end tests for BM25 retrieval, the compressed on-disk BM25
//! inverted index (`bm25-index.inv`), multilingual tokenization (EN/KO/JA/ZH/AR),
//! low-spec zero-ML mode (`--embedder none --sparse none`), and simultaneous
//! Dense + SPLADE + BM25 multi-retriever fusion for LTR features.

mod common;

use assert_cmd::Command;
use minsync::config::Config;
use std::path::Path;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

fn minsync(root: &Path, cache: &Path) -> Command {
    let mut command = Command::cargo_bin("minsync").expect("find minsync binary");
    command
        .current_dir(root)
        .env_remove("OPENAI_API_KEY")
        .env("MINSYNC_SERVER", "off")
        .env("HF_HOME", cache)
        .env("HF_HUB_OFFLINE", "1")
        .env("INL_CACHE_DIR", cache);
    command
}

fn run_json(root: &Path, cache: &Path, args: &[&str]) -> serde_json::Value {
    let output = minsync(root, cache)
        .args(["--format", "json"])
        .args(args)
        .output()
        .expect("run minsync");
    assert!(
        output.status.success(),
        "`minsync {}` failed (code {:?}):\nstdout: {}\nstderr: {}",
        args.join(" "),
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "`minsync {}` printed invalid JSON ({error}): {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stdout)
        )
    })
}

fn set_bm25_index_options(root: &Path, enabled: bool, min_rows: i64) {
    let path = root.join(".minsync/config.toml");
    let mut config = Config::load(&path).expect("load config");
    let mut options = match std::mem::replace(
        &mut config.vectorstore.options,
        toml::Value::Table(Default::default()),
    ) {
        toml::Value::Table(table) => table,
        _ => Default::default(),
    };
    options.insert("bm25_inverted_index".into(), toml::Value::Boolean(enabled));
    options.insert("bm25_index_min_rows".into(), toml::Value::Integer(min_rows));
    config.vectorstore.options = toml::Value::Table(options);
    config.save(&path).expect("save config");
}

fn ranked(results: &serde_json::Value) -> Vec<(String, f64)> {
    results
        .as_array()
        .expect("results array")
        .iter()
        .map(|hit| {
            (
                hit["doc_id"].as_str().expect("doc_id").to_string(),
                hit["score"].as_f64().expect("score"),
            )
        })
        .collect()
}

#[test]
fn bm25_only_workspace_works_completely_offline_with_zero_ml_models() {
    let root = tempfile::tempdir().expect("workspace");
    let cache = tempfile::tempdir().expect("empty cache");

    std::fs::write(
        root.path().join("orchard.md"),
        "# Orchard\n\nThe quince orchard grows heritage pears and crisp apples.\n",
    )
    .expect("write orchard");
    std::fs::write(
        root.path().join("harbor.md"),
        "# Harbor\n\nTrawlers unload Atlantic mackerel and herring at dawn.\n",
    )
    .expect("write harbor");

    // Low-spec PC configuration: neither dense nor SPLADE model is loaded.
    let init = run_json(
        root.path(),
        cache.path(),
        &["init", "--embedder", "none", "--sparse", "none"],
    );
    assert_eq!(init["embedder"]["id"], "none");
    assert_eq!(init["sparse"]["id"], "none");
    assert_eq!(init["bm25"]["enabled"], true);
    assert_eq!(init["bm25"]["language"], "multilingual");
    assert_eq!(init["vectorstore"]["options"]["dimension"], 1);

    // Health check passes offline without touching any ML model.
    let check = run_json(root.path(), cache.path(), &["check"]);
    assert_eq!(check["all_passed"], true);
    assert_eq!(check["embedder_ok"], true);
    assert_eq!(check["sparse_ok"], true);
    assert_eq!(check["vectorstore_ok"], true);

    // Sync writes chunks with zero ML inference calls.
    let sync = run_json(root.path(), cache.path(), &["sync"]);
    assert_eq!(sync["files_processed"], 2);
    assert_eq!(sync["embedding_api_calls"], 0);
    assert_eq!(sync["embedded_texts"], 0);
    assert_eq!(sync["sparse_encoded"], 0);
    assert_eq!(sync["partial"], false);

    // Query with omitted `--mode` defaults to `bm25`.
    let results = run_json(root.path(), cache.path(), &["query", "mackerel herring"]);
    assert_eq!(results[0]["path"], "harbor.md");
    assert_eq!(results[0]["mode"], "bm25");
    assert_eq!(results[0]["bm25_rank"], 1);
    assert!(results[0]["bm25_score"].as_f64().unwrap() > 0.0);
    assert!(results[0].get("vector_rank").is_none());
    assert!(results[0].get("sparse_rank").is_none());
    assert_eq!(results[0]["index_state"]["ann_indexed"], false);

    // Any combination requiring dense or SPLADE fails cleanly with guidance.
    for args in [
        vec!["query", "mackerel", "--vector"],
        vec!["query", "mackerel", "--sparse"],
        vec!["query", "mackerel", "--mode", "hybrid"],
        vec!["query", "mackerel", "--vector", "--bm25"],
        vec!["query", "mackerel", "--sparse", "--bm25"],
        vec!["query", "mackerel", "--vector", "--sparse", "--bm25"],
    ] {
        let out = minsync(root.path(), cache.path())
            .args(&args)
            .output()
            .expect("run rejected mode");
        assert!(
            !out.status.success(),
            "args {args:?} must fail in BM25-only workspace"
        );
    }

    // Incremental update and deletion work cleanly.
    std::fs::write(
        root.path().join("harbor.md"),
        "# Harbor\n\nFerries depart for the lighthouse island.\n",
    )
    .expect("edit harbor");
    std::fs::remove_file(root.path().join("orchard.md")).expect("delete orchard");
    let sync2 = run_json(root.path(), cache.path(), &["sync"]);
    assert_eq!(sync2["files_modified"], 1);
    assert_eq!(sync2["files_deleted"], 1);

    let after_edit = run_json(root.path(), cache.path(), &["query", "lighthouse"]);
    assert_eq!(after_edit[0]["path"], "harbor.md");
    let gone = run_json(root.path(), cache.path(), &["query", "quince"]);
    assert_eq!(gone.as_array().unwrap().len(), 0);

    // Verify passes on the BM25-only workspace.
    let verify = run_json(root.path(), cache.path(), &["verify"]);
    assert_eq!(verify["all_passed"], true);
}

#[test]
fn bm25_multilingual_search_works_across_en_ko_ja_zh_ar_and_identifiers() {
    let root = tempfile::tempdir().expect("workspace");
    let cache = tempfile::tempdir().expect("empty cache");

    let docs = [
        (
            "en.md",
            "Distributed systems and deterministic Rust indexing engine.",
        ),
        (
            "ko.md",
            "환불정책과배송안내입니다 서울맛집추천 아버지가방에들어가신다",
        ),
        (
            "ja.md",
            "関西国際空港限定トートバッグの販売情報と東京スカイツリー展望台",
        ),
        ("zh.md", "我们中出了一个叛徒 分布式全文搜索引擎与倒排索引"),
        ("ar.md", "والكتاب في المدرسة العربية لتعليم القراءة"),
        (
            "code.md",
            "Handle ERR_CONNECTION_REFUSED in auth-service.v2 middleware",
        ),
    ];
    for (name, body) in docs {
        std::fs::write(root.path().join(name), format!("{body}\n")).expect("write doc");
    }

    run_json(
        root.path(),
        cache.path(),
        &[
            "init",
            "--embedder",
            "none",
            "--sparse",
            "none",
            "--language",
            "multilingual",
        ],
    );
    // Force building bm25-index.inv even on a 6-file corpus.
    set_bm25_index_options(root.path(), true, 1);
    run_json(root.path(), cache.path(), &["sync"]);

    assert!(
        root.path().join(".minsync/store/bm25-index.inv").exists(),
        "bm25-index.inv must be created beside documents.lance"
    );

    let cases = [
        ("deterministic Rust", "en.md"),
        ("환불 정책", "ko.md"),
        ("関西国際空港", "ja.md"),
        ("倒排索引", "zh.md"),
        ("كتاب", "ar.md"),
        ("ERR_CONNECTION_REFUSED", "code.md"),
    ];

    let mut via_index = Vec::new();
    for (query, expected_path) in cases {
        let res = run_json(
            root.path(),
            cache.path(),
            &["query", query, "--mode", "bm25"],
        );
        assert!(
            !res.as_array().unwrap().is_empty(),
            "expected hit for multilingual query {query:?}"
        );
        assert_eq!(
            res[0]["path"], expected_path,
            "wrong top hit for {query:?}: {res}"
        );
        assert_eq!(res[0]["index_state"]["bm25_indexed"], true);
        via_index.push(ranked(&res));
    }

    // Disable on-disk BM25 index and verify exact score parity with scan
    // (exhaustive per-language parity is also verified in-process in store_index_tests).
    set_bm25_index_options(root.path(), false, 1);
    for ((query, _), expected_ranking) in cases.iter().zip(&via_index).take(3) {
        let res = run_json(
            root.path(),
            cache.path(),
            &["query", query, "--mode", "bm25"],
        );
        assert_eq!(res[0]["index_state"]["bm25_indexed"], false);
        assert_eq!(
            ranked(&res),
            *expected_ranking,
            "bm25-index.inv vs scan mismatch for {query:?}"
        );
    }

    // Switching [bm25].language in config.toml triggers bm25-index.inv rebuild on next sync.
    set_bm25_index_options(root.path(), true, 1);
    let cfg_path = root.path().join(".minsync/config.toml");
    let mut cfg = Config::load(&cfg_path).unwrap();
    cfg.bm25.language = "ko".to_string();
    cfg.save(&cfg_path).unwrap();

    let resync = run_json(root.path(), cache.path(), &["sync"]);
    assert_eq!(
        resync["already_up_to_date"], false,
        "language change must trigger bm25 index rebuild"
    );
    let ko_hit = run_json(root.path(), cache.path(), &["query", "환불 정책"]);
    assert_eq!(ko_hit[0]["path"], "ko.md");
    assert_eq!(ko_hit[0]["index_state"]["bm25_indexed"], true);

    // Subsequent sync with no changes is up to date.
    let noop = run_json(root.path(), cache.path(), &["sync"]);
    assert_eq!(noop["already_up_to_date"], true);
}

#[test]
fn bm25_inverted_index_lifecycle_edit_delete_and_corruption_recovery() {
    let root = tempfile::tempdir().expect("workspace");
    let cache = tempfile::tempdir().expect("empty cache");

    // 70 files > default bm25_index_min_rows (64), so default config builds bm25-index.inv!
    for i in 0..70 {
        let topic = match i % 5 {
            0 => "quantum lattice cryptography",
            1 => "distributed consensus raft",
            2 => "한국어 형태소 분석기 검색",
            3 => "日本語形態素解析エンジン",
            _ => "中文分词与倒排索引优化",
        };
        std::fs::write(
            root.path().join(format!("doc-{i:03}.md")),
            format!("document {i} tag_{i} {topic}\n"),
        )
        .expect("write doc");
    }

    run_json(
        root.path(),
        cache.path(),
        &["init", "--embedder", "none", "--sparse", "none"],
    );
    run_json(root.path(), cache.path(), &["sync"]);

    let inv_path = root.path().join(".minsync/store/bm25-index.inv");
    assert!(
        inv_path.exists(),
        "bm25-index.inv built by default at >=64 rows"
    );

    let hit = run_json(root.path(), cache.path(), &["query", "tag_42 cryptography"]);
    assert_eq!(hit[0]["path"], "doc-042.md");
    assert_eq!(hit[0]["index_state"]["bm25_indexed"], true);

    // Corrupt bm25-index.inv -> query still returns the exact same answer via scan.
    let baseline = ranked(&hit);
    std::fs::write(&inv_path, b"corrupted bm25 bytes").expect("corrupt index");
    let fallback = run_json(root.path(), cache.path(), &["query", "tag_42 cryptography"]);
    assert_eq!(fallback[0]["index_state"]["bm25_indexed"], false);
    assert_eq!(ranked(&fallback), baseline);

    // `minsync optimize` rebuilds the damaged bm25-index.inv.
    run_json(root.path(), cache.path(), &["optimize"]);
    let repaired = run_json(root.path(), cache.path(), &["query", "tag_42 cryptography"]);
    assert_eq!(repaired[0]["index_state"]["bm25_indexed"], true);
    assert_eq!(ranked(&repaired), baseline);
}

/// Deterministic 4-dim TEI mock embedder so we can run real Dense + real SPLADE
/// (tiny_splade) + real BM25 (`bm25-index.inv`) simultaneously in one workspace.
struct DeterministicTei;

impl Respond for DeterministicTei {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: serde_json::Value = serde_json::from_slice(&request.body).expect("json body");
        let inputs: Vec<String> = match &body["inputs"] {
            serde_json::Value::Array(arr) => arr
                .iter()
                .map(|v| v.as_str().unwrap_or_default().to_string())
                .collect(),
            serde_json::Value::String(s) => vec![s.clone()],
            _ => Vec::new(),
        };
        let vectors: Vec<[f32; 4]> = inputs
            .iter()
            .map(|text| {
                if text.contains("semantic_alpha") {
                    [1.0, 0.0, 0.0, 0.0]
                } else if text.contains("w1") {
                    [0.7, 0.7, 0.0, 0.0]
                } else if text.contains("환불") {
                    [0.0, 1.0, 0.0, 0.0]
                } else {
                    [0.0, 0.0, 1.0, 0.0]
                }
            })
            .collect();
        ResponseTemplate::new(200).set_body_json(vectors)
    }
}

#[tokio::test]
async fn all_three_retrievers_dense_splade_and_bm25_run_simultaneously_for_ltr() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(DeterministicTei)
        .mount(&server)
        .await;

    let root = tempfile::tempdir().expect("workspace");
    let cache = tempfile::tempdir().expect("empty cache");

    std::fs::write(
        root.path().join("alpha.md"),
        "semantic_alpha w1 w2 w3 exact_token_alpha 환불 정책\n",
    )
    .expect("write alpha");
    std::fs::write(
        root.path().join("beta.md"),
        "w1 w2 w3 w4 w5 exact_token_beta 배송 안내\n",
    )
    .expect("write beta");
    std::fs::write(
        root.path().join("gamma.md"),
        "w20 w21 w22 semantic_gamma 환불 정책 안내\n",
    )
    .expect("write gamma");

    // Initialize with TEI dense embedder + tiny native SPLADE + multilingual BM25.
    run_json(
        root.path(),
        cache.path(),
        &["init", "--embedder", "tei:mock/dense"],
    );
    let cfg_path = root.path().join(".minsync/config.toml");
    let mut cfg = Config::load(&cfg_path).unwrap();
    cfg.embedder.base_url = Some(server.uri());
    common::use_tiny_sparse(&mut cfg);
    let mut options = toml::value::Table::new();
    options.insert("dimension".into(), toml::Value::Integer(4));
    options.insert("sparse_index_min_rows".into(), toml::Value::Integer(1));
    options.insert("bm25_index_min_rows".into(), toml::Value::Integer(1));
    cfg.vectorstore.options = toml::Value::Table(options);
    cfg.save(&cfg_path).unwrap();

    let sync = run_json(root.path(), cache.path(), &["sync"]);
    assert_eq!(sync["files_processed"], 3);
    assert_eq!(sync["chunks_added"], 3);
    assert_eq!(sync["sparse_encoded"], 3);
    assert_eq!(sync["partial"], false);

    // BOTH sidecar inverted indexes exist simultaneously beside documents.lance!
    assert!(root.path().join(".minsync/store/sparse-index.inv").exists());
    assert!(root.path().join(".minsync/store/bm25-index.inv").exists());

    let query = "semantic_alpha w1 w2 환불";

    // 1. vector only
    let vec_res = run_json(
        root.path(),
        cache.path(),
        &["query", query, "--mode", "vector"],
    );
    assert_eq!(vec_res[0]["mode"], "vector");
    assert_eq!(vec_res[0]["path"], "alpha.md");
    assert_eq!(vec_res[0]["vector_rank"], 1);
    assert!(vec_res[0]["vector_score"].is_number());
    assert!(vec_res[0].get("sparse_rank").is_none());
    assert!(vec_res[0].get("bm25_rank").is_none());
    assert_eq!(vec_res[0]["index_state"]["sparse_indexed"], true);
    assert_eq!(vec_res[0]["index_state"]["bm25_indexed"], true);

    // 2. sparse only
    let sp_res = run_json(
        root.path(),
        cache.path(),
        &["query", query, "--mode", "sparse"],
    );
    assert_eq!(sp_res[0]["mode"], "sparse");
    assert!(sp_res[0]["sparse_rank"].is_number());
    assert!(sp_res[0]["sparse_score"].is_number());
    assert!(sp_res[0].get("vector_rank").is_none());
    assert!(sp_res[0].get("bm25_rank").is_none());

    // 3. bm25 only
    let bm_res = run_json(
        root.path(),
        cache.path(),
        &["query", query, "--mode", "bm25"],
    );
    assert_eq!(bm_res[0]["mode"], "bm25");
    assert_eq!(bm_res[0]["path"], "alpha.md");
    assert_eq!(bm_res[0]["bm25_rank"], 1);
    assert!(bm_res[0]["bm25_score"].as_f64().unwrap() > 0.0);
    assert!(bm_res[0].get("vector_rank").is_none());
    assert!(bm_res[0].get("sparse_rank").is_none());

    // 4. hybrid (vector + sparse)
    let hyb_res = run_json(
        root.path(),
        cache.path(),
        &["query", query, "--mode", "hybrid"],
    );
    assert_eq!(hyb_res[0]["mode"], "hybrid");
    assert!(hyb_res[0]["vector_rank"].is_number());
    assert!(hyb_res[0]["sparse_rank"].is_number());
    assert!(hyb_res[0].get("bm25_rank").is_none());

    // 5. --vector --bm25 (or --dense --bm25)
    let vb_res = run_json(
        root.path(),
        cache.path(),
        &["query", query, "--dense", "--bm25"],
    );
    assert_eq!(vb_res[0]["mode"], "vector+bm25");
    assert_eq!(vb_res[0]["path"], "alpha.md");
    assert!(vb_res[0]["vector_rank"].is_number());
    assert!(vb_res[0]["bm25_rank"].is_number());
    assert!(vb_res[0].get("sparse_rank").is_none());

    // 6. --sparse --bm25 (and negative toggle --vector=false gives the same set!)
    let sb_res = run_json(
        root.path(),
        cache.path(),
        &["query", query, "--sparse", "--bm25"],
    );
    assert_eq!(sb_res[0]["mode"], "sparse+bm25");
    assert!(sb_res[0]["sparse_rank"].is_number());
    assert!(sb_res[0]["bm25_rank"].is_number());
    assert!(sb_res[0].get("vector_rank").is_none());

    let neg_res = run_json(
        root.path(),
        cache.path(),
        &["query", query, "--vector=false"],
    );
    assert_eq!(neg_res[0]["mode"], "sparse+bm25");
    assert_eq!(ranked(&neg_res), ranked(&sb_res));

    // 7. --vector --sparse --bm25: full multi-feature LTR payload!
    let all_res = run_json(
        root.path(),
        cache.path(),
        &["query", query, "--vector", "--sparse", "--bm25"],
    );
    assert_eq!(all_res[0]["mode"], "vector+sparse+bm25");
    assert_eq!(all_res[0]["path"], "alpha.md");
    assert!(all_res[0]["vector_rank"].is_number());
    assert!(all_res[0]["vector_score"].is_number());
    assert!(all_res[0]["sparse_rank"].is_number());
    assert!(all_res[0]["sparse_score"].is_number());
    assert!(all_res[0]["bm25_rank"].is_number());
    assert!(all_res[0]["bm25_score"].is_number());
    assert_eq!(all_res[0]["index_state"]["sparse_indexed"], true);
    assert_eq!(all_res[0]["index_state"]["bm25_indexed"], true);

    // Even if the dense server goes down, `--bm25` and `--sparse --bm25`
    // keep answering because they never touch the dense embedder!
    drop(server);
    let still_ok = run_json(
        root.path(),
        cache.path(),
        &["query", "exact_token_beta", "--bm25"],
    );
    assert_eq!(still_ok[0]["path"], "beta.md");
}

#[test]
fn hardware_auto_selects_safe_retriever_plan_for_low_mid_and_high_spec_machines() {
    let gib: u64 = 1024 * 1024 * 1024;
    let cases = [
        (
            "8GB Windows laptop (CPU-only)",
            serde_json::json!({"os":"windows","arch":"x86_64","cores":4,"total_ram_bytes":8*gib,"cuda_gpus":[]}),
            "none",
            "none",
        ),
        (
            "8GB Apple Silicon Mac",
            serde_json::json!({"os":"macos","arch":"aarch64","cores":8,"total_ram_bytes":8*gib,"cuda_gpus":[]}),
            "none",
            "none",
        ),
        (
            "16GB Apple Silicon Mac",
            serde_json::json!({"os":"macos","arch":"aarch64","cores":8,"total_ram_bytes":16*gib,"cuda_gpus":[]}),
            "none",
            minsync::config::DEFAULT_SPARSE_ID,
        ),
        (
            "16GB Windows laptop (4 cores, no CUDA)",
            serde_json::json!({"os":"windows","arch":"x86_64","cores":4,"total_ram_bytes":16*gib,"cuda_gpus":[]}),
            "none",
            minsync::config::DEFAULT_SPARSE_ID,
        ),
        (
            "24GB Apple Silicon Mac",
            serde_json::json!({"os":"macos","arch":"aarch64","cores":8,"total_ram_bytes":24*gib,"cuda_gpus":[]}),
            minsync::config::DEFAULT_EMBEDDER_ID,
            minsync::config::DEFAULT_SPARSE_ID,
        ),
        (
            "16GB Windows PC with RTX 4060 8GB",
            serde_json::json!({"os":"windows","arch":"x86_64","cores":8,"total_ram_bytes":16*gib,"cuda_gpus":[{"name":"NVIDIA GeForce RTX 4060","vram_mib":8188}]}),
            minsync::config::DEFAULT_EMBEDDER_ID,
            minsync::config::DEFAULT_SPARSE_ID,
        ),
    ];

    let cache = tempfile::tempdir().expect("empty cache");
    for (label, hw_json, expected_embedder, expected_sparse) in cases {
        let root = tempfile::tempdir().expect("workspace");
        let output = minsync(root.path(), cache.path())
            .env("MINSYNC_HARDWARE_OVERRIDE", hw_json.to_string())
            .args(["--format", "json", "init"])
            .output()
            .expect("run init");
        assert!(
            output.status.success(),
            "{label} init failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let cfg: serde_json::Value = serde_json::from_slice(&output.stdout).expect("json");
        assert_eq!(
            cfg["embedder"]["id"], expected_embedder,
            "embedder for {label}"
        );
        assert_eq!(cfg["sparse"]["id"], expected_sparse, "sparse for {label}");
        assert_eq!(cfg["bm25"]["enabled"], true, "bm25 for {label}");
    }

    // End-to-end on simulated 8GB Windows laptop: `minsync init` (no flags!) ->
    // `minsync sync` -> `minsync query` works completely offline with zero ML!
    let root = tempfile::tempdir().expect("8gb workspace");
    std::fs::write(root.path().join("note.md"), "환불정책과배송안내입니다\n").unwrap();
    let low_hw = serde_json::json!({"os":"windows","arch":"x86_64","cores":4,"total_ram_bytes":8*gib,"cuda_gpus":[]}).to_string();
    let init_out = minsync(root.path(), cache.path())
        .env("MINSYNC_HARDWARE_OVERRIDE", &low_hw)
        .args(["init"])
        .output()
        .unwrap();
    assert!(init_out.status.success());
    let stdout = String::from_utf8_lossy(&init_out.stdout);
    assert!(
        stdout.contains("bm25-only"),
        "expected rationale in {stdout}"
    );

    let sync_res = run_json(root.path(), cache.path(), &["sync"]);
    assert_eq!(sync_res["embedding_api_calls"], 0);
    assert_eq!(sync_res["sparse_encoded"], 0);
    let hits = run_json(root.path(), cache.path(), &["query", "환불 정책"]);
    assert_eq!(hits[0]["path"], "note.md");
    assert_eq!(hits[0]["mode"], "bm25");
}
