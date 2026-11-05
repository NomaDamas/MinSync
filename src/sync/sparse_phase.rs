//! The sparse (SPLADE) phase of a sync: give every stored row a sparse vector
//! from the current model.
//!
//! The phase is driven by the store, not by the file changes of the run. It
//! asks for rows that hold no vector from `encoder.id()`, so one loop covers
//! a freshly indexed chunk, a row that predates sparse retrieval, a row left
//! over from an interrupted run and every row after a model change. Nothing
//! about it needs a journal: each batch is one idempotent commit, and an
//! interrupted phase simply finds the rest of the rows next time.

use crate::error::{MinSyncError, Result};
use crate::sparse::{SparseEncoder, SparseVector};
use crate::types::{FileFailure, FileStage};
use crate::vectorstore::{Filter, SparseTarget, VectorStore};
use std::collections::{HashMap, HashSet};

/// Rows read, encoded and committed per iteration. One commit per iteration,
/// so this trades commit count against memory held for chunk text.
pub(super) const ROWS_PER_BATCH: usize = 512;

/// Iterations between store checkpoints (compaction + pruning). Each commit
/// adds a table version and a fragment, so a long phase would otherwise leave
/// hundreds of them behind.
const CHECKPOINT_EVERY_BATCHES: usize = 16;

/// Chunks that fail in a row (any success resets it) after which the failure
/// is systemic (encoder down, model gone) and the run aborts instead of
/// skipping chunk after chunk.
const SYSTEMIC_STREAK: usize = 3;

#[derive(Debug, Default)]
pub(super) struct Outcome {
    /// Rows that got a vector from the encoder.
    pub encoded: usize,
    /// Rows that got a copy of the vector of an identical chunk.
    pub reused: usize,
    /// One entry per file that had a chunk the encoder rejected. The rows stay
    /// without a vector, so the next sync tries them again.
    pub failures: Vec<FileFailure>,
}

/// The sparse phase of one sync run, run in steps: after every dense batch
/// for the rows that batch wrote (so a committed batch is searchable before
/// the next one starts), and once more over the whole store at the end.
/// Steps share the run's failures, so a chunk the encoder rejected is
/// reported once and never retried within the run.
#[derive(Debug, Default)]
pub(super) struct CatchUp {
    outcome: Outcome,
    failed_ids: HashSet<String>,
    failed_paths: HashSet<String>,
    streak: usize,
    batches: usize,
}

impl CatchUp {
    /// Encode every row of `store` in `scope` (all rows when `None`) that
    /// lacks a vector from `encoder`'s model.
    ///
    /// `on_failure` is told about each failed file as soon as it is known; an
    /// `Err` from it stops the phase (the `on_file_error = "fail"` policy).
    /// Returns `Err` when the phase cannot go on (stuck encoder, systemic
    /// failure, store error); chunk-level failures are kept in the outcome.
    pub(super) async fn run(
        &mut self,
        store: &mut dyn VectorStore,
        encoder: &dyn SparseEncoder,
        scope: Option<&Filter>,
        rows_per_batch: usize,
        mut on_failure: impl FnMut(&FileFailure) -> Result<()>,
    ) -> Result<()> {
        let model = encoder.id().to_string();
        let rows_per_batch = rows_per_batch.max(1);
        loop {
            // Rows that already failed this run still look "missing"; ask
            // for enough extra rows to get past them.
            let wanted = rows_per_batch + self.failed_ids.len();
            let rows: Vec<SparseTarget> = store
                .rows_missing_sparse(&model, scope, wanted)?
                .into_iter()
                .filter(|row| !self.failed_ids.contains(&row.id))
                .take(rows_per_batch)
                .collect();
            if rows.is_empty() {
                return Ok(());
            }
            self.encode_rows(store, encoder, &model, &rows, &mut on_failure)
                .await?;
        }
    }

    /// Everything the steps of this run did.
    pub(super) fn outcome(self) -> Outcome {
        self.outcome
    }

    async fn encode_rows(
        &mut self,
        store: &mut dyn VectorStore,
        encoder: &dyn SparseEncoder,
        model: &str,
        rows: &[SparseTarget],
        on_failure: &mut impl FnMut(&FileFailure) -> Result<()>,
    ) -> Result<()> {
        let outcome = &mut self.outcome;
        let failed_ids = &mut self.failed_ids;
        let failed_paths = &mut self.failed_paths;
        let streak = &mut self.streak;

        // Identical text encodes identically: one vector per content hash,
        // taken from the store when another row already holds it.
        let mut unique: Vec<&SparseTarget> = Vec::new();
        let mut seen: HashSet<&str> = HashSet::new();
        for row in rows {
            if seen.insert(row.content_hash.as_str()) {
                unique.push(row);
            }
        }
        let hashes: Vec<String> = unique.iter().map(|row| row.content_hash.clone()).collect();
        let mut vectors: HashMap<String, std::result::Result<SparseVector, MinSyncError>> = store
            .fetch_sparse_by_content_hash(model, &hashes)?
            .into_iter()
            .map(|(hash, vector)| (hash, Ok(vector)))
            .collect();
        let to_encode: Vec<&SparseTarget> = unique
            .into_iter()
            .filter(|row| !vectors.contains_key(&row.content_hash))
            .collect();

        if !to_encode.is_empty() {
            let texts: Vec<String> = to_encode.iter().map(|row| row.text.clone()).collect();
            let encoded = encode_isolating(encoder, &texts, streak).await?;
            for (row, result) in to_encode.iter().zip(encoded) {
                vectors.insert(row.content_hash.clone(), result);
            }
        }

        let mut commit: Vec<(String, SparseVector)> = Vec::with_capacity(rows.len());
        let mut from_encoder: HashSet<&str> = HashSet::new();
        for row in rows {
            match vectors.get(&row.content_hash) {
                Some(Ok(vector)) => {
                    commit.push((row.id.clone(), vector.clone()));
                    // The first row of a hash that was actually encoded
                    // counts as encoded; its duplicates and store hits as
                    // reused.
                    let was_encoded = to_encode
                        .iter()
                        .any(|target| target.content_hash == row.content_hash);
                    if was_encoded && from_encoder.insert(row.content_hash.as_str()) {
                        outcome.encoded += 1;
                    } else {
                        outcome.reused += 1;
                    }
                }
                Some(Err(error)) => {
                    failed_ids.insert(row.id.clone());
                    if failed_paths.insert(row.path.clone()) {
                        let failure = FileFailure {
                            path: row.path.clone(),
                            stage: FileStage::Sparse,
                            error: error.to_string(),
                        };
                        on_failure(&failure)?;
                        outcome.failures.push(failure);
                    }
                }
                None => {
                    return Err(MinSyncError::Embedding(format!(
                        "missing sparse vector for {} ({})",
                        row.path, row.id
                    )));
                }
            }
        }
        if !commit.is_empty() {
            store.set_sparse(model, &commit)?;
        }

        self.batches += 1;
        if self.batches.is_multiple_of(CHECKPOINT_EVERY_BATCHES) {
            store.checkpoint()?;
        }
        tracing::info!(
            encoded = self.outcome.encoded,
            reused = self.outcome.reused,
            failed = self.failed_ids.len(),
            "sparse encoding progress"
        );
        Ok(())
    }
}

/// Encode `texts`, narrowing a failing call down to the chunks that cause it
/// (halving, like the dense path). A chunk that fails alone is returned as an
/// error in its slot; a deadline, a wrong answer count, or
/// [`SYSTEMIC_STREAK`] failed chunks in a row abort the whole phase.
async fn encode_isolating(
    encoder: &dyn SparseEncoder,
    texts: &[String],
    streak: &mut usize,
) -> Result<Vec<std::result::Result<SparseVector, MinSyncError>>> {
    let mut out: Vec<Option<std::result::Result<SparseVector, MinSyncError>>> =
        (0..texts.len()).map(|_| None).collect();
    let mut stack = Vec::new();
    stack.push(0..texts.len());
    while let Some(range) = stack.pop() {
        match encoder.encode(&texts[range.clone()]).await {
            Ok(vectors) if vectors.len() == range.len() => {
                *streak = 0;
                for (slot, vector) in out[range.clone()].iter_mut().zip(vectors) {
                    *slot = Some(Ok(vector));
                }
            }
            Ok(vectors) => {
                return Err(MinSyncError::Embedding(format!(
                    "expected {} sparse vectors, got {}",
                    range.len(),
                    vectors.len()
                )));
            }
            Err(error @ MinSyncError::Timeout(_)) => return Err(error),
            Err(error) if range.len() > 1 => {
                let middle = range.start + range.len() / 2;
                stack.push(middle..range.end);
                stack.push(range.start..middle);
                drop(error);
            }
            Err(error) => {
                *streak += 1;
                if *streak >= SYSTEMIC_STREAK {
                    return Err(error);
                }
                out[range.start] = Some(Err(error));
            }
        }
    }
    Ok(out
        .into_iter()
        .map(|slot| slot.expect("every text is answered or aborts"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vectorstore::memory::InMemoryStore;
    use crate::vectorstore::Document;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    /// One whole-store step, as the end of a sync runs it.
    async fn catch_up(
        store: &mut dyn VectorStore,
        encoder: &dyn SparseEncoder,
        rows_per_batch: usize,
        on_failure: impl FnMut(&FileFailure) -> Result<()>,
    ) -> Result<Outcome> {
        let mut phase = CatchUp::default();
        phase
            .run(store, encoder, None, rows_per_batch, on_failure)
            .await?;
        Ok(phase.outcome())
    }

    #[tokio::test]
    async fn a_scoped_step_encodes_only_its_rows_and_a_rejected_chunk_is_tried_once_per_run() {
        let mut store = store_with(&[
            doc("a", "a.txt", "term-1"),
            doc("b", "b.txt", "bad"),
            doc("c", "c.txt", "term-3"),
        ]);
        let mut encoder = Fake::new("m");
        encoder.reject = vec!["bad"];
        let mut phase = CatchUp::default();
        let mut reported = 0;

        let batch_one = Filter::In("path".into(), vec!["a.txt".into(), "b.txt".into()]);
        phase
            .run(&mut store, &encoder, Some(&batch_one), 10, |_| {
                reported += 1;
                Ok(())
            })
            .await
            .unwrap();
        assert_eq!(
            store.count_missing_sparse("m").unwrap(),
            2,
            "b failed, c out of scope"
        );
        let attempts_on_b = |encoder: &Fake| {
            encoder
                .encoded_texts()
                .iter()
                .filter(|text| text.as_str() == "bad")
                .count()
        };
        let tried = attempts_on_b(&encoder);

        phase
            .run(&mut store, &encoder, None, 10, |_| {
                reported += 1;
                Ok(())
            })
            .await
            .unwrap();

        assert_eq!(
            attempts_on_b(&encoder),
            tried,
            "b is not retried in the same run"
        );
        assert_eq!(reported, 1, "b is reported once");
        let outcome = phase.outcome();
        assert_eq!(outcome.encoded, 2);
        assert_eq!(outcome.failures.len(), 1);
        assert_eq!(store.count_missing_sparse("m").unwrap(), 1);
    }

    /// Encodes `"term-N"` as the single term N with weight 1; fails on
    /// request for the texts it is told to reject; records every call.
    struct Fake {
        model: &'static str,
        reject: Vec<&'static str>,
        calls: Mutex<Vec<Vec<String>>>,
        fail_all: bool,
        encode_calls: AtomicUsize,
    }

    impl Fake {
        fn new(model: &'static str) -> Self {
            Self {
                model,
                reject: Vec::new(),
                calls: Mutex::new(Vec::new()),
                fail_all: false,
                encode_calls: AtomicUsize::new(0),
            }
        }

        fn encoded_texts(&self) -> Vec<String> {
            self.calls
                .lock()
                .unwrap()
                .iter()
                .flatten()
                .cloned()
                .collect()
        }
    }

    #[async_trait]
    impl SparseEncoder for Fake {
        fn id(&self) -> &str {
            self.model
        }

        async fn encode(&self, texts: &[String]) -> Result<Vec<SparseVector>> {
            self.encode_calls.fetch_add(1, Ordering::SeqCst);
            self.calls.lock().unwrap().push(texts.to_vec());
            if self.fail_all {
                return Err(MinSyncError::Embedding("encoder is down".into()));
            }
            if let Some(bad) = texts.iter().find(|t| self.reject.contains(&t.as_str())) {
                return Err(MinSyncError::Embedding(format!("cannot encode {bad}")));
            }
            Ok(texts
                .iter()
                .map(|text| {
                    let term = text
                        .strip_prefix("term-")
                        .and_then(|n| n.parse::<u32>().ok())
                        .unwrap_or(0);
                    if term == 0 {
                        SparseVector::default()
                    } else {
                        SparseVector::from_pairs([(term, 1.0)])
                    }
                })
                .collect())
        }

        async fn encode_query(&self, text: &str) -> Result<SparseVector> {
            Ok(self.encode(&[text.to_string()]).await?.remove(0))
        }
    }

    fn doc(id: &str, path: &str, text: &str) -> Document {
        Document {
            id: id.to_string(),
            embedding: vec![0.0],
            text: text.to_string(),
            source_id: "s".to_string(),
            path: path.to_string(),
            chunk_schema_id: "recursive".to_string(),
            chunk_type: "text".to_string(),
            heading_path: String::new(),
            // Identical text must share a hash, like the real chunk hash.
            content_hash: format!("hash:{text}"),
            seen_token: "t".to_string(),
        }
    }

    fn store_with(docs: &[Document]) -> InMemoryStore {
        let mut store = InMemoryStore::new();
        store.upsert(docs).unwrap();
        store
    }

    #[tokio::test]
    async fn encodes_every_row_and_leaves_none_missing() {
        let mut store = store_with(&[
            doc("a", "a.txt", "term-1"),
            doc("b", "b.txt", "term-2"),
            doc("c", "c.txt", "term-3"),
        ]);
        let encoder = Fake::new("m");

        let outcome = catch_up(&mut store, &encoder, 2, |_| Ok(())).await.unwrap();

        assert_eq!((outcome.encoded, outcome.reused), (3, 0));
        assert!(outcome.failures.is_empty());
        assert_eq!(store.count_missing_sparse("m").unwrap(), 0);
        let hits = store
            .query_sparse("m", &SparseVector::from_pairs([(2, 1.0)]), None, 5)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "b");
    }

    #[tokio::test]
    async fn identical_text_is_encoded_once_even_across_batches() {
        let mut store = store_with(&[
            doc("a", "a.txt", "term-7"),
            doc("b", "b.txt", "term-7"),
            doc("c", "c.txt", "term-7"),
        ]);
        let encoder = Fake::new("m");

        // Batch size 1: the copies sit in later batches and must come from
        // the store, not from the encoder.
        let outcome = catch_up(&mut store, &encoder, 1, |_| Ok(())).await.unwrap();

        assert_eq!(encoder.encoded_texts(), vec!["term-7".to_string()]);
        assert_eq!((outcome.encoded, outcome.reused), (1, 2));
        assert_eq!(store.count_missing_sparse("m").unwrap(), 0);
    }

    #[tokio::test]
    async fn duplicates_inside_one_batch_encode_once() {
        let mut store = store_with(&[doc("a", "a.txt", "term-7"), doc("b", "b.txt", "term-7")]);
        let encoder = Fake::new("m");

        let outcome = catch_up(&mut store, &encoder, 10, |_| Ok(()))
            .await
            .unwrap();

        assert_eq!(encoder.encoded_texts().len(), 1);
        assert_eq!((outcome.encoded, outcome.reused), (1, 1));
    }

    #[tokio::test]
    async fn rows_already_encoded_by_the_current_model_are_not_encoded_again() {
        let mut store = store_with(&[doc("a", "a.txt", "term-1"), doc("b", "b.txt", "term-2")]);
        catch_up(&mut store, &Fake::new("m"), 10, |_| Ok(()))
            .await
            .unwrap();

        let encoder = Fake::new("m");
        let outcome = catch_up(&mut store, &encoder, 10, |_| Ok(()))
            .await
            .unwrap();

        assert!(encoder.encoded_texts().is_empty());
        assert_eq!((outcome.encoded, outcome.reused), (0, 0));
    }

    #[tokio::test]
    async fn a_new_model_re_encodes_every_row() {
        let mut store = store_with(&[doc("a", "a.txt", "term-1"), doc("b", "b.txt", "term-2")]);
        catch_up(&mut store, &Fake::new("old"), 10, |_| Ok(()))
            .await
            .unwrap();

        let encoder = Fake::new("new");
        let outcome = catch_up(&mut store, &encoder, 10, |_| Ok(()))
            .await
            .unwrap();

        assert_eq!(outcome.encoded, 2);
        assert_eq!(store.count_missing_sparse("new").unwrap(), 0);
        // A row holds one vector: the new model's replaces the old model's,
        // so the old model finds nothing and the new one finds everything.
        let probe = SparseVector::from_pairs([(1, 1.0)]);
        assert!(store
            .query_sparse("old", &probe, None, 5)
            .unwrap()
            .is_empty());
        assert_eq!(store.query_sparse("new", &probe, None, 5).unwrap().len(), 1);
    }

    #[tokio::test]
    async fn a_row_the_text_has_no_terms_for_is_done_and_never_retried() {
        let mut store = store_with(&[doc("a", "a.txt", "nothing")]);
        let encoder = Fake::new("m");
        catch_up(&mut store, &encoder, 10, |_| Ok(()))
            .await
            .unwrap();
        assert_eq!(store.count_missing_sparse("m").unwrap(), 0);

        let again = Fake::new("m");
        catch_up(&mut store, &again, 10, |_| Ok(())).await.unwrap();
        assert!(again.encoded_texts().is_empty());
    }

    #[tokio::test]
    async fn one_bad_chunk_fails_its_file_and_the_rest_is_still_encoded() {
        let mut store = store_with(&[
            doc("a", "good.txt", "term-1"),
            doc("b", "bad.txt", "poison"),
            doc("c", "good2.txt", "term-3"),
            doc("d", "bad.txt", "term-4"),
        ]);
        let mut encoder = Fake::new("m");
        encoder.reject = vec!["poison"];
        let mut reported = Vec::new();

        let outcome = catch_up(&mut store, &encoder, 10, |failure| {
            reported.push(failure.path.clone());
            Ok(())
        })
        .await
        .unwrap();

        assert_eq!(outcome.failures.len(), 1, "one failure per file");
        assert_eq!(outcome.failures[0].path, "bad.txt");
        assert_eq!(outcome.failures[0].stage, FileStage::Sparse);
        assert!(outcome.failures[0].error.contains("cannot encode poison"));
        assert_eq!(reported, vec!["bad.txt".to_string()]);
        // Only the poisoned row is left; its sibling in the same file is done.
        assert_eq!(store.count_missing_sparse("m").unwrap(), 1);
        assert_eq!(outcome.encoded, 3);
    }

    #[tokio::test]
    async fn an_error_from_the_failure_callback_stops_the_phase() {
        let mut store = store_with(&[doc("a", "bad.txt", "poison"), doc("b", "ok.txt", "term-2")]);
        let mut encoder = Fake::new("m");
        encoder.reject = vec!["poison"];

        let error = catch_up(&mut store, &encoder, 10, |failure| {
            Err(MinSyncError::Embedding(format!("{}: stop", failure.path)))
        })
        .await
        .expect_err("fail policy aborts");

        assert!(error.to_string().contains("bad.txt: stop"), "{error}");
    }

    #[tokio::test]
    async fn failed_rows_do_not_loop_forever_and_are_retried_by_the_next_run() {
        let mut store = store_with(&[doc("a", "bad.txt", "poison"), doc("b", "ok.txt", "term-2")]);
        let mut encoder = Fake::new("m");
        encoder.reject = vec!["poison"];
        catch_up(&mut store, &encoder, 1, |_| Ok(())).await.unwrap();
        assert_eq!(store.count_missing_sparse("m").unwrap(), 1);

        let healed = Fake::new("m");
        let outcome = catch_up(&mut store, &healed, 1, |_| Ok(())).await.unwrap();
        assert_eq!(outcome.encoded, 1);
        assert_eq!(store.count_missing_sparse("m").unwrap(), 0);
    }

    #[tokio::test]
    async fn a_dead_encoder_aborts_after_a_few_chunks_instead_of_failing_every_file() {
        let docs: Vec<Document> = (1..=20)
            .map(|i| {
                doc(
                    &format!("id{i}"),
                    &format!("f{i}.txt"),
                    &format!("term-{i}"),
                )
            })
            .collect();
        let mut store = store_with(&docs);
        let mut encoder = Fake::new("m");
        encoder.fail_all = true;

        let error = catch_up(&mut store, &encoder, 20, |_| Ok(()))
            .await
            .expect_err("a systemic failure aborts");

        assert!(error.to_string().contains("encoder is down"), "{error}");
        // Bisection plus the streak limit: nowhere near one call per chunk.
        assert!(encoder.encode_calls.load(Ordering::SeqCst) < 20);
        assert_eq!(store.count_missing_sparse("m").unwrap(), 20);
    }

    #[tokio::test]
    async fn a_wrong_answer_count_aborts() {
        struct Short;
        #[async_trait]
        impl SparseEncoder for Short {
            fn id(&self) -> &str {
                "m"
            }
            async fn encode(&self, _texts: &[String]) -> Result<Vec<SparseVector>> {
                Ok(Vec::new())
            }
            async fn encode_query(&self, _text: &str) -> Result<SparseVector> {
                Ok(SparseVector::default())
            }
        }
        let mut store = store_with(&[doc("a", "a.txt", "term-1")]);

        let error = catch_up(&mut store, &Short, 10, |_| Ok(()))
            .await
            .unwrap_err();

        assert!(error
            .to_string()
            .contains("expected 1 sparse vectors, got 0"));
    }

    #[tokio::test]
    async fn a_deadline_aborts_immediately() {
        struct Stuck;
        #[async_trait]
        impl SparseEncoder for Stuck {
            fn id(&self) -> &str {
                "m"
            }
            async fn encode(&self, _texts: &[String]) -> Result<Vec<SparseVector>> {
                Err(MinSyncError::Timeout("native batch stuck".into()))
            }
            async fn encode_query(&self, _text: &str) -> Result<SparseVector> {
                Ok(SparseVector::default())
            }
        }
        let mut store = store_with(&[doc("a", "a.txt", "term-1"), doc("b", "b.txt", "term-2")]);

        let error = catch_up(&mut store, &Stuck, 10, |_| Ok(()))
            .await
            .unwrap_err();

        assert!(matches!(error, MinSyncError::Timeout(_)));
    }

    #[tokio::test]
    async fn an_empty_store_encodes_nothing() {
        let mut store = InMemoryStore::new();
        let encoder = Fake::new("m");
        let outcome = catch_up(&mut store, &encoder, 10, |_| Ok(()))
            .await
            .unwrap();
        assert_eq!((outcome.encoded, outcome.reused), (0, 0));
        assert_eq!(encoder.encode_calls.load(Ordering::SeqCst), 0);
    }
}
