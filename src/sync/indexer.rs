//! Batch indexing: read, normalize, chunk and derive doc IDs for a group of
//! files, embed every missing chunk of the group in one call, and write the
//! whole group to the store with one commit per kind of write.
//!
//! Per batch of N files the store sees exactly: one `existing_ids`, one
//! `update` (reused chunks), one `insert_new` (new chunks), one stale-chunk
//! sweep, and one delete for removed files. The cost of a LanceDB write is
//! dominated by the commit, not the rows, so N files cost about what one did.
//!
//! One bad file must not sink the run. With `[sync] on_file_error = "skip"`
//! (the default) a file whose read, chunk, token-count, embed or store step
//! fails is recorded in `SyncResult.files_failed` and the rest proceeds; a
//! batch commit that fails is retried file by file to find the culprit.
//! `"fail"` aborts on the first error instead.

use crate::chunker::Chunker;
use crate::config::{Config, OnFileError};
use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use crate::id::doc_ids_for_chunks;
use crate::normalize::normalize_text;
use crate::types::{Chunk, FileFailure, FileStage, SyncResult};
use crate::vectorstore::{Document, DocumentUpdate, Filter, VectorStore};
use std::collections::HashSet;
use std::path::Path;

pub(super) struct SyncFileContext<'a> {
    pub config: &'a Config,
    pub chunker: &'a dyn Chunker,
    pub embedder: &'a dyn Embedder,
    pub store: &'a mut dyn VectorStore,
    pub sync_token: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BatchKind {
    /// Added or modified file: (re)index its chunks.
    Index,
    /// File removed from the tree: drop all of its chunks.
    Delete,
}

pub(super) struct BatchItem {
    pub path: String,
    pub kind: BatchKind,
    /// A resumed run already committed this path with older content. Its
    /// rows carry the reused sync token, so the stale sweep cannot tell them
    /// from fresh ones: clear the path before indexing.
    pub purge_first: bool,
}

struct PreparedFile<'a> {
    item: &'a BatchItem,
    chunks: Vec<(Chunk, String, String)>,
    truncated: usize,
}

enum Prepared<'a> {
    /// The file vanished since the scan; treat it like a deletion.
    Gone,
    File(PreparedFile<'a>),
}

/// Consecutive embed/store failures, with no success yet in the run, after
/// which the per-file retry concludes the failure is systemic.
const SYSTEMIC_STREAK: usize = 3;

/// An error tagged with the pipeline stage that produced it.
struct StageError {
    stage: FileStage,
    error: MinSyncError,
}

fn at(stage: FileStage) -> impl Fn(MinSyncError) -> StageError {
    move |error| StageError { stage, error }
}

fn path_filter(source_id: &str, paths: Vec<String>) -> Filter {
    Filter::And(vec![
        Filter::Eq("source_id".to_string(), source_id.to_string()),
        Filter::In("path".to_string(), paths),
    ])
}

/// Prefix `error` with `path` unless its text already names it, keeping the
/// variant so exit codes are unchanged.
fn with_path(error: MinSyncError, path: &str) -> MinSyncError {
    if error.to_string().contains(path) {
        return error;
    }
    match error {
        MinSyncError::Io(error) => MinSyncError::Io(std::io::Error::new(
            error.kind(),
            format!("{path}: {error}"),
        )),
        MinSyncError::Embedding(message) => MinSyncError::Embedding(format!("{path}: {message}")),
        MinSyncError::VectorStore(message) => {
            MinSyncError::VectorStore(format!("{path}: {message}"))
        }
        MinSyncError::Manifest(message) => MinSyncError::Manifest(format!("{path}: {message}")),
        MinSyncError::Config(message) => MinSyncError::Config(format!("{path}: {message}")),
        other => MinSyncError::Other(format!("{path}: {other}")),
    }
}

/// Record `failure` in `result` (skip mode) or abort with the path attached
/// (fail mode).
fn record_failure(
    skip: bool,
    path: &str,
    failure: StageError,
    result: &mut SyncResult,
) -> Result<()> {
    if !skip {
        return Err(with_path(failure.error, path));
    }
    tracing::warn!(
        path,
        stage = %failure.stage,
        error = %failure.error,
        "skipping file that failed to sync"
    );
    result.files_failed.push(FileFailure {
        path: path.to_string(),
        stage: failure.stage,
        error: failure.error.to_string(),
    });
    Ok(())
}

/// Read, normalize, chunk, count truncation and derive ids for one file.
async fn prepare_file<'a>(
    root: &Path,
    item: &'a BatchItem,
    context: &SyncFileContext<'_>,
) -> std::result::Result<Prepared<'a>, StageError> {
    let raw_text = match std::fs::read_to_string(root.join(&item.path)) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::InvalidData => String::new(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Prepared::Gone),
        Err(error) => return Err(at(FileStage::Read)(MinSyncError::Io(error))),
    };
    let text = normalize_text(&raw_text, &context.config.normalize);
    let chunks = context
        .chunker
        .chunk(&text, &item.path)
        .map_err(at(FileStage::Chunk))?;
    let chunk_texts: Vec<String> = chunks.iter().map(|chunk| chunk.text.clone()).collect();
    let truncated = context
        .embedder
        .count_truncated(&chunk_texts)
        .await
        .map_err(at(FileStage::Truncate))?;
    if truncated > 0 {
        if let Some(max_length) = context.embedder.max_length() {
            tracing::warn!(
                "{truncated} chunks in {} hit the {max_length}-token limit and were truncated",
                item.path
            );
        } else {
            tracing::warn!(
                "{truncated} chunks in {} hit the embedder token limit and were truncated",
                item.path
            );
        }
    }
    let ids = doc_ids_for_chunks(
        context.config.source_id.as_str(),
        &item.path,
        context.chunker.schema_id(),
        &chunks,
    );
    Ok(Prepared::File(PreparedFile {
        item,
        chunks: chunks
            .into_iter()
            .zip(ids)
            .map(|(chunk, (doc_id, content_hash))| (chunk, doc_id, content_hash))
            .collect(),
        truncated,
    }))
}

/// Index/delete `items` as one unit. Every item either lands in the store
/// and in `result.files_processed*`, or (skip mode) in `result.files_failed`;
/// in fail mode the first error aborts with nothing recorded for the batch.
pub(super) async fn index_batch(
    root: &Path,
    items: &[BatchItem],
    mut context: SyncFileContext<'_>,
    result: &mut SyncResult,
) -> Result<()> {
    let skip = context.config.sync.on_file_error == OnFileError::Skip;
    let failed_before = result.files_failed.len();

    // Read + chunk + id every file; files that vanished since the scan are
    // handled like deletions.
    let mut delete_paths: Vec<String> = Vec::new();
    let mut prepared: Vec<PreparedFile<'_>> = Vec::new();
    for item in items {
        if item.kind == BatchKind::Delete {
            delete_paths.push(item.path.clone());
            continue;
        }
        match prepare_file(root, item, &context).await {
            Ok(Prepared::Gone) => delete_paths.push(item.path.clone()),
            Ok(Prepared::File(file)) => prepared.push(file),
            Err(failure) => record_failure(skip, &item.path, failure, result)?,
        }
    }

    let prepared_items: Vec<&BatchItem> = prepared.iter().map(|file| file.item).collect();
    match commit(prepared, &delete_paths, &mut context, result).await {
        Ok(()) => {}
        Err(failure) if !skip => return Err(failure.error),
        Err(failure) => {
            tracing::warn!(
                stage = %failure.stage,
                error = %failure.error,
                "batch commit failed; retrying file by file"
            );
            // Embed/store failures that hit every file are an outage or a
            // misconfiguration (embedder down, bad key, disk full), not bad
            // files. While nothing has succeeded in this run, give up after
            // a few failures in a row with the original error so the hard
            // failure (and its exit code) is preserved.
            let mut succeeded = result.files_processed;
            let mut streak = 0;
            let streak_limit = prepared_items.len().min(SYSTEMIC_STREAK);
            // With one file and nothing else to write, a per-file retry would
            // repeat the identical commit: reuse the failure we already have.
            let mut first_failure =
                (prepared_items.len() == 1 && delete_paths.is_empty()).then_some(failure);
            for item in prepared_items {
                let outcome = if let Some(failure) = first_failure.take() {
                    Err(failure)
                } else {
                    match prepare_file(root, item, &context).await {
                        Ok(Prepared::File(file)) => {
                            commit(vec![file], &[], &mut context, result).await
                        }
                        Ok(Prepared::Gone) => {
                            commit(
                                Vec::new(),
                                std::slice::from_ref(&item.path),
                                &mut context,
                                result,
                            )
                            .await
                        }
                        Err(failure) => Err(failure),
                    }
                };
                match outcome {
                    Ok(()) => {
                        succeeded += 1;
                        streak = 0;
                    }
                    Err(failure) => {
                        let systemic = matches!(failure.stage, FileStage::Embed | FileStage::Store);
                        streak = if systemic { streak + 1 } else { 0 };
                        if succeeded == 0 && streak >= streak_limit {
                            result.files_failed.truncate(failed_before);
                            return Err(with_path(failure.error, &item.path));
                        }
                        record_failure(skip, &item.path, failure, result)?;
                    }
                }
            }
            for path in &delete_paths {
                if let Err(failure) =
                    commit(Vec::new(), std::slice::from_ref(path), &mut context, result).await
                {
                    record_failure(skip, path, failure, result)?;
                }
            }
        }
    }

    let failed: HashSet<&str> = result.files_failed[failed_before..]
        .iter()
        .map(|failure| failure.path.as_str())
        .collect();
    let done: Vec<String> = items
        .iter()
        .filter(|item| !failed.contains(item.path.as_str()))
        .map(|item| item.path.clone())
        .collect();
    result.files_processed += done.len();
    result.files_processed_paths.extend(done);
    Ok(())
}

/// Write `files` and delete `delete_paths` in one pass: existence check,
/// embed, update, insert, stale sweep, delete. Counters in `result` move
/// only when every write succeeded.
async fn commit(
    files: Vec<PreparedFile<'_>>,
    delete_paths: &[String],
    context: &mut SyncFileContext<'_>,
    result: &mut SyncResult,
) -> std::result::Result<(), StageError> {
    let source_id = context.config.source_id.as_str();
    let schema_id = context.chunker.schema_id();
    let store_stage = at(FileStage::Store);
    let embed_stage = at(FileStage::Embed);

    let purge: Vec<String> = files
        .iter()
        .filter(|file| file.item.purge_first)
        .map(|file| file.item.path.clone())
        .collect();
    let mut chunks_deleted = 0;
    if !purge.is_empty() {
        chunks_deleted += context
            .store
            .delete_by_filter(&path_filter(source_id, purge))
            .map_err(&store_stage)?;
    }
    let truncated_total: usize = files.iter().map(|file| file.truncated).sum();
    let indexed_paths: Vec<String> = files.iter().map(|file| file.item.path.clone()).collect();

    // One id-only existence check for the whole batch.
    let all_ids: Vec<String> = files
        .iter()
        .flat_map(|file| file.chunks.iter().map(|(_, doc_id, _)| doc_id.clone()))
        .collect();
    let existing: HashSet<String> = context.store.existing_ids(&all_ids).map_err(&store_stage)?;

    let mut reused: Vec<DocumentUpdate> = Vec::new();
    let mut docs_to_embed: Vec<Document> = Vec::new();
    // Position of each new chunk within its own file, for error messages.
    let mut chunk_in_file: Vec<usize> = Vec::new();
    for file in files {
        let path = file.item.path.as_str();
        let mut new_in_file = 0;
        for (chunk, doc_id, content_hash) in file.chunks {
            if existing.contains(&doc_id) {
                reused.push(DocumentUpdate {
                    id: doc_id,
                    seen_token: context.sync_token.to_string(),
                    path: path.to_string(),
                    heading_path: chunk.heading_path,
                });
            } else {
                docs_to_embed.push(Document {
                    id: doc_id,
                    embedding: Vec::new(),
                    text: chunk.text,
                    source_id: source_id.to_string(),
                    path: path.to_string(),
                    chunk_schema_id: schema_id.to_string(),
                    chunk_type: chunk.chunk_type,
                    heading_path: chunk.heading_path,
                    content_hash,
                    seen_token: context.sync_token.to_string(),
                });
                chunk_in_file.push(new_in_file);
                new_in_file += 1;
            }
        }
    }

    // Embed first: the slow, fallible step runs before any write, so a
    // failure leaves the store untouched by this batch.
    //
    // Identical chunk text (same content hash) embeds once, and vectors the
    // store already holds for that hash are reused. The rest go to the
    // embedder longest first so consecutive batches pad far less.
    if !docs_to_embed.is_empty() {
        let mut seen_hashes: HashSet<&str> = HashSet::new();
        let mut unique: Vec<usize> = Vec::new();
        for (index, doc) in docs_to_embed.iter().enumerate() {
            if seen_hashes.insert(doc.content_hash.as_str()) {
                unique.push(index);
            }
        }
        let hashes: Vec<String> = unique
            .iter()
            .map(|&index| docs_to_embed[index].content_hash.clone())
            .collect();
        let mut vectors = context
            .store
            .fetch_embeddings_by_content_hash(&hashes)
            .map_err(&store_stage)?;

        let mut to_embed: Vec<usize> = unique
            .into_iter()
            .filter(|&index| !vectors.contains_key(&docs_to_embed[index].content_hash))
            .collect();
        to_embed.sort_by_key(|&index| std::cmp::Reverse(docs_to_embed[index].text.len()));

        if !to_embed.is_empty() {
            let texts: Vec<String> = to_embed
                .iter()
                .map(|&index| docs_to_embed[index].text.clone())
                .collect();
            let mut embeddings = context.embedder.embed(&texts).await.map_err(&embed_stage)?;
            if embeddings.len() != texts.len() {
                return Err(embed_stage(MinSyncError::Embedding(format!(
                    "expected {} embeddings, got {}",
                    texts.len(),
                    embeddings.len()
                ))));
            }
            let located: Vec<(&Document, usize)> = to_embed
                .iter()
                .map(|&index| (&docs_to_embed[index], chunk_in_file[index]))
                .collect();
            repair_non_finite_embeddings(
                context.embedder,
                &texts,
                &mut embeddings,
                context.config.embedder.max_retries,
                &located,
            )
            .await
            .map_err(&embed_stage)?;

            result.embedding_api_calls += 1;
            result.embedded_texts += texts.len();
            result.estimated_tokens += texts
                .iter()
                .map(|text| estimate_tokens(text))
                .sum::<usize>();
            for (&index, embedding) in to_embed.iter().zip(embeddings) {
                vectors.insert(docs_to_embed[index].content_hash.clone(), embedding);
            }
        }

        result.embeddings_reused += docs_to_embed.len() - to_embed.len();
        for (index, doc) in docs_to_embed.iter_mut().enumerate() {
            doc.embedding = vectors.get(&doc.content_hash).cloned().ok_or_else(|| {
                embed_stage(MinSyncError::Embedding(format!(
                    "missing embedding for {} (chunk {})",
                    doc.path, chunk_in_file[index]
                )))
            })?;
        }
    }

    if !reused.is_empty() {
        context.store.update(&reused).map_err(&store_stage)?;
    }
    if !docs_to_embed.is_empty() {
        context
            .store
            .insert_new(&docs_to_embed)
            .map_err(&store_stage)?;
    }

    // Stale chunks of every indexed file (content changed: old ids no longer
    // carry this run's token) in one sweep, then removed files in one delete.
    if !indexed_paths.is_empty() {
        chunks_deleted += context
            .store
            .delete_by_filter(&Filter::And(vec![
                Filter::Eq("source_id".to_string(), source_id.to_string()),
                Filter::In("path".to_string(), indexed_paths),
                Filter::Neq("seen_token".to_string(), context.sync_token.to_string()),
            ]))
            .map_err(&store_stage)?;
    }
    if !delete_paths.is_empty() {
        chunks_deleted += context
            .store
            .delete_by_filter(&path_filter(source_id, delete_paths.to_vec()))
            .map_err(&store_stage)?;
    }

    result.chunks_updated += reused.len();
    result.chunks_added += docs_to_embed.len();
    result.chunks_deleted += chunks_deleted;
    result.chunks_truncated += truncated_total;
    Ok(())
}

fn is_finite_embedding(embedding: &[f32]) -> bool {
    embedding.iter().all(|value| value.is_finite())
}

/// Re-embed chunks whose vectors contain NaN/inf, one text per request.
///
/// Local GPU backends can emit non-finite values for a single chunk inside a
/// large batch without the input itself being invalid, so re-embedding it
/// alone usually yields a valid vector. Without this, one bad vector aborts
/// the whole sync at write time. Chunks that stay non-finite after
/// `max_retries` single-text attempts fail with the file path and chunk index
/// so the offending file can be inspected or excluded.
async fn repair_non_finite_embeddings(
    embedder: &dyn Embedder,
    texts: &[String],
    embeddings: &mut [Vec<f32>],
    max_retries: usize,
    located: &[(&Document, usize)],
) -> Result<()> {
    for index in 0..embeddings.len() {
        if is_finite_embedding(&embeddings[index]) {
            continue;
        }
        let (doc, chunk) = located[index];
        let path = doc.path.as_str();
        let mut attempts = 0;
        while attempts < max_retries.max(1) && !is_finite_embedding(&embeddings[index]) {
            attempts += 1;
            tracing::warn!(
                path,
                chunk_index = chunk,
                attempt = attempts,
                "embedding contains non-finite values; re-embedding chunk alone"
            );
            embeddings[index] = embedder.embed_single(&texts[index]).await?;
        }
        if !is_finite_embedding(&embeddings[index]) {
            return Err(MinSyncError::Embedding(format!(
                "embedding for {path} (chunk {chunk}, document {}) contains non-finite values after {attempts} single-chunk retries",
                doc.id
            )));
        }
    }
    Ok(())
}

fn estimate_tokens(text: &str) -> usize {
    let ascii_count = text.chars().filter(|char| char.is_ascii()).count();
    let non_ascii_count = text.chars().count() - ascii_count;
    ascii_count / 4 + non_ascii_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens_ascii() {
        assert_eq!(estimate_tokens("abcdefgh"), 2);
    }

    #[test]
    fn test_estimate_tokens_korean() {
        let text = "안녕하세요";
        assert_eq!(estimate_tokens(text), text.chars().count());
    }

    #[test]
    fn test_estimate_tokens_mixed_ascii_and_non_ascii() {
        assert_eq!(estimate_tokens("abcdefgh한글"), 4);
    }
}
