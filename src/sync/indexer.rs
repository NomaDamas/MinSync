//! Per-file indexing: read, normalize, chunk, derive doc IDs, queue missing
//! chunks into a cross-file embedding window, then embed, upsert, and sweep
//! stale chunks for every file in the window.

use crate::chunker::Chunker;
use crate::config::Config;
use crate::embedder::Embedder;
use crate::error::{MinSyncError, Result};
use crate::id::doc_ids_for_chunks;
use crate::normalize::normalize_text;
use crate::types::SyncResult;
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

/// Pending chunk count that triggers embedding a window. Large enough that
/// length-sorting across files removes most padding and duplicate chunks
/// across files embed once; small enough to keep memory and the work lost on
/// a crash bounded.
pub(super) const EMBED_WINDOW_CHUNKS: usize = 256;

/// A chunk that still needs a vector, with its position in its file for
/// error reporting.
struct PendingDoc {
    doc: Document,
    chunk_index: usize,
}

/// A file whose chunks are prepared but not yet embedded/upserted. Its stale
/// chunks are swept only after its new chunks are stored.
struct PendingFile {
    path: String,
    docs: Vec<PendingDoc>,
}

/// Files prepared for indexing whose new chunks are embedded together.
///
/// Batching across files lets the embedder see many chunks at once: they are
/// deduplicated by content hash (identical text embeds once, and vectors
/// already in the store are reused) and sorted by length so batches pad far
/// less than per-file batches did.
#[derive(Default)]
pub(super) struct EmbedWindow {
    files: Vec<PendingFile>,
    pending_chunks: usize,
}

impl EmbedWindow {
    pub(super) fn is_full(&self) -> bool {
        self.pending_chunks >= EMBED_WINDOW_CHUNKS
    }
}

/// Index one file immediately (a window of one file).
#[cfg(test)]
pub(super) async fn index_file(
    root: &Path,
    path: &str,
    context: SyncFileContext<'_>,
    result: &mut SyncResult,
) -> Result<()> {
    let mut window = EmbedWindow::default();
    let SyncFileContext {
        config,
        chunker,
        embedder,
        store,
        sync_token,
    } = context;
    prepare_file(
        root,
        path,
        SyncFileContext {
            config,
            chunker,
            embedder,
            store: &mut *store,
            sync_token,
        },
        &mut window,
        result,
    )?;
    flush_window(
        SyncFileContext {
            config,
            chunker,
            embedder,
            store,
            sync_token,
        },
        &mut window,
        result,
    )
    .await
}

/// Read, normalize and chunk one file; refresh chunks that already exist and
/// queue the rest in `window`. Missing files are deleted immediately.
pub(super) fn prepare_file(
    root: &Path,
    path: &str,
    context: SyncFileContext<'_>,
    window: &mut EmbedWindow,
    result: &mut SyncResult,
) -> Result<()> {
    let raw_text = match std::fs::read_to_string(root.join(path)) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::InvalidData => String::new(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            result.chunks_deleted += context.store.delete_by_filter(&Filter::And(vec![
                Filter::Eq("source_id".to_string(), context.config.source_id.clone()),
                Filter::Eq("path".to_string(), path.to_string()),
            ]))?;
            result.files_processed += 1;
            result.files_processed_paths.push(path.to_string());
            return Ok(());
        }
        Err(error) => return Err(MinSyncError::Io(error)),
    };
    let text = normalize_text(&raw_text, &context.config.normalize);
    let chunks = context.chunker.chunk(&text, path)?;
    let schema_id = context.chunker.schema_id();
    let doc_ids = doc_ids_for_chunks(&context.config.source_id, path, schema_id, &chunks);
    let existing_ids: HashSet<_> = context
        .store
        .fetch(
            &doc_ids
                .iter()
                .map(|(doc_id, _)| doc_id.clone())
                .collect::<Vec<_>>(),
        )?
        .into_iter()
        .map(|doc| doc.id)
        .collect();
    let mut docs = Vec::new();

    for (chunk_index, (chunk, (doc_id, chunk_content_hash))) in
        chunks.into_iter().zip(doc_ids).enumerate()
    {
        if existing_ids.contains(&doc_id) {
            context.store.update(&[DocumentUpdate {
                id: doc_id,
                seen_token: context.sync_token.to_string(),
                path: path.to_string(),
                heading_path: chunk.heading_path,
            }])?;
            result.chunks_updated += 1;
        } else {
            docs.push(PendingDoc {
                doc: Document {
                    id: doc_id,
                    embedding: Vec::new(),
                    text: chunk.text,
                    source_id: context.config.source_id.clone(),
                    path: path.to_string(),
                    chunk_schema_id: schema_id.to_string(),
                    chunk_type: chunk.chunk_type,
                    heading_path: chunk.heading_path,
                    content_hash: chunk_content_hash,
                    seen_token: context.sync_token.to_string(),
                },
                chunk_index,
            });
        }
    }

    result.files_processed += 1;
    result.files_processed_paths.push(path.to_string());
    window.pending_chunks += docs.len();
    window.files.push(PendingFile {
        path: path.to_string(),
        docs,
    });
    Ok(())
}

/// Embed every queued chunk, upsert it, then sweep each file's stale chunks.
pub(super) async fn flush_window(
    context: SyncFileContext<'_>,
    window: &mut EmbedWindow,
    result: &mut SyncResult,
) -> Result<()> {
    let files = std::mem::take(&mut window.files);
    window.pending_chunks = 0;

    // One representative per distinct chunk text.
    let mut unique: Vec<&PendingDoc> = Vec::new();
    let mut seen_hashes = HashSet::new();
    for file in &files {
        for pending in &file.docs {
            if seen_hashes.insert(pending.doc.content_hash.as_str()) {
                unique.push(pending);
            }
        }
    }

    let hashes: Vec<String> = unique
        .iter()
        .map(|pending| pending.doc.content_hash.clone())
        .collect();
    let mut vectors = context.store.fetch_embeddings_by_content_hash(&hashes)?;

    // Longest first so consecutive embedder batches have similar lengths.
    let mut to_embed: Vec<&PendingDoc> = unique
        .into_iter()
        .filter(|pending| !vectors.contains_key(&pending.doc.content_hash))
        .collect();
    to_embed.sort_by_key(|pending| std::cmp::Reverse(pending.doc.text.len()));

    if !to_embed.is_empty() {
        let texts: Vec<String> = to_embed
            .iter()
            .map(|pending| pending.doc.text.clone())
            .collect();
        let mut embeddings = context.embedder.embed(&texts).await?;
        if embeddings.len() != texts.len() {
            return Err(MinSyncError::Embedding(format!(
                "expected {} embeddings, got {}",
                texts.len(),
                embeddings.len()
            )));
        }
        repair_non_finite_embeddings(
            context.embedder,
            &texts,
            &mut embeddings,
            context.config.embedder.max_retries,
            &to_embed,
        )
        .await?;

        result.embedding_api_calls += 1;
        result.embedded_texts += texts.len();
        result.estimated_tokens += texts
            .iter()
            .map(|text| estimate_tokens(text))
            .sum::<usize>();
        for (pending, embedding) in to_embed.iter().zip(embeddings) {
            vectors.insert(pending.doc.content_hash.clone(), embedding);
        }
    }

    let mut docs = Vec::with_capacity(files.iter().map(|file| file.docs.len()).sum());
    for file in &files {
        for pending in &file.docs {
            let mut doc = pending.doc.clone();
            doc.embedding = vectors.get(&doc.content_hash).cloned().ok_or_else(|| {
                MinSyncError::Embedding(format!(
                    "missing embedding for {} (chunk {})",
                    file.path, pending.chunk_index
                ))
            })?;
            docs.push(doc);
        }
    }
    result.chunks_added += docs.len();
    result.embeddings_reused += docs.len() - to_embed.len();
    if !docs.is_empty() {
        context.store.upsert(&docs)?;
    }

    for file in &files {
        result.chunks_deleted += context.store.delete_by_filter(&Filter::And(vec![
            Filter::Eq("source_id".to_string(), context.config.source_id.clone()),
            Filter::Eq("path".to_string(), file.path.clone()),
            Filter::Neq("seen_token".to_string(), context.sync_token.to_string()),
        ]))?;
    }

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
/// the whole sync at upsert time. Chunks that stay non-finite after
/// `max_retries` single-text attempts fail with the file path and chunk index
/// so the offending file can be inspected or excluded.
async fn repair_non_finite_embeddings(
    embedder: &dyn Embedder,
    texts: &[String],
    embeddings: &mut [Vec<f32>],
    max_retries: usize,
    pending: &[&PendingDoc],
) -> Result<()> {
    for index in 0..embeddings.len() {
        if is_finite_embedding(&embeddings[index]) {
            continue;
        }
        let mut attempts = 0;
        while attempts < max_retries.max(1) && !is_finite_embedding(&embeddings[index]) {
            attempts += 1;
            tracing::warn!(
                path = pending[index].doc.path.as_str(),
                chunk_index = pending[index].chunk_index,
                attempt = attempts,
                "embedding contains non-finite values; re-embedding chunk alone"
            );
            embeddings[index] = embedder.embed_single(&texts[index]).await?;
        }
        if !is_finite_embedding(&embeddings[index]) {
            let doc = &pending[index];
            return Err(MinSyncError::Embedding(format!(
                "embedding for {} (chunk {}, document {}) contains non-finite values after {attempts} single-chunk retries",
                doc.doc.path, doc.chunk_index, doc.doc.id
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
