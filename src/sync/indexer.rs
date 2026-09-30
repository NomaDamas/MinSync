//! Per-file indexing: read, normalize, chunk, derive doc IDs, embed only
//! missing chunks, upsert, and sweep stale chunks for that file.

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

pub(super) async fn index_file(
    root: &Path,
    path: &str,
    context: SyncFileContext<'_>,
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
    let chunk_texts: Vec<String> = chunks.iter().map(|chunk| chunk.text.clone()).collect();
    let chunks_truncated = context.embedder.count_truncated(&chunk_texts).await?;
    if chunks_truncated > 0 {
        if let Some(max_length) = context.embedder.max_length() {
            tracing::warn!(
                "{} chunks in {} hit the {max_length}-token limit and were truncated",
                chunks_truncated,
                path
            );
        } else {
            tracing::warn!(
                "{} chunks in {} hit the embedder token limit and were truncated",
                chunks_truncated,
                path
            );
        }
        result.chunks_truncated += chunks_truncated;
    }
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
    let mut docs_to_embed = Vec::new();

    for (chunk, (doc_id, chunk_content_hash)) in chunks.into_iter().zip(doc_ids) {
        if existing_ids.contains(&doc_id) {
            context.store.update(&[DocumentUpdate {
                id: doc_id,
                seen_token: context.sync_token.to_string(),
                path: path.to_string(),
                heading_path: chunk.heading_path,
            }])?;
            result.chunks_updated += 1;
        } else {
            docs_to_embed.push(Document {
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
            });
        }
    }

    if !docs_to_embed.is_empty() {
        let texts: Vec<String> = docs_to_embed.iter().map(|doc| doc.text.clone()).collect();
        let mut embeddings = context.embedder.embed(&texts).await?;
        if embeddings.len() != docs_to_embed.len() {
            return Err(MinSyncError::Embedding(format!(
                "expected {} embeddings, got {}",
                docs_to_embed.len(),
                embeddings.len()
            )));
        }
        repair_non_finite_embeddings(
            context.embedder,
            &texts,
            &mut embeddings,
            context.config.embedder.max_retries,
            path,
            &docs_to_embed,
        )
        .await?;

        result.embedding_api_calls += 1;
        result.embedded_texts += texts.len();
        result.estimated_tokens += texts
            .iter()
            .map(|text| estimate_tokens(text))
            .sum::<usize>();

        for (doc, embedding) in docs_to_embed.iter_mut().zip(embeddings) {
            doc.embedding = embedding;
        }
        result.chunks_added += docs_to_embed.len();
        context.store.upsert(&docs_to_embed)?;
    }

    result.chunks_deleted += context.store.delete_by_filter(&Filter::And(vec![
        Filter::Eq("source_id".to_string(), context.config.source_id.clone()),
        Filter::Eq("path".to_string(), path.to_string()),
        Filter::Neq("seen_token".to_string(), context.sync_token.to_string()),
    ]))?;
    result.files_processed += 1;
    result.files_processed_paths.push(path.to_string());

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
    path: &str,
    docs: &[Document],
) -> Result<()> {
    for index in 0..embeddings.len() {
        if is_finite_embedding(&embeddings[index]) {
            continue;
        }
        let mut attempts = 0;
        while attempts < max_retries.max(1) && !is_finite_embedding(&embeddings[index]) {
            attempts += 1;
            tracing::warn!(
                path,
                chunk_index = index,
                attempt = attempts,
                "embedding contains non-finite values; re-embedding chunk alone"
            );
            embeddings[index] = embedder.embed_single(&texts[index]).await?;
        }
        if !is_finite_embedding(&embeddings[index]) {
            return Err(MinSyncError::Embedding(format!(
                "embedding for {path} (chunk {index}, document {}) contains non-finite values after {attempts} single-chunk retries",
                docs[index].id
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
