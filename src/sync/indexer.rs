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
        let embeddings = context.embedder.embed(&texts).await?;
        if embeddings.len() != docs_to_embed.len() {
            return Err(MinSyncError::Embedding(format!(
                "expected {} embeddings, got {}",
                docs_to_embed.len(),
                embeddings.len()
            )));
        }

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
