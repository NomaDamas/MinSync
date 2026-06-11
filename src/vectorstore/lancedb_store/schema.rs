//! Arrow schema definition and conversions between [`Document`]/[`QueryHit`]
//! values and Arrow record batches.

use super::{to_store_error, DISTANCE_COLUMN, VECTOR_COLUMN};
use crate::error::{MinSyncError, Result};
use crate::vectorstore::{Document, QueryHit};
use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{Array, FixedSizeListArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use std::collections::HashMap;
use std::sync::Arc;

pub(super) fn schema(dim: usize) -> Result<SchemaRef> {
    validate_dimension(dim)?;
    Ok(Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new(
            VECTOR_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
        Field::new("text", DataType::Utf8, false),
        Field::new("source_id", DataType::Utf8, false),
        Field::new("path", DataType::Utf8, false),
        Field::new("chunk_schema_id", DataType::Utf8, false),
        Field::new("chunk_type", DataType::Utf8, false),
        Field::new("heading_path", DataType::Utf8, false),
        Field::new("content_hash", DataType::Utf8, false),
        Field::new("seen_token", DataType::Utf8, false),
    ])))
}

pub(super) fn validate_schema(schema: &SchemaRef, dim: usize) -> Result<()> {
    validate_dimension(dim)?;
    for field_name in super::filters::FILTERABLE_FIELDS {
        let field = schema.field_with_name(field_name).map_err(to_store_error)?;
        if field.data_type() != &DataType::Utf8 {
            return Err(MinSyncError::VectorStore(format!(
                "LanceDB column {field_name} must be Utf8"
            )));
        }
    }

    let vector = schema
        .field_with_name(VECTOR_COLUMN)
        .map_err(to_store_error)?;
    match vector.data_type() {
        DataType::FixedSizeList(item, actual_dim)
            if item.data_type() == &DataType::Float32 && *actual_dim == dim as i32 =>
        {
            Ok(())
        }
        DataType::FixedSizeList(item, actual_dim) if item.data_type() == &DataType::Float32 => {
            Err(MinSyncError::VectorStore(format!(
                "dimension mismatch for LanceDB vector column: existing {actual_dim}, requested {dim}"
            )))
        }
        _ => Err(MinSyncError::VectorStore(
            "LanceDB vector column must be FixedSizeList(Float32, dimension)".into(),
        )),
    }
}

pub(super) fn docs_to_batch(docs: &[Document], dim: usize) -> Result<RecordBatch> {
    for doc in docs {
        validate_vector(
            &doc.embedding,
            dim,
            &format!("document {} embedding", doc.id),
        )?;
    }

    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        docs.iter()
            .map(|doc| Some(doc.embedding.iter().copied().map(Some).collect::<Vec<_>>())),
        dim as i32,
    );

    RecordBatch::try_new(
        schema(dim)?,
        vec![
            Arc::new(StringArray::from_iter_values(
                docs.iter().map(|doc| doc.id.as_str()),
            )),
            Arc::new(vectors),
            Arc::new(StringArray::from_iter_values(
                docs.iter().map(|doc| doc.text.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                docs.iter().map(|doc| doc.source_id.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                docs.iter().map(|doc| doc.path.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                docs.iter().map(|doc| doc.chunk_schema_id.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                docs.iter().map(|doc| doc.chunk_type.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                docs.iter().map(|doc| doc.heading_path.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                docs.iter().map(|doc| doc.content_hash.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                docs.iter().map(|doc| doc.seen_token.as_str()),
            )),
        ],
    )
    .map_err(to_store_error)
}

pub(super) fn batch_to_documents(batch: &RecordBatch) -> Result<Vec<Document>> {
    let ids = string_col(batch, "id")?;
    let vectors = batch
        .column_by_name(VECTOR_COLUMN)
        .ok_or_else(|| missing_column(VECTOR_COLUMN))?
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| MinSyncError::VectorStore("vector column has unexpected type".into()))?;
    let text = string_col(batch, "text")?;
    let source_ids = string_col(batch, "source_id")?;
    let paths = string_col(batch, "path")?;
    let schemas = string_col(batch, "chunk_schema_id")?;
    let chunk_types = string_col(batch, "chunk_type")?;
    let headings = string_col(batch, "heading_path")?;
    let hashes = string_col(batch, "content_hash")?;
    let tokens = string_col(batch, "seen_token")?;

    let mut docs = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let vector = vectors
            .value(row)
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        docs.push(Document {
            id: ids.value(row).to_string(),
            embedding: vector,
            text: text.value(row).to_string(),
            source_id: source_ids.value(row).to_string(),
            path: paths.value(row).to_string(),
            chunk_schema_id: schemas.value(row).to_string(),
            chunk_type: chunk_types.value(row).to_string(),
            heading_path: headings.value(row).to_string(),
            content_hash: hashes.value(row).to_string(),
            seen_token: tokens.value(row).to_string(),
        });
    }
    Ok(docs)
}

pub(super) fn batch_to_query_hits(batch: &RecordBatch) -> Result<Vec<QueryHit>> {
    let ids = string_col(batch, "id")?;
    let paths = string_col(batch, "path")?;
    let headings = string_col(batch, "heading_path")?;
    let chunk_types = string_col(batch, "chunk_type")?;
    let texts = string_col(batch, "text")?;
    let hashes = string_col(batch, "content_hash")?;
    let distances = batch
        .column_by_name(DISTANCE_COLUMN)
        .ok_or_else(|| missing_column(DISTANCE_COLUMN))?
        .as_primitive::<Float32Type>();

    let mut hits = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let distance = distances.value(row);
        hits.push(QueryHit {
            doc_id: ids.value(row).to_string(),
            path: paths.value(row).to_string(),
            heading_path: headings.value(row).to_string(),
            chunk_type: chunk_types.value(row).to_string(),
            text: texts.value(row).to_string(),
            // LanceDB returns cosine distance (lower is better). The
            // VectorStore contract requires similarity scores (higher is
            // better), matching the in-memory store, so reconcile here.
            score: 1.0 - distance,
            content_hash: hashes.value(row).to_string(),
        });
    }
    Ok(hits)
}

pub(super) fn dedupe_documents(docs: Vec<Document>) -> Vec<Document> {
    let mut last_by_id = HashMap::new();
    for (index, doc) in docs.iter().enumerate() {
        last_by_id.insert(doc.id.clone(), index);
    }
    docs.into_iter()
        .enumerate()
        .filter_map(|(index, doc)| (last_by_id.get(&doc.id) == Some(&index)).then_some(doc))
        .collect()
}

pub(super) fn validate_vector(vector: &[f32], dim: usize, label: &str) -> Result<()> {
    if vector.len() != dim {
        return Err(MinSyncError::VectorStore(format!(
            "{label} dimension {} does not match LanceDB table dimension {dim}",
            vector.len()
        )));
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(MinSyncError::VectorStore(format!(
            "{label} contains non-finite values"
        )));
    }
    Ok(())
}

pub(super) fn validate_dimension(dim: usize) -> Result<()> {
    if dim == 0 || dim > i32::MAX as usize {
        return Err(MinSyncError::VectorStore(format!(
            "invalid LanceDB vector dimension: {dim}"
        )));
    }
    Ok(())
}

pub(super) fn string_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    batch
        .column_by_name(name)
        .ok_or_else(|| missing_column(name))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| MinSyncError::VectorStore(format!("column {name} has unexpected type")))
}

pub(super) fn missing_column(name: &str) -> MinSyncError {
    MinSyncError::VectorStore(format!("missing LanceDB column {name}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(id: &str, embedding: Vec<f32>) -> Document {
        Document {
            id: id.to_string(),
            embedding,
            text: format!("text {id}"),
            source_id: "source-1".to_string(),
            path: format!("{id}.txt"),
            chunk_schema_id: "schema-1".to_string(),
            chunk_type: "text".to_string(),
            heading_path: String::new(),
            content_hash: format!("hash-{id}"),
            seen_token: "token".to_string(),
        }
    }

    #[test]
    fn test_docs_to_batch_roundtrip() {
        let docs = vec![doc("a", vec![1.0, 0.0]), doc("b", vec![0.5, 0.5])];

        let batch = docs_to_batch(&docs, 2).expect("convert to batch");
        let roundtripped = batch_to_documents(&batch).expect("convert back");

        assert_eq!(roundtripped.len(), docs.len());
        for (actual, expected) in roundtripped.iter().zip(&docs) {
            assert_eq!(actual.id, expected.id);
            assert_eq!(actual.embedding, expected.embedding);
            assert_eq!(actual.text, expected.text);
            assert_eq!(actual.source_id, expected.source_id);
            assert_eq!(actual.path, expected.path);
            assert_eq!(actual.chunk_schema_id, expected.chunk_schema_id);
            assert_eq!(actual.chunk_type, expected.chunk_type);
            assert_eq!(actual.heading_path, expected.heading_path);
            assert_eq!(actual.content_hash, expected.content_hash);
            assert_eq!(actual.seen_token, expected.seen_token);
        }
    }

    #[test]
    fn test_docs_to_batch_rejects_wrong_dimension() {
        let docs = vec![doc("a", vec![1.0, 0.0, 0.0])];

        assert!(docs_to_batch(&docs, 2).is_err());
    }

    #[test]
    fn test_dedupe_documents_last_wins() {
        let docs = vec![
            doc("a", vec![1.0, 0.0]),
            doc("b", vec![0.0, 1.0]),
            Document {
                text: "replacement".to_string(),
                ..doc("a", vec![0.5, 0.5])
            },
        ];

        let deduped = dedupe_documents(docs);

        assert_eq!(deduped.len(), 2);
        assert_eq!(deduped[1].id, "a");
        assert_eq!(deduped[1].text, "replacement");
    }

    #[test]
    fn test_validate_vector() {
        assert!(validate_vector(&[1.0, 2.0], 2, "v").is_ok());
        assert!(validate_vector(&[1.0], 2, "v").is_err());
        assert!(validate_vector(&[1.0, f32::NAN], 2, "v").is_err());
    }

    #[test]
    fn test_validate_dimension() {
        assert!(validate_dimension(1).is_ok());
        assert!(validate_dimension(0).is_err());
        assert!(validate_dimension(i32::MAX as usize + 1).is_err());
    }

    #[test]
    fn test_validate_schema_detects_mismatch() {
        let four = schema(4).expect("schema");
        assert!(validate_schema(&four, 4).is_ok());
        assert!(validate_schema(&four, 3).is_err());
    }
}
