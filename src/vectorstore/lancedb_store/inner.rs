//! Async LanceDB operations executed on the worker thread.

use super::filters::{filter_to_sql, id_in_filter, sql_literal};
use super::index::maintain_index;
use super::schema::{
    batch_to_documents, batch_to_query_hits, dedupe_documents, docs_to_batch, schema, string_col,
    validate_schema, validate_vector,
};
use super::{
    to_store_error, IndexingConfig, DISTANCE_COLUMN, DISTANCE_TYPE, TABLE_NAME, VECTOR_COLUMN,
};
use crate::error::Result;
use crate::vectorstore::{Document, DocumentUpdate, Filter, QueryHit};
use arrow_array::RecordBatch;
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::table::OptimizeAction;
use lancedb::{Connection, Table};
use std::cmp::Ordering;
use std::collections::HashSet;

pub(super) struct LanceDbInner {
    conn: Connection,
    table: Table,
    dim: usize,
    indexing: IndexingConfig,
}

impl LanceDbInner {
    pub(super) async fn open_or_create(
        uri: &str,
        dim: usize,
        indexing: IndexingConfig,
    ) -> Result<Self> {
        let conn = lancedb::connect(uri)
            .execute()
            .await
            .map_err(to_store_error)?;
        let table_names = conn.table_names().execute().await.map_err(to_store_error)?;
        let table = if table_names.iter().any(|name| name == TABLE_NAME) {
            let table = conn
                .open_table(TABLE_NAME)
                .execute()
                .await
                .map_err(to_store_error)?;
            validate_schema(&table.schema().await.map_err(to_store_error)?, dim)?;
            table
        } else {
            conn.create_empty_table(TABLE_NAME, schema(dim)?)
                .execute()
                .await
                .map_err(to_store_error)?
        };

        Ok(Self {
            conn,
            table,
            dim,
            indexing,
        })
    }

    pub(super) async fn upsert(&self, docs: Vec<Document>) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let docs = dedupe_documents(docs);
        let id_filter = id_in_filter(docs.iter().map(|doc| doc.id.as_str()));
        let batch = docs_to_batch(&docs, self.dim)?;
        self.table
            .delete(&id_filter)
            .await
            .map_err(to_store_error)?;
        self.table
            .add(batch)
            .execute()
            .await
            .map_err(to_store_error)?;
        Ok(())
    }

    pub(super) async fn update(&self, updates: Vec<DocumentUpdate>) -> Result<()> {
        for update in updates {
            let filter = format!("id = '{}'", super::filters::escape_sql_literal(&update.id));
            self.table
                .update()
                .only_if(filter)
                .column("seen_token", sql_literal(&update.seen_token))
                .column("path", sql_literal(&update.path))
                .column("heading_path", sql_literal(&update.heading_path))
                .execute()
                .await
                .map_err(to_store_error)?;
        }
        Ok(())
    }

    pub(super) async fn fetch(&self, ids: Vec<String>) -> Result<Vec<Document>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let filter = id_in_filter(ids.iter().map(String::as_str));
        let mut docs = self.scan_documents(Some(filter)).await?;
        docs.sort_by_key(|doc| ids.iter().position(|id| *id == doc.id).unwrap_or(ids.len()));
        Ok(docs)
    }

    pub(super) async fn delete_by_filter(&self, filter: Filter) -> Result<usize> {
        let sql = filter_to_sql(&filter)?;
        let result = self.table.delete(&sql).await.map_err(to_store_error)?;
        usize::try_from(result.num_deleted_rows).map_err(to_store_error)
    }

    pub(super) async fn query(
        &self,
        vector: Vec<f32>,
        filter: Option<Filter>,
        topk: usize,
    ) -> Result<Vec<QueryHit>> {
        if topk == 0 {
            return Ok(Vec::new());
        }
        validate_vector(&vector, self.dim, "query vector")?;
        let sql_filter = filter.as_ref().map(filter_to_sql).transpose()?;

        let mut query = self
            .table
            .vector_search(vector)
            .map_err(to_store_error)?
            .column(VECTOR_COLUMN)
            .distance_type(DISTANCE_TYPE)
            .limit(topk)
            .select(Select::columns(&[
                "id",
                "path",
                "heading_path",
                "chunk_type",
                "text",
                "content_hash",
                DISTANCE_COLUMN,
            ]));
        if let Some(sql) = sql_filter {
            query = query.only_if(sql);
        }

        let batches: Vec<RecordBatch> = query
            .execute()
            .await
            .map_err(to_store_error)?
            .try_collect()
            .await
            .map_err(to_store_error)?;

        let mut hits = Vec::new();
        for batch in batches {
            hits.extend(batch_to_query_hits(&batch)?);
        }

        hits.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
        });
        hits.truncate(topk);
        Ok(hits)
    }

    pub(super) async fn flush(&self) -> Result<()> {
        self.table
            .optimize(OptimizeAction::Compact {
                options: lancedb::table::CompactionOptions::default(),
                remap_options: None,
            })
            .await
            .map_err(to_store_error)?;
        maintain_index(&self.table, &self.indexing).await?;
        self.table
            .optimize(OptimizeAction::Prune {
                older_than: Some(chrono::Duration::zero()),
                delete_unverified: Some(true),
                error_if_tagged_old_versions: None,
            })
            .await
            .map_err(to_store_error)?;
        Ok(())
    }

    pub(super) async fn index_names(&self) -> Result<Vec<String>> {
        Ok(self
            .table
            .list_indices()
            .await
            .map_err(to_store_error)?
            .into_iter()
            .map(|index| index.name)
            .collect())
    }

    pub(super) async fn doc_count(&self) -> Result<usize> {
        let _ = self.conn.uri();
        self.table.count_rows(None).await.map_err(to_store_error)
    }

    pub(super) async fn all_paths(&self) -> Result<Vec<String>> {
        let mut paths: Vec<_> = self
            .scan_paths()
            .await?
            .into_iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        paths.sort();
        Ok(paths)
    }

    async fn scan_documents(&self, filter: Option<String>) -> Result<Vec<Document>> {
        let mut query = self.table.query();
        if let Some(filter) = filter {
            query = query.only_if(filter);
        }
        let batches: Vec<RecordBatch> = query
            .execute()
            .await
            .map_err(to_store_error)?
            .try_collect()
            .await
            .map_err(to_store_error)?;

        let mut docs = Vec::new();
        for batch in batches {
            docs.extend(batch_to_documents(&batch)?);
        }
        Ok(docs)
    }

    async fn scan_paths(&self) -> Result<Vec<String>> {
        let batches: Vec<RecordBatch> = self
            .table
            .query()
            .select(Select::Columns(vec!["path".to_string()]))
            .execute()
            .await
            .map_err(to_store_error)?
            .try_collect()
            .await
            .map_err(to_store_error)?;

        let mut paths = Vec::new();
        for batch in batches {
            let path_col = string_col(&batch, "path")?;
            for row in 0..batch.num_rows() {
                paths.push(path_col.value(row).to_string());
            }
        }
        Ok(paths)
    }
}
