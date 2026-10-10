//! Async LanceDB operations executed on the worker thread.

use super::filters::{filter_to_sql, id_in_filter};
use super::index::{compact_if_fragmented, maintain_index, maintain_scalar_indexes};
use super::schema::{
    batch_to_documents, batch_to_fts_hits, batch_to_query_hits, dedupe_documents, docs_to_batch,
    schema, string_col, updates_to_batch, validate_schema, validate_vector,
};
use super::{
    to_store_error, IndexingConfig, DISTANCE_COLUMN, DISTANCE_TYPE, TABLE_NAME, VECTOR_COLUMN,
};
use crate::error::{MinSyncError, Result};
use crate::types::IndexState;
use crate::vectorstore::{Document, DocumentUpdate, Filter, QueryHit};
use arrow_array::{RecordBatch, RecordBatchIterator};
use futures::TryStreamExt;
use lancedb::index::scalar::FullTextSearchQuery;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::table::NewColumnTransform;
use lancedb::table::OptimizeAction;
use lancedb::{Connection, Table};
use std::cmp::Ordering;
use std::collections::HashSet;

pub(super) struct LanceDbInner {
    conn: Connection,
    table: Table,
    dim: usize,
    indexing: IndexingConfig,
    language: String,
}

impl LanceDbInner {
    /// Open the table for writing (creating it and applying schema
    /// migrations when needed) or, with `read_only`, for concurrent reads
    /// alongside a running sync.
    ///
    /// Read-only mode never writes: a missing table is [`MinSyncError::NeverSynced`]
    /// instead of `create_empty_table`, and a pre-`lexical_text` table is an
    /// error instead of an in-place `add_columns` migration. Both cases are
    /// resolved by running a sync, which holds the writer lock.
    pub(super) async fn open(
        uri: &str,
        dim: usize,
        indexing: IndexingConfig,
        language: String,
        read_only: bool,
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
            if table
                .schema()
                .await
                .map_err(to_store_error)?
                .field_with_name("lexical_text")
                .is_err()
            {
                if read_only {
                    return Err(MinSyncError::VectorStore(
                        "index predates lexical search — run `minsync sync` once to migrate".into(),
                    ));
                }
                table
                    .add_columns(
                        NewColumnTransform::SqlExpressions(vec![(
                            "lexical_text".to_string(),
                            "text".to_string(),
                        )]),
                        None,
                    )
                    .await
                    .map_err(to_store_error)?;
            }
            validate_schema(&table.schema().await.map_err(to_store_error)?, dim)?;
            table
        } else if read_only {
            return Err(MinSyncError::NeverSynced);
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
            language,
        })
    }

    /// Replace-or-insert `docs` by `id` in ONE commit. (Was `delete` + `add`:
    /// two commits, and the delete scanned every fragment.)
    pub(super) async fn upsert(&self, docs: Vec<Document>) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let docs = dedupe_documents(docs);
        let batch = docs_to_batch(&docs, self.dim, &self.language)?;
        self.merge_by_id(batch, true).await
    }

    /// Rewrite `seen_token`/`path`/`heading_path` for every listed id in ONE
    /// commit. Ids not present in the table are ignored, never inserted.
    pub(super) async fn update(&self, updates: Vec<DocumentUpdate>) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }
        // merge_insert rejects several source rows matching one target row;
        // keep the last update per id, like `dedupe_documents` does for upsert.
        let mut last: std::collections::HashMap<&str, &DocumentUpdate> =
            std::collections::HashMap::with_capacity(updates.len());
        for update in &updates {
            last.insert(update.id.as_str(), update);
        }
        let mut ordered: Vec<&DocumentUpdate> = last.into_values().collect();
        ordered.sort_by(|a, b| a.id.cmp(&b.id));
        let batch = updates_to_batch(&ordered)?;
        self.merge_by_id(batch, false).await
    }

    async fn merge_by_id(&self, batch: RecordBatch, insert_new: bool) -> Result<()> {
        let batch_schema = batch.schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], batch_schema);
        let mut merge = self.table.merge_insert(&["id"]);
        merge.when_matched_update_all(None);
        if insert_new {
            merge.when_not_matched_insert_all();
        }
        merge
            .execute(Box::new(reader))
            .await
            .map_err(to_store_error)?;
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
        // A delete that removes nothing still commits a new table version.
        // The per-file stale sweep matches nothing for most files, so probe
        // first (an index lookup once `id`/`path` are indexed) to keep those
        // from piling up versions.
        if self
            .table
            .count_rows(Some(sql.clone()))
            .await
            .map_err(to_store_error)?
            == 0
        {
            return Ok(0);
        }
        let result = self.table.delete(&sql).await.map_err(to_store_error)?;
        usize::try_from(result.num_deleted_rows).map_err(to_store_error)
    }

    pub(super) async fn fragment_count(&self) -> Result<usize> {
        Ok(self
            .table
            .stats()
            .await
            .map_err(to_store_error)?
            .fragment_stats
            .num_fragments)
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

    pub(super) async fn query_text(
        &self,
        text: String,
        filter: Option<Filter>,
        topk: usize,
    ) -> Result<Vec<QueryHit>> {
        if topk == 0 {
            return Ok(Vec::new());
        }
        let sql_filter = filter.as_ref().map(filter_to_sql).transpose()?;
        let tokenized_text = crate::tokenizer::tokenize(&text, &self.language)?;
        let mut query = self
            .table
            .query()
            .full_text_search(
                FullTextSearchQuery::new(tokenized_text)
                    .with_column("lexical_text".to_string())
                    .map_err(to_store_error)?,
            )
            .limit(topk)
            .select(Select::columns(&[
                "id",
                "path",
                "heading_path",
                "chunk_type",
                "text",
                "content_hash",
                "_score",
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
            hits.extend(batch_to_fts_hits(&batch)?);
        }
        hits.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.doc_id.cmp(&right.doc_id))
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
        maintain_scalar_indexes(&self.table).await?;
        maintain_index(&self.table, &self.indexing).await?;
        self.prune().await
    }

    /// Mid-sync maintenance: keep `id`/`path` scalar indexes current (so the
    /// per-file filters stay index lookups instead of fragment scans), compact
    /// when fragments pile up, and prune old versions. Never retrains the
    /// ANN/FTS indexes — that stays in [`Self::flush`].
    pub(super) async fn checkpoint(&self) -> Result<()> {
        compact_if_fragmented(&self.table, self.indexing.compact_fragment_threshold).await?;
        maintain_scalar_indexes(&self.table).await?;
        self.prune().await
    }

    async fn prune(&self) -> Result<()> {
        // Keep recent versions alive long enough for lock-free concurrent
        // readers (`minsync query` no longer takes the writer lock): a reader
        // pins the latest version at open time, so pruning only versions older
        // than this window guarantees an in-flight query's files survive.
        // `delete_unverified: false` never deletes files Lance could not
        // verify as unreferenced.
        self.table
            .optimize(OptimizeAction::Prune {
                older_than: Some(chrono::Duration::minutes(10)),
                delete_unverified: Some(false),
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

    pub(super) async fn index_state(&self) -> Result<IndexState> {
        let indices = self.table.list_indices().await.map_err(to_store_error)?;
        let fts_indexed = indices
            .iter()
            .any(|index| index.columns.iter().any(|column| column == "lexical_text"));
        let ann = indices
            .iter()
            .find(|index| index.columns.iter().any(|column| column == VECTOR_COLUMN));
        let unindexed_rows = match ann {
            Some(index) => self
                .table
                .index_stats(&index.name)
                .await
                .map_err(to_store_error)?
                .map(|stats| stats.num_unindexed_rows),
            None => None,
        };
        Ok(IndexState {
            fts_indexed,
            ann_indexed: ann.is_some(),
            unindexed_rows,
        })
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
