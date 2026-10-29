//! Async LanceDB operations executed on the worker thread.

use super::filters::{
    column_in_filter, filter_to_sql, has_sparse_filter, id_in_filter, missing_sparse_filter,
};
use super::index::{compact_if_fragmented, maintain_index, maintain_scalar_indexes};
use super::schema::{
    batch_to_documents, batch_to_hash_embeddings, batch_to_hash_sparse, batch_to_query_hits,
    batch_to_scored_hits, batch_to_sparse_targets, dedupe_documents, docs_to_batch, schema,
    score_sparse_batch, sparse_fields, sparse_rows_to_batch, string_col, updates_to_batch,
    validate_schema, validate_vector,
};
use super::{
    to_store_error, IndexingConfig, DISTANCE_COLUMN, DISTANCE_TYPE, SPARSE_INDICES_COLUMN,
    SPARSE_VALUES_COLUMN, TABLE_NAME, VECTOR_COLUMN,
};
use crate::error::{MinSyncError, Result};
use crate::sparse::{SparseVector, TopK};
use crate::types::IndexState;
use crate::vectorstore::{Document, DocumentUpdate, Filter, QueryHit, SparseTarget};
use arrow_array::{RecordBatch, RecordBatchIterator};
use arrow_schema::Schema;
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::table::NewColumnTransform;
use lancedb::table::OptimizeAction;
use lancedb::{Connection, Table};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub(super) struct LanceDbInner {
    conn: Connection,
    table: Table,
    dim: usize,
    indexing: IndexingConfig,
}

impl LanceDbInner {
    /// Open the table for writing (creating it and applying schema
    /// migrations when needed) or, with `read_only`, for concurrent reads
    /// alongside a running sync.
    ///
    /// Read-only mode never writes: a missing table is [`MinSyncError::NeverSynced`]
    /// instead of `create_empty_table`, and a table from before sparse
    /// retrieval is an error instead of an in-place migration. Both cases are
    /// resolved by running a sync, which holds the writer lock.
    pub(super) async fn open(
        uri: &str,
        dim: usize,
        indexing: IndexingConfig,
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
            if Self::needs_migration(&table).await? {
                if read_only {
                    return Err(MinSyncError::VectorStore(
                        "index predates sparse retrieval — run `minsync sync` once to migrate"
                            .into(),
                    ));
                }
                Self::migrate(&table).await?;
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
        })
    }

    /// A table written by the BM25 era (it has `lexical_text`) or any table
    /// without the sparse columns.
    async fn needs_migration(table: &Table) -> Result<bool> {
        let schema = table.schema().await.map_err(to_store_error)?;
        Ok(schema.field_with_name("lexical_text").is_ok()
            || sparse_fields()
                .iter()
                .any(|field| schema.field_with_name(field.name()).is_err()))
    }

    /// Turn a BM25-era table into a sparse-ready one in place: drop the FTS
    /// index, add the sparse columns (all null, so every row reads as "not
    /// encoded yet" and the next sync encodes them), then drop `lexical_text`.
    /// Every step looks at the current state, so a run that was killed half
    /// way picks up where it stopped. Dense vectors are never touched.
    async fn migrate(table: &Table) -> Result<()> {
        let indices = table.list_indices().await.map_err(to_store_error)?;
        for index in indices
            .iter()
            .filter(|index| index.columns.iter().any(|column| column == "lexical_text"))
        {
            table
                .drop_index(&index.name)
                .await
                .map_err(to_store_error)?;
        }
        let schema = table.schema().await.map_err(to_store_error)?;
        let missing: Vec<_> = sparse_fields()
            .into_iter()
            .filter(|field| schema.field_with_name(field.name()).is_err())
            .collect();
        if !missing.is_empty() {
            table
                .add_columns(
                    NewColumnTransform::AllNulls(Arc::new(Schema::new(missing))),
                    None,
                )
                .await
                .map_err(to_store_error)?;
        }
        if schema.field_with_name("lexical_text").is_ok() {
            table
                .drop_columns(&["lexical_text"])
                .await
                .map_err(to_store_error)?;
        }
        Ok(())
    }

    /// Replace-or-insert `docs` by `id` in ONE commit. (Was `delete` + `add`:
    /// two commits, and the delete scanned every fragment.)
    pub(super) async fn upsert(&self, docs: Vec<Document>) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let docs = dedupe_documents(docs);
        let batch = docs_to_batch(&docs, self.dim)?;
        self.merge_by_id(batch, true).await
    }

    /// Append documents the caller knows are absent: a plain `add`, ~5x
    /// cheaper than `merge_insert`, which has to join against the table.
    pub(super) async fn insert_new(&self, docs: Vec<Document>) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let docs = dedupe_documents(docs);
        let batch = docs_to_batch(&docs, self.dim)?;
        self.table
            .add(batch)
            .execute()
            .await
            .map_err(to_store_error)?;
        Ok(())
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

    /// Read only `content_hash` and the vector so reuse lookups never pull
    /// chunk text (which can be megabytes for delimiter-less files).
    pub(super) async fn fetch_embeddings_by_content_hash(
        &self,
        hashes: Vec<String>,
    ) -> Result<HashMap<String, Vec<f32>>> {
        let mut found = HashMap::new();
        if hashes.is_empty() {
            return Ok(found);
        }
        let batches: Vec<RecordBatch> = self
            .table
            .query()
            .only_if(column_in_filter(
                "content_hash",
                hashes.iter().map(String::as_str),
            ))
            .select(Select::Columns(vec![
                "content_hash".to_string(),
                VECTOR_COLUMN.to_string(),
            ]))
            .execute()
            .await
            .map_err(to_store_error)?
            .try_collect()
            .await
            .map_err(to_store_error)?;
        for batch in batches {
            for (hash, vector) in batch_to_hash_embeddings(&batch)? {
                if vector.iter().all(|value| value.is_finite()) {
                    found.entry(hash).or_insert(vector);
                }
            }
        }
        Ok(found)
    }

    pub(super) async fn documents(&self) -> Result<Vec<Document>> {
        self.scan_documents(None).await
    }

    /// Existence check that projects only `id`: no vectors or text are read.
    pub(super) async fn existing_ids(&self, ids: Vec<String>) -> Result<HashSet<String>> {
        if ids.is_empty() {
            return Ok(HashSet::new());
        }
        let batches: Vec<RecordBatch> = self
            .table
            .query()
            .only_if(id_in_filter(ids.iter().map(String::as_str)))
            .select(Select::Columns(vec!["id".to_string()]))
            .execute()
            .await
            .map_err(to_store_error)?
            .try_collect()
            .await
            .map_err(to_store_error)?;
        let mut found = HashSet::new();
        for batch in &batches {
            let column = string_col(batch, "id")?;
            for row in 0..batch.num_rows() {
                found.insert(column.value(row).to_string());
            }
        }
        Ok(found)
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

    /// Rows that hold no sparse vector from `model`, at most `limit`.
    pub(super) async fn rows_missing_sparse(
        &self,
        model: String,
        limit: usize,
    ) -> Result<Vec<SparseTarget>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let batches: Vec<RecordBatch> = self
            .table
            .query()
            .only_if(missing_sparse_filter(&model))
            .select(Select::columns(&["id", "path", "content_hash", "text"]))
            .limit(limit)
            .execute()
            .await
            .map_err(to_store_error)?
            .try_collect()
            .await
            .map_err(to_store_error)?;
        let mut targets = Vec::new();
        for batch in &batches {
            targets.extend(batch_to_sparse_targets(batch)?);
        }
        targets.truncate(limit);
        Ok(targets)
    }

    pub(super) async fn count_missing_sparse(&self, model: String) -> Result<usize> {
        self.table
            .count_rows(Some(missing_sparse_filter(&model)))
            .await
            .map_err(to_store_error)
    }

    /// Fill the sparse columns of existing rows in ONE commit. Ids that are
    /// not in the table are ignored, never inserted.
    pub(super) async fn set_sparse(
        &self,
        model: String,
        rows: Vec<(String, SparseVector)>,
    ) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        // merge_insert rejects several source rows for one target row.
        let mut last: HashMap<&str, &(String, SparseVector)> = HashMap::with_capacity(rows.len());
        for row in &rows {
            last.insert(row.0.as_str(), row);
        }
        let mut ordered: Vec<&(String, SparseVector)> = last.into_values().collect();
        ordered.sort_by(|a, b| a.0.cmp(&b.0));
        let batch = sparse_rows_to_batch(&model, &ordered)?;
        self.merge_by_id(batch, false).await
    }

    /// Reads only `content_hash` and the sparse columns: no chunk text.
    pub(super) async fn fetch_sparse_by_content_hash(
        &self,
        model: String,
        hashes: Vec<String>,
    ) -> Result<HashMap<String, SparseVector>> {
        let mut found = HashMap::new();
        if hashes.is_empty() {
            return Ok(found);
        }
        let predicate = format!(
            "{} AND {}",
            column_in_filter("content_hash", hashes.iter().map(String::as_str)),
            has_sparse_filter(&model)
        );
        let batches: Vec<RecordBatch> = self
            .table
            .query()
            .only_if(predicate)
            .select(Select::columns(&[
                "content_hash",
                SPARSE_INDICES_COLUMN,
                SPARSE_VALUES_COLUMN,
            ]))
            .execute()
            .await
            .map_err(to_store_error)?
            .try_collect()
            .await
            .map_err(to_store_error)?;
        for batch in &batches {
            for (hash, vector) in batch_to_hash_sparse(batch)? {
                found.entry(hash).or_insert(vector);
            }
        }
        Ok(found)
    }

    /// Exact top-`topk` by inner product. The two sparse columns of every
    /// matching row are streamed batch by batch into a bounded heap, so the
    /// table is never materialized; only the winners' metadata is read after.
    pub(super) async fn query_sparse(
        &self,
        model: String,
        vector: SparseVector,
        filter: Option<Filter>,
        topk: usize,
    ) -> Result<Vec<QueryHit>> {
        if topk == 0 || vector.is_empty() {
            return Ok(Vec::new());
        }
        let mut predicate = has_sparse_filter(&model);
        if let Some(filter) = &filter {
            predicate = format!("({}) AND {predicate}", filter_to_sql(filter)?);
        }
        let mut stream = self
            .table
            .query()
            .only_if(predicate)
            .select(Select::columns(&[
                "id",
                SPARSE_INDICES_COLUMN,
                SPARSE_VALUES_COLUMN,
            ]))
            .execute()
            .await
            .map_err(to_store_error)?;
        let mut top = TopK::new(topk);
        while let Some(batch) = stream.try_next().await.map_err(to_store_error)? {
            score_sparse_batch(&batch, &vector, &mut top)?;
        }
        let ranked = top.into_sorted();
        if ranked.is_empty() {
            return Ok(Vec::new());
        }
        let scored: HashMap<String, f32> = ranked
            .iter()
            .map(|(score, id)| (id.clone(), *score))
            .collect();
        let batches: Vec<RecordBatch> = self
            .table
            .query()
            .only_if(id_in_filter(ranked.iter().map(|(_, id)| id.as_str())))
            .select(Select::columns(&[
                "id",
                "path",
                "heading_path",
                "chunk_type",
                "text",
                "content_hash",
            ]))
            .execute()
            .await
            .map_err(to_store_error)?
            .try_collect()
            .await
            .map_err(to_store_error)?;
        let mut by_id: HashMap<String, QueryHit> = HashMap::new();
        for batch in &batches {
            for hit in batch_to_scored_hits(batch, &scored)? {
                by_id.insert(hit.doc_id.clone(), hit);
            }
        }
        // A row replaced between the scan and this read simply drops out.
        Ok(ranked
            .into_iter()
            .filter_map(|(_, id)| by_id.remove(&id))
            .collect())
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
