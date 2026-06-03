use crate::error::{MinSyncError, Result};
use crate::vectorstore::{Document, DocumentUpdate, Filter, QueryHit, VectorStore};
use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{Array, FixedSizeListArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::TryStreamExt;
use lancedb::index::vector::IvfHnswSqIndexBuilder;
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::table::{OptimizeAction, OptimizeOptions};
use lancedb::{Connection, DistanceType, Table};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};

const DEFAULT_DIMENSION: usize = 1536;
const TABLE_NAME: &str = "documents";
const VECTOR_COLUMN: &str = "vector";
const DISTANCE_COLUMN: &str = "_distance";

/// The distance metric used for BOTH index construction and query time. These
/// must match: an index trained with one metric returns invalid results when
/// searched with another. Cosine matches the in-memory store's scoring.
const DISTANCE_TYPE: DistanceType = DistanceType::Cosine;

const DEFAULT_INDEX_BUILD_THRESHOLD: usize = 256;
const DEFAULT_INDEX_OPTIMIZE_DELTA_THRESHOLD: usize = 10_000;

/// ANN index maintenance thresholds. These are tuning knobs, not capacity
/// limits: any number of vectors can be stored and searched regardless of these
/// values. They only decide *when* index maintenance runs.
///
/// The two thresholds are NOT a min/max pair on the same quantity. They gate
/// two distinct, mutually exclusive events that measure different things:
/// `index_build_threshold` triggers the one-time `create_index` (compared
/// against TOTAL rows), while `index_optimize_delta_threshold` triggers the
/// recurring incremental `optimize` (compared against UNINDEXED-delta rows).
#[derive(Debug, Clone, Copy)]
pub struct IndexingConfig {
    /// Build the ANN index once the table's TOTAL row count reaches this value.
    /// Below it, an exact flat scan is fast enough and IVF lacks the data to
    /// train good partitions. This gates a one-time `create_index`.
    pub index_build_threshold: usize,
    /// Fold the unindexed delta into the existing index once that delta reaches
    /// this value. The delta counts only rows not yet covered by the index (not
    /// the total row count). New rows stay searchable (via flat scan over the
    /// delta) throughout; this bounds how large that flat portion grows before a
    /// recurring incremental `optimize` runs. Raise it to optimize less often,
    /// lower it to keep query latency tighter.
    pub index_optimize_delta_threshold: usize,
}

impl Default for IndexingConfig {
    fn default() -> Self {
        Self {
            index_build_threshold: DEFAULT_INDEX_BUILD_THRESHOLD,
            index_optimize_delta_threshold: DEFAULT_INDEX_OPTIMIZE_DELTA_THRESHOLD,
        }
    }
}

impl IndexingConfig {
    /// Parse indexing knobs from `[vectorstore.options]`. Unknown/missing keys
    /// fall back to defaults; `0` is rejected since it would index empty tables
    /// or optimize on every flush.
    pub fn from_options(options: Option<&toml::Value>) -> Result<Self> {
        let mut config = Self::default();
        let Some(options) = options else {
            return Ok(config);
        };
        if let Some(value) = options.get("index_build_threshold") {
            config.index_build_threshold = parse_positive_usize(value, "index_build_threshold")?;
        }
        if let Some(value) = options.get("index_optimize_delta_threshold") {
            config.index_optimize_delta_threshold =
                parse_positive_usize(value, "index_optimize_delta_threshold")?;
        }
        Ok(config)
    }
}

fn parse_positive_usize(value: &toml::Value, key: &str) -> Result<usize> {
    let integer = value.as_integer().ok_or_else(|| {
        MinSyncError::Config(format!("vectorstore.options.{key} must be an integer"))
    })?;
    let parsed = usize::try_from(integer)
        .map_err(|_| MinSyncError::Config(format!("vectorstore.options.{key} must be positive")))?;
    if parsed == 0 {
        return Err(MinSyncError::Config(format!(
            "vectorstore.options.{key} must be greater than 0"
        )));
    }
    Ok(parsed)
}

type Resp<T> = mpsc::Sender<Result<T>>;

enum Command {
    Upsert(Vec<Document>, Resp<()>),
    Update(Vec<DocumentUpdate>, Resp<()>),
    Fetch(Vec<String>, Resp<Vec<Document>>),
    Delete(Filter, Resp<usize>),
    Query {
        vector: Vec<f32>,
        filter: Option<Filter>,
        topk: usize,
        resp: Resp<Vec<QueryHit>>,
    },
    Flush(Resp<()>),
    DocCount(Resp<usize>),
    AllPaths(Resp<Vec<String>>),
    IndexNames(Resp<Vec<String>>),
    Shutdown,
}

pub struct LanceDbStore {
    tx: mpsc::Sender<Command>,
    worker: Option<JoinHandle<()>>,
    dim: usize,
}

struct LanceDbInner {
    conn: Connection,
    table: Table,
    dim: usize,
    indexing: IndexingConfig,
}

impl LanceDbStore {
    pub fn open_or_create(path: &Path, dimension: usize) -> Result<Self> {
        Self::open_with_indexing(path, dimension, IndexingConfig::default())
    }

    pub fn open_with_indexing(
        path: &Path,
        dimension: usize,
        indexing: IndexingConfig,
    ) -> Result<Self> {
        let dim = if dimension == 0 {
            DEFAULT_DIMENSION
        } else {
            dimension
        };
        validate_dimension(dim)?;
        let uri = path
            .to_str()
            .ok_or_else(|| MinSyncError::VectorStore("LanceDB path is not valid UTF-8".into()))?
            .to_string();

        let (cmd_tx, cmd_rx) = mpsc::channel();
        let (init_tx, init_rx) = mpsc::channel();
        let worker = thread::spawn(move || run_worker(uri, dim, indexing, cmd_rx, init_tx));

        match init_rx.recv().map_err(to_store_error)? {
            Ok(()) => Ok(Self {
                tx: cmd_tx,
                worker: Some(worker),
                dim,
            }),
            Err(error) => {
                let _ = worker.join();
                Err(error)
            }
        }
    }

    pub fn open_with_default_dimension(path: &Path) -> Result<Self> {
        Self::open_or_create(path, DEFAULT_DIMENSION)
    }

    pub fn dimension_from_options(options: Option<&toml::Value>) -> Result<usize> {
        let Some(options) = options else {
            return Ok(DEFAULT_DIMENSION);
        };
        let Some(value) = options.get("dimension") else {
            return Ok(DEFAULT_DIMENSION);
        };
        let Some(dimension) = value.as_integer() else {
            return Err(MinSyncError::Config(
                "vectorstore.options.dimension must be an integer".into(),
            ));
        };
        let dimension = usize::try_from(dimension).map_err(|_| {
            MinSyncError::Config("vectorstore.options.dimension must be positive".into())
        })?;
        if dimension == 0 {
            return Err(MinSyncError::Config(
                "vectorstore.options.dimension must be positive".into(),
            ));
        }
        validate_dimension(dimension).map_err(|error| MinSyncError::Config(error.to_string()))?;
        Ok(dimension)
    }

    pub fn dimension(&self) -> usize {
        self.dim
    }

    pub fn index_names(&self) -> Result<Vec<String>> {
        self.request(Command::IndexNames)
    }

    fn request<T>(&self, make: impl FnOnce(Resp<T>) -> Command) -> Result<T> {
        let (resp_tx, resp_rx) = mpsc::channel();
        self.tx.send(make(resp_tx)).map_err(to_store_error)?;
        resp_rx.recv().map_err(to_store_error)?
    }
}

impl Drop for LanceDbStore {
    fn drop(&mut self) {
        let _ = self.tx.send(Command::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl VectorStore for LanceDbStore {
    fn upsert(&mut self, docs: &[Document]) -> Result<()> {
        self.request(|resp| Command::Upsert(docs.to_vec(), resp))
    }

    fn update(&mut self, updates: &[DocumentUpdate]) -> Result<()> {
        self.request(|resp| Command::Update(updates.to_vec(), resp))
    }

    fn fetch(&self, ids: &[String]) -> Result<Vec<Document>> {
        self.request(|resp| Command::Fetch(ids.to_vec(), resp))
    }

    fn delete_by_filter(&mut self, filter: &Filter) -> Result<usize> {
        self.request(|resp| Command::Delete(filter.clone(), resp))
    }

    fn query(&self, vector: &[f32], filter: Option<&Filter>, topk: usize) -> Result<Vec<QueryHit>> {
        self.request(|resp| Command::Query {
            vector: vector.to_vec(),
            filter: filter.cloned(),
            topk,
            resp,
        })
    }

    fn flush(&mut self) -> Result<()> {
        self.request(Command::Flush)
    }

    fn doc_count(&self) -> usize {
        match self.request(Command::DocCount) {
            Ok(count) => count,
            Err(error) => {
                tracing::warn!(%error, "failed to count LanceDB documents");
                0
            }
        }
    }

    fn all_paths(&self) -> Vec<String> {
        match self.request(Command::AllPaths) {
            Ok(paths) => paths,
            Err(error) => {
                tracing::warn!(%error, "failed to list LanceDB paths");
                Vec::new()
            }
        }
    }
}

fn run_worker(
    uri: String,
    dim: usize,
    indexing: IndexingConfig,
    cmd_rx: mpsc::Receiver<Command>,
    init_tx: mpsc::Sender<Result<()>>,
) {
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(to_store_error)
    {
        Ok(rt) => rt,
        Err(error) => {
            let _ = init_tx.send(Err(error));
            return;
        }
    };

    let inner = match rt.block_on(LanceDbInner::open_or_create(&uri, dim, indexing)) {
        Ok(inner) => inner,
        Err(error) => {
            let _ = init_tx.send(Err(error));
            return;
        }
    };

    if init_tx.send(Ok(())).is_err() {
        return;
    }

    while let Ok(command) = cmd_rx.recv() {
        match command {
            Command::Upsert(docs, resp) => send_response(resp, rt.block_on(inner.upsert(docs))),
            Command::Update(updates, resp) => {
                send_response(resp, rt.block_on(inner.update(updates)))
            }
            Command::Fetch(ids, resp) => send_response(resp, rt.block_on(inner.fetch(ids))),
            Command::Delete(filter, resp) => {
                send_response(resp, rt.block_on(inner.delete_by_filter(filter)));
            }
            Command::Query {
                vector,
                filter,
                topk,
                resp,
            } => send_response(resp, rt.block_on(inner.query(vector, filter, topk))),
            Command::Flush(resp) => send_response(resp, rt.block_on(inner.flush())),
            Command::DocCount(resp) => send_response(resp, rt.block_on(inner.doc_count())),
            Command::AllPaths(resp) => send_response(resp, rt.block_on(inner.all_paths())),
            Command::IndexNames(resp) => send_response(resp, rt.block_on(inner.index_names())),
            Command::Shutdown => break,
        }
    }
}

fn send_response<T>(resp: Resp<T>, result: Result<T>) {
    let _ = resp.send(result);
}

impl LanceDbInner {
    async fn open_or_create(uri: &str, dim: usize, indexing: IndexingConfig) -> Result<Self> {
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

    async fn upsert(&self, docs: Vec<Document>) -> Result<()> {
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

    async fn update(&self, updates: Vec<DocumentUpdate>) -> Result<()> {
        for update in updates {
            let filter = format!("id = '{}'", escape_sql_literal(&update.id));
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

    async fn fetch(&self, ids: Vec<String>) -> Result<Vec<Document>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let filter = id_in_filter(ids.iter().map(String::as_str));
        let mut docs = self.scan_documents(Some(filter)).await?;
        docs.sort_by_key(|doc| ids.iter().position(|id| *id == doc.id).unwrap_or(ids.len()));
        Ok(docs)
    }

    async fn delete_by_filter(&self, filter: Filter) -> Result<usize> {
        let sql = filter_to_sql(&filter)?;
        let result = self.table.delete(&sql).await.map_err(to_store_error)?;
        usize::try_from(result.num_deleted_rows).map_err(to_store_error)
    }

    async fn query(
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
            let ids = string_col(&batch, "id")?;
            let paths = string_col(&batch, "path")?;
            let headings = string_col(&batch, "heading_path")?;
            let chunk_types = string_col(&batch, "chunk_type")?;
            let texts = string_col(&batch, "text")?;
            let hashes = string_col(&batch, "content_hash")?;
            let distances = batch
                .column_by_name(DISTANCE_COLUMN)
                .ok_or_else(|| missing_column(DISTANCE_COLUMN))?
                .as_primitive::<Float32Type>();

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

    async fn flush(&self) -> Result<()> {
        self.table
            .optimize(OptimizeAction::Compact {
                options: lancedb::table::CompactionOptions::default(),
                remap_options: None,
            })
            .await
            .map_err(to_store_error)?;
        self.maintain_index().await?;
        Ok(())
    }

    /// Ensure the vector column is ANN-indexed and keep that index current.
    ///
    /// LanceDB never auto-indexes inserted rows: a fresh table answers queries
    /// with an exhaustive `KNNFlatSearch`, and rows added after `create_index`
    /// stay in an unindexed delta that is flat-scanned on every query. This
    /// method (a) builds the index once enough rows exist, then (b) folds the
    /// unindexed delta into the existing index via an incremental `optimize`
    /// (no k-means retrain) once the delta grows large enough to hurt latency.
    async fn maintain_index(&self) -> Result<()> {
        let existing = self
            .table
            .list_indices()
            .await
            .map_err(to_store_error)?
            .into_iter()
            .find(|index| index.columns.iter().any(|column| column == VECTOR_COLUMN));

        match existing {
            None => {
                let total_rows = self.table.count_rows(None).await.map_err(to_store_error)?;
                if total_rows >= self.indexing.index_build_threshold {
                    self.table
                        .create_index(&[VECTOR_COLUMN], self.vector_index())
                        .execute()
                        .await
                        .map_err(to_store_error)?;
                }
            }
            Some(index) => {
                let unindexed = self
                    .table
                    .index_stats(&index.name)
                    .await
                    .map_err(to_store_error)?
                    .map(|stats| stats.num_unindexed_rows)
                    .unwrap_or(0);
                if unindexed >= self.indexing.index_optimize_delta_threshold {
                    self.table
                        .optimize(OptimizeAction::Index(OptimizeOptions::default()))
                        .await
                        .map_err(to_store_error)?;
                }
            }
        }
        Ok(())
    }

    /// IVF-HNSW-SQ index pinned to [`DISTANCE_TYPE`]. The metric is set
    /// explicitly because the builder defaults to L2, which would silently
    /// mismatch the Cosine queries this store issues.
    fn vector_index(&self) -> Index {
        Index::IvfHnswSq(IvfHnswSqIndexBuilder::default().distance_type(DISTANCE_TYPE))
    }

    async fn index_names(&self) -> Result<Vec<String>> {
        Ok(self
            .table
            .list_indices()
            .await
            .map_err(to_store_error)?
            .into_iter()
            .map(|index| index.name)
            .collect())
    }

    async fn doc_count(&self) -> Result<usize> {
        let _ = self.conn.uri();
        self.table.count_rows(None).await.map_err(to_store_error)
    }

    async fn all_paths(&self) -> Result<Vec<String>> {
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
            let ids = string_col(&batch, "id")?;
            let vectors = batch
                .column_by_name(VECTOR_COLUMN)
                .ok_or_else(|| missing_column(VECTOR_COLUMN))?
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| {
                    MinSyncError::VectorStore("vector column has unexpected type".into())
                })?;
            let text = string_col(&batch, "text")?;
            let source_ids = string_col(&batch, "source_id")?;
            let paths = string_col(&batch, "path")?;
            let schemas = string_col(&batch, "chunk_schema_id")?;
            let chunk_types = string_col(&batch, "chunk_type")?;
            let headings = string_col(&batch, "heading_path")?;
            let hashes = string_col(&batch, "content_hash")?;
            let tokens = string_col(&batch, "seen_token")?;

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

fn schema(dim: usize) -> Result<SchemaRef> {
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

fn validate_schema(schema: &SchemaRef, dim: usize) -> Result<()> {
    validate_dimension(dim)?;
    for field_name in FILTERABLE_FIELDS {
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

fn docs_to_batch(docs: &[Document], dim: usize) -> Result<RecordBatch> {
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

fn dedupe_documents(docs: Vec<Document>) -> Vec<Document> {
    let mut last_by_id = HashMap::new();
    for (index, doc) in docs.iter().enumerate() {
        last_by_id.insert(doc.id.clone(), index);
    }
    docs.into_iter()
        .enumerate()
        .filter_map(|(index, doc)| (last_by_id.get(&doc.id) == Some(&index)).then_some(doc))
        .collect()
}

pub fn filter_to_sql(filter: &Filter) -> Result<String> {
    match filter {
        Filter::Eq(field, value) => Ok(format!(
            "{} = '{}'",
            validate_filter_field(field)?,
            escape_sql_literal(value)
        )),
        Filter::Neq(field, value) => Ok(format!(
            "{} != '{}'",
            validate_filter_field(field)?,
            escape_sql_literal(value)
        )),
        Filter::And(filters) if filters.is_empty() => Ok("TRUE".to_string()),
        Filter::And(filters) => filters
            .iter()
            .map(|nested| filter_to_sql(nested).map(|sql| format!("({sql})")))
            .collect::<Result<Vec<_>>>()
            .map(|parts| parts.join(" AND ")),
    }
}

const FILTERABLE_FIELDS: &[&str] = &[
    "id",
    "text",
    "source_id",
    "path",
    "chunk_schema_id",
    "chunk_type",
    "heading_path",
    "content_hash",
    "seen_token",
];

fn validate_filter_field(field: &str) -> Result<&str> {
    if FILTERABLE_FIELDS.contains(&field) {
        Ok(field)
    } else {
        Err(MinSyncError::VectorStore(format!(
            "unsupported LanceDB filter field: {field}"
        )))
    }
}

fn id_in_filter<'a>(ids: impl Iterator<Item = &'a str>) -> String {
    let values = ids
        .map(|id| format!("'{}'", escape_sql_literal(id)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("id IN ({values})")
}

fn validate_vector(vector: &[f32], dim: usize, label: &str) -> Result<()> {
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

fn validate_dimension(dim: usize) -> Result<()> {
    if dim == 0 || dim > i32::MAX as usize {
        return Err(MinSyncError::VectorStore(format!(
            "invalid LanceDB vector dimension: {dim}"
        )));
    }
    Ok(())
}

fn sql_literal(value: &str) -> String {
    format!("'{}'", escape_sql_literal(value))
}

fn escape_sql_literal(value: &str) -> String {
    value.replace('\'', "''")
}

fn string_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    batch
        .column_by_name(name)
        .ok_or_else(|| missing_column(name))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| MinSyncError::VectorStore(format!("column {name} has unexpected type")))
}

fn missing_column(name: &str) -> MinSyncError {
    MinSyncError::VectorStore(format!("missing LanceDB column {name}"))
}

fn to_store_error(error: impl std::fmt::Display) -> MinSyncError {
    MinSyncError::VectorStore(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn doc(id: &str, path: &str, seen_token: &str, embedding: Vec<f32>) -> Document {
        Document {
            id: id.to_string(),
            embedding,
            text: format!("text {id}"),
            source_id: "source-1".to_string(),
            path: path.to_string(),
            chunk_schema_id: "schema-1".to_string(),
            chunk_type: "text".to_string(),
            heading_path: format!("heading {id}"),
            content_hash: format!("hash-{id}"),
            seen_token: seen_token.to_string(),
        }
    }

    fn store() -> (TempDir, LanceDbStore) {
        let dir = tempfile::tempdir().expect("create tempdir");
        let store = LanceDbStore::open_or_create(dir.path(), 4).expect("create lancedb store");
        (dir, store)
    }

    #[test]
    fn test_upsert_and_fetch() {
        let (_dir, mut store) = store();
        let docs = vec![
            doc("a", "a.txt", "token-1", vec![1.0, 0.0, 0.0, 0.0]),
            doc("b", "b.txt", "token-1", vec![0.0, 1.0, 0.0, 0.0]),
            doc("c", "c.txt", "token-1", vec![0.0, 0.0, 1.0, 0.0]),
        ];
        store.upsert(&docs).expect("upsert docs");
        let fetched = store
            .fetch(&["a".to_string(), "b".to_string(), "c".to_string()])
            .expect("fetch docs");
        assert_eq!(store.doc_count(), 3);
        assert_eq!(fetched.len(), 3);
        assert_eq!(fetched[0].id, "a");
        assert_eq!(fetched[1].embedding, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_upsert_replaces() {
        let (_dir, mut store) = store();
        store
            .upsert(&[doc("a", "old.txt", "old", vec![1.0, 0.0, 0.0, 0.0])])
            .expect("upsert old");
        store
            .upsert(&[doc("a", "new.txt", "new", vec![0.0, 1.0, 0.0, 0.0])])
            .expect("upsert new");
        let fetched = store.fetch(&["a".to_string()]).expect("fetch doc");
        assert_eq!(store.doc_count(), 1);
        assert_eq!(fetched[0].path, "new.txt");
        assert_eq!(fetched[0].seen_token, "new");
        assert_eq!(fetched[0].embedding, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_upsert_dedupes_input_batch_last_wins() {
        let (_dir, mut store) = store();
        store
            .upsert(&[
                doc("a", "old.txt", "old", vec![1.0, 0.0, 0.0, 0.0]),
                doc("a", "new.txt", "new", vec![0.0, 1.0, 0.0, 0.0]),
            ])
            .expect("upsert duplicate docs");
        let fetched = store.fetch(&["a".to_string()]).expect("fetch doc");
        assert_eq!(store.doc_count(), 1);
        assert_eq!(fetched[0].path, "new.txt");
    }

    #[test]
    fn test_update_metadata() {
        let (_dir, mut store) = store();
        store
            .upsert(&[doc("a", "old.txt", "old", vec![1.0, 0.0, 0.0, 0.0])])
            .expect("upsert doc");
        store
            .update(&[DocumentUpdate {
                id: "a".to_string(),
                seen_token: "new".to_string(),
                path: "new.txt".to_string(),
                heading_path: "new heading".to_string(),
            }])
            .expect("update doc");
        let fetched = store.fetch(&["a".to_string()]).expect("fetch doc");
        assert_eq!(fetched[0].seen_token, "new");
        assert_eq!(fetched[0].path, "new.txt");
        assert_eq!(fetched[0].heading_path, "new heading");
    }

    #[test]
    fn test_delete_by_eq_filter() {
        let (_dir, mut store) = store();
        store
            .upsert(&[
                doc("a", "x.txt", "token", vec![1.0, 0.0, 0.0, 0.0]),
                doc("b", "x.txt", "token", vec![0.0, 1.0, 0.0, 0.0]),
                doc("c", "y.txt", "token", vec![0.0, 0.0, 1.0, 0.0]),
            ])
            .expect("upsert docs");
        let deleted = store
            .delete_by_filter(&Filter::Eq("path".to_string(), "x.txt".to_string()))
            .expect("delete docs");
        assert_eq!(deleted, 2);
        assert_eq!(store.doc_count(), 1);
    }

    #[test]
    fn test_delete_by_and_filter() {
        let (_dir, mut store) = store();
        store
            .upsert(&[
                doc("a", "same.txt", "keep", vec![1.0, 0.0, 0.0, 0.0]),
                doc("b", "same.txt", "drop", vec![0.0, 1.0, 0.0, 0.0]),
                doc("c", "other.txt", "drop", vec![0.0, 0.0, 1.0, 0.0]),
            ])
            .expect("upsert docs");
        let deleted = store
            .delete_by_filter(&Filter::And(vec![
                Filter::Eq("path".to_string(), "same.txt".to_string()),
                Filter::Neq("seen_token".to_string(), "keep".to_string()),
            ]))
            .expect("delete docs");
        assert_eq!(deleted, 1);
        assert_eq!(store.doc_count(), 2);
        assert!(store.fetch(&["b".to_string()]).expect("fetch b").is_empty());
    }

    #[test]
    fn test_query_topk() {
        let (_dir, mut store) = store();
        store
            .upsert(&[
                doc("near", "path.txt", "token", vec![1.0, 0.0, 0.0, 0.0]),
                doc("mid", "path.txt", "token", vec![0.7, 0.7, 0.0, 0.0]),
                doc("far", "path.txt", "token", vec![0.0, 1.0, 0.0, 0.0]),
            ])
            .expect("upsert docs");
        let hits = store
            .query(&[1.0, 0.0, 0.0, 0.0], None, 2)
            .expect("query docs");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "near");
        assert!(hits[0].score >= hits[1].score);
        assert!(hits[0].score > 0.99);
    }

    #[test]
    fn test_query_with_filter() {
        let (_dir, mut store) = store();
        store
            .upsert(&[
                doc("a", "x.txt", "token", vec![1.0, 0.0, 0.0, 0.0]),
                doc("b", "y.txt", "token", vec![1.0, 0.0, 0.0, 0.0]),
                doc("c", "x.txt", "token", vec![0.0, 1.0, 0.0, 0.0]),
            ])
            .expect("upsert docs");
        let hits = store
            .query(
                &[1.0, 0.0, 0.0, 0.0],
                Some(&Filter::Eq("path".to_string(), "x.txt".to_string())),
                10,
            )
            .expect("query docs");
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|hit| hit.path == "x.txt"));
    }

    #[test]
    fn test_query_topk_zero_and_non_finite_rejected() {
        let (_dir, mut store) = store();
        store
            .upsert(&[doc("a", "x.txt", "token", vec![1.0, 0.0, 0.0, 0.0])])
            .expect("upsert docs");
        assert!(store
            .query(&[f32::NAN, 0.0, 0.0, 0.0], None, 0)
            .expect("zero topk")
            .is_empty());
        assert!(store.query(&[f32::NAN, 0.0, 0.0, 0.0], None, 1).is_err());
    }

    #[test]
    fn test_upsert_rejects_non_finite_embedding() {
        let (_dir, mut store) = store();
        assert!(store
            .upsert(&[doc(
                "a",
                "x.txt",
                "token",
                vec![f32::INFINITY, 0.0, 0.0, 0.0]
            )])
            .is_err());
    }

    #[test]
    fn test_filter_to_sql() {
        assert_eq!(
            filter_to_sql(&Filter::Eq("path".to_string(), "a'b".to_string())).expect("sql"),
            "path = 'a''b'"
        );
        assert_eq!(
            filter_to_sql(&Filter::Neq("seen_token".to_string(), "old".to_string())).expect("sql"),
            "seen_token != 'old'"
        );
        assert_eq!(
            filter_to_sql(&Filter::And(vec![
                Filter::Eq("path".to_string(), "x.txt".to_string()),
                Filter::Neq("seen_token".to_string(), "keep".to_string()),
            ]))
            .expect("sql"),
            "(path = 'x.txt') AND (seen_token != 'keep')"
        );
        assert_eq!(filter_to_sql(&Filter::And(vec![])).expect("sql"), "TRUE");
        assert!(filter_to_sql(&Filter::Eq("bad".to_string(), "x".to_string())).is_err());
    }

    #[test]
    fn test_all_paths() {
        let (_dir, mut store) = store();
        store
            .upsert(&[
                doc("a", "b.txt", "token", vec![1.0, 0.0, 0.0, 0.0]),
                doc("b", "a.txt", "token", vec![0.0, 1.0, 0.0, 0.0]),
                doc("c", "c.txt", "token", vec![0.0, 0.0, 1.0, 0.0]),
                doc("d", "a.txt", "token", vec![0.0, 0.0, 0.0, 1.0]),
            ])
            .expect("upsert docs");
        assert_eq!(store.all_paths(), vec!["a.txt", "b.txt", "c.txt"]);
    }

    #[test]
    fn test_persistence_roundtrip() {
        let dir = tempfile::tempdir().expect("create tempdir");
        {
            let mut store = LanceDbStore::open_or_create(dir.path(), 4).expect("create store");
            store
                .upsert(&[doc("a", "a.txt", "token", vec![1.0, 0.0, 0.0, 0.0])])
                .expect("upsert doc");
        }
        let loaded = LanceDbStore::open_or_create(dir.path(), 4).expect("reopen store");
        let fetched = loaded.fetch(&["a".to_string()]).expect("fetch loaded doc");
        assert_eq!(fetched.len(), 1);
        assert_eq!(fetched[0].path, "a.txt");
        assert_eq!(fetched[0].embedding, vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_open_existing_dimension_mismatch_errors() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let store = LanceDbStore::open_or_create(dir.path(), 4).expect("create store");
        drop(store);
        assert!(LanceDbStore::open_or_create(dir.path(), 3).is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_store_works_inside_multithread_tokio_runtime() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let mut store = LanceDbStore::open_or_create(dir.path(), 4).expect("create store");
        store
            .upsert(&[doc("a", "a.txt", "token", vec![1.0, 0.0, 0.0, 0.0])])
            .expect("upsert doc");
        let hits = store
            .query(&[1.0, 0.0, 0.0, 0.0], None, 1)
            .expect("query doc");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "a");
    }

    fn indexed_store(dir: &TempDir, index_build_threshold: usize) -> LanceDbStore {
        LanceDbStore::open_with_indexing(
            dir.path(),
            4,
            IndexingConfig {
                index_build_threshold,
                index_optimize_delta_threshold: 1,
            },
        )
        .expect("create lancedb store")
    }

    fn spread_doc(i: usize) -> Document {
        let angle = i as f32 * 0.013;
        doc(
            &format!("doc-{i}"),
            "corpus.txt",
            "token",
            vec![angle.cos(), angle.sin(), 0.0, 0.0],
        )
    }

    #[test]
    fn test_indexing_config_from_options() {
        let mut table = toml::value::Table::new();
        table.insert("index_build_threshold".into(), toml::Value::Integer(50));
        table.insert(
            "index_optimize_delta_threshold".into(),
            toml::Value::Integer(2000),
        );
        let config = IndexingConfig::from_options(Some(&toml::Value::Table(table)))
            .expect("parse indexing options");
        assert_eq!(config.index_build_threshold, 50);
        assert_eq!(config.index_optimize_delta_threshold, 2000);

        let defaults = IndexingConfig::from_options(None).expect("defaults");
        assert_eq!(
            defaults.index_build_threshold,
            DEFAULT_INDEX_BUILD_THRESHOLD
        );
        assert_eq!(
            defaults.index_optimize_delta_threshold,
            DEFAULT_INDEX_OPTIMIZE_DELTA_THRESHOLD
        );
    }

    #[test]
    fn test_indexing_config_rejects_zero() {
        let mut table = toml::value::Table::new();
        table.insert("index_build_threshold".into(), toml::Value::Integer(0));
        assert!(IndexingConfig::from_options(Some(&toml::Value::Table(table))).is_err());
    }

    #[test]
    fn test_no_index_built_below_threshold() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let mut store = indexed_store(&dir, 1000);
        let docs: Vec<_> = (0..50).map(spread_doc).collect();
        store.upsert(&docs).expect("upsert docs");
        store.flush().expect("flush");
        assert!(
            store.index_names().expect("list indices").is_empty(),
            "no ANN index should exist below the row threshold"
        );
    }

    #[test]
    fn test_index_built_above_threshold_and_query_still_accurate() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let mut store = indexed_store(&dir, 256);
        let docs: Vec<_> = (0..300).map(spread_doc).collect();
        store.upsert(&docs).expect("upsert docs");
        store.flush().expect("flush builds index");

        assert!(
            !store.index_names().expect("list indices").is_empty(),
            "ANN index should be built once the threshold is crossed"
        );

        // The nearest vector to doc-0's embedding is doc-0 itself. A Cosine-built
        // index queried with Cosine must return it first; an L2/Cosine mismatch
        // would scramble this ranking.
        let target = spread_doc(0).embedding;
        let hits = store.query(&target, None, 1).expect("query");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-0");
        assert!(hits[0].score > 0.99);
    }

    #[test]
    fn test_rows_added_after_index_remain_searchable() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let mut store = indexed_store(&dir, 256);
        let docs: Vec<_> = (0..300).map(spread_doc).collect();
        store.upsert(&docs).expect("upsert initial docs");
        store.flush().expect("flush builds index");

        store
            .upsert(&[doc("late", "corpus.txt", "token", vec![0.0, 0.0, 1.0, 0.0])])
            .expect("upsert late doc");
        store.flush().expect("flush folds delta");

        let hits = store
            .query(&[0.0, 0.0, 1.0, 0.0], None, 1)
            .expect("query late doc");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "late");
    }
}
