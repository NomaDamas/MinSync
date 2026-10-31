//! Dedicated worker thread that owns the LanceDB connection. The public
//! store communicates over an mpsc command channel so the blocking LanceDB
//! runtime never nests inside a caller's tokio runtime.

use super::inner::LanceDbInner;
use super::{to_store_error, IndexingConfig};
use crate::error::Result;
use crate::sparse::SparseVector;
use crate::types::IndexState;
use crate::vectorstore::{Document, DocumentUpdate, Filter, QueryHit, SparseTarget};
use std::collections::HashMap;
use std::sync::mpsc;

pub(super) type Resp<T> = mpsc::Sender<Result<T>>;

pub(super) enum Command {
    Upsert(Vec<Document>, Resp<()>),
    Update(Vec<DocumentUpdate>, Resp<()>),
    Fetch(Vec<String>, Resp<Vec<Document>>),
    FetchEmbeddingsByContentHash(Vec<String>, Resp<HashMap<String, Vec<f32>>>),
    Documents(Resp<Vec<Document>>),
    ExistingIds(Vec<String>, Resp<std::collections::HashSet<String>>),
    InsertNew(Vec<Document>, Resp<()>),
    Delete(Filter, Resp<usize>),
    Query {
        vector: Vec<f32>,
        filter: Option<Filter>,
        topk: usize,
        resp: Resp<Vec<QueryHit>>,
    },
    RowsMissingSparse {
        model: String,
        limit: usize,
        resp: Resp<Vec<SparseTarget>>,
    },
    CountMissingSparse(String, Resp<usize>),
    SetSparse {
        model: String,
        rows: Vec<(String, SparseVector)>,
        resp: Resp<()>,
    },
    FetchSparseByContentHash {
        model: String,
        hashes: Vec<String>,
        resp: Resp<HashMap<String, SparseVector>>,
    },
    QuerySparse {
        model: String,
        vector: SparseVector,
        filter: Option<Filter>,
        topk: usize,
        resp: Resp<Vec<QueryHit>>,
    },
    Flush(Resp<()>),
    Checkpoint(Resp<()>),
    FragmentCount(Resp<usize>),
    DocCount(Resp<usize>),
    AllPaths(Resp<Vec<String>>),
    IndexNames(Resp<Vec<String>>),
    IndexState(Resp<IndexState>),
    Shutdown,
}

pub(super) fn run_worker(
    uri: String,
    dim: usize,
    indexing: IndexingConfig,
    read_only: bool,
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

    let inner = match rt.block_on(LanceDbInner::open(&uri, dim, indexing, read_only)) {
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
            Command::FetchEmbeddingsByContentHash(hashes, resp) => send_response(
                resp,
                rt.block_on(inner.fetch_embeddings_by_content_hash(hashes)),
            ),
            Command::Documents(resp) => send_response(resp, rt.block_on(inner.documents())),
            Command::ExistingIds(ids, resp) => {
                send_response(resp, rt.block_on(inner.existing_ids(ids)))
            }
            Command::InsertNew(docs, resp) => {
                send_response(resp, rt.block_on(inner.insert_new(docs)))
            }
            Command::Delete(filter, resp) => {
                send_response(resp, rt.block_on(inner.delete_by_filter(filter)));
            }
            Command::Query {
                vector,
                filter,
                topk,
                resp,
            } => send_response(resp, rt.block_on(inner.query(vector, filter, topk))),
            Command::RowsMissingSparse { model, limit, resp } => {
                send_response(resp, rt.block_on(inner.rows_missing_sparse(model, limit)))
            }
            Command::CountMissingSparse(model, resp) => {
                send_response(resp, rt.block_on(inner.count_missing_sparse(model)))
            }
            Command::SetSparse { model, rows, resp } => {
                send_response(resp, rt.block_on(inner.set_sparse(model, rows)))
            }
            Command::FetchSparseByContentHash {
                model,
                hashes,
                resp,
            } => send_response(
                resp,
                rt.block_on(inner.fetch_sparse_by_content_hash(model, hashes)),
            ),
            Command::QuerySparse {
                model,
                vector,
                filter,
                topk,
                resp,
            } => send_response(
                resp,
                rt.block_on(inner.query_sparse(model, vector, filter, topk)),
            ),
            Command::Flush(resp) => send_response(resp, rt.block_on(inner.flush())),
            Command::Checkpoint(resp) => send_response(resp, rt.block_on(inner.checkpoint())),
            Command::FragmentCount(resp) => {
                send_response(resp, rt.block_on(inner.fragment_count()));
            }
            Command::DocCount(resp) => send_response(resp, rt.block_on(inner.doc_count())),
            Command::AllPaths(resp) => send_response(resp, rt.block_on(inner.all_paths())),
            Command::IndexNames(resp) => send_response(resp, rt.block_on(inner.index_names())),
            Command::IndexState(resp) => send_response(resp, rt.block_on(inner.index_state())),
            Command::Shutdown => break,
        }
    }
}

fn send_response<T>(resp: Resp<T>, result: Result<T>) {
    let _ = resp.send(result);
}
