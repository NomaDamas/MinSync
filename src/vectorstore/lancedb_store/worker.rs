//! Dedicated worker thread that owns the LanceDB connection. The public
//! store communicates over an mpsc command channel so the blocking LanceDB
//! runtime never nests inside a caller's tokio runtime.

use super::inner::LanceDbInner;
use super::{to_store_error, IndexingConfig};
use crate::error::Result;
use crate::vectorstore::{Document, DocumentUpdate, Filter, QueryHit};
use std::sync::mpsc;

pub(super) type Resp<T> = mpsc::Sender<Result<T>>;

pub(super) enum Command {
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

pub(super) fn run_worker(
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
