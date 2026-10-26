//! Reusable incremental-processing core: durable per-item completion state
//! and crash-resume, independent of what a stage actually produces.
//!
//! A *stage* turns stable items (usually files) into output: the vector
//! indexer writes chunks to a store, an upstream materialization stage writes
//! canonical text files. Every stage follows the same commit order, which is
//! the one invariant this module exists to enforce:
//!
//! ```text
//! read source -> compute output -> durably commit output -> append completion record
//! ```
//!
//! A completion record is appended only after the output commit returned, so
//! a crash can cause duplicate work (the commit must be idempotent) but never
//! a missing output that is treated as done.
//!
//! The core owns item identity, the append-only [`journal`], resume
//! classification, and the read-only [`view`]s a host uses to see a stage's
//! durable state without opening a run. It must not import embedders,
//! chunkers, vector stores, query code or parsers.

mod fsio;
pub mod item;
pub mod journal;
pub mod plan;
pub mod protocol;
pub mod snapshot;
pub mod stage;
pub mod view;
