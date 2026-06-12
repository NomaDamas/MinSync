//! ANN index configuration and maintenance policy.

use super::{to_store_error, DISTANCE_TYPE, VECTOR_COLUMN};
use crate::error::{MinSyncError, Result};
use lancedb::index::vector::IvfHnswSqIndexBuilder;
use lancedb::index::Index;
use lancedb::table::{OptimizeAction, OptimizeOptions};
use lancedb::Table;

pub(super) const DEFAULT_INDEX_BUILD_THRESHOLD: usize = 256;
pub(super) const DEFAULT_INDEX_OPTIMIZE_DELTA_THRESHOLD: usize = 10_000;

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

/// Ensure the vector column is ANN-indexed and keep that index current.
///
/// LanceDB never auto-indexes inserted rows: a fresh table answers queries
/// with an exhaustive `KNNFlatSearch`, and rows added after `create_index`
/// stay in an unindexed delta that is flat-scanned on every query. This
/// function (a) builds the index once enough rows exist, then (b) folds the
/// unindexed delta into the existing index via an incremental `optimize`
/// (no k-means retrain) once the delta grows large enough to hurt latency.
pub(super) async fn maintain_index(table: &Table, indexing: &IndexingConfig) -> Result<()> {
    let existing = table
        .list_indices()
        .await
        .map_err(to_store_error)?
        .into_iter()
        .find(|index| index.columns.iter().any(|column| column == VECTOR_COLUMN));

    match existing {
        None => {
            let total_rows = table.count_rows(None).await.map_err(to_store_error)?;
            if total_rows >= indexing.index_build_threshold {
                table
                    .create_index(&[VECTOR_COLUMN], vector_index())
                    .execute()
                    .await
                    .map_err(to_store_error)?;
            }
        }
        Some(index) => {
            let unindexed = table
                .index_stats(&index.name)
                .await
                .map_err(to_store_error)?
                .map(|stats| stats.num_unindexed_rows)
                .unwrap_or(0);
            if unindexed >= indexing.index_optimize_delta_threshold {
                table
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
fn vector_index() -> Index {
    Index::IvfHnswSq(IvfHnswSqIndexBuilder::default().distance_type(DISTANCE_TYPE))
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_indexing_config_rejects_non_integer() {
        let mut table = toml::value::Table::new();
        table.insert(
            "index_optimize_delta_threshold".into(),
            toml::Value::String("many".into()),
        );
        assert!(IndexingConfig::from_options(Some(&toml::Value::Table(table))).is_err());
    }

    #[test]
    fn test_indexing_config_rejects_negative() {
        let mut table = toml::value::Table::new();
        table.insert("index_build_threshold".into(), toml::Value::Integer(-5));
        assert!(IndexingConfig::from_options(Some(&toml::Value::Table(table))).is_err());
    }
}
