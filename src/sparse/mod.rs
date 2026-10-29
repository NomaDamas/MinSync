//! Learned sparse (SPLADE) retrieval: the vector type and the encoder
//! contract. Providers live in [`native`] (in-process candle), [`tei`]
//! (remote `/embed_sparse`) and [`served`] (a running `minsync serve`).

pub mod modernbert;
pub mod native;
pub mod served;
pub mod tei;

use crate::error::{MinSyncError, Result};
use async_trait::async_trait;
use serde::{Deserialize, Deserializer, Serialize};

/// A sparse term-weight vector: `indices` strictly ascending, every weight
/// finite and greater than zero (a zero weight is the absence of a term).
///
/// The invariant is what lets [`SparseVector::dot`] merge two vectors in one
/// pass and lets a store treat an entry list as an inverted-index posting set.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct SparseVector {
    indices: Vec<u32>,
    values: Vec<f32>,
}

impl SparseVector {
    /// Validate and wrap parallel `indices` / `values`.
    pub fn new(indices: Vec<u32>, values: Vec<f32>) -> Result<Self> {
        if indices.len() != values.len() {
            return Err(MinSyncError::Embedding(format!(
                "sparse vector has {} indices but {} values",
                indices.len(),
                values.len()
            )));
        }
        if let Some(pair) = indices.windows(2).find(|pair| pair[0] >= pair[1]) {
            return Err(MinSyncError::Embedding(format!(
                "sparse vector indices must be strictly ascending (saw {} then {})",
                pair[0], pair[1]
            )));
        }
        if let Some(value) = values.iter().find(|v| !v.is_finite() || **v <= 0.0) {
            return Err(MinSyncError::Embedding(format!(
                "sparse vector weights must be finite and positive (saw {value})"
            )));
        }
        Ok(Self { indices, values })
    }

    /// Build from arbitrary `(index, weight)` pairs: sorts, keeps the largest
    /// weight of a repeated index, and drops weights that are not finite and
    /// positive. Never fails; for producers that already compute weights.
    pub fn from_pairs(pairs: impl IntoIterator<Item = (u32, f32)>) -> Self {
        let mut pairs: Vec<(u32, f32)> = pairs
            .into_iter()
            .filter(|(_, weight)| weight.is_finite() && *weight > 0.0)
            .collect();
        pairs.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)));
        pairs.dedup_by_key(|pair| pair.0);
        Self {
            indices: pairs.iter().map(|pair| pair.0).collect(),
            values: pairs.iter().map(|pair| pair.1).collect(),
        }
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Inner product: the SPLADE relevance score.
    pub fn dot(&self, other: &Self) -> f32 {
        let (mut left, mut right) = (0, 0);
        let mut sum = 0.0;
        while left < self.indices.len() && right < other.indices.len() {
            match self.indices[left].cmp(&other.indices[right]) {
                std::cmp::Ordering::Less => left += 1,
                std::cmp::Ordering::Greater => right += 1,
                std::cmp::Ordering::Equal => {
                    sum += self.values[left] * other.values[right];
                    left += 1;
                    right += 1;
                }
            }
        }
        sum
    }
}

impl<'de> Deserialize<'de> for SparseVector {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            indices: Vec<u32>,
            values: Vec<f32>,
        }
        let raw = Raw::deserialize(deserializer)?;
        Self::new(raw.indices, raw.values).map_err(serde::de::Error::custom)
    }
}

/// Produces [`SparseVector`]s. Mirrors [`crate::embedder::Embedder`]: passages
/// at index time, queries at search time (the model marks the two
/// differently), no query-time shortcut.
#[async_trait]
pub trait SparseEncoder: Send + Sync {
    /// Model identity recorded in the cursor (`naver/...`, without provider).
    fn id(&self) -> &str;

    /// Encode passages, one vector per input, in order.
    async fn encode(&self, texts: &[String]) -> Result<Vec<SparseVector>>;

    /// Encode one search query.
    async fn encode_query(&self, text: &str) -> Result<SparseVector>;

    /// Passages the model had to cut to fit its context.
    async fn count_truncated(&self, _texts: &[String]) -> Result<usize> {
        Ok(0)
    }

    fn max_length(&self) -> Option<usize> {
        None
    }

    /// True when re-running identical input is guaranteed to give identical
    /// output (in-process inference on a fixed device).
    fn is_deterministic(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_accepts_ascending_unique_positive_finite_entries() {
        let vector = SparseVector::new(vec![1, 4, 9], vec![0.5, 2.0, 0.25]).expect("valid");
        assert_eq!(vector.indices(), &[1, 4, 9]);
        assert_eq!(vector.values(), &[0.5, 2.0, 0.25]);
        assert_eq!(vector.len(), 3);
        assert!(!vector.is_empty());
    }

    #[test]
    fn new_rejects_every_malformed_shape() {
        for (indices, values) in [
            (vec![1, 2], vec![1.0]),        // length mismatch
            (vec![2, 1], vec![1.0, 1.0]),   // descending
            (vec![1, 1], vec![1.0, 1.0]),   // duplicate
            (vec![1], vec![0.0]),           // zero weight is not a term
            (vec![1], vec![-0.5]),          // negative
            (vec![1], vec![f32::NAN]),      // non-finite
            (vec![1], vec![f32::INFINITY]), // non-finite
        ] {
            assert!(
                SparseVector::new(indices.clone(), values.clone()).is_err(),
                "{indices:?} {values:?} must be rejected"
            );
        }
    }

    #[test]
    fn from_pairs_sorts_merges_duplicates_by_max_and_drops_unusable_weights() {
        let vector = SparseVector::from_pairs([
            (7, 1.0),
            (3, 0.5),
            (7, 2.5),
            (5, 0.0),
            (6, -1.0),
            (8, f32::NAN),
        ]);
        assert_eq!(vector.indices(), &[3, 7]);
        assert_eq!(vector.values(), &[0.5, 2.5]);
    }

    #[test]
    fn dot_sums_only_the_shared_terms() {
        let left = SparseVector::new(vec![1, 3, 5], vec![2.0, 1.0, 4.0]).unwrap();
        let right = SparseVector::new(vec![3, 4, 5], vec![10.0, 7.0, 0.5]).unwrap();
        assert_eq!(left.dot(&right), 1.0 * 10.0 + 4.0 * 0.5);
        assert_eq!(right.dot(&left), left.dot(&right));
        assert_eq!(left.dot(&SparseVector::default()), 0.0);
    }

    #[test]
    fn json_roundtrip_revalidates_on_the_way_in() {
        let vector = SparseVector::new(vec![2, 9], vec![1.5, 0.5]).unwrap();
        let json = serde_json::to_string(&vector).unwrap();
        assert_eq!(serde_json::from_str::<SparseVector>(&json).unwrap(), vector);
        assert!(
            serde_json::from_str::<SparseVector>(r#"{"indices":[2,1],"values":[1.0,1.0]}"#)
                .is_err(),
            "a descending index list must not deserialize"
        );
    }
}
