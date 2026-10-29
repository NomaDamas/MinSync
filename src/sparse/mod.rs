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
        dot_terms(&self.indices, &self.values, &other.indices, &other.values)
    }
}

/// Inner product of two term lists, each strictly ascending by index. This is
/// the one scoring loop: [`SparseVector::dot`] and the stores that score rows
/// straight from their column buffers both run it.
pub fn dot_terms(
    left_indices: &[u32],
    left_values: &[f32],
    right_indices: &[u32],
    right_values: &[f32],
) -> f32 {
    let (mut left, mut right) = (0, 0);
    let mut sum = 0.0;
    while left < left_indices.len() && right < right_indices.len() {
        match left_indices[left].cmp(&right_indices[right]) {
            std::cmp::Ordering::Less => left += 1,
            std::cmp::Ordering::Greater => right += 1,
            std::cmp::Ordering::Equal => {
                sum += left_values[left] * right_values[right];
                left += 1;
                right += 1;
            }
        }
    }
    sum
}

/// The `k` best `(score, id)` pairs of a stream: highest score first, equal
/// scores toward the smaller id. A score that is not finite and positive never
/// qualifies, so a row that shares no term with the query is not a hit.
///
/// Memory is `O(k)` however many rows are offered, which is what lets a store
/// score a table batch by batch instead of materializing it.
pub struct TopK {
    k: usize,
    heap: std::collections::BinaryHeap<std::cmp::Reverse<Ranked>>,
}

#[derive(PartialEq, Eq)]
struct Ranked {
    score: OrderedScore,
    id: String,
}

/// `f32` ordered by [`f32::total_cmp`] (scores here are finite, so it is the
/// usual order).
#[derive(PartialEq, Eq)]
struct OrderedScore(u32);

impl OrderedScore {
    fn new(score: f32) -> Self {
        // Monotonic map from the IEEE bit pattern to an unsigned order.
        let bits = score.to_bits();
        Self(if bits >> 31 == 1 {
            !bits
        } else {
            bits | (1 << 31)
        })
    }
}

impl Ord for OrderedScore {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for OrderedScore {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ranked {
    /// Greater is better: a higher score, then the smaller id.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .cmp(&other.score)
            .then_with(|| other.id.cmp(&self.id))
    }
}

impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl TopK {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            heap: std::collections::BinaryHeap::new(),
        }
    }

    /// Offer one scored row. Allocates only when the row is kept.
    pub fn offer(&mut self, score: f32, id: &str) {
        if self.k == 0 || !score.is_finite() || score <= 0.0 {
            return;
        }
        let candidate = OrderedScore::new(score);
        if self.heap.len() == self.k {
            let worst = &self.heap.peek().expect("heap holds k entries").0;
            let better = match candidate.cmp(&worst.score) {
                std::cmp::Ordering::Greater => true,
                std::cmp::Ordering::Equal => id < worst.id.as_str(),
                std::cmp::Ordering::Less => false,
            };
            if !better {
                return;
            }
            self.heap.pop();
        }
        self.heap.push(std::cmp::Reverse(Ranked {
            score: candidate,
            id: id.to_string(),
        }));
    }

    /// The kept rows, best first, with their scores.
    pub fn into_sorted(self) -> Vec<(f32, String)> {
        let mut kept: Vec<Ranked> = self.heap.into_iter().map(|entry| entry.0).collect();
        kept.sort_by(|a, b| b.cmp(a));
        kept.into_iter()
            .map(|ranked| {
                let bits = ranked.score.0;
                let bits = if bits >> 31 == 1 {
                    bits & !(1 << 31)
                } else {
                    !bits
                };
                (f32::from_bits(bits), ranked.id)
            })
            .collect()
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

#[cfg(test)]
mod scoring_tests {
    use super::*;

    #[test]
    fn dot_terms_is_the_sum_over_shared_terms() {
        assert_eq!(
            dot_terms(&[1, 3, 5], &[2.0, 1.0, 4.0], &[3, 4, 5], &[10.0, 7.0, 0.5]),
            12.0
        );
        assert_eq!(dot_terms(&[], &[], &[1], &[1.0]), 0.0);
        assert_eq!(dot_terms(&[1], &[1.0], &[2], &[1.0]), 0.0);
    }

    #[test]
    fn top_k_returns_the_best_first_and_breaks_ties_toward_the_smaller_id() {
        let mut top = TopK::new(3);
        for (score, id) in [(1.0, "d"), (3.0, "b"), (3.0, "a"), (2.0, "c"), (0.5, "e")] {
            top.offer(score, id);
        }
        assert_eq!(
            top.into_sorted(),
            vec![
                (3.0, "a".to_string()),
                (3.0, "b".to_string()),
                (2.0, "c".to_string())
            ]
        );
    }

    #[test]
    fn top_k_never_keeps_a_row_that_scores_zero_or_is_not_finite() {
        let mut top = TopK::new(5);
        for score in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            top.offer(score, "x");
        }
        top.offer(0.25, "y");
        assert_eq!(top.into_sorted(), vec![(0.25, "y".to_string())]);
        assert!(TopK::new(0).into_sorted().is_empty());
    }

    #[test]
    fn top_k_equals_sorting_everything_for_many_random_streams() {
        // Deterministic LCG: no dev-dependency on rand for one property.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for round in 0..200 {
            let rows: Vec<(f32, String)> = (0..(round % 40 + 1))
                .map(|i| {
                    // Few distinct scores so ties are common.
                    let score = (next() % 6) as f32 * 0.5;
                    (score, format!("id{:03}", (i * 7 + next() as usize) % 90))
                })
                .collect();
            let k = (next() % 8) as usize;
            let mut top = TopK::new(k);
            for (score, id) in &rows {
                top.offer(*score, id);
            }
            let mut expected: Vec<(f32, String)> =
                rows.into_iter().filter(|(score, _)| *score > 0.0).collect();
            expected.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            expected.truncate(k);
            assert_eq!(top.into_sorted(), expected, "round {round}, k {k}");
        }
    }
}
