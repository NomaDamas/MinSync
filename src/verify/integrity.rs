//! `verify`'s embedding integrity check: two chunks whose vectors are
//! near-identical (cosine >= [`SIMILARITY_THRESHOLD`]) although their texts
//! differ point at an embedder that ignored its input.
//!
//! Comparing every pair is O(n²) and never finishes on a large index
//! (#152). Instead, rows that share a content hash and a vector collapse to
//! one representative, and random-hyperplane LSH (SimHash) proposes the
//! candidate pairs: [`TABLES`] tables, each keyed by [`BITS`] sign bits of
//! the centered unit vector. Each candidate is then confirmed with the exact
//! cosine and text test, so the check never reports a pair the all-pairs scan
//! would not. Before centering, a pair at the threshold (angle 0.0447 rad)
//! agrees on a 16-bit key with probability (1 - 0.0447/π)^16 = 0.795, so it
//! misses all 16 tables with probability 1e-11; centering widens angles near
//! the mean, and on two real 52k-row indexes (Qwen3-Embedding-0.6B,
//! EmbeddingGemma 2) 10^6 planted pairs at the threshold never missed. The
//! hyperplanes come from a fixed seed, so the result is deterministic.

use super::texts_differ_substantially;
use crate::error::Result;
use crate::types::{EmbeddingIntegrityPair, EmbeddingIntegrityResult};
use crate::vectorstore::similarity::cosine_similarity;
use crate::vectorstore::{StoredEmbedding, VectorStore};
use std::collections::{HashMap, HashSet};

pub(super) const SIMILARITY_THRESHOLD: f32 = 0.999;
/// The report lists at most this many pairs; an embedder that returns one
/// vector for every text would otherwise flag all n(n-1)/2 of them.
pub(super) const MAX_FLAGGED_PAIRS: usize = 1_000;
const TABLES: usize = 16;
const BITS: usize = 16;
const PLANES: usize = TABLES * BITS;
const SEED: u64 = 0x0152_9E37_79B9_7F4A;
/// Candidate pairs whose texts are fetched in one store round trip.
const TEXT_BATCH: usize = 1_024;

pub(super) fn check_embedding_integrity(
    store: &dyn VectorStore,
) -> Result<EmbeddingIntegrityResult> {
    let rows = store.embeddings()?;
    let reps = representatives(&rows);

    let mut by_dimension: HashMap<usize, Vec<usize>> = HashMap::new();
    for (rep, members) in reps.iter().enumerate() {
        by_dimension
            .entry(rows[members[0]].embedding.len())
            .or_default()
            .push(rep);
    }
    let mut dimensions: Vec<_> = by_dimension.into_iter().collect();
    dimensions.sort_unstable_by_key(|(dimension, _)| *dimension);

    let mut flagger = Flagger {
        store,
        rows: &rows,
        reps: &reps,
        pending: Vec::new(),
        texts: HashMap::new(),
        flagged: Vec::new(),
        truncated: false,
    };
    for (dimension, group) in dimensions {
        if !similar_pairs(&rows, &reps, dimension, &group, &mut flagger)? {
            break;
        }
    }
    flagger.finish()
}

/// Usable rows grouped by (content hash, vector): every member of a group
/// has the same text and the same vector, so one comparison covers them all.
/// Rows whose vector is empty, zero or non-finite have no cosine and are left
/// out, as the all-pairs scan never flagged them either.
fn representatives(rows: &[StoredEmbedding]) -> Vec<Vec<usize>> {
    let mut reps: Vec<Vec<usize>> = Vec::new();
    let mut by_hash: HashMap<&str, Vec<usize>> = HashMap::new();
    for (index, row) in rows.iter().enumerate() {
        let vector = &row.embedding;
        if vector.iter().any(|value| !value.is_finite()) || dot(vector, vector) == 0.0 {
            continue;
        }
        let candidates = by_hash.entry(row.content_hash.as_str()).or_default();
        match candidates
            .iter()
            .find(|&&rep| rows[reps[rep][0]].embedding == *vector)
        {
            Some(&rep) => reps[rep].push(index),
            None => {
                candidates.push(reps.len());
                reps.push(vec![index]);
            }
        }
    }
    reps
}

/// Feed every pair of representatives in `group` (all of width `dimension`)
/// whose cosine reaches the threshold to `flagger`. Returns false once the
/// flagger has stopped.
fn similar_pairs(
    rows: &[StoredEmbedding],
    reps: &[Vec<usize>],
    dimension: usize,
    group: &[usize],
    flagger: &mut Flagger<'_>,
) -> Result<bool> {
    if group.len() < 2 {
        return Ok(true);
    }
    let vectors: Vec<&[f32]> = group
        .iter()
        .map(|&rep| rows[reps[rep][0]].embedding.as_slice())
        .collect();

    // Sentence embeddings share a large common component; hashing around
    // the mean spreads them over far more buckets.
    let mut mean = vec![0.0f64; dimension];
    for vector in &vectors {
        let inverse_norm = 1.0 / f64::from(dot(vector, vector)).sqrt();
        for (sum, value) in mean.iter_mut().zip(*vector) {
            *sum += f64::from(*value) * inverse_norm;
        }
    }
    let mean: Vec<f32> = mean
        .iter()
        .map(|sum| (sum / vectors.len() as f64) as f32)
        .collect();
    let planes = hyperplanes(dimension);
    let offsets: Vec<f32> = planes
        .chunks_exact(dimension)
        .map(|plane| dot(&mean, plane))
        .collect();
    let signatures = signatures(&vectors, &planes, &offsets, dimension);

    let mut seen = HashSet::new();
    let mut keyed: Vec<(u16, u32)> = Vec::with_capacity(vectors.len());
    for table in 0..TABLES {
        keyed.clear();
        keyed.extend(
            signatures
                .iter()
                .enumerate()
                .map(|(index, signature)| (signature[table], index as u32)),
        );
        keyed.sort_unstable();
        for bucket in keyed.chunk_by(|left, right| left.0 == right.0) {
            for (offset, &(_, left)) in bucket.iter().enumerate() {
                for &(_, right) in &bucket[offset + 1..] {
                    let (left, right) = (left as usize, right as usize);
                    if cosine_similarity(vectors[left], vectors[right]) < SIMILARITY_THRESHOLD
                        || !seen.insert((left, right))
                    {
                        continue;
                    }
                    if !flagger.offer(group[left], group[right])? {
                        return Ok(false);
                    }
                }
            }
        }
    }
    Ok(true)
}

/// [`PLANES`] Gaussian directions of width `dimension`, row-major, from a
/// fixed seed (splitmix64 + Box-Muller).
fn hyperplanes(dimension: usize) -> Vec<f32> {
    let mut state = SEED ^ dimension as u64;
    let mut uniform = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // (0, 1]: never 0, so the logarithm below stays finite.
        ((z >> 11) + 1) as f64 / (1u64 << 53) as f64
    };
    (0..PLANES * dimension)
        .map(|_| {
            let radius = (-2.0 * uniform().ln()).sqrt();
            (radius * (std::f64::consts::TAU * uniform()).cos()) as f32
        })
        .collect()
}

/// One [`BITS`]-bit key per table for each vector: bit `p` is set when the
/// unit vector lies above hyperplane `p` shifted to the mean.
fn signatures(
    vectors: &[&[f32]],
    planes: &[f32],
    offsets: &[f32],
    dimension: usize,
) -> Vec<[u16; TABLES]> {
    let mut out = vec![[0u16; TABLES]; vectors.len()];
    let threads = std::thread::available_parallelism().map_or(1, |count| count.get());
    let per_thread = vectors.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (inputs, outputs) in vectors.chunks(per_thread).zip(out.chunks_mut(per_thread)) {
            scope.spawn(move || {
                for (vector, signature) in inputs.iter().zip(outputs) {
                    let inverse_norm = 1.0 / dot(vector, vector).sqrt();
                    for (plane_index, (plane, offset)) in
                        planes.chunks_exact(dimension).zip(offsets).enumerate()
                    {
                        if dot(vector, plane) * inverse_norm > *offset {
                            signature[plane_index / BITS] |= 1 << (plane_index % BITS);
                        }
                    }
                }
            });
        }
    });
    out
}

/// Dot product with eight independent accumulators so it vectorizes.
fn dot(left: &[f32], right: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let left_chunks = left.chunks_exact(8);
    let right_chunks = right.chunks_exact(8);
    let tail: f32 = left_chunks
        .remainder()
        .iter()
        .zip(right_chunks.remainder())
        .map(|(a, b)| a * b)
        .sum();
    for (a, b) in left_chunks.zip(right_chunks) {
        for lane in 0..8 {
            lanes[lane] += a[lane] * b[lane];
        }
    }
    lanes.iter().sum::<f32>() + tail
}

/// Confirms vector-similar pairs of representatives with the text test,
/// fetching texts in batches, and expands them to the rows they stand for.
struct Flagger<'a> {
    store: &'a dyn VectorStore,
    rows: &'a [StoredEmbedding],
    reps: &'a [Vec<usize>],
    pending: Vec<(usize, usize)>,
    texts: HashMap<usize, String>,
    flagged: Vec<EmbeddingIntegrityPair>,
    truncated: bool,
}

impl Flagger<'_> {
    /// Queue a pair of representatives; false once the report is full.
    fn offer(&mut self, left: usize, right: usize) -> Result<bool> {
        self.pending.push((left, right));
        if self.pending.len() >= TEXT_BATCH {
            self.resolve()?;
        }
        Ok(!self.truncated)
    }

    fn resolve(&mut self) -> Result<()> {
        let pending = std::mem::take(&mut self.pending);
        let mut missing: Vec<usize> = pending
            .iter()
            .flat_map(|&(left, right)| [left, right])
            .filter(|rep| !self.texts.contains_key(rep))
            .collect();
        missing.sort_unstable();
        missing.dedup();
        let rep_of_id: HashMap<&str, usize> = missing
            .iter()
            .map(|&rep| (self.rows[self.reps[rep][0]].id.as_str(), rep))
            .collect();
        let ids: Vec<String> = rep_of_id.keys().map(|id| id.to_string()).collect();
        for document in self.store.fetch(&ids)? {
            if let Some(&rep) = rep_of_id.get(document.id.as_str()) {
                self.texts.insert(rep, document.text);
            }
        }

        for (left, right) in pending {
            // A row deleted since the scan has no text left to compare.
            let (Some(left_text), Some(right_text)) =
                (self.texts.get(&left), self.texts.get(&right))
            else {
                continue;
            };
            if !texts_differ_substantially(left_text, right_text) {
                continue;
            }
            for &left_row in &self.reps[left] {
                for &right_row in &self.reps[right] {
                    let (first, second) = if self.rows[left_row].id < self.rows[right_row].id {
                        (&self.rows[left_row], &self.rows[right_row])
                    } else {
                        (&self.rows[right_row], &self.rows[left_row])
                    };
                    self.flagged.push(EmbeddingIntegrityPair {
                        chunk_ids: vec![first.id.clone(), second.id.clone()],
                        paths: vec![first.path.clone(), second.path.clone()],
                    });
                }
            }
            if self.flagged.len() > MAX_FLAGGED_PAIRS {
                self.truncated = true;
                return Ok(());
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<EmbeddingIntegrityResult> {
        if !self.truncated {
            self.resolve()?;
        }
        self.flagged
            .sort_unstable_by(|left, right| left.chunk_ids.cmp(&right.chunk_ids));
        self.flagged.truncate(MAX_FLAGGED_PAIRS);
        Ok(EmbeddingIntegrityResult {
            flagged_pairs: self.flagged,
            truncated: self.truncated,
        })
    }
}
