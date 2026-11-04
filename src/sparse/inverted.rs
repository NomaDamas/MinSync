//! An exact on-disk inverted index over sparse vectors.
//!
//! Scoring a query against every stored vector costs `N x (q + d)` steps. The
//! inverted index keeps, per term, the rows that hold it and their weights, so
//! a query reads only the posting lists of its own terms (about 2% of all
//! postings for a 150-term SPLADE query) and adds `query weight x row weight`
//! into one accumulator per row. The scores are the inner products the full
//! scan computes, summed in the same order, so the ranking is identical.
//!
//! File layout (little endian), written once and never modified:
//!
//! ```text
//! header (88 B)  magic, format, table version + timestamp, counts, lengths
//! meta           JSON: model and source id
//! directory      per term: first posting (u64), posting count (u32), pad (u32)
//! id offsets     (rows + 1) x u64 into the id blob
//! id blob        row ids, in ascending order
//! postings       per term, contiguous: (row slot u32, weight f32)
//! ```
//!
//! A row's *slot* is its rank in ascending id order, so "equal scores go to the
//! smaller id" (what [`super::TopK`] does) is "equal scores go to the smaller
//! slot", and ranking never needs the id strings.
//!
//! A query reads the directory entries and posting lists of its own terms with
//! positioned reads, so opening a 600 MB index costs a header read. The writer
//! keeps at most `memory_postings` postings in memory: terms are processed in
//! ranges, one scan of the rows per range, and each range is appended to the
//! file in term order.

use super::SparseVector;
use crate::error::{MinSyncError, Result};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

const MAGIC: &[u8; 8] = b"MSINVIDX";
const FORMAT: u32 = 1;
const HEADER_LEN: usize = 88;
const DIR_ENTRY: usize = 16;
const POSTING: usize = 8;

fn corrupt(what: impl std::fmt::Display) -> MinSyncError {
    MinSyncError::VectorStore(format!("sparse inverted index: {what}"))
}

/// What an index was built from. A reader compares it with the table it is
/// about to answer for: any difference means the index may not describe it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexMeta {
    /// Sparse model whose vectors were indexed.
    pub model: String,
    /// The one `source_id` every indexed row carries.
    pub source_id: String,
    /// Version of the table the rows were read from.
    pub table_version: u64,
    /// Creation time of that version, in nanoseconds.
    pub table_timestamp_nanos: u128,
}

#[derive(Serialize, Deserialize)]
struct StoredMeta {
    model: String,
    source_id: String,
}

/// What a finished build wrote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexStats {
    pub rows: usize,
    pub postings: u64,
    pub bytes: u64,
    /// Scans of the rows after the counting scan.
    pub passes: usize,
}

/// Builds an index in a bounded amount of memory.
///
/// The caller drives it with the same sequence of rows several times:
///
/// 1. [`IndexBuilder::count_row`] for every row, then [`IndexBuilder::plan`],
///    which returns how many passes follow;
/// 2. for each pass, [`IndexBuilder::add_row`] for every row in the order of
///    step 1, then [`IndexBuilder::end_pass`];
/// 3. [`IndexBuilder::finish`].
///
/// Rows without terms may be passed in every step; they are ignored.
pub struct IndexBuilder {
    final_path: PathBuf,
    model: String,
    table_version: u64,
    table_timestamp_nanos: u128,
    memory_postings: usize,

    ids: Vec<String>,
    source_id: Option<String>,
    counts: Vec<u32>,

    out: Option<BufWriter<NamedTempFile>>,
    rows: usize,
    slot_by_id: std::collections::HashMap<String, u32>,
    seen: Vec<bool>,
    ranges: Vec<(u32, u32)>,
    postings_total: u64,
    bytes_total: u64,
    written: u64,
    pass: usize,

    flat: Vec<(u32, f32)>,
    start: Vec<usize>,
    fill: Vec<usize>,
    ordinal: usize,
}

impl IndexBuilder {
    /// A builder that will write `path` atomically. `memory_postings` bounds
    /// the postings held in memory per pass (8 bytes each); a single term with
    /// more postings than that still forms one pass of its own.
    pub fn new(
        path: &Path,
        model: &str,
        table_version: u64,
        table_timestamp_nanos: u128,
        memory_postings: usize,
    ) -> Result<Self> {
        if path.parent().is_none() {
            return Err(corrupt("the index path has no directory"));
        }
        Ok(Self {
            final_path: path.to_path_buf(),
            model: model.to_string(),
            table_version,
            table_timestamp_nanos,
            memory_postings: memory_postings.max(1),
            ids: Vec::new(),
            source_id: None,
            counts: Vec::new(),
            out: None,
            rows: 0,
            slot_by_id: std::collections::HashMap::new(),
            seen: Vec::new(),
            ranges: Vec::new(),
            postings_total: 0,
            bytes_total: 0,
            written: 0,
            pass: 0,
            flat: Vec::new(),
            start: Vec::new(),
            fill: Vec::new(),
            ordinal: 0,
        })
    }

    pub fn count_row(&mut self, id: &str, source_id: &str, indices: &[u32]) -> Result<()> {
        if indices.is_empty() {
            return Ok(());
        }
        match &self.source_id {
            None => self.source_id = Some(source_id.to_string()),
            Some(seen) if seen != source_id => {
                return Err(corrupt(format!(
                    "rows from several sources ({seen:?} and {source_id:?}) cannot share one index"
                )));
            }
            Some(_) => {}
        }
        if self.ids.len() >= u32::MAX as usize {
            return Err(corrupt("more than 2^32 rows"));
        }
        self.ids.push(id.to_string());
        if let Some(&last) = indices.last() {
            if last as usize >= self.counts.len() {
                self.counts.resize(last as usize + 1, 0);
            }
        }
        for &term in indices {
            self.counts[term as usize] += 1;
        }
        Ok(())
    }

    /// Fix the layout and write everything but the postings. Returns the
    /// number of passes that follow (zero when no row has a term).
    pub fn plan(&mut self) -> Result<usize> {
        let rows = self.ids.len();
        if rows == 0 {
            return Ok(0);
        }
        let mut order: Vec<u32> = (0..rows as u32).collect();
        order.sort_unstable_by(|&a, &b| self.ids[a as usize].cmp(&self.ids[b as usize]));
        if let Some(pair) = order
            .windows(2)
            .find(|pair| self.ids[pair[0] as usize] == self.ids[pair[1] as usize])
        {
            return Err(corrupt(format!(
                "row id {:?} appears twice",
                self.ids[pair[0] as usize]
            )));
        }
        self.rows = rows;
        self.slot_by_id.reserve(rows);
        for (slot, &ordinal) in order.iter().enumerate() {
            self.slot_by_id
                .insert(self.ids[ordinal as usize].clone(), slot as u32);
        }
        self.seen = vec![false; rows];

        let meta = serde_json::to_vec(&StoredMeta {
            model: self.model.clone(),
            source_id: self.source_id.clone().unwrap_or_default(),
        })
        .map_err(corrupt)?;
        let mut id_offsets = Vec::with_capacity(rows + 1);
        let mut blob_len = 0u64;
        id_offsets.push(0u64);
        for &ordinal in &order {
            blob_len += self.ids[ordinal as usize].len() as u64;
            id_offsets.push(blob_len);
        }
        self.postings_total = self.counts.iter().map(|&c| c as u64).sum();
        let terms = self.counts.len() as u64;
        self.bytes_total = HEADER_LEN as u64
            + meta.len() as u64
            + terms * DIR_ENTRY as u64
            + (rows as u64 + 1) * 8
            + blob_len
            + self.postings_total * POSTING as u64;

        let dir = self
            .final_path
            .parent()
            .ok_or_else(|| corrupt("the index path has no directory"))?;
        let temp = NamedTempFile::new_in(dir)?;
        let mut out = BufWriter::with_capacity(1 << 20, temp);
        let mut header = Vec::with_capacity(HEADER_LEN);
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&FORMAT.to_le_bytes());
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(&self.table_version.to_le_bytes());
        header.extend_from_slice(&self.table_timestamp_nanos.to_le_bytes());
        header.extend_from_slice(&(rows as u64).to_le_bytes());
        header.extend_from_slice(&terms.to_le_bytes());
        header.extend_from_slice(&self.postings_total.to_le_bytes());
        header.extend_from_slice(&(meta.len() as u64).to_le_bytes());
        header.extend_from_slice(&blob_len.to_le_bytes());
        header.extend_from_slice(&self.bytes_total.to_le_bytes());
        debug_assert_eq!(header.len(), HEADER_LEN);
        out.write_all(&header)?;
        out.write_all(&meta)?;
        let mut first = 0u64;
        for &count in &self.counts {
            out.write_all(&first.to_le_bytes())?;
            out.write_all(&count.to_le_bytes())?;
            out.write_all(&0u32.to_le_bytes())?;
            first += count as u64;
        }
        for offset in &id_offsets {
            out.write_all(&offset.to_le_bytes())?;
        }
        for &ordinal in &order {
            out.write_all(self.ids[ordinal as usize].as_bytes())?;
        }
        self.written = self.bytes_total - self.postings_total * POSTING as u64;
        self.out = Some(out);
        self.ids = Vec::new();

        // Term ranges whose postings fit the budget; one pass per range.
        let (mut low, mut total) = (0u32, 0usize);
        for (term, &count) in self.counts.iter().enumerate() {
            let count = count as usize;
            if total > 0 && total + count > self.memory_postings {
                self.ranges.push((low, term as u32));
                low = term as u32;
                total = 0;
            }
            total += count;
        }
        self.ranges.push((low, self.counts.len() as u32));
        self.begin_pass();
        Ok(self.ranges.len())
    }

    fn begin_pass(&mut self) {
        let (low, high) = self.ranges[self.pass];
        self.start.clear();
        let mut at = 0usize;
        for term in low..high {
            self.start.push(at);
            at += self.counts[term as usize] as usize;
        }
        self.start.push(at);
        self.fill = self.start[..self.start.len() - 1].to_vec();
        self.flat.clear();
        self.flat.resize(at, (0, 0.0));
        self.ordinal = 0;
        self.seen.iter_mut().for_each(|seen| *seen = false);
    }

    /// Add one row of a pass. Rows are matched by `id`, not by position: the
    /// table does not promise that two scans return them in the same order.
    pub fn add_row(&mut self, id: &str, indices: &[u32], values: &[f32]) -> Result<()> {
        if indices.is_empty() {
            return Ok(());
        }
        if indices.len() != values.len() {
            return Err(corrupt("a row has a different number of terms and weights"));
        }
        let slot = *self
            .slot_by_id
            .get(id)
            .ok_or_else(|| corrupt(format!("row {id:?} was not in the counting scan")))?;
        if self.seen[slot as usize] {
            return Err(corrupt(format!("row {id:?} appeared twice in one pass")));
        }
        self.seen[slot as usize] = true;
        self.ordinal += 1;
        let (low, high) = self.ranges[self.pass];
        let from = indices.partition_point(|&term| term < low);
        let to = indices.partition_point(|&term| term < high);
        for at in from..to {
            let term = (indices[at] - low) as usize;
            let cursor = self.fill[term];
            if cursor >= self.start[term + 1] {
                return Err(corrupt(
                    "a pass saw a term more often than the counting scan",
                ));
            }
            self.flat[cursor] = (slot, values[at]);
            self.fill[term] = cursor + 1;
        }
        Ok(())
    }

    /// Append the pass's postings to the file.
    pub fn end_pass(&mut self) -> Result<()> {
        if self.ordinal != self.rows {
            return Err(corrupt("a pass saw fewer rows than the counting scan"));
        }
        if self
            .fill
            .iter()
            .zip(&self.start[1..])
            .any(|(cursor, end)| cursor != end)
        {
            return Err(corrupt(
                "a pass saw a term less often than the counting scan",
            ));
        }
        let out = self
            .out
            .as_mut()
            .ok_or_else(|| corrupt("end_pass before plan"))?;
        // Rows arrive in whatever order the table scan returns them; slots
        // are fixed by id, so sorting each list makes the file the same one
        // however the scan was ordered (and keeps lists ascending by row).
        for window in self.start.windows(2) {
            self.flat[window[0]..window[1]].sort_unstable_by_key(|&(slot, _)| slot);
        }
        let mut chunk = Vec::with_capacity(64 << 10);
        for &(slot, weight) in &self.flat {
            chunk.extend_from_slice(&slot.to_le_bytes());
            chunk.extend_from_slice(&weight.to_le_bytes());
            if chunk.len() >= 64 << 10 {
                out.write_all(&chunk)?;
                chunk.clear();
            }
        }
        out.write_all(&chunk)?;
        self.written += self.flat.len() as u64 * POSTING as u64;
        self.pass += 1;
        if self.pass < self.ranges.len() {
            self.begin_pass();
        } else {
            self.flat = Vec::new();
        }
        Ok(())
    }

    /// Move the finished file into place. `None` when no row had a term: there
    /// is nothing to index and no file is written.
    pub fn finish(mut self) -> Result<Option<IndexStats>> {
        let Some(out) = self.out.take() else {
            return Ok(None);
        };
        if self.pass != self.ranges.len() || self.written != self.bytes_total {
            return Err(corrupt("the index was not written completely"));
        }
        let temp = out
            .into_inner()
            .map_err(|error| MinSyncError::Io(error.into_error()))?;
        temp.as_file().sync_all()?;
        temp.persist(&self.final_path)
            .map_err(|error| MinSyncError::Io(error.error))?;
        Ok(Some(IndexStats {
            rows: self.rows,
            postings: self.postings_total,
            bytes: self.bytes_total,
            passes: self.ranges.len(),
        }))
    }
}

#[cfg(unix)]
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    let mut done = 0;
    while done < buf.len() {
        let read =
            std::os::windows::fs::FileExt::seek_read(file, &mut buf[done..], offset + done as u64)?;
        if read == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        done += read;
    }
    Ok(())
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("8 bytes"))
}

/// A built index, opened for queries.
pub struct InvertedFile {
    file: File,
    meta: IndexMeta,
    rows: usize,
    terms: usize,
    directory_at: u64,
    id_offsets_at: u64,
    id_blob_at: u64,
    postings_at: u64,
}

impl InvertedFile {
    /// Open and validate an index: magic, format, and that the file has
    /// exactly the length its header promises (a torn or truncated file does
    /// not).
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let length = file.metadata()?.len();
        let mut header = [0u8; HEADER_LEN];
        read_exact_at(&file, &mut header, 0)
            .map_err(|_| corrupt("file shorter than its header"))?;
        if &header[..8] != MAGIC {
            return Err(corrupt("not an index file"));
        }
        let format = u32::from_le_bytes(header[8..12].try_into().expect("4 bytes"));
        if format != FORMAT {
            return Err(corrupt(format!("unsupported format {format}")));
        }
        let table_version = u64_at(&header, 16);
        let table_timestamp_nanos =
            u128::from_le_bytes(header[24..40].try_into().expect("16 bytes"));
        let rows = u64_at(&header, 40);
        let terms = u64_at(&header, 48);
        let postings = u64_at(&header, 56);
        let meta_len = u64_at(&header, 64);
        let blob_len = u64_at(&header, 72);
        let total = u64_at(&header, 80);
        let expected = HEADER_LEN as u64
            + meta_len
            + terms * DIR_ENTRY as u64
            + (rows + 1) * 8
            + blob_len
            + postings * POSTING as u64;
        if total != expected || length != total {
            return Err(corrupt(format!(
                "file is {length} bytes, header says {total} (expected {expected})"
            )));
        }
        let mut meta_bytes = vec![0u8; meta_len as usize];
        read_exact_at(&file, &mut meta_bytes, HEADER_LEN as u64)?;
        let stored: StoredMeta = serde_json::from_slice(&meta_bytes).map_err(corrupt)?;
        let directory_at = HEADER_LEN as u64 + meta_len;
        let id_offsets_at = directory_at + terms * DIR_ENTRY as u64;
        let id_blob_at = id_offsets_at + (rows + 1) * 8;
        let postings_at = id_blob_at + blob_len;
        Ok(Self {
            file,
            meta: IndexMeta {
                model: stored.model,
                source_id: stored.source_id,
                table_version,
                table_timestamp_nanos,
            },
            rows: rows as usize,
            terms: terms as usize,
            directory_at,
            id_offsets_at,
            id_blob_at,
            postings_at,
        })
    }

    pub fn meta(&self) -> &IndexMeta {
        &self.meta
    }

    /// Rows with at least one term.
    pub fn len(&self) -> usize {
        self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// `(first posting, count)` of a term.
    fn entry(&self, term: u32) -> Result<(u64, usize)> {
        if term as usize >= self.terms {
            return Ok((0, 0));
        }
        let mut raw = [0u8; DIR_ENTRY];
        read_exact_at(
            &self.file,
            &mut raw,
            self.directory_at + term as u64 * DIR_ENTRY as u64,
        )?;
        let first = u64_at(&raw, 0);
        let count = u32::from_le_bytes(raw[8..12].try_into().expect("4 bytes")) as usize;
        Ok((first, count))
    }

    /// Length of one term's posting list: what a query containing it reads.
    pub fn posting_len(&self, term: u32) -> Result<usize> {
        Ok(self.entry(term)?.1)
    }

    fn id(&self, slot: u32) -> Result<String> {
        let mut raw = [0u8; 16];
        read_exact_at(&self.file, &mut raw, self.id_offsets_at + slot as u64 * 8)?;
        let (from, to) = (u64_at(&raw, 0), u64_at(&raw, 8));
        if to < from {
            return Err(corrupt("id offsets are not ascending"));
        }
        let mut bytes = vec![0u8; (to - from) as usize];
        read_exact_at(&self.file, &mut bytes, self.id_blob_at + from)?;
        String::from_utf8(bytes).map_err(|_| corrupt("a row id is not UTF-8"))
    }

    /// Exact top-`k` by inner product. Rows sharing no term with `query` are
    /// not hits; ties go to the smaller id, as in [`super::TopK`].
    pub fn search(&self, query: &SparseVector, k: usize) -> Result<Vec<(f32, String)>> {
        if k == 0 || query.is_empty() || self.rows == 0 {
            return Ok(Vec::new());
        }
        // Terms ascend, as in `dot_terms`, so every f32 sum is built in the
        // same order and comes out bit-identical.
        let mut scores = vec![0f32; self.rows];
        let mut buffer: Vec<u8> = Vec::new();
        for (&term, &weight) in query.indices().iter().zip(query.values()) {
            let (first, count) = self.entry(term)?;
            if count == 0 {
                continue;
            }
            buffer.resize(count * POSTING, 0);
            read_exact_at(
                &self.file,
                &mut buffer,
                self.postings_at + first * POSTING as u64,
            )?;
            for posting in buffer.chunks_exact(POSTING) {
                let slot = u32::from_le_bytes(posting[..4].try_into().expect("4 bytes")) as usize;
                let row_weight = f32::from_le_bytes(posting[4..].try_into().expect("4 bytes"));
                let score = scores.get_mut(slot).ok_or_else(|| {
                    corrupt(format!("a posting names row {slot} of {}", self.rows))
                })?;
                *score += weight * row_weight;
            }
        }

        // Smallest key first: the worst of the best k is on top.
        let mut best: BinaryHeap<Reverse<(u32, Reverse<u32>)>> =
            BinaryHeap::with_capacity(k.min(self.rows) + 1);
        for (slot, &score) in scores.iter().enumerate() {
            if !(score > 0.0 && score.is_finite()) {
                continue;
            }
            // Positive floats order like their bit patterns.
            let key = (score.to_bits(), Reverse(slot as u32));
            if best.len() == k {
                if key <= best.peek().expect("k entries").0 {
                    continue;
                }
                best.pop();
            }
            best.push(Reverse(key));
        }
        best.into_sorted_vec()
            .into_iter()
            .map(|Reverse((bits, Reverse(slot)))| Ok((f32::from_bits(bits), self.id(slot)?)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::TopK;

    fn vector(pairs: &[(u32, f32)]) -> SparseVector {
        SparseVector::from_pairs(pairs.iter().copied())
    }

    type Rows = Vec<(String, SparseVector)>;

    /// What the store does today: score every row, keep the best `k`.
    fn full_scan(rows: &Rows, query: &SparseVector, k: usize) -> Vec<(f32, String)> {
        let mut top = TopK::new(k);
        for (id, row) in rows {
            top.offer(query.dot(row), id);
        }
        top.into_sorted()
    }

    /// Drive the builder the way the store does: one counting scan, then one
    /// scan per pass. Every scan sees the rows in a different order, because
    /// the table makes no promise that two scans agree.
    fn write(path: &Path, rows: &Rows, memory_postings: usize) -> Result<Option<IndexStats>> {
        let mut builder = IndexBuilder::new(path, "m", 7, 99, memory_postings)?;
        for (id, row) in rows {
            builder.count_row(id, "src", row.indices())?;
        }
        for pass in 0..builder.plan()? {
            let mut order: Vec<usize> = (0..rows.len()).collect();
            order.rotate_left(pass % rows.len().max(1));
            order.reverse();
            for at in order {
                let (id, row) = &rows[at];
                builder.add_row(id, row.indices(), row.values())?;
            }
            builder.end_pass()?;
        }
        builder.finish()
    }

    fn build(dir: &Path, rows: &Rows, memory_postings: usize) -> InvertedFile {
        let path = dir.join("test.inv");
        write(&path, rows, memory_postings).expect("write index");
        InvertedFile::open(&path).expect("open index")
    }

    fn rows(items: &[(&str, &[(u32, f32)])]) -> Rows {
        items
            .iter()
            .map(|(id, pairs)| (id.to_string(), vector(pairs)))
            .collect()
    }

    /// Deterministic LCG, so the property test needs no `rand` dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    fn random_vector(rng: &mut Lcg, vocab: u64, max_terms: u64) -> SparseVector {
        let terms = rng.next() % (max_terms + 1);
        vector(
            &(0..terms)
                .map(|_| {
                    let weight = (rng.next() % 1000 + 1) as f32 / 100.0;
                    ((rng.next() % vocab) as u32, weight)
                })
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn a_query_scores_only_the_rows_that_share_a_term() {
        let dir = tempfile::tempdir().unwrap();
        let index = build(
            dir.path(),
            &rows(&[
                ("a", &[(1, 2.0), (5, 1.0)]),
                ("b", &[(5, 4.0)]),
                ("c", &[(9, 3.0)]),
            ]),
            1000,
        );

        let hits = index.search(&vector(&[(1, 1.0), (5, 0.5)]), 10).unwrap();

        assert_eq!(hits, vec![(2.5, "a".to_string()), (2.0, "b".to_string())]);
        assert_eq!(index.len(), 3);
        assert_eq!(index.meta().model, "m");
        assert_eq!(index.meta().source_id, "src");
        assert_eq!(index.meta().table_version, 7);
        assert_eq!(index.meta().table_timestamp_nanos, 99);
    }

    #[test]
    fn an_empty_query_or_k_zero_finds_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let index = build(dir.path(), &rows(&[("a", &[(1, 1.0)])]), 1000);

        assert!(index
            .search(&SparseVector::default(), 5)
            .unwrap()
            .is_empty());
        assert!(index.search(&vector(&[(1, 1.0)]), 0).unwrap().is_empty());
        assert!(index.search(&vector(&[(77, 1.0)]), 5).unwrap().is_empty());
    }

    #[test]
    fn equal_scores_go_to_the_smaller_id_like_the_full_scan() {
        let dir = tempfile::tempdir().unwrap();
        let index = build(
            dir.path(),
            &rows(&[
                ("d", &[(7, 1.0)]),
                ("b", &[(7, 1.0)]),
                ("c", &[(7, 1.0)]),
                ("a", &[(7, 1.0)]),
            ]),
            1000,
        );

        let hits = index.search(&vector(&[(7, 1.0)]), 3).unwrap();

        let ids: Vec<&str> = hits.iter().map(|(_, id)| id.as_str()).collect();
        assert_eq!(ids, ["a", "b", "c"]);
    }

    #[test]
    fn rows_without_terms_are_not_indexed_and_never_hits() {
        let dir = tempfile::tempdir().unwrap();
        let index = build(
            dir.path(),
            &rows(&[("empty", &[]), ("a", &[(1, 1.0)])]),
            1000,
        );

        assert_eq!(index.len(), 1);
        assert_eq!(index.search(&vector(&[(1, 1.0)]), 5).unwrap().len(), 1);
    }

    #[test]
    fn no_row_with_a_term_writes_no_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("none.inv");

        let stats = write(&path, &rows(&[("empty", &[])]), 1000).unwrap();

        assert!(stats.is_none());
        assert!(!path.exists());
    }

    /// A product that underflows to zero leaves the row's score at zero until a
    /// later term adds to it. The row must still be one hit, not two.
    #[test]
    fn a_row_whose_first_product_underflows_is_one_hit() {
        let dir = tempfile::tempdir().unwrap();
        let data = rows(&[("a", &[(1, 1e-30), (2, 1.0)]), ("b", &[(2, 0.5)])]);
        let index = build(dir.path(), &data, 1000);
        let query = vector(&[(1, 1e-30), (2, 1.0)]);

        let hits = index.search(&query, 10).unwrap();

        assert_eq!(hits, full_scan(&data, &query, 10));
        assert_eq!(hits.iter().filter(|(_, id)| id == "a").count(), 1);
    }

    #[test]
    fn search_equals_the_full_scan_for_many_random_corpora_and_memory_budgets() {
        let mut rng = Lcg(0x2545_F491_4F6C_DD1D);
        let dir = tempfile::tempdir().unwrap();
        for round in 0..24 {
            let vocab = [8u64, 40, 500][round % 3];
            let data: Rows = (0..(20 + rng.next() % 200))
                .map(|i| (format!("row-{i:04}"), random_vector(&mut rng, vocab, 12)))
                .collect();
            // 1 posting forces a pass per term; 7 a few terms per pass; the
            // last holds the whole index at once.
            for budget in [1usize, 7, 1_000_000] {
                let index = build(dir.path(), &data, budget);
                for _ in 0..12 {
                    let query = random_vector(&mut rng, vocab, 6);
                    for k in [1usize, 3, 10, 1000] {
                        assert_eq!(
                            index.search(&query, k).unwrap(),
                            full_scan(&data, &query, k),
                            "round {round} budget {budget} k {k} query {query:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_small_memory_budget_costs_passes_not_correctness() {
        let dir = tempfile::tempdir().unwrap();
        let data = rows(&[
            ("a", &[(1, 1.0), (2, 1.0), (3, 1.0)]),
            ("b", &[(1, 1.0), (2, 1.0)]),
        ]);

        let one_pass = write(&dir.path().join("x.inv"), &data, 1_000)
            .unwrap()
            .unwrap();
        let many = write(&dir.path().join("y.inv"), &data, 2).unwrap().unwrap();

        assert_eq!(one_pass.passes, 1);
        assert!(many.passes > 1);
        assert_eq!(one_pass.bytes, many.bytes);
        assert_eq!(
            std::fs::read(dir.path().join("x.inv")).unwrap(),
            std::fs::read(dir.path().join("y.inv")).unwrap(),
            "the file does not depend on how many passes wrote it"
        );
    }

    #[test]
    fn rows_from_two_sources_or_with_a_repeated_id_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.inv");
        let v = vector(&[(1, 1.0)]);

        let mut builder = IndexBuilder::new(&path, "m", 1, 1, 100).unwrap();
        builder.count_row("a", "one", v.indices()).unwrap();
        assert!(builder.count_row("b", "two", v.indices()).is_err());

        let mut builder = IndexBuilder::new(&path, "m", 1, 1, 100).unwrap();
        builder.count_row("a", "one", v.indices()).unwrap();
        builder.count_row("a", "one", v.indices()).unwrap();
        assert!(builder.plan().is_err());
    }

    #[test]
    fn a_pass_that_disagrees_with_the_counting_scan_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let data = rows(&[("a", &[(1, 1.0)]), ("b", &[(1, 1.0)])]);
        let start = |name: &str| {
            let mut builder = IndexBuilder::new(&dir.path().join(name), "m", 1, 1, 100).unwrap();
            for (id, row) in &data {
                builder.count_row(id, "s", row.indices()).unwrap();
            }
            assert_eq!(builder.plan().unwrap(), 1);
            builder
        };

        let mut missing = start("missing.inv");
        missing
            .add_row("a", data[0].1.indices(), data[0].1.values())
            .unwrap();
        assert!(missing.end_pass().is_err(), "row b was never seen");

        let mut twice = start("twice.inv");
        twice
            .add_row("a", data[0].1.indices(), data[0].1.values())
            .unwrap();
        assert!(
            twice
                .add_row("a", data[0].1.indices(), data[0].1.values())
                .is_err(),
            "row a was seen twice"
        );

        let mut stranger = start("stranger.inv");
        assert!(
            stranger
                .add_row("zzz", data[0].1.indices(), data[0].1.values())
                .is_err(),
            "a row the counting scan never saw"
        );
        assert!(!dir.path().join("missing.inv").exists());
    }

    #[test]
    fn a_row_with_no_terms_may_appear_in_every_scan() {
        let dir = tempfile::tempdir().unwrap();
        let mut builder = IndexBuilder::new(&dir.path().join("x.inv"), "m", 1, 1, 100).unwrap();
        builder.count_row("a", "s", &[1]).unwrap();
        builder.count_row("empty", "s", &[]).unwrap();
        assert_eq!(builder.plan().unwrap(), 1);
        builder.add_row("empty", &[], &[]).unwrap();
        builder.add_row("a", &[1], &[2.0]).unwrap();
        builder.end_pass().unwrap();

        assert_eq!(builder.finish().unwrap().unwrap().rows, 1);
    }

    #[test]
    fn a_damaged_file_is_rejected_when_opened_or_when_searched() {
        let dir = tempfile::tempdir().unwrap();
        let data = rows(&[("a", &[(1, 1.0), (2, 1.0)]), ("b", &[(2, 3.0)])]);
        let path = dir.path().join("x.inv");
        write(&path, &data, 100).unwrap();
        let good = std::fs::read(&path).unwrap();

        // Truncated: a torn copy must not be mistaken for an index.
        std::fs::write(&path, &good[..good.len() - 1]).unwrap();
        assert!(InvertedFile::open(&path).is_err());
        // Not an index at all.
        let mut bad = good.clone();
        bad[0] ^= 0xFF;
        std::fs::write(&path, &bad).unwrap();
        assert!(InvertedFile::open(&path).is_err());
        // A posting that names a row that does not exist.
        let mut bad = good.clone();
        let last = bad.len() - POSTING;
        bad[last..last + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        std::fs::write(&path, &bad).unwrap();
        let index = InvertedFile::open(&path).expect("header is intact");
        assert!(index.search(&vector(&[(1, 1.0), (2, 1.0)]), 5).is_err());
    }
}
