//! A compressed on-disk inverted index over sparse vectors that answers a
//! query with a small set of *candidate* rows containing the exact top-`k`.
//!
//! Scoring a query against every stored vector costs `N x (q + d)` steps. The
//! inverted index keeps, per term, the rows that hold it and their weights, so
//! a query reads only the posting lists of its own terms (about 2% of all
//! postings for a 150-term SPLADE query).
//!
//! # Exactness
//!
//! The index does not hold the weights, only an 8-bit bracket of each: a list
//! stores `q = floor(w / scale)` with `scale` fixed per term, so the stored
//! weight `w` lies in `[q * scale, (q + 1) * scale]`, to the last bit
//! ([`quantise`] checks both ends exactly). Summing `query weight x
//! (q + 1) x scale` per row gives an upper bound `U(row)` of the row's score;
//! `U(row) - S` with `S` the sum of `query weight x scale` over the query's
//! terms is a lower bound. The k-th largest lower bound is therefore at most
//! the k-th largest score, and any row whose upper bound falls below it
//! cannot be in the top `k`. [`InvertedFile::candidates`] returns every other
//! row, lowering the bar by a margin for the `f32` rounding of the scan's own
//! sum: `(q + 2) * 2^-21` of the k-th bound, plus `(q + 2) * 2^-148` for
//! underflowing products, `q` being the number of query terms. The scan's sum
//! of `m <= q` positive products is off from the exact one by about
//! `2 * m * 2^-24` relative, so the margin is several times that. Rows tied
//! with the k-th score are kept, and the caller's exact rescoring picks among
//! them by id, like [`super::TopK`].
//! The caller scores the candidates from the exact vectors with the scan's
//! own loop: the answer is the scan's, ids, order and `f32` bits.
//!
//! # File layout (little endian), written once and never modified
//!
//! ```text
//! postings   per term, contiguous; per block of 128 postings:
//!              width u8, 128 gaps bit-packed at `width` bits, 128 weight bytes
//! meta       JSON: model and source id
//! directory  (terms + 1) x { offset u64 into postings, count u32, scale f32 }
//! id offsets (rows + 1) x u64 into the id blob
//! id blob    row ids, in ascending order
//! trailer    (96 B) magic, format, table version + timestamp, counts, lengths
//! ```
//!
//! The directory sits after the postings because their sizes are only known
//! once they are written. A row's *slot* is its rank in ascending id order,
//! so "equal scores go to the smaller id" is "equal scores go to the smaller
//! slot". Slots in a list ascend; a block stores `slot - (previous slot + 1)`.
//!
//! A query reads the directory entries and posting lists of its own terms
//! with positioned reads, so opening a 100 MB index costs a trailer read. The
//! writer keeps at most `memory_postings` postings in memory: terms are
//! processed in ranges, one scan of the rows per range, and each range is
//! appended to the file in term order.

use super::SparseVector;
use crate::error::{MinSyncError, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

const MAGIC: &[u8; 8] = b"MSINVIDX";
const FORMAT: u32 = 2;
const FOOTER_LEN: usize = 96;
const DIR_ENTRY: usize = 16;
/// Postings per block: the unit of the bit width, so one large gap costs 128
/// postings a few bits, not the whole list.
const BLOCK: usize = 128;
/// Highest stored weight level (8 bits).
const LEVELS: u32 = 255;

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

/// Append the low `width` bits of every value, least significant first, in
/// `ceil(len * width / 8)` bytes. Every value must fit `width` bits.
fn pack(values: &[u32], width: u32, out: &mut Vec<u8>) {
    if width == 0 {
        return;
    }
    let (mut accumulator, mut bits) = (0u64, 0u32);
    for &value in values {
        debug_assert!(
            width == 32 || value >> width == 0,
            "value exceeds its width"
        );
        accumulator |= (value as u64) << bits;
        bits += width;
        while bits >= 8 {
            out.push(accumulator as u8);
            accumulator >>= 8;
            bits -= 8;
        }
    }
    if bits > 0 {
        out.push(accumulator as u8);
    }
}

/// Inverse of [`pack`]: `out.len()` values from `ceil(len * width / 8)` bytes.
fn unpack(bytes: &[u8], width: u32, out: &mut [u32]) {
    if width == 0 {
        out.fill(0);
        return;
    }
    let mask = (1u64 << width) - 1;
    let (mut accumulator, mut bits, mut at) = (0u64, 0u32, 0usize);
    for value in out.iter_mut() {
        while bits < width {
            accumulator |= (bytes[at] as u64) << bits;
            at += 1;
            bits += 8;
        }
        *value = (accumulator & mask) as u32;
        accumulator >>= width;
        bits -= width;
    }
}

/// The step of a list whose largest weight is `max`: positive, and 255 steps
/// reach `max`. `f32` arithmetic is avoided where it could round the wrong way.
fn scale_for(max: f32) -> f32 {
    let mut scale = (max as f64 / LEVELS as f64) as f32;
    if scale == 0.0 {
        scale = f32::from_bits(1);
    }
    while (scale as f64) * (LEVELS as f64) < max as f64 {
        scale = scale.next_up();
    }
    scale
}

/// The level `q` with `q * scale <= weight <= (q + 1) * scale`, for a weight
/// of at most `255 * scale`. The products are exact in `f64` (a 24-bit
/// mantissa times a 9-bit integer), so the checks below are exact too; only
/// the first guess is rounded.
fn quantise(weight: f32, scale: f32) -> u8 {
    let (weight, scale) = (weight as f64, scale as f64);
    let mut level = (weight / scale).floor().clamp(0.0, LEVELS as f64) as u32;
    while level > 0 && level as f64 * scale > weight {
        level -= 1;
    }
    while level < LEVELS && ((level + 1) as f64) * scale < weight {
        level += 1;
    }
    level as u8
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
    /// Bytes of postings written so far.
    written: u64,
    /// `(offset, count, scale)` of every term finished so far, in term order.
    directory: Vec<(u64, u32, f32)>,
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
            written: 0,
            directory: Vec::new(),
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

    /// Fix the slots and the passes. Returns the number of passes that follow
    /// (zero when no row has a term).
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
            let id = std::mem::take(&mut self.ids[ordinal as usize]);
            self.slot_by_id.insert(id, slot as u32);
        }
        self.ids = Vec::new();
        self.seen = vec![false; rows];
        self.postings_total = self.counts.iter().map(|&c| c as u64).sum();

        let dir = self
            .final_path
            .parent()
            .ok_or_else(|| corrupt("the index path has no directory"))?;
        let temp = NamedTempFile::new_in(dir)?;
        self.out = Some(BufWriter::with_capacity(1 << 20, temp));
        self.directory = Vec::with_capacity(self.counts.len() + 1);

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
        if values
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(corrupt(
                "a row has a weight that is not finite and positive",
            ));
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

    /// Compress the pass's posting lists and append them to the file.
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
        let mut chunk: Vec<u8> = Vec::with_capacity(96 << 10);
        let mut gaps = [0u32; BLOCK];
        for window in self.start.windows(2) {
            // Rows arrive in whatever order the table scan returns them; slots
            // are fixed by id, so sorting each list makes the file the same one
            // however the scan was ordered (and keeps lists ascending by row).
            let list = &mut self.flat[window[0]..window[1]];
            list.sort_unstable_by_key(|&(slot, _)| slot);
            if list.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                return Err(corrupt("a row lists one term twice"));
            }
            let offset = self.written + chunk.len() as u64;
            if list.is_empty() {
                self.directory.push((offset, 0, 0.0));
                continue;
            }
            let max = list.iter().map(|&(_, weight)| weight).fold(0.0, f32::max);
            let scale = scale_for(max);
            let mut next = 0u64;
            for block in list.chunks(BLOCK) {
                let mut widest = 0u32;
                for (gap, &(slot, _)) in gaps.iter_mut().zip(block) {
                    *gap = (slot as u64 - next) as u32;
                    next = slot as u64 + 1;
                    widest |= *gap;
                }
                let width = 32 - widest.leading_zeros();
                chunk.push(width as u8);
                pack(&gaps[..block.len()], width, &mut chunk);
                chunk.extend(block.iter().map(|&(_, weight)| quantise(weight, scale)));
            }
            self.directory.push((offset, list.len() as u32, scale));
            if chunk.len() >= 64 << 10 {
                out.write_all(&chunk)?;
                self.written += chunk.len() as u64;
                chunk.clear();
            }
        }
        out.write_all(&chunk)?;
        self.written += chunk.len() as u64;
        self.pass += 1;
        if self.pass < self.ranges.len() {
            self.begin_pass();
        } else {
            self.flat = Vec::new();
        }
        Ok(())
    }

    /// Write the directory, the ids and the trailer, then move the finished
    /// file into place. `None` when no row had a term: there is nothing to
    /// index and no file is written.
    pub fn finish(mut self) -> Result<Option<IndexStats>> {
        let Some(mut out) = self.out.take() else {
            return Ok(None);
        };
        if self.pass != self.ranges.len() || self.directory.len() != self.counts.len() {
            return Err(corrupt("the index was not written completely"));
        }
        let postings_len = self.written;
        let meta = serde_json::to_vec(&StoredMeta {
            model: self.model.clone(),
            source_id: self.source_id.clone().unwrap_or_default(),
        })
        .map_err(corrupt)?;
        out.write_all(&meta)?;
        for &(offset, count, scale) in &self.directory {
            out.write_all(&offset.to_le_bytes())?;
            out.write_all(&count.to_le_bytes())?;
            out.write_all(&scale.to_le_bytes())?;
        }
        out.write_all(&postings_len.to_le_bytes())?;
        out.write_all(&0u32.to_le_bytes())?;
        out.write_all(&0f32.to_le_bytes())?;

        let mut by_slot: Vec<&str> = vec![""; self.rows];
        for (id, &slot) in &self.slot_by_id {
            by_slot[slot as usize] = id;
        }
        let mut blob_len = 0u64;
        out.write_all(&blob_len.to_le_bytes())?;
        for id in &by_slot {
            blob_len += id.len() as u64;
            out.write_all(&blob_len.to_le_bytes())?;
        }
        for id in &by_slot {
            out.write_all(id.as_bytes())?;
        }

        let terms = self.counts.len() as u64;
        let total = postings_len
            + meta.len() as u64
            + (terms + 1) * DIR_ENTRY as u64
            + (self.rows as u64 + 1) * 8
            + blob_len
            + FOOTER_LEN as u64;
        let mut footer = Vec::with_capacity(FOOTER_LEN);
        footer.extend_from_slice(MAGIC);
        footer.extend_from_slice(&FORMAT.to_le_bytes());
        footer.extend_from_slice(&0u32.to_le_bytes());
        footer.extend_from_slice(&self.table_version.to_le_bytes());
        footer.extend_from_slice(&self.table_timestamp_nanos.to_le_bytes());
        footer.extend_from_slice(&(self.rows as u64).to_le_bytes());
        footer.extend_from_slice(&terms.to_le_bytes());
        footer.extend_from_slice(&self.postings_total.to_le_bytes());
        footer.extend_from_slice(&postings_len.to_le_bytes());
        footer.extend_from_slice(&(meta.len() as u64).to_le_bytes());
        footer.extend_from_slice(&blob_len.to_le_bytes());
        footer.extend_from_slice(&total.to_le_bytes());
        debug_assert_eq!(footer.len(), FOOTER_LEN);
        out.write_all(&footer)?;

        let temp = out
            .into_inner()
            .map_err(|error| MinSyncError::Io(error.into_error()))?;
        if temp.as_file().metadata()?.len() != total {
            return Err(corrupt("the index was not written completely"));
        }
        temp.as_file().sync_all()?;
        temp.persist(&self.final_path)
            .map_err(|error| MinSyncError::Io(error.error))?;
        Ok(Some(IndexStats {
            rows: self.rows,
            postings: self.postings_total,
            bytes: total,
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

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes"))
}

/// Decode one posting list, calling `visit(slot, level)` in ascending slot
/// order. The list must be exactly `count` postings long and end where
/// `bytes` does; anything else, or a slot outside `rows`, is an error.
fn for_each_posting(
    bytes: &[u8],
    count: usize,
    rows: usize,
    mut visit: impl FnMut(u32, u8),
) -> Result<()> {
    let (mut at, mut remaining, mut next) = (0usize, count, 0u64);
    let mut gaps = [0u32; BLOCK];
    while remaining > 0 {
        let n = remaining.min(BLOCK);
        let width = *bytes
            .get(at)
            .ok_or_else(|| corrupt("a posting list ends before its last block"))?
            as u32;
        if width > 32 {
            return Err(corrupt(format!("a block has a gap width of {width} bits")));
        }
        at += 1;
        let packed = (n * width as usize).div_ceil(8);
        let end = at + packed + n;
        if end > bytes.len() {
            return Err(corrupt("a posting list ends inside a block"));
        }
        unpack(&bytes[at..at + packed], width, &mut gaps[..n]);
        for (&gap, &level) in gaps[..n].iter().zip(&bytes[at + packed..end]) {
            let slot = next + gap as u64;
            if slot >= rows as u64 {
                return Err(corrupt(format!("a posting names row {slot} of {rows}")));
            }
            visit(slot as u32, level);
            next = slot + 1;
        }
        at = end;
        remaining -= n;
    }
    if at != bytes.len() {
        return Err(corrupt("a posting list is longer than its postings"));
    }
    Ok(())
}

/// Where one term's posting list sits and how to read its weights.
struct List {
    offset: u64,
    bytes: usize,
    count: usize,
    scale: f32,
}

/// A built index, opened for queries.
pub struct InvertedFile {
    file: File,
    meta: IndexMeta,
    rows: usize,
    terms: usize,
    postings_len: u64,
    directory_at: u64,
    id_offsets_at: u64,
    id_blob_at: u64,
}

impl InvertedFile {
    /// Open and validate an index: the trailer's magic and format, and that
    /// the file has exactly the length the trailer promises (a torn or
    /// truncated file does not).
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let length = file.metadata()?.len();
        if length < FOOTER_LEN as u64 {
            return Err(corrupt("file shorter than its trailer"));
        }
        let mut footer = [0u8; FOOTER_LEN];
        read_exact_at(&file, &mut footer, length - FOOTER_LEN as u64)?;
        if &footer[..8] != MAGIC {
            return Err(corrupt("not an index file"));
        }
        let format = u32_at(&footer, 8);
        if format != FORMAT {
            return Err(corrupt(format!("unsupported format {format}")));
        }
        let table_version = u64_at(&footer, 16);
        let table_timestamp_nanos =
            u128::from_le_bytes(footer[24..40].try_into().expect("16 bytes"));
        let rows = u64_at(&footer, 40);
        let terms = u64_at(&footer, 48);
        let postings_len = u64_at(&footer, 64);
        let meta_len = u64_at(&footer, 72);
        let blob_len = u64_at(&footer, 80);
        let total = u64_at(&footer, 88);
        let expected = [
            postings_len,
            meta_len,
            terms.saturating_add(1).saturating_mul(DIR_ENTRY as u64),
            rows.saturating_add(1).saturating_mul(8),
            blob_len,
            FOOTER_LEN as u64,
        ]
        .iter()
        .fold(0u64, |sum, &part| sum.saturating_add(part));
        if total != expected || length != total {
            return Err(corrupt(format!(
                "file is {length} bytes, trailer says {total} (expected {expected})"
            )));
        }
        let mut meta_bytes = vec![0u8; meta_len as usize];
        read_exact_at(&file, &mut meta_bytes, postings_len)?;
        let stored: StoredMeta = serde_json::from_slice(&meta_bytes).map_err(corrupt)?;
        let directory_at = postings_len + meta_len;
        let id_offsets_at = directory_at + (terms + 1) * DIR_ENTRY as u64;
        let id_blob_at = id_offsets_at + (rows + 1) * 8;
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
            postings_len,
            directory_at,
            id_offsets_at,
            id_blob_at,
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

    /// The list of a term, or `None` when no row holds it.
    fn list(&self, term: u32) -> Result<Option<List>> {
        if term as usize >= self.terms {
            return Ok(None);
        }
        let mut raw = [0u8; 2 * DIR_ENTRY];
        read_exact_at(
            &self.file,
            &mut raw,
            self.directory_at + term as u64 * DIR_ENTRY as u64,
        )?;
        let offset = u64_at(&raw, 0);
        let count = u32_at(&raw, 8);
        let scale = f32::from_bits(u32_at(&raw, 12));
        let end = u64_at(&raw, DIR_ENTRY);
        if offset > end || end > self.postings_len {
            return Err(corrupt("the directory points outside the postings"));
        }
        if count == 0 {
            return if offset == end {
                Ok(None)
            } else {
                Err(corrupt("an empty list has bytes"))
            };
        }
        if !(scale.is_finite() && scale > 0.0) {
            return Err(corrupt("a list has no valid weight scale"));
        }
        Ok(Some(List {
            offset,
            bytes: (end - offset) as usize,
            count: count as usize,
            scale,
        }))
    }

    /// Length of one term's posting list: what a query containing it reads.
    pub fn posting_len(&self, term: u32) -> Result<usize> {
        Ok(self.list(term)?.map_or(0, |list| list.count))
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

    /// The ids, in ascending order, of a superset of the exact top-`k` rows
    /// of `query`: every row of the top `k` by inner product, and every row
    /// tied with the k-th. Rows sharing no term with `query` are not hits and
    /// never returned. Score them from their exact vectors with the scan's
    /// loop to get the scan's answer (see the module docs for why).
    pub fn candidates(&self, query: &SparseVector, k: usize) -> Result<Vec<String>> {
        if k == 0 || query.is_empty() || self.rows == 0 {
            return Ok(Vec::new());
        }
        // `upper[row]`: sum of query weight x (level + 1) x scale, so at least
        // the row's score. Positive exactly for the rows that share a term.
        let mut upper = vec![0f64; self.rows];
        let mut touched: Vec<u32> = Vec::new();
        // At most how far below its upper bound a row's score can lie.
        let mut slack = 0f64;
        let mut buffer: Vec<u8> = Vec::new();
        for (&term, &weight) in query.indices().iter().zip(query.values()) {
            let Some(list) = self.list(term)? else {
                continue;
            };
            buffer.resize(list.bytes, 0);
            read_exact_at(&self.file, &mut buffer, list.offset)?;
            let step = weight as f64 * list.scale as f64;
            slack += step;
            for_each_posting(&buffer, list.count, self.rows, |slot, level| {
                let cell = &mut upper[slot as usize];
                if *cell == 0.0 {
                    touched.push(slot);
                }
                *cell += step * (level as f64 + 1.0);
            })?;
        }

        let mut keep = if touched.len() <= k {
            touched
        } else {
            let mut bounds: Vec<f64> = touched.iter().map(|&slot| upper[slot as usize]).collect();
            let ceiling = bounds.iter().copied().fold(0.0, f64::max);
            let (_, kth, _) = bounds.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
            let kth = *kth;
            if ceiling > f32::MAX as f64 / 2.0 {
                // A score this large can overflow `f32` in the scan, which
                // then drops the row from the ranking entirely; the rows that
                // took its place in the k-th bound would be mis-pruned. No
                // safe bar exists, so keep every row.
                touched
            } else {
                let terms = query.len() as f64 + 2.0;
                let bar = kth - slack - terms * 2f64.powi(-21) * kth - terms * 2f64.powi(-148);
                touched
                    .into_iter()
                    .filter(|&slot| upper[slot as usize] >= bar)
                    .collect()
            }
        };
        keep.sort_unstable();
        keep.into_iter().map(|slot| self.id(slot)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::TopK;
    use std::collections::{HashMap, HashSet};

    fn vector(pairs: &[(u32, f32)]) -> SparseVector {
        SparseVector::from_pairs(pairs.iter().copied())
    }

    type Rows = Vec<(String, SparseVector)>;

    /// What the store does today without an index: score every row, keep the
    /// best `k`.
    fn full_scan(rows: &Rows, query: &SparseVector, k: usize) -> Vec<(f32, String)> {
        let mut top = TopK::new(k);
        for (id, row) in rows {
            top.offer(query.dot(row), id);
        }
        top.into_sorted()
    }

    /// What the store does with the index: score only the candidates, with
    /// the same loop and the exact vectors.
    fn rescored(
        index: &InvertedFile,
        rows: &Rows,
        query: &SparseVector,
        k: usize,
    ) -> Vec<(f32, String)> {
        let by_id: HashMap<&str, &SparseVector> =
            rows.iter().map(|(id, row)| (id.as_str(), row)).collect();
        let mut top = TopK::new(k);
        for id in index.candidates(query, k).unwrap() {
            top.offer(query.dot(by_id[id.as_str()]), &id);
        }
        top.into_sorted()
    }

    /// The whole guarantee: no row of the exact top-`k` is pruned, so scoring
    /// the candidates gives the scan's answer, ids, order and f32 bits.
    fn assert_exact(index: &InvertedFile, rows: &Rows, query: &SparseVector, k: usize, ctx: &str) {
        let want = full_scan(rows, query, k);
        let candidates: HashSet<String> = index.candidates(query, k).unwrap().into_iter().collect();
        for (_, id) in &want {
            assert!(
                candidates.contains(id),
                "{ctx}: top-{k} row {id} was pruned"
            );
        }
        assert_eq!(rescored(index, rows, query, k), want, "{ctx}");
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

    /// Deterministic LCG, so the property tests need no `rand` dependency.
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
    fn a_query_finds_the_rows_that_share_a_term() {
        let dir = tempfile::tempdir().unwrap();
        let data = rows(&[
            ("a", &[(1, 2.0), (5, 1.0)]),
            ("b", &[(5, 4.0)]),
            ("c", &[(9, 3.0)]),
        ]);
        let index = build(dir.path(), &data, 1000);
        let query = vector(&[(1, 1.0), (5, 0.5)]);

        let candidates: HashSet<String> =
            index.candidates(&query, 10).unwrap().into_iter().collect();

        // "c" shares no term with the query: it is not even a candidate.
        let expected: HashSet<String> = ["a", "b"].iter().map(|id| id.to_string()).collect();
        assert_eq!(candidates, expected);
        assert_eq!(
            rescored(&index, &data, &query, 10),
            vec![(2.5, "a".to_string()), (2.0, "b".to_string())]
        );
        assert_eq!(index.len(), 3);
        assert_eq!(index.meta().model, "m");
        assert_eq!(index.meta().source_id, "src");
        assert_eq!(index.meta().table_version, 7);
        assert_eq!(index.meta().table_timestamp_nanos, 99);
        assert_eq!(index.posting_len(5).unwrap(), 2);
    }

    #[test]
    fn an_empty_query_or_k_zero_finds_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let index = build(dir.path(), &rows(&[("a", &[(1, 1.0)])]), 1000);

        assert!(index
            .candidates(&SparseVector::default(), 5)
            .unwrap()
            .is_empty());
        assert!(index
            .candidates(&vector(&[(1, 1.0)]), 0)
            .unwrap()
            .is_empty());
        assert!(index
            .candidates(&vector(&[(77, 1.0)]), 5)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn equal_scores_go_to_the_smaller_id_like_the_full_scan() {
        let dir = tempfile::tempdir().unwrap();
        let data = rows(&[
            ("d", &[(7, 1.0)]),
            ("b", &[(7, 1.0)]),
            ("c", &[(7, 1.0)]),
            ("a", &[(7, 1.0)]),
        ]);
        let index = build(dir.path(), &data, 1000);

        let hits = rescored(&index, &data, &vector(&[(7, 1.0)]), 3);

        let ids: Vec<&str> = hits.iter().map(|(_, id)| id.as_str()).collect();
        assert_eq!(ids, ["a", "b", "c"]);
    }

    /// The boundary of every `k` below falls inside a group of 60 rows with
    /// the same score: all of them must stay candidates, because which of
    /// them is in the answer is decided by id, not by score.
    #[test]
    fn rows_tied_with_the_kth_score_are_all_candidates() {
        let dir = tempfile::tempdir().unwrap();
        let mut data: Rows = (0..60)
            .map(|i| (format!("tie-{i:02}"), vector(&[(1, 2.0), (2, 1.0)])))
            .collect();
        data.extend((0..40).map(|i| (format!("low-{i:02}"), vector(&[(1, 0.5), (3, 1.0)]))));
        let index = build(dir.path(), &data, 1_000_000);
        let query = vector(&[(1, 1.0), (2, 1.0)]);

        for k in [1usize, 10, 59, 60, 61, 100] {
            assert_exact(&index, &data, &query, k, &format!("k {k}"));
        }
    }

    #[test]
    fn rows_without_terms_are_not_indexed_and_never_hits() {
        let dir = tempfile::tempdir().unwrap();
        let data = rows(&[("empty", &[]), ("a", &[(1, 1.0)])]);
        let index = build(dir.path(), &data, 1000);

        assert_eq!(index.len(), 1);
        assert_eq!(index.candidates(&vector(&[(1, 1.0)]), 5).unwrap().len(), 1);
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

        let hits = rescored(&index, &data, &query, 10);

        assert_eq!(hits, full_scan(&data, &query, 10));
        assert_eq!(hits.iter().filter(|(_, id)| id == "a").count(), 1);
    }

    #[test]
    fn candidates_contain_the_exact_top_k_for_many_random_corpora_and_memory_budgets() {
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
                        assert_exact(
                            &index,
                            &data,
                            &query,
                            k,
                            &format!("round {round} budget {budget} k {k} query {query:?}"),
                        );
                    }
                }
            }
        }
    }

    /// The small corpora above are mostly returned whole (k is as large as
    /// they are). Here the bounds actually prune, and must not prune a row of
    /// the answer.
    #[test]
    fn pruning_never_loses_a_top_k_row_on_larger_corpora() {
        let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
        let dir = tempfile::tempdir().unwrap();
        for round in 0..6 {
            let data: Rows = (0..(400 + rng.next() % 1200))
                .map(|i| (format!("row-{i:05}"), random_vector(&mut rng, 30, 40)))
                .collect();
            let index = build(dir.path(), &data, 1_000_000);
            for _ in 0..25 {
                let query = random_vector(&mut rng, 30, 25);
                for k in [1usize, 5, 20, 100] {
                    assert_exact(
                        &index,
                        &data,
                        &query,
                        k,
                        &format!("round {round} k {k} query {query:?}"),
                    );
                }
            }
        }
    }

    /// An index that returned every row would pass the tests above; the point
    /// is that the table is read for a handful of rows, not all of them.
    #[test]
    fn a_selective_query_leaves_only_a_few_candidates() {
        let mut rng = Lcg(7);
        let dir = tempfile::tempdir().unwrap();
        let data: Rows = (0..3000)
            .map(|i| (format!("row-{i:05}"), random_vector(&mut rng, 40, 30)))
            .collect();
        let index = build(dir.path(), &data, 1_000_000);
        let query = vector(&[(1, 3.0), (2, 2.0), (3, 1.0), (4, 2.5), (5, 1.5), (6, 3.5)]);

        let candidates = index.candidates(&query, 10).unwrap();

        assert!(candidates.len() >= 10);
        assert!(
            candidates.len() <= 150,
            "{} candidates of {} rows for k = 10",
            candidates.len(),
            data.len()
        );
        assert_exact(&index, &data, &query, 10, "selective query");
    }

    #[test]
    fn the_file_is_far_smaller_than_eight_bytes_per_posting() {
        let mut rng = Lcg(11);
        let dir = tempfile::tempdir().unwrap();
        let data: Rows = (0..2000)
            .map(|i| (format!("row-{i:05}"), random_vector(&mut rng, 64, 24)))
            .collect();

        let stats = write(&dir.path().join("x.inv"), &data, 1_000_000)
            .unwrap()
            .unwrap();

        // Everything counts: ids, directory and trailer included.
        assert!(
            stats.bytes * 2 < stats.postings * 8,
            "{} bytes for {} postings",
            stats.bytes,
            stats.postings
        );
        assert_eq!(
            stats.bytes,
            std::fs::metadata(dir.path().join("x.inv")).unwrap().len()
        );
    }

    /// Gaps of tens of thousands need 16-bit fields; a list that mixes small
    /// and large gaps exercises the widths a small corpus never reaches.
    #[test]
    fn a_list_with_big_gaps_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let rare = [0usize, 1, 40_000, 69_999];
        let data: Rows = (0..70_000usize)
            .map(|i| {
                let common = (6u32, 1.0 + (i % 1000) as f32 / 1000.0);
                let row = if rare.contains(&i) {
                    vector(&[(5, 1.0 + i as f32 / 70_000.0), common])
                } else {
                    vector(&[common])
                };
                (format!("row-{i:06}"), row)
            })
            .collect();
        let index = build(dir.path(), &data, 1_000_000);
        let query = vector(&[(5, 1.0)]);

        let mut candidates = index.candidates(&query, 10).unwrap();
        candidates.sort();

        assert_eq!(
            candidates,
            ["row-000000", "row-000001", "row-040000", "row-069999"]
        );
        assert_exact(&index, &data, &query, 10, "rare term");
        assert_exact(&index, &data, &query, 2, "rare term, k 2");
    }

    /// Found by searching for it: two rows whose exact `f32` scores are equal
    /// (1.25), where the one that wins by id has the smaller *quantised* upper
    /// bound, below the k-th bound by 5e-9. Without the margin that covers the
    /// scan's own rounding the winner is pruned and the tie goes to the wrong
    /// row. The weights are exact bit patterns, not decimal approximations.
    #[test]
    fn a_row_tied_with_the_kth_score_survives_f32_rounding_of_the_scan() {
        let w = f32::from_bits;
        let dir = tempfile::tempdir().unwrap();
        let data = rows(&[
            ("a-R", &[(1, w(0x3d12_2bc4)), (2, w(0x3f83_c3c2))]),
            ("b-anchor0", &[(1, w(0x3f33_3333))]),
            ("c-anchor1", &[(2, w(0x4020_0000))]),
            ("d-Z", &[(1, w(0x3e52_d2d4)), (2, w(0x4005_a5a6))]),
        ]);
        let index = build(dir.path(), &data, 1_000_000);
        let query = vector(&[(1, w(0x3f80_0000)), (2, w(0x3f00_0000))]);

        let want = full_scan(&data, &query, 1);
        assert_eq!(want[0].1, "c-anchor1", "the scan breaks the tie by id");
        assert_eq!(want[0].0, 1.25);
        assert!(index
            .candidates(&query, 1)
            .unwrap()
            .contains(&"c-anchor1".to_string()));
        assert_exact(&index, &data, &query, 1, "f32 tie");
    }

    /// A score the scan cannot represent (it overflows `f32` to infinity and
    /// is dropped) must neither be an answer nor push the bar above the rows
    /// that are.
    #[test]
    fn rows_whose_score_overflows_do_not_hide_the_real_answers() {
        let dir = tempfile::tempdir().unwrap();
        let mut data: Rows = vec![
            ("huge".to_string(), vector(&[(1, 3.0e38)])),
            ("also-huge".to_string(), vector(&[(1, 2.9e38)])),
        ];
        data.extend((0..50).map(|i| (format!("row-{i:02}"), vector(&[(1, 1.0 + i as f32)]))));
        let index = build(dir.path(), &data, 1_000_000);
        let query = vector(&[(1, 1.0e3)]);

        for k in [1usize, 2, 5, 60] {
            assert_exact(&index, &data, &query, k, &format!("overflow, k {k}"));
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

    /// A row that lists one term twice would put two postings of one row in a
    /// list, which the gap coding cannot express.
    #[test]
    fn a_term_listed_twice_in_a_row_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let mut builder = IndexBuilder::new(&dir.path().join("x.inv"), "m", 1, 1, 100).unwrap();
        builder.count_row("a", "s", &[1, 1]).unwrap();
        assert_eq!(builder.plan().unwrap(), 1);
        builder.add_row("a", &[1, 1], &[1.0, 2.0]).unwrap();

        assert!(builder.end_pass().is_err());
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
        let query = vector(&[(1, 1.0), (2, 1.0)]);

        // Truncated: a torn copy must not be mistaken for an index.
        std::fs::write(&path, &good[..good.len() - 1]).unwrap();
        assert!(InvertedFile::open(&path).is_err());
        // Shorter than its trailer.
        std::fs::write(&path, &good[..FOOTER_LEN - 1]).unwrap();
        assert!(InvertedFile::open(&path).is_err());
        // Not an index at all (the trailer's magic is damaged).
        let mut bad = good.clone();
        let footer = bad.len() - FOOTER_LEN;
        bad[footer] ^= 0xFF;
        std::fs::write(&path, &bad).unwrap();
        assert!(InvertedFile::open(&path).is_err());
        // A posting list whose first block claims an impossible width.
        let mut bad = good.clone();
        bad[0] = 0xFF;
        std::fs::write(&path, &bad).unwrap();
        let index = InvertedFile::open(&path).expect("the trailer is intact");
        assert!(index.candidates(&query, 5).is_err());
        // A posting list that ends where the directory does not expect it:
        // the width byte of the first block says 3 bits, not 0.
        let mut bad = good.clone();
        bad[0] = 3;
        std::fs::write(&path, &bad).unwrap();
        let index = InvertedFile::open(&path).expect("the trailer is intact");
        assert!(index.candidates(&query, 5).is_err());
    }

    #[test]
    fn bit_packing_round_trips_at_every_width() {
        let mut rng = Lcg(3);
        for width in 0..=32u32 {
            let mask = if width == 32 {
                u32::MAX
            } else {
                (1u32 << width) - 1
            };
            for len in [1usize, 2, 7, 127, 128] {
                let mut values: Vec<u32> = (0..len)
                    .map(|_| ((rng.next() << 8) ^ rng.next()) as u32 & mask)
                    .collect();
                // The extremes are where a bit reader breaks.
                values[0] = mask;
                if len > 1 {
                    values[1] = 0;
                }
                let mut packed = Vec::new();
                pack(&values, width, &mut packed);
                assert_eq!(packed.len(), (len * width as usize).div_ceil(8));

                let mut back = vec![0u32; len];
                unpack(&packed, width, &mut back);

                assert_eq!(back, values, "width {width} len {len}");
            }
        }
    }

    /// The pruning is exact only if every stored weight sits inside the
    /// interval its quantised value stands for, to the last bit.
    #[test]
    fn a_quantised_weight_is_bracketed_by_its_step() {
        let mut rng = Lcg(5);
        let mut weights: Vec<f32> = vec![
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            1e-30,
            1.0,
            3.0,
            f32::MAX,
        ];
        while weights.len() < 20_000 {
            let bits = ((rng.next() << 16) ^ rng.next()) as u32 & 0x7FFF_FFFF;
            let weight = f32::from_bits(bits);
            if weight.is_finite() && weight > 0.0 {
                weights.push(weight);
            }
        }
        for _ in 0..200 {
            // A list: its scale comes from its own maximum.
            let list: Vec<f32> = (0..50)
                .map(|_| weights[(rng.next() as usize) % weights.len()])
                .collect();
            let max = list.iter().copied().fold(0.0f32, f32::max);
            let scale = scale_for(max);
            assert!(scale > 0.0 && scale.is_finite());
            assert!(
                scale as f64 * 255.0 >= max as f64,
                "the maximum is in range"
            );
            for &weight in &list {
                let q = quantise(weight, scale) as f64;
                let low = q * scale as f64;
                let high = (q + 1.0) * scale as f64;
                assert!(low <= weight as f64, "{weight:e} below its step");
                assert!(weight as f64 <= high, "{weight:e} above its step");
            }
        }
    }
}
