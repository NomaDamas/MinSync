//! A compressed on-disk BM25 inverted index that reuses the block bit-packing
//! and positioned-read architecture of [`crate::sparse::inverted`].
//!
//! Unlike the SPLADE inverted index (which stores 8-bit quantized dot-product
//! bounds and rescores exact vectors from LanceDB), BM25 term frequencies are
//! exact integers (`tf >= 1`) and document lengths are exact `u32` values. By
//! storing exact gap-packed doc slots and bit-packed `(tf - 1)` values in
//! 128-posting blocks alongside a compact doc-length table and sorted term
//! dictionary, a BM25 query answers exact Okapi BM25 scores directly from the
//! sidecar file — zero full-table scans, zero extra LanceDB reads except
//! fetching the winning top-`k` rows' metadata!
//!
//! # File layout (little endian), written once and never modified
//!
//! ```text
//! postings      per term, contiguous; per block of 128 postings:
//!                 gap_width u8, tf_width u8,
//!                 128 slot gaps bit-packed at `gap_width` bits,
//!                 128 `(tf - 1)` values bit-packed at `tf_width` bits
//! meta          JSON: { language, source_id }
//! term offsets  (terms + 1) x u64 into term blob
//! term blob     UTF-8 term strings, strictly sorted lexicographically
//! directory     (terms + 1) x { offset u64 into postings, df u32, _pad u32 }
//! doc lengths   rows x u32 (token count per slot)
//! id offsets    (rows + 1) x u64 into id blob
//! id blob       row ids, in ascending order
//! trailer       (112 B) magic, format, table version + timestamp, counts,
//!               total_doc_len, section lengths, total file length
//! ```
//!
//! Terms are lexicographically sorted so a query binary-searches the term
//! offsets/blob with positioned reads (or caches only the small offsets array)
//! and reads ONLY the posting lists of the query terms. Opening a 100 MB index
//! costs a single trailer + meta read; answering a 5-term query does ~5
//! positioned reads.

use crate::bm25::{bm25_tf_component, idf, term_frequencies, DEFAULT_B, DEFAULT_K1};
use crate::error::{MinSyncError, Result};
use crate::sparse::TopK;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"MSBM25IX";
const FORMAT: u32 = 1;
const FOOTER_LEN: usize = 112;
const DIR_ENTRY: usize = 16;
const BLOCK: usize = 128;

fn corrupt(what: impl std::fmt::Display) -> MinSyncError {
    MinSyncError::VectorStore(format!("bm25 inverted index: {what}"))
}

/// What a BM25 index was built from. A reader compares it with the table and
/// configured language before answering from it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bm25IndexMeta {
    /// BM25 tokenizer language preset (`multilingual`, `ko`, `ja`, `zh`, `ar`, `simple`).
    pub language: String,
    /// The one `source_id` every indexed row carries.
    pub source_id: String,
    /// Version of the LanceDB table the rows were read from.
    pub table_version: u64,
    /// Creation time of that version, in nanoseconds.
    pub table_timestamp_nanos: u128,
}

#[derive(Serialize, Deserialize)]
struct StoredMeta {
    language: String,
    source_id: String,
}

/// What a finished build wrote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bm25IndexStats {
    pub rows: usize,
    pub terms: usize,
    pub postings: u64,
    pub bytes: u64,
}

/// Append the low `width` bits of every value, least significant first, in
/// `ceil(len * width / 8)` bytes. Reused directly from `sparse::inverted`.
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

/// Inverse of [`pack`].
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

fn bit_width(max_val: u32) -> u32 {
    32 - max_val.leading_zeros()
}

/// Builds a compressed BM25 inverted index atomically from tokenized rows.
///
/// The caller feeds every row once via [`Bm25IndexBuilder::add_row`], then
/// calls [`Bm25IndexBuilder::finish`]. Rows with zero tokens are still
/// recorded in the corpus document count (`rows`) with `doc_len = 0` so
/// `N` and `avgdl` match the full-table scan identically.
pub struct Bm25IndexBuilder {
    final_path: PathBuf,
    language: String,
    table_version: u64,
    table_timestamp_nanos: u128,
    source_id: Option<String>,
    /// Per-document `(id, doc_len)` in arrival order.
    docs: Vec<(String, u32)>,
    /// Term -> list of `(arrival_doc_idx, tf)`.
    postings_by_term: HashMap<String, Vec<(u32, u32)>>,
}

impl Bm25IndexBuilder {
    pub fn new(
        path: &Path,
        language: &str,
        table_version: u64,
        table_timestamp_nanos: u128,
    ) -> Result<Self> {
        if path.parent().is_none() {
            return Err(corrupt("the index path has no directory"));
        }
        Ok(Self {
            final_path: path.to_path_buf(),
            language: language.to_string(),
            table_version,
            table_timestamp_nanos,
            source_id: None,
            docs: Vec::new(),
            postings_by_term: HashMap::new(),
        })
    }

    /// Add one tokenized document to the builder.
    pub fn add_row(&mut self, id: &str, source_id: &str, tokens: &[String]) -> Result<()> {
        match &self.source_id {
            None => self.source_id = Some(source_id.to_string()),
            Some(seen) if seen != source_id => {
                return Err(corrupt(format!(
                    "rows from several sources ({seen:?} and {source_id:?}) cannot share one index"
                )));
            }
            Some(_) => {}
        }
        if self.docs.len() >= u32::MAX as usize {
            return Err(corrupt("more than 2^32 rows"));
        }
        let doc_idx = self.docs.len() as u32;
        let (tf_pairs, doc_len) = term_frequencies(tokens);
        self.docs.push((id.to_string(), doc_len));
        for (term, tf) in tf_pairs {
            self.postings_by_term
                .entry(term)
                .or_default()
                .push((doc_idx, tf));
        }
        Ok(())
    }

    /// Write the index atomically to disk. Returns `None` if no rows were added.
    pub fn finish(mut self) -> Result<Option<Bm25IndexStats>> {
        let rows = self.docs.len();
        if rows == 0 {
            return Ok(None);
        }

        // Assign slots in ascending `id` order (matching `sparse::inverted` tie-break order).
        let mut order: Vec<u32> = (0..rows as u32).collect();
        order.sort_unstable_by(|&a, &b| self.docs[a as usize].0.cmp(&self.docs[b as usize].0));
        if let Some(pair) = order
            .windows(2)
            .find(|pair| self.docs[pair[0] as usize].0 == self.docs[pair[1] as usize].0)
        {
            return Err(corrupt(format!(
                "row id {:?} appears twice",
                self.docs[pair[0] as usize].0
            )));
        }
        let mut arrival_to_slot = vec![0u32; rows];
        let mut slot_ids: Vec<String> = Vec::with_capacity(rows);
        let mut doc_lengths: Vec<u32> = Vec::with_capacity(rows);
        let mut total_doc_len: u64 = 0;
        for (slot, &arrival) in order.iter().enumerate() {
            arrival_to_slot[arrival as usize] = slot as u32;
            let id = std::mem::take(&mut self.docs[arrival as usize].0);
            let dl = self.docs[arrival as usize].1;
            total_doc_len = total_doc_len.saturating_add(dl as u64);
            slot_ids.push(id);
            doc_lengths.push(dl);
        }
        drop(self.docs);

        // Sort vocabulary lexicographically for binary search at query time.
        let mut vocab: Vec<(String, Vec<(u32, u32)>)> = self.postings_by_term.into_iter().collect();
        vocab.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let dir = self
            .final_path
            .parent()
            .ok_or_else(|| corrupt("the index path has no directory"))?;
        let temp = tempfile::Builder::new()
            .prefix(".minsync-bm25-index-")
            .tempfile_in(dir)?;
        let mut out = BufWriter::with_capacity(1 << 20, temp);

        // 1. Write compressed postings blocks and build directory.
        let mut written: u64 = 0;
        let mut postings_total: u64 = 0;
        let mut directory: Vec<(u64, u32)> = Vec::with_capacity(vocab.len());
        let mut chunk: Vec<u8> = Vec::with_capacity(96 << 10);
        let mut gaps = [0u32; BLOCK];
        let mut tfs = [0u32; BLOCK];

        for (_, list) in &mut vocab {
            // Remap arrival doc indices to sorted slots, then sort by slot.
            for (doc_idx, _) in list.iter_mut() {
                *doc_idx = arrival_to_slot[*doc_idx as usize];
            }
            list.sort_unstable_by_key(|&(slot, _)| slot);
            let offset = written + chunk.len() as u64;
            let df = list.len() as u32;
            postings_total += df as u64;

            let mut next = 0u64;
            for block in list.chunks(BLOCK) {
                let mut widest_gap = 0u32;
                let mut widest_tf = 0u32;
                for ((gap, tf_out), &(slot, tf)) in gaps.iter_mut().zip(tfs.iter_mut()).zip(block) {
                    *gap = (slot as u64 - next) as u32;
                    next = slot as u64 + 1;
                    widest_gap |= *gap;
                    let tf_minus_1 = tf.saturating_sub(1);
                    *tf_out = tf_minus_1;
                    widest_tf |= tf_minus_1;
                }
                let gap_width = bit_width(widest_gap);
                let tf_width = bit_width(widest_tf);
                chunk.push(gap_width as u8);
                chunk.push(tf_width as u8);
                pack(&gaps[..block.len()], gap_width, &mut chunk);
                pack(&tfs[..block.len()], tf_width, &mut chunk);
            }
            directory.push((offset, df));
            if chunk.len() >= 64 << 10 {
                out.write_all(&chunk)?;
                written += chunk.len() as u64;
                chunk.clear();
            }
        }
        out.write_all(&chunk)?;
        written += chunk.len() as u64;
        let postings_len = written;

        // 2. Meta JSON.
        let meta = serde_json::to_vec(&StoredMeta {
            language: self.language,
            source_id: self.source_id.unwrap_or_default(),
        })
        .map_err(corrupt)?;
        out.write_all(&meta)?;

        // 3. Term offsets + term blob.
        let mut term_blob_len: u64 = 0;
        out.write_all(&term_blob_len.to_le_bytes())?;
        for (term, _) in &vocab {
            term_blob_len += term.len() as u64;
            out.write_all(&term_blob_len.to_le_bytes())?;
        }
        for (term, _) in &vocab {
            out.write_all(term.as_bytes())?;
        }

        // 4. Directory: (terms + 1) entries of 16 bytes each: offset u64, df u32, pad u32.
        for &(offset, df) in &directory {
            out.write_all(&offset.to_le_bytes())?;
            out.write_all(&df.to_le_bytes())?;
            out.write_all(&0u32.to_le_bytes())?;
        }
        out.write_all(&postings_len.to_le_bytes())?;
        out.write_all(&0u32.to_le_bytes())?;
        out.write_all(&0u32.to_le_bytes())?;

        // 5. Doc lengths: rows x u32.
        for &dl in &doc_lengths {
            out.write_all(&dl.to_le_bytes())?;
        }

        // 6. ID offsets + ID blob.
        let mut id_blob_len: u64 = 0;
        out.write_all(&id_blob_len.to_le_bytes())?;
        for id in &slot_ids {
            id_blob_len += id.len() as u64;
            out.write_all(&id_blob_len.to_le_bytes())?;
        }
        for id in &slot_ids {
            out.write_all(id.as_bytes())?;
        }

        // 7. Footer (112 bytes).
        let terms = vocab.len() as u64;
        let total = postings_len
            + meta.len() as u64
            + (terms + 1) * 8
            + term_blob_len
            + (terms + 1) * DIR_ENTRY as u64
            + (rows as u64) * 4
            + (rows as u64 + 1) * 8
            + id_blob_len
            + FOOTER_LEN as u64;

        let mut footer = Vec::with_capacity(FOOTER_LEN);
        footer.extend_from_slice(MAGIC); // 0..8
        footer.extend_from_slice(&FORMAT.to_le_bytes()); // 8..12
        footer.extend_from_slice(&0u32.to_le_bytes()); // 12..16 (reserved)
        footer.extend_from_slice(&self.table_version.to_le_bytes()); // 16..24
        footer.extend_from_slice(&self.table_timestamp_nanos.to_le_bytes()); // 24..40
        footer.extend_from_slice(&(rows as u64).to_le_bytes()); // 40..48
        footer.extend_from_slice(&terms.to_le_bytes()); // 48..56
        footer.extend_from_slice(&postings_total.to_le_bytes()); // 56..64
        footer.extend_from_slice(&total_doc_len.to_le_bytes()); // 64..72
        footer.extend_from_slice(&postings_len.to_le_bytes()); // 72..80
        footer.extend_from_slice(&(meta.len() as u64).to_le_bytes()); // 80..88
        footer.extend_from_slice(&term_blob_len.to_le_bytes()); // 88..96
        footer.extend_from_slice(&id_blob_len.to_le_bytes()); // 96..104
        footer.extend_from_slice(&total.to_le_bytes()); // 104..112
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

        Ok(Some(Bm25IndexStats {
            rows,
            terms: vocab.len(),
            postings: postings_total,
            bytes: total,
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

fn for_each_posting(
    bytes: &[u8],
    count: usize,
    rows: usize,
    mut visit: impl FnMut(u32, u32),
) -> Result<()> {
    let (mut at, mut remaining, mut next) = (0usize, count, 0u64);
    let mut gaps = [0u32; BLOCK];
    let mut tfs = [0u32; BLOCK];
    while remaining > 0 {
        let n = remaining.min(BLOCK);
        let gap_width = *bytes
            .get(at)
            .ok_or_else(|| corrupt("posting list ends before gap width"))?
            as u32;
        let tf_width = *bytes
            .get(at + 1)
            .ok_or_else(|| corrupt("posting list ends before tf width"))?
            as u32;
        if gap_width > 32 || tf_width > 32 {
            return Err(corrupt(format!(
                "invalid block bit widths: gap={gap_width}, tf={tf_width}"
            )));
        }
        at += 2;
        let packed_gaps = (n * gap_width as usize).div_ceil(8);
        let packed_tfs = (n * tf_width as usize).div_ceil(8);
        let end = at + packed_gaps + packed_tfs;
        if end > bytes.len() {
            return Err(corrupt("posting list ends inside a block"));
        }
        unpack(&bytes[at..at + packed_gaps], gap_width, &mut gaps[..n]);
        unpack(&bytes[at + packed_gaps..end], tf_width, &mut tfs[..n]);
        for (&gap, &tf_m1) in gaps[..n].iter().zip(&tfs[..n]) {
            let slot = next + gap as u64;
            if slot >= rows as u64 {
                return Err(corrupt(format!("posting names row {slot} of {rows}")));
            }
            visit(slot as u32, tf_m1.saturating_add(1));
            next = slot + 1;
        }
        at = end;
        remaining -= n;
    }
    if at != bytes.len() {
        return Err(corrupt("posting list is longer than its postings"));
    }
    Ok(())
}

struct PostingList {
    offset: u64,
    bytes: usize,
    df: u32,
}

/// A built BM25 inverted index, opened for exact Okapi BM25 queries.
pub struct Bm25InvertedFile {
    file: File,
    meta: Bm25IndexMeta,
    rows: usize,
    terms: usize,
    total_doc_len: u64,
    postings_len: u64,
    term_offsets_at: u64,
    term_blob_at: u64,
    term_blob_len: u64,
    directory_at: u64,
    doc_lengths_at: u64,
    id_offsets_at: u64,
    id_blob_at: u64,
    id_blob_len: u64,
}

impl Bm25InvertedFile {
    /// Open and validate a BM25 inverted index file.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let length = file.metadata()?.len();
        if length < FOOTER_LEN as u64 {
            return Err(corrupt("file shorter than its trailer"));
        }
        let mut footer = [0u8; FOOTER_LEN];
        read_exact_at(&file, &mut footer, length - FOOTER_LEN as u64)?;
        if &footer[..8] != MAGIC {
            return Err(corrupt("not a bm25 index file"));
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
        let _postings_total = u64_at(&footer, 56);
        let total_doc_len = u64_at(&footer, 64);
        let postings_len = u64_at(&footer, 72);
        let meta_len = u64_at(&footer, 80);
        let term_blob_len = u64_at(&footer, 88);
        let id_blob_len = u64_at(&footer, 96);
        let total = u64_at(&footer, 104);

        let expected = [
            postings_len,
            meta_len,
            terms.saturating_add(1).saturating_mul(8),
            term_blob_len,
            terms.saturating_add(1).saturating_mul(DIR_ENTRY as u64),
            rows.saturating_mul(4),
            rows.saturating_add(1).saturating_mul(8),
            id_blob_len,
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

        let term_offsets_at = postings_len + meta_len;
        let term_blob_at = term_offsets_at + (terms + 1) * 8;
        let directory_at = term_blob_at + term_blob_len;
        let doc_lengths_at = directory_at + (terms + 1) * DIR_ENTRY as u64;
        let id_offsets_at = doc_lengths_at + rows * 4;
        let id_blob_at = id_offsets_at + (rows + 1) * 8;

        Ok(Self {
            file,
            meta: Bm25IndexMeta {
                language: stored.language,
                source_id: stored.source_id,
                table_version,
                table_timestamp_nanos,
            },
            rows: rows as usize,
            terms: terms as usize,
            total_doc_len,
            postings_len,
            term_offsets_at,
            term_blob_at,
            term_blob_len,
            directory_at,
            doc_lengths_at,
            id_offsets_at,
            id_blob_at,
            id_blob_len,
        })
    }

    pub fn meta(&self) -> &Bm25IndexMeta {
        &self.meta
    }

    pub fn len(&self) -> usize {
        self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    pub fn term_count(&self) -> usize {
        self.terms
    }

    fn term_at(&self, idx: usize) -> Result<String> {
        let mut raw = [0u8; 16];
        read_exact_at(&self.file, &mut raw, self.term_offsets_at + idx as u64 * 8)?;
        let (from, to) = (u64_at(&raw, 0), u64_at(&raw, 8));
        if to < from || to > self.term_blob_len {
            return Err(corrupt("term offsets out of bounds"));
        }
        let mut bytes = vec![0u8; (to - from) as usize];
        read_exact_at(&self.file, &mut bytes, self.term_blob_at + from)?;
        String::from_utf8(bytes).map_err(|_| corrupt("term is not valid UTF-8"))
    }

    /// Binary search the sorted on-disk term dictionary for `needle`.
    fn lookup_term(&self, needle: &str) -> Result<Option<usize>> {
        let mut lo = 0usize;
        let mut hi = self.terms;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let mid_term = self.term_at(mid)?;
            match mid_term.as_str().cmp(needle) {
                std::cmp::Ordering::Equal => return Ok(Some(mid)),
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
            }
        }
        Ok(None)
    }

    fn posting_list(&self, term_idx: usize) -> Result<Option<PostingList>> {
        if term_idx >= self.terms {
            return Ok(None);
        }
        let mut raw = [0u8; 2 * DIR_ENTRY];
        read_exact_at(
            &self.file,
            &mut raw,
            self.directory_at + term_idx as u64 * DIR_ENTRY as u64,
        )?;
        let offset = u64_at(&raw, 0);
        let df = u32_at(&raw, 8);
        let end = u64_at(&raw, DIR_ENTRY);
        if offset > end || end > self.postings_len {
            return Err(corrupt("directory points outside postings"));
        }
        if df == 0 {
            return Ok(None);
        }
        Ok(Some(PostingList {
            offset,
            bytes: (end - offset) as usize,
            df,
        }))
    }

    fn doc_length(&self, slot: u32) -> Result<u32> {
        let mut raw = [0u8; 4];
        read_exact_at(&self.file, &mut raw, self.doc_lengths_at + slot as u64 * 4)?;
        Ok(u32::from_le_bytes(raw))
    }

    fn id(&self, slot: u32) -> Result<String> {
        let mut raw = [0u8; 16];
        read_exact_at(&self.file, &mut raw, self.id_offsets_at + slot as u64 * 8)?;
        let (from, to) = (u64_at(&raw, 0), u64_at(&raw, 8));
        if to < from || to > self.id_blob_len {
            return Err(corrupt("id offsets out of bounds"));
        }
        let mut bytes = vec![0u8; (to - from) as usize];
        read_exact_at(&self.file, &mut bytes, self.id_blob_at + from)?;
        String::from_utf8(bytes).map_err(|_| corrupt("row id is not valid UTF-8"))
    }

    /// Answer exact Okapi BM25 top-`k` for `query_tokens`.
    /// Returns `(score, doc_id)` sorted by descending score then ascending id,
    /// matching [`crate::bm25::score_bm25_scan`] identically.
    pub fn search(&self, query_tokens: &[String], k: usize) -> Result<Vec<(f32, String)>> {
        if k == 0 || query_tokens.is_empty() || self.rows == 0 || self.total_doc_len == 0 {
            return Ok(Vec::new());
        }

        let mut query_tf: HashMap<&str, u32> = HashMap::new();
        for token in query_tokens {
            *query_tf.entry(token.as_str()).or_insert(0) += 1;
        }
        // Deterministic term iteration order so floating-point accumulation
        // matches `score_bm25_scan` bit-for-bit.
        let mut query_terms: Vec<(&str, u32)> = query_tf.into_iter().collect();
        query_terms.sort_unstable_by(|a, b| a.0.cmp(b.0));

        let avg_dl = (self.total_doc_len as f64 / self.rows as f64) as f32;
        let mut scores = vec![0.0f32; self.rows];
        let mut touched: Vec<u32> = Vec::new();
        let mut doc_len_cache: HashMap<u32, u32> = HashMap::new();
        // When the table is modest, reading doc_lengths in one sequential read
        // avoids thousands of 4-byte pread syscalls.
        let all_doc_lengths: Option<Vec<u32>> = if self.rows <= 250_000 {
            let mut raw = vec![0u8; self.rows * 4];
            read_exact_at(&self.file, &mut raw, self.doc_lengths_at)?;
            let (chunks, _) = raw.as_chunks::<4>();
            Some(chunks.iter().map(|c| u32::from_le_bytes(*c)).collect())
        } else {
            None
        };

        let mut buffer: Vec<u8> = Vec::new();
        for (term, qtf) in query_terms {
            let Some(term_idx) = self.lookup_term(term)? else {
                continue;
            };
            let Some(list) = self.posting_list(term_idx)? else {
                continue;
            };
            buffer.resize(list.bytes, 0);
            read_exact_at(&self.file, &mut buffer, list.offset)?;
            let term_idf = idf(self.rows as u64, list.df);
            let qtf_f = qtf as f32;
            let mut read_err: Option<MinSyncError> = None;
            for_each_posting(&buffer, list.df as usize, self.rows, |slot, tf| {
                if read_err.is_some() {
                    return;
                }
                let dl = match &all_doc_lengths {
                    Some(dls) => dls[slot as usize],
                    None => match doc_len_cache.get(&slot) {
                        Some(&dl) => dl,
                        None => match self.doc_length(slot) {
                            Ok(dl) => {
                                doc_len_cache.insert(slot, dl);
                                dl
                            }
                            Err(e) => {
                                read_err = Some(e);
                                return;
                            }
                        },
                    },
                };
                let cell = &mut scores[slot as usize];
                if *cell == 0.0 {
                    touched.push(slot);
                }
                *cell +=
                    qtf_f * term_idf * bm25_tf_component(tf, dl, avg_dl, DEFAULT_K1, DEFAULT_B);
            })?;
            if let Some(err) = read_err {
                return Err(err);
            }
        }

        if touched.is_empty() {
            return Ok(Vec::new());
        }

        // Slots are already in ascending id order, so resolving ids only for
        // the top-k slots gives the exact same tie-breaking as `TopK`.
        touched.sort_unstable_by(|&a, &b| {
            scores[b as usize]
                .total_cmp(&scores[a as usize])
                .then_with(|| a.cmp(&b))
        });
        touched.truncate(k);

        let mut top = TopK::new(k);
        for slot in touched {
            let id = self.id(slot)?;
            top.offer(scores[slot as usize], &id);
        }
        Ok(top.into_sorted())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bm25::score_bm25_scan;
    use crate::bm25::tokenizer::tokenize;

    fn build_and_open(
        dir: &Path,
        docs: &[(String, Vec<String>)],
        language: &str,
    ) -> Bm25InvertedFile {
        let path = dir.join("bm25-index.inv");
        let mut builder = Bm25IndexBuilder::new(&path, language, 42, 12345).unwrap();
        for (id, tokens) in docs {
            builder.add_row(id, "src-1", tokens).unwrap();
        }
        let stats = builder.finish().unwrap().expect("non-empty stats");
        assert_eq!(stats.rows, docs.len());
        Bm25InvertedFile::open(&path).unwrap()
    }

    #[test]
    fn inverted_index_matches_full_scan_exact_scores_and_order() {
        let dir = tempfile::tempdir().unwrap();
        let corpus = [
            ("doc-01", "Rust systems programming and BM25 inverted index"),
            ("doc-02", "환불정책과배송안내입니다 빠른 배송 안내"),
            ("doc-03", "関西国際空港限定トートバッグの販売情報"),
            ("doc-04", "我们中出了一个叛徒 全文搜索和索引"),
            ("doc-05", "والكتاب في المدرسة العربية"),
            ("doc-06", "Error ERR_404 in user-service.v2 retry timeout"),
            ("doc-07", "Rust Rust Rust BM25 search engine fast"),
            ("doc-08", ""),
        ];
        let docs: Vec<(String, Vec<String>)> = corpus
            .iter()
            .map(|(id, text)| (id.to_string(), tokenize(text, "multilingual").unwrap()))
            .collect();

        let index = build_and_open(dir.path(), &docs, "multilingual");
        assert_eq!(index.meta().language, "multilingual");
        assert_eq!(index.meta().source_id, "src-1");
        assert_eq!(index.meta().table_version, 42);
        assert_eq!(index.meta().table_timestamp_nanos, 12345);
        assert_eq!(index.len(), docs.len());

        for query_text in [
            "Rust BM25",
            "환불 정책",
            "関西国際空港",
            "叛徒",
            "كتاب",
            "ERR_404",
            "rust rust",
            "nonexistent_term_xyz",
        ] {
            let qtokens = tokenize(query_text, "multilingual").unwrap();
            for k in [1, 3, 10] {
                let via_index = index.search(&qtokens, k).unwrap();
                let via_scan = score_bm25_scan(&qtokens, &docs, k);
                assert_eq!(
                    via_index, via_scan,
                    "mismatch for query {query_text:?} k={k}"
                );
            }
        }
    }

    #[test]
    fn multi_block_posting_lists_roundtrip_exactly() {
        // > 256 docs so postings span multiple 128-entry blocks with varying gap & tf widths.
        let dir = tempfile::tempdir().unwrap();
        let mut docs = Vec::new();
        for i in 0..320 {
            let mut tokens = vec!["common".to_string()];
            if i % 3 == 0 {
                for _ in 0..((i % 7) + 1) {
                    tokens.push("alpha".to_string());
                }
            }
            if i % 11 == 0 {
                tokens.push("rare".to_string());
            }
            tokens.push(format!("unique_{i}"));
            docs.push((format!("id-{i:04}"), tokens));
        }

        let index = build_and_open(dir.path(), &docs, "simple");
        for query in [
            vec!["alpha".into(), "rare".into()],
            vec!["common".into()],
            vec!["unique_275".into()],
        ] {
            let via_index = index.search(&query, 15).unwrap();
            let via_scan = score_bm25_scan(&query, &docs, 15);
            assert_eq!(via_index, via_scan, "query={query:?}");
        }
    }

    #[test]
    fn corrupt_or_truncated_index_file_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let docs = vec![("a".to_string(), vec!["hello".to_string()])];
        let _ = build_and_open(dir.path(), &docs, "simple");
        let path = dir.path().join("bm25-index.inv");
        let bytes = std::fs::read(&path).unwrap();
        let trunc = dir.path().join("trunc.inv");
        std::fs::write(&trunc, &bytes[..bytes.len() - 5]).unwrap();
        assert!(Bm25InvertedFile::open(&trunc).is_err());

        let garbage = dir.path().join("garbage.inv");
        std::fs::write(&garbage, b"not a valid bm25 index").unwrap();
        assert!(Bm25InvertedFile::open(&garbage).is_err());
    }
}
