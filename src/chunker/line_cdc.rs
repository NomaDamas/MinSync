//! Line-aligned, token-budgeted content-defined chunking (`line-cdc`).
//!
//! Cut candidates are line ends only. Whether a line ends a chunk is decided by
//! a hash of that line's own content, scaled by its share of the target size,
//! so an edit changes only the chunks that hold edited lines: downstream
//! boundaries come back at the first unchanged cut. Sizes are counted with
//! [`tokens_upper`] against what both the dense embedder and the sparse
//! encoder take whole, so no chunk is truncated by either.

use crate::chunker::heading::{HeadingContext, Measure};
use crate::chunker::Chunker;
use crate::config::{Config, DEFAULT_SPARSE_MAX_LENGTH};
use crate::embedder::context_limit::budget;
use crate::error::Result;
use crate::types::Chunk;

/// A run of at least this many non-whitespace ASCII chars (ids, URLs, hex,
/// base64) counts a whole token per char.
const LONG_RUN: usize = 16;

/// Upper bound on the tokens of text for the tokenizers MinSync encodes with
/// (Qwen3, KURE-Sparse) and other byte-level BPE models. In half tokens per
/// char: ASCII letters and whitespace 1; other ASCII 2 (Qwen3 splits every
/// digit); every char of a run of [`LONG_RUN`]+ non-whitespace ASCII chars 2;
/// Hangul syllables `hangul`; any other char 2 per UTF-8 byte, the most a
/// byte-level tokenizer can spend on it. Unlike
/// [`crate::embedder::context_limit::estimate_tokens_upper`] it stays an upper
/// bound on digit-heavy tables, ids, URLs, base64 and unusual Unicode.
#[derive(Debug, Clone, Copy)]
struct Tokens {
    hangul: usize,
}

/// Hangul at 1.5 tokens: on real AutoRAG mirrors no 1843-token window of
/// consecutive lines exceeded 1707 Qwen3 or KURE tokens.
const LOCAL: Tokens = Tokens { hangul: 3 };
/// Hangul at 2 tokens, as the context guard of a remote (`openai:`, `tei:`)
/// embedder counts it: every other weight is already at least the guard's,
/// so the guard never cuts a chunk.
const REMOTE: Tokens = Tokens { hangul: 4 };

/// [`Tokens`] count for the local tokenizers.
pub(crate) fn tokens_upper(text: &str) -> usize {
    LOCAL.count(text)
}

fn remote_tokens_upper(text: &str) -> usize {
    REMOTE.count(text)
}

impl Tokens {
    fn count(self, text: &str) -> usize {
        let mut halves = 0;
        let mut run = 0;
        let mut run_halves = 0;
        for ch in text.chars() {
            if ch.is_ascii() && !ch.is_ascii_whitespace() {
                run += 1;
                run_halves += ascii_halves(ch);
                continue;
            }
            halves += run_weight(run, run_halves) + self.char_halves(ch);
            run = 0;
            run_halves = 0;
        }
        (halves + run_weight(run, run_halves)).div_ceil(2)
    }

    /// Half tokens of a char that is not part of an ASCII run.
    fn char_halves(self, ch: char) -> usize {
        match ch {
            _ if ch.is_ascii() => 1,
            '\u{AC00}'..='\u{D7A3}' => self.hangul,
            _ => 2 * ch.len_utf8(),
        }
    }

    /// Byte length of the longest prefix of `text` within `limit` tokens. A
    /// run still open at the prefix end is counted at its final weight once it
    /// is long, so the result never undercounts the piece it becomes.
    fn fitting_prefix(self, text: &str, limit: usize) -> usize {
        let limit = limit.saturating_mul(2);
        let mut halves = 0;
        let mut run = 0;
        let mut run_halves = 0;
        for (index, ch) in text.char_indices() {
            let open = if ch.is_ascii() && !ch.is_ascii_whitespace() {
                run += 1;
                run_halves += ascii_halves(ch);
                run_weight(run, run_halves)
            } else {
                halves += run_weight(run, run_halves) + self.char_halves(ch);
                run = 0;
                run_halves = 0;
                0
            };
            if halves + open > limit {
                return index;
            }
        }
        text.len()
    }
}

/// Half tokens of a non-whitespace ASCII char outside a long run.
fn ascii_halves(ch: char) -> usize {
    if ch.is_ascii_alphabetic() {
        1
    } else {
        2
    }
}

fn run_weight(run: usize, run_halves: usize) -> usize {
    if run >= LONG_RUN {
        2 * run
    } else {
        run_halves
    }
}

pub struct LineCdcChunker {
    budget: usize,
    tokens: Tokens,
    schema_id: String,
    context: HeadingContext,
}

impl LineCdcChunker {
    /// A chunker whose chunks never exceed `budget` [`tokens_upper`] tokens.
    pub fn new(budget: usize) -> Self {
        Self::with_context(budget, LOCAL, HeadingContext::off(tokens_upper))
    }

    /// Budget: `max_chunk_size` (in tokens), capped by 90% of the dense
    /// embedder's and of the sparse encoder's `max_length` less their passage
    /// prefix, less the heading context's room.
    pub fn from_config(config: &Config) -> Self {
        let id = config.embedder.id.as_str();
        let remote =
            !config.is_sparse_only() && (id.starts_with("openai:") || id.starts_with("tei:"));
        let (tokens, measure): (Tokens, Measure) = if remote {
            (REMOTE, remote_tokens_upper)
        } else {
            (LOCAL, tokens_upper)
        };
        let room = |max_length: usize, prefix: &Option<String>| {
            budget(max_length).saturating_sub(prefix.as_deref().map_or(0, measure))
        };
        let sparse = config
            .sparse
            .max_length
            .unwrap_or(DEFAULT_SPARSE_MAX_LENGTH);
        let mut cap = config
            .chunker
            .options
            .max_chunk_size
            .min(room(sparse, &config.sparse.passage_prefix));
        if let Some(max_length) = crate::embedder::dense_max_length(config) {
            cap = cap.min(room(max_length, &config.embedder.passage_prefix));
        }
        let context = HeadingContext::new(config.chunker.options.heading_context, cap, measure);
        Self::with_context(context.body_cap(cap), tokens, context)
    }

    fn with_context(budget: usize, tokens: Tokens, context: HeadingContext) -> Self {
        Self {
            budget: budget.max(1),
            tokens,
            schema_id: context.schema_id("line-cdc"),
            context,
        }
    }

    /// Largest chunk body, in tokens of the bound it counts with.
    pub fn budget(&self) -> usize {
        self.budget
    }
}

impl Chunker for LineCdcChunker {
    fn schema_id(&self) -> &str {
        &self.schema_id
    }

    fn chunk(&self, text: &str, path: &str) -> Result<Vec<Chunk>> {
        let count = |text: &str| self.tokens.count(text);
        let mut builder = Builder::new(self.budget, self.tokens);
        let groups = groups(text);
        let mut table = Table::None;
        for (index, group) in groups.iter().enumerate() {
            let line = &text[group.start..group.line_end];
            let tail_blank = group.end > group.line_end;
            let next = groups
                .get(index + 1)
                .map(|next| &text[next.start..next.line_end]);

            let role = match table {
                _ if !is_table_line(line) => Role::Text,
                Table::None if !tail_blank && next.is_some_and(is_table_separator) => Role::Header,
                Table::Header(start) if is_table_separator(line) => {
                    let header = &text[start..group.end];
                    table = Table::Rows(header, count(header));
                    Role::Separator
                }
                _ => Role::Row,
            };
            if role == Role::Header {
                table = Table::Header(group.start);
                // Keep the header, its separator and the first row together.
                let first_rows: usize = groups[index..groups.len().min(index + 3)]
                    .iter()
                    .map(|group| count(&text[group.start..group.end]))
                    .sum();
                builder.make_room(first_rows);
            }
            let header = match (role, table) {
                (Role::Row, Table::Rows(header, tokens)) => Some((header, tokens)),
                _ => None,
            };

            let body = &text[group.start..group.end];
            let tokens = count(body);
            builder.push_group(body, group.start, tokens, header);
            if matches!(role, Role::Text | Role::Row) {
                builder.maybe_cut(line, tokens, weight(line, tail_blank, next));
            }
            if role == Role::Text || tail_blank {
                table = Table::None;
            }
        }
        let chunks = builder
            .finish()
            .into_iter()
            .filter(|(_, text)| !text.trim().is_empty())
            .map(|(start, text)| {
                (
                    start,
                    Chunk {
                        text,
                        chunk_type: "chunk".to_string(),
                        heading_path: String::new(),
                    },
                )
            })
            .collect();
        Ok(self.context.apply(text, path, chunks))
    }
}

/// A content line and the blank lines after it: blank lines never start a
/// chunk, they stay with the paragraph they close.
struct Group {
    start: usize,
    line_end: usize,
    end: usize,
}

fn groups(text: &str) -> Vec<Group> {
    let mut groups: Vec<Group> = Vec::new();
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        let start = offset;
        offset += line.len();
        if line.trim().is_empty() {
            if let Some(last) = groups.last_mut() {
                last.end = offset;
                continue;
            }
        }
        groups.push(Group {
            start,
            line_end: offset,
            end: offset,
        });
    }
    groups
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Text,
    Header,
    Separator,
    Row,
}

/// Markdown pipe table being read.
#[derive(Clone, Copy)]
enum Table<'a> {
    None,
    /// Its header line starts at this offset; the separator comes next.
    Header(usize),
    /// Data rows: this header and separator start every chunk holding them.
    Rows(&'a str, usize),
}

/// Cut probability multiplier, in quarters: x4 where a paragraph, sentence or
/// section ends; /4 after a heading (keep it with its text) and where prose
/// wraps mid-sentence onto the next line.
fn weight(line: &str, tail_blank: bool, next: Option<&str>) -> u64 {
    const STRONG: u64 = 16;
    const NEUTRAL: u64 = 4;
    const WEAK: u64 = 1;
    let trimmed = line.trim();
    let next_opens_block =
        next.is_some_and(|next| is_heading(next) || (is_table_line(next) && !is_table_line(line)));
    if is_heading(line) {
        WEAK
    } else if tail_blank || next_opens_block || ends_sentence(trimmed) {
        STRONG
    } else if wraps(trimmed, next) {
        WEAK
    } else {
        NEUTRAL
    }
}

fn is_heading(line: &str) -> bool {
    let content = line.trim_start_matches(' ');
    if line.len() - content.len() > 3 {
        return false;
    }
    let level = content.bytes().take_while(|&byte| byte == b'#').count();
    (1..=6).contains(&level)
        && content[level..]
            .chars()
            .next()
            .is_none_or(char::is_whitespace)
}

fn ends_sentence(trimmed: &str) -> bool {
    trimmed
        .trim_end_matches(['"', '\'', ')', ']', '”', '’', '」', '』', '*'])
        .chars()
        .next_back()
        .is_some_and(is_sentence_end)
}

/// Punctuation that ends a sentence or a clause, and the syllables Korean
/// sentences end on (`...합니다`, `...해요`, `...함`).
fn is_sentence_end(ch: char) -> bool {
    matches!(
        ch,
        '.' | '!'
            | '?'
            | ':'
            | ';'
            | '。'
            | '！'
            | '？'
            | '…'
            | '다'
            | '요'
            | '죠'
            | '음'
            | '임'
            | '함'
            | '됨'
    )
}

/// A prose line that stops mid-sentence and continues on the next line.
fn wraps(trimmed: &str, next: Option<&str>) -> bool {
    !trimmed.contains('|')
        && trimmed
            .chars()
            .last()
            .is_some_and(|ch| ch.is_alphanumeric() || ch == ',' || ch == '-')
        && next
            .and_then(|next| next.trim_start().chars().next())
            .is_some_and(char::is_alphanumeric)
}

fn is_table_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with('|') && trimmed.matches('|').count() >= 2
}

fn is_table_separator(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with('|')
        && trimmed.ends_with('|')
        && trimmed.len() > 2
        && trimmed.trim_matches('|').split('|').all(|cell| {
            let cell = cell.trim();
            !cell.is_empty() && cell.chars().all(|ch| ch == '-' || ch == ':')
        })
}

/// 64-bit hash of a line's content, fixed across platforms and releases: the
/// cut points (and so every doc id) depend on it. FNV-1a, then the splitmix64
/// finalizer so similar lines (consecutive row numbers) spread evenly.
fn line_hash(line: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in line.trim_end().bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash ^= hash >> 30;
    hash = hash.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    hash ^= hash >> 27;
    hash = hash.wrapping_mul(0x94d0_49bb_1331_11eb);
    hash ^ (hash >> 31)
}

/// Accumulates chunk bodies with their start offsets.
struct Builder {
    budget: usize,
    tokens: Tokens,
    min: usize,
    avg: usize,
    chunks: Vec<(usize, String)>,
    current: String,
    start: usize,
    size: usize,
}

impl Builder {
    fn new(budget: usize, tokens: Tokens) -> Self {
        Self {
            budget,
            tokens,
            min: budget / 4,
            avg: budget / 2,
            chunks: Vec::new(),
            current: String::new(),
            start: 0,
            size: 0,
        }
    }

    fn emit(&mut self) {
        if !self.current.is_empty() {
            self.chunks
                .push((self.start, std::mem::take(&mut self.current)));
        }
        self.size = 0;
    }

    /// Cut now unless `tokens` more fit.
    fn make_room(&mut self, tokens: usize) {
        if self.size + tokens > self.budget {
            self.emit();
        }
    }

    /// Append one group of `tokens` tokens, cut into pieces if it alone is over
    /// the budget. A table row that opens a chunk gets its table's `header`
    /// (text and tokens) first.
    fn push_group(
        &mut self,
        text: &str,
        offset: usize,
        tokens: usize,
        header: Option<(&str, usize)>,
    ) {
        let header = header.filter(|&(_, tokens)| tokens <= self.budget / 2);
        let limit = self.budget - header.map_or(0, |(_, tokens)| tokens);
        let pieces = if tokens <= limit {
            vec![(text, tokens)]
        } else {
            split_long(text, limit, self.tokens)
                .into_iter()
                .map(|piece| (piece, self.tokens.count(piece)))
                .collect()
        };
        let mut piece_offset = offset;
        for (piece, tokens) in pieces {
            self.make_room(tokens);
            if self.current.is_empty() {
                self.start = piece_offset;
                if let Some((header, header_tokens)) = header {
                    self.current.push_str(header);
                    self.size = header_tokens;
                }
            }
            self.current.push_str(piece);
            self.size += tokens;
            piece_offset += piece.len();
        }
    }

    /// Content-defined cut after `line`: probability `tokens / target` times
    /// `weight` quarters, drawn from the line's hash, once the chunk holds
    /// `min` tokens. The target drops from `avg` to `avg / 4` past `avg`, so
    /// sizes gather around `avg` and the forced cut at the budget stays rare.
    fn maybe_cut(&mut self, line: &str, tokens: usize, weight: u64) {
        if self.size < self.min {
            return;
        }
        let target = if self.size < self.avg {
            self.avg
        } else {
            self.avg / 4
        }
        .max(1);
        // u < p with u = hash / 2^53 and p = tokens * weight / (4 * target).
        let draw = u128::from(line_hash(line) >> 11) * (4 * target as u128);
        if draw < (tokens as u128 * u128::from(weight)) << 53 {
            self.emit();
        }
    }

    fn finish(mut self) -> Vec<(usize, String)> {
        self.emit();
        self.chunks
    }
}

/// `text` in pieces of at most `limit` tokens, cut after the last sentence end,
/// else after the last whitespace, else at a char, within each piece's room.
fn split_long(text: &str, limit: usize, tokens: Tokens) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut rest = text;
    loop {
        let fit = tokens.fitting_prefix(rest, limit);
        if fit == rest.len() {
            if !rest.is_empty() {
                pieces.push(rest);
            }
            return pieces;
        }
        let cut = preferred_cut(&rest[..fit]).unwrap_or(fit);
        let cut = if cut == 0 {
            rest.chars().next().map_or(rest.len(), char::len_utf8)
        } else {
            cut
        };
        pieces.push(&rest[..cut]);
        rest = &rest[cut..];
    }
}

/// End of the last sentence in the back half of `window`, else of its last
/// whitespace, so a split piece ends at a natural break.
fn preferred_cut(window: &str) -> Option<usize> {
    let floor = window.len() / 2;
    let mut whitespace = None;
    let mut previous: Option<char> = None;
    for (index, ch) in window.char_indices().rev() {
        if index < floor {
            break;
        }
        if ch.is_whitespace() {
            let end = index + ch.len_utf8();
            if previous.is_none_or(|after| !after.is_whitespace()) {
                whitespace.get_or_insert(end);
            }
            let before = window[..index].chars().next_back();
            if before.is_some_and(is_sentence_end) {
                return Some(end);
            }
        }
        previous = Some(ch);
    }
    whitespace
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use std::collections::HashMap;

    const BUDGET: usize = 1843;

    fn chunks(text: &str) -> Vec<String> {
        LineCdcChunker::new(BUDGET)
            .chunk(text, "doc.md")
            .expect("chunk")
            .into_iter()
            .map(|chunk| chunk.text)
            .collect()
    }

    /// Chunks of `after` whose exact text is not a chunk of `before`.
    fn changed(before: &[String], after: &[String]) -> usize {
        let mut old: HashMap<&str, usize> = HashMap::new();
        for chunk in before {
            *old.entry(chunk.as_str()).or_default() += 1;
        }
        after
            .iter()
            .filter(|chunk| match old.get_mut(chunk.as_str()) {
                Some(count) if *count > 0 => {
                    *count -= 1;
                    false
                }
                _ => true,
            })
            .count()
    }

    /// One self-describing line per spreadsheet row, as AutoRAG renders sheets.
    fn record_rows(rows: std::ops::Range<usize>) -> String {
        rows.map(|row| {
            format!(
                "[예산!A{row}:F{row}] 구분: 인건비 | 비목: 항목 {row} | 편성액: {} | 비고: 내부\n",
                row * 1_000
            )
        })
        .collect()
    }

    fn pipe_table(rows: usize) -> String {
        let mut table = String::from("| 번호 | 비목 | 금액 |\n| --- | --- | --- |\n");
        for row in 0..rows {
            table.push_str(&format!("| {row} | 항목 {row} | {} |\n", row * 7));
        }
        table
    }

    /// PDF-like prose: sentences wrapped at about 60 chars, often mid-sentence,
    /// with a blank line between paragraphs.
    fn wrapped_prose(paragraphs: usize) -> String {
        let mut text = String::new();
        for paragraph in 0..paragraphs {
            let mut words = String::new();
            for sentence in 0..(3 + paragraph % 4) {
                words.push_str(&format!(
                    "Paragraph {paragraph} sentence {sentence} explains how the budget of \
                     department {} was allocated and reviewed by the committee. ",
                    paragraph * 31 + sentence
                ));
            }
            let mut line = String::new();
            for word in words.split_whitespace() {
                if line.len() + word.len() > 60 {
                    text.push_str(line.trim_end());
                    text.push('\n');
                    line.clear();
                }
                line.push_str(word);
                line.push(' ');
            }
            text.push_str(line.trim_end());
            text.push_str("\n\n");
        }
        text
    }

    fn ends_a_sentence(chunk: &str) -> bool {
        let trimmed = chunk.trim_end();
        trimmed.ends_with('.') || chunk.ends_with("\n\n")
    }

    #[test]
    fn no_chunk_exceeds_the_token_budget() {
        let blob: String = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo0NTY3ODkrLw".repeat(400);
        let korean = "예산 편성액은 742,000원이며 비고는 내부 검토 중입니다. ".repeat(600);
        let documents = [
            record_rows(0..5_000),
            pipe_table(3_000),
            wrapped_prose(300),
            format!("intro\n{blob}\noutro\n"),
            format!("{korean}\n짧은 줄\n"),
            "1234567890 ".repeat(5_000),
        ];
        for document in &documents {
            for chunk in chunks(document) {
                assert!(
                    tokens_upper(&chunk) <= BUDGET,
                    "{} > {BUDGET}: {:?}",
                    tokens_upper(&chunk),
                    &chunk[..chunk.len().min(120)]
                );
            }
        }
    }

    #[test]
    fn chunks_end_at_line_ends_and_reassemble_the_source() {
        for document in [record_rows(0..5_000), wrapped_prose(300)] {
            let chunks = chunks(&document);

            assert!(chunks.len() > 10, "{} chunks", chunks.len());
            assert!(chunks.iter().all(|chunk| chunk.ends_with('\n')));
            assert_eq!(chunks.concat(), document);
        }
    }

    #[test]
    fn only_a_line_over_the_budget_is_cut_inside() {
        let long = "가나다라 마바사 아자차. ".repeat(2_000);
        let document = format!("before\n{long}\nafter\n");

        let chunks = chunks(&document);

        assert_eq!(chunks.concat(), document);
        let inside = chunks.iter().filter(|chunk| !chunk.ends_with('\n')).count();
        assert!(inside >= 2, "the long line was not split: {inside}");
        for chunk in chunks.iter().filter(|chunk| !chunk.ends_with('\n')) {
            assert!(
                chunk.ends_with(". "),
                "cut mid-sentence: {:?}",
                &chunk[chunk.len() - 20..]
            );
        }
    }

    #[test]
    fn boundaries_resync_after_edits_to_a_30000_row_sheet() {
        let before = record_rows(0..30_000);
        let old = chunks(&before);

        let inserted = format!("{}{}", record_rows(100_000..100_001), before);
        let top = changed(&old, &chunks(&inserted));
        assert!(
            top * 100 <= old.len(),
            "insert at top changed {top} of {}",
            old.len()
        );

        let edited = before.replacen("비목: 항목 15000 ", "비목: 항목 15000 수정 ", 1);
        assert_ne!(edited, before);
        let middle = changed(&old, &chunks(&edited));
        assert!(middle <= 2, "editing one row changed {middle} chunks");

        let appended = format!("{before}{}", record_rows(100_000..100_001));
        let end = changed(&old, &chunks(&appended));
        assert!(end <= 1, "appending one row changed {end} chunks");
    }

    #[test]
    fn boundaries_resync_after_a_paragraph_is_added_to_prose() {
        let before = wrapped_prose(400);
        let old = chunks(&before);

        let after = format!("A new opening paragraph was added to the report.\n\n{before}");
        let top = changed(&old, &chunks(&after));

        assert!(
            top <= 3,
            "adding a paragraph changed {top} of {} chunks",
            old.len()
        );
    }

    #[test]
    fn prose_chunks_end_at_sentence_or_paragraph_ends() {
        let chunks = chunks(&wrapped_prose(400));
        let body = &chunks[..chunks.len() - 1];

        let sentence_ends = body.iter().filter(|chunk| ends_a_sentence(chunk)).count();

        assert!(
            sentence_ends * 100 >= body.len() * 85,
            "{sentence_ends} of {} chunks end a sentence",
            body.len()
        );
    }

    #[test]
    fn every_chunk_with_table_rows_starts_its_rows_with_the_header() {
        let document = format!(
            "# 예산\n\n머리말 문단입니다.\n\n{}\n맺음말입니다.\n",
            pipe_table(4_000)
        );

        let chunks = chunks(&document);

        let with_rows: Vec<&String> = chunks
            .iter()
            .filter(|chunk| chunk.contains("| 항목 "))
            .collect();
        assert!(with_rows.len() > 5, "{} chunks hold rows", with_rows.len());
        for chunk in with_rows {
            let header = chunk.find("| 번호 | 비목 | 금액 |\n| --- | --- | --- |\n");
            let first_row = chunk.find("| 항목 ").expect("row");
            assert!(
                header.is_some_and(|header| header < first_row),
                "rows without their header: {:?}",
                &chunk[..chunk.len().min(80)]
            );
        }
        assert!(chunks
            .iter()
            .all(|chunk| !chunk.ends_with("| --- | --- | --- |\n")));
    }

    #[test]
    fn cut_points_are_fixed_across_releases() {
        // The cut points decide every doc id: a change here re-embeds every
        // workspace, so it must be deliberate (and come with a new schema id).
        let document = format!("{}{}", record_rows(0..3_000), wrapped_prose(60));
        let lengths: Vec<usize> = chunks(&document).iter().map(String::len).collect();

        let digest = crate::id::content_hash(&format!("{lengths:?}"));

        assert_eq!(
            digest,
            "01768e30bcdb8d5d8bffb65d3cb3a73bf770add461d64594b44b47ab33662cb8"
        );
    }

    fn config(edit: impl FnOnce(&mut Config)) -> Config {
        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        config.chunker.id = "line-cdc".to_string();
        config.chunker.options.heading_context = false;
        edit(&mut config);
        config
    }

    #[test]
    fn the_budget_is_what_both_encoders_take_whole() {
        // Native Qwen3 (2048) and the KURE sparse encoder (2048): 90% of 2048,
        // less the 3 tokens of the sparse passage prefix `[D] `.
        assert_eq!(LineCdcChunker::from_config(&config(|_| {})).budget(), 1840);
        // `max_chunk_size` (tokens here) caps it further.
        let small = config(|config| config.chunker.options.max_chunk_size = 500);
        assert_eq!(LineCdcChunker::from_config(&small).budget(), 500);
        // A long-context remote model and sparse encoder: `max_chunk_size` binds.
        let remote = config(|config| {
            config.embedder.id = "openai:text-embedding-3-small".to_string();
            config.sparse.max_length = Some(8192);
        });
        assert_eq!(LineCdcChunker::from_config(&remote).budget(), 4096);
        // Sparse-only workspace: only the sparse encoder counts.
        let sparse_only = config(|config| {
            config.embedder.id = "none".to_string();
            config.sparse.max_length = Some(1024);
        });
        assert_eq!(LineCdcChunker::from_config(&sparse_only).budget(), 921 - 3);
        // A remote embedder's passage prefix counts too.
        let prefixed = config(|config| {
            config.embedder.id = "tei:intfloat/multilingual-e5-small".to_string();
            config.embedder.max_length = Some(512);
            config.embedder.passage_prefix = Some("passage: ".to_string());
        });
        assert_eq!(LineCdcChunker::from_config(&prefixed).budget(), 460 - 5);
        // The heading context's room comes out of the budget.
        let heading = config(|config| config.chunker.options.heading_context = true);
        assert_eq!(
            LineCdcChunker::from_config(&heading).budget(),
            1840 - 1840 / 8
        );
    }

    #[test]
    fn line_cdc_is_selectable_by_id() {
        let chunker = crate::chunker::create_chunker(&config(|_| {})).expect("create");
        assert_eq!(chunker.schema_id(), "line-cdc");
    }

    #[test]
    fn the_token_bound_counts_digits_punctuation_long_runs_and_unusual_chars_in_full() {
        // Letters and spaces in words: half a token each.
        assert_eq!(tokens_upper("budget is ok"), 6);
        // Digits and punctuation: a token each (Qwen3 splits every digit).
        assert_eq!(tokens_upper("742,000"), 7);
        // A run of 16+ non-space ASCII chars (ids, URLs, base64): a token each.
        assert_eq!(tokens_upper("abcdefghijklmnop"), 16);
        assert_eq!(tokens_upper("abcdefghijklmno"), 8);
        // Hangul syllables: one and a half tokens each.
        assert_eq!(tokens_upper("예산"), 3);
        // Anything else: a token per UTF-8 byte, the byte-level worst case
        // (U+034F and U+2007 fill e-mail preheaders).
        assert_eq!(tokens_upper("\u{034F}\u{2007}"), 5);
    }
}
