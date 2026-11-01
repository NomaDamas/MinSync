//! Heading context: a first line naming the source path and the Markdown
//! heading trail a chunk sits under (`docs/guide.md > Install > macOS`).
//!
//! The line is part of [`Chunk::text`], so the stored row, the dense embedder
//! and the sparse encoder all read exactly the same text, and the content hash
//! (and so the doc id) covers it. Its room is reserved out of the chunker's
//! cap, so a chunk with its context still never exceeds `max_chunk_size`.

use crate::types::Chunk;
use std::path::Path;

/// Measures text in the unit a chunker caps chunks in (chars, bytes, ...).
pub(crate) type Measure = fn(&str) -> usize;

/// Share of the cap held back for the context line: `cap / CONTEXT_SHARE`.
const CONTEXT_SHARE: usize = 8;

const SEPARATOR: &str = " > ";

/// File extensions whose `#` lines are Markdown headings. Anywhere else a `#`
/// line is a comment, a shebang or data, so those files get only their path.
const MARKDOWN_EXTENSIONS: [&str; 3] = ["md", "markdown", "mdx"];

/// Adds the heading context to chunk bodies, or leaves them as they are when
/// the context is off.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeadingContext {
    enabled: bool,
    /// Room for the context line and its newline, in the chunker's unit.
    reserve: usize,
    measure: Measure,
}

impl HeadingContext {
    /// The context for a chunker that caps chunks at `cap` units of `measure`.
    pub(crate) fn new(enabled: bool, cap: usize, measure: Measure) -> Self {
        Self {
            enabled,
            reserve: if enabled { cap / CONTEXT_SHARE } else { 0 },
            measure,
        }
    }

    pub(crate) fn off(measure: Measure) -> Self {
        Self::new(false, 0, measure)
    }

    /// The cap left for a chunk's body.
    pub(crate) fn body_cap(&self, cap: usize) -> usize {
        cap.saturating_sub(self.reserve)
    }

    /// Chunk schema id: the context changes every chunk, so it is part of it.
    pub(crate) fn schema_id(&self, base: &str) -> String {
        if self.enabled {
            format!("{base}+heading")
        } else {
            base.to_string()
        }
    }

    /// `chunks` pairs each chunk with the byte offset in `source` where its
    /// text begins, in source order. Each chunk gets the path and the
    /// headings in effect where it starts as its `heading_path` and as the
    /// first line of its text.
    pub(crate) fn apply(
        &self,
        source: &str,
        path: &str,
        chunks: Vec<(usize, Chunk)>,
    ) -> Vec<Chunk> {
        if !self.enabled {
            return chunks.into_iter().map(|(_, chunk)| chunk).collect();
        }
        let mut trail = Trail::new(source, is_markdown(path));
        chunks
            .into_iter()
            .map(|(start, mut chunk)| {
                let context = self.fit(path, &trail.at(start));
                if !context.is_empty() {
                    chunk.text = format!("{context}\n{}", chunk.text);
                }
                chunk.heading_path = context;
                chunk
            })
            .collect()
    }

    /// The longest context line that fits the reserve with its newline:
    /// outermost headings go first, then the end of the path.
    fn fit(&self, path: &str, headings: &[&str]) -> String {
        let fits = |line: &str| (self.measure)(line) + (self.measure)("\n") <= self.reserve;
        for skip in 0..headings.len() {
            let line = std::iter::once(path)
                .chain(headings[skip..].iter().copied())
                .collect::<Vec<_>>()
                .join(SEPARATOR);
            if fits(&line) {
                return line;
            }
        }
        let mut line = path.to_string();
        while !line.is_empty() && !fits(&line) {
            line.pop();
        }
        line
    }
}

fn is_markdown(path: &str) -> bool {
    Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            MARKDOWN_EXTENSIONS
                .iter()
                .any(|known| extension.eq_ignore_ascii_case(known))
        })
}

struct Heading<'a> {
    offset: usize,
    level: usize,
    text: &'a str,
}

/// The headings in effect at increasing offsets of one source text.
struct Trail<'a> {
    source: &'a str,
    headings: Vec<Heading<'a>>,
    /// Index of the first heading not yet on `stack`.
    next: usize,
    /// `(level, text)` of the open sections, outermost first.
    stack: Vec<(usize, &'a str)>,
}

impl<'a> Trail<'a> {
    fn new(source: &'a str, markdown: bool) -> Self {
        Self {
            source,
            headings: if markdown {
                atx_headings(source)
            } else {
                Vec::new()
            },
            next: 0,
            stack: Vec::new(),
        }
    }

    /// Headings a chunk starting at byte `start` sits under. A chunk whose
    /// first non-blank line is itself a heading names only that heading's
    /// ancestors: the heading is already in its text.
    fn at(&mut self, start: usize) -> Vec<&'a str> {
        let first = first_content(self.source, start);
        if self
            .next
            .checked_sub(1)
            .is_some_and(|last| self.headings[last].offset >= first)
        {
            // Out of order: rebuild from the top.
            self.next = 0;
            self.stack.clear();
        }
        while let Some(heading) = self.headings.get(self.next).filter(|h| h.offset < first) {
            close(&mut self.stack, heading.level);
            self.stack.push((heading.level, heading.text));
            self.next += 1;
        }
        let mut stack = self.stack.clone();
        if let Some(heading) = self.headings.get(self.next).filter(|h| h.offset == first) {
            close(&mut stack, heading.level);
        }
        stack
            .into_iter()
            .map(|(_, text)| text)
            .filter(|text| !text.is_empty())
            .collect()
    }
}

/// Close every open section at `level` or deeper.
fn close(stack: &mut Vec<(usize, &str)>, level: usize) {
    while stack.last().is_some_and(|&(open, _)| open >= level) {
        stack.pop();
    }
}

/// Offset of the first non-blank content at or after `start`. A start inside a
/// multi-byte char (byte-capped chunkers can cut there) moves to the next char.
fn first_content(source: &str, start: usize) -> usize {
    let mut offset = start.min(source.len());
    while !source.is_char_boundary(offset) {
        offset += 1;
    }
    for line in source[offset..].split_inclusive('\n') {
        if !line.trim().is_empty() {
            return offset;
        }
        offset += line.len();
    }
    offset
}

/// ATX headings (`#` to `######`) outside fenced code blocks, in order.
fn atx_headings(source: &str) -> Vec<Heading<'_>> {
    let mut headings = Vec::new();
    let mut fence: Option<(u8, usize)> = None;
    let mut offset = 0;
    for line in source.split_inclusive('\n') {
        let line_start = offset;
        offset += line.len();
        let line = line.trim_end_matches(['\n', '\r']);
        let content = line.trim_start_matches(' ');
        if line.len() - content.len() > 3 {
            continue; // indented code
        }
        if let Some((marker, length)) = fence_marker(content) {
            match fence {
                None => fence = Some((marker, length)),
                Some((open, open_length))
                    if marker == open
                        && length >= open_length
                        && content[length..].trim().is_empty() =>
                {
                    fence = None
                }
                Some(_) => {}
            }
            continue;
        }
        if fence.is_some() {
            continue;
        }
        if let Some((level, text)) = atx_heading(content) {
            headings.push(Heading {
                offset: line_start,
                level,
                text,
            });
        }
    }
    headings
}

/// A run of at least three backticks or tildes opening a line.
fn fence_marker(content: &str) -> Option<(u8, usize)> {
    let marker = *content.as_bytes().first()?;
    if marker != b'`' && marker != b'~' {
        return None;
    }
    let length = content.bytes().take_while(|&byte| byte == marker).count();
    (length >= 3).then_some((marker, length))
}

/// Level and text of an ATX heading line, without its optional closing `#`s.
fn atx_heading(content: &str) -> Option<(usize, &str)> {
    let level = content.bytes().take_while(|&byte| byte == b'#').count();
    if !(1..=6).contains(&level) {
        return None;
    }
    let rest = &content[level..];
    if !rest.is_empty() && !rest.starts_with([' ', '\t']) {
        return None;
    }
    let text = rest.trim();
    let open = text.trim_end_matches('#');
    let text = if open.is_empty() {
        ""
    } else if open.ends_with([' ', '\t']) {
        open.trim_end()
    } else {
        text
    };
    Some((level, text))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ChunkerOptions, Config};

    const GUIDE: &str = "# Guide\n\nIntro.\n\n## Install\n\nStep one.\n\n### macOS\n\nUse brew.\n\n## Usage\n\nRun it.\n";

    fn chars(text: &str) -> usize {
        text.chars().count()
    }

    fn body(text: &str) -> Chunk {
        Chunk {
            text: text.to_string(),
            chunk_type: "chunk".to_string(),
            heading_path: String::new(),
        }
    }

    /// Chunks starting at each of `starts` (a substring of `source`, searched
    /// from the previous start), each running to the next start.
    fn chunks_at(source: &str, starts: &[&str]) -> Vec<(usize, Chunk)> {
        let mut offsets = Vec::new();
        let mut from = 0;
        for start in starts {
            let offset = from + source[from..].find(start).expect("start in source");
            offsets.push(offset);
            from = offset + 1;
        }
        offsets
            .iter()
            .enumerate()
            .map(|(index, &offset)| {
                let end = offsets.get(index + 1).copied().unwrap_or(source.len());
                (offset, body(&source[offset..end]))
            })
            .collect()
    }

    fn context(cap: usize) -> HeadingContext {
        HeadingContext::new(true, cap, chars)
    }

    #[test]
    fn markdown_chunks_name_the_sections_they_start_in() {
        let chunks = context(4096).apply(
            GUIDE,
            "docs/guide.md",
            chunks_at(GUIDE, &["Intro.", "Step one.", "Use brew.", "Run it."]),
        );

        let trails: Vec<&str> = chunks.iter().map(|c| c.heading_path.as_str()).collect();
        assert_eq!(
            trails,
            [
                "docs/guide.md > Guide",
                "docs/guide.md > Guide > Install",
                "docs/guide.md > Guide > Install > macOS",
                "docs/guide.md > Guide > Usage",
            ]
        );
    }

    #[test]
    fn the_context_is_the_first_line_of_the_chunk_text() {
        let chunks = context(4096).apply(GUIDE, "docs/guide.md", chunks_at(GUIDE, &["Use brew."]));

        assert_eq!(
            chunks[0].text,
            "docs/guide.md > Guide > Install > macOS\nUse brew.\n\n## Usage\n\nRun it.\n"
        );
    }

    #[test]
    fn a_chunk_opening_with_a_heading_names_only_its_ancestors() {
        // The heading itself is the chunk's first body line; a blank line
        // before it does not change that.
        let chunks = context(4096).apply(
            GUIDE,
            "docs/guide.md",
            chunks_at(GUIDE, &["### macOS", "\n## Usage"]),
        );

        let trails: Vec<&str> = chunks.iter().map(|c| c.heading_path.as_str()).collect();
        assert_eq!(
            trails,
            ["docs/guide.md > Guide > Install", "docs/guide.md > Guide"]
        );
    }

    #[test]
    fn hash_lines_inside_code_fences_are_not_headings() {
        let source = "# Real\n\n```sh\n# not a heading\n~~~\n## still code\n```\n\nafter\n";

        let chunks = context(4096).apply(source, "x.md", chunks_at(source, &["after"]));

        assert_eq!(chunks[0].heading_path, "x.md > Real");
    }

    #[test]
    fn closing_hashes_and_indented_code_are_not_heading_text() {
        let source = "## Setup ##\n\n    # indented code\n\nbody\n";

        let chunks = context(4096).apply(source, "x.md", chunks_at(source, &["body"]));

        assert_eq!(chunks[0].heading_path, "x.md > Setup");
    }

    #[test]
    fn files_that_are_not_markdown_carry_only_their_path() {
        let source = "# a comment, not a heading\nprint('hi')\n";

        let chunks = context(4096).apply(source, "src/run.py", chunks_at(source, &["print"]));

        assert_eq!(chunks[0].heading_path, "src/run.py");
        assert_eq!(chunks[0].text, "src/run.py\nprint('hi')\n");
    }

    #[test]
    fn a_long_trail_drops_its_outermost_headings_first() {
        let outer = "O".repeat(30);
        let source = format!("# {outer}\n\n## Inner\n\nbody\n");
        // cap 320 chars reserves 40 for the line and its newline.
        let chunks = context(320).apply(&source, "p.md", chunks_at(&source, &["body"]));

        assert_eq!(chunks[0].heading_path, "p.md > Inner");
    }

    #[test]
    fn a_path_longer_than_the_reserve_is_cut_to_fit() {
        let path = format!("{}/file.md", "d".repeat(80));

        let chunks = context(320).apply("body\n", &path, chunks_at("body\n", &["body"]));

        // 40 chars reserved: 39 for the line, 1 for its newline.
        assert_eq!(chunks[0].heading_path, path[..39]);
    }

    #[test]
    fn chunk_starts_off_a_char_boundary_are_tolerated() {
        // Byte-capped chunkers can cut inside a multi-byte char.
        let source = "# 제목\n\n한국어 본문\n";
        let inside = source.find("국").expect("char") + 1;

        let chunks = context(4096).apply(source, "k.md", vec![(inside, body("x"))]);

        assert_eq!(chunks[0].heading_path, "k.md > 제목");
    }

    fn options(max_chunk_size: usize) -> ChunkerOptions {
        ChunkerOptions {
            max_chunk_size,
            heading_context: true,
            ..ChunkerOptions::default()
        }
    }

    fn chunker(id: &str, max_chunk_size: usize) -> Box<dyn crate::chunker::Chunker> {
        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        config.chunker.id = id.to_string();
        config.chunker.options = options(max_chunk_size);
        crate::chunker::create_chunker(&config).expect("create chunker")
    }

    /// Long nested Korean headings, paragraphs, a table and a delimiter-less
    /// line: every shape that pushes a chunker to its cap.
    fn stress_document() -> String {
        let mut document = String::new();
        for section in 0..6 {
            document.push_str(&format!(
                "# 장 {section} 매우 긴 제목이 붙은 문서의 첫 단계\n\n"
            ));
            for sub in 0..3 {
                document.push_str(&format!("## 절 {section}.{sub} 세부 항목 설명\n\n"));
                for sentence in 0..8 {
                    document.push_str(&format!(
                        "문장 {sentence}: 예산 편성액은 742,000원이며 비고는 내부 검토 중입니다. "
                    ));
                }
                document.push_str("\n\n| 항목 | 금액 |\n| --- | --- |\n");
                for row in 0..12 {
                    document.push_str(&format!("| 인건비 {row} | {} |\n", 1000 + row));
                }
                document.push('\n');
            }
        }
        document.push_str(&"x".repeat(900));
        document.push('\n');
        document
    }

    #[test]
    fn every_chunker_keeps_context_and_body_within_max_chunk_size() {
        let document = stress_document();
        let path = "보고서/2026/예산_편성_세부_내역서_최종본.hwp.md";
        // cdc pushes a cut forward to the next UTF-8 boundary: up to 3 bytes.
        let cases: [(&str, Measure, usize); 3] = [
            ("recursive", chars, 0),
            ("cdc", str::len, 3),
            ("chonkie", str::len, 0),
        ];
        for (id, measure, slack) in cases {
            for cap in [200, 512, 4096] {
                let chunks = chunker(id, cap).chunk(&document, path).expect("chunk");

                assert!(!chunks.is_empty(), "{id}/{cap}: no chunks");
                for chunk in &chunks {
                    assert!(
                        measure(&chunk.text) <= cap + slack,
                        "{id}/{cap}: {} > {cap}: {:?}",
                        measure(&chunk.text),
                        chunk.text
                    );
                    assert!(!chunk.heading_path.is_empty(), "{id}/{cap}: no context");
                    assert!(
                        chunk.text.starts_with(&format!("{}\n", chunk.heading_path)),
                        "{id}/{cap}: context is not the first line: {:?}",
                        chunk.text
                    );
                }
                // A small cap leaves room for (part of) the path only.
                if cap == 4096 {
                    assert!(
                        chunks.iter().any(|c| c.heading_path.contains(" > ")),
                        "{id}/{cap}: no chunk got a heading trail"
                    );
                }
            }
        }
    }

    #[test]
    fn the_context_is_part_of_the_chunk_schema() {
        for id in ["recursive", "cdc", "chonkie"] {
            assert_eq!(chunker(id, 4096).schema_id(), format!("{id}+heading"));
        }
    }

    #[test]
    fn chunks_without_the_context_keep_an_empty_heading_path() {
        let mut config = Config::default_for("12345678-1234-4234-9234-123456789abc");
        config.chunker.options.max_chunk_size = 64;
        let chunker = crate::chunker::create_chunker(&config).expect("create chunker");

        let chunks = chunker.chunk(GUIDE, "docs/guide.md").expect("chunk");

        assert_eq!(chunker.schema_id(), "recursive");
        assert!(chunks.iter().all(|c| c.heading_path.is_empty()));
        assert!(chunks.iter().all(|c| !c.text.starts_with("docs/guide.md")));
    }

    #[test]
    fn an_off_context_reserves_nothing() {
        assert_eq!(HeadingContext::off(chars).body_cap(4096), 4096);
        assert_eq!(HeadingContext::new(false, 4096, chars).body_cap(4096), 4096);
    }
}
