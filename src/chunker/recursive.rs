use crate::chunker::Chunker;
use crate::error::Result;
use crate::types::Chunk;
use chunk::{merge_splits, IncludeDelim, PatternSplitter};

pub struct RecursiveChunker {
    splitter: PatternSplitter,
    max_chunk_size: usize,
    schema_id: String,
}

impl RecursiveChunker {
    pub fn new(max_chunk_size: usize) -> Self {
        let patterns: &[&[u8]] = &[b"\n\n", b". ", b"? ", b"! ", b"\n"];

        Self {
            splitter: PatternSplitter::new(patterns),
            max_chunk_size,
            schema_id: "recursive".to_string(),
        }
    }

    pub fn from_config(options: &crate::config::ChunkerOptions) -> Self {
        Self::new(options.max_chunk_size)
    }
}

impl Chunker for RecursiveChunker {
    fn schema_id(&self) -> &str {
        &self.schema_id
    }

    fn chunk(&self, text: &str, _path: &str) -> Result<Vec<Chunk>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let mut chunks = Vec::new();
        let mut plain_start = 0;
        let mut lines = text.split_inclusive('\n').peekable();
        let mut offset = 0;

        while let Some(line) = lines.next() {
            let line_start = offset;
            offset += line.len();
            if !is_table_line(line) {
                continue;
            }

            let mut table = String::from(line);
            while let Some(next) = lines.next_if(|next| is_table_line(next)) {
                table.push_str(next);
                offset += next.len();
            }

            if line_start > plain_start {
                chunks.extend(self.chunk_plain(&text[plain_start..line_start]));
            }
            chunks.extend(self.chunk_table(&table));
            plain_start = offset;
        }

        if plain_start < text.len() {
            chunks.extend(self.chunk_plain(&text[plain_start..]));
        }

        Ok(chunks
            .into_iter()
            .filter(|chunk: &Chunk| !chunk.text.trim().is_empty())
            .collect())
    }
}

impl RecursiveChunker {
    fn chunk_plain(&self, text: &str) -> Vec<Chunk> {
        let bytes = text.as_bytes();
        let offsets = self.splitter.split(bytes, IncludeDelim::Prev, 0);
        let splits: Vec<&str> = offsets
            .into_iter()
            .filter_map(|(start, end)| std::str::from_utf8(&bytes[start..end]).ok())
            .collect();
        let merged = if splits.is_empty() {
            vec![text.to_string()]
        } else {
            let token_counts: Vec<usize> =
                splits.iter().map(|split| split.chars().count()).collect();
            merge_splits(&splits, &token_counts, self.max_chunk_size).merged
        };

        merged
            .into_iter()
            .flat_map(|text| self.hard_cap(&text))
            .map(chunk)
            .collect()
    }

    fn chunk_table(&self, table: &str) -> Vec<Chunk> {
        let lines: Vec<&str> = table.split_inclusive('\n').collect();
        let has_header = lines
            .get(1)
            .is_some_and(|separator| is_table_separator(separator));
        if !has_header {
            return lines
                .into_iter()
                .flat_map(|line| self.hard_cap(line))
                .map(chunk)
                .collect();
        }

        let header = format!("{}{}", lines[0], lines[1]);
        let mut chunks = Vec::new();
        let mut current = header.clone();
        for row in &lines[2..] {
            if current != header
                && current.chars().count() + row.chars().count() > self.max_chunk_size
            {
                tracing::warn!(
                    max_chunk_size = self.max_chunk_size,
                    "table chunk boundary falls inside a markdown table"
                );
                chunks.push(chunk(current));
                current = header.clone();
            }
            current.push_str(row);
        }
        if !current.is_empty() {
            chunks.push(chunk(current));
        }

        chunks
            .into_iter()
            .inspect(|chunk| {
                let length = chunk.text.chars().count();
                if length > self.max_chunk_size {
                    tracing::warn!(
                        original_length = length,
                        max_chunk_size = self.max_chunk_size,
                        "table chunk exceeds max_chunk_size because table rows are kept intact"
                    );
                }
            })
            .collect()
    }

    fn hard_cap(&self, text: &str) -> Vec<String> {
        let length = text.chars().count();
        if length <= self.max_chunk_size {
            return vec![text.to_string()];
        }

        tracing::warn!(
            original_length = length,
            max_chunk_size = self.max_chunk_size,
            "splitting overlong delimiter-less content at character boundaries"
        );
        let mut chunks = Vec::new();
        let mut current = String::new();
        for character in text.chars() {
            current.push(character);
            if current.chars().count() == self.max_chunk_size {
                chunks.push(std::mem::take(&mut current));
            }
        }
        if !current.is_empty() {
            chunks.push(current);
        }
        chunks
    }
}

fn is_table_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with('|') && trimmed.matches('|').count() >= 2
}

fn is_table_separator(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with('|')
        && trimmed.ends_with('|')
        && trimmed.trim_matches('|').split('|').all(|cell| {
            cell.trim()
                .chars()
                .all(|character| character == '-' || character == ':')
        })
}

fn chunk(text: String) -> Chunk {
    Chunk {
        text,
        chunk_type: "chunk".to_string(),
        heading_path: String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_text() {
        let chunker = RecursiveChunker::new(100);

        let chunks = chunker.chunk("", "f.md").expect("chunk empty text");

        assert!(chunks.is_empty());
    }

    #[test]
    fn test_single_short_line() {
        let chunker = RecursiveChunker::new(100);

        let chunks = chunker
            .chunk("Hello world", "f.md")
            .expect("chunk single line");

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Hello world");
    }

    #[test]
    fn test_multi_paragraph() {
        let chunker = RecursiveChunker::new(18);
        let text = "First paragraph\n\nSecond paragraph\n\nThird paragraph";

        let chunks = chunker.chunk(text, "f.md").expect("chunk paragraphs");

        assert!(chunks.len() > 1);
        assert_eq!(chunks.concat_text(), text);
        assert!(chunks.iter().any(|chunk| chunk.text.ends_with("\n\n")));
    }

    #[test]
    fn test_merge_respects_max_chunk_size() {
        let max_chunk_size = 8;
        let chunker = RecursiveChunker::new(max_chunk_size);
        let text = "a. bb. ccc. dddd. unsplittablelongword";

        let chunks = chunker.chunk(text, "f.md").expect("chunk with max size");

        assert!(chunks.len() > 1);
        assert_eq!(chunks.concat_text(), text);
        for chunk in chunks {
            assert!(chunk.text.chars().count() <= max_chunk_size);
        }
    }

    #[test]
    fn test_table_keeps_header_and_data_row_together() {
        let chunker = RecursiveChunker::new(32);
        let text = "| h0500 | other |\n| --- | --- |\n| v0500 | value |";

        let chunks = chunker.chunk(text, "f.md").expect("chunk markdown table");

        assert!(chunks
            .iter()
            .any(|chunk| chunk.text.contains("h0500") && chunk.text.contains("v0500")));
    }

    #[test]
    fn test_table_parts_repeat_header_and_separator() {
        let chunker = RecursiveChunker::new(40);
        let text = "| header |\n| --- |\n| first |\n| second |\n| third |";

        let chunks = chunker.chunk(text, "f.md").expect("chunk markdown table");

        assert!(chunks.len() > 1);
        let header = "| header |";
        let separator = "| --- |";
        for chunk in chunks.iter().skip(1) {
            assert!(chunk.text.starts_with(&format!("{header}\n{separator}\n")));
        }
    }

    #[test]
    fn test_delimiterless_line_is_hard_capped() {
        let max_chunk_size = 4096;
        let chunker = RecursiveChunker::new(max_chunk_size);
        let text = "x".repeat(8001);

        let chunks = chunker.chunk(&text, "f.md").expect("chunk long line");

        assert_eq!(chunks.concat_text(), text);
        assert!(chunks
            .iter()
            .all(|chunk| chunk.text.chars().count() <= max_chunk_size));
    }

    #[test]
    fn test_schema_id() {
        let chunker = RecursiveChunker::new(100);

        assert_eq!(chunker.schema_id(), "recursive");
    }

    #[test]
    fn test_whitespace_only_filtered() {
        let chunker = RecursiveChunker::new(10);
        let text = "  \n\nreal\n  ";

        let chunks = chunker.chunk(text, "f.md").expect("chunk whitespace");

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text.trim(), "real");
    }

    #[test]
    fn test_from_config() {
        let options = crate::config::ChunkerOptions {
            max_chunk_size: 8,
            delimiters: "\n".to_string(),
        };
        let chunker = RecursiveChunker::from_config(&options);

        let chunks = chunker
            .chunk("alpha\nbeta\ngamma", "f.md")
            .expect("chunk from config");

        assert_eq!(chunker.schema_id(), "recursive");
        assert!(chunks.len() > 1);
        assert_eq!(chunks.concat_text(), "alpha\nbeta\ngamma");
    }

    trait ChunkTextExt {
        fn concat_text(&self) -> String;
    }

    impl ChunkTextExt for [Chunk] {
        fn concat_text(&self) -> String {
            self.iter().map(|chunk| chunk.text.as_str()).collect()
        }
    }
}
