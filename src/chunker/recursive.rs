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

        let bytes = text.as_bytes();
        let offsets = self.splitter.split(bytes, IncludeDelim::Prev, 0);
        let splits: Vec<&str> = offsets
            .into_iter()
            .filter_map(|(start, end)| std::str::from_utf8(&bytes[start..end]).ok())
            .collect();

        if splits.is_empty() {
            return Ok(vec![Chunk {
                text: text.to_string(),
                chunk_type: "chunk".to_string(),
                heading_path: String::new(),
            }]);
        }

        let token_counts: Vec<usize> = splits.iter().map(|split| split.chars().count()).collect();
        let merged = merge_splits(&splits, &token_counts, self.max_chunk_size).merged;

        let result = merged
            .into_iter()
            .map(|text| Chunk {
                text,
                chunk_type: "chunk".to_string(),
                heading_path: String::new(),
            })
            .filter(|c| !c.text.trim().is_empty())
            .collect();

        Ok(result)
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
            let contains_boundary = chunk.text.matches(". ").count() > 1;
            if contains_boundary {
                assert!(chunk.text.chars().count() <= max_chunk_size);
            }
        }
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
