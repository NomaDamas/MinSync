use crate::chunker::Chunker;
use crate::error::Result;
use crate::types::Chunk;

pub struct ChonkieChunker {
    max_chunk_size: usize,
    delimiters: Vec<u8>,
    schema_id: String,
}

impl ChonkieChunker {
    pub fn new(max_chunk_size: usize, delimiters: &str) -> Self {
        Self {
            max_chunk_size,
            delimiters: delimiters.as_bytes().to_vec(),
            schema_id: "chonkie".to_string(),
        }
    }

    pub fn from_config(options: &crate::config::ChunkerOptions) -> Self {
        Self::new(options.max_chunk_size, &options.delimiters)
    }
}

impl Chunker for ChonkieChunker {
    fn schema_id(&self) -> &str {
        &self.schema_id
    }

    fn chunk(&self, text: &str, _path: &str) -> Result<Vec<Chunk>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let bytes = text.as_bytes();
        let chunks: Vec<&[u8]> = chunk::chunk(bytes)
            .size(self.max_chunk_size)
            .delimiters(&self.delimiters)
            .collect();

        let result = chunks
            .into_iter()
            .map(|c| {
                let text = String::from_utf8_lossy(c).into_owned();
                Chunk {
                    text,
                    chunk_type: "chunk".to_string(),
                    heading_path: String::new(),
                }
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
        let chunker = ChonkieChunker::new(100, "\n.?!");

        let chunks = chunker.chunk("", "f.md").expect("chunk empty text");

        assert!(chunks.is_empty());
    }

    #[test]
    fn test_single_line() {
        let chunker = ChonkieChunker::new(100, "\n.?!");

        let chunks = chunker
            .chunk("Hello world", "f.md")
            .expect("chunk single line");

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Hello world");
    }

    #[test]
    fn test_multi_paragraph() {
        let chunker = ChonkieChunker::new(18, "\n");
        let text = "First paragraph\n\nSecond paragraph\n\nThird paragraph";

        let chunks = chunker.chunk(text, "f.md").expect("chunk paragraphs");

        assert!(chunks.len() > 1);
        assert_eq!(chunks.concat_text(), text);
        assert!(chunks.iter().any(|chunk| chunk.text.ends_with("\n\n")));
    }

    #[test]
    fn test_custom_delimiters() {
        let chunker = ChonkieChunker::new(8, "\n");
        let text = "alpha\nbeta\ngamma";

        let chunks = chunker.chunk(text, "f.md").expect("chunk newlines");

        assert!(chunks.len() > 1);
        assert_eq!(chunks.concat_text(), text);
        assert!(chunks[..chunks.len() - 1]
            .iter()
            .all(|chunk| chunk.text.ends_with('\n') || chunk.text.len() <= 8));
    }

    #[test]
    fn test_chunk_type_is_chunk() {
        let chunker = ChonkieChunker::new(8, "\n");

        let chunks = chunker
            .chunk("alpha\nbeta\ngamma", "f.md")
            .expect("chunk text");

        assert!(chunks.iter().all(|chunk| chunk.chunk_type == "chunk"));
    }

    #[test]
    fn test_heading_path_empty() {
        let chunker = ChonkieChunker::new(8, "\n");

        let chunks = chunker
            .chunk("alpha\nbeta\ngamma", "f.md")
            .expect("chunk text");

        assert!(chunks.iter().all(|chunk| chunk.heading_path.is_empty()));
    }

    #[test]
    fn test_schema_id() {
        let chunker = ChonkieChunker::new(100, "\n.?!");

        assert_eq!(chunker.schema_id(), "chonkie");
    }

    #[test]
    fn test_whitespace_only_filtered() {
        let chunker = ChonkieChunker::new(10, "\n ");
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
        let chunker = ChonkieChunker::from_config(&options);

        let chunks = chunker
            .chunk("alpha\nbeta\ngamma", "f.md")
            .expect("chunk from config");

        assert_eq!(chunker.schema_id(), "chonkie");
        assert!(chunks.len() > 1);
        assert_eq!(chunks.concat_text(), "alpha\nbeta\ngamma");
    }

    #[test]
    fn test_large_text() {
        let chunker = ChonkieChunker::new(512, "\n");
        let text = (0..300)
            .map(|i| format!("line {i:03} abcdefghijklmnopqrstuvwxyz\n"))
            .collect::<String>();

        let chunks = chunker.chunk(&text, "f.md").expect("chunk large text");

        assert!(text.len() >= 10_000);
        assert!(chunks.len() > 1);
        assert_eq!(chunks.concat_text(), text);
        assert!(chunks
            .iter()
            .all(|chunk| chunk.text.as_bytes().len() <= 512 + 40));
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
