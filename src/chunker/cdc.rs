use crate::chunker::Chunker;
use crate::error::Result;
use crate::types::Chunk;

/// Content-defined chunker (FastCDC-style gear rolling hash).
///
/// Boundaries are chosen by content rather than position, so a small edit
/// near the top of a file only changes the chunk(s) it touches — downstream
/// chunk boundaries (and therefore content hashes / doc IDs) stay stable,
/// letting sync skip re-embedding unchanged chunks.
pub struct CdcChunker {
    min_size: usize,
    avg_size: usize,
    max_size: usize,
    schema_id: String,
}

impl CdcChunker {
    pub fn new(min_size: usize, avg_size: usize, max_size: usize) -> Self {
        Self {
            min_size,
            avg_size,
            max_size,
            schema_id: "cdc".to_string(),
        }
    }

    /// Derive CDC size parameters from `max_chunk_size`: max = max_chunk_size,
    /// avg = max/2, min = avg/4, with floors that keep the gear hash effective.
    pub fn from_config(options: &crate::config::ChunkerOptions) -> Self {
        let max_size = options.max_chunk_size.max(64);
        let avg_size = (max_size / 2).max(32);
        let min_size = (avg_size / 4).max(16);
        Self::new(min_size, avg_size, max_size)
    }
}

impl CdcChunker {
    /// FastCDC-style cut point for `bytes`: the gear hash restarts at each
    /// chunk, a stricter mask applies before `avg_size` and a looser one
    /// after, and `max_size` bounds the worst case.
    fn next_cut(&self, bytes: &[u8]) -> usize {
        let len = bytes.len();
        if len <= self.min_size {
            return len;
        }

        let max = self.max_size.min(len);
        let avg = self.avg_size.min(max);
        let bits = avg.next_power_of_two().trailing_zeros();
        let mask_strict: u64 = (1u64 << (bits + 2)) - 1;
        let mask_loose: u64 = (1u64 << bits.saturating_sub(2)) - 1;

        let table = gear_table();
        let mut hash: u64 = 0;
        for (index, &byte) in bytes.iter().enumerate().take(max).skip(self.min_size) {
            hash = (hash << 1).wrapping_add(table[byte as usize]);
            let mask = if index < avg { mask_strict } else { mask_loose };
            if hash & mask == 0 {
                return index + 1;
            }
        }
        max
    }
}

impl Chunker for CdcChunker {
    fn schema_id(&self) -> &str {
        &self.schema_id
    }

    fn chunk(&self, text: &str, _path: &str) -> Result<Vec<Chunk>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let mut chunks = Vec::new();
        let mut start = 0;
        while start < text.len() {
            let mut cut = start + self.next_cut(&text.as_bytes()[start..]);
            while cut < text.len() && !text.is_char_boundary(cut) {
                cut += 1;
            }
            chunks.push(&text[start..cut]);
            start = cut;
        }

        Ok(chunks
            .into_iter()
            .filter(|chunk| !chunk.trim().is_empty())
            .map(|chunk| Chunk {
                text: chunk.to_string(),
                chunk_type: "chunk".to_string(),
                heading_path: String::new(),
            })
            .collect())
    }
}

/// 256-entry gear table generated from a fixed seed (splitmix64) so cut
/// points are deterministic across runs and platforms.
fn gear_table() -> &'static [u64; 256] {
    static TABLE: std::sync::OnceLock<[u64; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut table = [0u64; 256];
        for entry in table.iter_mut() {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *entry = z ^ (z >> 31);
        }
        table
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::content_hash;

    fn sample_lines(count: usize) -> String {
        (0..count)
            .map(|i| format!("{i} {}\n", content_hash(&i.to_string())))
            .collect()
    }

    fn chunk_texts(chunker: &CdcChunker, text: &str) -> Vec<String> {
        chunker
            .chunk(text, "f.md")
            .expect("chunk succeeds")
            .into_iter()
            .map(|chunk| chunk.text)
            .collect()
    }

    #[test]
    fn test_empty_text() {
        let chunker = CdcChunker::new(16, 32, 64);

        let chunks = chunker.chunk("", "f.md").expect("chunk empty text");

        assert!(chunks.is_empty());
    }

    #[test]
    fn test_short_text_single_chunk() {
        let chunker = CdcChunker::new(16, 64, 128);

        let chunks = chunker.chunk("Hello world", "f.md").expect("chunk text");

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Hello world");
    }

    #[test]
    fn test_chunks_concat_to_original() {
        let chunker = CdcChunker::new(32, 64, 128);
        let text = sample_lines(200);

        let chunks = chunker.chunk(&text, "f.md").expect("chunk text");

        let concat: String = chunks.iter().map(|chunk| chunk.text.as_str()).collect();
        assert_eq!(concat, text);
        assert!(chunks.len() > 10, "expected many chunks for a large text");
    }

    #[test]
    fn test_respects_max_size() {
        let chunker = CdcChunker::new(32, 64, 128);
        let text = sample_lines(200);

        let chunks = chunker.chunk(&text, "f.md").expect("chunk text");

        // +3 slack: a cut may be pushed forward to the next UTF-8 boundary.
        assert!(chunks.iter().all(|chunk| chunk.text.len() <= 128 + 3));
    }

    #[test]
    fn test_boundaries_stable_under_top_insertion() {
        let chunker = CdcChunker::new(32, 64, 128);
        let original = sample_lines(200);
        let edited = {
            let mut lines: Vec<&str> = original.lines().collect();
            lines.insert(1, "INSERTED LINE");
            format!("{}\n", lines.join("\n"))
        };

        let before = chunk_texts(&chunker, &original);
        let after = chunk_texts(&chunker, &edited);

        let after_set: std::collections::HashSet<&String> = after.iter().collect();
        let retained = before
            .iter()
            .filter(|chunk| after_set.contains(chunk))
            .count();
        assert!(
            retained * 10 >= before.len() * 7,
            "expected >= 70% of chunks unchanged after a top insertion, got {retained}/{}",
            before.len()
        );
    }

    #[test]
    fn test_utf8_multibyte_safe() {
        let chunker = CdcChunker::new(16, 32, 64);
        let text: String = (0..200)
            .map(|i| format!("한글 텍스트 라인 {i} — émojis 😀 and 中文\n"))
            .collect();

        let chunks = chunker.chunk(&text, "f.md").expect("chunk multibyte text");

        let concat: String = chunks.iter().map(|chunk| chunk.text.as_str()).collect();
        assert_eq!(concat, text);
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_deterministic() {
        let chunker = CdcChunker::new(32, 64, 128);
        let text = sample_lines(100);

        assert_eq!(chunk_texts(&chunker, &text), chunk_texts(&chunker, &text));
    }

    #[test]
    fn test_schema_id() {
        let chunker = CdcChunker::new(16, 32, 64);

        assert_eq!(chunker.schema_id(), "cdc");
    }

    #[test]
    fn test_from_config() {
        let options = crate::config::ChunkerOptions {
            max_chunk_size: 256,
            delimiters: "\n".to_string(),
        };
        let chunker = CdcChunker::from_config(&options);

        assert_eq!(chunker.max_size, 256);
        assert_eq!(chunker.avg_size, 128);
        assert_eq!(chunker.min_size, 32);
    }

    #[test]
    fn test_from_config_floors_tiny_sizes() {
        let options = crate::config::ChunkerOptions {
            max_chunk_size: 1,
            delimiters: "\n".to_string(),
        };
        let chunker = CdcChunker::from_config(&options);

        assert!(chunker.min_size >= 16);
        assert!(chunker.avg_size >= 32);
        assert!(chunker.max_size >= 64);
        assert!(chunker.min_size < chunker.avg_size);
        assert!(chunker.avg_size <= chunker.max_size);
    }

    #[test]
    fn test_whitespace_only_filtered() {
        let chunker = CdcChunker::new(16, 32, 64);

        let chunks = chunker.chunk("   \n\n  \n", "f.md").expect("chunk text");

        assert!(chunks.is_empty());
    }
}
