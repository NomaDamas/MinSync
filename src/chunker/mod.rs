pub mod chonkie;

use crate::error::Result;
use crate::types::Chunk;

pub trait Chunker: Send + Sync {
    fn schema_id(&self) -> &str;
    fn chunk(&self, text: &str, path: &str) -> Result<Vec<Chunk>>;
}
