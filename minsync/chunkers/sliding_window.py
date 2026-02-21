"""Line-based sliding window chunker for non-markdown files."""

from __future__ import annotations

from minsync.protocols import Chunk


class SlidingWindowChunker:
    """Splits text using a line-based sliding window.

    Accumulates lines until *max_chunk_size* characters are reached, then
    starts a new chunk carrying over *overlap* lines from the previous chunk.
    The first chunk is ``chunk_type="parent"``; the rest are ``"child"``.
    All chunks have ``heading_path=""``.
    """

    def __init__(self, max_chunk_size: int = 1000, overlap: int = 100) -> None:
        self._max_chunk_size = max(max_chunk_size, 1)
        self._overlap = max(overlap, 0)

    def schema_id(self) -> str:
        return "sliding-window-v1"

    def chunk(self, text: str, path: str) -> list[Chunk]:
        stripped = text.strip()
        if not stripped:
            return []

        lines = stripped.split("\n")
        if len("\n".join(lines)) <= self._max_chunk_size:
            return [Chunk(chunk_type="parent", text=stripped, heading_path="")]

        chunks: list[Chunk] = []
        start = 0

        while start < len(lines):
            acc_len = 0
            end = start
            while end < len(lines):
                line_len = len(lines[end]) + (1 if end > start else 0)
                if acc_len + line_len > self._max_chunk_size and end > start:
                    break
                acc_len += line_len
                end += 1

            chunk_text = "\n".join(lines[start:end]).strip()
            if chunk_text:
                chunk_type = "parent" if not chunks else "child"
                chunks.append(Chunk(chunk_type=chunk_type, text=chunk_text, heading_path=""))

            # Calculate overlap lines to carry over
            overlap_chars = 0
            overlap_start = end
            for i in range(end - 1, start, -1):
                line_len = len(lines[i]) + 1
                if overlap_chars + line_len > self._overlap:
                    break
                overlap_chars += line_len
                overlap_start = i

            start = overlap_start if overlap_start < end else end

        return chunks
