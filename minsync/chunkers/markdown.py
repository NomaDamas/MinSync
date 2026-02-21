"""Markdown heading-based chunker (line-based splitting)."""

from __future__ import annotations

import re

from minsync.protocols import Chunk

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)")


class MarkdownHeadingChunker:
    """Splits markdown text on ``#``/``##``/``###`` headings.

    Each heading line becomes a *parent* chunk; the body text beneath it
    becomes one or more *child* chunks.  When the body exceeds
    *max_chunk_size* characters it is split on **line boundaries** so that
    chunks stay stable across small git diffs.

    Files with no headings are returned as a single parent chunk.
    """

    def __init__(self, max_chunk_size: int = 1000, overlap: int = 100) -> None:
        self._max_chunk_size = max(max_chunk_size, 1)
        self._overlap = max(overlap, 0)

    def schema_id(self) -> str:
        return "markdown-heading-v1"

    def chunk(self, text: str, path: str) -> list[Chunk]:
        lines = text.split("\n")
        chunks: list[Chunk] = []
        heading_stack: list[str] = []
        current_heading_line: str | None = None
        current_body_lines: list[str] = []

        def _flush_section() -> None:
            if current_heading_line is None:
                return
            hp = " > ".join(heading_stack)
            chunks.append(Chunk(chunk_type="parent", text=current_heading_line, heading_path=hp))
            body = "\n".join(current_body_lines).strip()
            if body:
                for sub_chunk in self._split_body(current_body_lines, hp):
                    chunks.append(sub_chunk)

        for line in lines:
            m = _HEADING_RE.match(line)
            if m:
                _flush_section()
                level = len(m.group(1))
                title = m.group(2).strip()
                while len(heading_stack) >= level:
                    heading_stack.pop()
                heading_stack.append(title)
                current_heading_line = line.strip()
                current_body_lines = []
            else:
                current_body_lines.append(line)

        _flush_section()

        if not chunks:
            stripped = text.strip()
            if stripped:
                chunks.append(Chunk(chunk_type="parent", text=stripped, heading_path=""))

        return chunks

    def _split_body(self, body_lines: list[str], heading_path: str) -> list[Chunk]:
        """Split body lines into child chunks respecting *max_chunk_size*."""
        stripped = _strip_empty_lines(body_lines)
        if not stripped:
            return []

        joined = "\n".join(stripped)
        if len(joined) <= self._max_chunk_size:
            return [Chunk(chunk_type="child", text=joined.strip(), heading_path=heading_path)]

        result: list[Chunk] = []
        start = 0
        while start < len(stripped):
            end = _accumulate_lines(stripped, start, self._max_chunk_size)
            chunk_text = "\n".join(stripped[start:end]).strip()
            if chunk_text:
                result.append(Chunk(chunk_type="child", text=chunk_text, heading_path=heading_path))
            start = _next_start_with_overlap(stripped, start, end, self._overlap)

        return result


def _accumulate_lines(lines: list[str], start: int, max_size: int) -> int:
    """Return the end index after accumulating lines up to *max_size* chars."""
    acc_len = 0
    end = start
    while end < len(lines):
        line_len = len(lines[end]) + (1 if end > start else 0)
        if acc_len + line_len > max_size and end > start:
            break
        acc_len += line_len
        end += 1
    return end


def _next_start_with_overlap(lines: list[str], start: int, end: int, overlap: int) -> int:
    """Compute the next start index, carrying back *overlap* chars of lines."""
    overlap_chars = 0
    overlap_start = end
    for i in range(end - 1, start, -1):
        line_len = len(lines[i]) + 1
        if overlap_chars + line_len > overlap:
            break
        overlap_chars += line_len
        overlap_start = i
    return end if overlap_start >= end else overlap_start


def _strip_empty_lines(lines: list[str]) -> list[str]:
    """Remove leading and trailing empty lines."""
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    end = len(lines)
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]
