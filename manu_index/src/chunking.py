from __future__ import annotations

import re

from typing import Any

from langchain_core.documents import Document


def _split_paragraphs(text: str) -> list[str]:
    """Split text into non-empty Markdown-ish paragraphs/blocks."""
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", text) if block.strip()]
    return blocks or [text.strip()] if text.strip() else []


def _split_long_block(block: str, max_chars: int) -> list[str]:
    """Split a long block on whitespace into bounded character chunks."""
    words = block.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_len + extra > max_chars:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra

    if current:
        chunks.append(" ".join(current))
    return chunks


def create_deterministic_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 0,
    *,
    doc_id: str | None = None,
) -> list[Document]:
    """Create stable bounded chunks without embedding-based semantic merging.

    ``chunk_size`` is treated as an approximate max character budget. The helper
    prefers paragraph boundaries and splits long paragraphs on whitespace.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")

    pieces: list[str] = []
    for block in _split_paragraphs(text):
        if len(block) <= chunk_size:
            pieces.append(block)
        else:
            pieces.extend(_split_long_block(block, chunk_size))

    documents: list[Document] = []
    for index, piece in enumerate(piece for piece in pieces if piece.strip()):
        metadata: dict[str, Any] = {
            "chunk_index": index,
        }
        if doc_id is not None:
            metadata["doc_id"] = doc_id
        documents.append(Document(page_content=piece, metadata=metadata))
    return documents
