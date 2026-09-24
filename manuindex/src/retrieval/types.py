from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IndexedChunk:
    doc_id: str
    chunk_index: int
    text: str
    vector: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, int]:
        return self.doc_id, self.chunk_index


@dataclass
class ScoredChunk:
    chunk: IndexedChunk
    score: float
