from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from .types import IndexedChunk, ScoredChunk


TOKEN_PATTERN = re.compile(r"\b\w+\b")


class BM25Retriever:
    def __init__(self, chunks: Sequence[IndexedChunk]):
        self.chunks = list(chunks)
        self.tokenized_documents = [self._tokenize(chunk.text) for chunk in self.chunks]
        self.index = (
            BM25Okapi(self.tokenized_documents)
            if self.chunks and any(self.tokenized_documents)
            else None
        )

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        if self.index is None or top_k <= 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)
        if len(scores) == 0:
            return []

        top_k = min(top_k, len(scores))
        indices = np.argpartition(scores, -top_k)[-top_k:]
        indices = indices[np.argsort(scores[indices])[::-1]]

        return [
            ScoredChunk(chunk=self.chunks[int(index)], score=float(scores[int(index)]))
            for index in indices
            if scores[int(index)] > 0
        ]

    def _tokenize(self, text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())
