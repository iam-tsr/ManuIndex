from __future__ import annotations

from typing import Sequence

import faiss
import numpy as np

from .types import IndexedChunk, ScoredChunk


class FaissRetriever:
    def __init__(self, chunks: Sequence[IndexedChunk]):
        self.chunks = [chunk for chunk in chunks if chunk.vector]
        self.index = None
        self.vectors = np.empty((0, 0), dtype=np.float32)

        if not self.chunks:
            return

        self.vectors = np.array([chunk.vector for chunk in self.chunks], dtype=np.float32)
        self.index = faiss.IndexHNSWFlat(self.vectors.shape[1], 32)
        self.index.add(self.vectors)

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        lambda_mult: float = 0.8,
    ) -> list[ScoredChunk]:
        if self.index is None or not self.chunks or top_k <= 0:
            return []

        query = np.array([query_vector], dtype=np.float32)
        fetch_k = min(len(self.chunks), max(top_k * 4, top_k))
        scores, indices = self.index.search(query, fetch_k)

        candidate_indices = [
            int(index)
            for index in indices[0]
            if int(index) >= 0
        ]
        candidate_scores = [float(score) for score in scores[0][:len(candidate_indices)]]
        selected = self._max_marginal_relevance(
            candidate_indices=candidate_indices,
            candidate_scores=candidate_scores,
            top_k=top_k,
            lambda_mult=lambda_mult,
        )

        score_by_index = dict(zip(candidate_indices, candidate_scores))
        return [
            ScoredChunk(chunk=self.chunks[index], score=score_by_index[index])
            for index in selected
        ]

    def _max_marginal_relevance(
        self,
        candidate_indices: list[int],
        candidate_scores: list[float],
        top_k: int,
        lambda_mult: float,
    ) -> list[int]:
        if not candidate_indices:
            return []

        lambda_mult = min(1.0, max(0.0, lambda_mult))
        selected = [candidate_indices[0]]
        remaining = candidate_indices[1:]
        score_by_index = dict(zip(candidate_indices, candidate_scores))

        while remaining and len(selected) < top_k:
            best_index = remaining[0]
            best_score = float("-inf")

            for index in remaining:
                relevance = score_by_index[index]
                diversity = max(
                    float(np.dot(self.vectors[index], self.vectors[selected_index]))
                    for selected_index in selected
                )
                mmr_score = lambda_mult * relevance - (1.0 - lambda_mult) * diversity
                if mmr_score > best_score:
                    best_index = index
                    best_score = mmr_score

            selected.append(best_index)
            remaining.remove(best_index)

        return selected