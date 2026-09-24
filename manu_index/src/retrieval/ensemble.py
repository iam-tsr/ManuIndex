from __future__ import annotations

from collections import OrderedDict

from .types import IndexedChunk, ScoredChunk


class EnsembleRetriever:
    def __init__(self, alpha: float = 0.5):
        self.alpha = min(1.0, max(0.0, alpha))

    def combine(
        self,
        dense_results: list[ScoredChunk],
        sparse_results: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        if top_k <= 0:
            return []

        dense_scores = self._normalize_scores(dense_results)
        sparse_scores = self._normalize_scores(sparse_results)
        chunks: OrderedDict[tuple[str, int], IndexedChunk] = OrderedDict()

        for result in dense_results + sparse_results:
            chunks.setdefault(result.chunk.key, result.chunk)

        combined = [
            ScoredChunk(
                chunk=chunk,
                score=(
                    self.alpha * dense_scores.get(key, 0.0)
                    + (1.0 - self.alpha) * sparse_scores.get(key, 0.0)
                ),
            )
            for key, chunk in chunks.items()
        ]
        combined.sort(key=lambda result: result.score, reverse=True)
        return combined[:top_k]

    def _normalize_scores(self, results: list[ScoredChunk]) -> dict[tuple[str, int], float]:
        if not results:
            return {}

        scores = [result.score for result in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return {result.chunk.key: 1.0 for result in results}

        return {
            result.chunk.key: (result.score - min_score) / (max_score - min_score)
            for result in results
        }
