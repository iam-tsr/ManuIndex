from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymupdf4llm.helpers.image_analyzer import BaseImageAnalyzer

from .summary import DocumentSummary
from .datastore import DataStore
from .retrieval import BM25Retriever, EnsembleRetriever, FaissRetriever, IndexedChunk, ScoredChunk
from .embed import Embedder
from .parser import image_analyzer as parse_pdf_document


DOC_TOP_K = 5

class ManuIndex:
    """
    ManuIndex is a class that provides functionality for indexing and searching documents.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str,
        embeddings: Embedder,
        datastore: DataStore,
        image_analyzer: BaseImageAnalyzer | None = None,
    ):
        """
        Args:
            api_key: API key used by PDF image analysis and summarization.
            embeddings: Embedding model used to encode summaries, chunks, and queries.
            model_name: Model name used for document summaries and PDF image analysis.
            base_url: API base URL used for OpenAI-compatible calls.
            datastore: Local datastore used for routing metadata and document chunks.
            image_analyzer: Preconfigured image analyzer for PDF image analysis.
        """
        self.api_key = api_key
        self.embeddings = embeddings
        self.model_name = model_name
        self.base_url = base_url
        self.datastore = datastore
        self.image_analyzer = image_analyzer

    def add_document(
        self,
        document: str | bytes | os.PathLike[str],
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 150,
    ) -> str:
        """Ingest text, a PDF path, or PDF bytes into local storage.

        Returns:
            The generated document id.
        """
        document_text = self._document_text(document)
        doc_id = uuid.uuid4().hex[:6]
        doc_metadata = self._document_metadata(
            metadata=metadata,
            source_name=self._source_name(document),
        )

        summary = self._create_summary(document_text)
        summary_embedding = self.embeddings.embed_query(summary)

        chunks = self._deterministic_splitter(
            document=document_text,
            doc_id=doc_id,
            chunk_size=chunk_size,
        )
        if not chunks:
            self.datastore.add_document(
                doc_id=doc_id,
                summary=summary,
                summary_vector=summary_embedding,
                metadata=doc_metadata,
                chunks=[],
            )
            return doc_id

        vectors = self.embeddings.embed_documents([chunk.text for chunk in chunks])
        records = [
            {
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "vector": list(vector),
                "text": chunk.text,
            }
            for chunk, vector in zip(chunks, vectors)
        ]
        self.datastore.add_document(
            doc_id=doc_id,
            summary=summary,
            summary_vector=summary_embedding,
            metadata=doc_metadata,
            chunks=records,
        )
        return doc_id

    def search(
        self,
        query: str,
        top_k: int = 5,
        top_c: int = 5,
        lambda_mult: float = 0.8,
        alpha: float = 0.5,
    ) -> list[str]:
        """Retrieve relevant passages for a query."""
        if top_k <= 0:
            return []

        query_embedding = self.embeddings.embed_query(query)
        doc_ids = self._find_collections(query_embedding, top_c=top_c)
        chunks = self._load_chunks(doc_ids)
        if not chunks:
            return []

        candidate_k = max(top_k, DOC_TOP_K * max(1, len(doc_ids)))
        candidates = self._hybrid_search(
            chunks=chunks,
            query=query,
            query_embedding=query_embedding,
            top_k=candidate_k,
            lambda_mult=lambda_mult,
            alpha=alpha,
        )
        expanded_chunks = self._neighbour_chunking(
            documents=candidates,
            chunks_by_key=self._chunks_by_key(chunks),
        )
        rerank_chunks = self._dedupe_chunks(expanded_chunks)
        if len(rerank_chunks) <= top_k:
            return [chunk.text for chunk in rerank_chunks]

        # The only cons in the whole process is that we have to re-embed the reranked chunks, which is a bit wasteful.
        final_vectors = self.embeddings.embed_documents([chunk.text for chunk in rerank_chunks])
        final_chunks = [
            IndexedChunk(
                doc_id=chunk.doc_id,
                chunk_index=index,
                text=chunk.text,
                vector=list(vector),
                metadata=chunk.metadata,
            )
            for index, (chunk, vector) in enumerate(zip(rerank_chunks, final_vectors))
        ]
        final_results = self._hybrid_search(
            chunks=final_chunks,
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            lambda_mult=lambda_mult,
            alpha=alpha,
        )
        return [result.chunk.text for result in final_results]

    def info(self) -> list[dict[str, Any]]:
        """Return metadata for every indexed document."""
        records = self.datastore.get_route_meta()
        return [
            {
                "doc_id": record["doc_id"],
                "summary": record.get("summary"),
                "metadata": record.get("metadata", {}),
            }
            for record in records
        ]

    def delete(self, doc_id: str) -> None:
        """Remove a document from local storage."""
        self.datastore.delete_document(doc_id)

    def clear(self) -> None:
        """Delete all indexed documents from local storage."""
        self.datastore.clear()

    def _document_text(
        self,
        document: str | bytes | os.PathLike[str],
    ) -> str:
        if isinstance(document, bytes):
            if self._is_pdf_bytes(document):
                return self._parse_pdf(document)
            return document.decode("utf-8", errors="ignore")

        if isinstance(document, os.PathLike):
            document = os.fspath(document)

        if not isinstance(document, str):
            raise ValueError("document must be text, a PDF path, or PDF bytes.")

        if Path(document).suffix.lower() == ".pdf":
            return self._parse_pdf(document)

        return document

    def _parse_pdf(self, document: str | bytes) -> str:
        # if self.image_analyzer:
        return parse_pdf_document(document, analyzer=self.image_analyzer)
        # return parse_pdf_document(document)

    def _is_pdf_bytes(self, document: bytes) -> bool:
        return b"%PDF" in document[:1024]

    def _source_name(self, document: str | bytes | os.PathLike[str]) -> str | None:
        if isinstance(document, os.PathLike):
            return Path(document).name

        if isinstance(document, str) and Path(document).suffix.lower() == ".pdf":
            return Path(document).name

        return None

    def _document_metadata(
        self,
        metadata: dict[str, Any] | None,
        source_name: str | None,
    ) -> dict[str, Any]:
        doc_metadata = dict(metadata or {})
        doc_metadata.setdefault("source", source_name or "unknown")
        doc_metadata.setdefault("created_at", datetime.now())
        return doc_metadata

    def _create_summary(self, document: str) -> str:
        return DocumentSummary(
            document=document,
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.model_name,
        ).summarize()

    def _find_collections(self, query_embedding: Sequence[float], top_c: int) -> list[str]:
        records = self.datastore.get_route_meta()
        records = [record for record in records if record.get("summary_vector")]
        if not records or top_c <= 0:
            return []

        matrix = np.array([record["summary_vector"] for record in records], dtype=np.float32)
        query = np.array(query_embedding, dtype=np.float32)
        scores = matrix @ query
        collection_count = min(top_c, len(records))
        top_indices = np.argpartition(scores, -collection_count)[-collection_count:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [records[int(index)]["doc_id"] for index in top_indices]

    def _load_chunks(self, doc_ids: Sequence[str]) -> list[IndexedChunk]:
        records = self.datastore.get_document_chunks(doc_ids)
        chunks = []
        for record in records:
            chunks.append(IndexedChunk(
                doc_id=record["doc_id"],
                chunk_index=int(record["chunk_index"]),
                text=record["text"],
                vector=list(record.get("vector") or []),
            ))
        return chunks

    def _hybrid_search(
        self,
        chunks: list[IndexedChunk],
        query: str,
        query_embedding: Sequence[float],
        top_k: int,
        lambda_mult: float,
        alpha: float,
    ) -> list[ScoredChunk]:
        dense_results = FaissRetriever(chunks).search(
            query_vector=query_embedding,
            top_k=top_k,
            lambda_mult=lambda_mult,
        )
        sparse_results = BM25Retriever(chunks).search(query=query, top_k=top_k)
        return EnsembleRetriever(alpha=alpha).combine(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=top_k,
        )

    def _deterministic_splitter(
        self,
        document: str,
        doc_id: str,
        chunk_size: int,
    ) -> list[IndexedChunk]:
        documents: list[IndexedChunk] = []

        for chunk_index, piece in enumerate(self._split_text(document, chunk_size)):
            if not piece.strip():
                continue

            documents.append(IndexedChunk(
                doc_id=doc_id,
                chunk_index=chunk_index,
                text=piece,
            ))

        return documents

    def _split_text(self, text: str, chunk_size: int) -> list[str]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", " "],
        )
        return text_splitter.split_text(text)

    def _chunks_by_key(self, chunks: list[IndexedChunk]) -> dict[tuple[str, int], IndexedChunk]:
        return {chunk.key: chunk for chunk in chunks}

    def _neighbour_chunking(
        self,
        documents: list[ScoredChunk],
        chunks_by_key: dict[tuple[str, int], IndexedChunk],
    ) -> list[IndexedChunk]:
        expanded_documents: list[IndexedChunk] = []
        used_chunk_keys: set[tuple[str, int]] = set()

        for result in documents:
            chunk = result.chunk
            if chunk.key in used_chunk_keys:
                continue

            neighbour_keys = [
                (chunk.doc_id, index)
                for index in (chunk.chunk_index - 1, chunk.chunk_index, chunk.chunk_index + 1)
                if (chunk.doc_id, index) in chunks_by_key
                and (chunk.doc_id, index) not in used_chunk_keys
            ]
            if not neighbour_keys:
                continue

            page_content = "\n".join(chunks_by_key[key].text for key in neighbour_keys)
            metadata = dict(chunk.metadata)
            metadata["chunk_indices"] = [key[1] for key in neighbour_keys]
            expanded_documents.append(IndexedChunk(
                doc_id=chunk.doc_id,
                chunk_index=chunk.chunk_index,
                text=page_content,
                metadata=metadata,
            ))
            used_chunk_keys.update(neighbour_keys)

        return expanded_documents

    def _dedupe_chunks(self, chunks: list[IndexedChunk]) -> list[IndexedChunk]:
        seen: set[str] = set()
        unique_documents: list[IndexedChunk] = []

        for chunk in chunks:
            if chunk.text in seen:
                continue
            seen.add(chunk.text)
            unique_documents.append(chunk)

        return unique_documents