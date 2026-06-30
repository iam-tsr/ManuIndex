import json
import os
import pickle
import shutil
import uuid
from typing import Any, List

import numpy as np

from openai import OpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from manu_index.src.summary import DocumentSummary
from manu_index.src.embed_infer import ONNXEmbedder
from manu_index.src.reranker_infer import ONNXReranker

META_FILENAME = "_meta.json"
DENSE_INDEX_SUFFIX = "dnse"
SPARSE_INDEX_SUFFIX = "spr.pkl"
DOC_TOP_K = 5


class ManuIndex:
    """Hybrid retrieval index that combines dense (FAISS) and lexical (BM25) search.

    Supports document ingestion with semantic chunking, persistent storage,
    and flexible retrieval strategies (dense, sparse, or hybrid).
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        embeddings: ONNXEmbedder,
        persist_directory: str = "manu_index_db"
    ):
        """
        Args:
            embeddings: Embedding model used to encode text into vectors.
            client: LLM client used for generating document summaries.
            model_name: Name of the language model to use.
            persist_directory: Directory where the index and metadata are stored.
        """
        self.embeddings = embeddings
        self.client = client
        self.model_name = model_name
        self.persist_directory = persist_directory

    def add_document(
        self,
        documents: str | bytes,
        chunk_size: int = 500
    ) -> FAISS:
        """Ingest a document into the index.

        Reads the document, splits it into semantic chunks, builds both a FAISS
        vector store and a BM25 lexical store, and persists them to disk.

        Args:
            documents: Raw document content — a UTF-8 string, bytes, or a path
                       to a Markdown file.
            chunk_size: Maximum size of chunks to create from the document.

        Returns:
            The FAISS vector store created for this document.
        """
        if not isinstance(documents, (str, bytes)):
            raise ValueError("documents must be a str or bytes.")

        if isinstance(documents, bytes):
            documents = documents.decode("utf-8", errors="ignore")

        if isinstance(documents, str) and documents.endswith(".md"):
            with open(documents, "r", encoding="utf-8") as f:
                documents = f.read()

        doc_id = uuid.uuid4().hex[:6]

        self._create_summary(documents, doc_id=doc_id)
        chunks = self._deterministic_splitter(documents, chunk_size=chunk_size)
        self._lexical_store(doc_id=doc_id, documents=chunks)

        vector_store = FAISS.from_documents(documents=chunks, embedding=self.embeddings)
        vector_store.save_local(self.persist_directory, index_name=self._dense_index_name(doc_id))
        return vector_store

    def search(
        self,
        query: str,
        top_k: int = 3,
        top_c: int = 5,
        lambda_mult: float = 0.8,
        alpha: float = 0.5,
        reranker: ONNXReranker | None = None,
    ) -> List[str]:
        """Retrieve relevant passages for a query.

        Args:
            query: Natural-language search query.
            top_k: Number of passages to return.
            top_c: Number of collections to retrieve from before reranking.
            lambda_mult: MMR diversity parameter (0 = max diversity, 1 = max relevance).
            alpha: Weight given to dense retrieval in hybrid mode (BM25 gets 1 - alpha).
            reranker: Optional reranking model used to re-rank retrieved documents. When omitted, retrieved candidates are returned in retrieval order.

        Returns:
            List of matching passage strings.
        """
        query_embedding = self.embeddings.embed_query(query)
        doc_ids = self._find_collections(query_embedding, top_c=top_c)

        candidates: list[Document] = []

        for doc_id in doc_ids:
            vector_store = FAISS.load_local(
                folder_path=self.persist_directory,
                embeddings=self.embeddings,
                index_name=self._dense_index_name(doc_id),
                allow_dangerous_deserialization=True,
            )

            retriever = self._hybrid_retrieval(
                dense=self._dense_retrieval(vector_store, DOC_TOP_K, lambda_mult),
                sparse=self._sparse_retrieval(doc_id, DOC_TOP_K),
                alpha=alpha,
            )
            chunks_by_index = self._documents_by_chunk_index(vector_store)
            retrieved_chunks = retriever.invoke(query)
            candidates.extend(
                self._neighbour_chunking(retrieved_chunks, chunks_by_index)
            )

        candidate_texts = self._dedupe_page_content(candidates)
        if not candidate_texts:
            return []

        if reranker is None:
            return candidate_texts[:top_k]

        ranked_documents = reranker.rerank(query=query, documents=candidate_texts)
        return [document for document, _score in ranked_documents[:top_k]]

    def info(self) -> List[dict]:
        """Return metadata (doc_id and summary) for every indexed document.

        Returns:
            List of ``{"doc_id": ..., "summary": ...}`` dicts.

        Raises:
            FileNotFoundError: If no documents have been indexed yet.
            ValueError: If the metadata file is corrupted.
        """
        data = self._load_meta()
        return [{"doc_id": e["doc_id"], "summary": e["summary"]} for e in data]

    def delete(self, doc_id: str) -> None:
        """Remove a document and all its associated files from the index.

        Args:
            doc_id: The document identifier returned when the document was added.
        """
        # Remove FAISS index files
        dense_index_name = self._dense_index_name(doc_id)
        for filename in (f"{dense_index_name}.faiss", f"{dense_index_name}.pkl"):
            dense_path = os.path.join(self.persist_directory, filename)
            if os.path.exists(dense_path):
                os.remove(dense_path)

        # Remove BM25 pickle
        bm25_path = self._sparse_index_path(doc_id)
        if os.path.exists(bm25_path):
            os.remove(bm25_path)

        # Remove from metadata
        meta_path = os.path.join(self.persist_directory, META_FILENAME)
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = []
            data = [e for e in data if e["doc_id"] != doc_id]
            with open(meta_path, "w") as f:
                json.dump(data, f, indent=2)

    def clear(self) -> None:
        """Delete the entire index directory and all its contents."""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)

    def _create_summary(self, document: str, doc_id: str) -> None:
        """Summarize a document and append the result to the metadata file."""
        summary = DocumentSummary(document=document, client=self.client, model_name=self.model_name).summarize()
        embedding = self.embeddings.embed_query(summary)

        entry = {
            "doc_id": doc_id,
            "values": embedding,
            "summary": summary,
        }

        os.makedirs(self.persist_directory, exist_ok=True)
        meta_path = os.path.join(self.persist_directory, META_FILENAME)

        existing = []
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []

        existing.append(entry)
        self._write_meta(existing)

    def _write_meta(self, entries: list) -> None:
        """Write metadata entries to disk with inline embedding arrays for readability."""
        meta_path = os.path.join(self.persist_directory, META_FILENAME)
        with open(meta_path, "w") as f:
            f.write("[\n")
            for i, entry in enumerate(entries):
                values_inline = json.dumps(entry["values"], separators=(", ", ": "))
                f.write(
                    f'  {{\n'
                    f'    "doc_id": {json.dumps(entry["doc_id"])},\n'
                    f'    "values": {values_inline},\n'
                    f'    "summary": {json.dumps(entry["summary"])}\n'
                    f'  }}'
                )
                f.write(",\n" if i < len(entries) - 1 else "\n")
            f.write("]\n")

    def _load_meta(self) -> list:
        """Load and return the metadata list, raising clear errors on failure."""
        meta_path = os.path.join(self.persist_directory, META_FILENAME)
        if not os.path.exists(meta_path):
            raise FileNotFoundError("Metadata file not found. Has any document been indexed?")
        with open(meta_path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                raise ValueError("Metadata file is corrupted.")

    def _dense_index_name(self, doc_id: str) -> str:
        """Return the FAISS index name for a document."""
        return f"{doc_id}{DENSE_INDEX_SUFFIX}"

    def _sparse_index_path(self, doc_id: str) -> str:
        """Return the BM25 pickle path for a document."""
        return os.path.join(self.persist_directory, f"{doc_id}{SPARSE_INDEX_SUFFIX}")

    def _find_collections(self, query_embedding, top_c: int) -> list[str]:
        """Return the top ``top_c`` doc_ids whose summaries are closest to the query."""
        data = self._load_meta()
        ids = [e["doc_id"] for e in data]
        matrix = np.array([e["values"] for e in data], dtype=np.float32)
        scores = np.dot(matrix, query_embedding)
        collection_count = max(1, min(top_c, len(ids)))
        top_indices = np.argpartition(scores, -collection_count)[-collection_count:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [ids[int(i)] for i in top_indices]

    def _lexical_store(self, doc_id: str, documents: List[Document]) -> None:
        """Build a BM25 retriever from documents and persist it to disk."""
        bm25_retriever = BM25Retriever.from_documents(documents)

        bm25_path = self._sparse_index_path(doc_id)
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_retriever, f)

    def _semantic_chunking(
        self,
        document: str,
        chunk_size: int,
        chunk_overlap: int,
        threshold: float,
    ) -> List[Document]:
        """Split a document into semantically coherent chunks.

        Steps:
            1. Split text into small units via RecursiveCharacterTextSplitter.
            2. Encode all units in one batch.
            3. Merge adjacent units whose cosine similarity meets ``threshold``.
        """
        raise NotImplementedError("This method is Deprecated. Use `deterministic_chunking` instead.")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[" "],
        )
        units = splitter.split_text(document)
        embeddings = self.embeddings.encode(units)

        chunks: list[str] = []
        current: list[str] = [units[0]]

        for i in range(1, len(units)):
            sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
            if sim >= threshold:
                current.append(units[i])
            else:
                chunks.append(" ".join(current))
                current = [units[i]]

        chunks.append(" ".join(current))
        return [Document(page_content=chunk) for chunk in chunks]
    
    def _deterministic_splitter(
        self,
        document: str,
        chunk_size: int,
    ) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", " "],
        )

        documents: list[Document] = []
        chunk_index = 0

        for piece in text_splitter.split_text(document):
            if not piece.strip():
                continue

            metadata: dict[str, Any] = {"chunk_index": chunk_index}
            documents.append(Document(page_content=piece, metadata=metadata))
            chunk_index += 1

        return documents

    def _documents_by_chunk_index(self, vector_store: FAISS) -> dict[int, Document]:
        """Return all documents from a FAISS store keyed by deterministic chunk index."""
        docstore = getattr(vector_store, "docstore", None)
        stored_documents = getattr(docstore, "_dict", {}).values()
        chunks_by_index: dict[int, Document] = {}

        for document in stored_documents:
            chunk_index = document.metadata.get("chunk_index")
            if chunk_index is None:
                continue
            chunks_by_index[int(chunk_index)] = document

        return chunks_by_index

    def _neighbour_chunking(
        self,
        documents: list[Document],
        chunks_by_index: dict[int, Document],
    ) -> list[Document]:
        """Merge retrieved chunks with neighbours while avoiding overlapping output.

        Retrieval can return adjacent chunks such as 4, 5, and 6. Expanding each
        seed independently would duplicate text across neighbouring windows, so
        this method only emits chunk indices that have not been emitted before.
        """
        expanded_documents: list[Document] = []
        used_chunk_indices: set[int] = set()

        for document in documents:
            chunk_index = document.metadata.get("chunk_index")
            if chunk_index is None:
                expanded_documents.append(document)
                continue

            chunk_index = int(chunk_index)
            if chunk_index in used_chunk_indices:
                continue

            neighbour_indices = [
                index
                for index in (chunk_index - 1, chunk_index, chunk_index + 1)
                if index in chunks_by_index and index not in used_chunk_indices
            ]
            if not neighbour_indices:
                continue

            page_content = "\n".join(
                chunks_by_index[index].page_content for index in neighbour_indices
            )
            metadata = dict(document.metadata)
            metadata["chunk_indices"] = neighbour_indices
            expanded_documents.append(Document(page_content=page_content, metadata=metadata))
            used_chunk_indices.update(neighbour_indices)

        return expanded_documents

    def _dedupe_page_content(self, documents: list[Document]) -> list[str]:
        """Return unique document text while preserving retrieval order."""
        seen: set[str] = set()
        unique_documents: list[str] = []

        for document in documents:
            page_content = document.page_content
            if page_content in seen:
                continue
            seen.add(page_content)
            unique_documents.append(page_content)

        return unique_documents

    def _dense_retrieval(self, vector_store: FAISS, top_k: int, lambda_mult: float):
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "lambda_mult": lambda_mult},
        )

    def _sparse_retrieval(self, doc_id: str, top_k: int):
        bm25_path = self._sparse_index_path(doc_id)
        with open(bm25_path, "rb") as f:
            retriever = pickle.load(f)
        retriever.k = top_k
        return retriever

    def _hybrid_retrieval(self, dense, sparse, alpha: float):
        return EnsembleRetriever(
            retrievers=[dense, sparse],
            weights=[alpha, 1 - alpha],
        )