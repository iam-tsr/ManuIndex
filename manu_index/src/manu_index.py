import json
import pickle
import shutil
import uuid
import os
from typing import List, Optional, Any

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from manu_index.src.summary import DocumentSummary

META_FILENAME = "_meta.json"


class ManuIndex:
    """Hybrid retrieval index that combines dense (FAISS) and lexical (BM25) search.

    Supports document ingestion with semantic chunking, persistent storage,
    and flexible retrieval strategies (dense, sparse, or hybrid).
    """

    def __init__(
        self,
        embeddings: Any,
        client: Any,
        persist_directory: str = "manu_index_data",
    ):
        """
        Args:
            embeddings: Embedding model used to encode text into vectors.
            client: LLM client used for generating document summaries.
            persist_directory: Directory where the index and metadata are stored.
        """
        self.embeddings = embeddings
        self.client = client
        self.persist_directory = persist_directory

    def add_document(
        self,
        documents: str | bytes,
        chunk_size: int = 100,
        chunk_overlap: int = 0,
        threshold: float = 0.7,
    ) -> FAISS:
        """Ingest a document into the index.

        Reads the document, splits it into semantic chunks, builds both a FAISS
        vector store and a BM25 lexical store, and persists them to disk.

        Args:
            documents: Raw document content — a UTF-8 string, bytes, or a path
                       to a Markdown file.
            chunk_size: Maximum character size of each initial (pre-semantic) chunk.
            chunk_overlap: Character overlap between adjacent initial chunks.
            threshold: Cosine-similarity threshold for the semantic chunking step.
                       Adjacent chunks below this value trigger a new chunk boundary.

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

        chunks = self._add(
            documents,
            doc_id=doc_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            threshold=threshold,
        )

        self._lexical_store(doc_id=doc_id, documents=chunks, top_k=3) # top_k for BM25 retrieval is cannot be changed.

        vector_store = FAISS.from_documents(documents=chunks, embedding=self.embeddings)
        vector_store.save_local(self.persist_directory, index_name=doc_id)
        return vector_store

    def search(
        self,
        query: str,
        top_k: int = 3,
        lambda_mult: float = 0.5,
        alpha: float = 0.5,
        search_strategy: Optional[str] = None,
    ) -> List[str]:
        """Retrieve relevant passages for a query.

        Args:
            query: Natural-language search query.
            top_k: Number of passages to return.
            lambda_mult: MMR diversity parameter (0 = max diversity, 1 = max relevance).
            alpha: Weight given to dense retrieval in hybrid mode (BM25 gets 1 - alpha).
            search_strategy: ``"dense"`` for vector-only, ``"sparse"`` for BM25-only,
                             or ``None`` (default) for hybrid.

        Returns:
            List of matching passage strings.
        """
        query_embedding = self.embeddings.embed_query(query)
        doc_id = self._find_collection(query_embedding)

        vector_store = FAISS.load_local(
            folder_path=self.persist_directory,
            embeddings=self.embeddings,
            index_name=doc_id,
            allow_dangerous_deserialization=True,
        )

        if search_strategy == "dense":
            retriever = self._dense_retrieval(vector_store, top_k, lambda_mult)
            return [doc.page_content for doc in retriever.invoke(query)]

        if search_strategy == "sparse":
            retriever = self._sparse_retrieval(doc_id)
            return [doc.page_content for doc in retriever.invoke(query)]

        # Default: hybrid
        retriever = self._hybrid_retrieval(
            dense=self._dense_retrieval(vector_store, top_k, lambda_mult),
            sparse=self._sparse_retrieval(doc_id),
            alpha=alpha,
        )
        return [doc.page_content for doc in retriever.invoke(query)]

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
        for filename in os.listdir(self.persist_directory):
            is_index_file = filename.endswith(".index") or filename.endswith(".faiss")
            if filename.startswith(doc_id) and is_index_file:
                os.remove(os.path.join(self.persist_directory, filename))

        # Remove BM25 pickle
        bm25_path = os.path.join(self.persist_directory, f"{doc_id}_tsr.pkl")
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

    @staticmethod
    def _normalize(v) -> list:
        arr = np.array(v, dtype=np.float32)
        return (arr / np.linalg.norm(arr)).tolist()

    def _add(self, document: str, doc_id: str, **kwargs) -> List[Document]:
        self._create_summary(document, doc_id=doc_id)
        return self._create_semantic_chunks(document, **kwargs)

    def _create_summary(self, document: str, doc_id: str) -> None:
        """Summarize a document and append the result to the metadata file."""
        summary = DocumentSummary(document=document, client=self.client).summarize()
        embedding = self._normalize(self.embeddings.embed_query(summary))

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

    def _find_collection(self, query_embedding) -> str:
        """Return the doc_id whose summary embedding is closest to the query."""
        data = self._load_meta()
        ids = [e["doc_id"] for e in data]
        matrix = np.array([e["values"] for e in data], dtype=np.float32)
        query = self._normalize(query_embedding)
        scores = np.dot(matrix, query)
        return ids[int(np.argmax(scores))]

    def _lexical_store(self, doc_id: str, documents: List[Document], top_k: int) -> None:
        """Build a BM25 retriever from documents and persist it to disk."""
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = top_k

        bm25_path = os.path.join(self.persist_directory, f"{doc_id}_tsr.pkl")
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_retriever, f)

    def _create_semantic_chunks(
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

    def _dense_retrieval(self, vector_store: FAISS, top_k: int, lambda_mult: float):
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "lambda_mult": lambda_mult},
        )

    def _sparse_retrieval(self, doc_id: str):
        bm25_path = os.path.join(self.persist_directory, f"{doc_id}_tsr.pkl")
        with open(bm25_path, "rb") as f:
            return pickle.load(f)

    def _hybrid_retrieval(self, dense, sparse, alpha: float):
        return EnsembleRetriever(
            retrievers=[dense, sparse],
            weights=[alpha, 1 - alpha],
        )