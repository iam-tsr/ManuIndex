import json
import pickle
import shutil
import uuid
import os
from typing import (
    List, 
    Optional, 
    Any
)
import logging

from sklearn.metrics.pairwise import cosine_similarity
from summary import DocumentSummary
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

logger = logging.getLogger(__name__)

META_FILENAME = "_meta.json"

class ManuIndex:
    def __init__(
            self,
            embeddings: Any,
            client: Any,
            persist_directory: str = "manu_index",
    ):
        """Initialize the ManuIndex with embeddings and a directory to persist the index.
        
        Args:
            embeddings (Any): The embedding model to use for creating vector representations of documents.
            persist_directory (str): The directory where the index and metadata will be stored.    
        """
        self.embeddings = embeddings
        self.client = client
        self.persist_directory = persist_directory

    def _add(self, document, doc_id, **kwargs):
        self._create_summary(document, doc_id=doc_id)
        return self._create_semantic_chunks(document, **kwargs)
    
    def add_document(
            self, 
            documents: str | bytes, 
            chunk_size: int = 100, 
            chunk_overlap: int = 0,
            threshold: float = 0.7
    ) -> FAISS:
        """Add documents to the index, creating a new FAISS vector store for each document.
        
        Args:
            documents (str | bytes): The document(s) to be added to the index. Can be a string (file path) or bytes.
            chunk_size (int): The size of each chunk for splitting the documents.
            chunk_overlap (int): The overlap size between chunks.
            threshold (float): The similarity threshold for semantic chunking.
        Returns:
            FAISS: The FAISS vector store containing the added documents.
        """
        doc_id = str(uuid.uuid4().hex[:6])
        # Read pdf files as bytes and convert to string if necessary
        if not isinstance(documents, (str, bytes)):
            raise ValueError("Documents must be a string, or bytes.")
        if isinstance(documents, bytes):
            documents = documents.decode("utf-8", errors="ignore")

        # Read markdown files as string if necessary
        if isinstance(documents, str) and documents.endswith(".md"):
            with open(documents, "r", encoding="utf-8") as f:
                documents = f.read()

        documents = self._add(
            documents, 
            doc_id=doc_id,
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            threshold=threshold
        )

        logger.info(f"Creating lexical store for document: {doc_id}")
        lexical_store = self._lexical_store(
            doc_id=doc_id, 
            documents=documents, 
            top_k=3     # Fixed top_k for lexical store, cannot be parameterized
        )
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
            # distance_strategy=DistanceStrategy.COSINE,
        )
        vector_store.save_local(self.persist_directory, index_name=doc_id)
        return vector_store
    
    def _lexical_store(
            self, 
            doc_id: str, 
            documents: List[Document], 
            top_k: int
    ):
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = top_k

        logger.info(f"Creating lexical store for document: {doc_id}")
        bm25_path = os.path.join(self.persist_directory, f"{doc_id}_tsr.pkl")
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_retriever, f)

    def _create_summary(self, document, doc_id: str):
        summary = DocumentSummary(document=document, client=self.client).summarize()
        embedding = self.embeddings.encode(summary)

        data = {"doc_id": doc_id, "values": embedding.tolist(), "summary": summary}
        logger.info(f"Created summary for document: {doc_id}")
        os.makedirs(self.persist_directory, exist_ok=True)
        meta_path = os.path.join(self.persist_directory, META_FILENAME)
        existing = []
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []

        existing.append(data)
        with open(meta_path, "w") as f:
            f.write("[\n")
            for i, entry in enumerate(existing):
                values_inline = json.dumps(entry["values"], separators=(", ", ": "))
                f.write(f'  {{\n    "doc_id": {json.dumps(entry["doc_id"])},\n    "values": {values_inline},\n    "summary": {json.dumps(entry["summary"])}\n  }}')
                f.write(",\n" if i < len(existing) - 1 else "\n")
            f.write("]\n")

    def _find_doc(self, query: List[float]) -> str:
        meta_path = os.path.join(self.persist_directory, META_FILENAME)
        if not os.path.exists(meta_path):
            raise FileNotFoundError("Metadata file not found.")

        with open(meta_path, "r") as f:
            try:
                data = json.load(f)
                logger.info(f"Finding document for query. Total documents in index: {len(data)}")
                data.sort(key=lambda x: cosine_similarity(query, [x["values"][0]])[0][0], reverse=True)
                # show similarity scores for debugging
                for entry in data:
                    sim_score = cosine_similarity(query, [entry["values"][0]])[0][0]
                    logger.info(f"Document ID: {entry['doc_id']}, Similarity Score: {sim_score}")
                return data[0]["doc_id"]  # Return the doc_id of the most relevant document
            except json.JSONDecodeError:
                raise ValueError("Metadata file is corrupted.")
    
    def _create_semantic_chunks(
            self, 
            document: str, 
            chunk_size: int, 
            chunk_overlap: int,
            threshold: float
    ) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=[" "]
        )
        
        split_document = text_splitter.split_text(document)
        def semantic_chunking(sentences: list[str], threshold: float) -> list[str]:
            embeddings = self.embeddings.encode(sentences)
            threshold = threshold
            chunks = []
            current_chunk=[sentences[0]]

            for i in range(1, len(sentences)):
                sim = cosine_similarity(
                    [embeddings[i - 1]],
                    [embeddings[i]]
                )[0][0]

                if sim>=threshold:
                    current_chunk.append(sentences[i])
                else:
                    chunks.append(" ".join(current_chunk))
                    current_chunk=[sentences[i]]

            chunks.append(" ".join(current_chunk))
            return chunks
        
        chunked_document = semantic_chunking(split_document, threshold=threshold)
        return [Document(page_content=chunk) for chunk in chunked_document]
    
    def info(self):
        """Retrieve metadata information about the indexed documents.

        Returns:
            List[dict]: A list of dictionaries containing metadata for each document, including 'doc_id' and 'summary'.
        """
        meta_path = os.path.join(self.persist_directory, META_FILENAME)
        if not os.path.exists(meta_path):
            raise FileNotFoundError("Metadata file not found.")

        with open(meta_path, "r") as f:
            try:
                data = json.load(f)
                # return only doc_id and summary for brevity
                return [{"doc_id": entry["doc_id"], "summary": entry["summary"]} for entry in data]
            except json.JSONDecodeError:
                raise ValueError("Metadata file is corrupted.")
            
    def delete(self, doc_id: str):
        """Delete a document and its associated vector store index.

        Args:
            doc_id (str): The unique identifier of the document to delete.
        """
        for filename in os.listdir(self.persist_directory):
            if filename.startswith(doc_id) and (filename.endswith(".index") or filename.endswith(".faiss")):
                os.remove(os.path.join(self.persist_directory, filename))

        meta_path = os.path.join(self.persist_directory, META_FILENAME)
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
            data = [entry for entry in data if entry["doc_id"] != doc_id]
            with open(meta_path, "w") as f:
                json.dump(data, f, indent=2)

    def clear(self):
        """
        Clear the entire index, including all documents and their associated vector store indices.
        """
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        logger.info(f"Cleared the entire index at {self.persist_directory}.")

    def search(
            self, 
            query: str, 
            top_k: int = 3, 
            lambda_mult: float = 0.5,
            alpha: float = 0.5, 
            search_strategy: Optional[str] = None
    ) -> List[Document]:
        """Search for relevant documents based on the query. Default search strategy is hybrid, which combines dense and lexical retrieval.
        
        Args:
            query (str): The search query.
            top_k (int): The number of `documents` to retrieve.
            lambda_mult (float): The lambda multiplier for MMR retrieval.
            alpha (float): The weight for dense retrieval in the ensemble (used in hybrid search).
            search_strategy (str, optional): The search strategy to use. Options are "dense", "lexical", or None (for hybrid).

        Returns:
            List[Document]: A list of retrieved documents based on the search strategy.
        """
        query_embedding = self.embeddings.encode(query)
        doc_id = self._find_doc(query_embedding)

        vector_store = FAISS.load_local(
            folder_path=self.persist_directory,
            embeddings=self.embeddings,
            index_name=doc_id,
            allow_dangerous_deserialization=True
        )

        if search_strategy == "dense":
            dense_retriever = self._dense_retrieval(vector_store, top_k, lambda_mult)
            dense_retriever_output = dense_retriever.invoke(query)
            return [doc.page_content for doc in dense_retriever_output]
        elif search_strategy == "sparse":
            return self._sparse_retrieval(doc_id)
        else:
            hybrid_retriever = self._hybrid_retrieval(
                self._dense_retrieval(vector_store, top_k, lambda_mult),
                self._sparse_retrieval(doc_id),
                alpha
            )
            hybrid_retriever_output = hybrid_retriever.invoke(query)
            return [doc.page_content for doc in hybrid_retriever_output]
    
    def _dense_retrieval(
            self, 
            vector_store: FAISS, 
            top_k: int,
            lambda_mult: float

    ):
        return vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": top_k,
                    "lambda_mult": lambda_mult,
                },
            )
    
    def _sparse_retrieval(
            self, 
            doc_id: str, 
    ):
        bm25_path = os.path.join(self.persist_directory, f"{doc_id}_tsr.pkl")
        with open(bm25_path, "rb") as f:
            bm25_retriever = pickle.load(f)
        return bm25_retriever
    
    def _hybrid_retrieval(
            self, 
            dense_retriever,
            sparse_retriever,
            alpha: float
    ):
        return EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=[alpha, 1 - alpha]
        )