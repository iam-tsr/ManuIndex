from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

from ._shared import split_documents


class HybridRAG:
    def __init__(
        self,
        embeddings,
        top_k: int = 5,
        chunk_size: int = 150,
        alpha: float = 0.7,
    ):
        self.embeddings = embeddings
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.alpha = alpha

    def get_chunks(self, document: str):
        return split_documents(document, chunk_size=self.chunk_size, chunk_overlap=50)

    def build_retriever(self, chunks):
        vector_store = FAISS.from_documents(chunks, embedding=self.embeddings)
        dense = vector_store.as_retriever(
            search_kwargs={"k": max(self.top_k * 2, self.top_k)},
        )
        sparse = BM25Retriever.from_documents(chunks)
        sparse.k = max(self.top_k * 2, self.top_k)
        return EnsembleRetriever(retrievers=[dense, sparse], weights=[self.alpha, 1 - self.alpha])

    def main(self, document: str, user_question: str) -> list[str]:
        chunks = self.get_chunks(document)
        retriever = self.build_retriever(chunks)
        retrieved = retriever.invoke(user_question)
        texts = [doc.page_content for doc in retrieved]
        texts = texts[: max(self.top_k * 2, self.top_k)]

        return texts[: self.top_k]
