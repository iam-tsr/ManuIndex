from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

from ._shared import join_top_k, split_documents


class FlatHybridRAG:
    def __init__(
        self,
        embeddings,
        top_k: int = 3,
        chunk_size: int = 150,
        alpha: float = 0.5,
        lambda_mult: float = 0.8,
        reranker=None,
    ):
        self.embeddings = embeddings
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.alpha = alpha
        self.lambda_mult = lambda_mult
        self.reranker = reranker

    def get_chunks(self, document: str):
        return split_documents(document, chunk_size=self.chunk_size)

    def build_retriever(self, chunks):
        vector_store = FAISS.from_documents(chunks, embedding=self.embeddings)
        dense = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": max(self.top_k * 2, self.top_k), "lambda_mult": self.lambda_mult},
        )
        sparse = BM25Retriever.from_documents(chunks)
        sparse.k = max(self.top_k * 2, self.top_k)
        return EnsembleRetriever(retrievers=[dense, sparse], weights=[self.alpha, 1 - self.alpha])

    def main(self, document: str, user_question: str) -> str:
        chunks = self.get_chunks(document)
        retriever = self.build_retriever(chunks)
        retrieved = retriever.invoke(user_question)
        texts = [doc.page_content for doc in retrieved]
        texts = texts[: max(self.top_k * 2, self.top_k)]

        if self.reranker is not None and texts:
            ranked = self.reranker.rerank(query=user_question, documents=texts)
            texts = [text for text, _score in ranked]

        return join_top_k(texts, self.top_k)

