from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ._shared import split_documents


class LongRAG:
    def __init__(
        self,
        embeddings,
        top_k: int = 5,
        chunk_size: int = 4096,
    ):
        self.embeddings = embeddings
        self.top_k = top_k
        self.chunk_size = chunk_size

    def get_chunks(self, document: str) -> list[Document]:
        return split_documents(
            document,
            chunk_size=self.chunk_size,
            chunk_overlap=0,
        )

    def main(self, document: str, user_question: str) -> list[str]:
        chunks = self.get_chunks(document)
        if not chunks:
            return []

        store = FAISS.from_documents(chunks, embedding=self.embeddings)
        docs = store.similarity_search(user_question, k=self.top_k)
        return [doc.page_content for doc in docs][: self.top_k]
