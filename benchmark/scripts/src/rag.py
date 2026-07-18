from langchain_community.vectorstores import FAISS

from ._shared import select_top_k, split_documents


class NaiveRAG:
    def __init__(
        self,
        embeddings,
        top_k: int,
        chunk_size: int,
    ):
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.embeddings = embeddings

    def get_chunks(self, document: str):
        return split_documents(document, chunk_size=self.chunk_size, chunk_overlap=30)

    def get_vector_store(self, chunks):
        return FAISS.from_documents(chunks, embedding=self.embeddings)

    def get_relevant_context(self, vector_store, user_question: str):
        return vector_store.similarity_search(user_question, k=self.top_k)

    def main(self, document: str, user_question: str) -> list[str]:
        chunks = self.get_chunks(document)
        vector_store = self.get_vector_store(chunks)
        docs = self.get_relevant_context(vector_store, user_question)
        return select_top_k((doc.page_content for doc in docs), self.top_k)
