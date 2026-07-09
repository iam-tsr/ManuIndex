from langchain_community.vectorstores import FAISS

from ._shared import join_top_k, split_documents


class ParentChildRAG:
    def __init__(
        self,
        embeddings,
        top_k: int = 3,
        parent_chunk_size: int = 150,
        child_chunk_size: int = 150,
    ):
        self.embeddings = embeddings
        self.top_k = top_k
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size

    def get_parent_chunks(self, document: str):
        return split_documents(document, chunk_size=self.parent_chunk_size, chunk_overlap=0)

    def get_child_chunks(self, parent_chunks):
        child_chunks = []
        for parent_index, parent in enumerate(parent_chunks):
            for child_index, child in enumerate(
                split_documents(parent.page_content, chunk_size=self.child_chunk_size, chunk_overlap=40)
            ):
                child.metadata["parent_index"] = parent_index
                child.metadata["child_index"] = child_index
                child_chunks.append(child)
        return child_chunks

    def main(self, document: str, user_question: str) -> str:
        parent_chunks = self.get_parent_chunks(document)
        child_chunks = self.get_child_chunks(parent_chunks)
        if not child_chunks:
            return ""

        child_store = FAISS.from_documents(child_chunks, embedding=self.embeddings)
        retrieved_children = child_store.similarity_search(user_question, k=max(self.top_k * 3, self.top_k))

        parent_texts = []
        used_parent_indexes: set[int] = set()
        for child in retrieved_children:
            parent_index = child.metadata.get("parent_index")
            if parent_index is None or parent_index in used_parent_indexes:
                continue
            used_parent_indexes.add(parent_index)
            parent_texts.append(parent_chunks[int(parent_index)].page_content)

        return join_top_k(parent_texts, self.top_k)
