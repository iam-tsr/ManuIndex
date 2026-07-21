import re

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ._shared import select_top_k, split_documents


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [part.strip() for part in parts if part and part.strip()]


class HierarchicalRAG:
    def __init__(
        self,
        embeddings,
        top_k: int = 5,
        parent_chunk_size: int = 512,
        child_window_sentences: int = 3,
        child_stride_sentences: int = 3,
    ):
        self.embeddings = embeddings
        self.top_k = top_k
        self.parent_chunk_size = parent_chunk_size
        self.child_window_sentences = child_window_sentences
        self.child_stride_sentences = child_stride_sentences

    def get_parents(self, document: str) -> list[Document]:
        return split_documents(document, chunk_size=self.parent_chunk_size, chunk_overlap=0)

    def get_children(self, parents: list[Document]) -> list[Document]:
        window = self.child_window_sentences
        stride = self.child_stride_sentences
        children: list[Document] = []

        for parent_index, parent in enumerate(parents):
            sentences = split_sentences(parent.page_content)
            if not sentences:
                continue

            if len(sentences) <= window:
                children.append(
                    Document(
                        page_content=" ".join(sentences),
                        metadata={"parent_index": parent_index},
                    )
                )
                continue

            for start in range(0, len(sentences) - window + 1, stride):
                children.append(
                    Document(
                        page_content=" ".join(sentences[start : start + window]),
                        metadata={"parent_index": parent_index},
                    )
                )

        return children

    def main(self, document: str, user_question: str) -> list[str]:
        parents = self.get_parents(document)
        if not parents:
            return []

        children = self.get_children(parents)
        if not children:
            return select_top_k((parent.page_content for parent in parents), self.top_k)

        candidate_k = max(self.top_k * 3, self.top_k)
        child_store = FAISS.from_documents(children, embedding=self.embeddings)

        dense = child_store.as_retriever(search_kwargs={"k": candidate_k})
        dense_docs = dense.invoke(user_question)
        dense_texts = {doc.page_content for doc in dense_docs}

        # Embedding rescoring over the dense shortlist (lower FAISS distance is better).
        scored = child_store.similarity_search_with_score(user_question, k=min(candidate_k, len(children)))
        rescored = [(doc, dist) for doc, dist in scored if doc.page_content in dense_texts]
        if not rescored:
            rescored = scored

        # Max-score assignment to parents (keep best / lowest distance per parent).
        parent_best: dict[int, float] = {}
        for doc, dist in rescored:
            parent_index = doc.metadata.get("parent_index")
            if parent_index is None:
                continue
            parent_index = int(parent_index)
            if parent_index not in parent_best or dist < parent_best[parent_index]:
                parent_best[parent_index] = float(dist)

        ranked_indexes = sorted(parent_best.keys(), key=lambda index: parent_best[index])
        parent_texts = [
            parents[index].page_content
            for index in ranked_indexes
            if 0 <= index < len(parents)
        ]
        return select_top_k(parent_texts, self.top_k)
