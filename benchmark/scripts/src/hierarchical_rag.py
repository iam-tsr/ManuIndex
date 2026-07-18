from langchain_community.vectorstores import FAISS

from ._shared import dedupe_texts, select_top_k, split_documents


class HierarchicalRAG:
    def __init__(
        self,
        embeddings,
        top_k: int = 3,
        section_top_k: int = 3,
        section_chunk_size: int = 150,
        chunk_size: int = 150,
    ):
        self.embeddings = embeddings
        self.top_k = top_k
        self.section_top_k = section_top_k
        self.section_chunk_size = section_chunk_size
        self.chunk_size = chunk_size

    def get_sections(self, document: str):
        sections = split_documents(document, chunk_size=self.section_chunk_size, chunk_overlap=0)
        for section_index, section in enumerate(sections):
            section.metadata["section_index"] = section_index
        return sections

    def get_section_chunks(self, sections):
        chunked_sections = []
        for section in sections:
            for chunk in split_documents(section.page_content, chunk_size=self.chunk_size, chunk_overlap=30):
                chunk.metadata["section_index"] = section.metadata.get("section_index")
                chunked_sections.append(chunk)
        return chunked_sections

    def main(self, document: str, user_question: str) -> list[str]:
        sections = self.get_sections(document)
        section_store = FAISS.from_documents(sections, embedding=self.embeddings)
        selected_sections = section_store.similarity_search(user_question, k=self.section_top_k)
        if not selected_sections:
            return []

        section_indexes = {section.metadata.get("section_index") for section in selected_sections}
        chunks = self.get_section_chunks(selected_sections)
        chunk_store = FAISS.from_documents(chunks, embedding=self.embeddings)
        retrieved_chunks = chunk_store.similarity_search(user_question, k=max(self.top_k * 2, self.top_k))

        texts = [chunk.page_content for chunk in retrieved_chunks if chunk.metadata.get("section_index") in section_indexes]
        return select_top_k(dedupe_texts(texts), self.top_k)
