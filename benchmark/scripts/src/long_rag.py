from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ._shared import select_top_k, split_documents


class LongRAG:
    def __init__(
        self,
        embeddings,
        top_k: int = 5,
        unit_size: int = 4096,
        maxp_chunk_size: int = 512,
    ):
        self.embeddings = embeddings
        self.top_k = top_k
        self.unit_size = unit_size
        self.maxp_chunk_size = maxp_chunk_size

    def get_maxp_chunks(self, document: str) -> list[Document]:
        return split_documents(
            document,
            chunk_size=self.maxp_chunk_size,
            chunk_overlap=0,
        )

    def get_long_units(self, document: str) -> tuple[list[Document], list[Document]]:
        """Pack 512-token MaxP chunks into ~4K-token long retrieval units.

        For a short document this yields a single unit (the full document text).
        For longer text, consecutive MaxP chunks are packed until ``unit_size``.
        """
        maxp_chunks = self.get_maxp_chunks(document)
        if not maxp_chunks:
            return [], []

        units: list[Document] = []
        unit_to_chunk_indexes: list[list[int]] = []
        current_parts: list[str] = []
        current_indexes: list[int] = []
        current_size = 0

        for chunk_index, chunk in enumerate(maxp_chunks):
            chunk_size = len(chunk.page_content)
            if current_parts and current_size + chunk_size > self.unit_size:
                units.append(
                    Document(
                        page_content="\n".join(current_parts),
                        metadata={"unit_index": len(units)},
                    )
                )
                unit_to_chunk_indexes.append(current_indexes)
                current_parts = []
                current_indexes = []
                current_size = 0

            current_parts.append(chunk.page_content)
            current_indexes.append(chunk_index)
            current_size += chunk_size
            chunk.metadata["unit_index"] = len(units)
            chunk.metadata["chunk_index"] = chunk_index

        if current_parts:
            units.append(
                Document(
                    page_content="\n".join(current_parts),
                    metadata={"unit_index": len(units)},
                )
            )
            unit_to_chunk_indexes.append(current_indexes)

        # Attach mapping on units for MaxP aggregation.
        for unit, chunk_indexes in zip(units, unit_to_chunk_indexes):
            unit.metadata["chunk_indexes"] = chunk_indexes

        return units, maxp_chunks

    def main(self, document: str, user_question: str) -> list[str]:
        units, maxp_chunks = self.get_long_units(document)
        if not units:
            return []
        if len(units) == 1:
            return select_top_k([units[0].page_content], self.top_k)

        # Index MaxP subchunks; score all for MaxP aggregation over units.
        store = FAISS.from_documents(maxp_chunks, embedding=self.embeddings)
        scored = store.similarity_search_with_score(user_question, k=len(maxp_chunks))

        # Lower FAISS distance is better; unit score = best (min) subchunk distance.
        best_distance: dict[int, float] = {}
        for doc, distance in scored:
            unit_index = doc.metadata.get("unit_index")
            if unit_index is None:
                continue
            unit_index = int(unit_index)
            distance = float(distance)
            if unit_index not in best_distance or distance < best_distance[unit_index]:
                best_distance[unit_index] = distance

        ranked_indexes = sorted(
            best_distance.keys(),
            key=lambda index: best_distance[index],
        )
        unit_texts = [
            units[index].page_content
            for index in ranked_indexes
            if 0 <= index < len(units)
        ]
        return select_top_k(unit_texts, self.top_k)
