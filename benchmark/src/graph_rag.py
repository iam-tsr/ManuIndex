from ._shared import (
    dedupe_texts,
    dot_similarity,
    join_top_k,
    l2_normalize,
    split_markdown_sections,
    tokenize_keywords,
)


class GraphRAG:
    def __init__(
        self,
        embeddings,
        top_k: int = 3,
        chunk_size: int = 150,
        seed_k: int = 2,
        hop_k: int = 1,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ):
        self.embeddings = embeddings
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.seed_k = seed_k
        self.hop_k = hop_k
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

    def build_graph(self, document: str):
        nodes = split_markdown_sections(document, fallback_chunk_size=self.chunk_size)
        adjacency: dict[int, set[int]] = {}

        for index, node in enumerate(nodes):
            adjacency[index] = set()
            if index > 0:
                adjacency[index].add(index - 1)
            if index + 1 < len(nodes):
                adjacency[index].add(index + 1)

        return nodes, adjacency

    def score_nodes(self, user_question: str, nodes):
        query_embedding = l2_normalize(self.embeddings.embed_query(user_question))
        node_embeddings = [l2_normalize(vector) for vector in self.embeddings.embed_documents([node.page_content for node in nodes])]
        query_terms = set(tokenize_keywords(user_question))

        scores: list[tuple[int, float]] = []
        for index, (node, node_embedding) in enumerate(zip(nodes, node_embeddings)):
            node_terms = set(tokenize_keywords(node.page_content))
            keyword_score = 0.0
            if query_terms:
                keyword_score = len(query_terms & node_terms) / len(query_terms)
            semantic_score = dot_similarity(query_embedding, node_embedding)
            total_score = (self.semantic_weight * semantic_score) + (self.keyword_weight * keyword_score)
            scores.append((index, total_score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores

    def main(self, document: str, user_question: str) -> str:
        nodes, adjacency = self.build_graph(document)
        if not nodes:
            return ""

        scored_nodes = self.score_nodes(user_question, nodes)
        ranked_indices: list[int] = []
        for node_index, _score in scored_nodes[: self.seed_k]:
            ranked_indices.append(node_index)
            neighbours = sorted(adjacency.get(node_index, set()))
            ranked_indices.extend(neighbours[: self.hop_k])

        texts = [nodes[index].page_content for index in ranked_indices if 0 <= index < len(nodes)]
        return join_top_k(dedupe_texts(texts), self.top_k)
