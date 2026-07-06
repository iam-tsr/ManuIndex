"""
Graph RAG Benchmark Evaluation Pipeline
=======================================
Metrics: RAGAS (faithfulness, answer relevancy, context precision,
         context recall, answer correctness)
"""

from dataclasses import dataclass

from ._common import embeddings, run_family_benchmark
from .src import GraphRAG


@dataclass(frozen=True)
class Config:
    top_k: int
    chunk_size: int
    emb_model: str
    llm_model: str


config = Config(
    top_k=3,
    chunk_size=150,
    emb_model="Qwen3-Embedding 0.6B (ONNX)",
    llm_model="Qwen3.5-2B",
)

graph_rag = GraphRAG(
    embeddings=embeddings,
    top_k=config.top_k,
    chunk_size=config.chunk_size,
)


def main():
    run_family_benchmark(
        report_title="Graph RAG Benchmark Report",
        run_label="graph RAG",
        report_filename="graph_rag_report.json",
        config=config,
        retriever=graph_rag,
    )


if __name__ == "__main__":
    main()
