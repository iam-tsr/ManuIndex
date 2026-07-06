"""
Parent-Child RAG Benchmark Evaluation Pipeline
==============================================
Metrics: RAGAS (faithfulness, answer relevancy, context precision,
         context recall, answer correctness)
"""

from dataclasses import dataclass

from ._common import embeddings, run_family_benchmark
from .src import ParentChildRAG


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

parent_child_rag = ParentChildRAG(
    embeddings=embeddings,
    top_k=config.top_k,
    parent_chunk_size=config.chunk_size,
    child_chunk_size=config.chunk_size,
)


def main():
    run_family_benchmark(
        report_title="Parent-Child RAG Benchmark Report",
        run_label="parent-child RAG",
        report_filename="parent_child_rag_report.json",
        config=config,
        retriever=parent_child_rag,
    )


if __name__ == "__main__":
    main()
