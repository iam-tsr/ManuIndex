"""
Flat Hybrid RAG Benchmark Evaluation Pipeline
=============================================
Metrics: RAGAS (faithfulness, answer relevancy, context precision,
         context recall, answer correctness)
"""

from dataclasses import dataclass

from ._common import embeddings, run_family_benchmark
from .src import FlatHybridRAG

@dataclass(frozen=True)
class Config:
    top_k: int
    chunk_size: int
    emb_model: str
    llm_model: str


config = Config(
    top_k=3,
    chunk_size=150,
    emb_model="BGE-M3 (ONNX)",
    llm_model="Qwen3.5-2B",
)

flat_hybrid_rag = FlatHybridRAG(
    embeddings=embeddings,
    top_k=config.top_k,
    chunk_size=config.chunk_size,
)


def main():
    run_family_benchmark(
        report_title="Flat Hybrid RAG Benchmark Report",
        run_label="flat hybrid RAG",
        report_filename="flat_hybrid_rag_report.json",
        config=config,
        retriever=flat_hybrid_rag,
    )


if __name__ == "__main__":
    main()