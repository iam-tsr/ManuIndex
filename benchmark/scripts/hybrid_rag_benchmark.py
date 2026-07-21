"""
Flat Hybrid RAG Benchmark Evaluation Pipeline
=============================================
Metrics:
  RAGAS — faithfulness, context precision, context recall
  HuggingFace evaluate — Token-level F1, Answer Recall
"""

from ._common import config, embeddings, run_family_benchmark
from .src import HybridRAG

flat_hybrid_rag = HybridRAG(
    embeddings=embeddings,
    top_k=config.top_k,
    chunk_size=150,
)


def main():
    run_family_benchmark(
        report_title="Hybrid RAG Benchmark Report",
        run_label="hybrid RAG",
        report_filename="hybrid_rag",
        retriever=flat_hybrid_rag,
    )


if __name__ == "__main__":
    main()
