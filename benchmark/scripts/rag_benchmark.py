"""
Naive RAG Benchmark Evaluation Pipeline
=======================================
Metrics:
  RAGAS — faithfulness, context precision, context recall
  HuggingFace evaluate — Token-level F1, Answer Recall
"""

from ._common import config, embeddings, run_family_benchmark
from .src import NaiveRAG

naive_rag = NaiveRAG(
    embeddings=embeddings,
    top_k=config.top_k,
    chunk_size=150
)


def main():
    run_family_benchmark(
        report_title="RAG Benchmark Report",
        run_label="naive RAG",
        report_filename="rag",
        retriever=naive_rag,
    )


if __name__ == "__main__":
    main()
