"""
LongRAG Benchmark Evaluation Pipeline
=====================================
Metrics:
  RAGAS — faithfulness, context precision, context recall
  HuggingFace evaluate — Token-level F1, Answer Recall
"""

from ._common import config, embeddings, run_family_benchmark
from .src import LongRAG

long_rag = LongRAG(
    embeddings=embeddings,
    top_k=config.top_k,
    chunk_size=4012,
)


def main():
    run_family_benchmark(
        report_title="LongRAG Benchmark Report",
        run_label="LongRAG",
        report_filename="long_rag",
        retriever=long_rag,
    )


if __name__ == "__main__":
    main()
