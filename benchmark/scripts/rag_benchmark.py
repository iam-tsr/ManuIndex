"""
Naive RAG Benchmark Evaluation Pipeline
=======================================
Metrics: RAGAS (faithfulness, answer relevancy, context precision,
         context recall, answer correctness)
"""

from ._common import config, embeddings, run_family_benchmark
from .src import NaiveRAG

naive_rag = NaiveRAG(embeddings=embeddings, top_k=config.top_k, chunk_size=config.chunk_size)


def main():
    run_family_benchmark(
        report_title="RAG Benchmark Report",
        run_label="naive RAG",
        report_filename="rag_report.json",
        retriever=naive_rag,
    )


if __name__ == "__main__":
    main()
