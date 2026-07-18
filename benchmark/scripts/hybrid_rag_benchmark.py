"""
Flat Hybrid RAG Benchmark Evaluation Pipeline
=============================================
Metrics: RAGAS (faithfulness, answer relevancy, context precision,
         context recall, answer correctness)
"""

from ._common import config, embeddings, run_family_benchmark
from .src import FlatHybridRAG

flat_hybrid_rag = FlatHybridRAG(
    embeddings=embeddings,
    top_k=config.top_k,
    chunk_size=config.chunk_size,
)


def main():
    run_family_benchmark(
        report_title="Hybrid RAG Benchmark Report",
        run_label="hybrid RAG",
        report_filename="hybrid_rag_report.json",
        retriever=flat_hybrid_rag,
    )


if __name__ == "__main__":
    main()
