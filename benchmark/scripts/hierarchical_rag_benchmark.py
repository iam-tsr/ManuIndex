"""
Hierarchical RAG Benchmark Evaluation Pipeline
==============================================
Metrics: RAGAS (faithfulness, answer relevancy, context precision,
         context recall, answer correctness)
"""

from ._common import config, embeddings, run_family_benchmark
from .src import HierarchicalRAG

hierarchical_rag = HierarchicalRAG(
    embeddings=embeddings,
    top_k=config.top_k,
    section_top_k=config.chunk_size,
    chunk_size=config.chunk_size,
)


def main():
    run_family_benchmark(
        report_title="Hierarchical RAG Benchmark Report",
        run_label="hierarchical RAG",
        report_filename="hierarchical_rag_report.json",
        retriever=hierarchical_rag,
    )


if __name__ == "__main__":
    main()
