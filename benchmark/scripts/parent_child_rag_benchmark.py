"""
Parent-Child RAG Benchmark Evaluation Pipeline
==============================================
Metrics: RAGAS (faithfulness, answer relevancy, context precision,
         context recall, answer correctness)
"""

from ._common import config, embeddings, run_family_benchmark
from .src import ParentChildRAG

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
        retriever=parent_child_rag,
    )


if __name__ == "__main__":
    main()
