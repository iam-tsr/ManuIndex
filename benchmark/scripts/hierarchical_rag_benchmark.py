"""
Hierarchical (H-RAG) Parent–Child Benchmark Evaluation Pipeline
===============================================================
Metrics:
  RAGAS — faithfulness, context precision, context recall
  HuggingFace evaluate — Token-level F1, Answer Recall
"""

from ._common import config, embeddings, run_family_benchmark
from .src import HierarchicalRAG

hierarchical_rag = HierarchicalRAG(
    embeddings=embeddings,
    top_k=config.top_k,
    parent_chunk_size=512,
    child_window_sentences=2,
    child_stride_sentences=1,
)


def main():
    run_family_benchmark(
        report_title="Hierarchical H-RAG Parent–Child Benchmark Report",
        run_label="hierarchical H-RAG",
        report_filename="hierarchical_rag",
        retriever=hierarchical_rag,
    )


if __name__ == "__main__":
    main()
