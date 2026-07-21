"""
Query Rewrite RAG Benchmark Evaluation Pipeline
===============================================
Metrics:
  RAGAS — faithfulness, context precision, context recall
  HuggingFace evaluate — Token-level F1, Answer Recall
"""

from ._common import client, config, embeddings, require_llm_model, run_family_benchmark
from .src import QueryRewriteRAG

query_rewrite_rag = QueryRewriteRAG(
    embeddings=embeddings,
    top_k=config.top_k,
    chunk_size=150,
    client=client,
    model_name=require_llm_model(),
)


def main():
    run_family_benchmark(
        report_title="Query Rewrite RAG Benchmark Report",
        run_label="query rewrite RAG",
        report_filename="query_rewrite_rag",
        retriever=query_rewrite_rag,
    )


if __name__ == "__main__":
    main()
