"""
Query Rewrite RAG Benchmark Evaluation Pipeline
===============================================
Metrics: RAGAS (faithfulness, answer relevancy, context precision,
         context recall, answer correctness)
"""

from dataclasses import dataclass

from ._common import client, embeddings, require_llm_model, run_family_benchmark
from .src import QueryRewriteRAG


@dataclass(frozen=True)
class Config:
    top_k: int
    chunk_size: int
    emb_model: str
    llm_model: str


config = Config(
    top_k=3,
    chunk_size=150,
    emb_model="Qwen3-Embedding 0.6B (ONNX)",
    llm_model="Qwen3.5-2B",
)

query_rewrite_rag = QueryRewriteRAG(
    embeddings=embeddings,
    top_k=config.top_k,
    chunk_size=config.chunk_size,
    client=client,
    model_name=require_llm_model(),
)


def main():
    run_family_benchmark(
        report_title="Query Rewrite RAG Benchmark Report",
        run_label="query rewrite RAG",
        report_filename="query_rewrite_rag_report.json",
        config=config,
        retriever=query_rewrite_rag,
    )


if __name__ == "__main__":
    main()
