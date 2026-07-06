"""
ManuIndex Benchmark Evaluation Pipeline
========================================
Metrics: RAGAS (faithfulness, answer relevancy, context precision,
         context recall, answer correctness)
"""

import time
from dataclasses import dataclass

from manu_index import ManuIndex, ONNXReranker

from ._common import (
    BENCHMARK_DIR,
    MAX_LENGTH,
    client,
    embeddings,
    generate_answer,
    load_evaluation_cases,
    print_report,
    require_llm_model,
    resolve_case_file_path,
    run_ragas,
    save_report,
    summarize_results,
)


RRNK_MODEL = "onnx_models/bge_reranker_v2_m3/onnx/model_q4.onnx"
RRNK_TOKENIZER = "onnx_models/bge_reranker_v2_m3"

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

def ingest_documents(cases: list[dict], persist_dir: str) -> ManuIndex:
    db = ManuIndex(
        embeddings=embeddings,
        client=client,
        model_name=require_llm_model(),
        persist_directory=persist_dir,
    )
    files_added: set[str] = set()
    for case in cases:
        fname = case["file"]
        path = resolve_case_file_path(fname)
        path_key = str(path.resolve())
        if path_key not in files_added:
            print(f"  Ingesting {path} …")
            db.add_document(str(path), chunk_size=config.chunk_size)
            files_added.add(path_key)
    return db

def collect_results(db: ManuIndex, cases: list[dict]) -> list[dict]:
    """Return a flat list of per-question result dicts."""
    results = []
    if config.reranker_model:
        reranker = ONNXReranker(RRNK_MODEL, RRNK_TOKENIZER, MAX_LENGTH, device="cuda")
    else:
        reranker = config.reranker_model
    for case in cases:
        for q in case["questions"]:
            retrieval_start = time.perf_counter()
            contexts = db.search(query=q["question"], reranker=reranker, top_k=config.top_k)
            retrieval_time = time.perf_counter() - retrieval_start

            answer_start = time.perf_counter()
            answer, input_tokens, output_tokens, total_tokens = generate_answer(q["question"], contexts)
            answer_time = time.perf_counter() - answer_start
            results.append({
                "id":       q["id"],
                "question": q["question"],
                "answer":   answer,
                "contexts": contexts,
                "ground_truth": q["expected_answer"],
                "retrieval_time_seconds": retrieval_time,
                "answer_time_seconds": answer_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            })
            print(f"  [{q['id']}] done")
    return results


def main():
    cases = load_evaluation_cases()

    # with tempfile.TemporaryDirectory(prefix="manu_eval_") as persist_dir:
    persist_dir = "manu_index_db"
    t0 = time.time()

    print("\n[1/3] Ingesting documents …")
    # db = ingest_documents(cases, persist_dir)
    db = ManuIndex(
        embeddings=embeddings,
        client=client,
        model_name=require_llm_model(),
        persist_directory=persist_dir,
    )

    print("\n[2/3] Running queries …")
    results = collect_results(db, cases)
    summary = summarize_results(results)

    print("\n[3/3] RAGAS evaluation …")
    ragas_scores = run_ragas(results)

    elapsed = time.time() - t0

    print_report("ManuIndex Benchmark Report", ragas_scores, summary, elapsed)
    save_report(BENCHMARK_DIR / "reports" / "grag_report.json", config, results, ragas_scores, summary)


if __name__ == "__main__":
    main()
