"""
ManuIndex Benchmark Evaluation Pipeline
========================================
Metrics:
  RAGAS — faithfulness, context precision, context recall
  HuggingFace evaluate — Token-level F1, Answer Recall
"""

import tempfile
import time

from manu_index import ManuIndex

from ._common import (
    BENCHMARK_DIR,
    case_document_text,
    client,
    config,
    embeddings,
    generate_answer,
    load_evaluation_cases,
    display_report,
    require_llm_model,
    run_hf_evaluate,
    run_ragas,
    save_report,
    summarize_results,
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
        doc_key = str(case["file"])
        if doc_key not in files_added:
            print(f"  Ingesting {doc_key} …")
            db.add_document(case_document_text(case), chunk_size=150)
            files_added.add(doc_key)
    return db

def collect_results(db: ManuIndex, cases: list[dict]) -> list[dict]:
    """Return a flat list of per-question result dicts."""
    results = []
    for case in cases:
        for q in case["questions"]:
            retrieval_start = time.perf_counter()
            contexts = db.search(query=q["question"], top_k=config.top_k, lambda_mult=0.9, alpha=0.5)
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

    with tempfile.TemporaryDirectory(prefix="manu_eval_") as persist_dir:
        t0 = time.time()

        print("\n[1/4] Ingesting documents …")
        db = ingest_documents(cases, persist_dir)

        print("\n[2/4] Running queries …")
        results = collect_results(db, cases)
        summary = summarize_results(results)

        print("\n[3/4] RAGAS evaluation …")
        ragas_scores = run_ragas(results)

        print("\n[4/4] HuggingFace evaluation …")
        hf_scores = run_hf_evaluate(results)

        elapsed = time.time() - t0

    display_report(
        "ManuIndex Benchmark Report",
        ragas_scores,
        summary,
        elapsed,
        hf_scores=hf_scores,
    )
    save_report(
        BENCHMARK_DIR / "reports" / "grag",
        config,
        results,
        ragas_scores,
        summary,
        hf_scores=hf_scores,
    )


if __name__ == "__main__":
    main()
