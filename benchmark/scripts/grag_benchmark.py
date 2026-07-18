"""
ManuIndex Benchmark Evaluation Pipeline
========================================
Metrics: RAGAS (faithfulness, answer relevancy, context precision,
         context recall, answer correctness)
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
            db.add_document(case_document_text(case), chunk_size=config.chunk_size)
            files_added.add(doc_key)
    return db

def collect_results(db: ManuIndex, cases: list[dict]) -> list[dict]:
    """Return a flat list of per-question result dicts."""
    results = []
    for case in cases:
        for q in case["questions"]:
            retrieval_start = time.perf_counter()
            contexts = db.search(query=q["question"], top_k=config.top_k, lambda_mult=0.8, alpha=0.5)
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

        print("\n[1/3] Ingesting documents …")
        db = ingest_documents(cases, persist_dir)

        print("\n[2/3] Running queries …")
        results = collect_results(db, cases)
        summary = summarize_results(results)

        print("\n[3/3] RAGAS evaluation …")
        ragas_scores = run_ragas(results)

        elapsed = time.time() - t0

    display_report("ManuIndex Benchmark Report", ragas_scores, summary, elapsed)
    save_report(BENCHMARK_DIR / "reports" / "grag_report.json", config, results, ragas_scores, summary)


if __name__ == "__main__":
    main()
