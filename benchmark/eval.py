"""
ManuIndex Benchmark Evaluation Pipeline
========================================
Metrics: RAGAS (faithfulness, answer relevancy, context precision,
         context recall)
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI
from manu_index import ManuIndex, ONNXEmbedder

from ragas import evaluate as ragas_evaluate
from ragas import metrics as ragas_metrics
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper


load_dotenv()

DATASET_DIR = Path(__file__).parent / "dataset"
CASES_FILE  = Path(__file__).parent / "evaluation_cases.json"

MODEL_DIR    = "onnx_models/qwen3_embedding_0dot6b/onnx/model.onnx"
TOKENIZER    = "onnx_models/qwen3_embedding_0dot6b"
MAX_LENGTH   = 1024
LLM_MODEL    = os.getenv("OPENAI_MODEL_NAME")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

embeddings = ONNXEmbedder(MODEL_DIR, TOKENIZER, MAX_LENGTH)


PARAMETERS = {
    "chunk_size": 130,
    "chunk_overlap": 0,
    "threshold": 0.7,
    "hybrid_top_k": [1, 1],
    "lambda_mult": 0.7,
    "alpha": 0.5,
}


def generate_answer(query: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(contexts)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "Answer the question using only the provided context. Be concise — no markdown, no headers, no bullet points."},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

def ingest_documents(cases: list[dict], persist_dir: str) -> ManuIndex:
    db = ManuIndex(embeddings=embeddings, client=client, model_name=LLM_MODEL, persist_directory=persist_dir)
    files_added: set[str] = set()
    for case in cases:
        fname = case["file"]
        if fname not in files_added:
            path = DATASET_DIR / fname
            print(f"  Ingesting {fname} …")
            db.add_document(
                str(path),
                chunk_size=PARAMETERS["chunk_size"],
                chunk_overlap=PARAMETERS["chunk_overlap"],
                threshold=PARAMETERS["threshold"],
            )
            files_added.add(fname)
    return db

def collect_results(db: ManuIndex, cases: list[dict]) -> list[dict]:
    """Return a flat list of per-question result dicts."""
    results = []
    for case in cases:
        for q in case["questions"]:
            contexts = db.search(
                q["question"],
                hybrid_top_k=PARAMETERS["hybrid_top_k"],
                lambda_mult=PARAMETERS["lambda_mult"],
                alpha=PARAMETERS["alpha"],
            )
            answer   = generate_answer(q["question"], contexts)
            results.append({
                "id":       q["id"],
                "question": q["question"],
                "answer":   answer,
                "contexts": contexts,
                "ground_truth": q["expected_answer"],
            })
            print(f"  [{q['id']}] done")
    return results

def _get_ragas_metric(*names: str):
    for name in names:
        metric = getattr(ragas_metrics, name, None)
        if metric is not None:
            return metric
    return None


def _configure_ragas_metric(metric, ragas_llm, ragas_embeddings):
    if hasattr(metric, "llm"):
        metric.llm = ragas_llm
    if hasattr(metric, "embeddings"):
        metric.embeddings = ragas_embeddings


def run_ragas(results: list[dict]) -> dict:
    ragas_llm = llm_factory(model=LLM_MODEL, provider="openai", client=client)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"],
        )
        for r in results
    ]
    dataset = EvaluationDataset(samples=samples)

    requested_metrics = [
        ("faithfulness", _get_ragas_metric("faithfulness")),
        ("answer_relevancy", _get_ragas_metric("answer_relevancy", "answer_relevance")),
        ("context_precision", _get_ragas_metric("context_precision")),
        ("context_recall", _get_ragas_metric("context_recall")),
    ]
    metrics = []
    labels_by_column = {}
    skipped_metrics = []
    for label, metric in requested_metrics:
        if metric is None:
            skipped_metrics.append(label)
            continue
        _configure_ragas_metric(metric, ragas_llm, ragas_embeddings)
        metrics.append(metric)
        labels_by_column[getattr(metric, "name", label)] = label

    if not metrics:
        raise RuntimeError("No requested RAGAS metrics are available in the installed ragas version.")

    scores = ragas_evaluate(dataset, metrics=metrics)
    score_df = scores.to_pandas()
    mean_scores = {}
    for column, label in labels_by_column.items():
        if column in score_df:
            mean_scores[label] = float(score_df[column].mean())  # pyright: ignore[reportArgumentType]
        else:
            skipped_metrics.append(label)

    if skipped_metrics:
        mean_scores["skipped_metrics"] = sorted(set(skipped_metrics))
    return mean_scores


def print_report(ragas_scores: dict, elapsed: float):
    sep = "─" * 52
    print(f"\n{'═' * 52}")
    print("  ManuIndex Benchmark Report")
    print(f"{'═' * 52}")

    print("\n  RAGAS")
    print(sep)
    for k, v in ragas_scores.items():
        print(f"  {k:<28} {v:.4f}" if isinstance(v, float) else f"  {k:<28} {v}")

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"{'═' * 52}\n")


def save_report(results: list[dict], ragas_scores: dict):
    report = {
        "ragas": ragas_scores,
        "per_question": results,
    }
    out = Path(__file__).parent / "report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  Report saved → {out}")



def main():
    cases = json.loads(CASES_FILE.read_text())["evaluation_cases"]

    # with tempfile.TemporaryDirectory(prefix="manu_eval_") as persist_dir:
    persist_dir = "manu_index_db"
    t0 = time.time()

    print("\n[1/3] Ingesting documents …")
    # db = ingest_documents(cases, persist_dir)
    db = ManuIndex(embeddings=embeddings, client=client, model_name=LLM_MODEL, persist_directory=persist_dir)  # Load existing index

    print("\n[2/3] Running queries …")
    results = collect_results(db, cases)

    print("\n[3/3] RAGAS evaluation …")
    ragas_scores = run_ragas(results)

    elapsed = time.time() - t0

    print_report(ragas_scores, elapsed)
    save_report(results, ragas_scores)


if __name__ == "__main__":
    main()