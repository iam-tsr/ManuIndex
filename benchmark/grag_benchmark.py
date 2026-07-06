"""
ManuIndex Benchmark Evaluation Pipeline
========================================
Metrics: RAGAS (faithfulness, answer relevancy, context precision,
         context recall, answer correctness)
"""

import os
import json
import time
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI
from manu_index import ManuIndex, ONNXEmbedder, ONNXReranker

from ragas import evaluate as ragas_evaluate
from ragas import metrics as ragas_metrics
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper


load_dotenv()

BENCHMARK_DIR = Path(__file__).parent
PROJECT_ROOT = BENCHMARK_DIR.parent
DATASET_DIR = BENCHMARK_DIR / "dataset" / "data"
CASES_FILE  = BENCHMARK_DIR / "dataset" / "evaluation_cases.json"

EMB_MODEL = 'onnx_models/qwen3_embedding_0.6b/onnx/model.onnx'
EMD_TOKENIZER = 'onnx_models/qwen3_embedding_0.6b'
RRNK_MODEL = 'onnx_models/bge_reranker_v2_m3/onnx/model_q4.onnx'
RRNK_TOKENIZER = 'onnx_models/bge_reranker_v2_m3'
MAX_LENGTH = 1024

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
LLM_MODEL = os.getenv("OPENAI_MODEL_NAME")

embeddings = ONNXEmbedder(EMB_MODEL, EMD_TOKENIZER, MAX_LENGTH, device="cpu")

@dataclass(frozen=True)
class Config:
    top_k: int
    chunk_size: int
    rrnk_model: str
    emb_model: str
    llm_model: str


config = Config(
    top_k=3,
    chunk_size=150,
    rrnk_model=None,
    emb_model="Qwen3-Embedding 0.6B (ONNX)",
    llm_model="Qwen3.5-2B",
)


def _require_llm_model() -> str:
    if not LLM_MODEL:
        raise RuntimeError("OPENAI_MODEL_NAME is required to run the benchmark.")
    return LLM_MODEL


def generate_answer(query: str, contexts: list[str]) -> tuple[str, int]:
    context_block = "\n\n".join(contexts)
    response = client.chat.completions.create(
        model=_require_llm_model(),
        messages=[
            {"role": "system", "content": "Answer the question using only the provided context. Give answer in simple terms — no markdown, no headers, no bullet points. If the context does not contain the answer, say 'I cannot answer based on the given context.'."},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
        ],
        temperature=0,
        max_tokens=2048,
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("LLM returned an empty response.")
    completion_tokens = getattr(response.usage, "completion_tokens", 0) if response.usage else 0
    return content.strip(), int(completion_tokens)

def resolve_case_file_path(file_name: str) -> Path:
    """Resolve a case document path from either old file names or new repo-relative paths."""
    path = Path(file_name)
    if path.is_absolute():
        return path

    repo_relative = PROJECT_ROOT / path
    if repo_relative.exists():
        return repo_relative

    benchmark_relative = BENCHMARK_DIR / path
    if benchmark_relative.exists():
        return benchmark_relative

    return DATASET_DIR / path


def load_evaluation_cases(cases_file: Path = CASES_FILE) -> list[dict]:
    """Load and normalize benchmark cases from the current or legacy JSON schema."""
    raw = json.loads(cases_file.read_text(encoding="utf-8"))
    raw_cases = raw.get("cases") if "cases" in raw else raw.get("evaluation_cases")
    if not isinstance(raw_cases, list):
        raise ValueError("evaluation cases JSON must contain a 'cases' or 'evaluation_cases' list")

    cases = []
    for case_index, case in enumerate(raw_cases, start=1):
        file_name = case.get("file_name") or case.get("file")
        if not file_name:
            raise ValueError(f"case {case_index} is missing 'file_name'")

        questions = []
        for question_index, question in enumerate(case.get("questions", []), start=1):
            query = question.get("question")
            expected_answer = question.get("answer") if "answer" in question else question.get("expected_answer")
            if not query or expected_answer is None:
                raise ValueError(f"case {case_index} question {question_index} is missing question/answer")
            questions.append({
                "id": question.get("id") or f"case_{case_index:03d}_q_{question_index:03d}",
                "question": query,
                "expected_answer": expected_answer,
            })

        cases.append({
            "file": file_name,
            "questions": questions,
        })
    return cases


def ingest_documents(cases: list[dict], persist_dir: str) -> ManuIndex:
    db = ManuIndex(embeddings=embeddings, client=client, model_name=LLM_MODEL, persist_directory=persist_dir)
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
    if config.rrnk_model:
        reranker = ONNXReranker(RRNK_MODEL, RRNK_TOKENIZER, MAX_LENGTH, device="cuda")
    else:
        reranker = config.rrnk_model  # None
    for case in cases:
        for q in case["questions"]:
            retrieval_start = time.perf_counter()
            contexts = db.search(query=q["question"], reranker=reranker, top_k=config.top_k)
            retrieval_time = time.perf_counter() - retrieval_start

            answer_start = time.perf_counter()
            answer, generation_tokens = generate_answer(q["question"], contexts)
            answer_time = time.perf_counter() - answer_start
            results.append({
                "id":       q["id"],
                "question": q["question"],
                "answer":   answer,
                "contexts": contexts,
                "ground_truth": q["expected_answer"],
                "retrieval_time_seconds": retrieval_time,
                "answer_time_seconds": answer_time,
                "generation_tokens": generation_tokens,
            })
            print(f"  [{q['id']}] done")
    return results


def summarize_results(results: list[dict]) -> dict:
    if not results:
        return {
            "average_retrieval_time_seconds": 0.0,
            "average_answer_time_seconds": 0.0,
            "average_generation_tokens": 0.0,
        }

    result_count = len(results)
    return {
        "average_retrieval_time_seconds": sum(r["retrieval_time_seconds"] for r in results) / result_count,
        "average_answer_time_seconds": sum(r["answer_time_seconds"] for r in results) / result_count,
        "average_generation_tokens": sum(r["generation_tokens"] for r in results) / result_count,
    }

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
    ragas_llm = llm_factory(model=LLM_MODEL, provider="openai", client=client, max_tokens=2048)
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
        ("Faithfulness", _get_ragas_metric("faithfulness")),
        ("Answer Relevancy", _get_ragas_metric("answer_relevancy", "answer_relevance")),
        ("Context Precision", _get_ragas_metric("context_precision")),
        ("Context Recall", _get_ragas_metric("context_recall")),
        ("F1 Score", _get_ragas_metric("answer_correctness")),
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


def print_report(ragas_scores: dict, summary: dict, elapsed: float):
    sep = "─" * 52
    print(f"\n{'═' * 52}")
    print("  ManuIndex Benchmark Report")
    print(f"{'═' * 52}")

    print("\n  RAGAS")
    print(sep)
    for k, v in ragas_scores.items():
        print(f"  {k:<28} {v:.4f}" if isinstance(v, float) else f"  {k:<28} {v}")

    print("\n  Runtime")
    print(sep)
    print(f"  {'Avg retrieval time (s)':<28} {summary['average_retrieval_time_seconds']:.4f}")
    print(f"  {'Avg answer time (s)':<28} {summary['average_answer_time_seconds']:.4f}")
    print(f"  {'Avg generation tokens':<28} {summary['average_generation_tokens']:.2f}")
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"{'═' * 52}\n")


def save_report(results: list[dict], ragas_scores: dict, summary: dict):
    report = {
        "config": {
            "top_k": config.top_k,
            "chunk_size": config.chunk_size,
            "reranker_model": config.rrnk_model,
            "embedding_model": config.emb_model,
            "llm_model": config.llm_model,
        },
        "ragas": ragas_scores,
        "runtime": summary,
        "per_question": results,
    }
    out = Path(__file__).parent / "grag_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  Report saved → {out}")



def main():
    cases = load_evaluation_cases()

    # with tempfile.TemporaryDirectory(prefix="manu_eval_") as persist_dir:
    persist_dir = "manu_index_db"
    t0 = time.time()

    print("\n[1/3] Ingesting documents …")
    # db = ingest_documents(cases, persist_dir)
    db = ManuIndex(embeddings=embeddings, client=client, model_name=LLM_MODEL, persist_directory=persist_dir)

    print("\n[2/3] Running queries …")
    results = collect_results(db, cases)
    summary = summarize_results(results)

    print("\n[3/3] RAGAS evaluation …")
    ragas_scores = run_ragas(results)

    elapsed = time.time() - t0

    print_report(ragas_scores, summary, elapsed)
    save_report(results, ragas_scores, summary)


if __name__ == "__main__":
    main()
