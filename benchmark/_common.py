import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from ragas import evaluate as ragas_evaluate
from ragas import metrics as ragas_metrics
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory

from manu_index import ONNXEmbedder


load_dotenv()

BENCHMARK_DIR = Path(__file__).parent
PROJECT_ROOT = BENCHMARK_DIR.parent
DATASET_DIR = BENCHMARK_DIR / "dataset" / "data"
CASES_FILE = BENCHMARK_DIR / "dataset" / "evaluation_cases.json"

EMB_MODEL = "onnx_models/qwen3_embedding_0.6b/onnx/model.onnx"
EMD_TOKENIZER = "onnx_models/qwen3_embedding_0.6b"
MAX_LENGTH = 1024

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
LLM_MODEL = os.getenv("OPENAI_MODEL_NAME")

embeddings = ONNXEmbedder(EMB_MODEL, EMD_TOKENIZER, MAX_LENGTH, batch_size=1, device="cuda")


def require_llm_model() -> str:
    if not LLM_MODEL:
        raise RuntimeError("OPENAI_MODEL_NAME is required to run the benchmark.")
    return LLM_MODEL


def generate_answer(query: str, contexts: list[str]) -> tuple[str, int, int, int]:
    context_block = "\n\n".join(contexts)
    response = client.chat.completions.create(
        model=require_llm_model(),
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
    prompt_tokens = getattr(response.usage, "prompt_tokens", 0) if response.usage else 0
    completion_tokens = getattr(response.usage, "completion_tokens", 0) if response.usage else 0
    total_tokens = getattr(response.usage, "total_tokens", prompt_tokens + completion_tokens) if response.usage else 0
    return content.strip(), int(prompt_tokens), int(completion_tokens), int(total_tokens)


def resolve_case_file_path(file_name: str) -> Path:
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
            questions.append(
                {
                    "id": question.get("id") or f"case_{case_index:03d}_q_{question_index:03d}",
                    "question": query,
                    "expected_answer": expected_answer,
                }
            )

        cases.append({"file": file_name, "questions": questions})
    return cases


def collect_results(cases: list[dict], retriever) -> list[dict]:
    results = []
    documents_by_path: dict[str, str] = {}

    for case in cases:
        path = resolve_case_file_path(case["file"])
        path_key = str(path.resolve())
        if path_key not in documents_by_path:
            print(f"  Loading {path} …")
            documents_by_path[path_key] = path.read_text(encoding="utf-8")

        document = documents_by_path[path_key]
        for q in case["questions"]:
            retrieval_start = time.perf_counter()
            context = retriever.main(document, q["question"])
            retrieval_time = time.perf_counter() - retrieval_start
            contexts = [context] if context else []
            answer_start = time.perf_counter()
            answer, input_tokens, output_tokens, total_tokens = generate_answer(q["question"], contexts)
            answer_time = time.perf_counter() - answer_start
            results.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": q["expected_answer"],
                    "retrieval_time_seconds": retrieval_time,
                    "answer_time_seconds": answer_time,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
            )
            print(f"  [{q['id']}] done")
    return results


def summarize_results(results: list[dict]) -> dict:
    if not results:
        return {
            "runtime": {
                "average_retrieval_time_seconds": 0.0,
                "average_answer_time_seconds": 0.0,
            },
            "cost": {
                "average_input_tokens": 0.0,
                "average_output_tokens": 0.0,
                "average_total_tokens": 0.0,
            },
        }

    result_count = len(results)
    return {
        "runtime": {
            "average_retrieval_time_seconds": sum(r["retrieval_time_seconds"] for r in results) / result_count,
            "average_answer_time_seconds": sum(r["answer_time_seconds"] for r in results) / result_count,
        },
        "cost": {
            "average_input_tokens": sum(r["input_tokens"] for r in results) / result_count,
            "average_output_tokens": sum(r["output_tokens"] for r in results) / result_count,
            "average_total_tokens": sum(r["total_tokens"] for r in results) / result_count,
        },
    }


def get_ragas_metric(*names: str):
    for name in names:
        metric = getattr(ragas_metrics, name, None)
        if metric is not None:
            return metric
    return None


def configure_ragas_metric(metric, ragas_llm, ragas_embeddings):
    if hasattr(metric, "llm"):
        metric.llm = ragas_llm
    if hasattr(metric, "embeddings"):
        metric.embeddings = ragas_embeddings


def run_ragas(results: list[dict]) -> dict:
    ragas_llm = llm_factory(model=require_llm_model(), provider="openai", client=client)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    samples = [
        SingleTurnSample(
            user_input=result["question"],
            response=result["answer"],
            retrieved_contexts=result["contexts"],
            reference=result["ground_truth"],
        )
        for result in results
    ]
    dataset = EvaluationDataset(samples=samples)

    requested_metrics = [
        ("Faithfulness", get_ragas_metric("faithfulness")),
        ("Answer Relevancy", get_ragas_metric("answer_relevancy", "answer_relevance")),
        ("Context Precision", get_ragas_metric("context_precision")),
        ("Context Recall", get_ragas_metric("context_recall")),
        ("Answer Correctness", get_ragas_metric("answer_correctness")),
    ]
    metrics = []
    labels_by_column = {}
    skipped_metrics = []
    for label, metric in requested_metrics:
        if metric is None:
            skipped_metrics.append(label)
            continue
        configure_ragas_metric(metric, ragas_llm, ragas_embeddings)
        metrics.append(metric)
        labels_by_column[getattr(metric, "name", label)] = label

    if not metrics:
        raise RuntimeError("No requested RAGAS metrics are available in the installed ragas version.")

    scores = ragas_evaluate(dataset, metrics=metrics, batch_size=16)
    score_df = scores.to_pandas()
    mean_scores = {}
    for column, label in labels_by_column.items():
        if column in score_df:
            mean_scores[label] = float(score_df[column].mean())
        else:
            skipped_metrics.append(label)

    context_precision = mean_scores.get("Context Precision")
    context_recall = mean_scores.get("Context Recall")
    if isinstance(context_precision, float) and isinstance(context_recall, float):
        denominator = context_precision + context_recall
        mean_scores["F1"] = (
            0.0 if denominator == 0 else (2 * context_precision * context_recall) / denominator
        )

    if skipped_metrics:
        mean_scores["skipped_metrics"] = sorted(set(skipped_metrics))
    return mean_scores


def print_report(report_title: str, ragas_scores: dict, summary: dict, elapsed: float):
    sep = "─" * 52
    runtime = summary["runtime"]
    cost = summary["cost"]
    print(f"\n{'═' * 52}")
    print(f"  {report_title}")
    print(f"{'═' * 52}")

    print("\n  RAGAS")
    print(sep)
    for key, value in ragas_scores.items():
        print(f"  {key:<28} {value:.4f}" if isinstance(value, float) else f"  {key:<28} {value}")

    print("\n  Runtime")
    print(sep)
    print(f"  {'Avg retrieval time (s)':<28} {runtime['average_retrieval_time_seconds']:.4f}")
    print(f"  {'Avg answer time (s)':<28} {runtime['average_answer_time_seconds']:.4f}")

    print("\n  Cost")
    print(sep)
    print(f"  {'Avg input tokens':<28} {cost['average_input_tokens']:.2f}")
    print(f"  {'Avg output tokens':<28} {cost['average_output_tokens']:.2f}")
    print(f"  {'Avg total tokens':<28} {cost['average_total_tokens']:.2f}")
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"{'═' * 52}\n")


def save_report(report_path: Path, config: Any, results: list[dict], ragas_scores: dict, summary: dict):
    if is_dataclass(config):
        config_data = asdict(config)
    elif isinstance(config, dict):
        config_data = config
    else:
        raise TypeError("config must be a dataclass instance or dict.")

    report = {
        "config": config_data,
        "ragas": ragas_scores,
        "runtime": summary["runtime"],
        "cost": summary["cost"],
        "per_question": results,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  Report saved → {report_path}")


def run_family_benchmark(
    *,
    report_title: str,
    run_label: str,
    report_filename: str,
    config: Any,
    retriever,
) -> None:
    cases = load_evaluation_cases()
    start_time = time.time()

    print(f"\n[1/2] Running {run_label} queries …")
    results = collect_results(cases, retriever)
    summary = summarize_results(results)

    print("\n[2/2] RAGAS evaluation …")
    ragas_scores = run_ragas(results)

    elapsed = time.time() - start_time
    print_report(report_title, ragas_scores, summary, elapsed)
    save_report(BENCHMARK_DIR / "reports" / report_filename, config, results, ragas_scores, summary)
