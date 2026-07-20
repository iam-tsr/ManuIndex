import json
import os
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, list_repo_files
from openai import OpenAI
from ragas import evaluate as ragas_evaluate
from ragas import metrics as ragas_metrics
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory

from manu_index import ONNXEmbedder


load_dotenv()

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BENCHMARK_DIR.parent

HF_DATASET_ID = "neural-bridge/rag-dataset-12000" # "iam-tsr/ragmix"  # "neural-bridge/rag-dataset-12000"
HF_DATASET_SPLIT = "test"

EMB_MODEL = "onnx_models/bge_m3/onnx/model.onnx"
EMD_TOKENIZER = "onnx_models/bge_m3"
MAX_LENGTH = 1024
EMBEDDING_MODEL_LABEL = "BGE-M3 (ONNX)"
BENCHMARK_LLM_LABEL = "Gemma-4-E2B" # "Gemma-4-E2B" # "Qwen3.5-2B"
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 100

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
LLM_MODEL = os.getenv("OPENAI_MODEL_NAME")

embeddings = ONNXEmbedder(EMB_MODEL, EMD_TOKENIZER, MAX_LENGTH, batch_size=4, device="cpu")


@dataclass(frozen=True)
class BenchmarkConfig:
    top_k: int = DEFAULT_TOP_K
    chunk_size: int = DEFAULT_CHUNK_SIZE
    emb_model: str = EMBEDDING_MODEL_LABEL
    llm_model: str = BENCHMARK_LLM_LABEL


config = BenchmarkConfig()


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


def _load_dataset(split: str = HF_DATASET_SPLIT) -> Dataset:
    """Load iam-tsr/ragmix, tolerating schema/metadata mismatches on the Hub card."""
    try:
        return load_dataset(HF_DATASET_ID, split=split)
    except Exception:
        raise RuntimeError(f"Failed to load dataset {HF_DATASET_ID} split={split!r}. Check that the dataset exists and is accessible.")


def _row_document_text(row: dict[str, Any]) -> str:
    document = row.get("document") if row.get("document") is not None else row.get("context")
    if document is None or not str(document).strip():
        raise ValueError("dataset row is missing document/context text")
    return str(document)


def load_evaluation_cases(
    dataset_id: str = HF_DATASET_ID,
    split: str = HF_DATASET_SPLIT,
) -> list[dict]:
    """Load evaluation cases from the Hugging Face dataset."""
    if dataset_id != HF_DATASET_ID:
        raise ValueError(f"Only {HF_DATASET_ID!r} is supported (got {dataset_id!r})")

    ds = _load_dataset(split=split).select(range(100)) # Limit to first 100 cases for benchmarking

    cases: list[dict] = []

    for case_index, row in enumerate(ds, start=1):
        document = _row_document_text(row)
        query = row.get("question")
        expected_answer = row.get("answer")
        if not query or expected_answer is None:
            raise ValueError(f"case {case_index} is missing question/answer")

        doc_id = f"case_{case_index:03d}"
        cases.append(
            {
                "file": doc_id,
                "document": document,
                "questions": [
                    {
                        "id": f"case_{case_index:03d}",
                        "question": str(query),
                        "expected_answer": str(expected_answer),
                    }
                ],
            }
        )

    if not cases:
        raise ValueError(f"No evaluation cases loaded from {HF_DATASET_ID} split={split!r}")
    return cases


def case_document_text(case: dict) -> str:
    """Return the full source document for a case (inlined from HF)."""
    document = case.get("document")
    if document is None or not str(document).strip():
        raise ValueError(f"case {case.get('file', '?')!r} is missing document text")
    return str(document)


def collect_results(cases: list[dict], retriever) -> list[dict]:
    results = []
    documents_by_id: dict[str, str] = {}

    for case in cases:
        doc_key = str(case.get("file") or id(case))
        if doc_key not in documents_by_id:
            print(f"  Loading {doc_key} …")
            documents_by_id[doc_key] = case_document_text(case)

        document = documents_by_id[doc_key]
        for q in case["questions"]:
            retrieval_start = time.perf_counter()
            retrieved = retriever.main(document, q["question"])
            retrieval_time = time.perf_counter() - retrieval_start
            auxiliary_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "estimated": False}
            if hasattr(retriever, "consume_token_usage"):
                raw_usage = retriever.consume_token_usage()
                if isinstance(raw_usage, dict):
                    auxiliary_usage = {
                        "input_tokens": int(raw_usage.get("input_tokens", 0) or 0),
                        "output_tokens": int(raw_usage.get("output_tokens", 0) or 0),
                        "total_tokens": int(raw_usage.get("total_tokens", 0) or 0),
                        "estimated": bool(raw_usage.get("estimated", False)),
                    }

            if isinstance(retrieved, list):
                contexts = [text for text in retrieved if isinstance(text, str) and text.strip()]
            elif isinstance(retrieved, str) and retrieved.strip():
                contexts = [retrieved]
            else:
                contexts = []
            answer_start = time.perf_counter()
            answer, answer_input_tokens, answer_output_tokens, answer_total_tokens = generate_answer(q["question"], contexts)
            answer_time = time.perf_counter() - answer_start
            input_tokens = answer_input_tokens + auxiliary_usage["input_tokens"]
            output_tokens = answer_output_tokens + auxiliary_usage["output_tokens"]
            total_tokens = answer_total_tokens + auxiliary_usage["total_tokens"]
            results.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": q["expected_answer"],
                    "retrieval_time_seconds": retrieval_time,
                    "answer_time_seconds": answer_time,
                    "additional_tokens": auxiliary_usage["total_tokens"],
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
                "average_additional_tokens": 0.0,
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
            "average_additional_tokens": sum(r.get("additional_tokens", 0) for r in results) / result_count,
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


def display_report(report_title: str, ragas_scores: dict, summary: dict, elapsed: float):
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
    print(f"  {'Avg additional tokens':<28} {cost['average_additional_tokens']:.2f}")
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
    display_report(report_title, ragas_scores, summary, elapsed)
    save_report(BENCHMARK_DIR / "reports" / report_filename, config, results, ragas_scores, summary)
