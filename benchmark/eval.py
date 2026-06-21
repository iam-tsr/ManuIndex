"""
ManuIndex Benchmark Evaluation Pipeline
========================================
Metrics: RAGAS (faithfulness, answer relevancy, context recall),
         RAGChecker, BLEU, ROUGE-L
"""

import os
import json
import time
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI
from manu_index import ManuIndex, ONNXEmbedder

from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI

from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


load_dotenv()

DATASET_DIR = Path(__file__).parent / "dataset"
CASES_FILE  = Path(__file__).parent / "evaluation_cases.json"

MODEL_DIR    = "onnx_models/embeddinggemma_300m/onnx/model.onnx"
TOKENIZER    = "onnx_models/embeddinggemma_300m"
MAX_LENGTH   = 768
LLM_MODEL    = os.getenv("OPENAI_MODEL_NAME")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

embeddings = ONNXEmbedder(MODEL_DIR, TOKENIZER, MAX_LENGTH)
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
smooth = SmoothingFunction().method1



def generate_answer(query: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(contexts)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "Answer the question using only the provided context."},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

def bleu(reference: str, hypothesis: str) -> float:
    ref_tokens  = reference.lower().split()
    hyp_tokens  = hypothesis.lower().split()
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth)

def rouge_l(reference: str, hypothesis: str) -> float:
    return rouge.score(reference, hypothesis)["rougeL"].fmeasure

def ingest_documents(cases: list[dict], persist_dir: str) -> ManuIndex:
    db = ManuIndex(embeddings=embeddings, client=client, persist_directory=persist_dir)
    files_added: set[str] = set()
    for case in cases:
        fname = case["file"]
        if fname not in files_added:
            path = DATASET_DIR / fname
            print(f"  Ingesting {fname} …")
            db.add_document(str(path))
            files_added.add(fname)
    return db

def collect_results(db: ManuIndex, cases: list[dict]) -> list[dict]:
    """Return a flat list of per-question result dicts."""
    results = []
    for case in cases:
        for q in case["questions"]:
            contexts = db.search(q["question"], top_k=3)
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

    faithfulness.llm = ragas_llm
    context_recall.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings

    scores = ragas_evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])
    return scores.to_pandas()[["faithfulness", "answer_relevancy", "context_recall"]].mean().to_dict()

def run_ragchecker(results: list[dict]) -> dict:
    rag_results_json = {
        "results": [
            {
                "query_id":       r["id"],
                "query":          r["question"],
                "gt_answer":      r["ground_truth"],
                "response":       r["answer"],
                "retrieved_context": [{"text": c} for c in r["contexts"]],
            }
            for r in results
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(rag_results_json, f)
        tmp_path = f.name

    api_base = os.getenv("OPENAI_BASE_URL")
    try:
        rag_results = RAGResults.from_json(Path(tmp_path).read_text())
        checker = RAGChecker(
            extractor_name=LLM_MODEL,
            checker_name=LLM_MODEL,
            extractor_api_base=api_base,
            checker_api_base=api_base,
            batch_size_extractor=10,
            batch_size_checker=10,
        )
        checker.evaluate(rag_results, all_metrics)
        merged = {}
        merged.update(rag_results.metrics.get("retriever_metrics", {}))
        merged.update(rag_results.metrics.get("generator_metrics", {}))
        return merged
    finally:
        os.unlink(tmp_path)

def run_traditional(results: list[dict]) -> dict:
    bleu_scores   = [bleu(r["ground_truth"], r["answer"]) for r in results]
    rouge_scores  = [rouge_l(r["ground_truth"], r["answer"]) for r in results]
    return {
        "bleu_avg":    round(sum(bleu_scores)  / len(bleu_scores),  4),
        "rouge_l_avg": round(sum(rouge_scores) / len(rouge_scores), 4),
    }

def print_report(ragas_scores: dict, ragchecker_scores: dict, trad_scores: dict, elapsed: float):
    sep = "─" * 52
    print(f"\n{'═' * 52}")
    print("  ManuIndex Benchmark Report")
    print(f"{'═' * 52}")

    print("\n  RAGAS")
    print(sep)
    for k, v in ragas_scores.items():
        print(f"  {k:<28} {v:.4f}")

    print("\n  RAGChecker")
    print(sep)
    for k, v in ragchecker_scores.items():
        print(f"  {k:<28} {v:.4f}" if isinstance(v, float) else f"  {k:<28} {v}")

    print("\n  Traditional (BLEU / ROUGE)")
    print(sep)
    for k, v in trad_scores.items():
        print(f"  {k:<28} {v:.4f}")

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"{'═' * 52}\n")


def save_report(results: list[dict], ragas_scores, ragchecker_scores, trad_scores):
    report = {
        "ragas":       ragas_scores,
        "ragchecker":  ragchecker_scores,
        "traditional": trad_scores,
        "per_question": [
            {
                "id":     r["id"],
                "bleu":   round(bleu(r["ground_truth"], r["answer"]), 4),
                "rougeL": round(rouge_l(r["ground_truth"], r["answer"]), 4),
            }
            for r in results
        ],
    }
    out = Path(__file__).parent / "report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"  Report saved → {out}")



def main():
    cases = json.loads(CASES_FILE.read_text())["evaluation_cases"]

    with tempfile.TemporaryDirectory(prefix="manu_eval_") as persist_dir:
        t0 = time.time()

        print("\n[1/4] Ingesting documents …")
        db = ingest_documents(cases, persist_dir)

        print("\n[2/4] Running queries …")
        results = collect_results(db, cases)

        print("\n[3/4] RAGAS evaluation …")
        ragas_scores = run_ragas(results)

        print("\n[3/4] RAGChecker evaluation …")
        ragchecker_scores = run_ragchecker(results)

        print("\n[4/4] BLEU / ROUGE …")
        trad_scores = run_traditional(results)

        elapsed = time.time() - t0

    print_report(ragas_scores, ragchecker_scores, trad_scores, elapsed)
    save_report(results, ragas_scores, ragchecker_scores, trad_scores)


if __name__ == "__main__":
    main()