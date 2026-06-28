# GRAG vs Naive RAG Benchmark: Two-LLM Comparison

## Purpose

This benchmark compares GRAG against a naive RAG baseline on the same heterogeneous document-question workload using two separate small LLMs. The goal is to measure whether GRAG improves retrieval-grounded answer quality across different generators, rather than only tuning for one model.

## Benchmark setup

- Dataset: 25 heterogeneous Markdown documents from `benchmark/dataset/`.
- Evaluation questions: 125 total questions from `benchmark/evaluation_cases.json`.
- Hardware setting: local inference on an NVIDIA RTX 3060 Ti with 8GB VRAM.
- Generator setting: weak 4-bit quantized GGUF language models were used to reflect a constrained local-inference setup.
- Compared systems:
  - GRAG: `benchmark/grag_benchmark.py`
  - Naive RAG: `benchmark/naive_rag_benchmark.py`
- LLM runs:
  - `benchmark/gemma4_2b/`
  - `benchmark/qwen2.5_3b/`
- Report files:
  - `benchmark/gemma4_2b/grag_report.json`
  - `benchmark/gemma4_2b/naive_rag_report.json`
  - `benchmark/qwen2.5_3b/grag_report.json`
  - `benchmark/qwen2.5_3b/naive_rag_report.json`
- Evaluation framework: RAGAS.
- Metrics:
  - faithfulness
  - answer_relevancy
  - context_precision
  - context_recall
  - answer_correctness

## System configuration

The naive RAG baseline builds one flat FAISS index over all benchmark documents. It splits each document with `RecursiveCharacterTextSplitter` using `chunk_size=500`, `chunk_overlap=0`, and retrieves `top_k=2` chunks per query.

GRAG uses `ManuIndex`, which indexes documents separately, creates document summaries, selects the most relevant document collections, combines dense FAISS retrieval with BM25 retrieval, expands retrieved chunks with neighbouring chunks, deduplicates candidate context, and reranks candidates before returning the final `top_k=2` contexts.

Both benchmark scripts use the same answer-generation instruction: answer only from the provided context, use simple terms, and say that the answer cannot be determined if the context does not contain the answer. The evaluated GGUF models were small and heavily quantized for the available 8GB GPU memory, so a high decoding temperature (`temperature=3`) was used because stricter decoding often failed to produce complete prompt-following answers. This setting should be treated as a limitation because it can increase answer variance.

## Primary paper metric

For the paper, the clearest single headline metric is answer correctness, because it directly measures final answer quality against the reference answer. The values below are RAGAS answer correctness scores in the range 0 to 1. They should be described as `RAGAS answer correctness`, not as a standard F1 score, unless a separate lexical, factual, token-level, or entity-level F1 computation is added.

| LLM | GRAG answer correctness | Naive RAG answer correctness | Absolute gain | Relative gain |
|---|---:|---:|---:|---:|
| gemma4_2b | 0.6929 | 0.6380 | +0.0549 | +8.6% |
| qwen2.5_3b | 0.7623 | 0.6873 | +0.0750 | +10.9% |
| Average | 0.7276 | 0.6626 | +0.0650 | +9.8% |

Paper-ready statement:

> Across two locally deployed 4-bit quantized GGUF LLMs, GRAG improved average RAGAS answer correctness from 0.6626 to 0.7276, corresponding to a +0.0650 absolute gain and approximately 9.8% relative improvement over naive RAG.

## Results with gemma4_2b

| Metric | GRAG | Naive RAG | Absolute gain | Relative gain |
|---|---:|---:|---:|---:|
| faithfulness | 0.9341 | 0.9105 | +0.0236 | +2.6% |
| answer_relevancy | 0.8252 | 0.7883 | +0.0369 | +4.7% |
| context_precision | 0.8480 | 0.7200 | +0.1280 | +17.8% |
| context_recall | 0.8787 | 0.8293 | +0.0493 | +5.9% |
| answer_correctness | 0.6929 | 0.6380 | +0.0549 | +8.6% |

GRAG outperforms naive RAG on all five RAGAS metrics with `gemma4_2b`. The largest improvement is in context precision, which indicates that GRAG supplies more relevant retrieved context to the generator.

## Results with qwen2.5_3b

| Metric | GRAG | Naive RAG | Absolute gain | Relative gain |
|---|---:|---:|---:|---:|
| faithfulness | 0.6512 | 0.6822 | -0.0309 | -4.5% |
| answer_relevancy | 0.7648 | 0.7182 | +0.0466 | +6.5% |
| context_precision | 0.9080 | 0.8240 | +0.0840 | +10.2% |
| context_recall | 0.9100 | 0.8317 | +0.0783 | +9.4% |
| answer_correctness | 0.7623 | 0.6873 | +0.0750 | +10.9% |

With `qwen2.5_3b`, GRAG improves answer relevancy, context precision, context recall, and answer correctness. Naive RAG is higher on faithfulness for this LLM, so the faithfulness result should be treated as mixed rather than uniformly favouring GRAG.

## Average across both LLMs

| Metric | GRAG average | Naive RAG average | Average absolute gain |
|---|---:|---:|---:|
| faithfulness | 0.7927 | 0.7963 | -0.0036 |
| answer_relevancy | 0.7950 | 0.7532 | +0.0418 |
| context_precision | 0.8780 | 0.7720 | +0.1060 |
| context_recall | 0.8943 | 0.8305 | +0.0638 |
| answer_correctness | 0.7276 | 0.6626 | +0.0650 |

Across both LLMs, GRAG produces consistent gains in answer relevancy, context precision, context recall, and answer correctness. The strongest average improvement is context precision, followed by answer correctness and context recall. Faithfulness is effectively tied on average, with naive RAG ahead by 0.0036.

## Key findings

1. GRAG consistently improves retrieval context quality.
   Context precision increases for both LLMs: +17.8% with `gemma4_2b` and +10.2% with `qwen2.5_3b`. Context recall also increases for both LLMs: +5.9% and +9.4%, respectively.

2. Better retrieval translates into better answers.
   Answer correctness improves for both LLMs: +8.6% with `gemma4_2b` and +10.9% with `qwen2.5_3b`.

3. The main GRAG advantage is evidence selection, not prompt dependence.
   Because the same benchmark is repeated with two different LLMs, the consistent gains in context precision, context recall, answer relevancy, and answer correctness suggest that the improvement comes from the retrieval structure rather than from a single generator's behavior.

4. Faithfulness should be reported carefully.
   GRAG improves faithfulness with `gemma4_2b`, but naive RAG is higher with `qwen2.5_3b`. This makes faithfulness a mixed result rather than a clear GRAG win.

## Paper-ready interpretation

Across two locally deployed 4-bit quantized GGUF LLMs and 125 questions over 25 heterogeneous documents, GRAG improves answer correctness compared with a naive flat-chunk FAISS baseline. The average RAGAS answer correctness score increases from 0.6626 to 0.7276, a +0.0650 absolute gain and approximately 9.8% relative improvement. The supporting retrieval metrics also improve consistently: GRAG achieves higher context precision, context recall, and answer relevancy for both evaluated LLMs. These results support the hypothesis that GRAG's document-aware retrieval structure improves evidence selection in document-zoo settings, where a single flat vector index can retrieve partially relevant or wrong-document chunks. Faithfulness remains mixed across LLMs, suggesting that future evaluation should separate retrieval quality from generator-specific answer behavior.

## Recommended caveats

- The benchmark compares GRAG's document-aware hybrid retrieval and reranking pipeline against a flat FAISS baseline, not against every possible optimized RAG baseline.
- The generator models are weak 4-bit quantized GGUF models running locally on an NVIDIA RTX 3060 Ti with 8GB VRAM. This makes the benchmark practical and resource-constrained, but stronger instruction-tuned LLMs may improve absolute answer quality.
- Both scripts use `temperature=3` because the small quantized models often failed to produce complete prompt-following answers at stricter settings. This may increase answer variance, so a lower-temperature rerun with stronger LLMs would provide a more stable estimate.
- GRAG and naive RAG both return two final contexts, but GRAG has a richer retrieval path before reranking: document-summary selection, dense retrieval, BM25 retrieval, neighbouring chunk expansion, deduplication, and reranking.
- The faithfulness result is mixed across LLMs and should not be presented as a uniform GRAG improvement.

## Future evaluation

Future experiments should repeat the benchmark with stronger instruction-tuned LLMs, lower decoding temperature, and a larger document-question set. This would help separate three effects: GRAG's retrieval-side contribution, generator quality, and decoding variance.

## Concise conclusion

The two-LLM benchmark shows that GRAG improves answer correctness under constrained local inference with weak 4-bit quantized GGUF models. The supporting retrieval metrics indicate that this gain comes from better evidence selection: GRAG retrieves more precise and complete context before generation. The evidence supports positioning GRAG as a retrieval-side improvement for heterogeneous document collections, while leaving stronger LLMs.