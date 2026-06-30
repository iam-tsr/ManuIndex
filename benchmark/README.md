# Benchmark Analysis

This benchmark suite compares the GRAG retrieval pipeline against a Naive RAG baseline on a mixed-document evaluation set. The goal is to measure whether improved document-aware retrieval produces better grounded answers without changing the evaluation corpus or the top-level retrieval budget.

## Purpose

The benchmark is designed to stress retrieval over a broad “document zoo” rather than only clean prose. The evaluation set includes job descriptions, religious essays, press releases, meeting minutes, regulatory and policy documents, privacy policies, clinical and research material, bus schedules, school safety plans, and other heterogeneous document types.

The main question is whether retrieval can surface the right evidence for exact factual answers across mixed document structures.

## Systems Compared

### Naive RAG

The baseline uses fixed-size text chunks, flat similarity search, and top-k chunk retrieval.

### GRAG

GRAG first routes the query to the most relevant document collections and then retrieves evidence within those collections.

### GRAG + reranker

This is the full GRAG pipeline with an additional reranking stage applied to the retrieved evidence before answer generation.

## Shared Evaluation Setup

All reports use the same evaluation corpus and the same retrieval budget:

| Setting | Value |
| --- | --- |
| Questions | 125 |
| Documents | 25 |
| Top-k | 3 |
| Chunk size | 500 |

What varies across the saved reports is the embedding backend, the generator, and whether reranking is enabled.

## Report Matrix

The saved reports cover four embedding/generator combinations:

| Embeddings | Generator |
| --- | --- |
| BGE-M3 | Gemma-4-E2B |
| BGE-M3 | Qwen3.5-2B |
| Qwen3-Embedding | Gemma-4-E2B |
| Qwen3-Embedding | Qwen3.5-2B |

For each combination there are three system variants: Naive RAG, GRAG, and GRAG + reranker.

## Metrics Used Here

This summary reports only two numbers:

- CR = context recall
- F1 = the harmonic mean of precision and recall

This keeps the analysis focused on retrieval completeness plus a single combined quality score, without surfacing the rest of the metric set.

## CR and F1 Results

| Embeddings | Generator | Naive CR | Naive F1 | GRAG CR | GRAG F1 | GRAG + reranker CR | GRAG + reranker F1 | Best F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BGE-M3 | Gemma-4-E2B | 0.8204 | 0.7174 | 0.8466 | 0.7328 | 0.9205 | **0.7869** | GRAG + reranker |
| BGE-M3 | Qwen3.5-2B | 0.8157 | 0.8484 | 0.8874 | 0.8843 | 0.9296 | **0.9236** | GRAG + reranker |
| Qwen3-Embedding | Gemma-4-E2B | 0.8129 | 0.7274 | 0.8130 | 0.7181 | 0.9184 | **0.7872** | GRAG + reranker |
| Qwen3-Embedding | Qwen3.5-2B | 0.8497 | 0.8728 | 0.8745 | 0.8753 | 0.9223 | **0.9186** | GRAG + reranker |

Average CR and F1 across all four report configurations:

| System | Average CR | Average F1 |
| --- | ---: | ---: |
| Naive RAG | 0.8247 | 0.7915 |
| GRAG | 0.8554 | 0.8026 |
| GRAG + reranker | **0.9227** | **0.8541** |

## Main Findings

The consistent result across the full report set is that GRAG + reranker is the strongest system in every configuration.

Compared against Naive RAG, GRAG + reranker improves both CR and F1 in all four embedding/generator combinations:

| Embeddings | Generator | CR gain vs Naive | F1 gain vs Naive | F1 relative gain |
| --- | --- | ---: | ---: | ---: |
| BGE-M3 | Gemma-4-E2B | +0.1001 | +0.0695 | +9.69% |
| BGE-M3 | Qwen3.5-2B | +0.1139 | +0.0751 | +8.86% |
| Qwen3-Embedding | Gemma-4-E2B | +0.1054 | +0.0599 | +8.23% |
| Qwen3-Embedding | Qwen3.5-2B | +0.0726 | +0.0457 | +5.24% |

Average improvement over Naive RAG:

- +0.0980 CR
- +0.0626 F1
- +7.90% F1 relative

## Interpretation

Two conclusions stand out from the report matrix.

1. Raw GRAG without reranking is not a uniform win.
   - It beats Naive RAG in three of the four configurations.
   - It slightly trails Naive RAG in the Qwen3-Embedding + Gemma-4-E2B setting.

2. The full GRAG pipeline is consistently stronger once reranking is enabled.
   - GRAG + reranker is best in every saved configuration.
   - The gain is stable across both embedding backends and both generators.

This supports a more precise version of the GRAG claim: the main benchmark win comes from the combination of document-aware retrieval plus reranking, not from retrieval routing alone.