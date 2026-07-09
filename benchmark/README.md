# Benchmark Suite

## Overview

This benchmark evaluates retrieval quality, latency, and token usage for seven retrieval-augmented generation pipelines on a heterogeneous document corpus. The benchmark is designed for the document-zoo setting: real collections contain job descriptions, minutes, policy text, privacy notices, clinical material, research-style prose, schedules, and corporate documents, all of which impose different retrieval demands.

The central question is not whether a larger language model can compensate for weak retrieval, but whether retrieval structure itself improves grounded answering under a fixed question set and a fixed top-k budget.

## Research Objective

The benchmark examines three related claims:

1. Retrieval architecture matters when the corpus is structurally heterogeneous.
2. Document-aware retrieval can improve answer quality without increasing the final retrieval budget.
3. Accuracy gains should be interpreted jointly with efficiency signals, especially latency and token consumption.

This makes the benchmark useful for both model comparison and retrieval-system design analysis.

## Systems Evaluated

The current report matrix compares seven retrieval families:

1. `Naive RAG`
   Fixed-size chunking with direct flat retrieval.

2. `Flat Hybrid RAG`
   Dense and sparse retrieval combined in a single flat retrieval space.

3. `Hierarchical RAG`
   Multi-stage retrieval that first selects broader sections and then narrows to finer text units.

4. `Parent-Child RAG`
   Retrieval using parent and child chunk relationships to preserve local context.

5. `Query Rewrite RAG`
   A pipeline that reformulates the question before retrieval.

6. `Graph RAG`
   A graph-oriented retrieval strategy with substantially richer context construction.

7. `GRAG`
   The document-aware retrieval design implemented in this repository. GRAG first routes a query toward the most relevant document collections and then retrieves evidence from within those selected collections.

## Experimental Design

All systems are evaluated on the same benchmark corpus and question set.

### Corpus Scale

| Item | Value |
| --- | ---: |
| Documents | 25 |
| Evaluation questions | 125 |
| Questions per document | 5 |

### Shared Retrieval Settings

| Setting | Value |
| --- | ---: |
| Top-k contexts returned for answer generation | 3 |
| Chunk size | 150 |

### Embedding and Generator Matrix

Each retrieval family is evaluated under four embedding-generator settings:

| Embedding backend | Generator |
| --- | --- |
| BGE-M3 (ONNX) | Gemma-4-E2B |
| BGE-M3 (ONNX) | Qwen3.5-2B |
| Qwen3-Embedding 0.6B (ONNX) | Gemma-4-E2B |
| Qwen3-Embedding 0.6B (ONNX) | Qwen3.5-2B |

The combined report therefore summarizes 28 system-condition pairs: 7 retrieval families across 4 model settings.

## Evaluation Procedure

For each question, the benchmark runs the following sequence:

1. Retrieve supporting context with the target pipeline.
2. Generate an answer using only the retrieved context.
3. Record retrieval latency, answer-generation latency, and token usage.
4. Score outputs with RAGAS-based evaluation metrics.

This design separates retrieval behavior from generation behavior while still reflecting end-to-end system performance.

## Metrics

The reports contain five primary RAGAS metrics plus a derived F1 score.

### Quality Metrics

- `Context Recall`
  Measures how much of the necessary supporting evidence is successfully retrieved.

- `F1`
  A derived harmonic mean of context precision and context recall:

  `F1 = 2 * P * R / (P + R)`

  where `P` is context precision and `R` is context recall.

### Efficiency Metrics

- `Average retrieval time`
- `Average answer time`
- `Average end-to-end time`
- `Average input tokens`
- `Average output tokens`
- `Average total tokens`

These allow quality improvements to be assessed against practical cost and latency tradeoffs.

## Comparison Tables

### GRAG vs Other RAG Families Excluding Graph RAG

This table isolates `GRAG` against the non-graph alternatives. It is useful when the research question is whether document-aware routing improves the quality-efficiency tradeoff over lighter or more conventional retrieval families.

| Method | Avg F1 | Avg Context Recall |
| --- | ---: | ---: |
| Naive RAG | 0.6224 | 0.6341 |
| Flat Hybrid RAG | 0.6237 | 0.6282 |
| Hierarchical RAG | 0.5941 | 0.6163 |
| Parent-Child RAG | 0.6163 | 0.6358 |
| Query Rewrite RAG | 0.6224 | 0.6341 |
| GRAG | **0.6631** | **0.7186** |

Interpretation:

- `GRAG` is the strongest method in this group on both average `F1` and average `Context Recall`.
- `GRAG` also has the lowest average end-to-end latency in this subset.

### GRAG vs Graph RAG

This table isolates the main frontier comparison in the benchmark: `GRAG` as the efficiency-oriented document-aware system versus `Graph RAG` as the highest-quality system.

| Method | Avg F1 | Avg Context Recall | Avg Total Tokens | Avg End-to-End Time (s) |
| --- | ---: | ---: | ---: | ---: |
| GRAG | 0.6631 | 0.7186 | **370.84** | **0.583** |
| Graph RAG | **0.9565** | **0.9473** | 1161.67 | 3.681 |

Interpretation:

- `Graph RAG` is clearly superior on retrieval quality.
- `GRAG` has the lowest token consumption and the lowest latency in this direct comparison.
- The benchmark therefore supports a frontier view: `Graph RAG` optimizes for quality, while `GRAG` optimizes for efficiency with a clear quality gain over simpler baselines.

## Main Findings

### 1. Graph RAG dominates on quality

Across all four model settings, Graph RAG achieves the best F1 score. It is the strongest system when the benchmark objective is raw retrieval completeness and answer support.

### 2. GRAG is the strongest efficiency-oriented method

GRAG is the fastest method overall on average end-to-end latency, despite using substantially more tokens than the lightest baselines. This indicates that document-aware routing reduces retrieval time enough to matter at system level.

### 3. GRAG improves quality over the lighter baseline family

Compared with Naive RAG, Flat Hybrid RAG, Query Rewrite RAG, and most of the hierarchical alternatives, GRAG delivers a better average F1 and stronger context recall. In other words, GRAG is not the highest-quality system overall, but it improves the quality-efficiency frontier relative to simpler baselines.

### 4. Quality and efficiency are not aligned

The highest-quality system is not the cheapest or fastest. Graph RAG produces the best retrieval quality, but it does so with the highest token usage by a large margin. This benchmark therefore supports a tradeoff interpretation rather than a single “best model” conclusion.

## How to Read the Plots

Two summary figures are most useful for interpretation:

### Overall Efficiency Comparison

This plot reports a single average latency value per retrieval family, aggregated across all benchmark settings. Shorter bars indicate lower end-to-end latency.

Interpretation:

- If latency is the main deployment constraint, this plot is the fastest high-level summary.
- GRAG should be read here as a retrieval architecture that prioritizes speed without collapsing quality.

### Cost-Quality Tradeoff

This plot places average total tokens on the x-axis and F1 on the y-axis for each embedding-generator setting.

Interpretation:

- Upper-left is preferable: higher quality at lower token cost.
- Systems farther right consume more context budget.
- Systems higher up recover more relevant evidence.
- A method that appears favorable in latency may still be weak on quality or expensive in prompt budget.

## Interpretation for Research Use

The benchmark supports a more careful conclusion than “one method wins.”

- If the objective is maximum answer support and retrieval completeness, Graph RAG is the best-performing family in the current report set.
- If the objective is balanced deployment efficiency with a clear quality gain over basic flat retrieval, GRAG is the more attractive operating point.
- If the objective is low token usage with minimal system complexity, Naive RAG and Flat Hybrid RAG remain relevant baselines, though they are clearly weaker on context recall than the top systems.

## Reproducing the Benchmark

At a high level, reproduction consists of:

1. Running each retrieval family on the shared evaluation set.
2. Saving per-method reports with quality, runtime, and token statistics.
3. Aggregating those results into the combined benchmark report.
4. Regenerating the summary plots for efficiency and cost-quality tradeoff analysis.

The key requirement is to keep the evaluation set, retrieval budget, and chunking settings fixed across methods so that the comparison remains controlled.

## Recommended Reporting Language

For paper-style summaries, the most defensible phrasing is:

- Graph RAG achieves the highest benchmark quality.
- GRAG provides the best average end-to-end efficiency.
- GRAG improves the quality-efficiency tradeoff over simpler flat baselines.
- Retrieval architecture choice should therefore depend on the target operating point rather than on F1 alone.

That framing is faithful to the benchmark evidence and avoids overstating the results.
