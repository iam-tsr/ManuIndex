# Benchmark Analysis

This benchmark suite compares the GRAG retrieval pipeline against a Naive RAG baseline on a mixed-document evaluation set. The goal is to measure whether improved document-aware retrieval produces better grounded answers without changing the generation setup.

## Purpose

The benchmark is designed to test retrieval quality across a broad "document zoo" rather than only clean prose. The evaluation set includes job descriptions, religious essays, press releases, meeting minutes, regulatory and policy documents, privacy policies, clinical and research material, bus schedules, school safety plans, and other heterogeneous document types.

The analysis focuses on whether the retrieval system can:

- Find exact facts such as dates, names, roles, prices, vote counts, and regulatory details.
- Preserve document-local context when questions depend on nearby clauses or sections.
- Avoid mixing unrelated content from different documents.
- Support answer generation with concise, grounded evidence.
- Improve recall while maintaining precision and faithfulness.

## Systems Compared

### GRAG

GRAG uses the ManuIndex retrieval stack with ONNX BGE-M3 embeddings and an ONNX BGE reranker. It retrieves multiple ranked contexts for each question and passes those contexts to the answer generator.

### Naive RAG

The baseline uses fixed-size text chunks, FAISS similarity search, and the same embedding model family. It retrieves the top matching chunks directly from a flat vector index and passes the retrieved text to the answer generator.

## Shared Evaluation Setup

Both systems use the same question set, same source documents, same top-k setting, same chunk-size setting, same embedding model family, and same answer-generation prompt. This keeps the comparison focused on retrieval behavior rather than differences in the generator or evaluation data.

Current shared configuration:

| Setting | Value |
| --- | --- |
| Questions | 125 |
| Documents | 25 |
| Top-k | 3 |
| Chunk size | 500 |
| Embedding model | BGE-M3 ONNX Q4 |
| Generator | Gemma-4-E2B GGUF Q4 |

## Metrics

The benchmark uses RAGAS-style metrics to evaluate both answer quality and retrieval quality.

| Metric | What it measures |
| --- | --- |
| Faithfulness | Whether the generated answer is supported by retrieved context. |
| Answer relevancy | Whether the answer directly addresses the question. |
| Context precision | Whether retrieved contexts are useful and not noisy. |
| Context recall | Whether retrieved contexts contain the information needed to answer. |
| Answer correctness | Whether the generated answer matches the expected answer. |

## Aggregate Results

| Metric | GRAG (Our) | Naive RAG | Absolute gain | Relative gain |
| --- | ---: | ---: | ---: | ---: |
| Faithfulness | **0.9799** | 0.9408 | +0.0391 | +4.15% |
| Answer relevancy | **0.8332** | 0.7993 | +0.0339 | +4.24% |
| Context precision | **0.8547** | 0.8240 | +0.0307 | +3.72% |
| Context recall | **0.9205** | 0.8204 | +0.1001 | +12.20% |
| Answer correctness | **0.6872** | 0.6374 | +0.0498 | +7.81% |

## Main Findings

GRAG outperforms the Naive RAG baseline on every reported metric. The largest improvement is in context recall, which suggests that GRAG retrieves more of the information needed to answer each question. This is important for heterogeneous documents where relevant evidence may be separated by formatting, section boundaries, tables, lists, or document-specific structure.

The gains in answer correctness and answer relevancy indicate that better retrieval also improves generation quality. The generator receives stronger evidence, so it is more likely to produce complete and directly useful answers.

The faithfulness score is high for both systems, but GRAG still improves it. This suggests that GRAG does not merely retrieve more text; it retrieves context that remains relevant enough for grounded answer generation.

## Retrieval Behavior Analysis

The Naive RAG baseline often retrieves broad chunks that contain some relevant information but may miss neighboring details needed for complete answers. This is especially visible when a question asks for multiple linked facts, such as names plus roles, dates plus outcomes, or policy conditions plus exceptions.

GRAG retrieves richer context sets and benefits from reranking. This helps when documents contain repeated labels, dense lists, tabular-style content, or mixed sections where a flat chunking strategy can blur boundaries.

The improvement in context recall shows that GRAG is better at surfacing the needed evidence. The simultaneous improvement in context precision shows that this recall gain does not come only from adding noise.

## Answer Quality Analysis

GRAG answers are generally more complete because the retrieved context more often includes all required facts. The largest practical improvements appear in questions that require:

- Combining multiple facts from the same document.
- Reading structured sections such as meeting minutes, policies, and reports.
- Extracting exact numeric values, dates, names, and procedural details.
- Understanding document-specific wording rather than broad semantic similarity alone.

The Naive RAG baseline can answer many direct questions, especially when the answer appears in a single obvious chunk. It is weaker when evidence is split across nearby sections or when the most similar chunk is not the most complete chunk.

## Interpretation

The benchmark supports the core GRAG hypothesis: retrieval improvements can raise answer quality without relying on larger prompts or heavier generation. The results point to retrieval structure, context selection, and reranking as useful levers for solving the document zoo problem.

The strongest evidence is the context recall gain. In RAG systems, missing evidence is a hard failure mode because the generator cannot faithfully answer from information it never receives. GRAG reduces that failure mode while also improving precision, correctness, and faithfulness.

## Reliability Note

These results should be interpreted as directional rather than definitive. The benchmark uses a small language model for answer generation and model-based evaluation, which may make the reported scores less reliable than results produced with a stronger evaluator or with human review. The comparison is still useful for observing retrieval behavior under the same setup, but the absolute numbers should not be treated as final benchmark claims.

## Conclusion

The benchmark shows that GRAG provides a consistent improvement over the Naive RAG baseline across retrieval and answer-quality metrics. The most important result is the strong context recall improvement, which directly supports the goal of making RAG systems more reliable on heterogeneous real-world documents.
