<img src="public/banner.png" alt="ManuIndex banner" width="100%">

> Core implementation of **GRAG**: a document-aware retrieval pipeline built for heterogeneous RAG corpora.

![Python](https://img.shields.io/badge/Python-3.11%2B-1f6feb)
![License](https://img.shields.io/badge/License-MIT-2ea043)
![Focus](https://img.shields.io/badge/RAG-Document--Aware-0a7ea4)

ManuIndex is designed for the "document zoo" problem: policies, reports, minutes, contracts, research notes, schedules, and other formats often behave poorly when everything is dumped into one flat vector index.

Instead of retrieving chunks from one mixed search space, ManuIndex routes the query to the most relevant documents first, then runs hybrid retrieval inside those selected documents only.

## Why It Works

- **Document routing first**: each document gets a compact LLM summary used as a routing vector.
- **Local retrieval second**: dense FAISS and sparse BM25 retrieval run inside selected documents.
- **Better context locality**: neighbor chunk expansion preserves nearby evidence.
- **Cleaner final ranking**: ONNX reranking filters noisy candidates before generation.
- **Practical deployment**: embeddings and reranking can run locally with ONNX Runtime.

## Retrieval Flow

```mermaid
flowchart TD
    A[Document] --> B[Summary generation]
    A --> C[Deterministic chunking]
    B --> D[Summary embedding]
    C --> E[Per-document FAISS index]
    C --> F[Per-document BM25 index]

    Q[Query] --> R[Query embedding]
    R --> D
    D --> G[Select top documents]
    G --> E
    G --> F
    E --> H[Hybrid retrieval]
    F --> H
    H --> I[Neighbor expansion]
    I --> J[ONNX reranking]
    J --> K[Final contexts]
```

## Benchmark Snapshot

The benchmark evaluates **7 retrieval pipelines** on a heterogeneous corpus of **25 documents** and **125 questions** with a fixed `top_k=3`.

| Method Group | Avg F1 | Avg Context Recall | Avg End-to-End Time |
| --- | ---: | ---: | ---: |
| GRAG | **0.6631** | **0.7186** | **0.583s** |
| Best non-GRAG flat / hierarchical baselines | 0.6237 | 0.6358 | slower |
| Graph RAG | **0.9565** | **0.9473** | 3.681s |

Interpretation:

- **GRAG is the best efficiency-oriented system** in this repo's benchmark.
- **Graph RAG is the quality leader overall**, but with materially higher latency and token cost.
- GRAG improves retrieval quality over simpler flat baselines without turning into the most expensive pipeline.

## Installation

ManuIndex requires **Python 3.11+** and uses `uv`.

```bash
uv sync
```

Core dependencies include FAISS, LangChain community utilities, ONNX Runtime via Optimum, Transformers, Rank-BM25, and PDF-to-Markdown tooling.

## Model Setup

Download the default embedding model:

```bash
python helpers/model_download.py
```

If you also want the reranker weights, call `download_onnx_models("reranker", "onnx_models")` from the helper module.

## Environment

Set these variables before running the examples:

```bash
OPENAI_API_KEY=...
OPENAI_MODEL_NAME=...
OPENAI_BASE_URL=...   # optional for OpenAI-compatible endpoints
```

If you use the PDF image-analysis helper, also set:

```bash
GROQ_API_KEY=...
```

## Quick Start

```python
import os
from openai import OpenAI
from manu_index import ManuIndex, ONNXEmbedder, ONNXReranker

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL"),
)

embeddings = ONNXEmbedder(
    model="onnx_models/bge_m3/onnx/model_q4.onnx",
    tokenizer="onnx_models/bge_m3",
    max_length=1024,
    device="cpu",
)

reranker = ONNXReranker(
    model="onnx_models/bge_reranker_v2_m3/onnx/model_q4.onnx",
    tokenizer="onnx_models/bge_reranker_v2_m3",
    max_length=1024,
    device="cpu",
    reranker_type="auto",
)

index = ManuIndex(
    client=client,
    model_name=os.environ["OPENAI_MODEL_NAME"],
    embeddings=embeddings,
    persist_directory="manu_index_db",
)

index.add_document("sample.md")

results = index.search(
    query="What role is being hired for?",
    reranker=reranker,
    top_k=3,
    top_c=5,
    alpha=0.5,
    lambda_mult=0.8,
)

for text in results:
    print(text)
```

## Public API

### `ManuIndex`

Main methods:

```python
index.add_document(documents, chunk_size=500)
index.search(query, top_k=3, top_c=5, lambda_mult=0.8, alpha=0.5, reranker=reranker)
index.info()
index.delete(doc_id)
index.clear()
```

Search behavior:

1. Embed the query.
2. Route it to the top document summaries.
3. Retrieve candidates with dense + sparse search.
4. Expand neighbor chunks.
5. Rerank the final candidate pool.

### `ONNXEmbedder`

LangChain-compatible embedding wrapper with:

- CPU and CUDA execution
- batched inference
- mean pooling
- optional normalization
- `embed_documents()` and `embed_query()`

### `ONNXReranker`

ONNX reranker supporting:

- BGE classifier rerankers
- BGE decoder rerankers
- Qwen decoder rerankers
- automatic reranker type inference
- CPU and CUDA execution

## PDF Ingestion

PDFs can be converted to Markdown before indexing, including optional image analysis for charts, figures, or visually rich pages.

```python
import pymupdf
import pymupdf4llm
from pymupdf4llm.helpers.image_analyzer import GroqImageAnalyzer

analyzer = GroqImageAnalyzer(
    api_key="...",
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
)

with pymupdf.open("report.pdf") as document:
    markdown = pymupdf4llm.to_markdown(document, analyze_image=analyzer)

index.add_document(markdown)
```

## Repository Highlights

- `manu_index`: core retrieval, embedding, reranking, and summary-routing logic
- `benchmark`: evaluation suite, saved reports, and comparison plots
- `helpers`: model download and PDF parsing utilities
- `tests`: usage examples for indexing, search, reranking, and summarization

## Notes

- Document summaries are generated with an LLM and stored as routing metadata.
- Each indexed document gets its own FAISS and BM25 stores rather than joining all chunks into one global index.
- `MATHS.md` contains the underlying retrieval formulations and scoring notes.

## License

MIT. See `LICENSE`.
