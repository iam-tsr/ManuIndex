import math
import re
from collections.abc import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    text: str,
    chunk_size: int,
    chunk_overlap: int = 30,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " "],
    )
    chunks = splitter.split_text(text)

    documents: list[Document] = []
    for chunk_index, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        documents.append(
            Document(
                page_content=chunk,
                metadata={"chunk_index": chunk_index},
            )
        )
    return documents


def dedupe_texts(texts: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def select_top_k(texts: Iterable[str], top_k: int) -> list[str]:
    """Dedupe and keep the first ``top_k`` passages as separate contexts."""
    return dedupe_texts(texts)[:top_k]


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def dot_similarity(left: list[float], right: list[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def tokenize_keywords(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
