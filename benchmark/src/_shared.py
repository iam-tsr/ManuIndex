import math
import re
from collections.abc import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    text: str,
    chunk_size: int,
    chunk_overlap: int = 0,
    separators: list[str] | None = None,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators or ["\n\n", "\n", " "],
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


def split_markdown_sections(text: str, fallback_chunk_size: int = 1200) -> list[Document]:
    lines = text.splitlines()
    sections: list[Document] = []
    current_heading = "Document"
    current_lines: list[str] = []
    section_index = 0

    def flush_section() -> None:
        nonlocal section_index, current_lines
        body = "\n".join(current_lines).strip()
        if not body:
            return
        content = f"{current_heading}\n{body}" if current_heading else body
        sections.append(
            Document(
                page_content=content,
                metadata={"section_index": section_index, "heading": current_heading},
            )
        )
        section_index += 1
        current_lines = []

    for line in lines:
        if line.lstrip().startswith("#"):
            flush_section()
            current_heading = line.strip()
            continue
        current_lines.append(line)

    flush_section()
    if sections:
        return sections

    return split_documents(text, chunk_size=fallback_chunk_size, chunk_overlap=120)


def dedupe_texts(texts: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def join_top_k(texts: Iterable[str], top_k: int) -> str:
    items = dedupe_texts(texts)
    return "\n\n".join(items[:top_k])


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def dot_similarity(left: list[float], right: list[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def tokenize_keywords(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
