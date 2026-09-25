from __future__ import annotations

import json
import sqlite3
import struct
import threading
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

import numpy as np


_VECTOR_MAGIC = b"MANUVEC1"
_VECTOR_HEADER = struct.Struct("<8sI")
_QUERY_BATCH_SIZE = 900


def _json_default(value: Any) -> str | int | float | bool:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _encode_vector(vector: Any) -> tuple[bytes, int]:
    if np.iscomplexobj(vector):
        raise ValueError("vectors must contain only real numbers")

    try:
        array = np.asarray(vector, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("vector must be a one-dimensional sequence of numbers") from error

    if array.ndim != 1 or array.size == 0:
        raise ValueError("vectors must be non-empty and one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError("vectors must contain only finite numbers")

    payload = _VECTOR_HEADER.pack(_VECTOR_MAGIC, int(array.shape[0]))
    return payload + array.tobytes(order="C"), int(array.shape[0])


def _decode_vector(value: bytes | bytearray | memoryview) -> list[float]:
    if len(value) < _VECTOR_HEADER.size:
        raise RuntimeError("stored vector is corrupt: missing header")

    magic, dimension = _VECTOR_HEADER.unpack_from(value)
    if magic != _VECTOR_MAGIC:
        raise RuntimeError("stored vector is corrupt: invalid format")

    expected_size = _VECTOR_HEADER.size + (dimension * np.dtype("<f4").itemsize)
    if dimension == 0 or len(value) != expected_size:
        raise RuntimeError("stored vector is corrupt: invalid length")

    payload = memoryview(value)[_VECTOR_HEADER.size :]
    array = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=True)
    return array.tolist()


class DataStore(Protocol):
    def add_document(
        self,
        doc_id: str,
        summary: str,
        summary_vector: Sequence[float],
        metadata: dict[str, Any],
        chunks: Iterable[dict[str, Any]],
    ) -> None:
        ...

    def get_route_meta(self) -> list[dict[str, Any]]:
        ...

    def get_document_chunks(self, doc_ids: Iterable[str]) -> list[dict[str, Any]]:
        ...

    def delete_document(self, doc_id: str) -> None:
        ...

    def clear(self) -> None:
        ...


class LocalDataStore:
    """SQLite-backed storage for document routing metadata and chunk vectors."""

    def __init__(self, path: str | Path, timeout: float = 30.0) -> None:
        self.path = Path(path).expanduser()
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.path),
            timeout=timeout,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._closed = False

        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 30000")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._initialize_schema()

    def _initialize_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS index_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS route_meta (
                doc_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                summary_vector BLOB NOT NULL,
                metadata TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS document_chunks (
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (doc_id, chunk_index),
                FOREIGN KEY (doc_id) REFERENCES route_meta(doc_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_id
                ON document_chunks(doc_id);

            INSERT OR IGNORE INTO index_meta(key, value)
                VALUES ('schema_version', '1');
            """
        )
        self._connection.commit()

    def add_document(
        self,
        doc_id: str,
        summary: str,
        summary_vector: Sequence[float],
        metadata: dict[str, Any],
        chunks: Iterable[dict[str, Any]],
    ) -> None:
        """Insert route metadata and all chunks in one atomic transaction."""
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError("doc_id must be a non-empty string")
        if not isinstance(summary, str):
            raise TypeError("summary must be a string")
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary")

        encoded_summary, dimension = _encode_vector(summary_vector)
        prepared_chunks = self._prepare_chunks(doc_id, chunks, dimension)
        metadata_json = json.dumps(
            metadata,
            default=_json_default,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        with self._lock:
            self._ensure_open()
            try:
                with self._connection:
                    existing = self._connection.execute(
                        "SELECT value FROM index_meta WHERE key = 'vector_dimension'"
                    ).fetchone()
                    if existing is not None and int(existing["value"]) != dimension:
                        raise ValueError(
                            "vector dimension does not match the existing index: "
                            f"expected {existing['value']}, got {dimension}"
                        )

                    self._connection.execute(
                        """
                        INSERT INTO route_meta(doc_id, summary, summary_vector, metadata)
                        VALUES (?, ?, ?, ?)
                        """,
                        (doc_id, summary, encoded_summary, metadata_json),
                    )
                    self._connection.executemany(
                        """
                        INSERT INTO document_chunks(doc_id, chunk_index, text, vector)
                        VALUES (?, ?, ?, ?)
                        """,
                        prepared_chunks,
                    )
                    if existing is None:
                        self._connection.execute(
                            """
                            INSERT INTO index_meta(key, value)
                                VALUES ('vector_dimension', ?)
                            """,
                            (str(dimension),),
                        )
            except sqlite3.IntegrityError as error:
                raise ValueError(f"document already exists: {doc_id}") from error

    def get_route_meta(self) -> list[dict[str, Any]]:
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                """
                SELECT doc_id, summary, summary_vector, metadata
                FROM route_meta
                ORDER BY doc_id
                """
            ).fetchall()

        return [
            {
                "doc_id": row["doc_id"],
                "summary": row["summary"],
                "summary_vector": _decode_vector(row["summary_vector"]),
                "metadata": json.loads(row["metadata"]),
            }
            for row in rows
        ]

    def get_document_chunks(self, doc_ids: Iterable[str]) -> list[dict[str, Any]]:
        ids = list(dict.fromkeys(doc_ids))
        if not ids:
            return []
        if not all(isinstance(doc_id, str) and doc_id for doc_id in ids):
            raise ValueError("doc_ids must contain only non-empty strings")

        rows: list[sqlite3.Row] = []
        with self._lock:
            self._ensure_open()
            for start in range(0, len(ids), _QUERY_BATCH_SIZE):
                batch = ids[start : start + _QUERY_BATCH_SIZE]
                placeholders = ",".join("?" for _ in batch)
                rows.extend(
                    self._connection.execute(
                        f"""
                        SELECT doc_id, chunk_index, text, vector
                        FROM document_chunks
                        WHERE doc_id IN ({placeholders})
                        ORDER BY doc_id, chunk_index
                        """,
                        batch,
                    ).fetchall()
                )

        return [
            {
                "doc_id": row["doc_id"],
                "chunk_index": int(row["chunk_index"]),
                "text": row["text"],
                "vector": _decode_vector(row["vector"]),
            }
            for row in rows
        ]

    def delete_document(self, doc_id: str) -> None:
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError("doc_id must be a non-empty string")

        with self._lock:
            self._ensure_open()
            with self._connection:
                self._connection.execute(
                    "DELETE FROM route_meta WHERE doc_id = ?",
                    (doc_id,),
                )

    def clear(self) -> None:
        with self._lock:
            self._ensure_open()
            with self._connection:
                self._connection.execute("DELETE FROM route_meta")
                self._connection.execute(
                    "DELETE FROM index_meta WHERE key = 'vector_dimension'"
                )

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> LocalDataStore:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _prepare_chunks(
        self,
        doc_id: str,
        chunks: Iterable[dict[str, Any]],
        dimension: int,
    ) -> list[tuple[str, int, str, bytes]]:
        prepared: list[tuple[str, int, str, bytes]] = []
        seen_indices: set[int] = set()

        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise TypeError("chunks must contain dictionaries")

            chunk_doc_id = chunk.get("doc_id")
            if chunk_doc_id != doc_id:
                raise ValueError("chunk doc_id must match the document doc_id")

            chunk_index = chunk.get("chunk_index")
            if isinstance(chunk_index, bool) or not isinstance(chunk_index, int):
                raise TypeError("chunk_index must be an integer")
            if chunk_index < 0:
                raise ValueError("chunk_index must be greater than or equal to zero")
            if chunk_index in seen_indices:
                raise ValueError(f"duplicate chunk_index: {chunk_index}")

            text = chunk.get("text")
            if not isinstance(text, str):
                raise TypeError("chunk text must be a string")

            encoded_vector, vector_dimension = _encode_vector(chunk.get("vector"))
            if vector_dimension != dimension:
                raise ValueError(
                    "chunk vector dimension does not match the summary vector: "
                    f"expected {dimension}, got {vector_dimension}"
                )

            seen_indices.add(chunk_index)
            prepared.append((doc_id, chunk_index, text, encoded_vector))

        return prepared

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("datastore is closed")
