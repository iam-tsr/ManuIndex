from __future__ import annotations

import math
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from manu_index.src.datastore import LocalDataStore


class LocalDataStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "manu_index.sqlite3"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def assert_vectors_close(
        self,
        actual: list[list[float]],
        expected: list[list[float]],
    ) -> None:
        self.assertEqual(len(actual), len(expected))
        for actual_vector, expected_vector in zip(actual, expected):
            self.assertEqual(len(actual_vector), len(expected_vector))
            for actual_value, expected_value in zip(actual_vector, expected_vector):
                self.assertTrue(math.isclose(actual_value, expected_value, rel_tol=1e-6))

    def test_persists_route_metadata_and_chunks(self) -> None:
        store = LocalDataStore(self.db_path)
        store.add_document(
            doc_id="doc-1",
            summary="A summary of document one.",
            summary_vector=[0.1, -0.2, 0.3],
            metadata={
                "created_at": datetime(2026, 9, 25, 12, 30),
                "title": "Document One",
            },
            chunks=[
                {
                    "doc_id": "doc-1",
                    "chunk_index": 0,
                    "text": "First chunk.",
                    "vector": [0.1, 0.2, 0.3],
                },
                {
                    "doc_id": "doc-1",
                    "chunk_index": 1,
                    "text": "Second chunk.",
                    "vector": [0.4, 0.5, 0.6],
                },
            ],
        )
        store.close()

        reopened = LocalDataStore(self.db_path)
        try:
            route_meta = reopened.get_route_meta()
            self.assertEqual(len(route_meta), 1)
            self.assertEqual(route_meta[0]["doc_id"], "doc-1")
            self.assertEqual(route_meta[0]["summary"], "A summary of document one.")
            self.assert_vectors_close(
                [route_meta[0]["summary_vector"]],
                [[0.1, -0.2, 0.3]],
            )
            self.assertEqual(
                route_meta[0]["metadata"],
                {
                    "created_at": "2026-09-25T12:30:00",
                    "title": "Document One",
                },
            )
            chunks = reopened.get_document_chunks(["doc-1"])
            self.assertEqual(
                [chunk["chunk_index"] for chunk in chunks],
                [0, 1],
            )
            self.assertEqual(
                [chunk["text"] for chunk in chunks],
                ["First chunk.", "Second chunk."],
            )
            self.assert_vectors_close(
                [chunk["vector"] for chunk in chunks],
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            )
        finally:
            reopened.close()

    def test_delete_and_clear_remove_documents(self) -> None:
        with LocalDataStore(self.db_path) as store:
            store.add_document(
                doc_id="doc-1",
                summary="Summary.",
                summary_vector=[1.0, 0.0],
                metadata={},
                chunks=[
                    {
                        "doc_id": "doc-1",
                        "chunk_index": 0,
                        "text": "Chunk.",
                        "vector": [1.0, 0.0],
                    }
                ],
            )
            store.delete_document("doc-1")
            self.assertEqual(store.get_route_meta(), [])
            self.assertEqual(store.get_document_chunks(["doc-1"]), [])

            store.add_document(
                doc_id="doc-2",
                summary="Summary.",
                summary_vector=[1.0, 0.0],
                metadata={},
                chunks=[],
            )
            store.clear()
            self.assertEqual(store.get_route_meta(), [])

    def test_add_document_is_atomic_when_chunk_is_invalid(self) -> None:
        with LocalDataStore(self.db_path) as store:
            with self.assertRaises(ValueError):
                store.add_document(
                    doc_id="doc-1",
                    summary="Summary.",
                    summary_vector=[1.0, 0.0],
                    metadata={},
                    chunks=[
                        {
                            "doc_id": "doc-1",
                            "chunk_index": 0,
                            "text": "Chunk.",
                            "vector": [1.0],
                        }
                    ],
                )

            self.assertEqual(store.get_route_meta(), [])
            self.assertEqual(store.get_document_chunks(["doc-1"]), [])

    def test_rejects_mismatched_vector_dimensions(self) -> None:
        with LocalDataStore(self.db_path) as store:
            store.add_document(
                doc_id="doc-1",
                summary="Summary.",
                summary_vector=[1.0, 0.0],
                metadata={},
                chunks=[],
            )
            with self.assertRaises(ValueError):
                store.add_document(
                    doc_id="doc-2",
                    summary="Summary.",
                    summary_vector=[1.0, 0.0, 0.0],
                    metadata={},
                    chunks=[],
                )

    def test_rejects_duplicate_document(self) -> None:
        with LocalDataStore(self.db_path) as store:
            store.add_document(
                    doc_id="doc-1",
                    summary="Summary.",
                    summary_vector=[1.0, 0.0],
                    metadata={},
                    chunks=[],
                )
            with self.assertRaises(ValueError):
                store.add_document(
                    doc_id="doc-1",
                    summary="Summary.",
                    summary_vector=[1.0, 0.0],
                    metadata={},
                    chunks=[],
                )


if __name__ == "__main__":
    unittest.main()
