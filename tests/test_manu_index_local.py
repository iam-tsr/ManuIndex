from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from manu_index import ManuIndex
from manu_index.src.datastore import LocalDataStore


class StaticEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in documents]


class SummarylessManuIndex(ManuIndex):
    def _create_summary(self, document: str) -> str:
        return document


class ManuIndexLocalDataStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "manu_index.sqlite3"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_index(self) -> tuple[LocalDataStore, ManuIndex]:
        store = LocalDataStore(self.db_path)
        index = SummarylessManuIndex(
            api_key="test-key",
            model_name="test-model",
            base_url="http://localhost",
            embeddings=StaticEmbedder(),
            datastore=store,
        )
        return store, index

    def test_add_search_delete_and_clear_use_local_storage(self) -> None:
        store, index = self.make_index()
        try:
            doc_id = index.add_document(
                "alpha beta retrieval document",
                metadata={"title": "Alpha"},
            )
            self.assertEqual(index.info(), [
                {
                    "doc_id": doc_id,
                    "summary": "alpha beta retrieval document",
                    "metadata": {
                        "title": "Alpha",
                        "source": "unknown",
                        "created_at": index.info()[0]["metadata"]["created_at"],
                    },
                }
            ])

            results = index.search("alpha", top_k=1)
            self.assertEqual(results, ["alpha beta retrieval document"])

            index.delete(doc_id)
            self.assertEqual(index.info(), [])
            self.assertEqual(index.search("alpha"), [])

            index.add_document("clear me", metadata={})
            index.clear()
            self.assertEqual(index.info(), [])
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
