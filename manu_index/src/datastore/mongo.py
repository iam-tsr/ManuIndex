from __future__ import annotations

from typing import Any, Iterable

from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import OperationFailure, PyMongoError


class RouteMetaStore:
    def __init__(self, collection: Collection):
        self.collection = collection

    def insert_route_meta(
        self,
        doc_id: str,
        summary: str,
        summary_vector: list[float],
        metadata: dict[str, Any],
    ):
        try:
            result = self.collection.insert_one({
                "doc_id": doc_id,
                "summary": summary,
                "summary_vector": summary_vector,
                "metadata": metadata,
            })
            return result.inserted_id
        except OperationFailure as error:
            raise RuntimeError(f"Failed to insert route meta: {error}") from error

    def get_route_meta(self, query: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        try:
            return list(self.collection.find(query or {}))
        except OperationFailure as error:
            raise RuntimeError(f"Failed to retrieve route meta: {error}") from error

    def delete_route_meta(self, doc_id: str) -> None:
        try:
            self.collection.delete_many({"doc_id": doc_id})
        except OperationFailure as error:
            raise RuntimeError(f"Failed to delete route meta: {error}") from error

    def clear(self) -> None:
        try:
            self.collection.delete_many({})
        except OperationFailure as error:
            raise RuntimeError(f"Failed to clear route meta: {error}") from error


class DocumentIndexStore:
    def __init__(self, collection: Collection):
        self.collection = collection

    def insert_document_chunks(self, chunks: list[dict[str, Any]]) -> list[Any]:
        if not chunks:
            return []

        try:
            result = self.collection.insert_many(chunks)
            return result.inserted_ids
        except OperationFailure as error:
            raise RuntimeError(f"Failed to insert document chunks: {error}") from error

    def get_document_chunks(self, doc_ids: Iterable[str]) -> list[dict[str, Any]]:
        ids = list(doc_ids)
        if not ids:
            return []

        try:
            return list(
                self.collection
                .find({"doc_id": {"$in": ids}})
                .sort([("doc_id", ASCENDING), ("chunk_index", ASCENDING)])
            )
        except OperationFailure as error:
            raise RuntimeError(f"Failed to retrieve document chunks: {error}") from error

    def delete_document_chunks(self, doc_id: str) -> None:
        try:
            self.collection.delete_many({"doc_id": doc_id})
        except OperationFailure as error:
            raise RuntimeError(f"Failed to delete document chunks: {error}") from error

    def clear(self) -> None:
        try:
            self.collection.delete_many({})
        except OperationFailure as error:
            raise RuntimeError(f"Failed to clear document chunks: {error}") from error


class MongoDBHandler:
    def __init__(
        self,
        connection_string: str,
        database_name: str = "manu_index_db",
        route_meta_collection: str = "routeMeta",
        document_index_collection: str = "documentIndex",
        server_selection_timeout_ms: int = 5000,
    ):
        self.connection_string = connection_string
        self.database_name = database_name
        self.client = MongoClient(
            self.connection_string,
            serverSelectionTimeoutMS=server_selection_timeout_ms,
        )
        self.route_meta_collection = route_meta_collection
        self.document_index_collection = document_index_collection

        try:
            self.client.admin.command("ping")
        except PyMongoError as error:
            raise RuntimeError(f"Failed to connect to MongoDB: {error}") from error

        self.db = self.client[self.database_name]
        self.route_meta = RouteMetaStore(collection=self.db[self.route_meta_collection])
        self.document_index = DocumentIndexStore(collection=self.db[self.document_index_collection])
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.route_meta.collection.create_index("doc_id")
        self.document_index.collection.create_index([
            ("doc_id", ASCENDING),
            ("chunk_index", ASCENDING),
        ])
