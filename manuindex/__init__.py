import os

from .src.index import ManuIndex
from .src.datastore.mongo import MongoDBHandler

__version__ = os.environ.get("MANUINDEX_VERSION", "0.1.0")

__all__ = [
    "ManuIndex",
    "MongoDBHandler",
]