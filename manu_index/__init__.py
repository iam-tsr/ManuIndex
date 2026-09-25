import os

from .src.datastore import DataStore, LocalDataStore
from .src.index import ManuIndex

__version__ = os.environ.get("MANUINDEX_VERSION", "0.1.0")

__all__ = [
    "DataStore",
    "LocalDataStore",
    "ManuIndex",
]
