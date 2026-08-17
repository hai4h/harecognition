"""Storage backend: MongoDB via pymongo. MongoDB is the single source of truth;
FAISS in-memory indices are rebuilt from it at boot (see core/vector_engine.py)."""

from __future__ import annotations

import os
import datetime
from abc import ABC, abstractmethod

import pymongo
import yaml

from core.paths import ROOT


def load_app_config() -> dict:
    with open(os.path.join(ROOT, "configs/app_config.yaml")) as f:
        return yaml.safe_load(f)


class StorageBackend(ABC):
    @abstractmethod
    def enroll(self, user_id: str, name: str, ghost_vector, arcface_vector) -> None:
        ...

    @abstractmethod
    def delete(self, user_id: str) -> bool:
        ...

    @abstractmethod
    def list(self) -> list[dict]:
        ...

    @abstractmethod
    def log_attendance(self, user_id: str, name: str, mode: int, confidence: float) -> None:
        ...

    @abstractmethod
    def load_all(self) -> list[dict]:
        ...


class MongoStorageBackend(StorageBackend):
    def __init__(self, uri: str, db_name: str):
        self._client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
        self._client.admin.command("ping")
        self._db = self._client[db_name]
        self._users = self._db["users"]
        self._logs = self._db["attendance_logs"]
        self._users.create_index("user_id", unique=True)
        self._logs.create_index("timestamp")

    def enroll(self, user_id: str, name: str, ghost_vector, arcface_vector) -> None:
        doc = {
            "user_id": user_id,
            "name": name,
            "ghost_vector": [float(v) for v in ghost_vector],
            "arcface_vector": [float(v) for v in arcface_vector],
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        self._users.replace_one({"user_id": user_id}, doc, upsert=True)

    def delete(self, user_id: str) -> bool:
        res = self._users.delete_one({"user_id": user_id})
        return res.deleted_count > 0

    def list(self) -> list[dict]:
        return list(self._users.find({}, {"_id": 0}))

    def load_all(self) -> list[dict]:
        return self.list()

    def log_attendance(self, user_id: str, name: str, mode, confidence: float) -> None:
        self._logs.insert_one({
            "user_id": user_id,
            "name": name,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "mode": mode,
            "confidence": float(confidence),
        })


def create_storage_backend(cfg: dict | None = None) -> StorageBackend:
    """Backend stays swappable via configuration; default is MongoDB."""
    cfg = cfg or load_app_config()
    backend_type = cfg.get("storage", {}).get("backend", "mongo")
    if backend_type != "mongo":
        raise ValueError(f"Unsupported storage backend: {backend_type}")
    return MongoStorageBackend(cfg["mongodb_uri"], cfg["mongodb_db"])