"""Dual FAISS vector engine.

Index #1: GhostFaceNet 512D  (Mode 1 continuous tracking)
Index #2: ArcFace 512D       (Mode 2 attendance, cosine threshold)

Embeddings are stored L2-normalized, so inner product == cosine similarity.

Crash-safety: indices are rebuilt entirely from MongoDB at boot
(sync_from_db) — MongoDB is the single source of truth; no disk checkpoints.
"""

import numpy as np
import faiss

EMBEDDING_DIM = 512
DEFAULT_THRESHOLD_MODE2 = 0.65


class DualVectorEngine:
    def __init__(self, storage):
        self._storage = storage
        self._dim = EMBEDDING_DIM
        self.index1 = self._new_index()
        self.index2 = self._new_index()
        self._rows1: list[str] = []
        self._rows2: list[str] = []
        self._meta: dict[str, str] = {}
        self.sync_from_db()

    def _new_index(self):
        """GPU IndexFlatIP when CUDA is available, otherwise CPU (guaranteed fallback)."""
        try:
            if faiss.get_num_gpus() > 0:
                return faiss.index_cpu_to_all_gpus(faiss.IndexFlatIP(self._dim))
        except Exception:
            pass
        return faiss.IndexFlatIP(self._dim)

    @staticmethod
    def _normalize(vec) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(arr)
        return arr

    def sync_from_db(self) -> None:
        """Rebuild both indices entirely from MongoDB (batched adds)."""
        self.index1 = self._new_index()
        self.index2 = self._new_index()
        self._rows1 = []
        self._rows2 = []
        self._meta = {}
        users = self._storage.load_all()
        if not users:
            return
        ghosts = np.empty((len(users), self._dim), dtype=np.float32)
        arcs = np.empty((len(users), self._dim), dtype=np.float32)
        for i, user in enumerate(users):
            ghosts[i] = user["ghost_vector"]
            arcs[i] = user["arcface_vector"]
            self._rows1.append(user["user_id"])
            self._rows2.append(user["user_id"])
            self._meta[user["user_id"]] = user["name"]
        faiss.normalize_L2(ghosts)
        faiss.normalize_L2(arcs)
        self.index1.add(ghosts)
        self.index2.add(arcs)

    def enroll_user(self, user_id: str, name: str, ghost_vec, arcface_vec) -> None:
        if user_id in self._meta:
            self.sync_from_db()  # re-enrollment: drop stale index rows first
        self._storage.enroll(user_id, name, ghost_vec, arcface_vec)
        ghost = self._normalize(ghost_vec)
        arcface = self._normalize(arcface_vec)
        self.index1.add(ghost)
        self.index2.add(arcface)
        self._rows1.append(user_id)
        self._rows2.append(user_id)
        self._meta[user_id] = name

    def delete_user(self, user_id: str) -> bool:
        if not self._storage.delete(user_id):
            return False
        self.sync_from_db()
        return user_id not in self._meta

    def search_mode1(self, query_vec, top_k: int) -> list[tuple[str, str, float]]:
        q = self._normalize(query_vec)
        if self.index1.ntotal == 0:
            return []
        scores, idx = self.index1.search(q, min(top_k, self.index1.ntotal))
        return [
            (self._rows1[i], self._meta[self._rows1[i]], float(scores[0][j]))
            for j, i in enumerate(idx[0])
        ]

    def search_mode2(self, query_vec, top_k: int, threshold: float = DEFAULT_THRESHOLD_MODE2):
        q = self._normalize(query_vec)
        if self.index2.ntotal == 0:
            return None
        scores, idx = self.index2.search(q, min(top_k, self.index2.ntotal))
        results = [
            (self._rows2[i], self._meta[self._rows2[i]], float(scores[0][j]))
            for j, i in enumerate(idx[0])
            if scores[0][j] >= threshold
        ]
        return results or None