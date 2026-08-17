"""Phase 2 verification: MongoDB vault + dual FAISS vector engine."""

import os
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from core.database import create_storage_backend, MongoStorageBackend  # noqa: E402
from core.vector_engine import DualVectorEngine, EMBEDDING_DIM  # noqa: E402

N_IDENTITIES = 10_000


def _clean_db() -> MongoStorageBackend:
    storage = create_storage_backend()
    storage._db["users"].delete_many({})
    storage._db["attendance_logs"].delete_many({})
    return storage


def _unit_vec(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def test_enroll_10000_metadata_and_benchmark():
    storage = _clean_db()
    engine = DualVectorEngine(storage)
    assert engine.index1.ntotal == 0 and engine.index2.ntotal == 0

    rng = np.random.default_rng(42)
    vectors = {}
    for i in range(N_IDENTITIES):
        uid = f"EMP_{i:04d}"
        ghost = _unit_vec(rng)
        arcface = _unit_vec(rng)
        vectors[uid] = (ghost, arcface)
        engine.enroll_user(uid, f"Identity {i}", ghost, arcface)

    assert engine.index1.ntotal == N_IDENTITIES
    assert engine.index2.ntotal == N_IDENTITIES
    assert storage._db["users"].count_documents({}) == N_IDENTITIES

    # Benchmark: query latency must be <= 0.8 ms on CPU (both indices).
    queries = 200
    for search in (engine.search_mode1, engine.search_mode2):
        probe = _unit_vec(np.random.default_rng(7))
        t0 = time.perf_counter()
        for _ in range(queries):
            search(probe, top_k=1)
        mean_ms = (time.perf_counter() - t0) / queries * 1000.0
        assert mean_ms <= 0.8, f"query latency {mean_ms:.3f} ms exceeds 0.8 ms"

    # Metadata verification: exact-vector probes must return themselves.
    for uid, (ghost, arcface) in list(vectors.items())[:50]:
        top1 = engine.search_mode1(ghost, top_k=1)[0]
        assert top1[0] == uid, f"mode1 top-1 mismatch: {top1[0]} != {uid}"
        assert top1[1] == f"Identity {int(uid.split('_')[1])}"
        assert top1[2] > 0.99
        top2 = engine.search_mode2(arcface, top_k=1)
        assert top2 and top2[0][0] == uid

    # Mode 2 threshold: dissimilar probe must return None.
    far = _unit_vec(np.random.default_rng(99))
    assert engine.search_mode2(far, top_k=3) is None

    # Attendance logging round-trip.
    storage.log_attendance("EMP_0000", "Identity 0", mode=2, confidence=0.97)
    log = storage._db["attendance_logs"].find_one({"user_id": "EMP_0000"})
    assert log and log["mode"] == 2 and log["confidence"] == 0.97 and "timestamp" in log

    # Crash-safety: a fresh engine must rebuild both indices from MongoDB.
    engine2 = DualVectorEngine(create_storage_backend())
    assert engine2.index1.ntotal == N_IDENTITIES
    assert engine2.index2.ntotal == N_IDENTITIES
    uid, (ghost, _) = list(vectors.items())[0]
    assert engine2.search_mode1(ghost, top_k=1)[0][0] == uid


def test_delete_purges_everywhere():
    storage = _clean_db()
    engine = DualVectorEngine(storage)
    rng = np.random.default_rng(1)
    for i in range(3):
        engine.enroll_user(f"DEL_{i}", f"User {i}", _unit_vec(rng), _unit_vec(rng))
    victim = "DEL_1"
    victim_ghost = np.asarray(
        storage._db["users"].find_one({"user_id": victim})["ghost_vector"],
        dtype=np.float32,
    )

    assert engine.delete_user(victim) is True
    assert victim not in engine._meta
    assert engine.index1.ntotal == 2 and engine.index2.ntotal == 2
    assert storage._db["users"].count_documents({"user_id": victim}) == 0
    res = engine.search_mode2(victim_ghost, top_k=1)
    assert res is None or all(r[0] != victim for r in res)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])