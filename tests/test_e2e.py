"""Phase 6 verification: E2E benchmarks, 100k scale, crash recovery, packaging."""

import os
import re
import select
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALE = 100_000
DIM = 512

from core.database import create_storage_backend  # noqa: E402
from core.vector_engine import DualVectorEngine  # noqa: E402


def _clean():
    storage = create_storage_backend()
    storage._db["users"].delete_many({})
    storage._db["attendance_logs"].delete_many({})
    return storage


def _unit_vecs(rng, n):
    v = rng.standard_normal((n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ------------------------------------------------------------- 100k scale

def test_100k_scale_latency_and_zero_far():
    storage = _clean()
    rng = np.random.default_rng(11)
    # Bulk-populate MongoDB (single source of truth) in blocks.
    users = storage._db["users"]
    batch = []
    for block in range(SCALE // 2000):
        ghosts = _unit_vecs(rng, 2000)
        arcs = _unit_vecs(rng, 2000)
        for i in range(2000):
            j = block * 2000 + i
            batch.append({
                "user_id": f"SCL_{j:05d}",
                "name": f"Scale Identity {j}",
                "ghost_vector": ghosts[i].tolist(),
                "arcface_vector": arcs[i].tolist(),
                "created_at": "2026-08-17T00:00:00+00:00",
            })
            if len(batch) == 2000:
                users.insert_many(batch)
                batch = []
    if batch:
        users.insert_many(batch)
    assert users.count_documents({}) == SCALE

    engine = DualVectorEngine(storage)  # boot-time rebuild from MongoDB
    assert engine.index1.ntotal == SCALE
    assert engine.index2.ntotal == SCALE

    probe = _unit_vecs(rng, 1)[0]
    for search in (engine.search_mode1, engine.search_mode2):
        search(probe, top_k=1)
        t0 = time.perf_counter()
        for _ in range(20):
            search(probe, top_k=1)
        mean_ms = (time.perf_counter() - t0) / 20 * 1000.0
        # Revised bound (user-approved): exact IndexFlatIP scales linearly;
        # measured GPU ~1.2 ms at 100k (10k bound of 0.8 ms preserved in Phase 2).
        assert mean_ms <= 2.5, f"GPU latency {mean_ms:.3f} ms > 2.5 ms"

    # CPU bound for the same workload (measured ~15.7 ms at 100k).
    import faiss

    cpu_index = faiss.IndexFlatIP(DIM)
    ghost = np.asarray([u["ghost_vector"] for u in users.find({}, {"ghost_vector": 1})],
                       dtype=np.float32)
    cpu_index.add(ghost)
    q = np.asarray(probe, dtype=np.float32).reshape(1, -1)
    cpu_index.search(q, 1)
    t0 = time.perf_counter()
    for _ in range(10):
        cpu_index.search(q, 1)
    mean_ms = (time.perf_counter() - t0) / 10 * 1000.0
    assert mean_ms <= 20.0, f"CPU latency {mean_ms:.3f} ms > 20 ms"

    # Mode 2 false-acceptance rate: negative probes must never pass >= 0.65.
    for _ in range(50):
        assert engine.search_mode2(_unit_vecs(rng, 1)[0], top_k=1) is None

    storage._db["users"].delete_many({})  # leave a clean vault for other tests


# ------------------------------------------------------------- benchmarks

def _parse_bench(output: str) -> dict:
    return {m.group(1): float(m.group(2))
            for m in re.finditer(r"BENCH (\w+)=([\d.]+)", output)}


def test_benchmark_gpu_acceptance():
    env = dict(os.environ)
    proc = subprocess.run(
        [sys.executable, "scripts/benchmark_pipeline.py",
         "--source", "virtual", "--seconds", "5", "--mode", "1"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    bench = _parse_bench(proc.stdout)
    assert bench["fps"] >= 30.0, f"CUDA fps {bench['fps']} < 30"
    assert bench["peak_rss_mb"] <= 2500.0, "peak RAM > 2.5 GB"
    assert "CUDAExecutionProvider" in proc.stdout


def test_benchmark_cpu_fallback_acceptance():
    env = dict(os.environ)
    proc = subprocess.run(
        [sys.executable, "scripts/benchmark_pipeline.py",
         "--source", "virtual", "--seconds", "5", "--mode", "1", "--cpu-only"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    bench = _parse_bench(proc.stdout)
    assert bench["fps"] >= 15.0, f"CPU fps {bench['fps']} < 15"
    assert bench["peak_rss_mb"] <= 2500.0


# --------------------------------------------------------- crash & signals

def _spawn_app(frames: int = 0):
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    return subprocess.Popen(
        [sys.executable, "main.py", "--test-mode", "--frames", str(frames)],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )


def _wait_for(proc, marker: str, timeout: float = 120) -> str:
    deadline = time.monotonic() + timeout
    collected = []
    while time.monotonic() < deadline:
        ready, _, _ = select.select([proc.stdout], [], [], 1.0)
        if ready:
            line = proc.stdout.readline()
            if not line:
                break
            collected.append(line)
            if marker in line:
                return "".join(collected)
        if proc.poll() is not None:
            break
    raise TimeoutError(f"marker {marker!r} not seen; got:\n{''.join(collected[-40:])}")


def test_crash_recovery_zero_data_loss():
    storage = _clean()
    rng = np.random.default_rng(5)
    engine = DualVectorEngine(storage)
    vectors = {}
    for i in range(200):
        uid = f"CRS_{i:03d}"
        ghost = _unit_vecs(rng, 1)[0]
        vectors[uid] = ghost
        engine.enroll_user(uid, f"Crash {i}", ghost, _unit_vecs(rng, 1)[0])

    proc = _spawn_app()
    _wait_for(proc, "AIWorkerThread ready")
    time.sleep(2.0)  # let it run with live frames
    proc.kill()  # SIGKILL: no cleanup possible
    proc.wait(timeout=10)

    engine2 = DualVectorEngine(create_storage_backend())  # rebuild from Mongo
    assert engine2.index1.ntotal == 200
    assert engine2.index2.ntotal == 200
    for uid, ghost in list(vectors.items())[:10]:
        top = engine2.search_mode1(ghost, top_k=1)
        assert top and top[0][0] == uid, f"zero-loss violation for {uid}"


def test_sigterm_clean_shutdown():
    storage = _clean()
    proc = _spawn_app()
    _wait_for(proc, "AIWorkerThread ready")
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 0, f"exit {proc.returncode}: {out[-2000:]}"
    assert "SHUTDOWN_COMPLETE" in out


# -------------------------------------------------------------- log integrity

def test_mode1_writes_no_attendance_logs():
    from pipelines.mode_tracking import Mode1Pipeline
    from core.tracker import IoUTracker

    storage = _clean()

    class FakeDet:
        def detect(self, frame):
            return [[0.1, 0.2, 0.3, 0.4, 0.9]]

    class FakeExt:
        def extract(self, crop):
            return np.zeros((1, DIM), dtype=np.float32)

    class FakeEng:
        def search_mode1(self, vec, top_k=1):
            return [("EMP_0001", "Alice", 0.99)]

    pipeline = Mode1Pipeline(FakeDet(), FakeExt(), FakeEng(), IoUTracker())
    pipeline.process(np.zeros((200, 200, 3), dtype=np.uint8))
    assert storage._db["attendance_logs"].count_documents({}) == 0


# ------------------------------------------------------------- packaging

BINARY = os.path.join(ROOT, "dist/harecognition/harecognition")


@pytest.mark.skipif(not os.path.exists(BINARY),
                    reason="PyInstaller binary not built (run scripts/package.sh)")
def test_frozen_binary_smoke():
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    proc = subprocess.run(
        [BINARY, "--test-mode", "--frames", "60"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, (
        f"exit {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "SHUTDOWN_COMPLETE" in proc.stdout + proc.stderr