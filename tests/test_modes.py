"""Phase 4 verification: gesture state machine + dual-mode pipelines."""

import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from core.gesture import (  # noqa: E402
    GestureStateMachine,
    detect_gesture,
    OPEN_PALM,
    THUMBS_UP,
)
from core.tracker import IoUTracker  # noqa: E402
from core.database import create_storage_backend  # noqa: E402
from core.vector_engine import DualVectorEngine, EMBEDDING_DIM  # noqa: E402
from pipelines.mode_tracking import Mode1Pipeline  # noqa: E402
from pipelines.mode_attendance import Mode2Pipeline, STATE_SUCCESS, STATE_ALIGN, LOG_MODE  # noqa: E402


# ---------------------------------------------------------------- landmarks

def _palm_landmarks() -> np.ndarray:
    """Open palm: all four fingers extended + thumb extended."""
    lms = np.zeros((21, 3), dtype=np.float32)
    lms[0] = (0.50, 0.90, 0.0)
    lms[1] = (0.34, 0.88, 0.0); lms[2] = (0.30, 0.78, 0.0)
    lms[3] = (0.29, 0.70, 0.0); lms[4] = (0.30, 0.60, 0.0)
    for base, tip_y in ((5, 0.40), (9, 0.36), (13, 0.41), (17, 0.47)):
        mcp_x = 0.44 + (base - 5) * 0.06
        lms[base] = (mcp_x, 0.72, 0.0)
        lms[base + 1] = (mcp_x, 0.60, 0.0)
        lms[base + 2] = (mcp_x, 0.50, 0.0)
        lms[base + 3] = (mcp_x, tip_y, 0.0)
    return lms


def _thumbs_up_landmarks() -> np.ndarray:
    """Thumbs up: thumb extended, four fingers curled."""
    lms = np.zeros((21, 3), dtype=np.float32)
    lms[0] = (0.50, 0.90, 0.0)
    lms[1] = (0.34, 0.88, 0.0); lms[2] = (0.30, 0.78, 0.0)
    lms[3] = (0.29, 0.70, 0.0); lms[4] = (0.30, 0.60, 0.0)
    for base in (5, 9, 13, 17):
        mcp_x = 0.44 + (base - 5) * 0.06
        lms[base] = (mcp_x, 0.70, 0.0)
        lms[base + 1] = (mcp_x, 0.67, 0.0)
        lms[base + 2] = (mcp_x, 0.68, 0.0)
        lms[base + 3] = (mcp_x, 0.68, 0.0)
    return lms


def _fist_landmarks() -> np.ndarray:
    """Fist: nothing extended."""
    lms = np.zeros((21, 3), dtype=np.float32)
    lms[0] = (0.50, 0.90, 0.0)
    lms[1] = (0.34, 0.88, 0.0); lms[2] = (0.34, 0.84, 0.0)
    lms[3] = (0.34, 0.81, 0.0); lms[4] = (0.34, 0.82, 0.0)
    for base in (5, 9, 13, 17):
        mcp_x = 0.44 + (base - 5) * 0.06
        lms[base] = (mcp_x, 0.70, 0.0)
        lms[base + 1] = (mcp_x, 0.67, 0.0)
        lms[base + 2] = (mcp_x, 0.68, 0.0)
        lms[base + 3] = (mcp_x, 0.68, 0.0)
    return lms


# ------------------------------------------------------------ gesture machine

def test_gesture_detection():
    assert detect_gesture(_palm_landmarks()) == OPEN_PALM
    assert detect_gesture(_thumbs_up_landmarks()) == THUMBS_UP
    assert detect_gesture(_fist_landmarks()) is None


def test_gesture_requires_five_sustained_frames():
    m = GestureStateMachine(require_frames=5)
    palm = _palm_landmarks()
    confirms = [m.update(palm) for _ in range(5)]
    assert confirms[:4] == [False] * 4
    assert confirms[4] is True  # confirmed only on the 5th sustained frame


def test_gesture_resets_on_interruption():
    m = GestureStateMachine(require_frames=5)
    palm = _palm_landmarks()
    fist = _fist_landmarks()
    for _ in range(3):
        assert m.update(palm) is False
    assert m.update(fist) is False  # interruption resets the counter
    assert m.progress == 0
    for _ in range(4):
        assert m.update(palm) is False  # restart from 0: no confirm within 4
    assert m.update(palm) is True


# ------------------------------------------------------------------- Mode 1

class FakeDetector:
    def __init__(self, dets):
        self._dets = dets

    def detect(self, frame):
        return list(self._dets)


class CountingExtractor:
    def __init__(self, vec):
        self._vec = vec
        self.calls = 0

    def extract(self, crop):
        self.calls += 1
        return self._vec


class FakeEngine:
    def __init__(self, top=None):
        self._top = top or []
        self.queries = 0

    def search_mode1(self, vec, top_k=1):
        self.queries += 1
        return list(self._top)


def test_mode1_caches_identity_and_skips_reinference():
    det = [0.10, 0.20, 0.30, 0.40, 0.90]
    extractor = CountingExtractor(np.zeros((1, EMBEDDING_DIM), dtype=np.float32))
    engine = FakeEngine([("EMP_0001", "Alice", 0.99)])
    pipeline = Mode1Pipeline(FakeDetector([det]), extractor, engine, IoUTracker())
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    overlay1, tel1 = pipeline.process(frame)
    assert tel1["embeddings_computed"] == 1
    assert extractor.calls == 1 and engine.queries == 1

    overlay2, tel2 = pipeline.process(frame)  # IoU match -> cached identity
    assert tel2["embeddings_computed"] == 0
    assert extractor.calls == 1 and engine.queries == 1  # no re-inference
    assert overlay2.shape == frame.shape
    tracks = pipeline._tracker.tracks
    assert tracks and tracks[0].identity == ("EMP_0001", "Alice")


# ------------------------------------------------------------------- Mode 2

class FakeGesture:
    def __init__(self, confirm: bool):
        self._confirm = confirm
        self.progress = 5

    def update(self, hand):
        return self._confirm

    def reset(self):
        self.progress = 0


GUIDE_ZONE = [0.2, 0.2, 0.6, 0.6]
DET_IN_ZONE = [0.30, 0.30, 0.50, 0.50, 0.90]
DET_OUT_ZONE = [0.85, 0.85, 0.95, 0.95, 0.90]


def _clean():
    storage = create_storage_backend()
    storage._db["users"].delete_many({})
    storage._db["attendance_logs"].delete_many({})
    return storage


def _unit(rng):
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _pipeline(confirm, arcface_vec, dets, on_confirmed=None):
    storage = _clean()
    engine = DualVectorEngine(storage)
    rng = np.random.default_rng(3)
    engine.enroll_user("EMP_0001", "Alice", _unit(rng), _unit(rng))

    class FakeHands:
        def track(self, frame_rgb):
            return [np.zeros((21, 3), dtype=np.float32)]

    class FakeArcFace:
        def __init__(self, vec):
            self._vec = vec

        def extract(self, crop):
            return self._vec

    return Mode2Pipeline(
        FakeDetector(dets), FakeHands(), FakeGesture(confirm),
        FakeArcFace(arcface_vec), engine, storage,
        guide_zone=GUIDE_ZONE, threshold=0.65, on_confirmed=on_confirmed,
    ), storage


def test_mode2_logs_only_on_zone_gesture_and_match():
    storage = _clean()
    rng = np.random.default_rng(3)
    engine = DualVectorEngine(storage)
    engine.enroll_user("EMP_0001", "Alice", _unit(rng), _unit(rng))
    enrolled = np.asarray(storage._db["users"].find_one({"user_id": "EMP_0001"})["arcface_vector"],
                          dtype=np.float32)

    calls = []
    p, storage = _pipeline(True, enrolled, [DET_IN_ZONE], on_confirmed=lambda u, n, c: calls.append((u, n, c)))
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    overlay, status = p.process(frame)
    assert status["state"] == STATE_SUCCESS
    assert storage._db["attendance_logs"].count_documents({}) == 1
    record = storage._db["attendance_logs"].find_one({})
    assert record["user_id"] == "EMP_0001"
    assert record["name"] == "Alice"
    assert record["mode"] == LOG_MODE
    assert isinstance(record["confidence"], float) and record["confidence"] >= 0.65
    import datetime

    datetime.datetime.fromisoformat(record["timestamp"])  # valid UTC ISO-8601
    assert calls == [("EMP_0001", "Alice", record["confidence"])]
    assert overlay.shape == frame.shape


def test_mode2_ignores_faces_outside_guide_zone():
    p, storage = _pipeline(True, np.zeros((1, EMBEDDING_DIM), dtype=np.float32), [DET_OUT_ZONE])
    overlay, status = p.process(np.zeros((200, 200, 3), dtype=np.uint8))
    assert status["state"] == STATE_ALIGN
    assert storage._db["attendance_logs"].count_documents({}) == 0


def test_mode2_requires_gesture_confirmation():
    rng = np.random.default_rng(3)
    storage = _clean()
    engine = DualVectorEngine(storage)
    engine.enroll_user("EMP_0001", "Alice", _unit(rng), _unit(rng))
    enrolled = np.asarray(storage._db["users"].find_one({"user_id": "EMP_0001"})["arcface_vector"],
                          dtype=np.float32)
    p, storage = _pipeline(False, enrolled, [DET_IN_ZONE])
    overlay, status = p.process(np.zeros((200, 200, 3), dtype=np.uint8))
    assert status["state"] == "gesture"
    assert storage._db["attendance_logs"].count_documents({}) == 0


def test_mode2_no_log_on_biometric_miss():
    rng = np.random.default_rng(7)
    p, storage = _pipeline(True, _unit(rng), [DET_IN_ZONE])  # random != enrolled
    overlay, status = p.process(np.zeros((200, 200, 3), dtype=np.uint8))
    assert status["state"] == STATE_ALIGN
    assert storage._db["attendance_logs"].count_documents({}) == 0