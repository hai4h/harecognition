"""Phase 3 verification: model pipeline (detector, embeddings, tracker)."""

import os

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "tests/fixtures/group_photo.jpg")

from pipelines.inference_manager import (  # noqa: E402
    FaceDetector,
    EmbeddingExtractor,
    HandTracker,
    get_optimized_session,
    load_model_config,
)
from core.tracker import IoUTracker  # noqa: E402

CFG = load_model_config()


@pytest.fixture(scope="module")
def sessions():
    return {
        name: get_optimized_session(CFG["models"][name]["path"], CFG)
        for name in ("yolo_face", "ghostfacenet_512", "arcface_512")
    }


@pytest.fixture(scope="module")
def frame():
    img = cv2.imread(FIXTURE)
    assert img is not None, f"fixture missing: {FIXTURE}"
    return img


@pytest.fixture(scope="module")
def face_crop(sessions, frame):
    det = FaceDetector(sessions["yolo_face"], conf_threshold=0.25)
    dets = det.detect(frame)
    assert dets, "no face detected in fixture"
    x1, y1, x2, y2, _ = dets[0]
    h, w = frame.shape[:2]
    return frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]


def test_face_detector_bbox_shape_and_range(sessions, frame):
    det = FaceDetector(sessions["yolo_face"], conf_threshold=0.25)
    dets = det.detect(frame)
    assert len(dets) >= 1
    for d in dets:
        assert len(d) == 5
        x1, y1, x2, y2, conf = d
        assert 0.0 <= x1 < x2 <= 1.0
        assert 0.0 <= y1 < y2 <= 1.0
        assert 0.0 <= conf <= 1.0


@pytest.mark.parametrize("model", ["ghostfacenet_512", "arcface_512"])
def test_embedding_shape_and_unit_norm(sessions, face_crop, model):
    extractor = EmbeddingExtractor(sessions[model])
    vec = extractor.extract(face_crop)
    assert vec.shape == (1, 512)
    assert vec.dtype == np.float32
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5


def test_optimized_session_respects_provider_priority():
    session = get_optimized_session(CFG["models"]["yolo_face"]["path"], CFG)
    providers = session.get_providers()
    assert providers[0] == "CUDAExecutionProvider"  # CUDA present on this host
    assert "CPUExecutionProvider" in providers


def test_cpu_fallback_when_cuda_unavailable(monkeypatch):
    import onnxruntime as ort

    real = ort.InferenceSession

    def fake(path, providers=None, **kwargs):
        names = [p if isinstance(p, str) else p[0] for p in (providers or [])]
        if "CUDAExecutionProvider" in names:
            raise RuntimeError("simulated CUDA failure")
        return real(path, providers=names, **kwargs)

    monkeypatch.setattr(ort, "InferenceSession", fake)
    session = get_optimized_session(CFG["models"]["yolo_face"]["path"], CFG)
    assert session.get_providers() == ["CPUExecutionProvider"]


def test_iou_tracker_stable_track_id():
    tracker = IoUTracker()
    det = [0.10, 0.20, 0.30, 0.40, 0.90]
    shifted = [0.11, 0.21, 0.31, 0.41, 0.90]  # high IoU -> same track
    far = [0.80, 0.80, 0.90, 0.90, 0.90]

    t1 = tracker.update([det])
    assert [t.track_id for t in t1] == [1]
    t2 = tracker.update([shifted])
    assert [t.track_id for t in t2] == [1]  # stable across frames
    t3 = tracker.update([shifted, far])
    assert sorted(t.track_id for t in t3) == [1, 2]  # far detection -> new track


def test_iou_tracker_prunes_lost_tracks():
    tracker = IoUTracker(max_lost=2)
    tracker.update([[0.1, 0.2, 0.3, 0.4, 0.9]])
    assert len(tracker.update([])) == 1
    assert len(tracker.update([])) == 1
    assert len(tracker.update([])) == 0  # pruned after max_lost frames


def test_cached_track_bypasses_embedding():
    tracker = IoUTracker()
    det = [0.10, 0.20, 0.30, 0.40, 0.90]
    shifted = [0.11, 0.21, 0.31, 0.41, 0.90]

    first = tracker.update([det], identities=[("EMP_0001", "Alice")])
    assert first[0].identity == ("EMP_0001", "Alice")
    assert first[0].needs_embedding is False  # matched -> cached

    second = tracker.update([shifted])
    assert second[0].needs_embedding is False  # cached track: bypass embedding

    third = tracker.update([[0.70, 0.70, 0.80, 0.80, 0.90]])
    new = [t for t in third if t.track_id not in {first[0].track_id}]
    assert new and new[0].needs_embedding is True  # unseen face needs extraction


def test_hand_tracker_returns_empty_on_face_image(frame):
    ht = HandTracker()
    hands = ht.track(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    assert hands == []  # no hands in fixture
    ht.close()