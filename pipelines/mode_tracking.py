"""Mode 1 - Continuous Tracking & Identification.

Full-frame YOLO Face detection -> IoU matching against active tracks
(IoU > 0.3). Matched tracks reuse the cached identity (bypassing BOTH
embedding extraction and FAISS retrieval). New faces get a GhostFaceNet
512D embedding, FAISS Index #1 query, and the name is cached on the track.
"""

import logging

import cv2
import numpy as np

log = logging.getLogger("mode_tracking")


def _crop_roi(frame: np.ndarray, det: list[float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v * s)) for v, s in zip(det[:4], (w, h, w, h)))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame[y1:y1 + 1, x1:x1 + 1]
    return frame[y1:y2, x1:x2]


class Mode1Pipeline:
    def __init__(self, detector, ghost_extractor, engine, tracker,
                 min_conf: float = 0.25):
        self._detector = detector
        self._extractor = ghost_extractor
        self._engine = engine
        self._tracker = tracker
        self._min_conf = min_conf

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        dets = [d for d in self._detector.detect(frame) if d[4] >= self._min_conf]
        identities: list[tuple[str, str] | None] = []
        embeddings_computed = 0
        for det in dets:
            track = self._tracker.match_detection(det)
            if track is not None and track.identity is not None:
                identities.append(track.identity)  # cached: no re-inference
                continue
            crop = _crop_roi(frame, det)
            vec = self._extractor.extract(crop)
            embeddings_computed += 1
            top = self._engine.search_mode1(vec, top_k=1)
            identities.append((top[0][0], top[0][1]) if top else None)

        tracks = self._tracker.update(dets, identities)
        overlay = self._annotate(frame.copy(), tracks)
        telemetry = {
            "faces": len(dets),
            "tracks": len(tracks),
            "embeddings_computed": embeddings_computed,
        }
        return overlay, telemetry

    @staticmethod
    def _annotate(frame: np.ndarray, tracks) -> np.ndarray:
        h, w = frame.shape[:2]
        for t in tracks:
            x1, y1, x2, y2 = (int(round(v * s)) for v, s in zip(t.bbox, (w, h, w, h)))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            name = t.identity[1] if t.identity else "unknown"
            cv2.putText(frame, f"#{t.track_id} {name}", (x1, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame