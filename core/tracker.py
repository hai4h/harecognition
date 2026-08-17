"""Lightweight IoU bounding-box tracker (per DESCRIPTION.md Section 3).

Matches new detections to active tracks when IoU > 0.3. Each track caches the
matched identity: while a track is active, subsequent frames bypass BOTH
feature extraction and FAISS retrieval (needs_embedding is False). Lost
tracks are pruned after max_lost frames.
"""

import logging

import numpy as np

log = logging.getLogger("tracker")


class Track:
    def __init__(self, track_id: int, bbox: list[float], conf: float):
        self.track_id = track_id
        self.bbox = bbox
        self.conf = conf
        self.identity = None      # (user_id, name) once matched
        self.frames_since_update = 0
        self.matches = 0

    @property
    def needs_embedding(self) -> bool:
        return self.identity is None

    def __repr__(self) -> str:
        return f"Track({self.track_id}, bbox={self.bbox}, id={self.identity})"


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 30):
        self._iou_threshold = iou_threshold
        self._max_lost = max_lost
        self._tracks: list[Track] = []
        self._next_id = 1

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks)

    def match_detection(self, det: list[float]) -> Track | None:
        """Best track with IoU > threshold for this detection, else None."""
        best_iou, best_t = 0.0, None
        for t in self._tracks:
            iou = _iou(det[:4], t.bbox)
            if iou > best_iou:
                best_iou, best_t = iou, t
        return best_t if (best_t is not None and best_iou > self._iou_threshold) else None

    def update(self, detections: list[list[float]],
               identities: list[tuple[str, str] | None] | None = None) -> list[Track]:
        """Match detections [x1,y1,x2,y2,conf] to active tracks. Returns active
        tracks; `identities` (parallel to detections) carry match results for
        new/updated tracks."""
        identities = identities or [None] * len(detections)
        existing = list(self._tracks)
        matched = [False] * len(existing)
        for det, identity in zip(detections, identities):
            best_iou, best_t = 0.0, None
            for t in existing:
                iou = _iou(det[:4], t.bbox)
                if iou > best_iou:
                    best_iou, best_t = iou, t
            if best_t is not None and best_iou > self._iou_threshold:
                best_t.bbox = det[:4]
                best_t.conf = det[4]
                best_t.frames_since_update = 0
                best_t.matches += 1
                if identity is not None:
                    best_t.identity = identity
                matched[existing.index(best_t)] = True
            else:
                track = Track(self._next_id, det[:4], det[4])
                if identity is not None:
                    track.identity = identity
                    track.matches = 1
                self._next_id += 1
                self._tracks.append(track)
                log.debug("new track %d for %s", track.track_id, det)
        for i, t in enumerate(existing):
            if not matched[i]:
                t.frames_since_update += 1
        self._tracks = [t for t in self._tracks if t.frames_since_update <= self._max_lost]
        return self.tracks