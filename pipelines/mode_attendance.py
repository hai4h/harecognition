"""Mode 2 - High-Precision Time Attendance.

Guide-zone gate (face center inside the central zone) -> gesture
confirmation (sustained >= 5 frames) -> ArcFace 512D embedding ->
FAISS Index #2 search with strict threshold (cosine >= 0.65) -> immutable
attendance_logs record + UI confirmation trigger callback.
"""

import logging

import cv2
import numpy as np

log = logging.getLogger("mode_attendance")

STATE_ALIGN = "align"
STATE_GESTURE = "gesture"
STATE_SUCCESS = "success"
LOG_MODE = "GESTURE_CONFIRMED"


def _face_center(det: list[float]) -> tuple[float, float]:
    return ((det[0] + det[2]) / 2.0, (det[1] + det[3]) / 2.0)


def _inside_guide_zone(center: tuple[float, float], zone: list[float]) -> bool:
    gx, gy, gw, gh = zone
    return gx <= center[0] <= gx + gw and gy <= center[1] <= gy + gh


def _crop_roi(frame: np.ndarray, det: list[float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v * s)) for v, s in zip(det[:4], (w, h, w, h)))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame[y1:y1 + 1, x1:x1 + 1]
    return frame[y1:y2, x1:x2]


class Mode2Pipeline:
    def __init__(self, detector, hand_tracker, gesture_machine, arcface_extractor,
                 engine, storage, guide_zone: list[float],
                 threshold: float = 0.65, min_conf: float = 0.25,
                 on_confirmed=None):
        self._detector = detector
        self._hands = hand_tracker
        self._gesture = gesture_machine
        self._extractor = arcface_extractor
        self._engine = engine
        self._storage = storage
        self._zone = guide_zone
        self._threshold = threshold
        self._min_conf = min_conf
        self._on_confirmed = on_confirmed  # UI callback (visual + audio trigger)

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        dets = [d for d in self._detector.detect(frame) if d[4] >= self._min_conf]
        in_zone = [d for d in dets if _inside_guide_zone(_face_center(d), self._zone)]
        confirmed = False
        if in_zone:
            hands = self._hands.track(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            confirmed = any(self._gesture.update(h) for h in hands) if hands else False
            if not hands:
                self._gesture.reset()

        state = STATE_ALIGN
        last_match = None
        if in_zone and not confirmed:
            state = STATE_GESTURE
        if confirmed:
            target = max(in_zone, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
            crop = _crop_roi(frame, target)
            vec = self._extractor.extract(crop)
            top = self._engine.search_mode2(vec, top_k=1, threshold=self._threshold)
            if top:
                user_id, name, confidence = top[0]
                self._storage.log_attendance(user_id, name, LOG_MODE, confidence)
                last_match = (user_id, name, confidence)
                state = STATE_SUCCESS
                log.info("attendance confirmed: %s %s (%.3f)", user_id, name, confidence)
                if self._on_confirmed:
                    self._on_confirmed(user_id, name, confidence)
            else:
                log.info("gesture confirmed but no biometric match")
                state = STATE_ALIGN
                self._gesture.reset()

        overlay = self._annotate(frame.copy(), in_zone, state)
        status = {
            "state": state,
            "gesture_progress": self._gesture.progress,
            "faces_in_zone": len(in_zone),
            "last_match": last_match,
        }
        return overlay, status

    def _annotate(self, frame: np.ndarray, faces_in_zone: list[list[float]],
                  state: str) -> np.ndarray:
        h, w = frame.shape[:2]
        gx, gy, gw, gh = self._zone
        cv2.rectangle(frame,
                      (int(gx * w), int(gy * h)),
                      (int((gx + gw) * w), int((gy + gh) * h)),
                      (0, 200, 255), 1)
        for det in faces_in_zone:
            x1, y1, x2, y2 = (int(round(v * s)) for v, s in zip(det[:4], (w, h, w, h)))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        color = {"align": (0, 200, 255), "gesture": (255, 200, 0),
                 "success": (0, 255, 0)}[state]
        cv2.putText(frame, f"STATE: {state}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame