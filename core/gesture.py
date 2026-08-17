"""Gesture verification: geometric checks + sustained-gesture state machine.

MediaPipe hand landmark layout (21 points, normalized coords):
  WRIST=0; thumb 1-4 (CMC, MCP, IP, TIP); index 5-8 (MCP, PIP, DIP, TIP);
  middle 9-12; ring 13-16; pinky 17-20.

Confirmation gestures: OPEN_PALM and THUMBS_UP. A gesture must hold
continuously for >= require_frames frames (~150 ms at 30 FPS); any
interruption resets the counter (prevents accidental triggers).
"""

import math

WRIST = 0
THUMB_TIP = 4
FINGERS = {
    "index": (8, 6, 5),    # tip, pip, mcp
    "middle": (12, 10, 9),
    "ring": (16, 14, 13),
    "pinky": (20, 18, 17),
}
THUMB_MCP, THUMB_IP = 2, 3

EXTENSION_RATIO = 1.25
OPEN_PALM = "open_palm"
THUMBS_UP = "thumbs_up"
CONFIRM_GESTURES = {OPEN_PALM, THUMBS_UP}


def _dist(a, b) -> float:
    return math.dist((a[0], a[1]), (b[0], b[1]))


def is_finger_extended(lms, tip, pip, mcp) -> bool:
    return _dist(lms[mcp], lms[tip]) > EXTENSION_RATIO * _dist(lms[mcp], lms[pip])


def is_thumb_extended(lms) -> bool:
    return _dist(lms[THUMB_MCP], lms[THUMB_TIP]) > EXTENSION_RATIO * _dist(
        lms[THUMB_MCP], lms[THUMB_IP]
    )


def detect_gesture(lms) -> str | None:
    """Return the detected confirmation gesture name, or None."""
    if len(lms) != 21:
        return None
    fingers_extended = [
        is_finger_extended(lms, tip, pip, mcp) for tip, pip, mcp in FINGERS.values()
    ]
    thumb_extended = is_thumb_extended(lms)
    if all(fingers_extended) and thumb_extended:
        return OPEN_PALM
    if thumb_extended and not any(fingers_extended):
        return THUMBS_UP
    return None


class GestureStateMachine:
    """Confirms a gesture only after >= require_frames sustained frames."""

    def __init__(self, require_frames: int = 5):
        self._require = require_frames
        self._counter = 0
        self._last = None

    @property
    def progress(self) -> int:
        return self._counter

    def reset(self) -> None:
        self._counter = 0
        self._last = None

    def update(self, landmarks) -> bool:
        """Feed one frame of hand landmarks (or None / multiple hands handled
        by the caller). Returns True only on the confirming transition frame."""
        gesture = detect_gesture(landmarks) if landmarks is not None else None
        if gesture is not None and gesture == self._last:
            self._counter += 1
        elif gesture is not None:
            self._last = gesture
            self._counter = 1
        else:
            self.reset()
            return False
        if self._counter >= self._require:
            self.reset()
            return True
        return False