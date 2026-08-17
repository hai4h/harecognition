# Phase 4: Dual-Mode Controllers & Gesture Verification

## Objectives
1. Implement `core/gesture.py`:
   - MediaPipe Hands 21-landmark (3D) tracking via the `HandTracker` from Phase 3.
   - Deterministic geometric checks for confirmation gestures: **Open Palm** and **Thumbs Up** (e.g., finger-extension angle/position rules over the 21 joints).
   - Sustained-gesture state machine: a gesture must hold continuously for >= 5 frames (~150 ms) to confirm; any interruption resets the counter (prevents accidental triggers).
2. Implement `pipelines/mode_tracking.py` (Mode 1 - Continuous Tracking & Identification):
   - Detect faces across the FULL frame with YOLO Face.
   - IoU match against active tracks (IoU > 0.3); on match return the cached identity without re-inference.
   - On new face: crop ROI -> GhostFaceNet 512D -> L2-normalize -> query FAISS Index #1 -> return name + cache to track.
   - Emit per-frame annotated overlays (bboxes + names) via OpenCV drawing.
3. Implement `pipelines/mode_attendance.py` (Mode 2 - High-Precision Attendance):
   - Guide-zone check: faces whose center is outside the central `guide_zone` (from `configs/app_config.yaml`) are ignored.
   - Gesture validation via `core/gesture.py` (sustained >= 5 frames).
   - On confirmation: crop ROI -> ArcFace 512D -> L2-normalize -> query FAISS Index #2 with strict threshold (cosine >= 0.65).
   - On match: write immutable record (`user_id`, `name`, UTC `timestamp`, `mode: "GESTURE_CONFIRMED"`, `confidence`) to MongoDB `attendance_logs` and emit the UI confirmation trigger signal (visual + audio).
4. Mode dispatcher: branch per frame on the active mode from the UI state machine (Mode 1 or Mode 2); state transitions ("Align Face" -> "Show Gesture" -> "Success") are driven by the pipeline's status metadata.

## Verification Checkpoint
Create `tests/test_modes.py`:
- Feed synthetic MediaPipe landmark sequences: assert the gesture state machine confirms ONLY after >= 5 sustained frames and resets on interruption.
- Feed synthetic detections: assert Mode 1 caches identity and skips re-extraction on IoU match.
- Feed synthetic embedding queries: assert Mode 2 writes to MongoDB `attendance_logs` only when cosine >= 0.65 AND the face center is inside the guide zone; otherwise nothing is logged.
- Assert logged records contain valid `user_id`, `name`, UTC `timestamp`, `mode`, and `confidence`.