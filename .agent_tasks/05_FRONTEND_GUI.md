# Phase 5: PyQt6 Native Threading Engine & Kiosk UI

## Objectives
1. Implement the PyQt6 threading engine in `main.py` (application bootstrap & thread wiring, per `DESCRIPTION.md` Section 7):
   - `CameraThread` (QThread): dedicated `cv2.VideoCapture` loop pulling frames at 30-60 FPS; emits raw `np.ndarray` via `pyqtSignal`; under backpressure (AI worker busy) DROPS frames instead of queueing (no accumulation).
   - `AIWorkerThread` (QThread): owns ALL model memory contexts (ONNX sessions, FAISS dual indices, MediaPipe trackers, mode pipelines); consumes frames, runs the active dual-mode pipeline, renders bbox/name overlays directly onto the matrix via OpenCV, emits annotated frame + metadata (faces, people count, mode state, FPS).
   - Zero-copy matrix pipeline: `np.ndarray` -> `QImage` -> `QPixmap` (no deep copies; one shared buffer owner at a time).
   - Thread synchronization strictly via Qt signal/slot (`pyqtSignal`) per `DESCRIPTION.md` Section 6 topology.
2. Implement `ui/main_window.py`:
   - `QMainWindow` shell & view manager.
   - Mode switching (Mode 1 / Mode 2) and the visual state machine: "Align Face" -> "Show Gesture" -> "Success".
   - Camera device selector, FPS/latency counters, status bar.
   - `closeEvent` must stop both threads, release the camera, and join cleanly without hangs.
3. Implement `ui/video_widget.py`:
   - Optimized QPainter/QPixmap viewport rendering the annotated camera feed.
   - Draws the Mode 2 central guide-zone overlay on the widget layer.
4. Implement `ui/components/`:
   - `attendance_card.py`: Mode 2 user confirmation card (name, confidence, timestamp) triggered by the attendance signal.
   - `stats_panel.py`: telemetry panel throttled to 2 Hz (FPS, people count, system status, active mode).
5. Apply kiosk QSS styles (`assets/qss/dark_theme.qss`, `assets/qss/kiosk_overlay.qss`) and audio feedback (`assets/sounds/success.wav`, `assets/sounds/error.wav`) on success/error.
6. Support headless verification: `--test-mode` flag runs the app with a virtual camera (synthetic frames or bundled MP4 loop) and no display dependency.

## Verification Checkpoint
Launch the client with `--test-mode` and verify:
- CameraThread and AIWorkerThread initialize and terminate cleanly (no deadlocks, no leaked resources).
- Signal-slot bindings deliver annotated frames + metadata to the main thread at 30+ FPS.
- Mode 1 and Mode 2 pipeline hooks are exercised by the dispatcher.
- `closeEvent` shuts down both threads and releases the camera without hangs (exit code 0).
- QSS files load without warnings; telemetry updates are throttled to 2 Hz.