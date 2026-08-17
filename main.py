"""HARecognition entry point: PyQt6 threading engine + kiosk UI.

CameraThread (QThread): dedicated cv2.VideoCapture loop at target_fps,
emits raw frames via pyqtSignal; DROPS frames under AI-worker backpressure
(no queue accumulation).

AIWorkerThread (QThread): owns ALL model memory contexts (ONNX sessions,
FAISS dual indices, MediaPipe tracker, mode pipelines, dispatcher); runs the
active dual-mode pipeline per frame, emits annotated frame + metadata.

Thread sync strictly via Qt signal/slot (queued connections).

Headless verification:
    ./run.sh -- python main.py --test-mode --frames 90 --cycle-modes
"""

import argparse
import logging
import os
import signal
import sys
import time

import cv2
import numpy as np

from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication

from core.paths import ROOT

log = logging.getLogger("harecognition")


# ------------------------------------------------------------- virtual camera

class VirtualCamera:
    """Synthetic camera for --test-mode: moving shapes + real photo frames
    (exercises real face detection without a physical device)."""

    def __init__(self, resolution=(1280, 720), fps=30):
        self._w, self._h = resolution
        self._fps = fps
        self._n = 0
        photo = cv2.imread(os.path.join(ROOT, "tests/fixtures/group_photo.jpg"))
        self._photo = cv2.resize(photo, (self._w, self._h)) if photo is not None else None

    def read(self):
        self._n += 1
        if self._photo is not None and self._n % 45 == 0:
            return True, self._photo.copy()
        frame = np.zeros((self._h, self._w, 3), dtype=np.uint8)
        x = int((self._n * 8) % self._w)
        cv2.rectangle(frame, (x, self._h // 2), (x + 80, self._h // 2 + 80),
                      (0, 120, 255), -1)
        cv2.circle(frame, (self._w // 2, self._h // 2), 30 + (self._n % 40),
                   (80, 80, 80), 2)
        return True, frame

    def release(self):
        pass


# ---------------------------------------------------------------- camera thread

class CameraThread(QThread):
    frame_ready = pyqtSignal(object)

    def __init__(self, source, target_fps: int, test_mode: bool = False, parent=None):
        super().__init__(parent)
        self._source = source
        self._fps = target_fps
        self._test_mode = test_mode
        self._stop_flag = False
        self.backpressure_fn = None  # callable() -> bool, set by MainWindow

    def run(self) -> None:
        cap = VirtualCamera() if self._test_mode else cv2.VideoCapture(self._source)
        if not self._test_mode and not cap.isOpened():
            log.error("cannot open camera %r", self._source)
            return
        try:
            period = 1.0 / max(1, self._fps)
            while not self._stop_flag:
                ok, frame = cap.read()
                if not ok:
                    log.warning("frame read failed")
                    break
                if self.backpressure_fn is None or self.backpressure_fn():
                    self.frame_ready.emit(frame)  # drop frame when AI busy
                time.sleep(period)
        finally:
            cap.release()
        log.info("CameraThread stopped")

    def stop(self, timeout_ms: int = 5000) -> None:
        self._stop_flag = True
        self.wait(timeout_ms)


# ---------------------------------------------------------------- AI worker

class AIWorkerThread(QThread):
    frame_annotated = pyqtSignal(object)
    metadata_ready = pyqtSignal(dict)
    attendance_confirmed = pyqtSignal(str, str, float)
    error = pyqtSignal(str)
    ready = pyqtSignal()

    def __init__(self, mode: int = 1, parent=None):
        super().__init__(parent)
        self._mode = mode
        self._busy = False
        self._sessions = None
        self._fps = 0.0
        self._fps_t0 = None
        self._fps_count = 0

    # -- model context setup (runs in the worker thread's run()) -------------

    def _build_context(self):
        from core.database import create_storage_backend
        from core.tracker import IoUTracker
        from core.vector_engine import DualVectorEngine
        from core.gesture import GestureStateMachine
        from pipelines.inference_manager import (
            FaceDetector,
            EmbeddingExtractor,
            HandTracker,
            get_optimized_session,
            load_model_config,
        )
        from pipelines.mode_tracking import Mode1Pipeline
        from pipelines.mode_attendance import Mode2Pipeline
        from pipelines.mode_dispatcher import ModeDispatcher

        cfg = load_model_config()
        sessions = {
            name: get_optimized_session(cfg["models"][name]["path"], cfg)
            for name in ("yolo_face", "ghostfacenet_512", "arcface_512")
        }
        storage = create_storage_backend()
        engine = DualVectorEngine(storage)
        from core.database import load_app_config

        app_cfg = load_app_config()
        hand = HandTracker()
        tracker = IoUTracker()
        self._mode1 = Mode1Pipeline(
            FaceDetector(sessions["yolo_face"]),
            EmbeddingExtractor(sessions["ghostfacenet_512"]),
            engine, tracker,
        )
        self._mode2 = Mode2Pipeline(
            FaceDetector(sessions["yolo_face"]),
            hand,
            GestureStateMachine(require_frames=app_cfg.get("gesture_min_frames", 5)),
            EmbeddingExtractor(sessions["arcface_512"]),
            engine, storage,
            guide_zone=app_cfg["guide_zone"],
            threshold=app_cfg.get("faiss_threshold_mode2", 0.65),
            on_confirmed=self._on_attendance,
        )
        self._dispatcher = ModeDispatcher(self._mode1, self._mode2)
        self._sessions = sessions

    # -- slots (invoked via queued connections from CameraThread) ------------

    def can_accept(self) -> bool:
        return not self._busy

    def set_mode(self, mode: int) -> None:
        self._mode = mode

    def _on_attendance(self, user_id: str, name: str, confidence: float) -> None:
        self.attendance_confirmed.emit(user_id, name, confidence)

    def on_frame(self, frame) -> None:
        if self._sessions is None:
            return
        self._busy = True
        try:
            overlay, meta = self._dispatcher.process(frame, self._mode)
            meta["state"] = meta.get("state", "align")
            now = time.monotonic()
            if self._fps_t0 is None:
                self._fps_t0 = now
                self._fps_count = 0
            self._fps_count += 1
            dt = now - self._fps_t0
            if dt >= 1.0:
                self._fps = self._fps_count / dt
                self._fps_t0, self._fps_count = now, 0
            meta["fps"] = self._fps
            meta["people"] = meta.get("faces", meta.get("faces_in_zone", 0))
            self.frame_annotated.emit(overlay)
            self.metadata_ready.emit(meta)
        except Exception as exc:  # never crash the UI thread
            log.exception("pipeline error")
            self.error.emit(str(exc))
        finally:
            self._busy = False

    # -- thread lifecycle ----------------------------------------------------

    def run(self) -> None:
        self._build_context()
        log.info("AIWorkerThread ready (sessions: %s)",
                 {k: v.get_providers()[0] for k, v in self._sessions.items()})
        self.ready.emit()
        self.exec()  # event loop for queued slot delivery
        log.info("AIWorkerThread stopped")

    def stop(self, timeout_ms: int = 10000) -> None:
        if not self.isRunning():
            return
        self.quit()
        self.wait(timeout_ms)


# ---------------------------------------------------------------- bootstrap

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="HARecognition kiosk")
    parser.add_argument("--test-mode", action="store_true",
                        help="headless: virtual camera, no display dependency")
    parser.add_argument("--frames", type=int, default=0,
                        help="auto-close after N frames (test-mode)")
    parser.add_argument("--mode", type=int, default=None, help="initial mode 1|2")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.test_mode:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from core.database import load_app_config
    from ui.main_window import MainWindow

    cfg = load_app_config()
    app = QApplication(sys.argv[:1])

    def load_qss(name: str) -> str:
        path = os.path.join(ROOT, f"assets/qss/{name}")
        with open(path) as f:
            return f.read()

    app.setStyleSheet(load_qss("dark_theme.qss") + load_qss("kiosk_overlay.qss"))

    camera = CameraThread(source=cfg.get("camera_id", 0),
                          target_fps=cfg.get("target_fps", 30),
                          test_mode=args.test_mode)
    worker = AIWorkerThread(mode=args.mode or cfg.get("mode", 1))
    window = MainWindow(camera, worker, cfg, test_mode=args.test_mode)

    camera.start()
    worker.start()
    window.show()

    def _request_shutdown(signum, frame):
        """SIGINT/SIGTERM -> clean closeEvent path (joins both threads)."""
        log.info("signal %d received; shutting down cleanly", signum)
        QTimer.singleShot(0, window.close)
        QTimer.singleShot(100, app.quit)

    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)

    if args.frames > 0:
        def _auto_close():
            window.close()
            app.quit()

        timer = QTimer()
        timer.setSingleShot(True)
        timer.setInterval(int(args.frames / cfg.get("target_fps", 30) * 1000))
        timer.timeout.connect(_auto_close)
        worker.ready.connect(timer.start)  # countdown begins once AI context is live

    code = app.exec()
    camera.stop()
    worker.stop()
    return code


if __name__ == "__main__":
    sys.exit(main())