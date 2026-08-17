"""Main kiosk window: view manager, mode switching, status bar, threads."""

import os
import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.video_widget import VideoWidget
from ui.components.attendance_card import AttendanceCard
from ui.components.stats_panel import StatsPanel
from core.paths import ROOT


class MainWindow(QMainWindow):
    def __init__(self, camera, ai_worker, cfg: dict, test_mode: bool = False):
        super().__init__()
        self._camera = camera
        self._worker = ai_worker
        self._cfg = cfg
        self._test_mode = test_mode
        self._mode = cfg.get("mode", 1)
        self._last_meta = {}
        self.setWindowTitle("HARecognition Kiosk")
        self.resize(1280, 800)

        central = QWidget()
        root = QVBoxLayout(central)

        controls = QHBoxLayout()
        self._btn_mode1 = QPushButton("Mode 1 - Tracking")
        self._btn_mode2 = QPushButton("Mode 2 - Attendance")
        self._btn_mode1.setCheckable(True)
        self._btn_mode2.setCheckable(True)
        self._btn_mode1.clicked.connect(lambda: self.set_mode(1))
        self._btn_mode2.clicked.connect(lambda: self.set_mode(2))
        self._cam_selector = QComboBox()
        self._cam_selector.addItem("Virtual (test)", -1)
        for dev in sorted(self._list_cameras()):
            self._cam_selector.addItem(dev, dev)
        self._mode_state_label = QLabel("Mode 1")
        self._mode_state_label.setObjectName("modeStateLabel")
        self._mode_state_label.setProperty("state", "align")
        controls.addWidget(self._btn_mode1)
        controls.addWidget(self._btn_mode2)
        controls.addWidget(QLabel("Camera:"))
        controls.addWidget(self._cam_selector)
        controls.addStretch(1)
        controls.addWidget(self._mode_state_label)
        root.addLayout(controls)

        video_row = QHBoxLayout()
        self._video = VideoWidget()
        video_row.addWidget(self._video, 1)
        self._stats = StatsPanel()
        video_row.addWidget(self._stats)
        root.addLayout(video_row, 1)

        self._card = AttendanceCard()
        self._card.setMaximumWidth(420)
        root.addWidget(self._card, 0, Qt.AlignmentFlag.AlignHCenter)
        self.setCentralWidget(central)

        self._camera.frame_ready.connect(self._worker.on_frame)
        self._worker.frame_annotated.connect(self._video.set_frame)
        self._worker.metadata_ready.connect(self._on_metadata)
        self._worker.attendance_confirmed.connect(self._on_attendance)
        self._worker.error.connect(self._on_error)
        self._camera.backpressure_fn = self._worker.can_accept

        self._telemetry_timer = QTimer(self)  # 2 Hz throttle (spec)
        self._telemetry_timer.setInterval(500)
        self._telemetry_timer.timeout.connect(self._refresh_telemetry)
        self._telemetry_timer.start()

        if test_mode:
            self._cycle_timer = QTimer(self)  # exercise both dispatcher hooks
            self._cycle_timer.setInterval(2000)
            self._cycle_timer.timeout.connect(
                lambda: self.set_mode(2 if self._mode == 1 else 1))
            self._cycle_timer.start()

        self.set_mode(self._mode)

    @staticmethod
    def _list_cameras():
        devs = []
        for i in range(8):
            if os.path.exists(f"/dev/video{i}"):
                devs.append(f"/dev/video{i}")
        return devs

    def set_mode(self, mode: int) -> None:
        self._mode = mode
        self._btn_mode1.setChecked(mode == 1)
        self._btn_mode2.setChecked(mode == 2)
        self._worker.set_mode(mode)
        self._mode_state_label.setText(
            "Mode 1 - Tracking" if mode == 1 else "Mode 2 - Attendance")

    def _on_metadata(self, metadata: dict) -> None:
        metadata["mode"] = self._mode
        self._last_meta = metadata

    def _refresh_telemetry(self) -> None:
        meta = dict(self._last_meta)
        if not meta:
            return
        meta.setdefault("fps", 0.0)
        meta.setdefault("people", 0)
        meta.setdefault("state", "align")
        meta.setdefault("status", "running")
        meta["mode"] = self._mode
        self._stats.update_stats(meta)
        state = meta.get("state", "align")
        self._mode_state_label.setProperty("state", state)
        self._mode_state_label.style().unpolish(self._mode_state_label)
        self._mode_state_label.style().polish(self._mode_state_label)
        self.statusBar().showMessage(
            f"FPS {meta.get('fps', 0):.1f} | state {state} | mode {self._mode}")
        if self._test_mode:
            print(f"TELEMETRY fps={meta.get('fps', 0):.1f} people={meta.get('people', 0)} "
                  f"state={state} mode={self._mode}", flush=True)

    def _on_attendance(self, user_id: str, name: str, confidence: float) -> None:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._card.show_confirmation(user_id, name, confidence, timestamp)
        self._play_audio("success")
        self._mode_state_label.setProperty("state", "success")
        self._mode_state_label.style().unpolish(self._mode_state_label)
        self._mode_state_label.style().polish(self._mode_state_label)

    def _on_error(self, message: str) -> None:
        print(f"AI_ERROR: {message}", flush=True)
        self.statusBar().showMessage(f"ERROR: {message}")
        self._play_audio("error")

    def _play_audio(self, kind: str) -> None:
        if self._test_mode or not self._cfg.get("audio_feedback_enabled", True):
            return
        try:
            from PyQt6.QtCore import QUrl
            from PyQt6.QtMultimedia import QSoundEffect

            effect = QSoundEffect()
            effect.setSource(QUrl.fromLocalFile(
                os.path.join(ROOT, f"assets/sounds/{kind}.wav")))
            effect.play()
        except Exception:
            pass  # audio is best-effort (no device in headless/CI)

    def closeEvent(self, event) -> None:
        self._telemetry_timer.stop()
        if self._test_mode:
            self._cycle_timer.stop()
        print("SHUTDOWN: stopping threads...", flush=True)
        self._camera.stop()
        self._worker.stop()
        print("SHUTDOWN_COMPLETE", flush=True)
        event.accept()