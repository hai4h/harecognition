"""Telemetry panel: FPS, people count, system status, active mode (2 Hz)."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QLabel, QWidget


class StatsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("StatsPanel")
        self.setFixedWidth(260)
        layout = QGridLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self._labels = {}
        for row, (key, title) in enumerate([
            ("fps", "FPS"),
            ("people", "People"),
            ("state", "Mode State"),
            ("mode", "Active Mode"),
            ("status", "System"),
        ]):
            name = QLabel(title)
            name.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value = QLabel("-")
            value.setProperty("role", "value")
            value.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            layout.addWidget(name, row, 0)
            layout.addWidget(value, row, 1)
            self._labels[key] = value

    def update_stats(self, metadata: dict) -> None:
        self._labels["fps"].setText(f"{metadata.get('fps', 0):.1f}")
        self._labels["people"].setText(str(metadata.get("people", 0)))
        self._labels["state"].setText(str(metadata.get("state", "-")))
        self._labels["mode"].setText(f"Mode {metadata.get('mode', '-')}")
        self._labels["status"].setText(str(metadata.get("status", "running")))