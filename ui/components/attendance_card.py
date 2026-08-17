"""Mode 2 user confirmation card (name, confidence, timestamp)."""

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout


class AttendanceCard(QFrame):
    def __init__(self, auto_hide_ms: int = 4000, parent=None):
        super().__init__(parent)
        self.setObjectName("AttendanceCard")
        self.setVisible(False)
        layout = QVBoxLayout(self)
        self._name = QLabel()
        self._name.setObjectName("cardName")
        self._name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._meta = QLabel()
        self._meta.setObjectName("cardMeta")
        self._meta.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._name)
        layout.addWidget(self._meta)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(auto_hide_ms)
        self._timer.timeout.connect(self.hide)

    def show_confirmation(self, user_id: str, name: str, confidence: float,
                          timestamp: str) -> None:
        self._name.setText(name)
        self._meta.setText(f"{user_id}  |  {confidence:.1%}  |  {timestamp}")
        self.show()
        self._timer.start()