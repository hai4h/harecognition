"""Optimized QPainter/QPixmap viewport for the annotated camera feed.

The Mode 2 guide zone is drawn by the pipeline itself on the annotated
frame (pipelines/mode_attendance.py) — never here, so it cannot leak
into Mode 1.
"""

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import QWidget


class VideoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self.setMinimumSize(640, 360)
        self.setObjectName("VideoWidget")

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        """Convert BGR np.ndarray -> QPixmap (single conversion, no deep copies)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(image.copy())
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        if self._pixmap is not None:
            scaled = self._pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        painter.end()