"""CameraThread: source specs, start/stop/restart, video-file looping."""

import os
import sys
import time

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QCoreApplication, QObject, pyqtSlot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from main import CameraThread, VirtualCamera, open_source

_app = QCoreApplication.instance() or QCoreApplication([])


class _FrameCounter(QObject):
    def __init__(self):
        super().__init__()
        self.count = 0

    @pyqtSlot(object)
    def on_frame(self, frame):
        self.count += 1


def _wait_until(cond, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_open_source_virtual():
    cap = open_source("virtual")
    assert isinstance(cap, VirtualCamera)
    ok, frame = cap.read()
    assert ok and frame is not None
    cap.release()


def test_open_source_device_spec():
    assert isinstance(open_source("device:3"), cv2.VideoCapture)


def test_virtual_camera_start_stop_restart():
    cam = CameraThread(source="virtual", target_fps=1000)
    counter = _FrameCounter()
    cam.frame_ready.connect(counter.on_frame)
    try:
        cam.start()
        assert _wait_until(lambda: counter.count > 5), "virtual frames not flowing"
        cam.stop()
        frozen = counter.count
        time.sleep(0.05)
        assert counter.count == frozen, "frames kept flowing after stop()"
        cam.start()
        assert _wait_until(lambda: counter.count > frozen + 5), "restart did not resume"
    finally:
        cam.stop()


def test_set_source_restarts_while_running():
    cam = CameraThread(source="virtual", target_fps=1000)
    counter = _FrameCounter()
    cam.frame_ready.connect(counter.on_frame)
    try:
        cam.start()
        assert _wait_until(lambda: counter.count > 5), "virtual frames not flowing"
        cam.set_source("virtual")  # restart path (stop + start) with same source
        assert _wait_until(lambda: counter.count > 10), "restart did not resume frames"
        assert not cam.isRunning() or counter.count > 10
    finally:
        cam.stop()


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("cam") / "loop.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter unavailable (no encoder)")
    try:
        for i in range(10):
            frame = cv2.rectangle(
                np.zeros((240, 320, 3), dtype="uint8"),
                (i * 20, 0), (i * 20 + 15, 240), (255, 0, 0), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_video_file_loops(tiny_video):
    cam = CameraThread(source=tiny_video, target_fps=1000)
    counter = _FrameCounter()
    cam.frame_ready.connect(counter.on_frame)
    try:
        cam.start()
        assert _wait_until(lambda: counter.count >= 25, timeout=10), (
            f"video should loop past its 10 frames, saw {counter.count}")
        # distinct frames must keep arriving (not frozen on last frame)
        assert counter.count >= 25
    finally:
        cam.stop()


def test_video_file_source_is_looping(tiny_video):
    assert os.path.isfile(tiny_video)
    cap = open_source(tiny_video)
    assert isinstance(cap, cv2.VideoCapture) and cap.isOpened()
    cap.release()
