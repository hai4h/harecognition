"""Phase 5 verification: PyQt6 threading engine + kiosk UI (headless)."""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_qss_assets_load():
    for name in ("dark_theme.qss", "kiosk_overlay.qss"):
        path = os.path.join(ROOT, f"assets/qss/{name}")
        assert os.path.isfile(path), f"missing {name}"
        with open(path) as f:
            assert f.read().strip(), f"empty {name}"


def test_audio_assets_exist():
    for name in ("success.wav", "error.wav"):
        assert os.path.isfile(os.path.join(ROOT, f"assets/sounds/{name}")), name


def test_app_runs_and_shuts_down_cleanly():
    """--test-mode: threads initialize/terminate cleanly, both modes are
    exercised by the dispatcher, telemetry throttled at 2 Hz, exit code 0."""
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "main.py"),
         "--test-mode", "--frames", "120"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, (
        f"exit {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    out = proc.stdout + proc.stderr
    assert "AIWorkerThread ready" in out, "worker never became ready"
    assert "CameraThread stopped" in out, "camera thread leaked"
    assert "AIWorkerThread stopped" in out, "worker thread leaked"
    assert "SHUTDOWN_COMPLETE" in out, "closeEvent did not join threads"
    assert "mode=2" in out or "mode 2" in out, "Mode 2 dispatcher hook not exercised"
    telemetry_lines = [l for l in out.splitlines() if l.startswith("TELEMETRY ")]
    assert telemetry_lines, "no telemetry updates"
    assert len(telemetry_lines) <= 12, (
        f"telemetry not throttled to 2 Hz ({len(telemetry_lines)} updates in 3 s)"
    )
    nonzero_fps = [
        l for l in telemetry_lines
        if l.split("fps=")[1].split(" ")[0].strip() not in ("0.0", "0")
    ]
    assert nonzero_fps, f"FPS never became nonzero:\n{telemetry_lines}"
    assert any("people=" in l for l in telemetry_lines), "people telemetry missing"


def test_fatal_init_error_exits_gracefully():
    """Unreachable MongoDB -> AI_FATAL + clean shutdown (exit 1), not SIGABRT."""
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "main.py"),
         "--test-mode", "--frames", "10",
         "--mongodb-uri", "mongodb://127.0.0.1:59999",
         "--mongodb-db", "harecognition_unreachable"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 1, (
        f"expected exit 1 (clean init failure), got {proc.returncode}\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "AI_FATAL:" in out, "fatal handler never ran"
    assert "SHUTDOWN_COMPLETE" in out, "did not shut down cleanly"
    assert "initialization failed" in out, "missing root-cause message"


def test_camera_off_at_startup():
    """Default launch keeps the camera stopped until Start Camera is pressed."""
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "main.py"),
         "--frames", "30", "--camera-source", "virtual"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, (
        f"exit {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "camera OFF at startup (start via UI)" in out, "camera should be off"
    assert "CameraThread started" not in out, "camera started despite being off"
    assert "SHUTDOWN_COMPLETE" in out, "did not shut down cleanly"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])