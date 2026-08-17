"""End-to-end pipeline benchmark: ingest FPS, glass-to-glass latency, RAM.

    .venv/bin/python scripts/benchmark_pipeline.py --source virtual --seconds 10
    .venv/bin/python scripts/benchmark_pipeline.py --source mp4 --path clip.mp4 --seconds 60
    .venv/bin/python scripts/benchmark_pipeline.py --source camera --seconds 60 --cpu-only

Prints BENCH <key>=<value> lines for machine parsing plus a human summary.
"""

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _build_pipeline(mode: int, cpu_only: bool):
    import onnxruntime as ort
    from core.database import create_storage_backend
    from core.gesture import GestureStateMachine
    from core.tracker import IoUTracker
    from core.vector_engine import DualVectorEngine
    from pipelines.inference_manager import (
        EmbeddingExtractor,
        FaceDetector,
        HandTracker,
        get_optimized_session,
        load_model_config,
    )
    from pipelines.mode_tracking import Mode1Pipeline
    from pipelines.mode_attendance import Mode2Pipeline

    cfg = load_model_config()
    if cpu_only:
        cpu_opts = cfg["cpu_session_options"]
        sessions = {
            name: ort.InferenceSession(
                cfg["models"][name]["path"],
                providers=[("CPUExecutionProvider", cpu_opts)],
            )
            for name in ("yolo_face", "ghostfacenet_512", "arcface_512")
        }
    else:
        sessions = {
            name: get_optimized_session(cfg["models"][name]["path"], cfg)
            for name in ("yolo_face", "ghostfacenet_512", "arcface_512")
        }
    providers = {name: s.get_providers()[0] for name, s in sessions.items()}
    storage = create_storage_backend()
    engine = DualVectorEngine(storage)
    app_cfg_path = os.path.join(ROOT, "configs/app_config.yaml")
    import yaml

    with open(app_cfg_path) as f:
        app_cfg = yaml.safe_load(f)
    detector = FaceDetector(sessions["yolo_face"])
    if mode == 1:
        pipeline = Mode1Pipeline(
            detector,
            EmbeddingExtractor(sessions["ghostfacenet_512"]),
            engine, IoUTracker(),
        )
    else:
        pipeline = Mode2Pipeline(
            detector,
            HandTracker(),
            GestureStateMachine(require_frames=app_cfg.get("gesture_min_frames", 5)),
            EmbeddingExtractor(sessions["arcface_512"]),
            engine, storage,
            guide_zone=app_cfg["guide_zone"],
            threshold=app_cfg.get("faiss_threshold_mode2", 0.65),
        )
    return pipeline, providers


def _open_source(source: str, path: str | None):
    if source == "virtual":
        from main import VirtualCamera

        return VirtualCamera()
    if source == "mp4":
        assert path and os.path.isfile(path), f"mp4 not found: {path}"
        return cv2.VideoCapture(path)
    if source == "camera":
        cap = cv2.VideoCapture(0)
        assert cap.isOpened(), "cannot open /dev/video0"
        return cap
    raise SystemExit(f"unknown source: {source}")


class _RssWatcher(threading.Thread):
    """Samples VmRSS post-exec (ru_maxrss is polluted by fork inheritance)."""

    def __init__(self):
        super().__init__(daemon=True)
        self._done = threading.Event()
        self.peak_kb = 0

    def run(self):
        while not self._done.wait(0.25):
            try:
                with open("/proc/self/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            self.peak_kb = max(self.peak_kb, int(line.split()[1]))
                            break
            except OSError:
                pass

    def stop(self):
        self._done.set()
        self.join(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description="HARecognition end-to-end benchmark")
    parser.add_argument("--source", choices=["virtual", "mp4", "camera"], default="virtual")
    parser.add_argument("--path", default=None, help="mp4 path")
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--mode", type=int, choices=[1, 2], default=1)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()

    pipeline, providers = _build_pipeline(args.mode, args.cpu_only)
    src = _open_source(args.source, args.path)
    print(f"BENCH providers={providers}")
    print(f"BENCH source={args.source} mode={args.mode} seconds={args.seconds}")

    deadline = time.monotonic() + args.seconds
    latencies = []
    frames = 0
    errors = 0
    rss = _RssWatcher()
    rss.start()
    try:
        while time.monotonic() < deadline:
            ok, frame = src.read()
            if not ok:
                time.sleep(0.01)
                continue
            t0 = time.perf_counter()
            try:
                pipeline.process(frame)
                latencies.append((time.perf_counter() - t0) * 1000.0)
                frames += 1
            except Exception:
                errors += 1
    finally:
        src.release()
        rss.stop()

    elapsed = args.seconds
    fps = frames / elapsed
    lat = np.asarray(latencies)
    peak_rss_mb = rss.peak_kb / 1024.0
    p50 = float(np.percentile(lat, 50)) if lat.size else 0.0
    p95 = float(np.percentile(lat, 95)) if lat.size else 0.0
    mx = float(lat.max()) if lat.size else 0.0

    print(f"BENCH fps={fps:.2f}")
    print(f"BENCH latency_p50_ms={p50:.2f} latency_p95_ms={p95:.2f} latency_max_ms={mx:.2f}")
    print(f"BENCH peak_rss_mb={peak_rss_mb:.1f} errors={errors}")
    print(f"SUMMARY: {fps:.1f} FPS | p50 {p50:.2f} ms | p95 {p95:.2f} ms | "
          f"max {mx:.2f} ms | peak RSS {peak_rss_mb:.0f} MB | {providers}")


if __name__ == "__main__":
    main()