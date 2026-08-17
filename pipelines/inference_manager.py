"""Inference manager: ONNX Runtime sessions + detection/embedding/hand wrappers.

All neural models (except MediaPipe Hands) run through ONNX Runtime with the
provider chain from configs/model_config.yaml:
  - default: CUDAExecutionProvider (device 0) -> CPUExecutionProvider
  - ROCm: prepended ONLY when enable_rocm_workaround is true; on session
    failure we transparently fall back to CUDA, then CPU (guaranteed).
"""

import logging
import os

import cv2
import numpy as np
import onnxruntime as ort
import yaml

from core.paths import ROOT
log = logging.getLogger("inference_manager")

CUDA_OPTS = {"device_id": 0}


def load_model_config() -> dict:
    with open(os.path.join(ROOT, "configs/model_config.yaml")) as f:
        return yaml.safe_load(f)


def get_optimized_session(onnx_path: str, model_cfg: dict | None = None) -> ort.InferenceSession:
    cfg = model_cfg or load_model_config()
    cpu_opts = cfg["cpu_session_options"]
    providers: list[tuple[str, dict]] = []
    if cfg.get("enable_rocm_workaround"):
        providers.append((cfg["rocm_execution_provider"], {"device_id": 0}))
        log.info("ROCm workaround enabled: prepending %s", cfg["rocm_execution_provider"])
    for name in cfg["provider_priority"]:
        if name == "CUDA":
            providers.append(("CUDAExecutionProvider", dict(CUDA_OPTS)))
        elif name == "CPU":
            providers.append(("CPUExecutionProvider", dict(cpu_opts)))
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
    except Exception as exc:  # e.g. CUDA libs missing: fall back CUDA-less, then CPU
        log.warning("Session creation failed (%s); retrying without CUDA", exc)
        fallback = [("CPUExecutionProvider", dict(cpu_opts))]
        try:
            session = ort.InferenceSession(onnx_path, providers=fallback)
        except Exception as exc2:
            raise RuntimeError(f"No execution provider usable for {onnx_path}") from exc2
    log.info("Active providers for %s: %s", os.path.basename(onnx_path),
             session.get_providers())
    return session


class FaceDetector:
    """YOLO Face ONNX detector. Returns [x1, y1, x2, y2, conf] normalized 0.0-1.0."""

    def __init__(self, session: ort.InferenceSession, input_size: int = 640,
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self._session = session
        self._size = input_size
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._input_name = session.get_inputs()[0].name

    @staticmethod
    def _letterbox(img: np.ndarray, size: int):
        h, w = img.shape[:2]
        scale = min(size / h, size / w)
        nh, nw = round(h * scale), round(w * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        pad_x = (size - nw) // 2
        pad_y = (size - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        return canvas, scale, pad_x, pad_y

    @staticmethod
    def _decode(output: np.ndarray):
        """Generic YOLO head: (1, 4+nc, N) -> (N, 4+nc) xyxy."""
        out = output[0]
        if out.ndim == 3 and out.shape[1] <= out.shape[2] and out.shape[1] < 20:
            out = out.transpose(0, 2, 1)[0]
        elif out.ndim == 2 and out.shape[0] < out.shape[1]:
            out = out.T
        n, c = out.shape
        xywh = out[:, :4]
        scores = out[:, 4:c]
        box_conf = scores.max(axis=1)
        class_id = scores.argmax(axis=1)
        conf = np.where(class_id >= 0, box_conf, 0.0)
        boxes = np.empty_like(xywh)
        boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # x1
        boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # y1
        boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # x2
        boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # y2
        return boxes, conf

    def detect(self, frame_bgr: np.ndarray) -> list[list[float]]:
        h, w = frame_bgr.shape[:2]
        canvas, scale, pad_x, pad_y = self._letterbox(frame_bgr, self._size)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]  # NCHW
        output = self._session.run(None, {self._input_name: blob})[0]
        boxes, conf = self._decode(output)
        keep = conf >= self._conf
        boxes, conf = boxes[keep], conf[keep]
        if boxes.size == 0:
            return []
        xyxy = boxes.copy()
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale
        xyxy[:, 0::2] = np.clip(xyxy[:, 0::2], 0, w) / w
        xyxy[:, 1::2] = np.clip(xyxy[:, 1::2], 0, h) / h
        keep_idx = cv2.dnn.NMSBoxes(
            xyxy.tolist(), conf.tolist(), self._conf, self._iou
        )
        if isinstance(keep_idx, tuple):
            keep_idx = keep_idx[0]
        out = np.hstack([xyxy[keep_idx], conf[keep_idx, None]])
        return [list(map(float, row)) for row in out]


class EmbeddingExtractor:
    """ONNX 512-d embedding extractor (GhostFaceNet / ArcFace). Outputs are
    L2-normalized (unit norm) thanks to the export wrapper."""

    def __init__(self, session: ort.InferenceSession, size: int = 112):
        self._session = session
        self._size = size
        self._input_name = session.get_inputs()[0].name

    def extract(self, face_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_bgr, (self._size, self._size),
                             interpolation=cv2.INTER_AREA)
        blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob[None]  # NHWC
        out = self._session.run(None, {self._input_name: blob})[0]
        norm = np.linalg.norm(out)
        if norm > 0 and abs(norm - 1.0) > 1e-3:
            out = out / norm
        return out.astype(np.float32)


class HandTracker:
    """MediaPipe Hands (Tasks API): 21 3D landmarks per hand, CPU optimized."""

    def __init__(self, model_path: str | None = None, max_num_hands: int = 1,
                 detection_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if model_path is None:
            model_path = os.path.join(ROOT, "models/hand_landmarker.task")
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def track(self, frame_rgb: np.ndarray) -> list[np.ndarray]:
        """Return a list of (21, 3) landmark arrays in normalized coords."""
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self._landmarker.detect(mp_image)
        if not results.hand_landmarks:
            return []
        return [
            np.asarray([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
            for hand in results.hand_landmarks
        ]

    def close(self) -> None:
        self._landmarker.close()