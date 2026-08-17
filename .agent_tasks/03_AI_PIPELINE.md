# Phase 3: Model Optimization & Inference Pipeline

## Objectives
1. Implement `scripts/convert_to_onnx.py` — YOLO Face detector export:
   - Load `models/yolov8p-face-v2.pt` (PyTorch) and export to ONNX (`models/yolo_face_custom.onnx`) with dynamic batch, fixed 640x640 input.
   - Post-processing must yield detections `[x1, y1, x2, y2, conf]` normalized to 0.0-1.0.
   - Latency target: 8-12 ms per full-frame inference.
2. Implement `scripts/export_embedding_onnx.py` — **runs ONLY inside `.venv-dev`**:
   - Use the `deepface` library to download the GhostFaceNet and ArcFace `.h5` weights (deepface's `GhostFaceNet` and `ArcFace` backends).
   - Convert each `.h5` model to ONNX with `tf2onnx.convert.from_keras`.
   - Outputs: `models/ghostfacenet_512.onnx` and `models/arcface_512.onnx`.
   - Assert inputs are 112x112x3 RGB (NHWC) and output dims are `(1, 512)` for GhostFaceNet and `(1, 512)` for ArcFace; assert L2 unit norm of outputs (embeddings must be pre-normalized or normalized in the export wrapper).
   - Latency targets: GhostFaceNet 4-6 ms, ArcFace 15-25 ms.
   - NOTE: the runtime venv `.venv` MUST NOT contain tensorflow/deepface/tf2onnx (per `00_ORCHESTRATOR.md`).
3. Implement `pipelines/inference_manager.py`:
   - `get_optimized_session(onnx_path)` builds the provider chain from `configs/model_config.yaml`:
     - Default: `CUDAExecutionProvider` (device_id 0) -> `CPUExecutionProvider` (`arena_extend_strategy='kSameAsRequested'`, `intra_op_num_threads=4`).
     - If `enable_rocm_workaround: true`: prepend `ROCMExecutionProvider` (device_id 0) and log the workaround; on failure, transparently fall back to CUDA, then CPU.
   - `FaceDetector`: wraps `models/yolo_face_custom.onnx`, returns `[x1, y1, x2, y2, conf]` normalized bboxes.
   - `EmbeddingExtractor` (GhostFaceNet 512D) and `EmbeddingExtractor` (ArcFace 512D): RGB 112x112x3 pre-processing, L2-normalized output vectors.
   - `HandTracker`: MediaPipe Hands wrapper producing 21 3D landmarks (CPU optimized).
4. Implement `core/tracker.py`:
   - Lightweight bounding box IoU tracker: matches new detections to active tracks when IoU > 0.3 (per `DESCRIPTION.md` Section 3).
   - Track cache stores the matched identity; subsequent frames bypass BOTH feature extraction and FAISS retrieval until track loss.
   - Assigns stable `track_id` per face; prunes lost tracks.
5. Quality gatekeeper: skip embedding extraction for low-confidence detections and for already-identified (cached) tracks.

## Verification Checkpoint
Create `tests/test_pipeline.py` against a sample image/video fixture (e.g., a labeled group photo):
- Asserts bounding box shape/range `[x1, y1, x2, y2, conf]` with values in 0.0-1.0.
- Asserts GhostFaceNet output shape `(1, 512)` and ArcFace output shape `(1, 512)`.
- Asserts L2 unit norm for both embeddings (`||v|| == 1.0 +/- 1e-5`).
- Asserts the IoU tracker assigns a stable `track_id` across consecutive frames.
- Asserts cached tracks bypass embedding extraction (call count of the embedder drops to 0 for matched tracks).
- Asserts `get_optimized_session` respects `provider_priority` from config (CUDA first when available, CPU fallback otherwise).