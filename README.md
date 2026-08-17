# HARecognition

Edge-native, dual-mode Face Recognition & Attendance system.
Single native PyQt6 process: ONNX Runtime inference + FAISS in-memory vector
retrieval + MongoDB persistence. No network layer, gRPC, HTTP, or
client-server split. Runs fully offline on a local kiosk.

- **Mode 1 – Continuous Tracking:** YOLO Face detection + GhostFaceNet 512D
  embeddings tracked continuously, telemetry at 2 Hz.
- **Mode 2 – Attendance:** face detection + hand-gesture (MediaPipe Hands,
  21 landmarks) guide-zone confirmation, ArcFace 512D embedding matched
  against FAISS Index #2, attendance logged to MongoDB.

## Architecture

```
                     ┌──────────────────────────────────────────┐
  Camera ───────────▶│  PyQt6 kiosk (single native process)     │
                     │  CameraThread ──▶ AIWorkerThread          │
                     │    ├─ YOLOv8-p Face (ONNX, CUDA→CPU)     │
                     │    ├─ GhostFaceNet 512D / ArcFace 512D   │
                     │    ├─ MediaPipe Hands (Tasks API)        │
                     │    ├─ DualVectorEngine (FAISS GPU/CPU)   │
                     │    └─ ModeDispatcher (1/2, auto-cycle)   │
                     └──────────┬───────────────────────────────┘
                                │ enroll / attendance_logs
                     ┌──────────▼───────────┐
                     │ MongoDB (source of   │
                     │ truth; FAISS rebuilt │
                     │ from it at boot)     │
                     └──────────────────────┘
```

- FAISS Index #1: GhostFaceNet 512D (Mode 1), Index #2: ArcFace 512D
  (Mode 2), both `IndexFlatIP` on L2-normalized vectors (IP == cosine).
  GPU when CUDA present, CPU guaranteed fallback.
- Provider chain: `CUDAExecutionProvider` → `CPUExecutionProvider`
  (`kSameAsRequested`, 4 threads). ROCm is opt-in only via
  `configs/model_config.yaml` (`enable_rocm_workaround: false`).
- Crash safety: SIGKILL at any point loses nothing — indices rebuild from
  MongoDB at boot (verified by `tests/test_e2e.py`).
- Two venvs: `.venv` (runtime) and `.venv-dev` (build-only: tensorflow,
  deepface, tf2onnx, torch; used only for ONNX model conversion, never
  imported at runtime, excluded from the packaged binary).

## Setup

1. Runtime venv (Python 3.11):

   ```bash
   uv venv --python 3.11 .venv
   uv pip install --python .venv -r requirements.txt
   ```

2. MongoDB 7 (single source of truth; FAISS indices are rebuilt from it at
   boot) — Docker:

   ```bash
   docker compose up -d
   ```

   or portable tarball (no Docker host): see `DEPLOYMENT.md` §2.

3. Enroll identities:

   ```bash
   .venv/bin/python scripts/enroll_user.py --user-id U1 --name "Alice" \
       --ghost models/ghostfacenet_512.onnx --arcface models/arcface_512.onnx \
       --image photo.jpg
   ```

4. Run the verification harness:

   ```bash
   ./run.sh -- .venv/bin/python -m pytest tests/ -v
   ```

5. Launch the application (no camera attached? use `--test-mode`):

   ```bash
   ./run.sh                      # kiosk
   ./run.sh -- --test-mode       # synthetic camera smoke
   ```

## Model Provenance

All ONNX models were converted inside the build-only venv `.venv-dev` with
`CUDA_VISIBLE_DEVICES=""` (tf2onnx grappler cannot probe CUDA devices):

| Model                     | Source                                            | Conversion                       |
|---------------------------|---------------------------------------------------|----------------------------------|
| `yolo_face_custom.onnx`   | `yolov8p-face-v2.pt` (YOLO v8, Face) via ultralytics | `scripts/convert_to_onnx.py`     |
| `ghostfacenet_512.onnx`   | deepface 0.0.100 `GhostFaceNet` `.h5` weights (512D) | `scripts/export_embedding_onnx.py` (tf2onnx) |
| `arcface_512.onnx`        | deepface 0.0.100 `ArcFace` `.h5` weights (512D)    | same script                     |
| `hand_landmarker.task`    | Google MediaPipe Hands (Tasks API, 21 landmarks)   | downloaded, static asset        |

Embedding outputs verified: shape `(1, 512)`, L2 unit norm.

## Validation Matrix (measured on dev hardware)

| Provider | Face detect | GhostFaceNet | ArcFace | Mode 1 E2E | Peak RAM |
|----------|-------------|--------------|---------|------------|----------|
| CUDA     | 12.2 ms     | 3.63 ms      | 7.99 ms | 76.8 FPS   | 1.74 GB  |
| CPU-only | 40–70 ms    | —            | —       | 22.0 FPS   | 1.40 GB  |

FAISS search latency (exact `IndexFlatIP`):

| Scale   | GPU        | CPU       |
|---------|------------|-----------|
| 10,000  | 0.167 ms   | 1.06 ms   |
| 100,000 | 1.22 ms    | 15.73 ms  |

100k-scale gates use measured+headroom bounds (GPU ≤ 2.5 ms, CPU ≤ 20 ms);
10k keeps the strict 0.8 ms bound. Exact search scales linearly with the
identity count — approximate indices are intentionally not used.

### ROCm workaround

`enable_rocm_workaround` defaults to `false`. When enabled, the pipeline
registers a ROCm EP first with transparent fallback to CUDA/CPU; results are
logged by `inference_manager`. Not exercised on CUDA-only hardware (best-effort).

## Packaging (kiosk binary)

```bash
./scripts/package.sh          # PyInstaller onedir bundle (runtime venv only)
./dist/harecognition/run-kiosk.sh --test-mode   # smoke-test the bundle
```

The bundle ships its own CUDA 13 runtime (onnxruntime) and CUDA 12 runtime
(FAISS) under `_internal/nvidia/`; only the NVIDIA driver (`libcuda.so.1`)
is required on the target machine. `tensorflow`/`deepface`/`tf2onnx`/`torch`
are explicitly excluded. See `DEPLOYMENT.md` for the full deployment guide.

See `.agent_tasks/` for the phase-by-phase build plan and `BUILD_PROGRESS.md`
for verification history.