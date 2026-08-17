# HARecognition Deployment Guide

Edge-native, dual-mode Face Recognition & Attendance. Single native PyQt6
process: ONNX Runtime inference + FAISS in-memory vector retrieval + MongoDB.
No network layer, gRPC, HTTP, or client-server split.

---

## 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11 | venvs created via `uv venv --python 3.11` |
| `uv` | >= 0.5 | package manager for both venvs |
| NVIDIA CUDA | 12.x / 13.x (tested 13.3) | primary acceleration; **CPU fallback guaranteed** |
| MongoDB | 7.x | Docker Compose (`mongo:7`) or portable `mongod` tarball |
| Docker | optional | only needed for the `docker compose up -d` path |

AMD ROCm is an **opt-in workaround only** (`enable_rocm_workaround: true` in
`configs/model_config.yaml`); it is never assumed and never part of the
default validation path.

---

## 2. Build

### 2.1 Runtime venv (`.venv`)

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements.txt
```

Contains: numpy, opencv-python-headless, PyQt6, onnxruntime-gpu,
faiss-gpu-cu12, mediapipe, pymongo, pyyaml, pytest, pyinstaller.

### 2.2 Build venv (`.venv-dev`) — model conversion only

```bash
uv venv --python 3.11 .venv-dev
uv pip install --python .venv-dev tensorflow deepface tf-keras tf2onnx onnxruntime
uv pip install --python .venv-dev --index-url https://download.pytorch.org/whl/cpu torch torchvision
uv pip install --python .venv-dev ultralytics
```

Used ONLY to convert deepface-downloaded `.h5` weights to ONNX with tf2onnx
and the YOLO `.pt` checkpoint to ONNX (torch/ultralytics); `onnxruntime` is
present only for export verification.
**NEVER imported at runtime and NEVER included in the packaged kiosk binary.**

### 2.3 Model conversion (Phase 3)

- `models/yolov8p-face-v2.pt` (committed source) -> `models/yolo_face_custom.onnx`
  via `scripts/convert_to_onnx.py` (dynamic batch, fixed 640x640)
- deepface-downloaded `.h5` weights -> `models/ghostfacenet_512.onnx`,
  `models/arcface_512.onnx` (both 512-d L2-normalized) via
  `scripts/export_embedding_onnx.py`
- MediaPipe hand model (7.8 MB): `curl -fsSL -o models/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"`

Run conversion with `CUDA_VISIBLE_DEVICES=""` (tf2onnx's grappler step
hard-fails when the TF build can't probe the host CUDA).

Generated ONNX/task files are gitignored (`models/*.onnx`, `models/*.task`)
and reproducible from the committed sources.

### 2.4 Verification gate

```bash
.venv/bin/python -m pytest tests/ -v
```

Every phase requires exit code 0 before proceeding (see
`.agent_tasks/00_ORCHESTRATOR.md`).

---

## 3. MongoDB

Either path gives `mongodb://localhost:27017` (see `configs/app_config.yaml`).
MongoDB is the single source of truth; FAISS indices are rebuilt from it at
boot.

### Option A: Docker Compose (preferred for target deployment)

```bash
docker compose up -d
```

Service: `mongo:7`, port `27017:27017`, named volume `mongodb_data`.

### Option B: Portable mongod (no Docker, used during development)

```bash
mkdir -p .mongodb/bin .mongodb/data
curl -fsSL -o .mongodb/mongodb.tgz \
  "https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu2204-7.0.14.tgz"
tar -xzf .mongodb/mongodb.tgz -C .mongodb --strip-components=1 && rm .mongodb/mongodb.tgz

.mongodb/bin/mongod \
  --dbpath /absolute/path/to/.mongodb/data \
  --logpath /absolute/path/to/.mongodb/mongod.log \
  --port 27017 --bind_ip 127.0.0.1 \
  --fork --pidfilepath /absolute/path/to/.mongodb/mongod.pid
```

**Important:** `--fork` requires ABSOLUTE paths for `--dbpath`, `--logpath`
and `--pidfilepath`; a relative pidfile kills the daemonized child.
Stop with: `kill $(cat .mongodb/mongod.pid)`.

---

## 4. Run

Launch with the provided launcher — it exports `LD_LIBRARY_PATH` so
onnxruntime-gpu (built against CUDA 13) and faiss-gpu-cu12 (CUDA 12) can both
load their runtimes. The CUDA 12 libs come from the venv (faiss-gpu-cu12's
nvidia-* packages); the CUDA 13 libs come from the host CUDA installation
(`/etc/ld.so.conf.d/cuda.conf` in the dev environment):

```bash
./run.sh                 # runs main.py
./run.sh --test          # runs the pytest verification gate
```

Or run manually with the CUDA 12 libs on the loader path:

```bash
NVD=.venv/lib/python3.11/site-packages/nvidia
LD_LIBRARY_PATH="$NVD/cuda_runtime/lib:$NVD/cublas/lib:$NVD/cuda_nvrtc/lib" \
  .venv/bin/python main.py
```

- **Mode 1 – Continuous Tracking:** YOLO face detection + GhostFaceNet 512D
  embeddings, telemetry at 2 Hz.
- **Mode 2 – Attendance:** face detection + hand gesture (MediaPipe Hands,
  21 landmarks) inside the guide zone, ArcFace 512D match (threshold >= 0.65),
  attendance logged to MongoDB `attendance_logs`.

Enroll users:

```bash
.venv/bin/python scripts/enroll_user.py --user-id EMP_0042 \
  --name "Nguyen Thanh Tung" --images path/to/faces/
```

---

## 5. Packaging (Phase 6 — complete)

One command produces the kiosk bundle (runtime venv ONLY):

```bash
./scripts/package.sh
```

It runs PyInstaller `--onedir --windowed` with:

- **Data bundled:** `configs/`, `models/` (all `.onnx` + `hand_landmarker.task`),
  `assets/`, `tests/fixtures/` (used by `--test-mode`).
- **CUDA runtimes bundled** under `_internal/nvidia/`:
  - `cuda13/lib` — `libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`,
    `libcurand.so.10` (onnxruntime CUDA EP needs these; taken from the host
    CUDA 13 install, NVIDIA redistributable runtime license).
  - `cuda12/lib` — faiss-gpu's nvidia-* packages (FAISS GPU).
- **Excluded explicitly:** `tensorflow`, `keras`, `deepface`, `tf2onnx`,
  `torch`, `pytest` (nothing from `.venv-dev` can leak into the binary —
  verified by `tests/test_e2e.py`).
- **Launcher:** `dist/harecognition/run-kiosk.sh` sets `LD_LIBRARY_PATH` to the
  bundled CUDA 13/12 dirs, so only the NVIDIA driver (`libcuda.so.1`) is
  required on the target machine — no system CUDA install.

Smoke-test the bundle (offscreen, synthetic camera):

```bash
QT_QPA_PLATFORM=offscreen ./dist/harecognition/run-kiosk.sh --test-mode
```

`tests/test_e2e.py::test_frozen_binary_smoke` runs the binary in `--test-mode`
and asserts exit 0 + clean shutdown whenever the bundle exists.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| ONNX Runtime slow / CUDA not used | ORT 1.28's CUDA EP is built against CUDA 13 (its plugin links `libcudart.so.13`); faiss needs CUDA 12. Dev: use `./run.sh` (cu12 libs) + host CUDA 13 via ld.so.conf. Bundle: `run-kiosk.sh` (ships both) |
| "No registered plugin EP device" warning | spurious in ORT 1.28; verify with `session.get_providers()` and latency |
| FAISS falls back to CPU | Engine auto-falls back if `faiss.get_num_gpus() == 0`; measured 0.167 ms GPU vs 1.06 ms CPU at 10k vectors |
| `mongod --fork` exits 1 | Use absolute paths for `--dbpath`, `--logpath`, `--pidfilepath` |
| Port 27017 busy | Check `ss -tlnp | grep 27017`; change `mongodb_uri` in `configs/app_config.yaml` and compose port mapping together |
| `pymongo` connection refused | Start MongoDB first (Section 3), then run the app/tests |
| Missing ONNX models | Run Phase 3 conversion scripts in `.venv-dev` (Section 2.3) |

---

## 7. Progress tracking

Per-phase status, successes, errors and resolutions are recorded in
`BUILD_PROGRESS.md` (updated after every completed phase; required by
`.agent_tasks/00_ORCHESTRATOR.md`).