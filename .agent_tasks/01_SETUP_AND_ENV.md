# Phase 1: Environment & Project Scaffolding

## Objectives
1. Initialize the runtime venv: `uv venv --python 3.11` (creates `.venv`).
2. Install runtime baseline packages with `uv pip install` (log each to `installed_packages_log.txt`):
   - `numpy`, `opencv-python-headless` (headless avoids Qt DLL conflicts with PyQt6)
   - `PyQt6`
   - `onnxruntime-gpu` on CUDA targets, otherwise `onnxruntime` (CPU). The provider chain in `configs/model_config.yaml` must list `CUDA` then `CPU` by default.
   - `faiss-cpu` (FAISS RAM vector indices)
   - `mediapipe` (21-landmark hand tracking)
   - `pymongo` (MongoDB driver)
   - `pyyaml` (config parsing)
   - `pytest` (verification harness)
   - `pyinstaller` (final packaging, Phase 6)
3. Initialize the **build venv** for model conversion only: `uv venv --python 3.11 .venv-dev` and install `tensorflow`, `deepface`, `tf2onnx` there. This venv is NEVER imported at runtime and NEVER included in the packaged kiosk binary. Record its packages in `installed_packages_log.txt` under a `[.venv-dev]` section.
4. Scaffold the complete project tree from `DESCRIPTION.md` Section 7:
   ```text
   edgeface-ai/
   ├── assets/qss/            # dark_theme.qss, kiosk_overlay.qss
   ├── assets/sounds/         # success.wav, error.wav
   ├── configs/               # app_config.yaml, model_config.yaml
   ├── core/                  # database.py, vector_engine.py, tracker.py, gesture.py
   ├── models/                # yolo_face_custom.onnx, ghostfacenet_512.onnx, arcface_512.onnx
   ├── pipelines/             # inference_manager.py, mode_tracking.py, mode_attendance.py
   ├── scripts/               # convert_to_onnx.py, export_embedding_onnx.py, enroll_user.py, benchmark_pipeline.py
   ├── ui/components/         # attendance_card.py, stats_panel.py
   ├── main.py
   ├── requirements.txt
   └── README.md
   ```
5. Create `configs/app_config.yaml`:
   - `camera_id`, `target_fps`, `camera_resolution`
   - `mode` (1 or 2, initial UI state)
   - Mode 2: `guide_zone` (normalized central region), `gesture_min_frames: 5`
   - FAISS/MongoDB: `faiss_threshold_mode2: 0.65`, `mongodb_uri`, `mongodb_db`
   - UI: telemetry throttle `2 Hz`, audio feedback toggles
6. Create `configs/model_config.yaml`:
   - Model paths + input dimensions (`yolo_face` 640x640, `ghostfacenet_512` 112x112x3 RGB, `arcface_512` 112x112x3 RGB)
   - `provider_priority: [CUDA, CPU]` and `enable_rocm_workaround: false` (ROCm workaround entry `ROCMExecutionProvider` present but only prepended when the flag is `true`)
   - CPU session options: `arena_extend_strategy: kSameAsRequested`, `intra_op_num_threads: 4`
7. Add `docker-compose.yml` for MongoDB 7 (service `mongodb`, image `mongo:7`, port `27017:27017`, named volume for persistence). Document `docker compose up -d` in `README.md`.
8. Generate `requirements.txt` from the runtime venv (`.venv` only; `.venv-dev` is build-only).

## Verification Checkpoint
Create `tests/test_env.py` and run it:
- Imports `cv2`, `numpy`, `PyQt6.QtWidgets`, `onnxruntime`, `faiss`, `mediapipe`, `pymongo`, `yaml` successfully.
- Parses both YAML configs; asserts `provider_priority == ["CUDA", "CPU"]` and `enable_rocm_workaround == false`.
- Asserts all directories from the scaffold tree exist.
- Asserts `.venv-dev` exists and contains `tensorflow`/`deepface`/`tf2onnx` while the runtime venv does NOT import `tensorflow`.
- Asserts `docker compose config` validates (if Docker is available; otherwise defer to Phase 2).