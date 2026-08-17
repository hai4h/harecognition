# 00_ORCHESTRATOR.md: Agent Execution Rules

## Operational Directives
1. Execute markdown phases in strict numerical order (01 -> 06).
2. Do NOT proceed to the next markdown file until the current phase's verification script passes with exit code 0.
3. Every package installation MUST use `uv pip install` inside the active venv and append to `installed_packages_log.txt`.
4. Update `BUILD_PROGRESS.md` after completing each phase (successes, errors, and resolutions).
5. All neural models (except MediaPipe Hands) MUST run through ONNX Runtime. The **designed** provider chain is `CUDAExecutionProvider -> CPUExecutionProvider`. ROCm is an **opt-in workaround only**: it is enabled exclusively via `configs/model_config.yaml` (`enable_rocm_workaround: true`), is never assumed, and is never part of the default validation path.
6. Embedding ONNX models MUST be produced from deepface-downloaded `.h5` weights converted with `tf2onnx` inside the **separate build venv** `.venv-dev`. The runtime venv (`.venv`) MUST NOT contain tensorflow/deepface/tf2onnx.
7. Persistent storage is MongoDB (collections `users`, `attendance_logs`). Vector retrieval is FAISS in-memory (`IndexFlatIP(512)` for GhostFaceNet, `IndexFlatIP(512)` for ArcFace). There is NO network layer, gRPC, HTTP, or client-server split — everything runs in one native PyQt6 process.

## Fallback & Error Handling
- If a compilation or test fails, debug within the current phase scope.
- Log error messages and resolutions in `BUILD_PROGRESS.md`.
- If the ROCm workaround flag is enabled and session creation fails, the inference manager MUST transparently fall back to CUDA, then CPU, and log the fallback.

## Architecture Reference
- The authoritative system specification is `DESCRIPTION.md` (EdgeFace-AI: edge-native, dual-mode Face Recognition & Attendance).
- System components: `core/` (database, vector_engine, tracker, gesture), `pipelines/` (inference_manager, mode_tracking, mode_attendance), `ui/` (main_window, video_widget, components), `models/`, `configs/`, `assets/`, `scripts/`.
- Primary acceleration target: NVIDIA CUDA. Guaranteed fallback: x86_64 CPU (AVX2/AVX-512). AMD ROCm: experimental workaround, disabled by default.