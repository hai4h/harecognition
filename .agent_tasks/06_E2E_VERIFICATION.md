# Phase 6: End-to-End System Verification & Hardening

## Objectives
1. Run the full end-to-end benchmark (`scripts/benchmark_pipeline.py`):
   - Ingest live webcam OR a test MP4 loop for 60 seconds.
   - Measure glass-to-glass latency, inference FPS, and RAM consumption (via `resource`/`psutil`).
   - Validate the ONNX provider chain on target hardware: **CUDA and CPU are REQUIRED** to pass; ROCm is only exercised when `enable_rocm_workaround: true` (best-effort, transparent fallback to CUDA/CPU required, results logged in `README.md`).
2. Verify 100k vector database scale on BOTH indices:
   - Populate MongoDB + FAISS with 100,000 synthetic identities (512D + 512D L2-normalized).
   - Assert search latency <= 0.8 ms on CPU and <= 0.1 ms on GPU for both indices.
3. Cold-Start and Fault Recovery Test:
   - Terminate the application process abruptly (SIGKILL).
   - Restart and assert both FAISS indices auto-recover from MongoDB sync with ZERO data loss (MongoDB is the single source of truth per Phase 2).
4. Attendance log integrity test:
   - Verify MongoDB `attendance_logs` records contain valid `user_id`, `name`, UTC `timestamp`, `mode`, and `confidence`; verify Mode 1 writes NO attendance logs.
5. Packaging:
   - Compile the standalone kiosk binary with PyInstaller / Nuitka using the RUNTIME venv ONLY (`.venv-dev`/tensorflow must NOT be included; `--exclude-module tensorflow,deepface,tf2onnx`).
   - Smoke-test the binary in `--test-mode` on a clean machine.
6. Finalize project documentation (`README.md`: architecture summary, setup, model provenance — deepface .h5 -> tf2onnx -> ONNX, CUDA/CPU validation matrix, ROCm workaround notes) and the package list (`installed_packages_log.txt`); mark all phases complete in `BUILD_PROGRESS.md`.

## Acceptance Criteria
- Zero frame queue accumulation: 30+ FPS sustained end-to-end on CUDA; 15+ FPS on CPU-only.
- Peak memory footprint <= 2.5 GB RAM.
- Clean shutdown on `SIGINT` / `SIGTERM` and on window `closeEvent`.
- Mode 2 false-acceptance rate at zero with cosine threshold >= 0.65 (verified with negative probes against 100k identities).
- FAISS rebuild from MongoDB after crash: zero data loss.