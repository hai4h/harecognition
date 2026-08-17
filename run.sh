#!/usr/bin/env bash
# HARecognition launcher.
# onnxruntime-gpu 1.28 requires CUDA 12 runtime libs; this machine ships
# cu12 libs inside the venv (nvidia-* packages pulled in by faiss-gpu-cu12).
# LD_LIBRARY_PATH must be set at process start for dlopen to find them.
set -euo pipefail
cd "$(dirname "$0")"

NVD=".venv/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$NVD/cuda_runtime/lib:$NVD/cublas/lib:$NVD/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [ "${1:-}" = "--test" ]; then
    shift
    exec .venv/bin/python -m pytest "$@"
fi
exec .venv/bin/python main.py "$@"