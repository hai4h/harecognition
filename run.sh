#!/usr/bin/env bash
# HARecognition launcher.
# onnxruntime-gpu 1.28 requires CUDA 12 runtime libs; this machine ships
# cu12 libs inside the venv (nvidia-* packages pulled in by faiss-gpu-cu12).
# LD_LIBRARY_PATH must be set at process start for dlopen to find them.
set -euo pipefail
cd "$(dirname "$0")"

NVD=".venv/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$NVD/cuda_runtime/lib:$NVD/cublas/lib:$NVD/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Auto-start the portable MongoDB if it is down and present (kiosk one-command
# launch). No-op when a server already listens on 27017 or mongod is absent.
if [ -x ".mongodb/bin/mongod" ]; then
    if ! (exec 3<>/dev/tcp/127.0.0.1/27017) 2>/dev/null; then
        echo "starting portable mongod (.mongodb/bin/mongod) ..."
        ./.mongodb/bin/mongod \
            --dbpath "$PWD/.mongodb/data" \
            --logpath "$PWD/.mongodb/mongod.log" \
            --bind_ip 127.0.0.1 --port 27017 \
            --fork --pidfilepath "$PWD/.mongodb/mongod.pid" >/dev/null
    fi
fi

if [ "${1:-}" = "--test" ]; then
    shift
    exec .venv/bin/python -m pytest "$@"
fi
exec .venv/bin/python main.py "$@"