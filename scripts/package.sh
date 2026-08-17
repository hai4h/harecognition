#!/usr/bin/env bash
# Build the standalone kiosk bundle with the RUNTIME venv ONLY.
# .venv-dev (tensorflow/deepface/tf2onnx) must never be included.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NVIDIA=".venv/lib/python3.11/site-packages/nvidia"

.venv/bin/pyinstaller --noconfirm --clean --onedir --windowed --name harecognition \
  --add-data "configs:configs" \
  --add-data "models:models" \
  --add-data "assets:assets" \
  --add-data "tests/fixtures:tests/fixtures" \
  --add-binary "$NVIDIA/cuda_runtime/lib:./nvidia/cuda12/lib" \
  --add-binary "$NVIDIA/cublas/lib:./nvidia/cuda12/lib" \
  --add-binary "$NVIDIA/cuda_nvrtc/lib:./nvidia/cuda12/lib" \
  --add-binary "/opt/cuda/lib64/libcudart.so.13:./nvidia/cuda13/lib" \
  --add-binary "/opt/cuda/lib64/libcublas.so.13:./nvidia/cuda13/lib" \
  --add-binary "/opt/cuda/lib64/libcublasLt.so.13:./nvidia/cuda13/lib" \
  --add-binary "/opt/cuda/lib64/libcurand.so.10:./nvidia/cuda13/lib" \
  --collect-all mediapipe \
  --exclude-module tensorflow \
  --exclude-module keras \
  --exclude-module deepface \
  --exclude-module tf2onnx \
  --exclude-module torch \
  --exclude-module pytest \
  main.py

echo "OK: dist/harecognition/harecognition"

cat > dist/harecognition/run-kiosk.sh <<'EOF'
#!/usr/bin/env bash
# Deployable kiosk launcher: exposes the bundled CUDA 13 runtime (onnxruntime)
# and CUDA 12 runtime (FAISS) so the GPU provider chain works on a machine
# with only the NVIDIA driver installed (libcuda.so.1).
set -euo pipefail
BUNDLE="$(cd "$(dirname "$0")" && pwd)"
NVD="$BUNDLE/_internal/nvidia"

if [ -d "$NVD/cuda13/lib" ]; then
    export LD_LIBRARY_PATH="$NVD/cuda13/lib:$NVD/cuda12/lib:${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}"
fi

exec "$BUNDLE/harecognition" "$@"
EOF
chmod +x dist/harecognition/run-kiosk.sh
echo "OK: dist/harecognition/run-kiosk.sh"