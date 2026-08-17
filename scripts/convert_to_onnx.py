"""Convert the YOLO Face PyTorch checkpoint to ONNX.

Runs in the BUILD venv (.venv-dev): requires torch + ultralytics.
Output: models/yolo_face_custom.onnx (dynamic batch, fixed 640x640 input).

Usage (from project root):
    .venv-dev/bin/python scripts/convert_to_onnx.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SRC = os.path.join(ROOT, "models/yolov8p-face-v2.pt")
OUT = os.path.join(ROOT, "models/yolo_face_custom.onnx")
IMGSZ = 640


def main() -> None:
    from ultralytics import YOLO

    if not os.path.isfile(SRC):
        raise SystemExit(f"Source checkpoint not found: {SRC}")
    model = YOLO(SRC)
    exported = model.export(
        format="onnx",
        imgsz=IMGSZ,
        dynamic=True,
        opset=17,
        simplify=False,
    )
    if os.path.abspath(exported) != os.path.abspath(OUT):
        if os.path.isfile(OUT):
            os.remove(OUT)
        os.rename(exported, OUT)
    assert os.path.isfile(OUT), f"export failed, expected {OUT}"
    print(f"Exported YOLO face -> {OUT}")


if __name__ == "__main__":
    main()