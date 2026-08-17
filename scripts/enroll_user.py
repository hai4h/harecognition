"""Enroll a user from face images.

Runs both embedding ONNX models (Phase 3 outputs: GhostFaceNet 512D +
ArcFace 512D) on each face image, L2-normalizes, and stores dual vectors
in MongoDB `users` AND appends to both FAISS indices.

Usage:
    .venv/bin/python scripts/enroll_user.py --user-id EMP_0042 \
        --name "Nguyen Thanh Tung" --images path/to/faces/
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def embed_faces(images_dir: str):
    """Embed every face image with both models. Requires Phase 3 artifacts."""
    from core.database import load_app_config
    from pipelines.inference_manager import InferenceManager

    cfg = load_app_config()
    model_cfg_path = os.path.join(ROOT, "configs/model_config.yaml")
    manager = InferenceManager(model_cfg_path)
    ghost_vecs = []
    arcface_vecs = []
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        ghost, arcface = manager.embed_face(os.path.join(images_dir, fname))
        ghost_vecs.append(ghost)
        arcface_vecs.append(arcface)
    if not ghost_vecs:
        raise SystemExit(f"No images found in {images_dir}")
    import numpy as np

    return np.mean(ghost_vecs, axis=0), np.mean(arcface_vecs, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll a user (dual 512D embeddings)")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--images", required=True, help="Directory of face images")
    args = parser.parse_args()

    from core.database import create_storage_backend
    from core.vector_engine import DualVectorEngine

    ghost_vec, arcface_vec = embed_faces(args.images)
    engine = DualVectorEngine(create_storage_backend())
    engine.enroll_user(args.user_id, args.name, ghost_vec, arcface_vec)
    print(f"Enrolled {args.user_id} ({args.name}) - ghost/arcface vectors mean of "
          f"{len(os.listdir(args.images))} images")


if __name__ == "__main__":
    main()