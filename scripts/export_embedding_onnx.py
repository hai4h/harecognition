"""Export deepface embedding models (.h5) to L2-normalized ONNX.

Runs ONLY in the BUILD venv (.venv-dev): requires tensorflow, deepface,
tf-keras, tf2onnx. NEVER run from the runtime venv.

Deepface downloads the weights on first use to ~/.deepface/weights/:
  - GhostFaceNet -> GhostFaceNet_W1.3_S1_ArcFace.h5 (HamadYA/GhostFaceNets)
  - ArcFace      -> arcface_weights.h5
Both models output a 512-d embedding; the exported ONNX graph ends with an
L2-normalization so every output vector has unit norm (IP == cosine).

Outputs:
  models/ghostfacenet_512.onnx
  models/arcface_512.onnx

Usage (from project root):
    .venv-dev/bin/python scripts/export_embedding_onnx.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODELS = {
    "GhostFaceNet": "models/ghostfacenet_512.onnx",
    "ArcFace": "models/arcface_512.onnx",
}


def _build_normalized(model):
    """Wrap a Keras model with an L2-normalization output layer."""
    import tensorflow as tf

    inputs = model.inputs
    norm = tf.keras.layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1), name="l2_normalize"
    )
    return tf.keras.Model(inputs=inputs, outputs=norm(model.outputs[0]))


def main() -> None:
    import numpy as np
    import tensorflow as tf
    import tf2onnx
    from deepface import DeepFace

    for model_name, out_rel in MODELS.items():
        # build_model downloads the .h5 weights on first use (~/.deepface/weights/);
        # the client wrapper exposes the Keras model via .model.
        keras_model = DeepFace.build_model(model_name=model_name, task="facial_recognition").model
        wrapped = _build_normalized(keras_model)

        # Verify inputs are NHWC 112x112x3 RGB.
        ins = wrapped.inputs[0]
        print(f"[{model_name}] input: {ins.name} shape={ins.shape} "
              f"dtype={ins.dtype}")
        assert tuple(ins.shape[1:]) == (112, 112, 3), f"unexpected input shape: {ins.shape}"

        out_path = os.path.join(ROOT, out_rel)
        onnx_model, _ = tf2onnx.convert.from_keras(wrapped, opset=17)
        import onnx

        onnx.save(onnx_model, out_path)

        # Verify with onnxruntime: output shape (1, 512) and L2 unit norm.
        import onnxruntime as ort

        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(0)
        probe = rng.standard_normal((1, 112, 112, 3)).astype(np.float32) / 255.0
        out = sess.run(None, {sess.get_inputs()[0].name: probe})[0]
        assert out.shape == (1, 512), f"[{model_name}] output shape {out.shape} != (1, 512)"
        norm = float(np.linalg.norm(out))
        assert abs(norm - 1.0) < 1e-5, f"[{model_name}] L2 norm {norm} != 1.0"
        print(f"[{model_name}] OK: {out_path} (norm={norm:.6f})")


if __name__ == "__main__":
    main()