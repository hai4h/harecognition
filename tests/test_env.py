"""Phase 1 verification: environment & scaffolding."""

import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    import cv2
    import numpy
    import faiss
    import mediapipe
    import pymongo
    import yaml
    import onnxruntime
    from PyQt6 import QtWidgets


def test_configs_parse():
    import yaml

    with open(os.path.join(ROOT, "configs/model_config.yaml")) as f:
        model_cfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, "configs/app_config.yaml")) as f:
        app_cfg = yaml.safe_load(f)

    assert model_cfg["provider_priority"] == ["CUDA", "CPU"]
    assert model_cfg["enable_rocm_workaround"] is False
    assert model_cfg["cpu_session_options"]["arena_extend_strategy"] == "kSameAsRequested"
    assert model_cfg["cpu_session_options"]["intra_op_num_threads"] == 4
    assert model_cfg["models"]["yolo_face"]["input"] == [640, 640, 3]
    assert model_cfg["models"]["ghostfacenet_512"]["input"] == [112, 112, 3]
    assert model_cfg["models"]["arcface_512"]["input"] == [112, 112, 3]

    assert app_cfg["gesture_min_frames"] == 5
    assert app_cfg["faiss_threshold_mode2"] == 0.65
    assert app_cfg["telemetry_throttle_hz"] == 2


def test_scaffold_tree():
    expected = [
        "assets/qss", "assets/sounds",
        "configs", "core", "models", "pipelines",
        "scripts", "ui/components", "tests",
    ]
    for rel in expected:
        assert os.path.isdir(os.path.join(ROOT, rel)), f"missing dir: {rel}"

    for pkg in ["core", "pipelines", "ui", "ui/components", "tests"]:
        assert os.path.isfile(os.path.join(ROOT, pkg, "__init__.py")), f"missing {pkg}/__init__.py"

    for f in ["main.py", "requirements.txt", "README.md", "docker-compose.yml",
              "installed_packages_log.txt", "configs/app_config.yaml",
              "configs/model_config.yaml"]:
        assert os.path.isfile(os.path.join(ROOT, f)), f"missing file: {f}"


def test_build_venv_isolation():
    dev_py = os.path.join(ROOT, ".venv-dev/bin/python")
    assert os.path.exists(dev_py), ".venv-dev missing"

    for pkg in ["tensorflow", "deepface", "tf2onnx"]:
        res = subprocess.run(
            [dev_py, "-c", f"import {pkg}"],
            capture_output=True, text=True,
        )
        assert res.returncode == 0, f".venv-dev cannot import {pkg}"

    runtime_py = os.path.join(ROOT, ".venv/bin/python")
    res = subprocess.run(
        [runtime_py, "-c", "import tensorflow"],
        capture_output=True, text=True,
    )
    assert res.returncode != 0, "runtime venv must NOT import tensorflow"


def test_cuda_provider_available():
    import onnxruntime as ort

    assert "CUDAExecutionProvider" in ort.get_available_providers()


def test_docker_compose_config():
    docker = subprocess.run(["which", "docker"], capture_output=True, text=True)
    if docker.returncode != 0:
        return  # Docker unavailable; deferred to Phase 2
    res = subprocess.run(
        ["docker", "compose", "config", "--quiet"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr


def test_requirements_track_runtime_venv():
    with open(os.path.join(ROOT, "requirements.txt")) as f:
        reqs = f.read()
    assert "tensorflow" not in reqs
    assert "deepface" not in reqs
    assert "tf2onnx" not in reqs
    assert "onnxruntime-gpu" in reqs