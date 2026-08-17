# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('configs', 'configs'), ('models', 'models'), ('assets', 'assets'), ('tests/fixtures', 'tests/fixtures')]
binaries = [('.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib', 'nvidia/cuda12/lib'), ('.venv/lib/python3.11/site-packages/nvidia/cublas/lib', 'nvidia/cuda12/lib'), ('.venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib', 'nvidia/cuda12/lib'), ('/opt/cuda/lib64/libcudart.so.13', 'nvidia/cuda13/lib'), ('/opt/cuda/lib64/libcublas.so.13', 'nvidia/cuda13/lib'), ('/opt/cuda/lib64/libcublasLt.so.13', 'nvidia/cuda13/lib'), ('/opt/cuda/lib64/libcurand.so.10', 'nvidia/cuda13/lib')]
hiddenimports = []
tmp_ret = collect_all('mediapipe')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tensorflow', 'keras', 'deepface', 'tf2onnx', 'torch', 'pytest'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='harecognition',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='harecognition',
)
