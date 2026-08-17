Here is the complete, comprehensive architecture specification file written in English Markdown. You can save this file directly as `ARCHITECTURE.md` or `README.md` in your newly structured repository.

---

# Architecture Specification: Edge-Native Dual-Mode Face Recognition & Attendance System

## 1. Executive Summary

**EdgeFace-AI** is an edge-native, zero-latency computer vision system designed for standalone kiosks, physical security checkpoints, and enterprise time-attendance terminals. 

By eliminating the network overhead of client-server (HTTP/WebRTC) models and hosting the entire pipeline directly on local hardware using **PyQt6**, **ONNX Runtime**, and **FAISS in-memory vector indices**, the system achieves native camera frame rates (30–60+ FPS) with zero frame-dropping and sub-millisecond query latency.

```
+-----------------------------------------------------------------------------------+
|                               SYSTEM HIGHLIGHTS                                   |
+-----------------------------------------------------------------------------------+
|  * Native IPC / Shared Memory: Zero network serialization/compression overhead.  |
|  * Hardware Agnostic: Seamless execution on AMD ROCm, NVIDIA CUDA, or x86 CPU.   |
|  * Dual Inference Pipeline: GhostFaceNet (Realtime) vs. ArcFace (SOTA Biometrics).|
|  * Natural Interaction: MediaPipe Hands for zero-training gesture confirmation.  |
+-----------------------------------------------------------------------------------+
```

---

## 2. High-Level System Architecture

The application runs as a native desktop process leveraging a decoupled, multi-threaded pipeline via Qt’s inter-thread signal/slot architecture over shared memory (`numpy.ndarray`).

```
+---------------------------------------------------------------------------------------------------+
|                                   PyQt6 MAIN UI THREAD (GUI)                                      |
|  - Real-time Video Rendering (QImage / QPixmap on QLabel)                                         |
|  - Mode Switching & Visual State Machine ("Align Face" -> "Show Gesture" -> "Success")          |
|  - Low-Frequency Telemetry (FPS, People Count, System Status throttled to 2 Hz)                   |
+---------------------------------▲---------------------------------------▲-------------------------+
                                  │ (Frame Pixmap Signal)                 │ (Inference Meta Signal)
                                  │                                       │
+---------------------------------┴---------+           +-----------------┴-------------------------+
|           CAMERA QTHREAD                  |           |            AI WORKER QTHREAD              |
|  - Dedicated cv2.VideoCapture loop        |           |  - Hardware-Agnostic ONNX Runtime         |
|  - Continuous frame pulling (30-60 FPS)   |──────────►|  - Lightweight YOLO Face Detector         |
|  - Zero-Copy Memory Pointers              | (Shared   |  - MediaPipe 21-Landmark Hand Tracker     |
|  - Backpressure Frame Dropping            |  Memory)  |  - Bounding Box IoU Tracker Cache         |
+-------------------------------------------+           +-----------------┬-------------------------+
                                                                          │
                                            ┌─────────────────────────────┴─────────────────────────┐
                                            ▼                                                       ▼
                         ┌────────────────────────────────────┐                  ┌────────────────────────────────────┐
                         │   FAISS RAM VECTOR DATABASE #1     │                  │    FAISS RAM VECTOR DATABASE #2    │
                         │   - Model: GhostFaceNet (512D)     │                  │    - Model: ArcFace SOTA (512D)    │
                         │   - Target: Mode 1 (Continuous)    │                  │    - Target: Mode 2 (Attendance)   │
                         │   - Query Latency: < 0.01 ms       │                  │    - Metric: Cosine Sim > 0.65     │
                         └────────────────────────────────────┘                  └──────────────────┬─────────────────┘
                                                                                                    │
                                                                                                    ▼
                                                                                 ┌────────────────────────────────────┐
                                                                                 │       PERSISTENT STORAGE           │
                                                                                 │  - MongoDB: User Profiles,         │
                                                                                 │    Dual Embeddings, Access Logs    │
                                                                                 └────────────────────────────────────┘
```

---

## 3. Core Operational Modes

The core innovation is the decoupling of **Situational Awareness (Tracking)** from **High-Security Verification (Attendance)**.

```
                              Operational Mode Branching
                              
                                [ Input Video Frame ]
                                          │
                                 [ Custom YOLO Face ]
                                          │
                         +----------------┴----------------+
                         │                                 │
                 [ MODE 1 Active ]                 [ MODE 2 Active ]
                         │                                 │
              [ IoU Bounding Box Match? ]           [ Face in Guide Zone? ]
                   /          \                            │ (Yes)
             (Yes)/            \(No)              [ MediaPipe 21 Hand Joints ]
                 /              \                          │
       [ Return Cached ]  [ Crop Face ROI ]       [ Stable Gesture >= 5 frames? ]
       [     Name      ]         │                         │ (Yes)
                                 │                   [ Crop Face ROI ]
                       [ GhostFaceNet 512D ]               │
                                 │                   [ ArcFace SOTA 512D ]
                       [ FAISS Index #1 (IP) ]             │
                                 │                   [ FAISS Index #2 (IP) ]
                       [ Return Name + Cache ]             │
                                                     [ Cosine Sim >= Threshold ]
                                                           │
                                                     [ Write Log to MongoDB ]
                                                     [ Trigger UI Confirmation ]
```

### Mode 1: Continuous Tracking & Identification
* **Objective:** Scan, track, and label multiple individuals across the entire camera field of view in real time at high FPS.
* **Pipeline:**
  1. **Detection:** Custom lightweight YOLO Face detector scans the full frame.
  2. **Tracking:** An **IoU (Intersection-over-Union) Tracker** evaluates bounding box drift against active tracks ($IoU > 0.3$).
  3. **Feature Extraction:** If a face is *new*, it is cropped and fed into the ultra-lightweight **GhostFaceNet ONNX (512-dimensional)**.
  4. **Vector Retrieval:** The 512D vector queries **FAISS Index #1** ($<0.01\text{ ms}$).
  5. **Caching:** The matched identity is cached to the track. Subsequent frames bypass both neural feature extraction and vector retrieval entirely until track loss.

### Mode 2: High-Precision Time Attendance
* **Objective:** Zero-false-acceptance biometric attendance logging triggered via non-intrusive gesture confirmation.
* **Pipeline:**
  1. **Zone Alignment:** UI defines a central target zone. Faces outside this boundary are ignored.
  2. **Gesture Validation:** **Google MediaPipe Hands** tracks the user's hand skeleton (21 3D landmarks).
  3. **Deterministic Confirmation:** A heuristic state machine checks for a confirmation gesture (e.g., Open Palm or Thumbs Up) sustained continuously for $\ge 5$ consecutive frames ($\sim 150\text{ ms}$) to prevent accidental triggers.
  4. **High-Precision Inference:** The aligned face is cropped and processed by **ArcFace SOTA ONNX (512-dimensional)**.
  5. **Biometric Search:** The 512D vector queries **FAISS Index #2** with a strict verification threshold ($\text{Cosine Similarity} \ge 0.65$).
  6. **Persistence:** On match, an immutable record containing `user_id`, `name`, and a UTC `timestamp` is written to MongoDB's `attendance_logs` collection, triggering UI visual/auditory feedback.

---

## 4. Machine Learning & Vision Pipeline

| Task | Component / Model | Output Dimension | Framework & Provider | Latency Target |
| :--- | :--- | :--- | :--- | :--- |
| **Face Detection** | Custom Pruned YOLO Face | Bounding Boxes `[x1, y1, x2, y2, conf]` | PyTorch / ONNX Runtime | $8 - 12\text{ ms}$ |
| **Hand Tracking** | MediaPipe Hands | 21 3D Landmark Coordinates | MediaPipe / CPU optimized | $3 - 5\text{ ms}$ |
| **Fast Embedding** | GhostFaceNet | $1 \times 512$ L2-Normalized Vector | ONNX Runtime (ROCm/CUDA/CPU) | $4 - 6\text{ ms}$ |
| **SOTA Embedding** | ArcFace (ResNet/MobileFaceNet) | $1 \times 512$ L2-Normalized Vector | ONNX Runtime (ROCm/CUDA/CPU) | $15 - 25\text{ ms}$ |

### Hardware Abstraction Strategy
All neural models (excluding MediaPipe) are stored in open **ONNX** format. At runtime, the inference manager queries hardware availability and sets up the execution provider chain dynamically:

```python
import onnxruntime as ort

def get_optimized_session(onnx_path: str) -> ort.InferenceSession:
    providers = [
        ('ROCMExecutionProvider', {'device_id': 0}), # AMD Radeon GPUs
        ('CUDAExecutionProvider', {'device_id': 0}), # NVIDIA GeForce/RTX GPUs
        ('CPUExecutionProvider', {                   # x86_64 AVX2/AVX512
            'arena_extend_strategy': 'kSameAsRequested',
            'intra_op_num_threads': 4
        })
    ]
    return ort.InferenceSession(onnx_path, providers=providers)
```

---

## 5. Database & Vector Memory Architecture

```
                    [ MongoDB Persistent Layer ]
                                 │
       ┌─────────────────────────┴─────────────────────────┐
       ▼                                                   ▼
[ Collection: users ]                     [ Collection: attendance_logs ]
  - user_id: "EMP_0042"                     - log_id: ObjectId(...)
  - name: "Nguyen Thanh Tung"               - user_id: "EMP_0042"
  - ghost_vector: [512 floats]              - timestamp: ISODate(...)
  - arcface_vector: [512 floats]            - mode: "GESTURE_CONFIRMED"
  - created_at: ISODate(...)                - confidence: 0.892
       │
       │ (1. Synchronized at application boot)
       │ (2. Appended dynamically on new user enrollment)
       ▼
[ FAISS In-Memory Layer (RAM) ]
  ├─► FAISS Index #1 (GhostFaceNet): IndexFlatIP(512)
  └─► FAISS Index #2 (ArcFace):      IndexFlatIP(512)
```

### Vector Index Optimization
Because embeddings are generated with an internal $L_2$ normalization layer ($\|v\|_2 = 1$), cosine similarity is mathematically equivalent to the inner dot product:
$$\text{CosineSim}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = u \cdot v$$
Using `faiss.IndexFlatIP` allows SIMD vector units (AVX2 / AVX-512) to compute dot products at sub-microsecond speeds per frame without running explicit square roots or normalizations during query time.

---

## 6. Native Multithreading Engine (PyQt6)

To prevent UI lockup and maintain a rock-solid 60 FPS display, computational tasks are strictly separated across distinct operating system threads via `QThread`.

```
                    Thread Synchronization Topology
                    
   [ CameraThread ]               [ AIWorkerThread ]              [ MainUIThread ]
          │                               │                              │
          │ Frame Capture                 │                              │
          ├──────────────────────────────►│                              │
          │ (pyqtSignal: np.ndarray)      │ Frame Processing             │
          │                               ├─────────────────────────────►│
          │                               │ (pyqtSignal: Frame + Meta)   │ Render Pixmap
          │                               │                              │ Paint Widgets
          │ Frame Capture                 │                              │
          ├──────────────────────────────►│ (Busy? Drop Frame)           │
          │ (Backpressure Protection)     │                              │
```

1. **`CameraThread`**: Interacts with the hardware bus via OpenCV `VideoCapture`. Emits raw `np.ndarray` frames at camera-native refresh rates.
2. **`AIWorkerThread`**: Owns model memory contexts (ONNX sessions, FAISS indices, MediaPipe trackers). Consumes incoming frames, computes state transitions, renders bounding box overlays directly onto the matrix via OpenCV, and emits the annotated frame with metadata.
3. **`MainUIThread`**: Executes the Qt event loop. Converts the incoming matrix to `QPixmap`, updates status indicators, and renders the kiosk HUD.

---

## 7. Recommended Project Layout

```text
edgeface-ai/
├── assets/
│   ├── qss/                          # Qt stylesheets (Modern Kiosk themes)
│   │   ├── dark_theme.qss
│   │   └── kiosk_overlay.qss
│   └── sounds/                       # Audio confirmation feedback
│       ├── success.wav
│       └── error.wav
├── configs/
│   ├── app_config.yaml               # Thresholds, active camera ID, ports
│   └── model_config.yaml             # Input dimensions, ONNX provider priorities
├── core/
│   ├── __init__.py
│   ├── database.py                   # MongoDB driver & connection pools
│   ├── vector_engine.py              # FAISS Dual-Index controller
│   ├── tracker.py                    # Lightweight bounding box IoU Tracker
│   └── gesture.py                    # MediaPipe Hands deterministic gesture logic
├── models/
│   ├── yolo_face_custom.onnx         # Latency-optimized Face Detector
│   ├── ghostfacenet_512.onnx         # 512D Fast Embedding Model (Mode 1)
│   └── arcface_512.onnx              # 512D SOTA Embedding Model (Mode 2)
├── pipelines/
│   ├── __init__.py
│   ├── inference_manager.py          # Unified ONNX execution context
│   ├── mode_tracking.py              # Mode 1 pipeline controller
│   └── mode_attendance.py            # Mode 2 state-machine pipeline controller
├── scripts/
│   ├── convert_to_onnx.py            # Model export/quantization tools
│   ├── enroll_user.py                # Dual-embedding enrollment script
│   └── benchmark_pipeline.py         # Latency & throughput profiler
├── ui/
│   ├── __init__.py
│   ├── main_window.py                # QMainWindow shell & view manager
│   ├── video_widget.py               # Optimized QPainter viewport
│   └── components/
│       ├── attendance_card.py        # Mode 2 user confirmation card
│       └── stats_panel.py            # Throttled telemetry panel
├── main.py                           # Application bootstrap & thread wiring
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## 8. Implementation Roadmap

```
===================================================================================
PHASE 1: MODEL OPTIMIZATION & VECTOR SETUP
-----------------------------------------------------------------------------------
[ ] Prune and train custom lightweight YOLO Face model.
[ ] Convert GhostFaceNet (512D) and ArcFace (512D) weights to ONNX format.
[ ] Implement `core/vector_engine.py` to instantiate and synchronize Dual FAISS indices.
[ ] Create `scripts/enroll_user.py` to generate dual embeddings for MongoDB records.

===================================================================================
PHASE 2: NATIVE THREADING & CAMERA PIPELINE
-----------------------------------------------------------------------------------
[ ] Construct `CameraThread` with OpenCV hardware frame capture.
[ ] Build decoupled `AIWorkerThread` with non-blocking backpressure protection.
[ ] Implement zero-copy matrix conversion pipeline (`np.ndarray` -> `QImage` -> `QPixmap`).

===================================================================================
PHASE 3: DUAL-MODE LOGIC & GESTURE INTEGRATION
-----------------------------------------------------------------------------------
[ ] Implement `core/tracker.py` for IoU-based face tracking and feature caching.
[ ] Implement `core/gesture.py` using MediaPipe Hands (21-point geometric checks).
[ ] Build Mode 1 (Continuous Tracking + GhostFaceNet + FAISS-1).
[ ] Build Mode 2 (Target Zone + Sustained Gesture Verification + ArcFace + FAISS-2).

===================================================================================
PHASE 4: UI/UX & HARDWARE ACCELERATION
-----------------------------------------------------------------------------------
[ ] Apply modern kiosk QSS styles, guide boxes, and visual confirmation alerts.
[ ] Validate ONNX dynamic provider fallback on target hardware (ROCm / CUDA / CPU).
[ ] Connect MongoDB attendance transaction logging on successful Mode 2 matches.

===================================================================================
PHASE 5: PACKAGING & DEPLOYMENT
-----------------------------------------------------------------------------------
[ ] Run end-to-end benchmark suite (`scripts/benchmark_pipeline.py`).
[ ] Compile standalone kiosk binary using PyInstaller / Nuitka.
===================================================================================
```
