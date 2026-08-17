# Phase 2: MongoDB Vault & Dual FAISS Vector Engine

## Objectives
1. Start MongoDB: `docker compose up -d` (service from Phase 1) and verify connectivity with `pymongo`.
2. Implement `core/database.py`:
   - MongoDB backend via `pymongo` (URI/db from `configs/app_config.yaml`).
   - Collection `users`: `user_id` (TEXT PK, e.g. "EMP_0042"), `name`, `ghost_vector` (array of 512 float32), `arcface_vector` (array of 512 float32), `created_at` (UTC ISO-8601). Create unique index on `user_id`.
   - Collection `attendance_logs`: `log_id` (ObjectId), `user_id`, `name`, `timestamp` (UTC ISO-8601), `mode`, `confidence`. Create index on `timestamp` for fast range queries.
   - Expose a thin `StorageBackend` interface (`enroll`, `delete`, `list`, `log_attendance`, `load_all`) so the backend remains swappable via configuration.
3. Implement `core/vector_engine.py` — `DualVectorEngine`:
   - Index #1: `faiss.IndexFlatIP(512)` for GhostFaceNet (Mode 1 continuous tracking).
   - Index #2: `faiss.IndexFlatIP(512)` for ArcFace (Mode 2 attendance, cosine threshold >= 0.65).
   - Boot-time synchronization: load ALL user records from MongoDB into both indices. Embeddings are stored L2-normalized, so inner product == cosine similarity (per `DESCRIPTION.md` Section 5).
   - Dynamic append on new user enrollment; removal on delete.
   - Methods: `search_mode1(query_vec, top_k)`, `search_mode2(query_vec, top_k, threshold)`, `enroll_user(user_id, name, ghost_vec, arcface_vec)`, `delete_user(user_id)`, `sync_from_db()`.
   - Mode 2 search MUST return `(user_id, name, confidence)` only when similarity >= threshold, otherwise `None`.
4. Implement `scripts/enroll_user.py`:
   - CLI: `--user-id EMP_0042 --name "Nguyen Thanh Tung" --images path/to/faces/`
   - Runs both embedding ONNX models (Phase 3 outputs) on each face image, L2-normalizes, stores dual vectors in MongoDB `users` AND appends to both FAISS indices.
5. Crash-safety: on application boot, FAISS indices are rebuilt entirely from MongoDB (`sync_from_db`) — MongoDB is the single source of truth; no separate disk checkpoint files are required.

## Verification Checkpoint
Create `tests/test_vault.py`:
- Enroll 10,000 synthetic identities (512D + 512D L2-normalized vectors) through `enroll_user`.
- Benchmark query latency at 10,000 vectors on both indices (must be <= 0.8 ms on CPU).
- Verify MongoDB metadata (user_id, name) matches FAISS search results for random probes.
- Verify Mode 2 threshold: a probe with similarity < 0.65 returns `None`.
- Verify delete purges records from both FAISS indices and MongoDB `users`.