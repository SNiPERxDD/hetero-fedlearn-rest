# Feature Extension PRD v1.1
**Project:** Iterative Federated Learning and Parameter Aggregation across Heterogeneous Clusters
**Extension Module:** DFS-Lite Storage Layer & Telemetry Dashboards
**Date:** March 2026

## 1. Executive Summary
This extension bridges the architectural gap between pure Distributed Computing (Federated Learning) and Distributed File Systems (DFS). To satisfy strict Big Data academic criteria, the Master node will now operate as a **Metadata NameNode**, and the Windows Workers will operate as persistent **DataNodes**.

Furthermore, zero-configuration onboarding scripts and real-time telemetry dashboards are introduced to provide visual proof of distributed storage, data locality, and parallel computation during live demonstrations.

---

## 2. Architectural Updates: The "DFS-Lite" Pivot
We are transitioning from in-memory data transmission to physical block storage. This proves the system adheres to the core DFS principle of "Data Locality" (moving computation to the data, not data to the computation).

### 2.1 Master Node (Acts as NameNode)
*   **Role Update:** Master no longer just "sends data." It generates discrete data chunks (Blocks) and maintains a **Block Metadata Map** (e.g., `Block_01 -> Worker_1_IP`).
*   **Replication (Optional):** The system can be configured to send `Block_01` to both Worker 1 and Worker 2. If Worker 1 drops, the Master reroutes computation to Worker 2.

### 2.2 Worker Node (Acts as DataNode)
*   **Role Update:** Upon receiving data via `/initialize`, the Worker **must write the data to local physical disk** inside a dedicated `/datanode_storage/` directory.
*   **Data Locality:** During `/train_round`, the worker must read from its local disk block into memory, execute `.partial_fit()`, and release the memory.

---

## 3. Feature 1: Real-Time Telemetry Dashboards
To prevent terminal-blindness and prove the cluster is active, both components will serve lightweight, built-in web dashboards.

### 3.1 Master Dashboard (`GET /`)
*   **Technology:** Flask rendering a single HTML file with Vanilla JS polling (`setInterval` to `GET /api/cluster_status`).
*   **Visuals Required:**
    *   **Global Epoch Chart:** Line graph showing Validation Accuracy vs. Communication Round.
    *   **DFS Block Map:** A UI table showing which Worker holds which Data Block (proving the NameNode logic).
    *   **Cluster Health:** Live Ping status of Windows Workers.

### 3.2 Worker Dashboard (`GET /`)
*   **Technology:** Flask + Vanilla JS.
*   **Visuals Required:**
    *   **Storage Metrics:** Size of local data blocks stored on disk (proving DataNode persistence).
    *   **Compute Metrics:** Current local loss and samples processed.

### ⚠️ Failure Node Mitigation (Thread Blocking)
**Failure Node:** Serving a web UI while running a heavy `while` loop for training will lock the Flask server, causing timeouts.
**Strict Mitigation:** The Master's training loop `master.run()` **must** be moved to a Python `threading.Thread(daemon=True)`. The main thread must strictly serve the Flask API and Dashboards. State will be shared via a thread-safe `State` dataclass.

---

## 4. Feature 2: Zero-Failure Onboarding Scripts
To eliminate environment issues on demo day, the deployment must be heavily automated with pre-flight checks.

### 4.1 Master Onboarding (`macOS / Linux`)
**Script:** `start_master.sh`
*   **Checks:** Verifies Python 3.14+ is installed.
*   **Action:** Creates isolated `venv`, installs `requirements.txt`, binds to `0.0.0.0:8080`, and opens the default web browser automatically.

### 4.2 Worker Onboarding (`Windows`)
**Script:** `start_worker.bat`
*   **Checks:** Executes `docker info >nul 2>&1` to explicitly verify the Docker Daemon is running. If not, it halts with a clear error ("Please start Docker Desktop").
*   **Cleanup:** Runs `docker rm -f worker-node` to kill dangling containers from previous failed runs.
*   **Action:** Builds the image and runs it with a **Volume Mount** (`-v %cd%\storage:/app/datanode_storage`).
*   *Why the Volume Mount?* This proves to the examiner that the data is physically resting on the Windows Host OS drive, perfectly mimicking HDFS.

---

## 5. API Contract Modifications

### 5.1 DFS Initialization (`POST /initialize`)
**Payload from Master:**
```json
{
  "block_id": "blk_90812",
  "worker_id": "worker_1",
  "features": [...],
  "labels": [...],
  "classes": [...]
}
```
**Worker Execution:**
Worker intercepts payload, writes to `./datanode_storage/blk_90812.csv`.
**Response (200 OK):**
```json
{
  "status": "block_committed",
  "block_id": "blk_90812",
  "bytes_written": 45012
}
```

### 5.2 Telemetry Endpoints
*   **Master `GET /api/status`**: Returns current round, accuracy, and Block Map.
*   **Worker `GET /api/status`**: Returns local disk usage (`os.path.getsize()`), worker ID, and readiness state.

---

## 6. Implementation Execution Plan

### Phase 1: Storage Layer Rewrite (DFS-Lite)
1.  Modify `worker.py` to include file I/O operations in `/initialize`.
2.  Modify `worker.py` `/train_round` to load from the `.csv` instead of holding data in RAM.
3.  Modify `master.py` to generate UUIDs for data chunks (representing HDFS Blocks) and maintain a dictionary of `Block -> Worker`.

### Phase 2: Asynchronous Master Orchestration
1.  Wrap the `FederatedMaster.run()` logic inside a background thread.
2.  Expose the Flask UI on port 8080.
3.  Add the JSON state-polling endpoint so the UI updates dynamically.

### Phase 3: Bootstrap Scripts
1.  Write `start_master.sh` with `chmod +x` instructions.
2.  Write `start_worker.bat` with the Docker lifecycle commands and the local host volume mount.

## 7. Directory Structure Update
```text
project_root/
│
├── config_extended.json      # NEW: DFS-lite config and dashboard settings
├── start_dashboard.py        # NEW: macOS/Linux full dashboard quick start
├── start_master.sh           # NEW: Mac bootstrap
├── start_worker.bat          # NEW: Windows bootstrap
├── website/                  # NEW: React project website
│
├── master/
│   ├── master.py             # PRESERVED: v1 baseline
│   ├── master_dfs.py         # NEW: Flask UI + Background Thread + NameNode metadata
│   ├── requirements_extended.txt
│   ├── templates/
│   │   └── index_dfs.html    # NEW: Master Real-time Dashboard
│   └── data/
│
├── worker/
│   ├── worker.py             # PRESERVED: v1 baseline
│   ├── worker_dfs.py         # NEW: Disk I/O Block Storage (DataNode)
│   ├── Dockerfile_extended   # NEW: DFS-lite worker image
│   ├── templates/
│   │   └── index_dfs.html    # NEW: Worker Dashboard
│   └── datanode_storage/     # NEW: Physical block storage directory
│
└── tests/
    └── test_dfs_lite_workflow.py  # NEW: DFS-lite integration coverage
```

## 8. Implementation Snapshot
*   **Doctrine Procedure Followed:** The working v1 runtime remains intact in `master/master.py` and `worker/worker.py`. The DFS-lite upgrade was implemented in copied variants so the known-good baseline stays unchanged.
*   **DFS-Lite Storage Layer:** Workers now write block CSV files to disk via `worker/worker_dfs.py`, expose local storage telemetry, and reload those blocks during `/train_round`.
*   **Asynchronous Master:** `master/master_dfs.py` now serves a Flask dashboard on `/`, exposes `/api/status` and `/api/start_training`, and runs the training loop inside a daemon thread.
*   **Bootstrap Path:** `start_dashboard.py`, `start_master.sh`, and `start_worker.bat` implement the bootstrap flow required by this extension, including Python version checks, virtual environment setup, Docker daemon checks, stale container cleanup, localhost worker health checks, and Windows host volume mounts.
*   **Website Layer:** `website/` now provides a clean React front-end package for presenting the architecture, telemetry model, onboarding path, and current validation state without depending on the AI Studio browser scaffold.
*   **Validation Status:** The DFS-lite worker persistence tests and asynchronous master tests pass under `pytest`, and a live local smoke run successfully served the dashboards, persisted block files to disk, and completed 10 rounds to 0.9737 validation accuracy.
