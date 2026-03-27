
*The Fresh Repo URL : `https://github.com/SNiPERxDD/hetero-fedlearn-rest.git`*
*Iterative Federated Learning and Parameter Aggregation across Heterogeneous Clusters*

# Project Requirements Document (PRD)
**Project Title**: Distributed Model Training System via Iterative Federated Learning
**Target Environment**: Heterogeneous Cluster (1x macOS Master, 2x Windows Workers)
**Date**: March 2026

## 1. Executive Summary
This document outlines the architecture, execution phases, and failure mitigation strategies for a distributed machine learning system. The system implements Iterative Federated Learning (FedAvg) using a Master-Worker architecture over HTTP. It avoids heavyweight orchestration frameworks (like Kubernetes or Ray) in favor of a custom, lightweight REST-based synchronization layer. 

## 2. System Architecture
The system employs a decentralized training methodology synchronized by a central aggregator.

### 2.1 Component Roles
*   **Master Node (macOS):** 
    *   Acts as the network coordinator and state manager.
    *   Partitions and distributes the dataset to workers.
    *   Broadcasts global model weights at the start of each communication round.
    *   Aggregates worker updates via Federated Averaging (FedAvg).
    *   Evaluates the global model against a holdout validation set.
*   **Worker Nodes (Windows, Containerized):**
    *   Expose a REST API to receive instructions and model weights.
    *   Maintain a local instance of the model (`SGDClassifier`).
    *   Execute partial training (local epochs) on their isolated data partition.
    *   Return only computed gradients/weights (never raw data) to the Master.

### 2.2 Technology Stack
*   **Language:** Python 3.14+
*   **ML Library:** Scikit-learn 1.8+ (specifically `sklearn.linear_model.SGDClassifier` with `warm_start=True`)
*   **API Framework:** Flask 3.x (served via Waitress for Windows compatibility/stability)
*   **Containerization:** Docker 24+
*   **Data Serialization:** JSON (NumPy arrays explicitly cast via `.tolist()`)

## 3. Algorithmic Workflow (Iterative Federated Averaging)
The training occurs in discrete **Communication Rounds**.
For each round $t \in \{1, 2, ..., T\}$:
1.  Master broadcasts global weights $W_t$ and intercept $B_t$ to all active workers.
2.  Worker $k$ injects $W_t$ and $B_t$ into its local model.
3.  Worker $k$ trains on its local dataset partition $D_k$ for $E$ local epochs using `partial_fit`.
4.  Worker $k$ returns its updated weights $W_k'$ and $B_k'$ to the Master.
5.  Master aggregates updates using weighted averaging based on partition size $n_k$:
    $$W_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n_{total}} W_k'$$
    $$B_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n_{total}} B_k'$$

## 4. API Contracts & Data Models

### 4.1 Master -> Worker: Training Trigger (`POST /train_round`)
**Payload:**
```json
{
  "round_number": 1,
  "global_weights":[0.15, -0.42, 0.88, ...], 
  "global_intercept": [0.05],
  "local_epochs": 5
}
```
*(Note: Data $D_k$ is sent only once in a separate `/initialize` endpoint to prevent network saturation during rounds).*

### 4.2 Worker -> Master: Update Response
**Response (200 OK):**
```json
{
  "worker_id": "worker_1",
  "samples_processed": 5000,
  "updated_weights":[0.18, -0.39, 0.91, ...],
  "updated_intercept": [0.07],
  "local_loss": 0.342
}
```

## 5. Execution Plan (Zero-Failure Methodology)
To eliminate hardware and OS variables during development, the implementation must strictly follow this phased deployment strategy.

### Phase 1: Local Simulation (Single Machine)
**Objective:** Validate algorithmic convergence and serialization logic.
*   **Action 1:** Create `worker.py`. Configure it to accept a port argument via CLI.
*   **Action 2:** Spawn two worker instances on the Master machine in separate terminals (`python worker.py --port 5001` and `python worker.py --port 5002`).
*   **Action 3:** Create `master.py`. Hardcode worker endpoints to `http://127.0.0.1:5001` and `http://127.0.0.1:5002`.
*   **Action 4:** Execute the full $T$-round training loop. 
*   **Exit Criteria:** Master successfully aggregates weights, and global validation accuracy monotonically increases.

### Phase 2: Containerization (Docker)
**Objective:** Abstract OS dependencies (Windows vs. macOS) and ensure runtime consistency.
*   **Action 1:** Create a minimal `Dockerfile` for the Worker application.
    ```dockerfile
    FROM python:3.14-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    COPY worker.py .
    EXPOSE 5000
    CMD["waitress-serve", "--port=5000", "worker:app"]
    ```
*   **Action 2:** Build and run two containers on the Master machine, mapping to different host ports (`docker run -p 5001:5000 worker-image`, `docker run -p 5002:5000 worker-image`).
*   **Exit Criteria:** Phase 1 tests pass against the containerized workers.

### Phase 3: Physical Distributed Network
**Objective:** Deploy across heterogeneous hardware over Wi-Fi.
*   **Action 1:** Install Docker Desktop on the 2 Windows laptops.
*   **Action 2:** Deploy the Worker image to both Windows machines and run the containers exposing port 5000 (`docker run -p 5000:5000 worker-image`).
*   **Action 3:** Identify the IPv4 addresses of the Windows machines on the local Wi-Fi subnet (e.g., `192.168.1.101`, `192.168.1.102`).
*   **Action 4:** Update `master.py` to target the physical Windows IPs.
*   **Action 5:** Execute distributed training.
*   **Exit Criteria:** Master successfully communicates with physical workers, processes rounds without timeouts, and achieves target model accuracy.

## 6. Known Failure Nodes & Mitigations

| Failure Node | Cause | Strict Mitigation Strategy |
| :--- | :--- | :--- |
| **Silent Array Corruption** | `numpy.ndarray` cannot be natively serialized to JSON. | Master and Worker **must** explicitly cast via `.tolist()` before HTTP transmission and `numpy.array()` upon receipt. |
| **Model Reinitialization** | Scikit-learn models wipe their state when `.fit()` is called. | Use `SGDClassifier(warm_start=True)`. Training iterations must strictly use `.partial_fit(X, y, classes=np.unique(y))`. |
| **Network Latency/Timeout** | Wi-Fi packet drops or long local training times cause the `requests.post()` to time out. | Set explicit, high timeouts on the Master: `requests.post(url, json=data, timeout=120)`. Implement `try/except` blocks to retry failed requests 3 times before dropping a worker for that round. |
| **Windows Firewall Blocking** | Windows Defender blocks incoming requests to Docker ports. | Manually add an Inbound Rule in Windows Firewall to allow TCP traffic on port 5000, OR set the Wi-Fi network profile to "Private". |
| **Bandwidth Saturation** | Sending raw training data during every round crashes the network. | Send $X$ and $y$ partitions strictly **once** at Phase 0 (Initialization). Rounds 1 to $T$ must transfer **only** $W_t$ and $B_t$ arrays [1]. |

## 7. Directory Structure
```text
project_root/
│
├── master/
│   ├── master.py             # Orchestration, data partitioning, FedAvg math
│   ├── requirements.txt      # scikit-learn, numpy, requests
│   └── data/                 # Raw dataset (e.g., CSV)
│
├── worker/
│   ├── worker.py             # Flask/Waitress API, SGDClassifier local state
│   ├── requirements.txt      # Flask, waitress, scikit-learn, numpy
│   └── Dockerfile            # Container specification
│
└── config.json               # Defines total rounds, epochs, and worker IPs
```