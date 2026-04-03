# hetero-fedlearn-rest

Distributed iterative federated learning over HTTP for a heterogeneous cluster with one master node and multiple worker nodes. The repository now contains both the preserved v1 baseline and the v1.1 DFS-lite extension, which adds disk-backed block storage, asynchronous dashboards, and zero-failure bootstrap scripts.

## Operating Modes

- `v1 baseline`: `master/master.py`, `worker/worker.py`, and `config.json`
- `v1.1 DFS-lite extension`: `master/master_dfs.py`, `worker/worker_dfs.py`, and `config_extended.json`

## Architecture

- `master/master.py` orchestrates dataset preparation, worker initialization, round execution, retry-aware HTTP communication, FedAvg aggregation, and validation.
- `worker/worker.py` exposes `/health`, `/initialize`, and `/train_round` and keeps local model state across communication rounds.
- `master/master_dfs.py` upgrades the master into a NameNode-style service with a background training thread, live telemetry APIs, and a dashboard on `/`.
- `worker/worker_dfs.py` upgrades the worker into a DataNode-style service that persists blocks to disk, reloads them per round, and serves storage or compute telemetry on `/`.
- `config.json` defines the baseline dataset, model hyperparameters, training schedule, timeouts, retries, and worker endpoints.
- `config_extended.json` defines the DFS-lite dashboard poll rate, block replication factor, and the extended worker topology.
- `worker/Dockerfile` packages the baseline worker on `python:3.14-slim` with Flask, Waitress, scikit-learn, and a native Docker health check.
- `worker/Dockerfile_extended` packages the DFS-lite worker with templates and a persistent datanode storage directory.
- `scripts/windows/onboard_worker.ps1` automates Windows worker setup for firewall rules, optional network profile hardening, image build or pull, container launch, and health verification.
- `start_dashboard.py`, `start_master.sh`, and `start_worker.bat` provide the strict bootstrap path required by the extended PRD.

## Repository Layout

```text
.
├── README.md
├── CHANGELOG.md
├── PRD.md
├── PRD_Extended.md
├── config.json
├── config_extended.json
├── master/
├── worker/
├── tests/
└── scripts/windows/
```

## Requirements

- Python 3.12+ for local development and testing in this repository
- Python 3.14-compatible runtime for the worker container image and the strict master bootstrap script
- Docker Desktop 24+
- macOS or Linux for the master path tested here
- Windows with Docker Desktop for physical worker deployment

## Local Development Setup

Install the baseline master dependencies:

```bash
python3 -m pip install -r master/requirements.txt
```

Install the baseline worker dependencies:

```bash
python3 -m pip install -r worker/requirements.txt
```

Install the DFS-lite master dependencies:

```bash
python3 -m pip install -r master/requirements_extended.txt
```

## Phase 1: Baseline Local Simulation

Start two baseline workers in separate terminals:

```bash
python3 -m worker.worker --port 5001 --worker-id worker_1
python3 -m worker.worker --port 5002 --worker-id worker_2
```

Run the baseline master:

```bash
python3 -m master.master --config config.json --log-level INFO
```

Expected baseline outcome:

- 10 communication rounds complete successfully
- validation accuracy rises from `0.3684` to `0.9737`
- only model parameters and intercepts are transmitted after initialization

## Phase 2: Baseline Containerized Workers

Build the baseline worker image:

```bash
docker build -t hetero-fedlearn-worker:test worker
```

Run two baseline workers locally in containers:

```bash
docker run --rm -d -p 5001:5000 --name hetero-fedlearn-worker-1 -e WORKER_ID=worker_1 hetero-fedlearn-worker:test
docker run --rm -d -p 5002:5000 --name hetero-fedlearn-worker-2 -e WORKER_ID=worker_2 hetero-fedlearn-worker:test
```

Verify health:

```bash
curl http://127.0.0.1:5001/health
curl http://127.0.0.1:5002/health
docker inspect --format '{{.Name}} {{.State.Health.Status}}' hetero-fedlearn-worker-1 hetero-fedlearn-worker-2
```

Run the baseline master against those containerized workers:

```bash
python3 -m master.master --config config.json --log-level INFO
```

## DFS-Lite Extension Demo

For the fastest local dashboard demo on macOS or Linux, run:

```bash
python3 start_dashboard.py --allow-unsupported-python
```

This launcher builds the DFS-lite worker image, starts one local worker container per localhost endpoint declared in [`config_extended.json`](config_extended.json), waits for each `/health` endpoint, and then chains into [`start_master.sh`](start_master.sh). When the master process exits, the worker containers are removed automatically.

If `8080` is already occupied on the host, override it explicitly:

```bash
python3 start_dashboard.py --allow-unsupported-python --master-port 18080
```

Start two DFS-lite workers in separate terminals:

```bash
python3 -m worker.worker_dfs --port 5001 --worker-id worker_1 --storage-dir /tmp/hetero-fedlearn-worker-1
python3 -m worker.worker_dfs --port 5002 --worker-id worker_2 --storage-dir /tmp/hetero-fedlearn-worker-2
```

Start the DFS-lite master dashboard with the background training thread enabled:

```bash
python3 -m master.master_dfs --config config_extended.json --host 127.0.0.1 --port 8080 --auto-start
```

Open the dashboards:

- `http://127.0.0.1:8080/` for the NameNode-style master view
- `http://127.0.0.1:5001/` and `http://127.0.0.1:5002/` for the DataNode worker views

Validated outcome on the default extended configuration:

- workers commit physical CSV blocks to local storage
- the asynchronous master thread completes 10 communication rounds without blocking Flask
- the master dashboard serves live block-map and worker-health state on `/api/status`
- validation accuracy reaches `0.9737`

## DFS-Lite Docker Packaging

Build the DFS-lite worker image:

```bash
docker build -t hetero-fedlearn-worker-dfs:test -f worker/Dockerfile_extended worker
```

The extended Windows bootstrap script builds this image and mounts host storage into `/app/datanode_storage` so the block files visibly persist on the Windows host filesystem.

## Phase 3: Windows Worker Onboarding

On each Windows worker host, open an elevated PowerShell session and run:

```powershell
pwsh -ExecutionPolicy Bypass -File .\scripts\windows\onboard_worker.ps1 `
  -BuildContext .\worker `
  -Image hetero-fedlearn-worker:test `
  -ContainerName hetero-fedlearn-worker-1 `
  -WorkerId worker_1 `
  -HostPort 5000 `
  -SetActiveNetworkPrivate
```

What the PowerShell script does:

- verifies Docker availability
- creates or refreshes the inbound firewall rule for the worker port
- optionally switches connected networks to `Private`
- optionally builds or pulls the worker image
- starts the worker container with `--restart unless-stopped`
- waits for `/health` to report `status=ok`

After onboarding, update the `workers` section in [`config.json`](config.json) or [`config_extended.json`](config_extended.json) to use the Windows machines' IPv4 addresses.

For the strict extended-PRD bootstrap path, use `start_worker.bat`. It verifies `docker info`, removes the stale container, builds `worker/Dockerfile_extended`, mounts `%cd%\\storage` into `/app/datanode_storage`, and opens the worker dashboard automatically.

For the master side, use `start_master.sh`. It verifies Python `3.14+` by default, creates an isolated virtual environment, installs `master/requirements_extended.txt`, binds the DFS-lite master dashboard to `0.0.0.0:8080` by default, and opens the browser automatically. `CONFIG_PATH` and `MASTER_PORT` can be overridden when another DFS-lite config or dashboard port is required.

## Configuration

The default [`config.json`](config.json) uses:

- the built-in breast cancer dataset
- `10` communication rounds
- `5` local epochs per round
- `120` second HTTP timeout
- `3` retry attempts per worker request
- two local worker endpoints on ports `5001` and `5002`

The default [`config_extended.json`](config_extended.json) adds:

- `replication_factor` for DFS-lite block placement
- `health_timeout_seconds` for dashboard-safe worker polling
- `dashboard.poll_interval_ms` for the UI refresh cadence

## Worker API

### Baseline

- `GET /health`
- `POST /initialize`
- `POST /train_round`

### DFS-Lite Extension

- `GET /`
- `GET /health`
- `GET /api/status`
- `POST /initialize` with `block_id`
- `POST /train_round` with `block_id`

## Verification

Run the full test suite:

```bash
pytest
```

Current validated paths:

- baseline in-process HTTP integration tests pass
- DFS-lite worker persistence and asynchronous master tests pass
- baseline local Python worker simulation passes
- baseline Docker image build passes on `python:3.14-slim`
- two live baseline worker containers pass health checks and complete a full master training run
- the DFS-lite master and worker dashboards served successfully during a live smoke run
- DFS-lite block CSV files were written to disk and reused during local training rounds
- `start_dashboard.py` now provides a one-command local dashboard bootstrap path for macOS and Linux

## Notes

- NumPy arrays are serialized with `.tolist()` and reconstructed with `numpy.asarray(...)`.
- Training uses `partial_fit(...)` with `warm_start=True` to preserve model state across rounds.
- The master drops only failing workers for a round after exhausting the configured retry budget.
- The DFS-lite extension preserves the original v1 files unchanged and implements the new storage or dashboard features in copied variant files to match the repository doctrine.
