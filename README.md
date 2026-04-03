# Hetero FedLearn REST

[![Tests](https://img.shields.io/badge/tests-23%20passed-1f6feb)](./README.md#-verification)
[![Runtime](https://img.shields.io/badge/runtime-baseline%20%2B%20DFS--lite-0a7f6f)](./README.md#-operating-modes)
[![Launchers](https://img.shields.io/badge/launchers-python--first-c26a14)](./README.md#-preferred-entry-points)
[![Workers](https://img.shields.io/badge/workers-http%20%2B%20docker-5b6b73)](./README.md#-deployment-paths)

Federated learning over HTTP for heterogeneous clusters, with a preserved baseline runtime and a DFS-lite extension for disk-backed locality, live dashboards, browser-driven control, and cross-platform Python launchers.

The repository is structured as an engineering runtime first, not a demo shell: the baseline path remains intact, the DFS-lite path is implemented as a separate extension, and the documented operator flows map directly to the code that is currently tested.

## ◇ Snapshot

| Area | Baseline | DFS-lite |
| --- | --- | --- |
| Master runtime | `master/master.py` | `master/master_dfs.py` |
| Worker runtime | `worker/worker.py` | `worker/worker_dfs.py` |
| Config | `config.json` | `config_extended.json` |
| Operator UI | none | master and worker dashboards |
| Preferred launch | direct module run | `start_master.py`, `start_worker.py`, `start_dashboard.py` |

## ◇ Current State

- Verified suite state: `23 passed`
- Verified DFS-lite validation accuracy: `0.9737`
- Preferred master launcher: `start_master.py`
- Preferred worker launcher: `start_worker.py`
- Preferred cleanup launcher: `stop_all.py`
- Project website: `website/`
- PRD archive: `archive/`

## ◇ Operating Modes

- `v1 baseline`: `master/master.py`, `worker/worker.py`, and `config.json`
- `v1.1 DFS-lite extension`: `master/master_dfs.py`, `worker/worker_dfs.py`, and `config_extended.json`

## ◇ Preferred Entry Points

| Task | Command |
| --- | --- |
| Start local DFS-lite demo | `python3 start_dashboard.py --allow-unsupported-python` |
| Start DFS-lite master | `python3 start_master.py --allow-unsupported-python --config config_extended.json --host 127.0.0.1 --port 18080` |
| Start native DFS-lite worker | `python3 start_worker.py --mode native --allow-unsupported-python --worker-id worker_1 --port 5001 --storage-dir /tmp/hetero-fedlearn-worker-1` |
| Start Docker DFS-lite worker | `python3 start_worker.py --mode docker --worker-id worker_1 --port 5000` |
| Stop repo-managed services | `python3 stop_all.py` |

## ◇ Architecture

- `master/master.py` orchestrates dataset preparation, worker initialization, round execution, retry-aware HTTP communication, FedAvg aggregation, and validation.
- `worker/worker.py` exposes `/health`, `/initialize`, and `/train_round` and keeps local model state across communication rounds.
- `master/master_dfs.py` upgrades the master into a NameNode-style service with a background training thread, live telemetry APIs, worker registration, dataset upload controls, and a dashboard on `/`.
- `worker/worker_dfs.py` upgrades the worker into a DataNode-style service that persists blocks to disk, reloads them per round, serves storage or compute telemetry on `/`, and can register itself with the master from its own UI.
- `config.json` defines the baseline dataset, model hyperparameters, training schedule, timeouts, retries, and worker endpoints.
- `config_extended.json` defines the DFS-lite dashboard poll rate, block replication factor, and the extended worker topology.
- `worker/Dockerfile` packages the baseline worker on `python:3.14-slim` with Flask, Waitress, scikit-learn, and a native Docker health check.
- `worker/Dockerfile_extended` packages the DFS-lite worker with templates and a persistent datanode storage directory.
- `scripts/windows/onboard_worker.ps1` automates Windows worker setup for firewall rules, optional network profile hardening, image build or pull, container launch, and health verification.
- `start_master.py` provides the preferred cross-platform DFS-lite master bootstrap.
- `start_worker.py` provides the preferred cross-platform DFS-lite worker bootstrap in either native Python or Docker mode.
- `stop_all.py` provides a repo-scoped cleanup utility that stops the managed master, native workers, and known worker containers so ports can be reclaimed quickly.
- `start_dashboard.py`, `start_master.sh`, and `start_worker.bat` remain available as compatibility bootstrap paths.
- `website/` contains a clean React website package for the project overview, architecture, validation summary, and quick-start flows.

## ◇ Repository Layout

```text
.
├── README.md
├── CHANGELOG.md
├── archive/
│   ├── PRD.md
│   └── PRD_Extended.md
├── config.json
├── config_extended.json
├── start_dashboard.py
├── start_master.py
├── start_master.sh
├── start_worker.py
├── start_worker.bat
├── stop_all.py
├── master/
├── worker/
├── tests/
├── scripts/windows/
└── website/
```

## ◇ Requirements

- Python 3.12+ for local development and testing in this repository
- Python 3.14-compatible runtime for the worker container image and the preferred Python launchers
- Docker Desktop 24+
- macOS or Linux for the localhost dashboard bootstrap path tested here
- Windows with Docker Desktop for physical worker deployment
- Role-swapped operation is supported through `start_master.py` and `start_worker.py`

## ◇ Local Development Setup

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

## ▣ Baseline Runtime

### ◎ Phase 1: Baseline Local Simulation

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

### ◎ Phase 2: Baseline Containerized Workers

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

## ▣ DFS-lite Control Plane

### ◎ Control-Plane Routing

- `http://127.0.0.1:4173` is the separate React website in `website/`. It is a project overview and quick-start surface, not the live training control plane.
- `http://127.0.0.1:18080` is the actual DFS-lite master control panel served by `master/master_dfs.py`.
- `http://127.0.0.1:5001`, `http://127.0.0.1:5002`, and other worker ports are the actual DFS-lite worker dashboards served by `worker/worker_dfs.py`.

### ◎ Local Demo Bootstrap

For the fastest local dashboard demo on macOS or Linux, run:

```bash
python3 start_dashboard.py --allow-unsupported-python
```

This launcher builds the DFS-lite worker image, starts one local worker container per localhost endpoint declared in [`config_extended.json`](config_extended.json), waits for each `/health` endpoint, and then chains into [`start_master.py`](start_master.py). When the master process exits, the worker containers are removed automatically.

The default master dashboard port is `18080`, which avoids common local collisions with other tools that often bind `8080`.

```bash
python3 start_dashboard.py --allow-unsupported-python --master-port 18080
```

### ◎ Direct Master And Worker Launch

Start two DFS-lite workers in separate terminals:

```bash
python3 start_worker.py --mode native --allow-unsupported-python --worker-id worker_1 --port 5001 --storage-dir /tmp/hetero-fedlearn-worker-1
python3 start_worker.py --mode native --allow-unsupported-python --worker-id worker_2 --port 5002 --storage-dir /tmp/hetero-fedlearn-worker-2
```

Start the DFS-lite master dashboard:

```bash
python3 start_master.py --allow-unsupported-python --config config_extended.json --host 127.0.0.1 --port 18080
```

Open the dashboards:

- `http://127.0.0.1:18080/` for the NameNode-style master view
- `http://127.0.0.1:5001/` and `http://127.0.0.1:5002/` for the DataNode worker views

### ◎ Validated Outcome

- workers commit physical CSV blocks to local storage
- the asynchronous master thread completes 10 communication rounds without blocking Flask
- the master dashboard serves live block-map and worker-health state on `/api/status`
- the master dashboard can register workers, upload a CSV dataset, and update training settings from the browser
- the worker dashboard can register itself with the master by posting its advertised endpoint
- validation accuracy reaches `0.9737`

## ▣ Deployment Paths

### ◎ DFS-lite Docker Packaging

Build the DFS-lite worker image:

```bash
docker build -t hetero-fedlearn-worker-dfs:test -f worker/Dockerfile_extended worker
```

The extended Windows bootstrap script builds this image and mounts host storage into `/app/datanode_storage` so the block files visibly persist on the Windows host filesystem.

### ◎ Windows Worker Onboarding

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

### ◎ Preferred Worker Bootstrap

For the preferred cross-platform worker bootstrap path, use `start_worker.py`.

Native mode is suitable for a macOS worker or a direct Python worker on any host:

```bash
python3 start_worker.py --mode native --allow-unsupported-python --worker-id mac_worker --port 5000 --storage-dir ~/hetero-fedlearn-worker
```

Docker mode is suitable for Windows or Docker-backed worker hosts:

```bash
python3 start_worker.py --mode docker --worker-id worker_1 --port 5000
```

For the strict compatibility path, `start_worker.bat` still verifies `docker info`, removes the stale container, builds `worker/Dockerfile_extended`, mounts `%cd%\\storage` into `/app/datanode_storage`, and opens the worker dashboard automatically.

### ◎ Preferred Master Bootstrap

For the preferred cross-platform master bootstrap path, use `start_master.py`. It verifies Python `3.14+` by default, creates an isolated virtual environment, installs `master/requirements_extended.txt`, binds the DFS-lite master dashboard to `0.0.0.0:18080` by default, and opens the browser automatically.

`start_master.sh` remains as a compatibility wrapper that delegates to `start_master.py`.

## ▣ Role-Swapped Topologies

The runtime is no longer tied to "mac master, Windows workers" at the launcher level. The preferred Python entry points support role-swapped layouts directly:

- Windows master: `py start_master.py --config config_extended.json --host 0.0.0.0 --port 18080`
- macOS worker: `python3 start_worker.py --mode native --allow-unsupported-python --worker-id mac_worker --port 5000 --storage-dir ~/hetero-fedlearn-worker`
- Windows worker in Docker mode: `py start_worker.py --mode docker --worker-id worker_1 --port 5000`

`start_dashboard.py` remains a localhost demo path for macOS/Linux because it builds and orchestrates local worker containers on the same machine.

## ▣ Stop And Cleanup

To free the repo-managed control-plane ports quickly, run:

```bash
python3 stop_all.py
```

Default behavior:

- stops repo-managed listeners on `18080`, `5000`, `5001`, and `5002`
- removes the known worker containers started by the Python launchers or dashboard bootstrap

You can override the target ports or container names:

```bash
python3 stop_all.py --ports 18180 15050 --container-name worker-node --container-prefix hetero-fedlearn-dashboard
```

To inspect what would be stopped without terminating anything:

```bash
python3 stop_all.py --dry-run
```

## ◇ Runtime Workflow

After the services are up:

- use the master dashboard to register worker endpoints, change training settings, switch back to the builtin dataset, or upload a CSV dataset
- use the worker dashboard to register a worker with the master by entering the master URL and the advertised worker endpoint
- start the federated run directly from the master dashboard without editing the config file manually

## ◇ Configuration

| File | Purpose |
| --- | --- |
| `config.json` | baseline dataset, schedule, retries, and worker endpoints |
| `config_extended.json` | DFS-lite topology, replication, network settings, and dashboard polling |

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

## ◇ React Website

The repository includes a clean React website package in [`website/`](website) for presenting the architecture, DFS-lite extension, quick-start flows, and current verification state.

Important separation:

- the React site is intentionally separate from the live Flask control plane
- `npm run dev` in `website/` serves the documentation layer on `4173` by default
- the real DFS-lite master operator UI is served by `master/master_dfs.py` on `18080`
- the real worker operator UIs are served by `worker/worker_dfs.py` on worker ports such as `5001` and `5002`

Run it locally:

```bash
cd website
npm install
npm run dev
```

Build it for production:

```bash
cd website
npm run build
```

Notes:

- the generated AI Studio browser mock was preserved separately during integration, but the committed deliverable is the cleaned `website/` package
- the website copy reflects the real project entry points such as `start_dashboard.py`, `start_master.py`, `start_worker.py`, `stop_all.py`, `start_worker.bat`, and `scripts/windows/onboard_worker.ps1`
- the current site content includes the verified `23 passed` suite state and the `0.9737` DFS-lite validation result

## ◇ Worker API

### Baseline

- `GET /health`
- `POST /initialize`
- `POST /train_round`

### DFS-lite Extension

- `GET /`
- `GET /health`
- `GET /api/status`
- `GET /api/config` on the master
- `POST /api/config` on the master
- `POST /api/workers/register` on the master
- `POST /api/workers/remove` on the master
- `POST /api/dataset/upload` on the master
- `POST /api/connect_master` on the worker
- `POST /initialize` with `block_id`
- `POST /train_round` with `block_id`

## ◇ Verification

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
- the DFS-lite master and worker dashboards served successfully during live smoke runs
- DFS-lite block CSV files were written to disk and reused during local training rounds
- `start_dashboard.py` provides a one-command local dashboard bootstrap path for macOS and Linux
- `start_master.py` and `start_worker.py` provide Python-first cross-platform launchers for role-swapped deployments
- `stop_all.py` reclaims repo-managed ports cleanly in a live smoke run
- the DFS-lite control plane supports worker self-registration, browser-side training configuration, and CSV dataset upload

## ◇ Implementation Notes

- NumPy arrays are serialized with `.tolist()` and reconstructed with `numpy.asarray(...)`
- training uses `partial_fit(...)` with `warm_start=True` to preserve model state across rounds
- the master drops only failing workers for a round after exhausting the configured retry budget
- the DFS-lite extension preserves the original v1 files unchanged and implements the new storage or dashboard features in copied variant files to match the repository doctrine
