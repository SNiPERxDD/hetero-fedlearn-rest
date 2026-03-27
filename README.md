# hetero-fedlearn-rest

Distributed iterative federated learning over HTTP for a heterogeneous cluster with one master node and multiple worker nodes. The current implementation uses a lightweight REST control plane, `SGDClassifier` local training, weighted FedAvg aggregation, and Docker-packaged workers for Windows-friendly deployment.

## Architecture

- `master/master.py` orchestrates dataset preparation, worker initialization, round execution, retry-aware HTTP communication, FedAvg aggregation, and validation.
- `worker/worker.py` exposes `/health`, `/initialize`, and `/train_round` and keeps local model state across communication rounds.
- `config.json` defines the dataset, model hyperparameters, training schedule, timeouts, retries, and worker endpoints.
- `worker/Dockerfile` packages the worker on `python:3.14-slim` with Flask, Waitress, scikit-learn, and a native Docker health check.
- `scripts/windows/onboard_worker.ps1` automates Windows worker setup for firewall rules, optional network profile hardening, image build or pull, container launch, and health verification.

## Repository Layout

```text
.
├── README.md
├── CHANGELOG.md
├── PRD.md
├── config.json
├── master/
├── worker/
├── tests/
└── scripts/windows/
```

## Requirements

- Python 3.12+ for local development
- Python 3.14-compatible runtime for the worker container image
- Docker Desktop 24+
- macOS or Linux for the master path tested here
- Windows with Docker Desktop for physical worker deployment

## Local Development Setup

Install the master dependencies:

```bash
python3 -m pip install -r master/requirements.txt
```

Install the worker dependencies for local non-container simulation:

```bash
python3 -m pip install -r worker/requirements.txt
```

## Phase 1: Local Simulation

Start two workers in separate terminals:

```bash
python3 -m worker.worker --port 5001 --worker-id worker_1
python3 -m worker.worker --port 5002 --worker-id worker_2
```

Run the master:

```bash
python3 -m master.master --config config.json --log-level INFO
```

Expected outcome on the default configuration:

- 10 communication rounds complete successfully
- validation accuracy rises from `0.3684` to `0.9737`
- only model parameters and intercepts are transmitted after initialization

## Phase 2: Containerized Workers

Build the worker image:

```bash
docker build -t hetero-fedlearn-worker:test worker
```

Run two workers locally in containers:

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

Run the master against those containerized workers:

```bash
python3 -m master.master --config config.json --log-level INFO
```

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

What the script does:

- verifies Docker availability
- creates or refreshes the inbound firewall rule for the worker port
- optionally switches connected networks to `Private`
- optionally builds or pulls the worker image
- starts the worker container with `--restart unless-stopped`
- waits for `/health` to report `status=ok`

After onboarding, update the `workers` section in [`config.json`](config.json) to use the Windows machines' IPv4 addresses.

## Configuration

The default [`config.json`](config.json) uses:

- the built-in breast cancer dataset
- `10` communication rounds
- `5` local epochs per round
- `120` second HTTP timeout
- `3` retry attempts per worker request
- two local worker endpoints on ports `5001` and `5002`

## Worker API

### `GET /health`

Returns worker liveness and initialization state.

### `POST /initialize`

Loads the worker-local shard exactly once.

Required payload fields:

- `worker_id`
- `features`
- `labels`
- `classes`
- `model_config`

### `POST /train_round`

Runs one local training round with broadcast global parameters.

Required payload fields:

- `round_number`
- `global_weights`
- `global_intercept`
- `local_epochs`

## Verification

Run the test suite:

```bash
pytest
```

Current validated paths:

- in-process HTTP integration tests pass
- local Python worker simulation passes
- Docker image build passes on `python:3.14-slim`
- two live worker containers pass health checks and complete a full master training run

## Notes

- NumPy arrays are serialized with `.tolist()` and reconstructed with `numpy.asarray(...)`.
- Training uses `partial_fit(...)` with `warm_start=True` to preserve model state across rounds.
- The master drops only failing workers for a round after exhausting the configured retry budget.
