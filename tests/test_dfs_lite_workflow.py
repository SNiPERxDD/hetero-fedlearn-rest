"""Integration tests for the DFS-lite federated learning extension."""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

from werkzeug.serving import make_server

from master.master_dfs import FederatedMasterDFS, create_app as create_master_app, load_config
from worker.worker_dfs import create_app as create_worker_app


def allocate_port() -> int:
    """Allocate an ephemeral TCP port for a test server."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LocalDFSWorkerServer:
    """Run a DFS-lite worker app in-process on a disposable local port."""

    def __init__(self, worker_id: str, port: int, storage_dir: Path) -> None:
        """Create the WSGI server and serving thread."""

        self.worker_id = worker_id
        self.port = port
        self.server = make_server(
            "127.0.0.1",
            port,
            create_worker_app(default_worker_id=worker_id, storage_dir=storage_dir),
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        """Start serving requests in the background."""

        self.thread.start()
        time.sleep(0.1)

    def stop(self) -> None:
        """Shutdown the server and join its serving thread."""

        self.server.shutdown()
        self.thread.join(timeout=5)


def write_extended_config(config_path: Path, worker_ports: list[int]) -> None:
    """Write a deterministic DFS-lite configuration for integration tests."""

    config = {
        "dataset": {
            "source": "builtin",
            "name": "breast_cancer",
            "validation_fraction": 0.2,
        },
        "model": {
            "loss": "log_loss",
            "alpha": 0.0001,
            "eta0": 0.001,
            "learning_rate": "constant",
            "penalty": "l2",
            "random_state": 42,
        },
        "training": {
            "rounds": 6,
            "local_epochs": 4,
            "random_seed": 42,
            "replication_factor": 2,
        },
        "network": {
            "timeout_seconds": 30,
            "retry_attempts": 2,
            "retry_backoff_seconds": 0.05,
            "health_timeout_seconds": 1.5,
        },
        "dashboard": {
            "poll_interval_ms": 250,
        },
        "workers": [
            {"worker_id": "worker_1", "endpoint": f"http://127.0.0.1:{worker_ports[0]}"},
            {"worker_id": "worker_2", "endpoint": f"http://127.0.0.1:{worker_ports[1]}"},
        ],
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")


def test_worker_persists_blocks_and_reports_status(tmp_path: Path) -> None:
    """A worker must write a block to disk and expose its telemetry."""

    storage_dir = tmp_path / "storage"
    app = create_worker_app(default_worker_id="worker_test", storage_dir=storage_dir)
    client = app.test_client()

    initialize_response = client.post(
        "/initialize",
        json={
            "block_id": "blk_test",
            "worker_id": "worker_test",
            "features": [[0.1, 0.2], [0.3, 0.4]],
            "labels": [0, 1],
            "classes": [0, 1],
            "model_config": {
                "loss": "log_loss",
                "alpha": 0.0001,
                "eta0": 0.001,
                "learning_rate": "constant",
                "penalty": "l2",
                "random_state": 42,
            },
        },
    )
    assert initialize_response.status_code == 200
    assert (storage_dir / "blk_test.csv").exists()

    status_response = client.get("/api/status")
    status_payload = status_response.get_json()
    assert status_response.status_code == 200
    assert status_payload["storage_bytes"] > 0
    assert status_payload["block_count"] == 1

    train_response = client.post(
        "/train_round",
        json={
            "block_id": "blk_test",
            "round_number": 1,
            "global_weights": [0.0, 0.0],
            "global_intercept": [0.0],
            "local_epochs": 1,
        },
    )
    assert train_response.status_code == 200
    assert train_response.get_json()["samples_processed"] == 2


def test_dfs_master_training_thread_and_dashboard(tmp_path: Path) -> None:
    """The DFS-lite master should complete training and expose dashboard state."""

    worker_ports = [allocate_port(), allocate_port()]
    storage_dirs = [tmp_path / "worker_1_storage", tmp_path / "worker_2_storage"]
    servers = [
        LocalDFSWorkerServer(worker_id="worker_1", port=worker_ports[0], storage_dir=storage_dirs[0]),
        LocalDFSWorkerServer(worker_id="worker_2", port=worker_ports[1], storage_dir=storage_dirs[1]),
    ]
    for server in servers:
        server.start()

    try:
        config_path = tmp_path / "config_extended.json"
        write_extended_config(config_path, worker_ports)

        service = FederatedMasterDFS(load_config(config_path))
        assert service.start_training_thread() is True
        service.wait_for_training(timeout=30)

        snapshot = service.state.snapshot()
        assert snapshot["training_completed"] is True
        assert snapshot["training_error"] is None
        assert len(snapshot["validation_history"]) == 7
        assert snapshot["latest_validation_accuracy"] >= 0.95
        assert len(snapshot["block_map"]) == 2
        assert all(len(block["replicas"]) == 2 for block in snapshot["block_map"])

        for storage_dir in storage_dirs:
            persisted_blocks = list(storage_dir.glob("*.csv"))
            assert len(persisted_blocks) == 2

        app = create_master_app(config_path=config_path, autostart=False, service=service)
        client = app.test_client()
        dashboard_response = client.get("/")
        status_response = client.get("/api/status")
        assert dashboard_response.status_code == 200
        assert status_response.status_code == 200
        assert "DFS-Lite" in dashboard_response.get_data(as_text=True)
        assert status_response.get_json()["training_completed"] is True
    finally:
        for server in servers:
            server.stop()
