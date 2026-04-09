"""Integration and endpoint tests for the federated learning workflow."""

from __future__ import annotations

import json
import numpy as np
import socket
import threading
import time
from pathlib import Path

from werkzeug.serving import make_server

from master.master import run_training_from_config
from master.master import FederatedMaster, load_config
from worker.worker import create_app


def allocate_port() -> int:
    """Allocate an ephemeral TCP port for a test worker server."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LocalWorkerServer:
    """Run a worker app in-process on a disposable local port."""

    def __init__(self, worker_id: str, port: int) -> None:
        """Create the WSGI server and serving thread."""

        self.worker_id = worker_id
        self.port = port
        self.server = make_server("127.0.0.1", port, create_app(default_worker_id=worker_id))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        """Start serving requests in the background."""

        self.thread.start()
        time.sleep(0.1)

    def stop(self) -> None:
        """Shutdown the server and join its serving thread."""

        self.server.shutdown()
        self.thread.join(timeout=5)


def write_test_config(config_path: Path, worker_ports: list[int]) -> None:
    """Write a deterministic master configuration for the integration test."""

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
            "rounds": 10,
            "local_epochs": 5,
            "random_seed": 42,
        },
        "network": {
            "timeout_seconds": 120,
            "retry_attempts": 3,
            "retry_backoff_seconds": 0.05,
        },
        "workers": [
            {"worker_id": "worker_1", "endpoint": f"http://127.0.0.1:{worker_ports[0]}"},
            {"worker_id": "worker_2", "endpoint": f"http://127.0.0.1:{worker_ports[1]}"},
        ],
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")


def test_worker_rejects_training_before_initialisation() -> None:
    """The worker must reject `/train_round` before `/initialize` succeeds."""

    app = create_app(default_worker_id="worker_test")
    client = app.test_client()
    response = client.post(
        "/train_round",
        json={
            "round_number": 1,
            "global_weights": [0.0, 0.0],
            "global_intercept": [0.0],
            "local_epochs": 1,
        },
    )
    assert response.status_code == 409


def test_master_supports_digits_as_larger_builtin_dataset(tmp_path: Path) -> None:
    """The baseline master should expose the larger builtin digits dataset."""

    config_path = tmp_path / "config_digits.json"
    write_test_config(config_path, [5001, 5002])
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config_payload["dataset"]["name"] = "digits"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    master = FederatedMaster(load_config(config_path))
    features, labels = master.load_dataset()

    assert features.shape[0] > 1000
    assert len(np.unique(labels)) == 10


def test_worker_rejects_weight_dimension_mismatch() -> None:
    """The baseline worker must reject rounds with incompatible weight dimensions."""

    app = create_app(default_worker_id="worker_test")
    client = app.test_client()
    initialize_response = client.post(
        "/initialize",
        json={
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

    train_response = client.post(
        "/train_round",
        json={
            "round_number": 1,
            "global_weights": [0.0, 0.0, 0.0],
            "global_intercept": [0.0],
            "local_epochs": 1,
        },
    )
    assert train_response.status_code == 400
    assert "dimension" in train_response.get_json()["error"].lower()


def test_federated_training_roundtrip(tmp_path: Path) -> None:
    """The master should complete a full multi-round federated training session."""

    worker_ports = [allocate_port(), allocate_port()]
    servers = [
        LocalWorkerServer(worker_id="worker_1", port=worker_ports[0]),
        LocalWorkerServer(worker_id="worker_2", port=worker_ports[1]),
    ]
    for server in servers:
        server.start()

    try:
        config_path = tmp_path / "config.json"
        write_test_config(config_path, worker_ports)
        summary = run_training_from_config(config_path)
    finally:
        for server in servers:
            server.stop()

    round_accuracies = [entry["validation_accuracy"] for entry in summary["history"]]
    assert len(summary["history"]) == 10
    assert summary["active_workers"] == ["worker_1", "worker_2"]
    assert summary["final_validation_accuracy"] >= 0.97
    assert round_accuracies == sorted(round_accuracies)
