"""Integration tests for the DFS-lite federated learning extension."""

from __future__ import annotations

import io
import json
import socket
import threading
import time
from pathlib import Path

from werkzeug.serving import make_server

from master.master_dfs import FederatedMasterDFS, create_app as create_master_app, load_config
from worker.worker_dfs import create_app as create_worker_app
from sklearn.datasets import load_breast_cancer


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


def test_worker_train_round_reads_block_from_disk_each_time(tmp_path: Path) -> None:
    """The DFS-lite worker must fail if the persisted block on disk becomes invalid."""

    storage_dir = tmp_path / "storage"
    app = create_worker_app(default_worker_id="worker_test", storage_dir=storage_dir)
    client = app.test_client()

    initialize_response = client.post(
        "/initialize",
        json={
            "block_id": "blk_disk",
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

    block_path = storage_dir / "blk_disk.csv"
    block_path.write_text("feature_0,feature_1,label\n0.2,0.1\n", encoding="utf-8")

    train_response = client.post(
        "/train_round",
        json={
            "block_id": "blk_disk",
            "round_number": 1,
            "global_weights": [0.0, 0.0],
            "global_intercept": [0.0],
            "local_epochs": 1,
        },
    )
    assert train_response.status_code == 400


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


def test_dfs_master_fails_over_to_replica_when_primary_worker_drops(tmp_path: Path) -> None:
    """The DFS-lite master must reroute block training to a surviving replica."""

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

        block_payloads, _, _ = service.prepare_blocks()
        assignments = service.initialise_blocks(block_payloads)

        servers[0].stop()
        worker_health = service.refresh_worker_health()
        assert worker_health["worker_1"]["healthy"] is False
        assert worker_health["worker_2"]["healthy"] is True

        block_updates = [
            service.train_block(
                assignment=assignment,
                round_number=1,
                local_epochs=int(service.training_config["local_epochs"]),
            )
            for assignment in assignments
        ]
        assert all(update.worker_id == "worker_2" for update in block_updates)

        service.global_weights, service.global_intercept = service.aggregate_updates(block_updates)
        validation_accuracy, validation_loss = service.evaluate_global_model()
        assert validation_accuracy >= 0.9
        assert validation_loss < 0.4
    finally:
        servers[1].stop()


def test_master_start_endpoint_is_idempotent(tmp_path: Path) -> None:
    """The dashboard start endpoint must not spawn duplicate training threads."""

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
        app = create_master_app(config_path=config_path, autostart=False, service=service)
        client = app.test_client()

        first_response = client.post("/api/start_training")
        second_response = client.post("/api/start_training")
        assert first_response.status_code == 200
        assert second_response.status_code == 200
        assert first_response.get_json()["started"] is True
        assert second_response.get_json()["started"] is False

        service.wait_for_training(timeout=30)
        final_status = client.get("/api/status").get_json()
        assert final_status["training_completed"] is True
        assert final_status["training_error"] is None
    finally:
        for server in servers:
            if server.thread.is_alive():
                server.stop()


def test_master_runtime_config_and_dataset_upload_workflow(tmp_path: Path) -> None:
    """The master control plane must accept config edits and CSV dataset uploads."""

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
        service = FederatedMasterDFS(load_config(config_path), upload_dir=tmp_path / "uploads")
        app = create_master_app(config_path=config_path, autostart=False, service=service)
        client = app.test_client()

        config_response = client.post(
            "/api/config",
            json={
                "dataset": {"validation_fraction": 0.25},
                "training": {"rounds": 3, "local_epochs": 2, "replication_factor": 1},
                "network": {"timeout_seconds": 20},
            },
        )
        assert config_response.status_code == 200
        updated_config = config_response.get_json()["config"]
        assert updated_config["training"]["rounds"] == 3
        assert updated_config["training"]["local_epochs"] == 2
        assert updated_config["dataset"]["validation_fraction"] == 0.25
        assert updated_config["network"]["timeout_seconds"] == 20

        features, labels = load_breast_cancer(return_X_y=True)
        header = ",".join([f"feature_{index}" for index in range(features.shape[1])] + ["label"])
        rows = [
            ",".join([*(f"{value:.10f}" for value in row), str(int(label))])
            for row, label in zip(features, labels, strict=True)
        ]
        csv_payload = f"{header}\n" + "\n".join(rows)
        upload_response = client.post(
            "/api/dataset/upload",
            data={
                "label_column": "-1",
                "dataset": (io.BytesIO(csv_payload.encode("utf-8")), "breast_cancer_copy.csv"),
            },
            content_type="multipart/form-data",
        )
        assert upload_response.status_code == 200
        upload_payload = upload_response.get_json()
        assert upload_payload["dataset"]["source"] == "csv"
        assert Path(upload_payload["dataset"]["csv_path"]).exists()

        start_response = client.post("/api/start_training")
        assert start_response.status_code == 200
        service.wait_for_training(timeout=30)
        final_status = client.get("/api/status").get_json()
        assert final_status["training_completed"] is True
        assert final_status["training_error"] is None
        assert final_status["latest_validation_accuracy"] >= 0.9
    finally:
        for server in servers:
            if server.thread.is_alive():
                server.stop()


def test_worker_can_register_itself_with_master_control_plane(tmp_path: Path) -> None:
    """A worker must be able to register itself with a running master from its own API."""

    worker_port = allocate_port()
    master_port = allocate_port()
    worker_storage_dir = tmp_path / "worker_storage"
    worker_app = create_worker_app(default_worker_id="worker_self", storage_dir=worker_storage_dir)
    worker_server = make_server("127.0.0.1", worker_port, worker_app)
    worker_thread = threading.Thread(target=worker_server.serve_forever, daemon=True)
    worker_thread.start()
    time.sleep(0.1)

    try:
        config_path = tmp_path / "config_extended.json"
        write_extended_config(config_path, [allocate_port(), allocate_port()])
        service = FederatedMasterDFS(load_config(config_path), upload_dir=tmp_path / "uploads")
        master_app = create_master_app(config_path=config_path, autostart=False, service=service)
        master_server = make_server("127.0.0.1", master_port, master_app)
        master_thread = threading.Thread(target=master_server.serve_forever, daemon=True)
        master_thread.start()
        time.sleep(0.1)

        try:
            worker_client = worker_app.test_client()
            connect_response = worker_client.post(
                "/api/connect_master",
                json={
                    "master_endpoint": f"http://127.0.0.1:{master_port}",
                    "advertised_endpoint": f"http://127.0.0.1:{worker_port}",
                },
            )
            assert connect_response.status_code == 200
            connect_payload = connect_response.get_json()
            assert connect_payload["connection"]["master_endpoint"] == f"http://127.0.0.1:{master_port}"

            master_client = master_app.test_client()
            config_payload = master_client.get("/api/config").get_json()
            registered_worker = next(
                worker for worker in config_payload["workers"] if worker["worker_id"] == "worker_self"
            )
            assert registered_worker["endpoint"] == f"http://127.0.0.1:{worker_port}"

            status_payload = worker_client.get("/api/status").get_json()
            assert status_payload["master_endpoint"] == f"http://127.0.0.1:{master_port}"
            assert status_payload["advertised_endpoint"] == f"http://127.0.0.1:{worker_port}"
            assert status_payload["last_registration_status"] == "connected"
        finally:
            master_server.shutdown()
            master_thread.join(timeout=5)
    finally:
        worker_server.shutdown()
        worker_thread.join(timeout=5)
