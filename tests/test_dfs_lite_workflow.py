"""Integration tests for the DFS-lite federated learning extension."""

from __future__ import annotations

import io
import json
import numpy as np
import socket
import threading
import time
from pathlib import Path

from werkzeug.serving import make_server

from master.master_dfs import FederatedMasterDFS, create_app as create_master_app, load_config
from worker.worker_dfs import beacon_targets, create_app as create_worker_app
from sklearn.datasets import make_classification


def allocate_port() -> int:
    """Allocate an ephemeral TCP port for a test server."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def allocate_udp_port() -> int:
    """Allocate an ephemeral UDP port for discovery listener tests."""

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
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


def write_csv_dataset(path: Path, features, labels) -> None:
    """Write a feature matrix and label vector to a CSV dataset file."""

    header = ",".join([f"feature_{index}" for index in range(features.shape[1])] + ["label"])
    rows = [
        ",".join([*(f"{value:.10f}" for value in row), str(int(label))])
        for row, label in zip(features, labels, strict=True)
    ]
    path.write_text(f"{header}\n" + "\n".join(rows), encoding="utf-8")


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
    assert status_payload["lan_endpoint"].startswith("http://")
    assert status_payload["udp_beacon_enabled"] is False

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

        features, labels = make_classification(
            n_samples=240,
            n_features=12,
            n_informative=8,
            n_redundant=0,
            n_clusters_per_class=1,
            class_sep=2.2,
            random_state=42,
        )
        dataset_path = tmp_path / "synthetic_dataset.csv"
        write_csv_dataset(dataset_path, features, labels)
        upload_response = client.post(
            "/api/dataset/upload",
            data={
                "label_column": "-1",
                "dataset": (io.BytesIO(dataset_path.read_bytes()), dataset_path.name),
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
        assert final_status["latest_validation_accuracy"] >= 0.8
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


def test_worker_auto_registers_with_master_endpoint_env(tmp_path: Path, monkeypatch) -> None:
    """A worker should auto-register when MASTER_ENDPOINT is provided in the environment."""

    master_port = allocate_port()
    worker_port = allocate_port()

    config_path = tmp_path / "config_extended.json"
    write_extended_config(config_path, [allocate_port(), allocate_port()])
    service = FederatedMasterDFS(load_config(config_path), upload_dir=tmp_path / "uploads")
    master_app = create_master_app(config_path=config_path, autostart=False, service=service)
    master_server = make_server("127.0.0.1", master_port, master_app)
    master_thread = threading.Thread(target=master_server.serve_forever, daemon=True)
    master_thread.start()
    time.sleep(0.1)

    worker_storage_dir = tmp_path / "worker_storage"
    monkeypatch.setenv("MASTER_ENDPOINT", f"http://127.0.0.1:{master_port}")
    monkeypatch.setenv("ADVERTISED_ENDPOINT", f"http://127.0.0.1:{worker_port}")
    worker_app = create_worker_app(default_worker_id="worker_env", storage_dir=worker_storage_dir)
    worker_server = make_server("127.0.0.1", worker_port, worker_app)
    worker_thread = threading.Thread(target=worker_server.serve_forever, daemon=True)
    worker_thread.start()
    time.sleep(0.5)

    try:
        master_client = master_app.test_client()
        config_payload = master_client.get("/api/config").get_json()
        registered_worker = next(
            worker for worker in config_payload["workers"] if worker["worker_id"] == "worker_env"
        )
        assert registered_worker["endpoint"] == f"http://127.0.0.1:{worker_port}"

        status_payload = worker_app.test_client().get("/api/status").get_json()
        assert status_payload["master_endpoint"] == f"http://127.0.0.1:{master_port}"
        assert status_payload["advertised_endpoint"] == f"http://127.0.0.1:{worker_port}"
        assert status_payload["last_registration_status"] == "connected"
    finally:
        master_server.shutdown()
        master_thread.join(timeout=5)
        worker_server.shutdown()
        worker_thread.join(timeout=5)


def test_master_udp_discovery_auto_registers_beaconed_workers(tmp_path: Path) -> None:
    """The master should auto-register workers announced over UDP beacons."""

    discovery_port = allocate_udp_port()
    config_path = tmp_path / "config_extended.json"
    write_extended_config(config_path, [allocate_port(), allocate_port()])

    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config_payload["workers"] = []
    config_payload["network"]["discovery_port"] = discovery_port
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    service = FederatedMasterDFS(load_config(config_path), upload_dir=tmp_path / "uploads")
    app = create_master_app(
        config_path=config_path,
        autostart=False,
        service=service,
        enable_udp_discovery=True,
    )
    client = app.test_client()

    beacon_payload = {
        "worker_id": "worker_udp",
        "endpoint": "http://127.0.0.1:5999",
    }
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as beacon_socket:
        beacon_socket.sendto(json.dumps(beacon_payload).encode("utf-8"), ("127.0.0.1", discovery_port))

    deadline = time.time() + 2.0
    while time.time() < deadline:
        workers = client.get("/api/config").get_json()["workers"]
        if any(worker["worker_id"] == "worker_udp" for worker in workers):
            break
        time.sleep(0.1)

    config_after_discovery = client.get("/api/config").get_json()
    discovered_worker = next(
        worker for worker in config_after_discovery["workers"] if worker["worker_id"] == "worker_udp"
    )
    assert discovered_worker["endpoint"] == "http://127.0.0.1:5999"


def test_worker_beacon_targets_cover_non_class_c_private_networks(monkeypatch) -> None:
    """The worker beacon targets should not assume /24-only directed broadcast."""

    monkeypatch.setattr("worker.worker_dfs.get_all_lan_ips", lambda: ["10.136.149.171"])

    targets = beacon_targets("10.136.149.171")

    assert "255.255.255.255" in targets
    assert "10.136.149.255" in targets
    assert "10.136.255.255" in targets
    assert "10.255.255.255" in targets
    assert "10.136.149.171" in targets


def test_worker_auto_registers_after_discovering_master_beacon(tmp_path: Path) -> None:
    """A worker should learn the master endpoint from UDP and self-register."""

    discovery_port = allocate_udp_port()
    master_port = allocate_port()
    worker_port = allocate_port()
    config_path = tmp_path / "config_extended.json"
    write_extended_config(config_path, [allocate_port(), allocate_port()])

    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config_payload["workers"] = []
    config_payload["network"]["discovery_port"] = discovery_port
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    master_service = FederatedMasterDFS(load_config(config_path), upload_dir=tmp_path / "uploads")
    master_app = create_master_app(
        config_path=config_path,
        autostart=False,
        service=master_service,
        enable_udp_discovery=False,
        bound_host="127.0.0.1",
        bound_port=master_port,
    )
    master_server = make_server("127.0.0.1", master_port, master_app)
    master_thread = threading.Thread(target=master_server.serve_forever, daemon=True)
    master_thread.start()
    time.sleep(0.1)

    worker_app = create_worker_app(
        default_worker_id="worker_discovered_master",
        storage_dir=tmp_path / "worker_storage",
        bound_host="127.0.0.1",
        bound_port=worker_port,
        enable_udp_beacon=False,
        enable_master_discovery=True,
        udp_discovery_port=discovery_port,
    )
    worker_server = make_server("127.0.0.1", worker_port, worker_app)
    worker_thread = threading.Thread(target=worker_server.serve_forever, daemon=True)
    worker_thread.start()
    time.sleep(0.1)

    try:
        beacon_payload = {
            "node_type": "master",
            "master_endpoint": f"http://127.0.0.1:{master_port}",
        }
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as beacon_socket:
            beacon_socket.sendto(json.dumps(beacon_payload).encode("utf-8"), ("127.0.0.1", discovery_port))

        deadline = time.time() + 3.0
        while time.time() < deadline:
            workers = master_app.test_client().get("/api/config").get_json()["workers"]
            if any(worker["worker_id"] == "worker_discovered_master" for worker in workers):
                break
            time.sleep(0.1)

        workers = master_app.test_client().get("/api/config").get_json()["workers"]
        discovered_worker = next(
            worker for worker in workers if worker["worker_id"] == "worker_discovered_master"
        )
        assert discovered_worker["endpoint"] == f"http://127.0.0.1:{worker_port}"
    finally:
        master_server.shutdown()
        master_thread.join(timeout=5)
        worker_server.shutdown()
        worker_thread.join(timeout=5)


def test_master_autostart_waits_for_worker_registration(tmp_path: Path, monkeypatch) -> None:
    """Autostart should wait for zero-arg worker registration instead of failing immediately."""

    master_port = allocate_port()
    worker_port = allocate_port()
    config_path = tmp_path / "config_extended.json"
    write_extended_config(config_path, [allocate_port(), allocate_port()])

    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config_payload["workers"] = []
    config_payload["network"]["autostart_wait_seconds"] = 5.0
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    master_service = FederatedMasterDFS(load_config(config_path), upload_dir=tmp_path / "uploads")
    master_app = create_master_app(
        config_path=config_path,
        autostart=True,
        service=master_service,
        enable_udp_discovery=False,
        bound_host="127.0.0.1",
        bound_port=master_port,
    )
    master_server = make_server("127.0.0.1", master_port, master_app)
    master_thread = threading.Thread(target=master_server.serve_forever, daemon=True)
    master_thread.start()
    time.sleep(0.2)

    worker_storage_dir = tmp_path / "worker_storage"
    monkeypatch.setenv("MASTER_ENDPOINT", f"http://127.0.0.1:{master_port}")
    monkeypatch.setenv("ADVERTISED_ENDPOINT", f"http://127.0.0.1:{worker_port}")
    worker_app = create_worker_app(
        default_worker_id="worker_autostart",
        storage_dir=worker_storage_dir,
        bound_host="127.0.0.1",
        bound_port=worker_port,
        enable_udp_beacon=False,
        enable_master_discovery=False,
    )
    worker_server = make_server("127.0.0.1", worker_port, worker_app)
    worker_thread = threading.Thread(target=worker_server.serve_forever, daemon=True)
    worker_thread.start()

    try:
        deadline = time.time() + 30.0
        while time.time() < deadline:
            snapshot = master_service.state.snapshot()
            if snapshot["training_completed"]:
                break
            if snapshot["training_error"]:
                raise AssertionError(snapshot["training_error"])
            time.sleep(0.2)

        snapshot = master_service.state.snapshot()
        assert snapshot["training_completed"] is True
        assert snapshot["training_error"] is None
        registered_workers = master_app.test_client().get("/api/config").get_json()["workers"]
        assert any(worker["worker_id"] == "worker_autostart" for worker in registered_workers)
    finally:
        master_server.shutdown()
        master_thread.join(timeout=5)
        worker_server.shutdown()
        worker_thread.join(timeout=5)


def test_master_prefers_reachable_worker_endpoint_candidate(tmp_path: Path) -> None:
    """The master should choose a worker endpoint candidate it can actually reach."""

    worker_port = allocate_port()
    worker_storage_dir = tmp_path / "worker_storage"
    worker_app = create_worker_app(
        default_worker_id="worker_reachable_candidate",
        storage_dir=worker_storage_dir,
        bound_host="127.0.0.1",
        bound_port=worker_port,
        enable_udp_beacon=False,
        enable_master_discovery=False,
    )
    worker_server = make_server("127.0.0.1", worker_port, worker_app)
    worker_thread = threading.Thread(target=worker_server.serve_forever, daemon=True)
    worker_thread.start()
    time.sleep(0.1)

    try:
        config_path = tmp_path / "config_extended.json"
        write_extended_config(config_path, [allocate_port(), allocate_port()])
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        config_payload["workers"] = []
        config_path.write_text(json.dumps(config_payload), encoding="utf-8")

        service = FederatedMasterDFS(load_config(config_path), upload_dir=tmp_path / "uploads")
        response_payload = service.register_worker(
            worker_id="worker_reachable_candidate",
            endpoint="http://10.255.255.1:5000",
            endpoint_candidates=[f"http://127.0.0.1:{worker_port}"],
        )

        assert response_payload["endpoint"] == f"http://127.0.0.1:{worker_port}"
        assert service.runtime_config_snapshot()["workers"][0]["endpoint"] == f"http://127.0.0.1:{worker_port}"
    finally:
        worker_server.shutdown()
        worker_thread.join(timeout=5)


def test_worker_registration_payload_includes_endpoint_candidates(tmp_path: Path, monkeypatch) -> None:
    """Worker self-registration should provide alternate endpoint candidates to the master."""

    monkeypatch.setattr("worker.worker_dfs.get_all_lan_ips", lambda: ["10.0.0.8", "192.168.1.20"])
    worker_app = create_worker_app(
        default_worker_id="worker_candidates",
        storage_dir=tmp_path / "storage",
        bound_host="0.0.0.0",
        bound_port=5000,
        enable_udp_beacon=False,
        enable_master_discovery=False,
    )
    state = worker_app.config["WORKER_STATE"]

    assert "http://10.0.0.8:5000" in state.endpoint_candidates
    assert "http://192.168.1.20:5000" in state.endpoint_candidates


def test_dfs_master_accepts_digits_as_builtin_dataset(tmp_path: Path) -> None:
    """The DFS-lite master should accept digits as a larger builtin dataset option."""

    config_path = tmp_path / "config_extended.json"
    write_extended_config(config_path, [allocate_port(), allocate_port()])
    service = FederatedMasterDFS(load_config(config_path), upload_dir=tmp_path / "uploads")

    updated_config = service.update_runtime_config({"dataset": {"name": "digits"}})
    features, labels = service.load_dataset()

    assert updated_config["dataset"]["name"] == "digits"
    assert features.shape[0] > 1000
    assert len(np.unique(labels)) == 10
