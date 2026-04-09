"""Browser-level UI tests for the DFS-lite dashboards."""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect
from sklearn.datasets import make_classification
from werkzeug.serving import make_server

from master.master_dfs import FederatedMasterDFS, create_app as create_master_app, load_config
from worker.worker_dfs import create_app as create_worker_app


def allocate_port() -> int:
    """Allocate an ephemeral TCP port for a test server."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LocalServer:
    """Run a WSGI app in-process on a disposable local port."""

    def __init__(self, port: int, app) -> None:
        """Create the WSGI server and serving thread."""

        self.port = port
        self.server = make_server("127.0.0.1", port, app)
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


def write_extended_config(
    config_path: Path,
    worker_ports: list[int],
    *,
    worker_count: int | None = None,
) -> None:
    """Write a deterministic DFS-lite configuration for UI tests."""

    selected_ports = worker_ports[:worker_count] if worker_count is not None else worker_ports
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
            "replication_factor": 1,
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
            {"worker_id": f"worker_{index + 1}", "endpoint": f"http://127.0.0.1:{port}"}
            for index, port in enumerate(selected_ports)
        ],
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")


@pytest.fixture(name="dfs_ui_stack")
def fixture_dfs_ui_stack(tmp_path: Path) -> tuple[str, list[str], list[Path]]:
    """Start a disposable DFS-lite master and two workers for browser tests."""

    worker_ports = [allocate_port(), allocate_port()]
    worker_storage_dirs = [tmp_path / "worker_1_storage", tmp_path / "worker_2_storage"]
    master_port = allocate_port()

    worker_servers = [
        LocalServer(worker_ports[0], create_worker_app(default_worker_id="worker_1", storage_dir=worker_storage_dirs[0])),
        LocalServer(worker_ports[1], create_worker_app(default_worker_id="worker_2", storage_dir=worker_storage_dirs[1])),
    ]
    for server in worker_servers:
        server.start()

    master_server = None
    try:
        config_path = tmp_path / "config_extended.json"
        write_extended_config(config_path, worker_ports)
        service = FederatedMasterDFS(load_config(config_path))
        master_app = create_master_app(config_path=config_path, autostart=False, service=service)
        master_server = LocalServer(master_port, master_app)
        master_server.start()

        yield (
            f"http://127.0.0.1:{master_port}",
            [f"http://127.0.0.1:{worker_ports[0]}", f"http://127.0.0.1:{worker_ports[1]}"],
            worker_storage_dirs,
        )
    finally:
        if master_server is not None:
            master_server.stop()
        for server in worker_servers:
            server.stop()


def test_dashboard_ui_flow(page: Page, dfs_ui_stack: tuple[str, list[str], list[Path]]) -> None:
    """The live dashboards must reflect a real DFS-lite training run."""

    master_url, worker_urls, worker_storage_dirs = dfs_ui_stack

    page.goto(master_url, wait_until="domcontentloaded")
    expect(page.get_by_role("heading", name="Move Training To The Blocks, Not The Blocks To Training.")).to_be_visible()
    expect(page.locator("#cluster-state")).to_have_text("Idle")
    expect(page.locator("#config-rounds")).to_have_value("6")
    expect(page.locator("#worker-form")).to_be_visible()
    page.get_by_role("button", name="Start Background Training").click()
    expect(page.locator("#cluster-state")).to_have_text("Completed", timeout=30000)
    expect(page.locator("#current-round")).to_have_text("6 / 6", timeout=30000)

    accuracy_text = page.locator("#validation-accuracy").text_content(timeout=30000)
    assert accuracy_text is not None
    assert float(accuracy_text) >= 0.95
    expect(page.locator("#accuracy-dots circle.dot")).to_have_count(7, timeout=30000)
    expect(page.locator("#block-map-body tr")).to_have_count(2, timeout=30000)
    expect(page.locator("#worker-health-body tr")).to_have_count(2, timeout=30000)

    for index, worker_url in enumerate(worker_urls):
        page.goto(worker_url, wait_until="domcontentloaded")
        expect(page.locator("#worker-title")).to_have_text(f"worker_{index + 1}")
        expect(page.locator("#connect-form")).to_be_visible()
        expect(page.locator("#ready-pill")).to_have_text("Ready", timeout=30000)
        expect(page.locator("#block-count")).to_have_text("1", timeout=30000)
        expect(page.locator("#storage-dir")).to_contain_text(str(worker_storage_dirs[index]))
        assert len(list(worker_storage_dirs[index].glob("*.csv"))) == 1


def test_dashboard_ui_mobile_layout(
    page: Page,
    dfs_ui_stack: tuple[str, list[str], list[Path]],
) -> None:
    """The master dashboard must stay usable at a mobile viewport."""

    master_url, _, _ = dfs_ui_stack

    page.set_viewport_size({"width": 430, "height": 932})
    page.goto(master_url, wait_until="domcontentloaded")

    hero_actions = page.locator(".hero-actions")
    hero_actions_box = hero_actions.bounding_box()
    assert hero_actions_box is not None
    assert hero_actions_box["width"] <= 430

    hero_grid_template = page.locator(".hero").evaluate(
        """(element) => getComputedStyle(element).gridTemplateColumns"""
    )
    assert len(hero_grid_template.split()) == 1

    page.get_by_role("button", name="Refresh").click()
    expect(page.locator("#worker-health-body tr")).to_have_count(2)


def test_browser_control_plane_workflow(page: Page, tmp_path: Path) -> None:
    """The browser UI must support worker registration, CSV upload, config edits, and training."""

    worker_ports = [allocate_port(), allocate_port()]
    worker_storage_dirs = [tmp_path / "worker_1_storage", tmp_path / "worker_2_storage"]
    master_port = allocate_port()

    worker_servers = [
        LocalServer(worker_ports[0], create_worker_app(default_worker_id="worker_1", storage_dir=worker_storage_dirs[0])),
        LocalServer(worker_ports[1], create_worker_app(default_worker_id="worker_2", storage_dir=worker_storage_dirs[1])),
    ]
    for server in worker_servers:
        server.start()

    master_server = None
    try:
        config_path = tmp_path / "config_extended.json"
        write_extended_config(config_path, worker_ports, worker_count=1)
        service = FederatedMasterDFS(load_config(config_path), upload_dir=tmp_path / "uploads")
        master_app = create_master_app(config_path=config_path, autostart=False, service=service)
        master_server = LocalServer(master_port, master_app)
        master_server.start()

        master_url = f"http://127.0.0.1:{master_port}"
        worker_2_url = f"http://127.0.0.1:{worker_ports[1]}"

        page.goto(worker_2_url, wait_until="domcontentloaded")
        page.locator("#master-endpoint").fill(master_url)
        page.locator("#advertised-endpoint").fill(worker_2_url)
        page.get_by_role("button", name="Register With Master").click()
        expect(page.locator("#connection-status")).to_contain_text("Connected", timeout=30000)
        expect(page.locator("#registration-state")).to_have_text("connected", timeout=30000)

        features, labels = make_classification(
            n_samples=240,
            n_features=12,
            n_informative=8,
            n_redundant=0,
            n_clusters_per_class=1,
            class_sep=2.2,
            random_state=42,
        )
        dataset_path = tmp_path / "browser_upload.csv"
        write_csv_dataset(dataset_path, features, labels)

        page.goto(master_url, wait_until="domcontentloaded")
        expect(page.locator("#worker-config-body tr")).to_have_count(2, timeout=30000)
        page.locator("#config-rounds").fill("3")
        page.locator("#config-local-epochs").fill("2")
        page.get_by_role("button", name="Save Settings").click()
        expect(page.locator("#action-banner")).to_have_text("Training settings updated.", timeout=30000)

        page.locator("#dataset-file").set_input_files(str(dataset_path))
        page.get_by_role("button", name="Upload CSV Dataset").click()
        expect(page.locator("#dataset-summary")).to_contain_text("Dataset source: csv", timeout=30000)

        page.get_by_role("button", name="Start Background Training").click()
        expect(page.locator("#cluster-state")).to_have_text("Completed", timeout=30000)
        expect(page.locator("#current-round")).to_have_text("3 / 3", timeout=30000)
        expect(page.locator("#worker-health-body tr")).to_have_count(2, timeout=30000)

        accuracy_text = page.locator("#validation-accuracy").text_content(timeout=30000)
        assert accuracy_text is not None
        assert float(accuracy_text) >= 0.8
    finally:
        if master_server is not None:
            master_server.stop()
        for server in worker_servers:
            server.stop()


def test_worker_dashboard_registration_failure_state(page: Page, tmp_path: Path) -> None:
    """The worker dashboard must surface a failed master registration attempt."""

    worker_port = allocate_port()
    unavailable_master_port = allocate_port()
    worker_storage_dir = tmp_path / "worker_storage"
    worker_server = LocalServer(
        worker_port,
        create_worker_app(default_worker_id="worker_1", storage_dir=worker_storage_dir),
    )
    worker_server.start()

    try:
        worker_url = f"http://127.0.0.1:{worker_port}"
        page.goto(worker_url, wait_until="domcontentloaded")
        page.locator("#master-endpoint").fill(f"http://127.0.0.1:{unavailable_master_port}")
        page.locator("#advertised-endpoint").fill(worker_url)
        page.get_by_role("button", name="Register With Master").click()
        expect(page.locator("#connection-status")).to_contain_text(
            "Worker could not register with the master.",
            timeout=30000,
        )
        expect(page.locator("#registration-state")).to_have_text("failed", timeout=30000)
        expect(page.locator("#registration-error")).not_to_have_text("n/a", timeout=30000)
    finally:
        worker_server.stop()


def test_worker_dashboard_auto_registration_state_updates_banner(page: Page, tmp_path: Path, monkeypatch) -> None:
    """The worker dashboard banner should reflect auto-registration discovered via polling."""

    worker_port = allocate_port()
    master_port = allocate_port()
    worker_storage_dir = tmp_path / "worker_storage"

    config_path = tmp_path / "config_extended.json"
    write_extended_config(config_path, [allocate_port(), allocate_port()], worker_count=0)
    service = FederatedMasterDFS(load_config(config_path), upload_dir=tmp_path / "uploads")
    master_server = LocalServer(master_port, create_master_app(config_path=config_path, autostart=False, service=service))
    master_server.start()

    monkeypatch.setenv("MASTER_ENDPOINT", f"http://127.0.0.1:{master_port}")
    monkeypatch.setenv("ADVERTISED_ENDPOINT", f"http://127.0.0.1:{worker_port}")
    worker_server = LocalServer(
        worker_port,
        create_worker_app(
            default_worker_id="worker_auto_ui",
            storage_dir=worker_storage_dir,
            bound_host="127.0.0.1",
            bound_port=worker_port,
            enable_udp_beacon=False,
            enable_master_discovery=False,
        ),
    )
    worker_server.start()

    try:
        worker_url = f"http://127.0.0.1:{worker_port}"
        page.goto(worker_url, wait_until="domcontentloaded")
        expect(page.locator("#connection-status")).to_contain_text("Connected to", timeout=30000)
        expect(page.locator("#connection-status")).to_contain_text(f"http://127.0.0.1:{master_port}", timeout=30000)
        expect(page.locator("#registration-state")).to_have_text("connected", timeout=30000)
    finally:
        worker_server.stop()
        master_server.stop()


def test_master_dashboard_exposes_digits_builtin_dataset(page: Page, tmp_path: Path) -> None:
    """The master dashboard should surface the larger builtin digits dataset option."""

    worker_ports = [allocate_port(), allocate_port()]
    config_path = tmp_path / "config_extended.json"
    write_extended_config(config_path, worker_ports, worker_count=0)
    service = FederatedMasterDFS(load_config(config_path), upload_dir=tmp_path / "uploads")
    master_port = allocate_port()
    master_server = LocalServer(master_port, create_master_app(config_path=config_path, autostart=False, service=service))
    master_server.start()

    try:
        master_url = f"http://127.0.0.1:{master_port}"
        page.goto(master_url, wait_until="domcontentloaded")
        page.locator("#dataset-source").select_option("builtin")
        page.locator("#dataset-name").select_option("digits")
        page.get_by_role("button", name="Use Builtin Dataset").click()
        expect(page.locator("#dataset-summary")).to_contain_text("digits", timeout=30000)
        expect(page.locator("#action-banner")).to_contain_text("Builtin dataset selected: digits.", timeout=30000)
    finally:
        master_server.stop()


def test_master_dashboard_save_settings_preserves_selected_builtin_dataset(page: Page, tmp_path: Path) -> None:
    """Saving settings should not reset the selected builtin dataset back to breast_cancer."""

    worker_ports = [allocate_port(), allocate_port()]
    config_path = tmp_path / "config_extended.json"
    write_extended_config(config_path, worker_ports, worker_count=0)
    service = FederatedMasterDFS(load_config(config_path), upload_dir=tmp_path / "uploads")
    master_port = allocate_port()
    master_server = LocalServer(master_port, create_master_app(config_path=config_path, autostart=False, service=service))
    master_server.start()

    try:
        master_url = f"http://127.0.0.1:{master_port}"
        page.goto(master_url, wait_until="domcontentloaded")
        page.locator("#dataset-source").select_option("builtin")
        page.locator("#dataset-name").select_option("digits")
        page.locator("#config-rounds").fill("4")
        page.get_by_role("button", name="Save Settings").click()
        expect(page.locator("#dataset-summary")).to_contain_text("builtin · digits", timeout=30000)
        expect(page.locator("#dataset-name")).to_have_value("digits", timeout=30000)
        expect(page.locator("#action-banner")).to_contain_text("Training settings updated.", timeout=30000)
    finally:
        master_server.stop()
