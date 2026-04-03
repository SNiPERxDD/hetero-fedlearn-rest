"""Browser-level UI tests for the DFS-lite dashboards."""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect
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


def write_extended_config(config_path: Path, worker_ports: list[int]) -> None:
    """Write a deterministic DFS-lite configuration for UI tests."""

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
            {"worker_id": "worker_1", "endpoint": f"http://127.0.0.1:{worker_ports[0]}"},
            {"worker_id": "worker_2", "endpoint": f"http://127.0.0.1:{worker_ports[1]}"},
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
