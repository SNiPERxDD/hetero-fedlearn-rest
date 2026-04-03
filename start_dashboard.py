"""One-command DFS-lite dashboard bootstrap for localhost demos."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the dashboard bootstrap."""

    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build local DFS-lite worker containers and launch the master dashboard.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "config_extended.json",
        help="Path to the DFS-lite JSON configuration file.",
    )
    parser.add_argument(
        "--image",
        default="hetero-fedlearn-worker-dfs:test",
        help="Docker image tag for the DFS-lite worker container.",
    )
    parser.add_argument(
        "--container-prefix",
        default="hetero-fedlearn-dashboard",
        help="Prefix used for worker container names.",
    )
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=repo_root / ".dashboard_storage",
        help="Host directory that receives the worker block files.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse the existing worker image instead of rebuilding it.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=8080,
        help="Host port for the DFS-lite master dashboard.",
    )
    parser.add_argument(
        "--allow-unsupported-python",
        action="store_true",
        help="Forward ALLOW_UNSUPPORTED_PYTHON=1 to start_master.sh for local smoke tests.",
    )
    return parser.parse_args()


def ensure_command_available(command_name: str) -> None:
    """Exit with a clear message when a required external command is missing."""

    if shutil.which(command_name) is None:
        raise SystemExit(f"Required command '{command_name}' is not available on PATH.")


def run_command(
    arguments: list[str],
    *,
    env: dict[str, str] | None = None,
    stdout: int | None = None,
    stderr: int | None = None,
) -> None:
    """Run a subprocess and fail immediately if it exits unsuccessfully."""

    subprocess.run(arguments, check=True, env=env, stdout=stdout, stderr=stderr)


def load_worker_specs(config_path: Path) -> list[tuple[str, int]]:
    """Load localhost worker ids and ports from the DFS-lite configuration."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    workers = config.get("workers", [])
    if not workers:
        raise SystemExit("No workers are defined in the configured workers list.")

    worker_specs: list[tuple[str, int]] = []
    for worker in workers:
        worker_id = str(worker["worker_id"])
        endpoint = str(worker["endpoint"])
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise SystemExit(
                "start_dashboard.py only supports localhost worker endpoints. "
                f"Received {endpoint!r} for worker {worker_id!r}."
            )
        if parsed.port is None:
            raise SystemExit(f"Worker endpoint {endpoint!r} is missing an explicit port.")
        worker_specs.append((worker_id, int(parsed.port)))
    return worker_specs


def ensure_port_available(port: int, label: str) -> None:
    """Fail fast when a required localhost port is already bound."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(f"{label} port {port} is already in use on 127.0.0.1.")


def wait_for_worker_health(port: int, worker_id: str) -> None:
    """Wait until the worker health endpoint responds with HTTP 200."""

    health_url = f"http://127.0.0.1:{port}/health"
    for attempt in range(20):
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 19:
                raise SystemExit(f"Worker {worker_id} did not become healthy in time.")
            time.sleep(2)


def cleanup_workers(container_names: list[str]) -> None:
    """Remove worker containers that were started by this bootstrap."""

    for container_name in container_names:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def register_cleanup(container_names: list[str]) -> None:
    """Register cleanup handlers for normal process exit and termination signals."""

    def handle_signal(signum: int, _frame: object) -> None:
        """Stop worker containers before propagating the terminating signal."""

        cleanup_workers(container_names)
        raise SystemExit(128 + signum)

    atexit.register(cleanup_workers, container_names)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


def main() -> int:
    """Build local DFS-lite workers, wait for health, then launch the master."""

    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    config_path = Path(args.config).resolve()
    storage_root = Path(args.storage_root).resolve()

    ensure_command_available("docker")
    run_command(["docker", "info"], stdout=subprocess.DEVNULL)

    worker_specs = load_worker_specs(config_path)
    container_names = [f"{args.container_prefix}-{worker_id}" for worker_id, _ in worker_specs]
    register_cleanup(container_names)

    storage_root.mkdir(parents=True, exist_ok=True)

    ensure_port_available(args.master_port, "Master dashboard")
    for worker_id, host_port in worker_specs:
        ensure_port_available(host_port, f"Worker {worker_id}")

    if not args.skip_build:
        print(f"Building DFS-lite worker image: {args.image}")
        run_command(
            [
                "docker",
                "build",
                "-t",
                args.image,
                "-f",
                str(repo_root / "worker" / "Dockerfile_extended"),
                str(repo_root / "worker"),
            ]
        )

    for worker_id, host_port in worker_specs:
        storage_dir = storage_root / worker_id
        storage_dir.mkdir(parents=True, exist_ok=True)
        container_name = f"{args.container_prefix}-{worker_id}"

        subprocess.run(
            ["docker", "rm", "-f", container_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"Starting worker {worker_id} on http://127.0.0.1:{host_port}")
        run_command(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                container_name,
                "-e",
                f"WORKER_ID={worker_id}",
                "-p",
                f"{host_port}:5000",
                "-v",
                f"{storage_dir}:/app/datanode_storage",
                args.image,
            ]
        )

    for worker_id, host_port in worker_specs:
        print(f"Waiting for {worker_id} health at http://127.0.0.1:{host_port}/health")
        wait_for_worker_health(host_port, worker_id)

    print(f"Workers are healthy. Launching master dashboard on http://127.0.0.1:{args.master_port}")
    print(f"Worker storage root: {storage_root}")

    env = os.environ.copy()
    env["CONFIG_PATH"] = str(config_path)
    env["MASTER_PORT"] = str(args.master_port)
    if args.allow_unsupported_python:
        env["ALLOW_UNSUPPORTED_PYTHON"] = "1"

    master_script = repo_root / "start_master.sh"
    run_command([str(master_script)], env=env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
