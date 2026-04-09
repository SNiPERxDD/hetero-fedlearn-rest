"""Cross-platform DFS-lite worker bootstrap."""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the worker bootstrap."""

    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Launch a DFS-lite worker natively or in Docker from a cross-platform Python entry point.",
    )
    parser.add_argument(
        "--mode",
        choices=("native", "docker"),
        default="native",
        help="Choose native Python execution or Docker container execution.",
    )
    parser.add_argument(
        "--python-bin",
        default=os.environ.get("PYTHON_BIN", sys.executable),
        help="Python interpreter used to create the worker virtual environment in native mode.",
    )
    parser.add_argument(
        "--venv-dir",
        type=Path,
        default=repo_root / ".venv_worker_dfs",
        help="Virtual environment directory for native worker mode.",
    )
    parser.add_argument(
        "--worker-id",
        default=os.environ.get("WORKER_ID"),
        help="Worker identifier exposed by the DFS-lite worker service. Defaults to <hostname>-<port>.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("WORKER_HOST", "0.0.0.0"),
        help="Host interface to bind in native mode.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("WORKER_PORT", "5000")),
        help="Host port for the worker dashboard and health endpoint.",
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=None,
        help="Host storage directory for block persistence. Defaults to ./storage/<worker-id>.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("WORKER_LOG_LEVEL", "INFO"),
        help="Logging level for native worker mode.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Reuse the existing virtual environment without reinstalling dependencies in native mode.",
    )
    parser.add_argument(
        "--allow-unsupported-python",
        action="store_true",
        help="Allow Python versions below 3.14 for local development smoke tests.",
    )
    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="Do not open the worker dashboard automatically.",
    )
    parser.add_argument(
        "--image",
        default="hetero-fedlearn-worker-dfs:test",
        help="Docker image tag for worker docker mode.",
    )
    parser.add_argument(
        "--container-name",
        default="worker-node",
        help="Container name used in docker mode.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse the existing image instead of rebuilding it in docker mode.",
    )
    parser.add_argument(
        "--restart-policy",
        default="unless-stopped",
        help="Docker restart policy used in docker mode.",
    )
    parser.add_argument(
        "--udp-discovery-targets",
        default=os.environ.get("UDP_DISCOVERY_TARGETS"),
        help=(
            "Optional comma-separated IPv4 targets for UDP beacon unicast fallback "
            "(for networks where broadcast is filtered)."
        ),
    )
    return parser.parse_args()


def venv_python_path(venv_dir: Path) -> Path:
    """Return the Python executable path inside a virtual environment."""

    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def ensure_supported_python(*, allow_unsupported: bool, script_name: str) -> None:
    """Enforce the repository's preferred Python floor for native launchers."""

    if sys.version_info < (3, 14) and not allow_unsupported:
        raise SystemExit(
            f"Python 3.14+ is required for {script_name}. "
            "Set --allow-unsupported-python only for local development smoke tests."
        )


def run_command(arguments: list[str], *, env: dict[str, str] | None = None) -> None:
    """Run a subprocess and stop immediately if it fails."""

    subprocess.run(arguments, check=True, env=env)


def ensure_command_available(command_name: str) -> None:
    """Exit with a clear message when a required external command is missing."""

    if shutil.which(command_name) is None:
        raise SystemExit(f"Required command '{command_name}' is not available on PATH.")


def ensure_virtualenv(args: argparse.Namespace, requirements_path: Path) -> Path:
    """Create the worker virtual environment and optionally install dependencies."""

    venv_dir = args.venv_dir.resolve()
    venv_python = venv_python_path(venv_dir)
    if not venv_python.exists():
        run_command([args.python_bin, "-m", "venv", str(venv_dir)])

    if not args.skip_install:
        run_command([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
        run_command([str(venv_python), "-m", "pip", "install", "-r", str(requirements_path)])

    return venv_python


def maybe_open_browser(url: str, *, disabled: bool) -> None:
    """Open the worker dashboard URL in the default browser after a short delay."""

    if disabled:
        return

    def open_url() -> None:
        """Launch the user's default browser for the worker URL."""

        webbrowser.open(url)

    threading.Timer(2.0, open_url).start()


def wait_for_worker_health(port: int) -> None:
    """Wait until the worker health endpoint responds with HTTP 200."""

    health_url = f"http://127.0.0.1:{port}/health"
    for attempt in range(20):
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 19:
                raise SystemExit(f"Worker health endpoint did not become ready at {health_url}.")
            time.sleep(2)


def default_storage_dir(repo_root: Path, worker_id: str) -> Path:
    """Build the default worker storage directory path."""

    return repo_root / "storage" / worker_id


def is_port_available(host: str, port: int) -> bool:
    """Return whether a TCP port can be bound on the requested host."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe_socket:
        probe_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe_socket.bind((host, port))
        except OSError:
            return False
    return True


def resolve_port(host: str, requested_port: int, max_offset: int = 100) -> int:
    """Resolve the first free TCP port at or above the requested port."""

    if requested_port <= 0:
        raise SystemExit("--port must be a positive integer.")

    for offset in range(max_offset + 1):
        candidate = requested_port + offset
        if candidate > 65535:
            break
        if is_port_available(host, candidate):
            return candidate
    raise SystemExit(
        f"No available port found from {requested_port} to {min(requested_port + max_offset, 65535)} on {host}."
    )


def resolve_worker_id(requested_worker_id: str | None, port: int) -> str:
    """Resolve the worker identifier, defaulting to hostname-port."""

    if requested_worker_id and requested_worker_id.strip():
        return requested_worker_id.strip()
    return f"{socket.gethostname()}-{port}"


def run_native_worker(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    storage_dir: Path,
    worker_id: str,
    port: int,
) -> int:
    """Launch the DFS-lite worker directly through a Python virtual environment."""

    allow_unsupported_env = os.environ.get("ALLOW_UNSUPPORTED_PYTHON")
    allow_unsupported_default = allow_unsupported_env != "0"
    allow_unsupported = args.allow_unsupported_python or allow_unsupported_default
    ensure_supported_python(allow_unsupported=allow_unsupported, script_name="start_worker.py")
    storage_dir.mkdir(parents=True, exist_ok=True)

    requirements_path = repo_root / "worker" / "requirements.txt"
    worker_url = f"http://127.0.0.1:{port}"
    venv_python = ensure_virtualenv(args, requirements_path)
    child_env = os.environ.copy()
    if allow_unsupported:
        child_env["ALLOW_UNSUPPORTED_PYTHON"] = "1"
    child_env["WORKER_PORT"] = str(port)
    child_env["WORKER_ID"] = worker_id
    if args.udp_discovery_targets:
        child_env["UDP_DISCOVERY_TARGETS"] = args.udp_discovery_targets

    maybe_open_browser(worker_url, disabled=args.no_open_browser)
    command = [
        str(venv_python),
        "-m",
        "worker.worker_dfs",
        "--host",
        args.host,
        "--port",
        str(port),
        "--worker-id",
        worker_id,
        "--storage-dir",
        str(storage_dir),
        "--log-level",
        args.log_level,
    ]
    completed = subprocess.run(command, env=child_env)
    return int(completed.returncode)


def run_docker_worker(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    storage_dir: Path,
    worker_id: str,
    port: int,
) -> int:
    """Launch the DFS-lite worker in Docker and wait for its health endpoint."""

    ensure_command_available("docker")
    run_command(["docker", "info"], env=os.environ.copy())
    storage_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["docker", "rm", "-f", args.container_name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not args.skip_build:
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

    docker_run_command = [
        "docker",
        "run",
        "-d",
        "--restart",
        args.restart_policy,
        "--name",
        args.container_name,
        "-e",
        f"WORKER_ID={worker_id}",
    ]
    if args.udp_discovery_targets:
        docker_run_command.extend(["-e", f"UDP_DISCOVERY_TARGETS={args.udp_discovery_targets}"])

    docker_run_command.extend(
        [
            "-p",
            f"{port}:5000",
            "-v",
            f"{storage_dir.resolve()}:/app/datanode_storage",
            args.image,
        ]
    )
    run_command(docker_run_command)
    wait_for_worker_health(port)
    maybe_open_browser(f"http://127.0.0.1:{port}", disabled=args.no_open_browser)
    print(f"DFS-lite worker started on http://127.0.0.1:{port}")
    return 0


def main() -> int:
    """Launch the DFS-lite worker in the requested execution mode."""

    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    port_probe_host = args.host if args.mode == "native" else "127.0.0.1"
    resolved_port = resolve_port(port_probe_host, int(args.port))
    if resolved_port != int(args.port):
        print(f"Port {args.port} is in use; automatically selected port {resolved_port}.")

    resolved_worker_id = resolve_worker_id(args.worker_id, resolved_port)
    storage_dir = (args.storage_dir or default_storage_dir(repo_root, resolved_worker_id)).resolve()
    if args.mode == "native":
        return run_native_worker(
            args,
            repo_root=repo_root,
            storage_dir=storage_dir,
            worker_id=resolved_worker_id,
            port=resolved_port,
        )
    return run_docker_worker(
        args,
        repo_root=repo_root,
        storage_dir=storage_dir,
        worker_id=resolved_worker_id,
        port=resolved_port,
    )


if __name__ == "__main__":
    raise SystemExit(main())
