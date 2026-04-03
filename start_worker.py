"""Cross-platform DFS-lite worker bootstrap."""

from __future__ import annotations

import argparse
import os
import shutil
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
        default=os.environ.get("WORKER_ID", "worker-node"),
        help="Worker identifier exposed by the DFS-lite worker service.",
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


def run_native_worker(args: argparse.Namespace, *, repo_root: Path, storage_dir: Path) -> int:
    """Launch the DFS-lite worker directly through a Python virtual environment."""

    allow_unsupported = args.allow_unsupported_python or os.environ.get("ALLOW_UNSUPPORTED_PYTHON") == "1"
    ensure_supported_python(allow_unsupported=allow_unsupported, script_name="start_worker.py")
    storage_dir.mkdir(parents=True, exist_ok=True)

    requirements_path = repo_root / "worker" / "requirements.txt"
    worker_url = f"http://127.0.0.1:{args.port}"
    venv_python = ensure_virtualenv(args, requirements_path)
    child_env = os.environ.copy()
    if allow_unsupported:
        child_env["ALLOW_UNSUPPORTED_PYTHON"] = "1"

    maybe_open_browser(worker_url, disabled=args.no_open_browser)
    command = [
        str(venv_python),
        "-m",
        "worker.worker_dfs",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--worker-id",
        args.worker_id,
        "--storage-dir",
        str(storage_dir),
        "--log-level",
        args.log_level,
    ]
    completed = subprocess.run(command, env=child_env)
    return int(completed.returncode)


def run_docker_worker(args: argparse.Namespace, *, repo_root: Path, storage_dir: Path) -> int:
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

    run_command(
        [
            "docker",
            "run",
            "-d",
            "--restart",
            args.restart_policy,
            "--name",
            args.container_name,
            "-e",
            f"WORKER_ID={args.worker_id}",
            "-p",
            f"{args.port}:5000",
            "-v",
            f"{storage_dir.resolve()}:/app/datanode_storage",
            args.image,
        ]
    )
    wait_for_worker_health(args.port)
    maybe_open_browser(f"http://127.0.0.1:{args.port}", disabled=args.no_open_browser)
    print(f"DFS-lite worker started on http://127.0.0.1:{args.port}")
    return 0


def main() -> int:
    """Launch the DFS-lite worker in the requested execution mode."""

    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    storage_dir = (args.storage_dir or default_storage_dir(repo_root, args.worker_id)).resolve()
    if args.mode == "native":
        return run_native_worker(args, repo_root=repo_root, storage_dir=storage_dir)
    return run_docker_worker(args, repo_root=repo_root, storage_dir=storage_dir)


if __name__ == "__main__":
    raise SystemExit(main())
