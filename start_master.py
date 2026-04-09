"""Cross-platform DFS-lite master bootstrap."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the master bootstrap."""

    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Create the DFS-lite master virtual environment and launch the master dashboard.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("CONFIG_PATH", repo_root / "config_extended.json")),
        help="Path to the DFS-lite JSON configuration file.",
    )
    parser.add_argument(
        "--python-bin",
        default=os.environ.get("PYTHON_BIN", sys.executable),
        help="Python interpreter used to create the virtual environment.",
    )
    parser.add_argument(
        "--venv-dir",
        type=Path,
        default=repo_root / ".venv_master_dfs",
        help="Virtual environment directory for the master dependencies.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("MASTER_HOST", "0.0.0.0"),
        help="Host interface to bind.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MASTER_PORT", "18080")),
        help="Host port for the DFS-lite master dashboard.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("MASTER_LOG_LEVEL", "INFO"),
        help="Logging level for the master process.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Reuse the existing virtual environment without reinstalling dependencies.",
    )
    parser.add_argument(
        "--allow-unsupported-python",
        action="store_true",
        help="Allow Python versions below 3.14 for local development smoke tests.",
    )
    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="Do not open the master dashboard automatically.",
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Start the Flask dashboard without automatically starting background training.",
    )
    return parser.parse_args()


def venv_python_path(venv_dir: Path) -> Path:
    """Return the Python executable path inside a virtual environment."""

    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def ensure_supported_python(*, allow_unsupported: bool) -> None:
    """Enforce the repository's preferred Python floor for the master bootstrap."""

    if sys.version_info < (3, 14) and not allow_unsupported:
        raise SystemExit(
            "Python 3.14+ is required for start_master.py. "
            "Set --allow-unsupported-python only for local development smoke tests."
        )


def run_command(arguments: list[str], *, env: dict[str, str] | None = None) -> None:
    """Run a subprocess and stop immediately if it fails."""

    subprocess.run(arguments, check=True, env=env)


def ensure_virtualenv(args: argparse.Namespace, requirements_path: Path) -> Path:
    """Create the master virtual environment and optionally install dependencies."""

    venv_dir = args.venv_dir.resolve()
    venv_python = venv_python_path(venv_dir)
    if not venv_python.exists():
        run_command([args.python_bin, "-m", "venv", str(venv_dir)])

    if not args.skip_install:
        run_command([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
        run_command([str(venv_python), "-m", "pip", "install", "-r", str(requirements_path)])

    return venv_python


def maybe_open_browser(url: str, *, disabled: bool) -> None:
    """Open the dashboard URL in the default browser after a short delay."""

    if disabled:
        return

    def open_url() -> None:
        """Launch the user's default browser for the dashboard URL."""

        webbrowser.open(url)

    threading.Timer(2.0, open_url).start()


def get_lan_ip() -> str:
    """Return the best-effort LAN IPv4 address for this host."""

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe_socket:
        try:
            probe_socket.connect(("8.8.8.8", 80))
            return str(probe_socket.getsockname()[0])
        except OSError:
            return "127.0.0.1"


def main() -> int:
    """Create the virtual environment and launch the DFS-lite master process."""

    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    allow_unsupported_env = os.environ.get("ALLOW_UNSUPPORTED_PYTHON")
    allow_unsupported_default = allow_unsupported_env != "0"
    allow_unsupported = args.allow_unsupported_python or allow_unsupported_default
    ensure_supported_python(allow_unsupported=allow_unsupported)

    requirements_path = repo_root / "master" / "requirements_extended.txt"
    browser_host = get_lan_ip() if args.host in {"0.0.0.0", "::"} else args.host
    master_url = f"http://{browser_host}:{args.port}"
    venv_python = ensure_virtualenv(args, requirements_path)

    child_env = os.environ.copy()
    if allow_unsupported:
        child_env["ALLOW_UNSUPPORTED_PYTHON"] = "1"

    maybe_open_browser(master_url, disabled=args.no_open_browser)
    command = [
        str(venv_python),
        "-m",
        "master.master_dfs",
        "--config",
        str(args.config.resolve()),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--log-level",
        args.log_level,
    ]
    if not args.no_auto_start:
        command.append("--auto-start")

    completed = subprocess.run(command, env=child_env)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
