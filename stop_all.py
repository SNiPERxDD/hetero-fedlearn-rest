"""Stop repo-managed DFS-lite master and worker processes or containers."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the repo cleanup utility."""

    parser = argparse.ArgumentParser(
        description="Stop the repo-managed DFS-lite master and worker services so their ports are freed.",
    )
    parser.add_argument(
        "--ports",
        nargs="*",
        type=int,
        default=[18080, 5000, 5001, 5002],
        help="Ports whose Python listeners should be terminated if they belong to this repo runtime.",
    )
    parser.add_argument(
        "--container-prefix",
        default="hetero-fedlearn-dashboard",
        help="Dashboard worker container prefix to remove.",
    )
    parser.add_argument(
        "--container-name",
        action="append",
        default=["worker-node"],
        help="Additional exact Docker container names to remove. May be provided multiple times.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=5.0,
        help="How long to wait after SIGTERM before escalating to SIGKILL on POSIX.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be stopped without terminating anything.",
    )
    return parser.parse_args()


def command_exists(command_name: str) -> bool:
    """Return whether an external command is available on PATH."""

    from shutil import which

    return which(command_name) is not None


def list_repo_containers(*, container_prefix: str, container_names: list[str]) -> list[str]:
    """Return Docker container names managed by the repository launchers."""

    if not command_exists("docker"):
        return []
    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []

    discovered_names: list[str] = []
    known_exact = set(container_names)
    prefix = f"{container_prefix}-"
    for line in result.stdout.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if candidate in known_exact or candidate.startswith(prefix):
            discovered_names.append(candidate)
    return sorted(set(discovered_names))


def stop_containers(container_names: list[str], *, dry_run: bool) -> list[str]:
    """Remove the requested Docker containers and return the stopped names."""

    stopped_names: list[str] = []
    for container_name in container_names:
        if dry_run:
            stopped_names.append(container_name)
            continue
        result = subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            stopped_names.append(container_name)
    return stopped_names


def pid_listening_on_port(port: int) -> set[int]:
    """Return the process ids listening on a given port for the current platform."""

    if os.name == "nt":
        result = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return set()
        pid_matches: set[int] = set()
        suffix = f":{port}"
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0].upper() == "TCP" and parts[1].endswith(suffix) and parts[3] == "LISTENING":
                pid_matches.add(int(parts[4]))
        return pid_matches

    if not command_exists("lsof"):
        return set()
    result = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in {0, 1}:
        return set()
    return {int(line.strip()) for line in result.stdout.splitlines() if line.strip()}


def process_command_line(pid: int) -> str:
    """Return the command line for a process id or an empty string if unavailable."""

    if os.name == "nt":
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(Get-CimInstance Win32_Process -Filter \"ProcessId = {pid}\").CommandLine",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip()

    if Path(f"/proc/{pid}/cmdline").exists():
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ")
            return raw.decode("utf-8", errors="ignore").strip()
        except OSError:
            return ""

    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def is_repo_runtime_process(command_line: str) -> bool:
    """Return whether a command line belongs to this repository's master or worker runtime."""

    return "-m master.master_dfs" in command_line or "-m worker.worker_dfs" in command_line


def terminate_pid(pid: int, *, dry_run: bool, timeout_seconds: float) -> bool:
    """Terminate a single process id and return whether it was handled successfully."""

    if dry_run:
        return True

    if os.name == "nt":
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.2)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    return True


def main() -> int:
    """Stop repo-managed master or worker services and report the cleanup summary."""

    args = parse_args()

    candidate_pids: set[int] = set()
    for port in args.ports:
        candidate_pids.update(pid_listening_on_port(port))

    handled_processes: list[tuple[int, int, str]] = []
    for pid in sorted(candidate_pids):
        command_line = process_command_line(pid)
        if not is_repo_runtime_process(command_line):
            continue
        if terminate_pid(pid, dry_run=args.dry_run, timeout_seconds=args.timeout_seconds):
            handled_processes.append((pid, _infer_port_for_pid(pid, args.ports), command_line))

    container_names = list_repo_containers(
        container_prefix=args.container_prefix,
        container_names=args.container_name,
    )
    stopped_containers = stop_containers(container_names, dry_run=args.dry_run)

    mode_label = "Would stop" if args.dry_run else "Stopped"
    if handled_processes:
        for pid, port, command_line in handled_processes:
            port_label = f"port {port}" if port > 0 else "unknown port"
            print(f"{mode_label} process {pid} ({port_label}): {command_line}")
    if stopped_containers:
        for container_name in stopped_containers:
            print(f"{mode_label} container {container_name}")
    if not handled_processes and not stopped_containers:
        print("No repo-managed master or worker services were found.")
    return 0


def _infer_port_for_pid(pid: int, ports: list[int]) -> int:
    """Infer the managed port owned by a process id when possible."""

    for port in ports:
        if pid in pid_listening_on_port(port):
            return port
    return -1


if __name__ == "__main__":
    raise SystemExit(main())
