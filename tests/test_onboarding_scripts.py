"""Verification tests for the repository onboarding scripts."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_start_master_shell_script_is_valid_bash() -> None:
    """The master bootstrap script must parse and preserve the DFS-lite entrypoint."""

    script_path = REPO_ROOT / "start_master.sh"
    subprocess.run(["bash", "-n", str(script_path)], check=True, capture_output=True, text=True)

    script_text = script_path.read_text(encoding="utf-8")
    assert "Python 3.14+ is required for start_master.sh" in script_text
    assert 'CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/config_extended.json}"' in script_text
    assert 'MASTER_PORT="${MASTER_PORT:-18080}"' in script_text
    assert "--config \"$CONFIG_PATH\"" in script_text
    assert "--auto-start" in script_text
    assert "master/requirements_extended.txt" in script_text


def test_start_dashboard_python_launcher_compiles_and_declares_contract() -> None:
    """The dashboard quick-start launcher must build workers and chain into the master bootstrap."""

    script_path = REPO_ROOT / "start_dashboard.py"
    subprocess.run(["python3", "-m", "py_compile", str(script_path)], check=True, capture_output=True, text=True)

    script_text = script_path.read_text(encoding="utf-8")
    assert 'parser.add_argument(' in script_text
    assert '"--master-port"' in script_text
    assert '"docker"' in script_text
    assert '"build"' in script_text
    assert 'ensure_port_available' in script_text
    assert 'wait_for_worker_health' in script_text
    assert 'start_dashboard.py only supports localhost worker endpoints' in script_text
    assert 'start_master.sh' in script_text


def test_windows_batch_onboarding_contract_is_present() -> None:
    """The Windows batch launcher must build, mount storage, and expose the worker port."""

    script_text = (REPO_ROOT / "start_worker.bat").read_text(encoding="utf-8")

    assert "docker info >nul 2>&1" in script_text
    assert "docker build -t %IMAGE_NAME% -f worker\\Dockerfile_extended worker" in script_text
    assert "-v \"%cd%\\storage:/app/datanode_storage\"" in script_text
    assert "Please start Docker Desktop before running start_worker.bat" in script_text
    assert "DFS-lite worker started on http://127.0.0.1:%HOST_PORT%" in script_text


def test_windows_powershell_onboarding_contract_is_present() -> None:
    """The PowerShell onboarding helper must enforce firewall, Docker, and health checks."""

    script_text = (REPO_ROOT / "scripts" / "windows" / "onboard_worker.ps1").read_text(encoding="utf-8")

    assert "function Ensure-FirewallRule" in script_text
    assert "function Set-ConnectedNetworksPrivate" in script_text
    assert "function Wait-WorkerHealth" in script_text
    assert "Invoke-RestMethod -Method Get -Uri $healthUri -TimeoutSec 5" in script_text
    assert 'throw "Run this script from an elevated PowerShell session' in script_text
    assert 'Invoke-Docker -Arguments @("version") | Out-Null' in script_text
    assert '"--restart"' in script_text
    assert '"unless-stopped"' in script_text
    assert 'Write-Host "Worker onboarding completed successfully."' in script_text
