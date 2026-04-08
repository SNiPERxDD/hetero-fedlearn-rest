"""Verification tests for the repository onboarding scripts."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_start_master_python_launcher_compiles_and_declares_contract() -> None:
    """The Python master launcher must provision a venv and launch the DFS-lite master."""

    script_path = REPO_ROOT / "start_master.py"
    subprocess.run(["python3", "-m", "py_compile", str(script_path)], check=True, capture_output=True, text=True)

    script_text = script_path.read_text(encoding="utf-8")
    assert "Python 3.14+ is required for start_master.py" in script_text
    assert '"--skip-install"' in script_text
    assert '"--no-open-browser"' in script_text
    assert '"--no-auto-start"' in script_text
    assert '"master" / "requirements_extended.txt"' in script_text
    assert '"master.master_dfs"' in script_text
    assert '"18080"' in script_text


def test_start_master_shell_script_delegates_to_python_launcher() -> None:
    """The legacy shell launcher must forward to the Python master bootstrap."""

    script_path = REPO_ROOT / "start_master.sh"
    subprocess.run(["bash", "-n", str(script_path)], check=True, capture_output=True, text=True)

    script_text = script_path.read_text(encoding="utf-8")
    assert 'exec "$PYTHON_BIN" "$ROOT_DIR/start_master.py" "$@"' in script_text


def test_start_dashboard_python_launcher_compiles_and_declares_contract() -> None:
    """The dashboard quick-start launcher must build workers and chain into the Python master bootstrap."""

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
    assert 'start_master.py' in script_text


def test_start_worker_python_launcher_compiles_and_declares_contract() -> None:
    """The Python worker launcher must support both native and Docker worker startup."""

    script_path = REPO_ROOT / "start_worker.py"
    subprocess.run(["python3", "-m", "py_compile", str(script_path)], check=True, capture_output=True, text=True)

    script_text = script_path.read_text(encoding="utf-8")
    assert '"--mode"' in script_text
    assert '"native"' in script_text
    assert '"docker"' in script_text
    assert '"worker.worker_dfs"' in script_text
    assert '"worker" / "requirements.txt"' in script_text
    assert 'wait_for_worker_health' in script_text
    assert 'DFS-lite worker started on http://127.0.0.1:' in script_text


def test_stop_all_python_launcher_compiles_and_declares_contract() -> None:
    """The stop launcher must target repo-managed ports and worker containers."""

    script_path = REPO_ROOT / "stop_all.py"
    subprocess.run(["python3", "-m", "py_compile", str(script_path)], check=True, capture_output=True, text=True)

    script_text = script_path.read_text(encoding="utf-8")
    assert '"--ports"' in script_text
    assert '"--container-prefix"' in script_text
    assert '"hetero-fedlearn-dashboard"' in script_text
    assert '"worker-node"' in script_text
    assert '"-m master.master_dfs"' in script_text
    assert '"-m worker.worker_dfs"' in script_text
    assert '"No repo-managed master or worker services were found."' in script_text


def test_windows_batch_onboarding_contract_is_present() -> None:
    """The Windows batch launcher must build, mount storage, and expose the worker port."""

    script_text = (REPO_ROOT / "start_worker.bat").read_text(encoding="utf-8")

    assert "docker info >nul 2>&1" in script_text
    assert "docker build -t %IMAGE_NAME% -f worker\\Dockerfile_extended worker" in script_text
    assert "-v \"%cd%\\storage:/app/datanode_storage\"" in script_text
    assert "Please start Docker Desktop before running start_worker.bat" in script_text
    assert "DFS-lite worker started on http://127.0.0.1:%HOST_PORT%" in script_text


def test_windows_powershell_onboarding_contract_is_present() -> None:
    """The PowerShell onboarding helper must enforce Tailscale, SSH, worker launch, and health checks."""

    script_text = (REPO_ROOT / "scripts" / "windows" / "onboard_worker.ps1").read_text(encoding="utf-8")

    assert "function Ensure-FirewallRule" in script_text
    assert 'ValidateSet("Docker", "Native")' in script_text
    assert "function Set-ConnectedNetworksPrivate" in script_text
    assert "function Ensure-TailscaleInstalled" in script_text
    assert "function Ensure-TailscaleConnected" in script_text
    assert "function Ensure-OpenSshServer" in script_text
    assert "function Ensure-AuthorizedKey" in script_text
    assert "function Register-WorkerWithMaster" in script_text
    assert "function Wait-WorkerHealth" in script_text
    assert "Invoke-RestMethod -Method Get -Uri $healthUri -TimeoutSec 5" in script_text
    assert 'Join-Path $env:ProgramData "ssh\\administrators_authorized_keys"' in script_text
    assert 'Invoke-Tailscale -Arguments @("up") -AllowFailure' in script_text
    assert 'Invoke-Tailscale -Arguments @("ip", "-4")' in script_text
    assert 'start_worker.py --mode native' in script_text
    assert 'New-ScheduledTaskAction' in script_text
    assert 'Register-ScheduledTask' in script_text
    assert 'Start-ScheduledTask -TaskName $taskName' in script_text
    assert 'HeteroFedLearnWorker-$HostPort' in script_text
    assert 'throw "Run this script from an elevated PowerShell session' in script_text
    assert 'Invoke-Docker -Arguments @("version") | Out-Null' in script_text
    assert '"--restart"' in script_text
    assert '"unless-stopped"' in script_text
    assert 'Write-Host "Worker onboarding completed successfully."' in script_text


def test_readme_documents_tailscale_and_ssh_onboarding_flow() -> None:
    """The README must document the validated cross-machine Tailscale and SSH worker path."""

    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "same Tailscale tailnet" in readme_text
    assert "Install Tailscale on both machines." in readme_text
    assert "ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ''" in readme_text
    assert "python3 start_master.py --allow-unsupported-python --config config_extended.json --host 0.0.0.0 --port 18080 --no-auto-start" in readme_text
    assert "AuthorizedPublicKey" in readme_text
    assert "-UseTailscale" in readme_text
    assert "-EnsureSsh" in readme_text
    assert "administrators_authorized_keys" in readme_text
    assert "persistent elevated scheduled task" in readme_text
