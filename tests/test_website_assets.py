"""Validation tests for the React website package."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_website_package_declares_buildable_react_app() -> None:
    """The website package must declare a Vite React build pipeline."""

    package_path = REPO_ROOT / "website" / "package.json"
    package = json.loads(package_path.read_text(encoding="utf-8"))

    assert package["name"] == "hetero-fedlearn-website"
    assert package["scripts"]["build"] == "tsc -b && vite build"
    assert "react" in package["dependencies"]
    assert "vite" in package["devDependencies"]


def test_website_copy_references_real_project_entry_points() -> None:
    """The site content must reference the real runtime commands and launchers."""

    app_text = (REPO_ROOT / "website" / "src" / "App.tsx").read_text(encoding="utf-8")

    assert "Hetero FedLearn REST" in app_text
    assert "python3 start_dashboard.py --allow-unsupported-python --master-port 18080" in app_text
    assert "start_worker.bat" in app_text
    assert "scripts/windows/onboard_worker.ps1" in app_text
    assert "20 passed" in app_text
