# Changelog

## 2026-03-27 19:06:11 IST
- Added a standard repository `README.md` covering architecture, setup, local simulation, container execution, Windows worker onboarding, configuration, API usage, and verification flow. Files: `README.md`
- Updated the PRD directory tree and implementation snapshot to include the standard project README. Files: `PRD.md`, `CHANGELOG.md`

## 2026-03-27 19:02:08 IST
- Hardened the worker container for deployment by enabling unbuffered logging and a native Docker health check, then validated the image build on `python:3.14-slim` with Flask 3.1.3, scikit-learn 1.8.0, and Waitress 3.0.2. Files: `worker/Dockerfile`
- Added a Windows onboarding script that can set the active network to `Private`, create or refresh the inbound firewall rule, optionally build or pull the worker image, launch the worker container, and verify `/health` locally. Files: `scripts/windows/onboard_worker.ps1`
- Updated the PRD implementation snapshot and directory tree to reflect the Windows bootstrap path and the hardened container health model. Files: `PRD.md`, `CHANGELOG.md`

## 2026-03-27 13:48:53 IST
- Bootstrapped the federated learning runtime with a REST master, stateful worker service, stratified shard initialisation, JSON-safe parameter transport, retry-aware round execution, and weighted FedAvg aggregation. Files: `config.json`, `master/__init__.py`, `master/master.py`, `master/requirements.txt`, `worker/__init__.py`, `worker/worker.py`, `worker/requirements.txt`, `worker/Dockerfile`
- Added deterministic local verification for worker API safety and full two-worker HTTP training convergence, plus pytest configuration for repository-local execution. Files: `pyproject.toml`, `tests/test_federated_workflow.py`
- Updated repository hygiene and requirements documentation to reflect the committed implementation layout and validation path. Files: `.gitignore`, `PRD.md`, `master/data/.gitkeep`, `CHANGELOG.md`
