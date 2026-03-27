# Changelog

## 2026-03-27 13:48:53 IST
- Bootstrapped the federated learning runtime with a REST master, stateful worker service, stratified shard initialisation, JSON-safe parameter transport, retry-aware round execution, and weighted FedAvg aggregation. Files: `config.json`, `master/__init__.py`, `master/master.py`, `master/requirements.txt`, `worker/__init__.py`, `worker/worker.py`, `worker/requirements.txt`, `worker/Dockerfile`
- Added deterministic local verification for worker API safety and full two-worker HTTP training convergence, plus pytest configuration for repository-local execution. Files: `pyproject.toml`, `tests/test_federated_workflow.py`
- Updated repository hygiene and requirements documentation to reflect the committed implementation layout and validation path. Files: `.gitignore`, `PRD.md`, `master/data/.gitkeep`, `CHANGELOG.md`
