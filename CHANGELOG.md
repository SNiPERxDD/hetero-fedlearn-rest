# Changelog

## 2026-04-04 00:48:14 IST
- Expanded `.gitignore` to cover broader Python cache artifacts, virtual environments, logs, local archives, editor metadata, notebook checkpoints, Node build outputs, and repo-managed runtime storage so tracked source stays clean while local launcher or build noise remains untracked. Files: `.gitignore`, `CHANGELOG.md`

## 2026-04-04 00:39:48 IST
- Refined the README presentation to match the current repository state with cleaner hierarchy, compact operator tables, restrained icon markers, and clearer scan paths for launch, cleanup, deployment, and verification. Files: `README.md`, `CHANGELOG.md`

## 2026-04-04 00:36:43 IST
- Moved the PRD source files out of the repo root into `archive/` and updated the README repository layout so the archive location is reflected in the operator-facing documentation. Files: `archive/PRD.md`, `archive/PRD_Extended.md`, `README.md`, `CHANGELOG.md`

## 2026-04-04 00:05:54 IST
- Added `stop_all.py`, a repo-scoped Python cleanup utility that terminates the managed DFS-lite master or native workers on known ports and removes the known worker containers so demo ports can be reclaimed quickly. Files: `stop_all.py`, `CHANGELOG.md`
- Added stop-script coverage and updated the operator docs or website copy to document the cleanup entry point and reflect the expanded `23 passed` suite state. Files: `tests/test_onboarding_scripts.py`, `tests/test_website_assets.py`, `README.md`, `PRD.md`, `PRD_Extended.md`, `website/src/App.tsx`, `CHANGELOG.md`

## 2026-04-04 00:03:00 IST
- Added Python-first cross-platform launchers for the DFS-lite control plane: `start_master.py` for the master, `start_worker.py` for native or Docker worker startup, and updated `start_dashboard.py` or `start_master.sh` so the dashboard bootstrap now chains into the Python master launcher while the shell script remains a compatibility wrapper. Files: `start_master.py`, `start_worker.py`, `start_dashboard.py`, `start_master.sh`, `CHANGELOG.md`
- Expanded onboarding verification and operator documentation for role-swapped deployments, updated `.gitignore` for launcher-generated environments or storage, and refreshed the website copy to reflect the new Python entry points and the `22 passed` suite state. Files: `.gitignore`, `tests/test_onboarding_scripts.py`, `tests/test_website_assets.py`, `README.md`, `PRD.md`, `PRD_Extended.md`, `website/src/App.tsx`, `CHANGELOG.md`

## 2026-04-03 23:38:25 IST
- Strengthened dataset and UI verification by switching the CSV-upload workflow tests to a genuinely different synthetic classification dataset and adding a worker-dashboard failure-path browser test for unsuccessful master registration. Files: `tests/test_dashboard_ui.py`, `tests/test_dfs_lite_workflow.py`, `CHANGELOG.md`
- Refreshed the website and README verification copy to reflect the expanded `20 passed` suite state. Files: `website/src/App.tsx`, `tests/test_website_assets.py`, `README.md`, `CHANGELOG.md`

## 2026-04-03 23:38:25 IST
- Clarified the operator documentation so the README and PRDs explicitly distinguish the separate React website on port `4173` from the real DFS-lite Flask control plane on port `18080`, and corrected the remaining extended-PRD references that still mentioned `8080`. Files: `README.md`, `PRD.md`, `PRD_Extended.md`, `CHANGELOG.md`

## 2026-04-03 22:39:08 IST
- Changed the DFS-lite master dashboard default port from `8080` to `18080` across the bootstrap scripts, manual CLI path, worker UI placeholder, and website or README copy to avoid the verified local Appsmith collision on `8080`. Files: `start_dashboard.py`, `start_master.sh`, `master/master_dfs.py`, `worker/templates/index_dfs.html`, `website/src/App.tsx`, `README.md`, `tests/test_onboarding_scripts.py`, `CHANGELOG.md`

## 2026-04-03 21:48:15 IST
- Added a browser-driven end-to-end control-plane test that registers a worker from the worker UI, uploads a CSV dataset from the master UI, updates training settings, and starts training entirely through the dashboards. Files: `tests/test_dashboard_ui.py`, `CHANGELOG.md`
- Refreshed the website and README verification copy to reflect the stronger `19 passed` suite state after the new browser control-plane coverage. Files: `website/src/App.tsx`, `tests/test_website_assets.py`, `README.md`, `CHANGELOG.md`

## 2026-04-03 21:43:02 IST
- Extended the DFS-lite master into a browser-driven control plane with runtime training config updates, worker registration or removal, CSV dataset upload, and safe mutation guards while training is active. Files: `master/master_dfs.py`, `master/templates/index_dfs.html`, `CHANGELOG.md`
- Extended the DFS-lite worker dashboard so a worker can register itself with a master endpoint and expose connection state in its runtime telemetry, and added the required HTTP dependency for the worker service. Files: `worker/worker_dfs.py`, `worker/templates/index_dfs.html`, `worker/requirements.txt`, `CHANGELOG.md`
- Added integration and browser coverage for the new control-plane workflows and refreshed repository or website copy to reflect the verified `18 passed` state. Files: `tests/test_dfs_lite_workflow.py`, `tests/test_dashboard_ui.py`, `tests/test_website_assets.py`, `website/src/App.tsx`, `README.md`, `PRD.md`, `PRD_Extended.md`, `CHANGELOG.md`

## 2026-04-03 21:16:51 IST
- Added a clean React website package that replaces the unusable AI Studio browser scaffold with a project-specific frontend covering architecture, DFS-lite telemetry, quick-start commands, onboarding paths, and verified runtime evidence. Files: `website/package.json`, `website/package-lock.json`, `website/index.html`, `website/src/App.tsx`, `website/src/main.tsx`, `website/src/styles.css`, `website/tsconfig.json`, `website/vite.config.ts`, `CHANGELOG.md`
- Added repository integration for the website package, including build-artifact ignores, website validation tests, and README or PRD updates describing how to run and build the frontend. Files: `.gitignore`, `tests/test_website_assets.py`, `README.md`, `PRD.md`, `PRD_Extended.md`, `CHANGELOG.md`

## 2026-04-03 21:04:56 IST
- Added a Python quick-start launcher for the DFS-lite dashboards that builds or reuses local worker containers, preflights localhost ports, waits for worker health, forwards master config and port overrides, and cleans up worker containers on exit. Files: `start_dashboard.py`, `start_master.sh`, `CHANGELOG.md`
- Updated repository documentation and script verification to use the Python quick-start path, including the new `--master-port` override and the launcher contract checks in the automated test suite. Files: `README.md`, `PRD.md`, `PRD_Extended.md`, `tests/test_onboarding_scripts.py`, `CHANGELOG.md`

## 2026-04-03 20:52:00 IST
- Added browser-level DFS-lite dashboard verification with a live Chromium flow that starts background training, validates worker telemetry, checks disk-backed block persistence, and exercises the mobile responsive layout. Files: `tests/test_dashboard_ui.py`, `CHANGELOG.md`
- Added onboarding-script verification covering Bash syntax plus Windows launcher and PowerShell contract checks for Docker availability, storage mounting, firewall setup, and health probing. Files: `tests/test_onboarding_scripts.py`, `CHANGELOG.md`

## 2026-04-03 20:10:40 IST
- Hardened the verification suite with stricter negative-path coverage, including baseline weight-dimension rejection, DFS-lite disk-read validation, replica failover, and idempotent dashboard start control checks. Files: `tests/test_federated_workflow.py`, `tests/test_dfs_lite_workflow.py`, `CHANGELOG.md`

## 2026-04-03 12:55:22 IST
- Added the DFS-lite extension as preserved-copy variants with a NameNode-style master service, a DataNode-style worker service, block metadata tracking, disk-backed block commits, and background-thread orchestration for dashboards. Files: `config_extended.json`, `master/master_dfs.py`, `master/requirements_extended.txt`, `worker/worker_dfs.py`
- Added telemetry dashboards, bootstrap scripts, and DFS-lite packaging assets for live demos and OS onboarding. Files: `master/templates/index_dfs.html`, `worker/templates/index_dfs.html`, `worker/Dockerfile_extended`, `worker/datanode_storage/.gitkeep`, `start_master.sh`, `start_worker.bat`, `scripts/windows/onboard_worker.ps1`
- Added DFS-lite verification coverage and updated project documentation to distinguish the preserved v1 baseline from the v1.1 extension. Files: `tests/test_dfs_lite_workflow.py`, `README.md`, `PRD.md`, `PRD_Extended.md`, `CHANGELOG.md`

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
