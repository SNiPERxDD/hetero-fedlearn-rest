"""DFS-lite federated learning master with telemetry dashboards."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import requests
from flask import Flask, jsonify, render_template, request
from requests import Session
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)

DEFAULT_UDP_DISCOVERY_PORT = 54321


def get_lan_ip() -> str:
    """Return the best-effort LAN IPv4 address for this host."""

    # Try socket probe to external address (detects routing interface)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe_socket:
        try:
            probe_socket.connect(("8.8.8.8", 80))
            lan_ip = str(probe_socket.getsockname()[0])
            if lan_ip and lan_ip != "127.0.0.1":
                return lan_ip
        except OSError:
            pass

    # Try getaddrinfo on hostname to get all known addresses
    try:
        hostname = socket.gethostname()
        results = socket.getaddrinfo(hostname, None, socket.AF_INET, socket.SOCK_DGRAM)
        for _, _, _, _, sockaddr in results:
            ip = sockaddr[0]
            if ip and ip != "127.0.0.1":
                return ip
    except (OSError, socket.gaierror):
        pass

    # Try alternative external addresses for more robust detection
    for external_host in [("1.1.1.1", 80), ("9.9.9.9", 80)]:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe_socket:
            try:
                probe_socket.connect(external_host)
                ip = str(probe_socket.getsockname()[0])
                if ip and ip != "127.0.0.1":
                    return ip
            except OSError:
                pass

    return "127.0.0.1"


def udp_discovery_listener(runtime_service: "FederatedMasterDFS", discovery_port: int) -> None:
    """Listen for worker UDP beacons and auto-register discovered endpoints."""

    known_endpoints: dict[str, str] = {}
    beacon_log_count = 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as listener_socket:
        listener_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        reuse_port = getattr(socket, "SO_REUSEPORT", None)
        if reuse_port is not None:
            try:
                listener_socket.setsockopt(socket.SOL_SOCKET, reuse_port, 1)
            except OSError:
                LOGGER.debug("SO_REUSEPORT is unavailable for UDP discovery listener.")

        try:
            listener_socket.bind(("0.0.0.0", discovery_port))
        except OSError as error:
            LOGGER.warning(
                "UDP discovery listener could not bind to port %s: %s",
                discovery_port,
                error,
            )
            return

        listener_socket.settimeout(1.0)
        LOGGER.info("UDP worker discovery listener active on 0.0.0.0:%s", discovery_port)
        while True:
            try:
                beacon_bytes, sender_addr = listener_socket.recvfrom(2048)
                beacon_log_count += 1
            except TimeoutError:
                continue
            except OSError as error:
                LOGGER.warning("UDP discovery listener stopped: %s", error)
                return

            try:
                beacon_payload = json.loads(beacon_bytes.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            worker_id = str(beacon_payload.get("worker_id", "")).strip()
            endpoint = str(beacon_payload.get("endpoint", "")).strip().rstrip("/")
            if not worker_id or not endpoint:
                continue
            if not endpoint.startswith(("http://", "https://")):
                continue
            if known_endpoints.get(worker_id) == endpoint:
                # Log every 10th duplicate to avoid spam
                if beacon_log_count % 10 == 0:
                    LOGGER.debug("Duplicate beacon from %s (known endpoint: %s) sent from %s", 
                               worker_id, endpoint, sender_addr[0])
                continue

            try:
                runtime_service.register_worker(worker_id=worker_id, endpoint=endpoint)
                known_endpoints[worker_id] = endpoint
                LOGGER.info("Discovered worker %s at %s via UDP beacon from %s", 
                           worker_id, endpoint, sender_addr[0])
            except (RuntimeError, ValueError) as error:
                LOGGER.debug("Worker beacon ignored for %s (%s) from %s", 
                           worker_id, error, sender_addr[0])


@dataclass(frozen=True)
class WorkerSpec:
    """Configuration for a federated worker endpoint."""

    worker_id: str
    endpoint: str


@dataclass(frozen=True)
class BlockAssignment:
    """A DFS-style data block and its replica placement."""

    block_id: str
    sample_count: int
    replicas: list[str]


@dataclass(frozen=True)
class BlockUpdate:
    """A completed block-local training update."""

    block_id: str
    worker_id: str
    samples_processed: int
    updated_weights: np.ndarray
    updated_intercept: np.ndarray
    local_loss: float


@dataclass
class MasterDashboardState:
    """Thread-safe state shared between the training loop and Flask routes."""

    total_rounds: int = 0
    current_round: int = 0
    training_active: bool = False
    training_completed: bool = False
    training_error: str | None = None
    initial_validation_accuracy: float | None = None
    initial_validation_loss: float | None = None
    latest_validation_accuracy: float | None = None
    latest_validation_loss: float | None = None
    validation_history: list[dict[str, Any]] = field(default_factory=list)
    round_summaries: list[dict[str, Any]] = field(default_factory=list)
    block_map: dict[str, dict[str, Any]] = field(default_factory=dict)
    worker_health: dict[str, dict[str, Any]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def begin_training(
        self,
        total_rounds: int,
        initial_accuracy: float,
        initial_loss: float,
    ) -> None:
        """Reset the state for a fresh training run."""

        with self.lock:
            self.total_rounds = total_rounds
            self.current_round = 0
            self.training_active = True
            self.training_completed = False
            self.training_error = None
            self.initial_validation_accuracy = initial_accuracy
            self.initial_validation_loss = initial_loss
            self.latest_validation_accuracy = initial_accuracy
            self.latest_validation_loss = initial_loss
            self.validation_history = [
                {
                    "round_number": 0,
                    "validation_accuracy": initial_accuracy,
                    "validation_loss": initial_loss,
                }
            ]
            self.round_summaries = []
            self.summary = {}

    def replace_block_map(self, block_map: dict[str, dict[str, Any]]) -> None:
        """Replace the DFS block metadata map."""

        with self.lock:
            self.block_map = copy.deepcopy(block_map)

    def update_block_runtime(
        self,
        block_id: str,
        *,
        last_worker: str | None = None,
        replicas: list[str] | None = None,
        bytes_written: int | None = None,
    ) -> None:
        """Update runtime metadata for a single block."""

        with self.lock:
            block_entry = self.block_map.setdefault(block_id, {"block_id": block_id})
            if last_worker is not None:
                block_entry["last_worker"] = last_worker
            if replicas is not None:
                block_entry["replicas"] = list(replicas)
            if bytes_written is not None:
                block_entry["bytes_written"] = int(bytes_written)

    def replace_worker_health(self, worker_health: dict[str, dict[str, Any]]) -> None:
        """Replace the live worker health snapshot."""

        with self.lock:
            self.worker_health = copy.deepcopy(worker_health)

    def append_round(
        self,
        round_number: int,
        validation_accuracy: float,
        validation_loss: float,
        mean_local_loss: float,
        block_workers: dict[str, str],
    ) -> None:
        """Append a completed round to the telemetry history."""

        with self.lock:
            self.current_round = round_number
            self.latest_validation_accuracy = validation_accuracy
            self.latest_validation_loss = validation_loss
            self.validation_history.append(
                {
                    "round_number": round_number,
                    "validation_accuracy": validation_accuracy,
                    "validation_loss": validation_loss,
                }
            )
            self.round_summaries.append(
                {
                    "round_number": round_number,
                    "validation_accuracy": validation_accuracy,
                    "validation_loss": validation_loss,
                    "mean_local_loss": mean_local_loss,
                    "block_workers": block_workers,
                }
            )

    def complete(self, summary: dict[str, Any]) -> None:
        """Mark the training run as complete."""

        with self.lock:
            self.training_active = False
            self.training_completed = True
            self.summary = copy.deepcopy(summary)

    def fail(self, error_message: str) -> None:
        """Mark the training run as failed."""

        with self.lock:
            self.training_active = False
            self.training_completed = False
            self.training_error = error_message

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot of the current dashboard state."""

        with self.lock:
            return {
                "total_rounds": self.total_rounds,
                "current_round": self.current_round,
                "training_active": self.training_active,
                "training_completed": self.training_completed,
                "training_error": self.training_error,
                "initial_validation_accuracy": self.initial_validation_accuracy,
                "initial_validation_loss": self.initial_validation_loss,
                "latest_validation_accuracy": self.latest_validation_accuracy,
                "latest_validation_loss": self.latest_validation_loss,
                "validation_history": copy.deepcopy(self.validation_history),
                "round_summaries": copy.deepcopy(self.round_summaries),
                "block_map": sorted(copy.deepcopy(self.block_map).values(), key=lambda item: item["block_id"]),
                "worker_health": sorted(
                    copy.deepcopy(self.worker_health).values(),
                    key=lambda item: item["worker_id"],
                ),
                "summary": copy.deepcopy(self.summary),
            }


def configure_logging(level: str = "INFO") -> None:
    """Configure the module logging output."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a JSON configuration file from disk."""

    path = Path(config_path)
    return json.loads(path.read_text(encoding="utf-8"))


def serialise_array(array: np.ndarray) -> list[Any]:
    """Convert a NumPy array into a JSON-safe list representation."""

    if array.ndim == 2 and array.shape[0] == 1:
        return array[0].tolist()
    return array.tolist()


def deserialise_weights(payload: Sequence[Any]) -> np.ndarray:
    """Convert a weight payload into a 2D NumPy array."""

    weights = np.asarray(payload, dtype=float)
    if weights.ndim == 1:
        return weights.reshape(1, -1)
    return weights


def deserialise_intercept(payload: Sequence[Any]) -> np.ndarray:
    """Convert an intercept payload into a 1D NumPy array."""

    return np.asarray(payload, dtype=float).reshape(-1)


def build_classifier(model_config: dict[str, Any]) -> SGDClassifier:
    """Create an SGDClassifier configured for incremental training."""

    return SGDClassifier(
        loss=model_config.get("loss", "log_loss"),
        alpha=float(model_config.get("alpha", 0.0001)),
        eta0=float(model_config.get("eta0", 0.001)),
        learning_rate=model_config.get("learning_rate", "constant"),
        penalty=model_config.get("penalty", "l2"),
        random_state=int(model_config.get("random_state", 42)),
        max_iter=1,
        tol=None,
        warm_start=True,
    )


def seed_classifier(
    model: SGDClassifier,
    classes: np.ndarray,
    n_features: int,
) -> SGDClassifier:
    """Initialise classifier metadata and zero its parameters."""

    seed_features = np.zeros((len(classes), n_features), dtype=float)
    seed_labels = classes.copy()
    model.partial_fit(seed_features, seed_labels, classes=classes)
    model.coef_[:] = 0.0
    model.intercept_[:] = 0.0
    return model


def apply_model_parameters(
    model: SGDClassifier,
    weights: np.ndarray,
    intercept: np.ndarray,
    classes: np.ndarray,
    n_features: int,
) -> SGDClassifier:
    """Inject broadcast model parameters into a seeded classifier."""

    model.coef_ = weights.copy()
    model.intercept_ = intercept.copy()
    model.classes_ = classes.copy()
    model.n_features_in_ = n_features
    return model


def partition_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    partition_count: int,
    random_seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partition a dataset into stratified shards."""

    if partition_count <= 0:
        raise ValueError("partition_count must be positive.")

    rng = np.random.default_rng(random_seed)
    partitions: list[list[int]] = [[] for _ in range(partition_count)]
    for class_label in np.unique(labels):
        class_indices = np.where(labels == class_label)[0]
        rng.shuffle(class_indices)
        for partition_index, class_split in enumerate(np.array_split(class_indices, partition_count)):
            partitions[partition_index].extend(class_split.tolist())

    shards: list[tuple[np.ndarray, np.ndarray]] = []
    for partition_indices in partitions:
        rng.shuffle(partition_indices)
        shard_indices = np.asarray(partition_indices, dtype=int)
        shards.append((features[shard_indices], labels[shard_indices]))
    return shards


class FederatedMasterDFS:
    """Coordinate DFS-lite block placement, telemetry, and FedAvg aggregation."""

    def __init__(
        self,
        config: dict[str, Any],
        session: Session | None = None,
        upload_dir: str | Path | None = None,
    ) -> None:
        """Initialise the master service from a configuration dictionary."""

        self.config = config
        self.session = session or requests.Session()
        self.config_lock = threading.RLock()
        self.dataset_config = config.get("dataset", {})
        self.model_config = config.get("model", {})
        self.training_config = config.get("training", {})
        self.network_config = config.get("network", {})
        self.dashboard_config = config.get("dashboard", {})
        self.upload_dir = Path(upload_dir or (Path(__file__).resolve().parent / "uploads"))
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.workers = self._normalise_workers(config.get("workers", []))

        self.worker_map = {worker.worker_id: worker for worker in self.workers}
        self.random_seed = int(self.training_config.get("random_seed", 42))
        self.scaler = StandardScaler()
        self.state = MasterDashboardState()
        self.training_thread: threading.Thread | None = None
        self.training_thread_lock = threading.RLock()

        self.classes_: np.ndarray | None = None
        self.global_weights: np.ndarray | None = None
        self.global_intercept: np.ndarray | None = None
        self.validation_features: np.ndarray | None = None
        self.validation_labels: np.ndarray | None = None
        self.block_assignments: list[BlockAssignment] = []

    def _normalise_workers(self, workers_payload: Sequence[dict[str, Any]]) -> list[WorkerSpec]:
        """Validate and normalise worker payloads into WorkerSpec instances."""

        normalised_workers: list[WorkerSpec] = []
        seen_worker_ids: set[str] = set()
        seen_endpoints: set[str] = set()
        for raw_worker in workers_payload:
            worker_id = str(raw_worker.get("worker_id", "")).strip()
            endpoint = str(raw_worker.get("endpoint", "")).strip().rstrip("/")
            if not worker_id:
                raise ValueError("Each worker requires a non-empty worker_id.")
            if not endpoint:
                raise ValueError(f"Worker {worker_id!r} requires a non-empty endpoint.")
            if not endpoint.startswith(("http://", "https://")):
                raise ValueError(f"Worker endpoint {endpoint!r} must start with http:// or https://.")
            if worker_id in seen_worker_ids:
                raise ValueError(f"Worker id {worker_id!r} is duplicated.")
            if endpoint in seen_endpoints:
                raise ValueError(f"Worker endpoint {endpoint!r} is duplicated.")
            seen_worker_ids.add(worker_id)
            seen_endpoints.add(endpoint)
            normalised_workers.append(WorkerSpec(worker_id=worker_id, endpoint=endpoint))

        return normalised_workers

    def _sync_worker_runtime(self) -> None:
        """Rebuild worker runtime lookup structures from the current config."""

        self.workers = self._normalise_workers(self.config.get("workers", []))
        self.worker_map = {worker.worker_id: worker for worker in self.workers}

    def training_is_active(self) -> bool:
        """Return whether the training thread is currently active."""

        thread = self.training_thread
        return bool(thread and thread.is_alive())

    def ensure_mutable(self) -> None:
        """Guard configuration changes while training is running."""

        if self.training_is_active() or self.state.training_active:
            raise RuntimeError("Configuration cannot be changed while training is running.")

    def runtime_config_snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot of the editable runtime configuration."""

        with self.config_lock:
            return {
                "dataset": copy.deepcopy(self.dataset_config),
                "model": copy.deepcopy(self.model_config),
                "training": copy.deepcopy(self.training_config),
                "network": copy.deepcopy(self.network_config),
                "dashboard": copy.deepcopy(self.dashboard_config),
                "workers": [
                    {"worker_id": worker.worker_id, "endpoint": worker.endpoint}
                    for worker in self.workers
                ],
            }

    def update_runtime_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Update editable runtime configuration values safely."""

        with self.config_lock:
            self.ensure_mutable()
            dataset_payload = payload.get("dataset", {})
            training_payload = payload.get("training", {})
            network_payload = payload.get("network", {})

            if "source" in dataset_payload:
                source = str(dataset_payload["source"]).strip()
                if source not in {"builtin", "csv"}:
                    raise ValueError("dataset.source must be either 'builtin' or 'csv'.")
                self.dataset_config["source"] = source
            if "name" in dataset_payload:
                dataset_name = str(dataset_payload["name"]).strip()
                if self.dataset_config.get("source", "builtin") == "builtin" and dataset_name != "breast_cancer":
                    raise ValueError("Only the builtin 'breast_cancer' dataset is currently supported.")
                self.dataset_config["name"] = dataset_name
            if "validation_fraction" in dataset_payload:
                validation_fraction = float(dataset_payload["validation_fraction"])
                if not 0.05 <= validation_fraction < 0.5:
                    raise ValueError("dataset.validation_fraction must be between 0.05 and 0.5.")
                self.dataset_config["validation_fraction"] = validation_fraction
            if "label_column" in dataset_payload:
                self.dataset_config["label_column"] = int(dataset_payload["label_column"])

            if "rounds" in training_payload:
                rounds = int(training_payload["rounds"])
                if rounds <= 0:
                    raise ValueError("training.rounds must be positive.")
                self.training_config["rounds"] = rounds
            if "local_epochs" in training_payload:
                local_epochs = int(training_payload["local_epochs"])
                if local_epochs <= 0:
                    raise ValueError("training.local_epochs must be positive.")
                self.training_config["local_epochs"] = local_epochs
            if "replication_factor" in training_payload:
                replication_factor = int(training_payload["replication_factor"])
                if replication_factor <= 0:
                    raise ValueError("training.replication_factor must be positive.")
                self.training_config["replication_factor"] = replication_factor

            if "timeout_seconds" in network_payload:
                timeout_seconds = float(network_payload["timeout_seconds"])
                if timeout_seconds <= 0:
                    raise ValueError("network.timeout_seconds must be positive.")
                self.network_config["timeout_seconds"] = timeout_seconds
            if "discovery_port" in network_payload:
                discovery_port = int(network_payload["discovery_port"])
                if not 1 <= discovery_port <= 65535:
                    raise ValueError("network.discovery_port must be between 1 and 65535.")
                self.network_config["discovery_port"] = discovery_port
            if "enable_udp_discovery" in network_payload:
                self.network_config["enable_udp_discovery"] = bool(network_payload["enable_udp_discovery"])

            self.config["dataset"] = self.dataset_config
            self.config["training"] = self.training_config
            self.config["network"] = self.network_config
            return self.runtime_config_snapshot()

    def register_worker(self, worker_id: str, endpoint: str) -> dict[str, Any]:
        """Register or update a worker endpoint in the runtime configuration."""

        resolved_worker_id = worker_id.strip()
        resolved_endpoint = endpoint.strip().rstrip("/")
        if not resolved_worker_id:
            raise ValueError("worker_id must be non-empty.")
        if not resolved_endpoint:
            raise ValueError("endpoint must be non-empty.")

        with self.config_lock:
            self.ensure_mutable()
            next_workers = self.runtime_config_snapshot()["workers"]
            worker_registered = False
            for worker in next_workers:
                if worker["worker_id"] == resolved_worker_id:
                    worker["endpoint"] = resolved_endpoint
                    worker_registered = True
                    break
            if not worker_registered:
                next_workers.append({"worker_id": resolved_worker_id, "endpoint": resolved_endpoint})

            validated_workers = self._normalise_workers(next_workers)
            self.config["workers"] = next_workers
            self.workers = validated_workers
            self.worker_map = {worker.worker_id: worker for worker in self.workers}
            self.refresh_worker_health()
            return {
                "worker_id": resolved_worker_id,
                "endpoint": resolved_endpoint,
                "worker_count": len(self.workers),
                "config": self.runtime_config_snapshot(),
            }

    def remove_worker(self, worker_id: str) -> dict[str, Any]:
        """Remove a worker from the runtime configuration."""

        with self.config_lock:
            self.ensure_mutable()
            next_workers = [
                {"worker_id": worker.worker_id, "endpoint": worker.endpoint}
                for worker in self.workers
                if worker.worker_id != worker_id
            ]
            if len(next_workers) == len(self.workers):
                raise ValueError(f"Worker {worker_id!r} is not configured.")
            validated_workers = self._normalise_workers(next_workers)
            self.config["workers"] = next_workers
            self.workers = validated_workers
            self.worker_map = {worker.worker_id: worker for worker in self.workers}
            self.refresh_worker_health()
            return {"worker_count": len(self.workers), "config": self.runtime_config_snapshot()}

    def save_uploaded_dataset(
        self,
        *,
        filename: str,
        file_bytes: bytes,
        label_column: int,
    ) -> dict[str, Any]:
        """Persist an uploaded CSV dataset and switch the runtime to CSV mode."""

        with self.config_lock:
            self.ensure_mutable()
            if not filename.lower().endswith(".csv"):
                raise ValueError("Uploaded dataset must be a CSV file.")
            safe_stem = "".join(character for character in Path(filename).stem if character.isalnum() or character in {"-", "_"})
            safe_stem = safe_stem or "dataset"
            destination = self.upload_dir / f"{safe_stem}_{uuid.uuid4().hex[:8]}.csv"
            destination.write_bytes(file_bytes)

            try:
                raw_data = np.genfromtxt(destination, delimiter=",", skip_header=1)
                raw_data = np.atleast_2d(raw_data)
            except Exception as error:
                destination.unlink(missing_ok=True)
                raise ValueError("Uploaded CSV dataset could not be parsed.") from error

            if raw_data.shape[1] < 2:
                destination.unlink(missing_ok=True)
                raise ValueError("Uploaded CSV dataset must contain at least one feature column and one label column.")
            if not (-raw_data.shape[1] <= label_column < raw_data.shape[1]):
                destination.unlink(missing_ok=True)
                raise ValueError("dataset.label_column is out of range for the uploaded CSV dataset.")

            self.dataset_config["source"] = "csv"
            self.dataset_config["name"] = destination.stem
            self.dataset_config["csv_path"] = str(destination)
            self.dataset_config["label_column"] = int(label_column)
            self.config["dataset"] = self.dataset_config
            return {
                "dataset": self.runtime_config_snapshot()["dataset"],
                "rows": int(raw_data.shape[0]),
                "columns": int(raw_data.shape[1]),
            }

    def load_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """Load the configured dataset into memory."""

        source = self.dataset_config.get("source", "builtin")
        if source == "builtin":
            dataset_name = self.dataset_config.get("name", "breast_cancer")
            if dataset_name != "breast_cancer":
                raise ValueError(f"Unsupported builtin dataset: {dataset_name}")
            return load_breast_cancer(return_X_y=True)

        if source == "csv":
            csv_path = self.dataset_config.get("csv_path")
            if not csv_path:
                raise ValueError("dataset.csv_path is required when source is 'csv'.")
            raw_data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
            label_column = int(self.dataset_config.get("label_column", -1))
            features = np.delete(raw_data, label_column, axis=1)
            labels = raw_data[:, label_column]
            return features, labels

        raise ValueError(f"Unsupported dataset source: {source}")

    def prepare_blocks(self) -> tuple[list[tuple[BlockAssignment, np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
        """Split, scale, and convert the training dataset into DFS-like blocks."""

        if not self.workers:
            raise RuntimeError(
                "No workers are registered. Start at least one worker and wait for discovery before training."
            )

        features, labels = self.load_dataset()
        validation_fraction = float(self.dataset_config.get("validation_fraction", 0.2))
        train_features, validation_features, train_labels, validation_labels = train_test_split(
            features,
            labels,
            test_size=validation_fraction,
            random_state=self.random_seed,
            stratify=labels,
        )

        train_features = self.scaler.fit_transform(train_features)
        validation_features = self.scaler.transform(validation_features)
        self.classes_ = np.unique(train_labels)
        self.validation_features = validation_features
        self.validation_labels = validation_labels

        seeded_model = seed_classifier(
            build_classifier(self.model_config),
            classes=self.classes_,
            n_features=train_features.shape[1],
        )
        self.global_weights = seeded_model.coef_.copy()
        self.global_intercept = seeded_model.intercept_.copy()

        initial_accuracy, initial_loss = self.evaluate_global_model()
        total_rounds = int(self.training_config.get("rounds", 10))
        self.state.begin_training(
            total_rounds=total_rounds,
            initial_accuracy=initial_accuracy,
            initial_loss=initial_loss,
        )

        shards = partition_dataset(
            train_features,
            train_labels,
            partition_count=len(self.workers),
            random_seed=self.random_seed,
        )
        replication_factor = max(1, int(self.training_config.get("replication_factor", 1)))
        replication_factor = min(replication_factor, len(self.workers))

        block_payloads: list[tuple[BlockAssignment, np.ndarray, np.ndarray]] = []
        block_map: dict[str, dict[str, Any]] = {}
        for block_index, (local_features, local_labels) in enumerate(shards):
            block_id = f"blk_{uuid.uuid4().hex[:8]}"
            replica_ids = [
                self.workers[(block_index + replica_offset) % len(self.workers)].worker_id
                for replica_offset in range(replication_factor)
            ]
            assignment = BlockAssignment(
                block_id=block_id,
                sample_count=int(local_features.shape[0]),
                replicas=replica_ids,
            )
            block_payloads.append((assignment, local_features, local_labels))
            block_map[block_id] = {
                "block_id": block_id,
                "sample_count": int(local_features.shape[0]),
                "replicas": replica_ids,
                "bytes_written": 0,
                "last_worker": None,
            }

        self.state.replace_block_map(block_map)
        return block_payloads, train_features, train_labels

    def evaluate_global_model(self) -> tuple[float, float]:
        """Evaluate the current global model on the validation set."""

        if self.classes_ is None or self.validation_features is None or self.validation_labels is None:
            raise RuntimeError("Validation data is not ready.")
        if self.global_weights is None or self.global_intercept is None:
            raise RuntimeError("Global parameters are not initialised.")

        model = seed_classifier(
            build_classifier(self.model_config),
            classes=self.classes_,
            n_features=self.validation_features.shape[1],
        )
        apply_model_parameters(
            model,
            weights=self.global_weights,
            intercept=self.global_intercept,
            classes=self.classes_,
            n_features=self.validation_features.shape[1],
        )
        predictions = model.predict(self.validation_features)
        probabilities = model.predict_proba(self.validation_features)
        return (
            float(accuracy_score(self.validation_labels, predictions)),
            float(log_loss(self.validation_labels, probabilities, labels=self.classes_)),
        )

    def request_json(
        self,
        method: str,
        worker: WorkerSpec,
        route: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
        retry_attempts: int | None = None,
    ) -> dict[str, Any]:
        """Send an HTTP request to a worker and parse its JSON response."""

        request_timeout = float(timeout_seconds or self.network_config.get("timeout_seconds", 120))
        request_retries = int(retry_attempts or self.network_config.get("retry_attempts", 3))
        retry_backoff_seconds = float(self.network_config.get("retry_backoff_seconds", 1.0))
        url = f"{worker.endpoint}{route}"
        last_exception: Exception | None = None

        for attempt in range(1, request_retries + 1):
            try:
                if method.upper() == "GET":
                    response = self.session.get(url, timeout=request_timeout)
                else:
                    response = self.session.post(url, json=payload, timeout=request_timeout)
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as error:
                last_exception = error
                LOGGER.warning(
                    "Worker request failed | worker=%s route=%s attempt=%s/%s",
                    worker.worker_id,
                    route,
                    attempt,
                    request_retries,
                )
                if attempt < request_retries:
                    time.sleep(retry_backoff_seconds)

        raise RuntimeError(
            f"Worker {worker.worker_id} failed to respond successfully on {route} "
            f"after {request_retries} attempts."
        ) from last_exception

    def refresh_worker_health(self) -> dict[str, dict[str, Any]]:
        """Poll all workers for a live health snapshot."""

        health_timeout_seconds = float(self.network_config.get("health_timeout_seconds", 3.0))
        worker_health: dict[str, dict[str, Any]] = {}
        for worker in self.workers:
            try:
                payload = self.request_json(
                    "GET",
                    worker,
                    "/api/status",
                    timeout_seconds=health_timeout_seconds,
                    retry_attempts=1,
                )
                worker_health[worker.worker_id] = {
                    "worker_id": worker.worker_id,
                    "endpoint": worker.endpoint,
                    "healthy": True,
                    "ready": bool(payload.get("ready", False)),
                    "storage_bytes": int(payload.get("storage_bytes", 0)),
                    "block_count": int(payload.get("block_count", 0)),
                    "last_local_loss": payload.get("last_local_loss"),
                    "last_samples_processed": int(payload.get("last_samples_processed", 0)),
                }
            except RuntimeError:
                worker_health[worker.worker_id] = {
                    "worker_id": worker.worker_id,
                    "endpoint": worker.endpoint,
                    "healthy": False,
                    "ready": False,
                    "storage_bytes": 0,
                    "block_count": 0,
                    "last_local_loss": None,
                    "last_samples_processed": 0,
                }

        self.state.replace_worker_health(worker_health)
        return worker_health

    def initialise_blocks(
        self,
        block_payloads: list[tuple[BlockAssignment, np.ndarray, np.ndarray]],
    ) -> list[BlockAssignment]:
        """Commit every block to one or more workers and update the metadata map."""

        if self.classes_ is None:
            raise RuntimeError("Classes are not initialised.")

        committed_assignments: list[BlockAssignment] = []
        for assignment, local_features, local_labels in block_payloads:
            committed_replicas: list[str] = []
            max_bytes_written = 0
            for worker_id in assignment.replicas:
                worker = self.worker_map[worker_id]
                payload = {
                    "block_id": assignment.block_id,
                    "worker_id": worker.worker_id,
                    "features": local_features.tolist(),
                    "labels": local_labels.tolist(),
                    "classes": self.classes_.tolist(),
                    "model_config": self.model_config,
                }
                try:
                    response = self.request_json("POST", worker, "/initialize", payload)
                except RuntimeError:
                    LOGGER.error("Failed to commit block %s to worker %s.", assignment.block_id, worker.worker_id)
                    continue

                committed_replicas.append(worker.worker_id)
                max_bytes_written = max(max_bytes_written, int(response.get("bytes_written", 0)))

            if not committed_replicas:
                raise RuntimeError(f"Block {assignment.block_id} could not be committed to any worker.")

            committed_assignment = BlockAssignment(
                block_id=assignment.block_id,
                sample_count=assignment.sample_count,
                replicas=committed_replicas,
            )
            committed_assignments.append(committed_assignment)
            self.state.update_block_runtime(
                assignment.block_id,
                replicas=committed_replicas,
                bytes_written=max_bytes_written,
            )

        self.block_assignments = committed_assignments
        self.refresh_worker_health()
        return committed_assignments

    def train_block(
        self,
        assignment: BlockAssignment,
        round_number: int,
        local_epochs: int,
    ) -> BlockUpdate:
        """Train a single block on the first healthy replica available."""

        if self.global_weights is None or self.global_intercept is None:
            raise RuntimeError("Global parameters must be initialised before training.")

        for worker_id in assignment.replicas:
            worker = self.worker_map[worker_id]
            payload = {
                "block_id": assignment.block_id,
                "round_number": round_number,
                "global_weights": serialise_array(self.global_weights),
                "global_intercept": serialise_array(self.global_intercept),
                "local_epochs": local_epochs,
            }
            try:
                response = self.request_json("POST", worker, "/train_round", payload)
            except RuntimeError:
                LOGGER.error(
                    "Block %s failed on worker %s during round %s.",
                    assignment.block_id,
                    worker_id,
                    round_number,
                )
                continue

            self.state.update_block_runtime(assignment.block_id, last_worker=worker_id)
            return BlockUpdate(
                block_id=assignment.block_id,
                worker_id=response["worker_id"],
                samples_processed=int(response["samples_processed"]),
                updated_weights=deserialise_weights(response["updated_weights"]),
                updated_intercept=deserialise_intercept(response["updated_intercept"]),
                local_loss=float(response["local_loss"]),
            )

        raise RuntimeError(f"No healthy replica completed block {assignment.block_id} during round {round_number}.")

    def aggregate_updates(self, updates: Sequence[BlockUpdate]) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate block updates via weighted FedAvg."""

        total_samples = sum(update.samples_processed for update in updates)
        if total_samples <= 0:
            raise RuntimeError("Block aggregation received zero samples.")

        aggregated_weights = sum(
            (update.samples_processed / total_samples) * update.updated_weights for update in updates
        )
        aggregated_intercept = sum(
            (update.samples_processed / total_samples) * update.updated_intercept for update in updates
        )
        return aggregated_weights, aggregated_intercept

    def run_training(self) -> dict[str, Any]:
        """Run the DFS-lite federated training loop inside the background thread."""

        try:
            block_payloads, train_features, train_labels = self.prepare_blocks()
            self.initialise_blocks(block_payloads)

            rounds = int(self.training_config.get("rounds", 10))
            local_epochs = int(self.training_config.get("local_epochs", 5))
            round_history: list[dict[str, Any]] = []
            for round_number in range(1, rounds + 1):
                self.refresh_worker_health()
                block_updates: list[BlockUpdate] = []
                block_workers: dict[str, str] = {}
                for assignment in self.block_assignments:
                    block_update = self.train_block(
                        assignment=assignment,
                        round_number=round_number,
                        local_epochs=local_epochs,
                    )
                    block_updates.append(block_update)
                    block_workers[assignment.block_id] = block_update.worker_id

                self.global_weights, self.global_intercept = self.aggregate_updates(block_updates)
                validation_accuracy, validation_loss = self.evaluate_global_model()
                mean_local_loss = float(np.mean([update.local_loss for update in block_updates]))
                round_summary = {
                    "round_number": round_number,
                    "validation_accuracy": validation_accuracy,
                    "validation_loss": validation_loss,
                    "mean_local_loss": mean_local_loss,
                    "block_workers": block_workers,
                }
                round_history.append(round_summary)
                self.state.append_round(
                    round_number=round_number,
                    validation_accuracy=validation_accuracy,
                    validation_loss=validation_loss,
                    mean_local_loss=mean_local_loss,
                    block_workers=block_workers,
                )
                LOGGER.info(
                    "Round %s complete | blocks=%s | val_acc=%.4f | val_loss=%.4f",
                    round_number,
                    len(block_updates),
                    validation_accuracy,
                    validation_loss,
                )

            final_accuracy, final_loss = self.evaluate_global_model()
            summary = {
                "dataset": {
                    "source": self.dataset_config.get("source", "builtin"),
                    "name": self.dataset_config.get("name", "breast_cancer"),
                    "train_samples": int(train_features.shape[0]),
                    "validation_samples": int(self.validation_features.shape[0]),
                    "feature_count": int(train_features.shape[1]),
                    "classes": self.classes_.tolist() if self.classes_ is not None else [],
                },
                "training": {
                    "rounds": rounds,
                    "local_epochs": local_epochs,
                    "random_seed": self.random_seed,
                    "replication_factor": int(self.training_config.get("replication_factor", 1)),
                    "train_label_cardinality": int(len(np.unique(train_labels))),
                },
                "block_map": self.state.snapshot()["block_map"],
                "worker_health": self.state.snapshot()["worker_health"],
                "initial_validation_accuracy": self.state.initial_validation_accuracy,
                "initial_validation_loss": self.state.initial_validation_loss,
                "final_validation_accuracy": final_accuracy,
                "final_validation_loss": final_loss,
                "history": round_history,
            }
            self.state.complete(summary)
            return summary
        except Exception as error:
            LOGGER.exception("DFS-lite training failed.")
            self.state.fail(str(error))
            raise

    def start_training_thread(self) -> bool:
        """Start the background training thread if it is not already running."""

        with self.training_thread_lock:
            if self.training_thread and self.training_thread.is_alive():
                return False
            if not self.workers:
                self.state.fail(
                    "No workers are currently registered. Start workers and let UDP discovery register them first."
                )
                return False
            self.training_thread = threading.Thread(target=self.run_training, daemon=True)
            self.training_thread.start()
            return True

    def wait_for_training(self, timeout: float | None = None) -> None:
        """Join the background training thread."""

        thread = self.training_thread
        if thread is not None:
            thread.join(timeout=timeout)


def create_app(
    config_path: str | Path,
    *,
    autostart: bool = False,
    service: FederatedMasterDFS | None = None,
    enable_udp_discovery: bool | None = None,
) -> Flask:
    """Create and configure the DFS-lite master Flask application."""

    app = Flask(__name__, template_folder="templates")
    runtime_service = service or FederatedMasterDFS(
        load_config(config_path),
        upload_dir=Path(config_path).resolve().parent / "uploads",
    )
    app.config["MASTER_SERVICE"] = runtime_service
    app.config["CONFIG_PATH"] = str(config_path)
    app.config["MASTER_LAN_IP"] = get_lan_ip()

    discovery_port = int(runtime_service.network_config.get("discovery_port", DEFAULT_UDP_DISCOVERY_PORT))
    if enable_udp_discovery is None:
        discovery_enabled = bool(runtime_service.network_config.get("enable_udp_discovery", True))
        if "PYTEST_CURRENT_TEST" in os.environ:
            discovery_enabled = False
    else:
        discovery_enabled = bool(enable_udp_discovery)

    app.config["UDP_DISCOVERY_PORT"] = discovery_port
    app.config["UDP_DISCOVERY_ENABLED"] = discovery_enabled
    if discovery_enabled:
        discovery_thread = threading.Thread(
            target=udp_discovery_listener,
            kwargs={"runtime_service": runtime_service, "discovery_port": discovery_port},
            daemon=True,
            name=f"master-udp-discovery-{discovery_port}",
        )
        discovery_thread.start()

    if autostart:
        runtime_service.start_training_thread()

    @app.get("/")
    def index() -> str:
        """Render the master telemetry dashboard."""

        host_header = request.host
        resolved_port = host_header.split(":")[-1] if ":" in host_header else "18080"

        return render_template(
            "index_dfs.html",
            worker_count=len(runtime_service.workers),
            poll_interval_ms=int(runtime_service.dashboard_config.get("poll_interval_ms", 1500)),
            total_rounds=int(runtime_service.training_config.get("rounds", 10)),
            master_lan_ip=app.config["MASTER_LAN_IP"],
            master_port=resolved_port,
            udp_discovery_port=app.config["UDP_DISCOVERY_PORT"],
            udp_discovery_enabled=app.config["UDP_DISCOVERY_ENABLED"],
        )

    @app.get("/api/status")
    @app.get("/api/cluster_status")
    def status() -> Any:
        """Return the current cluster state for dashboard polling."""

        runtime_service.refresh_worker_health()
        status_payload = runtime_service.state.snapshot()
        status_payload["discovery"] = {
            "enabled": bool(app.config["UDP_DISCOVERY_ENABLED"]),
            "port": int(app.config["UDP_DISCOVERY_PORT"]),
            "master_lan_ip": str(app.config["MASTER_LAN_IP"]),
        }
        return jsonify(status_payload)

    @app.post("/api/start_training")
    def start_training() -> Any:
        """Start the background training thread on demand."""

        started = runtime_service.start_training_thread()
        runtime_service.refresh_worker_health()
        return jsonify({"started": started, "state": runtime_service.state.snapshot()})

    @app.get("/api/config")
    def config_snapshot() -> Any:
        """Return the editable runtime configuration."""

        return jsonify(runtime_service.runtime_config_snapshot())

    @app.post("/api/config")
    def update_config() -> Any:
        """Update editable training or dataset configuration."""

        payload = request.get_json(silent=True) or {}
        try:
            config_snapshot = runtime_service.update_runtime_config(payload)
        except (RuntimeError, ValueError) as error:
            status_code = 409 if isinstance(error, RuntimeError) else 400
            return jsonify({"error": str(error)}), status_code
        return jsonify({"config": config_snapshot}), 200

    @app.post("/api/workers/register")
    def register_worker() -> Any:
        """Register or update a worker endpoint from the UI or a worker node."""

        payload = request.get_json(silent=True) or {}
        worker_id = str(payload.get("worker_id", "")).strip()
        endpoint = str(payload.get("endpoint", "")).strip()
        if not worker_id or not endpoint:
            return jsonify({"error": "worker_id and endpoint are required."}), 400

        try:
            response_payload = runtime_service.register_worker(worker_id=worker_id, endpoint=endpoint)
        except (RuntimeError, ValueError) as error:
            status_code = 409 if isinstance(error, RuntimeError) else 400
            return jsonify({"error": str(error)}), status_code
        return jsonify(response_payload), 200

    @app.post("/api/workers/remove")
    def remove_worker() -> Any:
        """Remove a worker endpoint from the runtime configuration."""

        payload = request.get_json(silent=True) or {}
        worker_id = str(payload.get("worker_id", "")).strip()
        if not worker_id:
            return jsonify({"error": "worker_id is required."}), 400

        try:
            response_payload = runtime_service.remove_worker(worker_id=worker_id)
        except (RuntimeError, ValueError) as error:
            status_code = 409 if isinstance(error, RuntimeError) else 400
            return jsonify({"error": str(error)}), status_code
        return jsonify(response_payload), 200

    @app.post("/api/dataset/upload")
    def upload_dataset() -> Any:
        """Persist an uploaded CSV dataset and switch the runtime to CSV mode."""

        uploaded_file = request.files.get("dataset")
        if uploaded_file is None or not uploaded_file.filename:
            return jsonify({"error": "A CSV dataset file is required."}), 400

        try:
            label_column = int(request.form.get("label_column", "-1"))
        except ValueError:
            return jsonify({"error": "label_column must be an integer."}), 400

        try:
            response_payload = runtime_service.save_uploaded_dataset(
                filename=uploaded_file.filename,
                file_bytes=uploaded_file.read(),
                label_column=label_column,
            )
        except (RuntimeError, ValueError) as error:
            status_code = 409 if isinstance(error, RuntimeError) else 400
            return jsonify({"error": str(error)}), status_code
        return jsonify(response_payload), 200

    return app


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the DFS-lite master service."""

    parser = argparse.ArgumentParser(description="DFS-lite federated learning master service.")
    parser.add_argument("--config", default="config_extended.json", help="Path to the JSON configuration file.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind.")
    parser.add_argument("--port", default=18080, type=int, help="Port to bind.")
    parser.add_argument("--log-level", default="INFO", help="Logging level for the master process.")
    parser.add_argument(
        "--auto-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to start the background training thread immediately.",
    )
    return parser


def main() -> int:
    """Run the DFS-lite master CLI entry point."""

    args = build_argument_parser().parse_args()
    configure_logging(args.log_level)
    app = create_app(config_path=args.config, autostart=args.auto_start)
    try:
        from waitress import serve
    except ImportError:
        LOGGER.warning("Waitress is unavailable. Falling back to Flask's built-in server.")
        app.run(host=args.host, port=args.port, threaded=True)
        return 0

    serve(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
