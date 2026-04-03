"""DFS-lite federated learning master with telemetry dashboards."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import requests
from flask import Flask, jsonify, render_template
from requests import Session
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)


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
    ) -> None:
        """Initialise the master service from a configuration dictionary."""

        self.config = config
        self.session = session or requests.Session()
        self.dataset_config = config.get("dataset", {})
        self.model_config = config.get("model", {})
        self.training_config = config.get("training", {})
        self.network_config = config.get("network", {})
        self.dashboard_config = config.get("dashboard", {})
        self.workers = [
            WorkerSpec(worker_id=item["worker_id"], endpoint=item["endpoint"].rstrip("/"))
            for item in config.get("workers", [])
        ]
        if not self.workers:
            raise ValueError("Configuration must define at least one worker.")

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
) -> Flask:
    """Create and configure the DFS-lite master Flask application."""

    app = Flask(__name__, template_folder="templates")
    runtime_service = service or FederatedMasterDFS(load_config(config_path))
    app.config["MASTER_SERVICE"] = runtime_service
    app.config["CONFIG_PATH"] = str(config_path)

    if autostart:
        runtime_service.start_training_thread()

    @app.get("/")
    def index() -> str:
        """Render the master telemetry dashboard."""

        return render_template(
            "index_dfs.html",
            worker_count=len(runtime_service.workers),
            poll_interval_ms=int(runtime_service.dashboard_config.get("poll_interval_ms", 1500)),
            total_rounds=int(runtime_service.training_config.get("rounds", 10)),
        )

    @app.get("/api/status")
    @app.get("/api/cluster_status")
    def status() -> Any:
        """Return the current cluster state for dashboard polling."""

        runtime_service.refresh_worker_health()
        return jsonify(runtime_service.state.snapshot())

    @app.post("/api/start_training")
    def start_training() -> Any:
        """Start the background training thread on demand."""

        started = runtime_service.start_training_thread()
        runtime_service.refresh_worker_health()
        return jsonify({"started": started, "state": runtime_service.state.snapshot()})

    return app


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the DFS-lite master service."""

    parser = argparse.ArgumentParser(description="DFS-lite federated learning master service.")
    parser.add_argument("--config", default="config_extended.json", help="Path to the JSON configuration file.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind.")
    parser.add_argument("--port", default=8080, type=int, help="Port to bind.")
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
