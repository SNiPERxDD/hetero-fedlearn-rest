"""Federated learning master orchestration entry point."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import requests
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
class WorkerUpdate:
    """A completed worker round update returned to the master."""

    worker_id: str
    samples_processed: int
    updated_weights: np.ndarray
    updated_intercept: np.ndarray
    local_loss: float


@dataclass(frozen=True)
class RoundMetrics:
    """Validation and aggregation metrics for a communication round."""

    round_number: int
    successful_workers: int
    dropped_workers: list[str]
    mean_local_loss: float
    validation_accuracy: float
    validation_loss: float


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
    worker_count: int,
    random_seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partition the dataset into stratified worker-local shards."""

    if worker_count <= 0:
        raise ValueError("worker_count must be positive.")

    rng = np.random.default_rng(random_seed)
    partitions: list[list[int]] = [[] for _ in range(worker_count)]
    for class_label in np.unique(labels):
        class_indices = np.where(labels == class_label)[0]
        rng.shuffle(class_indices)
        for partition_index, class_split in enumerate(np.array_split(class_indices, worker_count)):
            partitions[partition_index].extend(class_split.tolist())

    shards: list[tuple[np.ndarray, np.ndarray]] = []
    for partition_indices in partitions:
        rng.shuffle(partition_indices)
        shard_indices = np.asarray(partition_indices, dtype=int)
        shards.append((features[shard_indices], labels[shard_indices]))
    return shards


class FederatedMaster:
    """Coordinate data partitioning, worker communication, and FedAvg aggregation."""

    def __init__(
        self,
        config: dict[str, Any],
        session: Session | None = None,
    ) -> None:
        """Initialise the master from a configuration dictionary."""

        self.config = config
        self.session = session or requests.Session()
        self.dataset_config = config.get("dataset", {})
        self.model_config = config.get("model", {})
        self.training_config = config.get("training", {})
        self.network_config = config.get("network", {})
        self.workers = [
            WorkerSpec(worker_id=item["worker_id"], endpoint=item["endpoint"].rstrip("/"))
            for item in config.get("workers", [])
        ]
        if not self.workers:
            raise ValueError("Configuration must define at least one worker.")

        self.random_seed = int(self.training_config.get("random_seed", 42))
        self.scaler = StandardScaler()
        self.classes_: np.ndarray | None = None
        self.global_weights: np.ndarray | None = None
        self.global_intercept: np.ndarray | None = None
        self.validation_features: np.ndarray | None = None
        self.validation_labels: np.ndarray | None = None
        self.active_workers: list[WorkerSpec] = []

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

    def prepare_data(
        self,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, float, float]:
        """Split, standardise, seed the global model, and create worker shards."""

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
        shards = partition_dataset(
            train_features,
            train_labels,
            worker_count=len(self.workers),
            random_seed=self.random_seed,
        )
        return shards, train_features, train_labels, initial_accuracy, initial_loss

    def evaluate_global_model(self) -> tuple[float, float]:
        """Evaluate the current global weights on the validation set."""

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

    def post_json(self, worker: WorkerSpec, route: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON to a worker with retries and a strict timeout."""

        timeout_seconds = float(self.network_config.get("timeout_seconds", 120))
        retry_attempts = int(self.network_config.get("retry_attempts", 3))
        retry_backoff_seconds = float(self.network_config.get("retry_backoff_seconds", 1.0))
        url = f"{worker.endpoint}{route}"
        last_exception: Exception | None = None

        for attempt in range(1, retry_attempts + 1):
            try:
                response = self.session.post(url, json=payload, timeout=timeout_seconds)
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as error:
                last_exception = error
                LOGGER.warning(
                    "Worker request failed",
                    extra={
                        "worker_id": worker.worker_id,
                        "route": route,
                        "attempt": attempt,
                        "max_attempts": retry_attempts,
                    },
                )
                if attempt < retry_attempts:
                    time.sleep(retry_backoff_seconds)

        raise RuntimeError(
            f"Worker {worker.worker_id} failed to respond successfully on {route} after "
            f"{retry_attempts} attempts."
        ) from last_exception

    def initialise_workers(self, shards: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """Send each worker its fixed training partition exactly once."""

        if self.classes_ is None:
            raise RuntimeError("Classes are not initialised.")

        active_workers: list[WorkerSpec] = []
        for worker, (local_features, local_labels) in zip(self.workers, shards, strict=True):
            payload = {
                "worker_id": worker.worker_id,
                "features": local_features.tolist(),
                "labels": local_labels.tolist(),
                "classes": self.classes_.tolist(),
                "model_config": self.model_config,
            }
            try:
                response = self.post_json(worker, "/initialize", payload)
            except RuntimeError:
                LOGGER.error("Failed to initialise worker %s.", worker.worker_id)
                continue

            LOGGER.info(
                "Initialised worker %s with %s samples.",
                worker.worker_id,
                response.get("samples_loaded", 0),
            )
            active_workers.append(worker)

        if not active_workers:
            raise RuntimeError("No workers initialised successfully.")

        self.active_workers = active_workers

    def aggregate_worker_updates(self, updates: Sequence[WorkerUpdate]) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate successful worker updates via weighted FedAvg."""

        total_samples = sum(update.samples_processed for update in updates)
        if total_samples <= 0:
            raise RuntimeError("Worker aggregation received zero samples.")

        aggregated_weights = sum(
            (update.samples_processed / total_samples) * update.updated_weights for update in updates
        )
        aggregated_intercept = sum(
            (update.samples_processed / total_samples) * update.updated_intercept for update in updates
        )
        return aggregated_weights, aggregated_intercept

    def run_round(self, round_number: int, local_epochs: int) -> RoundMetrics:
        """Execute one communication round across all currently active workers."""

        if self.global_weights is None or self.global_intercept is None:
            raise RuntimeError("Global parameters must be initialised before training.")

        updates: list[WorkerUpdate] = []
        dropped_workers: list[str] = []
        for worker in self.active_workers:
            payload = {
                "round_number": round_number,
                "global_weights": serialise_array(self.global_weights),
                "global_intercept": serialise_array(self.global_intercept),
                "local_epochs": local_epochs,
            }
            try:
                response = self.post_json(worker, "/train_round", payload)
            except RuntimeError:
                LOGGER.error("Dropping worker %s for round %s.", worker.worker_id, round_number)
                dropped_workers.append(worker.worker_id)
                continue

            updates.append(
                WorkerUpdate(
                    worker_id=response["worker_id"],
                    samples_processed=int(response["samples_processed"]),
                    updated_weights=deserialise_weights(response["updated_weights"]),
                    updated_intercept=deserialise_intercept(response["updated_intercept"]),
                    local_loss=float(response["local_loss"]),
                )
            )

        if not updates:
            raise RuntimeError(f"Round {round_number} failed because no worker updates were received.")

        self.global_weights, self.global_intercept = self.aggregate_worker_updates(updates)
        validation_accuracy, validation_loss = self.evaluate_global_model()
        return RoundMetrics(
            round_number=round_number,
            successful_workers=len(updates),
            dropped_workers=dropped_workers,
            mean_local_loss=float(np.mean([update.local_loss for update in updates])),
            validation_accuracy=validation_accuracy,
            validation_loss=validation_loss,
        )

    def run(self) -> dict[str, Any]:
        """Run the configured federated training session and return a JSON-safe summary."""

        shards, train_features, train_labels, initial_accuracy, initial_loss = self.prepare_data()
        self.initialise_workers(shards)

        rounds = int(self.training_config.get("rounds", 10))
        local_epochs = int(self.training_config.get("local_epochs", 5))
        history: list[dict[str, Any]] = []
        for round_number in range(1, rounds + 1):
            metrics = self.run_round(round_number=round_number, local_epochs=local_epochs)
            history.append(asdict(metrics))
            LOGGER.info(
                "Round %s complete | workers=%s | val_acc=%.4f | val_loss=%.4f",
                metrics.round_number,
                metrics.successful_workers,
                metrics.validation_accuracy,
                metrics.validation_loss,
            )

        final_accuracy, final_loss = self.evaluate_global_model()
        return {
            "dataset": {
                "source": self.dataset_config.get("source", "builtin"),
                "name": self.dataset_config.get("name", "breast_cancer"),
                "train_samples": int(train_features.shape[0]),
                "validation_samples": int(self.validation_features.shape[0]),
                "feature_count": int(train_features.shape[1]),
                "classes": self.classes_.tolist() if self.classes_ is not None else [],
            },
            "active_workers": [worker.worker_id for worker in self.active_workers],
            "initial_validation_accuracy": initial_accuracy,
            "initial_validation_loss": initial_loss,
            "final_validation_accuracy": final_accuracy,
            "final_validation_loss": final_loss,
            "history": history,
            "training": {
                "rounds": rounds,
                "local_epochs": local_epochs,
                "random_seed": self.random_seed,
                "train_label_cardinality": int(len(np.unique(train_labels))),
            },
        }


def run_training_from_config(config_path: str | Path) -> dict[str, Any]:
    """Convenience wrapper used by the CLI and tests."""

    master = FederatedMaster(load_config(config_path))
    return master.run()


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the master process."""

    parser = argparse.ArgumentParser(description="Federated learning master orchestrator.")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to the JSON configuration file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for the master process.",
    )
    return parser


def main() -> int:
    """Run the master CLI entry point."""

    parser = build_argument_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    summary = run_training_from_config(args.config)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
