"""REST worker service for local federated training rounds."""

from __future__ import annotations

import argparse
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from flask import Flask, jsonify, request
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

LOGGER = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure worker logging output."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_classifier(model_config: dict[str, Any]) -> SGDClassifier:
    """Create an SGDClassifier configured for incremental learning."""

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
    """Initialise a classifier instance so weights can be injected safely."""

    seed_features = np.zeros((len(classes), n_features), dtype=float)
    seed_labels = classes.copy()
    model.partial_fit(seed_features, seed_labels, classes=classes)
    model.coef_[:] = 0.0
    model.intercept_[:] = 0.0
    return model


def deserialise_weights(payload: Sequence[Any]) -> np.ndarray:
    """Convert a JSON payload into a 2D weight matrix."""

    weights = np.asarray(payload, dtype=float)
    if weights.ndim == 1:
        return weights.reshape(1, -1)
    return weights


def deserialise_intercept(payload: Sequence[Any]) -> np.ndarray:
    """Convert a JSON payload into a 1D intercept vector."""

    return np.asarray(payload, dtype=float).reshape(-1)


def serialise_array(array: np.ndarray) -> list[Any]:
    """Convert a NumPy array into a JSON-safe list."""

    if array.ndim == 2 and array.shape[0] == 1:
        return array[0].tolist()
    return array.tolist()


@dataclass
class WorkerState:
    """Mutable worker-local training state guarded by a lock."""

    worker_id: str
    model_config: dict[str, Any] = field(default_factory=dict)
    features: np.ndarray | None = None
    labels: np.ndarray | None = None
    classes: np.ndarray | None = None
    model: SGDClassifier | None = None
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def is_initialised(self) -> bool:
        """Return whether the worker has been initialised with local data."""

        return self.features is not None and self.labels is not None and self.model is not None

    def initialise(
        self,
        worker_id: str,
        features: Sequence[Sequence[float]],
        labels: Sequence[float],
        classes: Sequence[float],
        model_config: dict[str, Any] | None = None,
    ) -> int:
        """Store the worker-local dataset and prepare the SGD classifier."""

        local_features = np.asarray(features, dtype=float)
        local_labels = np.asarray(labels)
        local_classes = np.asarray(classes)
        if local_features.ndim != 2:
            raise ValueError("features must be a 2D array.")
        if local_features.shape[0] == 0:
            raise ValueError("features must contain at least one sample.")
        if local_features.shape[0] != local_labels.shape[0]:
            raise ValueError("features and labels must contain the same number of rows.")

        with self.lock:
            self.worker_id = worker_id or self.worker_id
            self.model_config = model_config or self.model_config or {}
            self.features = local_features
            self.labels = local_labels
            self.classes = local_classes
            self.model = seed_classifier(
                build_classifier(self.model_config),
                classes=self.classes,
                n_features=self.features.shape[1],
            )
        return int(local_features.shape[0])

    def train_round(
        self,
        round_number: int,
        global_weights: Sequence[Any],
        global_intercept: Sequence[Any],
        local_epochs: int,
    ) -> dict[str, Any]:
        """Run a communication round using the master-broadcast parameters."""

        if local_epochs <= 0:
            raise ValueError("local_epochs must be positive.")

        with self.lock:
            if not self.is_initialised() or self.features is None or self.labels is None:
                raise RuntimeError("Worker has not been initialised.")
            if self.model is None or self.classes is None:
                raise RuntimeError("Worker model state is unavailable.")

            updated_weights = deserialise_weights(global_weights)
            updated_intercept = deserialise_intercept(global_intercept)
            if updated_weights.shape[1] != self.features.shape[1]:
                raise ValueError("Weight dimension does not match local feature dimension.")

            self.model.coef_ = updated_weights.copy()
            self.model.intercept_ = updated_intercept.copy()
            self.model.classes_ = self.classes.copy()
            self.model.n_features_in_ = self.features.shape[1]

            base_seed = int(self.model_config.get("random_state", 42))
            rng = np.random.default_rng(base_seed + round_number)
            for _ in range(local_epochs):
                sample_order = rng.permutation(self.features.shape[0])
                self.model.partial_fit(
                    self.features[sample_order],
                    self.labels[sample_order],
                    classes=self.classes,
                )

            probabilities = self.model.predict_proba(self.features)
            local_loss = float(log_loss(self.labels, probabilities, labels=self.classes))
            return {
                "worker_id": self.worker_id,
                "samples_processed": int(self.features.shape[0]),
                "updated_weights": serialise_array(self.model.coef_),
                "updated_intercept": serialise_array(self.model.intercept_),
                "local_loss": local_loss,
            }


def create_app(default_worker_id: str | None = None) -> Flask:
    """Create and configure the Flask application."""

    app = Flask(__name__)
    state = WorkerState(worker_id=default_worker_id or "worker")
    app.config["WORKER_STATE"] = state

    @app.get("/health")
    def health() -> tuple[Any, int]:
        """Report worker liveness and initialisation status."""

        runtime_state: WorkerState = app.config["WORKER_STATE"]
        return (
            jsonify(
                {
                    "worker_id": runtime_state.worker_id,
                    "status": "ok",
                    "initialised": runtime_state.is_initialised(),
                }
            ),
            200,
        )

    @app.post("/initialize")
    def initialize() -> tuple[Any, int]:
        """Load the worker-local dataset and instantiate the model."""

        payload = request.get_json(silent=True) or {}
        required_fields = {"features", "labels", "classes"}
        missing_fields = required_fields - payload.keys()
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {sorted(missing_fields)}"}), 400

        runtime_state: WorkerState = app.config["WORKER_STATE"]
        try:
            samples_loaded = runtime_state.initialise(
                worker_id=payload.get("worker_id", runtime_state.worker_id),
                features=payload["features"],
                labels=payload["labels"],
                classes=payload["classes"],
                model_config=payload.get("model_config", {}),
            )
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

        LOGGER.info("Initialised worker %s with %s samples.", runtime_state.worker_id, samples_loaded)
        return (
            jsonify(
                {
                    "worker_id": runtime_state.worker_id,
                    "samples_loaded": samples_loaded,
                    "feature_count": int(runtime_state.features.shape[1]),
                }
            ),
            200,
        )

    @app.post("/train_round")
    def train_round() -> tuple[Any, int]:
        """Execute local training for a single communication round."""

        payload = request.get_json(silent=True) or {}
        required_fields = {"round_number", "global_weights", "global_intercept", "local_epochs"}
        missing_fields = required_fields - payload.keys()
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {sorted(missing_fields)}"}), 400

        runtime_state: WorkerState = app.config["WORKER_STATE"]
        try:
            response_payload = runtime_state.train_round(
                round_number=int(payload["round_number"]),
                global_weights=payload["global_weights"],
                global_intercept=payload["global_intercept"],
                local_epochs=int(payload["local_epochs"]),
            )
        except (RuntimeError, ValueError) as error:
            status_code = 409 if isinstance(error, RuntimeError) else 400
            return jsonify({"error": str(error)}), status_code

        return jsonify(response_payload), 200

    return app


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the worker service."""

    parser = argparse.ArgumentParser(description="Federated learning worker service.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind.")
    parser.add_argument("--port", default=5000, type=int, help="Port to bind.")
    parser.add_argument("--worker-id", default=os.environ.get("WORKER_ID", "worker"), help="Worker identifier.")
    parser.add_argument("--log-level", default="INFO", help="Logging level for the worker process.")
    return parser


def main() -> int:
    """Run the worker service via Waitress when available."""

    args = build_argument_parser().parse_args()
    configure_logging(args.log_level)
    app_instance = create_app(default_worker_id=args.worker_id)
    try:
        from waitress import serve
    except ImportError:
        LOGGER.warning("Waitress is unavailable. Falling back to Flask's built-in server.")
        app_instance.run(host=args.host, port=args.port)
        return 0

    serve(app_instance, host=args.host, port=args.port)
    return 0


app = create_app(default_worker_id=os.environ.get("WORKER_ID"))


if __name__ == "__main__":
    raise SystemExit(main())
