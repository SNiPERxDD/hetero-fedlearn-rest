"""DFS-lite worker service with disk-backed block storage and telemetry."""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from ipaddress import IPv4Address
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import requests
from flask import Flask, jsonify, render_template, request
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

LOGGER = logging.getLogger(__name__)

DEFAULT_UDP_DISCOVERY_PORT = 54321


def get_all_lan_ips() -> list[str]:
    """Enumerate all non-loopback IPv4 addresses on this host."""
    
    ips: list[str] = []
    
    # Try getaddrinfo on hostname to get all known addresses
    try:
        hostname = socket.gethostname()
        results = socket.getaddrinfo(hostname, None, socket.AF_INET, socket.SOCK_DGRAM)
        for _, _, _, _, sockaddr in results:
            ip = sockaddr[0]
            if ip and ip != "127.0.0.1" and ip not in ips:
                ips.append(ip)
    except (OSError, socket.gaierror):
        pass
    
    # Try socket probe to external address (detects routing interface)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe_socket:
        try:
            probe_socket.connect(("8.8.8.8", 80))
            ip = str(probe_socket.getsockname()[0])
            if ip and ip != "127.0.0.1" and ip not in ips:
                ips.append(ip)
        except OSError:
            pass
    
    # Try alternative external addresses for more robust detection
    for external_host in [("1.1.1.1", 80), ("9.9.9.9", 80)]:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe_socket:
            try:
                probe_socket.connect(external_host)
                ip = str(probe_socket.getsockname()[0])
                if ip and ip != "127.0.0.1" and ip not in ips:
                    ips.append(ip)
                    break  # Found one, stop
            except OSError:
                pass
    
    return ips


def get_lan_ip() -> str:
    """Return the best-effort LAN IPv4 address for this host."""
    
    ips = get_all_lan_ips()
    if ips:
        LOGGER.debug("Detected LAN IPs: %s, using: %s", ips, ips[0])
        return ips[0]
    
    LOGGER.warning("No non-loopback IPv4 addresses detected, falling back to 127.0.0.1")
    return "127.0.0.1"


def worker_lan_endpoint(lan_ip: str, port: int) -> str:
    """Build the worker's LAN endpoint URL from IP and port."""

    return f"http://{lan_ip}:{port}"


def default_advertised_endpoint(bound_host: str, lan_ip: str, port: int) -> str:
    """Return the default worker endpoint that should be advertised to the master."""

    if bound_host in {"127.0.0.1", "localhost"}:
        return f"http://127.0.0.1:{port}"
    return worker_lan_endpoint(lan_ip, port)


def private_broadcast_candidates(lan_ip: str) -> tuple[str, ...]:
    """Return additional directed broadcast candidates for RFC1918 networks."""

    octets = lan_ip.split(".")
    if len(octets) != 4 or not all(part.isdigit() for part in octets):
        return ()

    first_octet, second_octet, third_octet, _ = (int(part) for part in octets)
    candidates: list[str] = []

    def append_candidate(candidate: str) -> None:
        """Append a broadcast candidate only once while preserving order."""

        if candidate not in candidates:
            candidates.append(candidate)

    if first_octet == 10:
        append_candidate(f"{first_octet}.{second_octet}.{third_octet}.255")
        append_candidate(f"{first_octet}.{second_octet}.255.255")
        append_candidate(f"{first_octet}.255.255.255")
    elif first_octet == 172 and 16 <= second_octet <= 31:
        append_candidate(f"{first_octet}.{second_octet}.{third_octet}.255")
        append_candidate(f"{first_octet}.{second_octet}.255.255")
    elif first_octet == 192 and second_octet == 168:
        append_candidate(f"{first_octet}.{second_octet}.{third_octet}.255")
        append_candidate(f"{first_octet}.{second_octet}.255.255")
    else:
        append_candidate(f"{first_octet}.{second_octet}.{third_octet}.255")

    return tuple(candidates)


def beacon_targets(lan_ip: str, extra_targets: Sequence[str] | None = None) -> tuple[str, ...]:
    """Return UDP discovery targets for robust LAN and same-host registration."""

    targets: list[str] = ["255.255.255.255"]

    for candidate in private_broadcast_candidates(lan_ip):
        if candidate not in targets:
            targets.append(candidate)

    # Add the primary LAN IP itself
    if lan_ip and lan_ip not in targets:
        targets.append(lan_ip)

    # Add all other detected LAN IPs as additional targets for robustness
    all_ips = get_all_lan_ips()
    for ip in all_ips:
        if ip not in targets:
            targets.append(ip)
        for candidate in private_broadcast_candidates(ip):
            if candidate not in targets:
                targets.append(candidate)

    # Add explicit extra targets
    for target in extra_targets or []:
        try:
            normalised_target = str(IPv4Address(str(target).strip()))
        except ValueError:
            continue
        if normalised_target not in targets:
            targets.append(normalised_target)

    LOGGER.debug("Beacon targets for %s: %s", lan_ip, targets)
    return tuple(targets)


def udp_beacon_thread(state: "WorkerDFSState") -> None:
    """Broadcast worker endpoint beacons for master auto-discovery."""

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as beacon_socket:
        beacon_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        beacon_log_count = 0
        while True:
            with state.lock:
                payload = {
                    "worker_id": state.worker_id,
                    "endpoint": state.advertised_endpoint or worker_lan_endpoint(state.lan_ip, state.service_port),
                    "endpoint_candidates": list(state.endpoint_candidates),
                }
                discovery_port = int(state.udp_discovery_port)
                interval_seconds = float(state.beacon_interval_seconds)
                targets = tuple(state.udp_beacon_targets or beacon_targets(state.lan_ip))

            beacon_bytes = json.dumps(payload).encode("utf-8")
            beacon_log_count += 1
            
            # Log every 10th beacon to avoid log spam
            if beacon_log_count % 10 == 0:
                LOGGER.debug("Broadcasting beacon for %s to targets: %s (port %s)", 
                           payload["worker_id"], targets, discovery_port)
            
            failed_targets = []
            for target in targets:
                try:
                    beacon_socket.sendto(beacon_bytes, (target, discovery_port))
                except OSError as e:
                    failed_targets.append((target, str(e)))
            
            if failed_targets and beacon_log_count % 10 == 0:
                LOGGER.debug("Failed to send beacon to some targets: %s", failed_targets)
                
            time.sleep(interval_seconds)


def master_discovery_listener(state: "WorkerDFSState") -> None:
    """Listen for master beacons and self-register without manual endpoint arguments."""

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as listener_socket:
        listener_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        reuse_port = getattr(socket, "SO_REUSEPORT", None)
        if reuse_port is not None:
            try:
                listener_socket.setsockopt(socket.SOL_SOCKET, reuse_port, 1)
            except OSError:
                LOGGER.debug("SO_REUSEPORT is unavailable for worker master discovery listener.")

        try:
            listener_socket.bind(("0.0.0.0", int(state.udp_discovery_port)))
        except OSError as error:
            LOGGER.warning(
                "Worker master discovery listener could not bind to UDP port %s: %s",
                state.udp_discovery_port,
                error,
            )
            return

        listener_socket.settimeout(1.0)
        while True:
            try:
                beacon_bytes, _ = listener_socket.recvfrom(2048)
            except TimeoutError:
                continue
            except OSError as error:
                LOGGER.warning("Worker master discovery listener stopped: %s", error)
                return

            try:
                beacon_payload = json.loads(beacon_bytes.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            master_endpoint = str(beacon_payload.get("master_endpoint", "")).strip().rstrip("/")
            if not master_endpoint.startswith(("http://", "https://")):
                continue

            with state.lock:
                current_master_endpoint = state.master_endpoint
                current_registration_status = state.last_registration_status
                advertised_endpoint = state.advertised_endpoint or worker_lan_endpoint(state.lan_ip, state.service_port)

            if current_master_endpoint == master_endpoint and current_registration_status == "connected":
                continue

            try:
                state.connect_to_master(
                    master_endpoint=master_endpoint,
                    advertised_endpoint=advertised_endpoint,
                    timeout_seconds=5.0,
                )
                LOGGER.info("Worker %s discovered master %s and self-registered.", state.worker_id, master_endpoint)
            except RuntimeError as error:
                LOGGER.debug(
                    "Worker %s heard master beacon from %s but registration is not ready yet: %s",
                    state.worker_id,
                    master_endpoint,
                    error,
                )


def auto_register_master_thread(state: "WorkerDFSState", master_endpoint: str, advertised_endpoint: str) -> None:
    """Retry worker registration against a configured master endpoint."""

    retry_interval_seconds = 5.0
    while True:
        try:
            state.connect_to_master(master_endpoint=master_endpoint, advertised_endpoint=advertised_endpoint)
            LOGGER.info("Worker %s auto-registered with master at %s.", state.worker_id, master_endpoint)
            return
        except RuntimeError as error:
            LOGGER.warning("Worker %s could not auto-register with %s: %s", state.worker_id, master_endpoint, error)
            time.sleep(retry_interval_seconds)


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


def write_block_csv(path: Path, features: np.ndarray, labels: np.ndarray) -> None:
    """Persist a training block to disk as a CSV file."""

    header = ",".join([f"feature_{index}" for index in range(features.shape[1])] + ["label"])
    block_matrix = np.column_stack([features, labels])
    np.savetxt(path, block_matrix, delimiter=",", header=header, comments="")


def read_block_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a persisted training block from disk."""

    block_matrix = np.loadtxt(path, delimiter=",", skiprows=1)
    block_matrix = np.atleast_2d(block_matrix)
    features = block_matrix[:, :-1]
    labels = block_matrix[:, -1]
    return features, labels


@dataclass
class BlockMetadata:
    """Metadata describing a block persisted on local disk."""

    block_id: str
    path: Path
    sample_count: int
    bytes_written: int
    feature_count: int


@dataclass
class WorkerDFSState:
    """Mutable worker state guarded by a lock."""

    worker_id: str
    storage_dir: Path
    model_config: dict[str, Any] = field(default_factory=dict)
    classes: np.ndarray | None = None
    model: SGDClassifier | None = None
    blocks: dict[str, BlockMetadata] = field(default_factory=dict)
    last_local_loss: float | None = None
    last_samples_processed: int = 0
    last_round_number: int | None = None
    master_endpoint: str | None = None
    advertised_endpoint: str | None = None
    last_registration_status: str | None = None
    last_registration_error: str | None = None
    lan_ip: str = "127.0.0.1"
    service_port: int = 5000
    udp_discovery_port: int = DEFAULT_UDP_DISCOVERY_PORT
    udp_beacon_enabled: bool = False
    beacon_interval_seconds: float = 3.0
    udp_beacon_targets: list[str] = field(default_factory=list)
    endpoint_candidates: list[str] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def ensure_storage_dir(self) -> None:
        """Ensure the datanode storage directory exists on disk."""

        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def is_ready(self) -> bool:
        """Return whether the worker has at least one committed block and a seeded model."""

        return self.model is not None and self.classes is not None and bool(self.blocks)

    def initialise_block(
        self,
        *,
        block_id: str,
        worker_id: str,
        features: Sequence[Sequence[float]],
        labels: Sequence[float],
        classes: Sequence[float],
        model_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a block to local disk and prepare the local model state."""

        local_features = np.asarray(features, dtype=float)
        local_labels = np.asarray(labels)
        local_classes = np.asarray(classes)
        if local_features.ndim != 2:
            raise ValueError("features must be a 2D array.")
        if local_features.shape[0] == 0:
            raise ValueError("features must contain at least one sample.")
        if local_features.shape[0] != local_labels.shape[0]:
            raise ValueError("features and labels must contain the same number of rows.")
        if not block_id:
            raise ValueError("block_id is required.")

        with self.lock:
            self.ensure_storage_dir()
            self.worker_id = worker_id or self.worker_id
            self.model_config = model_config or self.model_config or {}

            if self.classes is None:
                self.classes = local_classes
            elif not np.array_equal(self.classes, local_classes):
                raise ValueError("All blocks on a worker must share the same class space.")

            if self.model is None:
                self.model = seed_classifier(
                    build_classifier(self.model_config),
                    classes=self.classes,
                    n_features=local_features.shape[1],
                )
            elif self.model.coef_.shape[1] != local_features.shape[1]:
                raise ValueError("All blocks on a worker must share the same feature dimension.")

            block_path = self.storage_dir / f"{block_id}.csv"
            write_block_csv(block_path, local_features, local_labels)
            bytes_written = int(block_path.stat().st_size)
            self.blocks[block_id] = BlockMetadata(
                block_id=block_id,
                path=block_path,
                sample_count=int(local_features.shape[0]),
                bytes_written=bytes_written,
                feature_count=int(local_features.shape[1]),
            )
            return {
                "status": "block_committed",
                "block_id": block_id,
                "bytes_written": bytes_written,
            }

    def train_round(
        self,
        *,
        block_id: str,
        round_number: int,
        global_weights: Sequence[Any],
        global_intercept: Sequence[Any],
        local_epochs: int,
    ) -> dict[str, Any]:
        """Run a communication round for a persisted local block."""

        if local_epochs <= 0:
            raise ValueError("local_epochs must be positive.")

        with self.lock:
            if not self.is_ready():
                raise RuntimeError("Worker has not been initialised.")
            if self.model is None or self.classes is None:
                raise RuntimeError("Worker model state is unavailable.")
            if block_id not in self.blocks:
                raise RuntimeError(f"Block {block_id} is not present on this worker.")

            block_metadata = self.blocks[block_id]
            updated_weights = deserialise_weights(global_weights)
            updated_intercept = deserialise_intercept(global_intercept)
            if updated_weights.shape[1] != block_metadata.feature_count:
                raise ValueError("Weight dimension does not match block feature dimension.")

            self.model.coef_ = updated_weights.copy()
            self.model.intercept_ = updated_intercept.copy()
            self.model.classes_ = self.classes.copy()
            self.model.n_features_in_ = block_metadata.feature_count

            block_features, block_labels = read_block_csv(block_metadata.path)
            try:
                base_seed = int(self.model_config.get("random_state", 42))
                rng = np.random.default_rng(base_seed + round_number)
                for _ in range(local_epochs):
                    sample_order = rng.permutation(block_features.shape[0])
                    self.model.partial_fit(
                        block_features[sample_order],
                        block_labels[sample_order],
                        classes=self.classes,
                    )

                probabilities = self.model.predict_proba(block_features)
                local_loss = float(log_loss(block_labels, probabilities, labels=self.classes))
            finally:
                del block_features
                del block_labels

            self.last_local_loss = local_loss
            self.last_samples_processed = block_metadata.sample_count
            self.last_round_number = round_number
            return {
                "worker_id": self.worker_id,
                "block_id": block_id,
                "samples_processed": block_metadata.sample_count,
                "updated_weights": serialise_array(self.model.coef_),
                "updated_intercept": serialise_array(self.model.intercept_),
                "local_loss": local_loss,
            }

    def status_payload(self) -> dict[str, Any]:
        """Return a JSON-safe telemetry payload for the worker dashboard."""

        with self.lock:
            self.ensure_storage_dir()
            disk_inventory = []
            total_bytes = 0
            known_blocks = {metadata.path.name: metadata for metadata in self.blocks.values()}
            for block_path in sorted(self.storage_dir.glob("*.csv")):
                bytes_written = int(block_path.stat().st_size)
                total_bytes += bytes_written
                block_metadata = known_blocks.get(block_path.name)
                disk_inventory.append(
                    {
                        "block_id": block_path.stem,
                        "bytes_written": bytes_written,
                        "sample_count": block_metadata.sample_count if block_metadata else None,
                        "path": str(block_path),
                    }
                )

            return {
                "worker_id": self.worker_id,
                "status": "ok",
                "ready": self.is_ready(),
                "storage_dir": str(self.storage_dir),
                "storage_bytes": total_bytes,
                "block_count": len(disk_inventory),
                "blocks": disk_inventory,
                "last_local_loss": self.last_local_loss,
                "last_samples_processed": self.last_samples_processed,
                "last_round_number": self.last_round_number,
                "master_endpoint": self.master_endpoint,
                "advertised_endpoint": self.advertised_endpoint,
                "last_registration_status": self.last_registration_status,
                "last_registration_error": self.last_registration_error,
                "lan_ip": self.lan_ip,
                "service_port": self.service_port,
                "lan_endpoint": worker_lan_endpoint(self.lan_ip, self.service_port),
                "udp_discovery_port": self.udp_discovery_port,
                "udp_beacon_enabled": self.udp_beacon_enabled,
                "udp_beacon_targets": list(self.udp_beacon_targets),
                "endpoint_candidates": list(self.endpoint_candidates),
            }

    def connect_to_master(
        self,
        *,
        master_endpoint: str,
        advertised_endpoint: str,
        timeout_seconds: float = 15.0,
    ) -> dict[str, Any]:
        """Register this worker with a master control plane."""

        resolved_master_endpoint = master_endpoint.strip().rstrip("/")
        resolved_advertised_endpoint = advertised_endpoint.strip().rstrip("/")
        if not resolved_master_endpoint.startswith(("http://", "https://")):
            raise ValueError("master_endpoint must start with http:// or https://.")
        if not resolved_advertised_endpoint.startswith(("http://", "https://")):
            raise ValueError("advertised_endpoint must start with http:// or https://.")

        payload = {
            "worker_id": self.worker_id,
            "endpoint": resolved_advertised_endpoint,
            "endpoint_candidates": list(self.endpoint_candidates),
        }
        try:
            response = requests.post(
                f"{resolved_master_endpoint}/api/workers/register",
                json=payload,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            master_payload = response.json()
        except (requests.RequestException, ValueError) as error:
            with self.lock:
                self.master_endpoint = resolved_master_endpoint
                self.advertised_endpoint = resolved_advertised_endpoint
                self.last_registration_status = "failed"
                self.last_registration_error = str(error)
            raise RuntimeError("Worker could not register with the master.") from error

        with self.lock:
            self.master_endpoint = resolved_master_endpoint
            self.advertised_endpoint = resolved_advertised_endpoint
            self.last_registration_status = "connected"
            self.last_registration_error = None
            return {
                "worker_id": self.worker_id,
                "master_endpoint": self.master_endpoint,
                "advertised_endpoint": self.advertised_endpoint,
                "master_response": master_payload,
            }


def create_app(
    default_worker_id: str | None = None,
    *,
    storage_dir: str | Path | None = None,
    bound_host: str = "0.0.0.0",
    bound_port: int | None = None,
    enable_udp_beacon: bool | None = None,
    enable_master_discovery: bool | None = None,
    udp_discovery_port: int | None = None,
) -> Flask:
    """Create and configure the DFS-lite worker Flask application."""

    app = Flask(__name__, template_folder="templates")
    resolved_port = int(bound_port or os.environ.get("WORKER_PORT", "5000"))
    resolved_discovery_port = int(udp_discovery_port or os.environ.get("UDP_DISCOVERY_PORT", DEFAULT_UDP_DISCOVERY_PORT))
    raw_targets = os.environ.get("UDP_DISCOVERY_TARGETS", "")
    explicit_targets = [value.strip() for value in raw_targets.split(",") if value.strip()]
    detected_ips = get_all_lan_ips()
    resolved_lan_ip = detected_ips[0] if detected_ips else get_lan_ip()
    resolved_targets = list(beacon_targets(resolved_lan_ip, explicit_targets))
    master_endpoint = os.environ.get("MASTER_ENDPOINT", "").strip()
    advertised_endpoint = os.environ.get("ADVERTISED_ENDPOINT", "").strip() or default_advertised_endpoint(
        bound_host,
        resolved_lan_ip,
        resolved_port,
    )
    if enable_udp_beacon is None:
        beacon_enabled = os.environ.get("ENABLE_UDP_BEACON", "1") != "0"
        if "PYTEST_CURRENT_TEST" in os.environ:
            beacon_enabled = False
    else:
        beacon_enabled = bool(enable_udp_beacon)
    if enable_master_discovery is None:
        master_discovery_enabled = os.environ.get("ENABLE_MASTER_DISCOVERY", "1") != "0"
        if "PYTEST_CURRENT_TEST" in os.environ:
            master_discovery_enabled = False
    else:
        master_discovery_enabled = bool(enable_master_discovery)
    endpoint_candidates: list[str] = []
    preferred_candidates = [advertised_endpoint, *(f"http://{ip}:{resolved_port}" for ip in detected_ips)]
    for candidate in preferred_candidates:
        if candidate and candidate not in endpoint_candidates:
            endpoint_candidates.append(candidate)
    if bound_host in {"127.0.0.1", "localhost"}:
        loopback_candidate = f"http://127.0.0.1:{resolved_port}"
        if loopback_candidate not in endpoint_candidates:
            endpoint_candidates.append(loopback_candidate)

    resolved_storage_dir = Path(storage_dir or os.environ.get("DATANODE_STORAGE_DIR") or (
        Path(__file__).resolve().parent / "datanode_storage"
    ))
    state = WorkerDFSState(
        worker_id=default_worker_id or os.environ.get("WORKER_ID", "worker"),
        storage_dir=resolved_storage_dir,
        lan_ip=resolved_lan_ip,
        service_port=resolved_port,
        udp_discovery_port=resolved_discovery_port,
        udp_beacon_enabled=beacon_enabled,
        advertised_endpoint=advertised_endpoint,
        udp_beacon_targets=resolved_targets,
        endpoint_candidates=endpoint_candidates,
    )
    state.ensure_storage_dir()
    app.config["WORKER_STATE"] = state
    app.config["WORKER_HOST"] = bound_host
    app.config["WORKER_PORT"] = resolved_port

    if beacon_enabled:
        thread = threading.Thread(
            target=udp_beacon_thread,
            args=(state,),
            daemon=True,
            name=f"worker-udp-beacon-{resolved_port}",
        )
        thread.start()
    if master_discovery_enabled:
        discovery_thread = threading.Thread(
            target=master_discovery_listener,
            args=(state,),
            daemon=True,
            name=f"worker-master-discovery-{resolved_port}",
        )
        discovery_thread.start()

    if master_endpoint:
        if advertised_endpoint and advertised_endpoint.startswith("http://127.0.0.1") or advertised_endpoint.startswith("http://localhost"):
            LOGGER.warning(
                "advertised_endpoint %s is localhost; worker will not be reachable from master on %s; "
                "set --advertised-endpoint to the worker's LAN IP when registering across networks.",
                advertised_endpoint,
                master_endpoint,
            )
        registration_thread = threading.Thread(
            target=auto_register_master_thread,
            args=(state, master_endpoint, advertised_endpoint),
            daemon=True,
            name=f"worker-master-register-{resolved_port}",
        )
        registration_thread.start()

    @app.get("/")
    def index() -> str:
        """Render the worker telemetry dashboard."""

        return render_template(
            "index_dfs.html",
            worker_id=state.worker_id,
            worker_lan_ip=state.lan_ip,
            worker_port=state.service_port,
            worker_lan_endpoint=worker_lan_endpoint(state.lan_ip, state.service_port),
            udp_discovery_port=state.udp_discovery_port,
            udp_beacon_enabled=state.udp_beacon_enabled,
        )

    @app.get("/health")
    def health() -> tuple[Any, int]:
        """Report worker liveness and readiness state."""

        return (
            jsonify(
                {
                    "worker_id": state.worker_id,
                    "status": "ok",
                    "initialised": state.is_ready(),
                }
            ),
            200,
        )

    @app.get("/api/status")
    def api_status() -> tuple[Any, int]:
        """Return the worker telemetry payload."""

        return jsonify(state.status_payload()), 200

    @app.post("/initialize")
    def initialize() -> tuple[Any, int]:
        """Persist a training block to local disk."""

        payload = request.get_json(silent=True) or {}
        required_fields = {"block_id", "worker_id", "features", "labels", "classes"}
        missing_fields = required_fields - payload.keys()
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {sorted(missing_fields)}"}), 400

        try:
            response_payload = state.initialise_block(
                block_id=str(payload["block_id"]),
                worker_id=str(payload["worker_id"]),
                features=payload["features"],
                labels=payload["labels"],
                classes=payload["classes"],
                model_config=payload.get("model_config", {}),
            )
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

        LOGGER.info(
            "Committed block %s on worker %s.",
            response_payload["block_id"],
            state.worker_id,
        )
        return jsonify(response_payload), 200

    @app.post("/train_round")
    def train_round() -> tuple[Any, int]:
        """Execute local training for a persisted block."""

        payload = request.get_json(silent=True) or {}
        required_fields = {"block_id", "round_number", "global_weights", "global_intercept", "local_epochs"}
        missing_fields = required_fields - payload.keys()
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {sorted(missing_fields)}"}), 400

        try:
            response_payload = state.train_round(
                block_id=str(payload["block_id"]),
                round_number=int(payload["round_number"]),
                global_weights=payload["global_weights"],
                global_intercept=payload["global_intercept"],
                local_epochs=int(payload["local_epochs"]),
            )
        except (RuntimeError, ValueError) as error:
            status_code = 409 if isinstance(error, RuntimeError) else 400
            return jsonify({"error": str(error)}), status_code

        return jsonify(response_payload), 200

    @app.post("/api/connect_master")
    def connect_master() -> tuple[Any, int]:
        """Register this worker endpoint with a master from the worker UI."""

        payload = request.get_json(silent=True) or {}
        master_endpoint = str(payload.get("master_endpoint", "")).strip()
        advertised_endpoint = str(payload.get("advertised_endpoint", "")).strip()
        if not master_endpoint:
            return jsonify({"error": "master_endpoint is required."}), 400
        if not advertised_endpoint:
            advertised_endpoint = worker_lan_endpoint(state.lan_ip, state.service_port)

        try:
            response_payload = state.connect_to_master(
                master_endpoint=master_endpoint,
                advertised_endpoint=advertised_endpoint,
            )
        except (RuntimeError, ValueError) as error:
            status_code = 409 if isinstance(error, RuntimeError) else 400
            return jsonify({"error": str(error), "state": state.status_payload()}), status_code

        return jsonify({"connection": response_payload, "state": state.status_payload()}), 200

    return app


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the DFS-lite worker service."""

    parser = argparse.ArgumentParser(description="DFS-lite federated learning worker service.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind.")
    parser.add_argument("--port", default=5000, type=int, help="Port to bind.")
    parser.add_argument("--worker-id", default=os.environ.get("WORKER_ID", "worker"), help="Worker identifier.")
    parser.add_argument(
        "--storage-dir",
        default=os.environ.get("DATANODE_STORAGE_DIR"),
        help="Directory used for DFS-lite block persistence.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level for the worker process.")
    return parser


def main() -> int:
    """Run the DFS-lite worker service via Waitress when available."""

    args = build_argument_parser().parse_args()
    configure_logging(args.log_level)
    app_instance = create_app(
        default_worker_id=args.worker_id,
        storage_dir=args.storage_dir,
        bound_host=args.host,
        bound_port=args.port,
        enable_udp_beacon=True,
        enable_master_discovery=True,
    )
    try:
        from waitress import serve
    except ImportError:
        LOGGER.warning("Waitress is unavailable. Falling back to Flask's built-in server.")
        app_instance.run(host=args.host, port=args.port)
        return 0

    serve(app_instance, host=args.host, port=args.port)
    return 0


app = create_app(
    default_worker_id=os.environ.get("WORKER_ID"),
    enable_udp_beacon=False,
    enable_master_discovery=False,
)


if __name__ == "__main__":
    raise SystemExit(main())
