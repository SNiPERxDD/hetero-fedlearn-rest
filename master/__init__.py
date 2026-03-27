"""Master package exports."""

from __future__ import annotations

from typing import Any

__all__ = ["FederatedMaster", "run_training_from_config"]


def __getattr__(name: str) -> Any:
    """Lazily expose the public master module symbols."""

    if name in __all__:
        from .master import FederatedMaster, run_training_from_config

        exports = {
            "FederatedMaster": FederatedMaster,
            "run_training_from_config": run_training_from_config,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
