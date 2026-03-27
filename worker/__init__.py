"""Worker package exports."""

from __future__ import annotations

from typing import Any

__all__ = ["app", "create_app"]


def __getattr__(name: str) -> Any:
    """Lazily expose the public worker module symbols."""

    if name in __all__:
        from .worker import app, create_app

        exports = {
            "app": app,
            "create_app": create_app,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
