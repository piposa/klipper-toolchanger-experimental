"""Compatibility helpers for running the toolchanger extras on Kalico."""

from __future__ import annotations

from typing import Any, List

from . import probe_backports


def ensure_probe_backports(probe_mod: Any) -> List[str]:
    """Patch the supplied ``probe_mod`` with missing helpers if needed."""
    return probe_backports.apply(probe_mod)

__all__ = ["ensure_probe_backports"]
