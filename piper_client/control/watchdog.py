"""Freshness tracking for sensors and commands.

Everything that must stay fresh (joint feedback, camera frames, incoming
policy commands) beats a named heartbeat here; consumers ask whether a
signal is stale before trusting it or moving the arm.
"""

from __future__ import annotations

import threading
import time


class Watchdog:
    def __init__(self):
        self._lock = threading.Lock()
        self._beats: dict[str, float] = {}

    def beat(self, name: str, t: float | None = None):
        with self._lock:
            self._beats[name] = time.monotonic() if t is None else t

    def age(self, name: str) -> float:
        with self._lock:
            t = self._beats.get(name)
        return float("inf") if t is None else time.monotonic() - t

    def is_fresh(self, name: str, max_age_s: float) -> bool:
        return self.age(name) <= max_age_s

    def check_fresh(self, requirements: dict[str, float]) -> list[str]:
        """Return the names in {name: max_age_s} that are stale."""
        return [name for name, max_age in requirements.items() if not self.is_fresh(name, max_age)]
