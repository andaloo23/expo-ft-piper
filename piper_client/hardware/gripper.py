"""Gripper model: normalized position <-> physical opening.

Convention (matches DROID so Pi0.5-DROID initialization transfers):
- observation gripper_position in [0, 1]: 0 = fully open, 1 = fully closed.
- action gripper_velocity in [-1, 1]: positive closes, negative opens.

Physical opening is meters of stroke (Piper: 0 = closed, `max_open_m` = open).
"""

from __future__ import annotations

import numpy as np


class GripperModel:
    def __init__(self, max_open_m: float = 0.07):
        self.max_open_m = float(max_open_m)

    def opening_to_normalized(self, opening_m: float) -> float:
        frac = np.clip(opening_m / self.max_open_m, 0.0, 1.0)
        return float(1.0 - frac)

    def normalized_to_opening(self, normalized: float) -> float:
        normalized = float(np.clip(normalized, 0.0, 1.0))
        return (1.0 - normalized) * self.max_open_m

    def integrate(self, opening_m: float, close_velocity_mps: float, dt: float) -> float:
        """New opening after applying a closing velocity (m/s, positive closes)."""
        return float(np.clip(opening_m - close_velocity_mps * dt, 0.0, self.max_open_m))
