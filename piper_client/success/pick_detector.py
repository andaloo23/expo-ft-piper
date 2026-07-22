"""State-based pick success detector (no camera).

Success once (gripper partially closed AND EE lifted above z threshold AND
gripper still commanded closed) holds for `consecutive_steps` steps.

Add this only after the rest of the pipeline works — manual labeling
(success/manual.py) is the default.
"""

from __future__ import annotations

import numpy as np


class PiperPickDetector:
    def __init__(
        self,
        grip_closed_min: float = 0.15,   # normalized gripper_position (0 open, 1 closed)
        grip_closed_max: float = 0.85,   # fully closed = grasped nothing
        lift_z_m: float = 0.20,          # EE z in base frame
        consecutive_steps: int = 3,
    ):
        self.grip_closed_min = float(grip_closed_min)
        self.grip_closed_max = float(grip_closed_max)
        self.lift_z_m = float(lift_z_m)
        self.consecutive_steps = int(consecutive_steps)
        self._streak = 0

    def reset_sequence(self):
        self._streak = 0

    def detect(self, gripper_position: float, cartesian_position: np.ndarray) -> bool:
        z = float(np.asarray(cartesian_position).ravel()[2])
        g = float(np.asarray(gripper_position).ravel()[0])
        holding = self.grip_closed_min <= g <= self.grip_closed_max
        lifted = z >= self.lift_z_m
        self._streak = self._streak + 1 if (holding and lifted) else 0
        return self._streak >= self.consecutive_steps
