"""PiperPickEnv: pick task on one Piper follower arm.

Success labeling is manual by default (type 1/2 in the actor terminal —
see success/manual.py, already wired in PiperEnv.get_info_for_step). The
automatic state-based detector is opt-in via use_auto_detector once the
rest of the pipeline works.
"""

from __future__ import annotations

import numpy as np

from piper_client.envs.base import PiperEnv
from piper_client.success.pick_detector import PiperPickDetector


class PiperPickEnv(PiperEnv):
    def __init__(
        self,
        use_auto_detector: bool = False,
        pick_lift_z_m: float = 0.20,
        success_reset_randomize_magnitude: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_auto_detector = use_auto_detector
        self.pick_detector = PiperPickDetector(lift_z_m=pick_lift_z_m)
        # Reserved for randomized drop position after success (upstream parity);
        # keep 0.0 until auto-reset behaviors are validated on hardware.
        self.success_reset_randomize_magnitude = float(success_reset_randomize_magnitude)

    def _before_reset(self):
        self.pick_detector.reset_sequence()

    def detect(self, obs):
        if not self.use_auto_detector or obs is None:
            return False, False
        success = self.pick_detector.detect(
            gripper_position=obs["gripper_position"],
            cartesian_position=np.asarray(obs["cartesian_position"], dtype=np.float64),
        )
        return bool(success), False


__all__ = ["PiperPickEnv"]
