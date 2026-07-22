"""Human-override detection during online RL.

The operator intervenes simply by moving the master arm. When master EE
speed exceeds `activate_speed`, the sample is flagged human; the flag is
held for `hold_s` after the last active sample so brief pauses mid-correction
don't flip control back to the policy.
"""

from __future__ import annotations

import time

import numpy as np

from piper_client.teleop.master_arm import MasterArm, MasterSample


class InterventionDetector:
    def __init__(self, master: MasterArm, activate_speed: float = 0.008, hold_s: float = 0.5):
        self.master = master
        self.activate_speed = float(activate_speed)  # m/s-equivalent of MasterSample.raw_speed
        self.hold_s = float(hold_s)
        self._active_until = 0.0

    def get_human_action(self) -> tuple[np.ndarray | None, bool]:
        """(action_7d, is_human). action is None when the human is idle."""
        try:
            sample: MasterSample = self.master.read()
        except Exception:
            return None, False
        now = time.monotonic()
        if sample.raw_speed > self.activate_speed:
            self._active_until = now + self.hold_s
        if now <= self._active_until:
            return sample.action, True
        return None, False
