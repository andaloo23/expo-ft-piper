"""Thin wrapper over piper_sdk C_PiperInterface_V2.

Unit conventions at this boundary (and nowhere else):
- joints: SDK speaks 0.001 degrees; we speak radians.
- gripper: SDK speaks 0.001 mm; we speak meters of opening.

The wrapper never converts anywhere else in the codebase — everything above
this file is rad / m / s.
"""

from __future__ import annotations

import logging
import time

import numpy as np

RAD_TO_MDEG = 180.0 / np.pi * 1000.0

logger = logging.getLogger(__name__)


class PiperArm:
    """One follower (or master, read-only) Piper arm on a CAN interface."""

    def __init__(self, can_name: str, read_only: bool = False):
        self.can_name = can_name
        self.read_only = read_only
        self._piper = None

    def connect(self):
        from piper_sdk import C_PiperInterface_V2

        self._piper = C_PiperInterface_V2(self.can_name)
        self._piper.ConnectPort()
        # Give the SDK reader thread a moment to populate feedback.
        time.sleep(0.2)

    def enable(self, timeout_s: float = 5.0):
        if self.read_only:
            raise RuntimeError(f"{self.can_name}: refusing to enable a read-only arm")
        t0 = time.monotonic()
        while not self._piper.EnablePiper():
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError(f"{self.can_name}: EnablePiper timed out after {timeout_s}s")
            time.sleep(0.01)
        logger.info("%s: arm enabled", self.can_name)

    def set_joint_motion_mode(self, speed_rate: int = 100):
        """CAN command control, MOVE J, firmware position mode (no MIT)."""
        if self.read_only:
            raise RuntimeError(f"{self.can_name}: refusing to command a read-only arm")
        self._piper.MotionCtrl_2(0x01, 0x01, int(speed_rate), 0x00)

    def set_mit_motion_mode(self):
        """CAN command control, MOVE M, MIT mode (0xAD). The SDK demo sends
        this alongside each JointMitCtrl batch; we do the same."""
        if self.read_only:
            raise RuntimeError(f"{self.can_name}: refusing to command a read-only arm")
        self._piper.MotionCtrl_2(0x01, 0x04, 0, 0xAD)

    # ---------------- feedback ----------------

    def read_joints(self) -> tuple[np.ndarray, float]:
        """(q[6] in rad, sdk feedback timestamp)."""
        msg = self._piper.GetArmJointMsgs()
        js = msg.joint_state
        q_mdeg = np.array(
            [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6],
            dtype=np.float64,
        )
        return q_mdeg / RAD_TO_MDEG, float(msg.time_stamp)

    def read_gripper(self) -> tuple[float, float]:
        """(opening in meters, sdk feedback timestamp)."""
        msg = self._piper.GetArmGripperMsgs()
        return float(msg.gripper_state.grippers_angle) * 1e-6, float(msg.time_stamp)

    # ---------------- commands ----------------

    def command_joints(self, q_rad: np.ndarray):
        if self.read_only:
            raise RuntimeError(f"{self.can_name}: refusing to command a read-only arm")
        q = np.asarray(q_rad, dtype=np.float64).reshape(6)
        mdeg = [int(round(v * RAD_TO_MDEG)) for v in q]
        self._piper.JointCtrl(*mdeg)

    def command_joints_mit(self, q_rad: np.ndarray, v_rads: np.ndarray,
                           kp: float, kd: float, t_ff_nm: np.ndarray):
        """Per-joint MIT command (pos rad, vel rad/s, gains, feedforward N·m).
        Caller is responsible for clamping t_ff (±8 N·m on this firmware —
        larger values alias) and for gravity feedforward."""
        if self.read_only:
            raise RuntimeError(f"{self.can_name}: refusing to command a read-only arm")
        q = np.asarray(q_rad, dtype=np.float64).reshape(6)
        v = np.asarray(v_rads, dtype=np.float64).reshape(6)
        t = np.asarray(t_ff_nm, dtype=np.float64).reshape(6)
        for j in range(6):
            self._piper.JointMitCtrl(j + 1, float(q[j]), float(v[j]),
                                     float(kp), float(kd), float(t[j]))

    def command_gripper(self, opening_m: float, effort_sdk: int = 1000):
        if self.read_only:
            raise RuntimeError(f"{self.can_name}: refusing to command a read-only arm")
        sdk_gripper = int(round(max(opening_m, 0.0) * 1_000_000.0))
        self._piper.GripperCtrl(sdk_gripper, int(effort_sdk), 0x01, 0)

    def emergency_stop(self):
        try:
            self._piper.EmergencyStop(0x01)
        except Exception:
            logger.exception("%s: emergency stop failed", self.can_name)

    def disconnect(self):
        if self._piper is not None:
            try:
                self._piper.DisconnectPort()
            except Exception:
                logger.exception("%s: disconnect failed", self.can_name)
            self._piper = None
