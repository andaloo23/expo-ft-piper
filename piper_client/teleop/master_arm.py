"""Master (leader) Piper arm as a 7D Cartesian-velocity teleop source.

Reads can_left_mst / can_right_mst joints (read-only — the master is never
enabled or commanded), runs FK with the same URDF chain as the follower,
differentiates successive EE poses into Cartesian velocity, and normalizes
by the configured velocity limits so demonstrations use the exact action
contract the policy trains on.
"""

from __future__ import annotations

import dataclasses
import logging
import time

import numpy as np
import pinocchio as pin

from piper_client.control.safety import SafetyLimits
from piper_client.hardware.arm import PiperArm
from piper_client.hardware.gripper import GripperModel
from piper_client.hardware.kinematics import PiperKinematics, pose_to_se3

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MasterSample:
    action: np.ndarray        # (7,) normalized [-1, 1]
    pose6: np.ndarray         # master EE pose (follower base frame)
    gripper_opening_m: float
    raw_speed: float          # unnormalized |v| for intervention detection


class MasterArm:
    def __init__(
        self,
        can_name: str,
        kinematics: PiperKinematics,
        gripper: GripperModel,
        limits: SafetyLimits,
        master_to_follower_base: np.ndarray | None = None,  # 4x4; identity if co-mounted
        smoothing: float = 0.5,  # EMA on velocities; 1.0 = no smoothing
    ):
        self.arm = PiperArm(can_name, read_only=True)
        self.kin = kinematics
        self.gripper = gripper
        self.limits = limits
        self._bTm = pin.SE3(np.eye(4)) if master_to_follower_base is None else pin.SE3(
            np.asarray(master_to_follower_base, dtype=np.float64))
        self.smoothing = float(smoothing)
        self._prev_pose: np.ndarray | None = None
        self._prev_grip: float | None = None
        self._prev_time: float | None = None
        self._vel_ema = np.zeros(7)

    def connect(self):
        self.arm.connect()

    def read(self) -> MasterSample:
        q, _ = self.arm.read_joints()
        grip_m, _ = self.arm.read_gripper()
        now = time.monotonic()

        M = self._bTm * self.kin.fk_se3(q)  # master EE in follower base frame
        pose6 = np.concatenate([
            np.asarray(M.translation).ravel(),
            np.asarray(pin.rpy.matrixToRpy(np.asarray(M.rotation))).ravel(),
        ])

        if self._prev_pose is None or self._prev_time is None:
            self._prev_pose, self._prev_grip, self._prev_time = pose6, grip_m, now
            return MasterSample(np.zeros(7), pose6, grip_m, 0.0)

        dt = max(now - self._prev_time, 1e-3)
        M0 = pose_to_se3(self._prev_pose)
        v_lin = (np.asarray(M.translation).ravel() - np.asarray(M0.translation).ravel()) / dt
        dR = np.asarray(M.rotation) @ np.asarray(M0.rotation).T
        v_ang = np.asarray(pin.log3(dR)).ravel() / dt
        v_grip = (self._prev_grip - grip_m) / dt  # positive closes

        raw = np.concatenate([v_lin, v_ang, [v_grip]])
        self._vel_ema = self.smoothing * raw + (1.0 - self.smoothing) * self._vel_ema
        self._prev_pose, self._prev_grip, self._prev_time = pose6, grip_m, now

        lim = self.limits
        action = np.concatenate([
            self._vel_ema[0:3] / lim.max_linear_velocity_mps,
            self._vel_ema[3:6] / lim.max_angular_velocity_radps,
            [self._vel_ema[6] / lim.max_gripper_velocity_mps],
        ])
        raw_speed = float(np.linalg.norm(self._vel_ema[0:3])) + 0.1 * float(np.linalg.norm(self._vel_ema[3:6]))
        return MasterSample(np.clip(action, -1.0, 1.0), pose6, grip_m, raw_speed)

    def reset_state(self):
        self._prev_pose = self._prev_grip = self._prev_time = None
        self._vel_ema = np.zeros(7)

    def close(self):
        self.arm.disconnect()
