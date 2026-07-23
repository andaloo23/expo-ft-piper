"""MIT-mode servo backend: JointMitCtrl instead of firmware MOVE-J.

Bypasses the firmware position planner for crisp tracking (what the ROS
dagger teleop stack runs), at the cost of doing gravity feedforward
ourselves. Parameters proven on this rig's gravity-comp project:

    kp = 25, kd = 0.8 (kd >= ~1.2 buzzes at firmware rate)
    t_ff = gravity_scale * RNEA gravity, clamped to ±8 N·m
           (values beyond ±8 ALIAS on this firmware family — clamp is
           mandatory, not advisory)
    vel_ref = interpolated target velocity (deriv feedforward halves
              teleop lag at kd=0.8)

Left-arm gravity_scale [1, 1.09, 1.06, 0.88, 1, 1] comes from the
known-mass/gravscale calibration; the right arm is uncalibrated (1.0).

SAFETY: in MIT mode the arm goes SOFT if commands stop. stop() reverts
the firmware to position mode and holds the current measured pose; the
env must call stop() (or park the arm) before exiting.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from piper_client.control.calibration import split_gravity_torque
from piper_client.control.servo import ServoLoop
from piper_client.hardware.arm import PiperArm
from piper_client.hardware.kinematics import PiperKinematics

logger = logging.getLogger(__name__)


class MitServoLoop(ServoLoop):
    def __init__(
        self,
        arm: PiperArm,
        kin: PiperKinematics,
        servo_hz: float = 100.0,
        command_timeout_s: float = 0.25,
        gripper_effort_sdk: int = 1000,
        dry_run: bool = True,
        kp: float = 25.0,
        kd: float = 0.8,
        t_ref_clamp_nm: float = 8.0,
        vel_ff_gain: float = 1.0,
        vel_clamp_rads: float = 4.0,
        gravity_scale=None,
        offset_stiffness=None,        # (6,) from the calibration store, kp-indexed
        deflection_tables=None,       # per-joint [(|tau|,|defl|),...] or None
    ):
        super().__init__(arm, servo_hz=servo_hz, command_timeout_s=command_timeout_s,
                         gripper_effort_sdk=gripper_effort_sdk, dry_run=dry_run)
        self.kin = kin
        self.kp = float(kp)
        self.kd = float(kd)
        self.t_ref_clamp = abs(float(t_ref_clamp_nm))
        self.vel_ff_gain = float(vel_ff_gain)
        self.vel_clamp = abs(float(vel_clamp_rads))
        self.gravity_scale = (
            np.ones(6) if gravity_scale is None
            else np.asarray(gravity_scale, dtype=np.float64).reshape(6)
        )
        self.offset_stiffness = (
            np.zeros(6) if offset_stiffness is None
            else np.asarray(offset_stiffness, dtype=np.float64).reshape(6)
        )
        self.deflection_tables = deflection_tables or [None] * 6
        self._q_prev_sent: np.ndarray | None = None

    def _send(self, q_send, grip_send):
        q = np.asarray(q_send, dtype=np.float64)
        if self._q_prev_sent is None:
            vel = np.zeros(6)
        else:
            vel = (q - self._q_prev_sent) / self.dt
        self._q_prev_sent = q.copy()
        vel = np.clip(vel * self.vel_ff_gain, -self.vel_clamp, self.vel_clamp)

        tau_desired = self.kin.gravity_torques(q) * self.gravity_scale
        t_ref = np.empty(6)
        q_cmd = q.copy()
        for j in range(6):
            t_ref[j], offset = split_gravity_torque(
                tau_desired[j], self.t_ref_clamp, self.kp,
                offset_stiffness=self.offset_stiffness[j],
                deflection_table=self.deflection_tables[j],
            )
            q_cmd[j] += offset

        self.arm.set_mit_motion_mode()
        self.arm.command_joints_mit(q_cmd, vel, self.kp, self.kd, t_ref)
        if grip_send is not None:
            self.arm.command_gripper(grip_send, self.gripper_effort_sdk)

    def stop(self):
        """Stop streaming and revert the firmware to position-mode hold —
        without this the arm goes soft and sags when the process exits."""
        super().stop()
        if self.dry_run:
            return
        try:
            q, _ = self.arm.read_joints()
            self.arm.set_joint_motion_mode(speed_rate=20)
            for _ in range(5):
                self.arm.command_joints(q)
                time.sleep(0.02)
            logger.info("%s: reverted to position-mode hold", self.arm.can_name)
        except Exception:
            logger.exception("%s: failed to revert to position mode — arm may be soft",
                             self.arm.can_name)
