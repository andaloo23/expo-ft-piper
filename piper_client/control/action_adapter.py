"""Normalized 7D policy action -> safe joint + gripper target.

Pipeline (one call per policy step, 1/policy_hz seconds of motion):

    [vx, vy, vz, wx, wy, wz, v_grip] in [-1, 1]
        -> clamp/validate (safety.sanitize_normalized_action)
        -> physical Cartesian velocity (config limits)
        -> integrate dt over the *commanded* pose (not measured, so bounded
           tracking error cannot wind up the target)
        -> workspace filter on position
        -> Pinocchio IK
        -> joint limit + max-delta filter
        -> (ServoLoop interpolates to 50 Hz and sends JointCtrl)

Returns the action that was actually applied, recomputed from the filtered
target — EXPO-FT stores this executed action in its replay buffer.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pinocchio as pin

from piper_client.control.safety import (
    SafetyLimits,
    clamp_position_to_workspace,
    filter_joint_target,
    sanitize_normalized_action,
)
from piper_client.hardware.gripper import GripperModel
from piper_client.hardware.kinematics import PiperKinematics, pose_to_se3, se3_to_pose

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AdapterResult:
    q_target: np.ndarray            # (6,) rad
    gripper_opening_m: float
    executed_action: np.ndarray     # (7,) normalized, what was actually applied
    target_pose6: np.ndarray        # (6,) commanded EE pose after all filters
    ik_converged: bool
    ik_error: float


class ActionAdapter:
    def __init__(
        self,
        kinematics: PiperKinematics,
        gripper: GripperModel,
        limits: SafetyLimits,
        policy_hz: float = 10.0,
    ):
        self.kin = kinematics
        self.gripper = gripper
        self.limits = limits
        self.dt = 1.0 / float(policy_hz)
        self._cmd_pose6: np.ndarray | None = None    # commanded EE pose (base frame)
        self._cmd_q: np.ndarray | None = None        # last commanded joint target
        self._cmd_gripper_m: float | None = None     # last commanded opening

    def reset(self, q_measured: np.ndarray, gripper_opening_m: float):
        """Re-anchor the commanded state to measured state (call on env reset)."""
        q = np.asarray(q_measured, dtype=np.float64).reshape(6)
        self._cmd_q = q.copy()
        self._cmd_pose6 = self.kin.fk(q)
        self._cmd_gripper_m = float(np.clip(gripper_opening_m, 0.0, self.gripper.max_open_m))

    @property
    def commanded_pose6(self) -> np.ndarray:
        return None if self._cmd_pose6 is None else self._cmd_pose6.copy()

    @property
    def commanded_gripper_opening(self) -> float | None:
        return self._cmd_gripper_m

    def compute(self, action) -> AdapterResult:
        if self._cmd_pose6 is None:
            raise RuntimeError("ActionAdapter.compute() before reset()")
        lim = self.limits
        a = sanitize_normalized_action(action)

        v_lin = a[0:3] * lim.max_linear_velocity_mps
        v_ang = a[3:6] * lim.max_angular_velocity_radps
        v_grip = a[6] * lim.max_gripper_velocity_mps

        # Integrate over the commanded pose. Angular velocity is in the base
        # frame: R_new = R(dtheta) * R_old.
        prev = pose_to_se3(self._cmd_pose6)
        new_pos = np.asarray(prev.translation).ravel() + v_lin * self.dt
        new_pos = clamp_position_to_workspace(new_pos, lim.workspace_bounds)
        dR = pin.exp3(v_ang * self.dt)
        target = pin.SE3(dR @ np.asarray(prev.rotation), new_pos)
        target_pose6 = se3_to_pose(target)

        q_ik, converged, ik_err = self.kin.ik(target_pose6, self._cmd_q)
        if not converged or ik_err > lim.ik_max_error:
            # Reject the Cartesian step entirely: hold the previous target.
            logger.warning("IK rejected (converged=%s err=%.4f); holding pose", converged, ik_err)
            q_new = self._cmd_q.copy()
            achieved_pose6 = self._cmd_pose6.copy()
        else:
            q_new = filter_joint_target(
                q_ik, self._cmd_q, self.kin.lower_limits, self.kin.upper_limits, lim
            )
            achieved_pose6 = self.kin.fk(q_new)

        new_grip_m = self.gripper.integrate(self._cmd_gripper_m, v_grip, self.dt)

        executed = self._executed_from_poses(self._cmd_pose6, achieved_pose6,
                                             self._cmd_gripper_m, new_grip_m)

        self._cmd_pose6 = achieved_pose6
        self._cmd_q = q_new
        self._cmd_gripper_m = new_grip_m
        return AdapterResult(
            q_target=q_new.copy(),
            gripper_opening_m=new_grip_m,
            executed_action=executed,
            target_pose6=achieved_pose6.copy(),
            ik_converged=converged,
            ik_error=ik_err,
        )

    def _executed_from_poses(self, pose_before, pose_after, grip_before, grip_after) -> np.ndarray:
        """Recover the normalized action that the filtered motion corresponds to."""
        lim = self.limits
        M0, M1 = pose_to_se3(pose_before), pose_to_se3(pose_after)
        v_lin = (np.asarray(M1.translation).ravel() - np.asarray(M0.translation).ravel()) / self.dt
        dR = np.asarray(M1.rotation) @ np.asarray(M0.rotation).T
        v_ang = np.asarray(pin.log3(dR)).ravel() / self.dt
        v_grip = (grip_before - grip_after) / self.dt  # positive closes

        executed = np.concatenate([
            v_lin / lim.max_linear_velocity_mps,
            v_ang / lim.max_angular_velocity_radps,
            [v_grip / lim.max_gripper_velocity_mps],
        ])
        return np.clip(executed, -1.0, 1.0)
