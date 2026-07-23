"""Pinocchio FK/IK for one Piper arm chain out of the dual-arm URDF.

Poses are 6D [x, y, z, roll, pitch, yaw] (meters, radians, extrinsic xyz
Euler — same convention as DROID's cartesian_position), expressed in the
configured base frame (e.g. left_base_link).
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    return pin.rpy.rpyToMatrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))


def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    return np.asarray(pin.rpy.matrixToRpy(R), dtype=np.float64)


def pose_to_se3(pose6: np.ndarray) -> pin.SE3:
    pose6 = np.asarray(pose6, dtype=np.float64)
    return pin.SE3(rpy_to_matrix(pose6[3:6]), pose6[:3].copy())


def se3_to_pose(M: pin.SE3) -> np.ndarray:
    return np.concatenate([np.asarray(M.translation).ravel(), matrix_to_rpy(np.asarray(M.rotation))])


class PiperKinematics:
    """FK/IK on the 6-DoF chain `<side>_joint1..6` of the piper_x URDF."""

    def __init__(
        self,
        urdf_path: str,
        base_frame: str = "left_base_link",
        ee_frame: str = "left_gripper_base",
        side: str = "left",
    ):
        full_model = pin.buildModelFromUrdf(str(urdf_path))
        self.side = side
        self.joint_names = [f"{side}_joint{i}" for i in range(1, 7)]

        keep_ids = set()
        for name in self.joint_names:
            if not full_model.existJointName(name):
                raise ValueError(f"Joint {name} not found in {urdf_path}")
            keep_ids.add(full_model.getJointId(name))
        lock_ids = [jid for jid in range(1, full_model.njoints) if jid not in keep_ids]
        q_neutral = pin.neutral(full_model)
        self.model = pin.buildReducedModel(full_model, lock_ids, q_neutral)
        if self.model.nq != 6:
            raise RuntimeError(f"Expected 6-DoF reduced model, got nq={self.model.nq}")
        self.data = self.model.createData()

        for frame in (base_frame, ee_frame):
            if not self.model.existFrame(frame):
                raise ValueError(f"Frame {frame} not found in {urdf_path}")
        self.base_frame_id = self.model.getFrameId(base_frame)
        self.ee_frame_id = self.model.getFrameId(ee_frame)

        self.lower_limits = np.asarray(self.model.lowerPositionLimit, dtype=np.float64)
        self.upper_limits = np.asarray(self.model.upperPositionLimit, dtype=np.float64)

    def clamp_to_limits(self, q: np.ndarray, margin: float = 0.0) -> np.ndarray:
        return np.clip(q, self.lower_limits + margin, self.upper_limits - margin)

    def gravity_torques(self, q: np.ndarray) -> np.ndarray:
        """Joint torques (N·m) holding configuration q against gravity."""
        q = np.asarray(q, dtype=np.float64).reshape(self.model.nq)
        return np.asarray(
            pin.computeGeneralizedGravity(self.model, self.data, q), dtype=np.float64
        ).copy()

    def fk(self, q: np.ndarray) -> np.ndarray:
        """EE pose (6D) in the base frame for joint config q (6, rad)."""
        return se3_to_pose(self.fk_se3(q))

    def fk_se3(self, q: np.ndarray) -> pin.SE3:
        q = np.asarray(q, dtype=np.float64).reshape(self.model.nq)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        oMb = self.data.oMf[self.base_frame_id]
        oMe = self.data.oMf[self.ee_frame_id]
        return oMb.inverse() * oMe

    def ik(
        self,
        target_pose6: np.ndarray,
        q_init: np.ndarray,
        max_iters: int = 100,
        tol: float = 1e-4,
        damping: float = 1e-6,
        step_scale: float = 0.5,
    ) -> tuple[np.ndarray, bool, float]:
        """Damped-least-squares IK toward target EE pose (6D, base frame).

        Returns (q, converged, final_error_norm). q is clamped to joint
        limits every iteration; caller must still validate the result.
        """
        pin.forwardKinematics(self.model, self.data, np.asarray(q_init, dtype=np.float64))
        pin.updateFramePlacements(self.model, self.data)
        oMb = self.data.oMf[self.base_frame_id]
        target = oMb * pose_to_se3(target_pose6)  # base frame -> model world frame

        q = np.asarray(q_init, dtype=np.float64).copy()
        err_norm = np.inf
        for _ in range(max_iters):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            oMe = self.data.oMf[self.ee_frame_id]
            err = pin.log(oMe.inverse() * target).vector
            err_norm = float(np.linalg.norm(err))
            if err_norm < tol:
                return q, True, err_norm
            J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id, pin.ReferenceFrame.LOCAL)
            JJt = J @ J.T + damping * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            q = self.clamp_to_limits(q + step_scale * dq)
        return q, err_norm < tol, err_norm
