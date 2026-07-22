"""Safety enforcement for the Piper actor.

All policy/teleop output passes through here before touching the arm:
- NaN/Inf rejection and clamping of normalized actions.
- Cartesian workspace bounds on integrated targets.
- Joint limits and max joint change per command.
- IK convergence gating.
- Sensor/command freshness (see watchdog.py); stale => hold position.
- Exclusive CAN ownership via an OS-level file lock.
"""

from __future__ import annotations

import dataclasses
import fcntl
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


class SafetyError(RuntimeError):
    pass


@dataclasses.dataclass
class SafetyLimits:
    max_linear_velocity_mps: float = 0.03
    max_angular_velocity_radps: float = 0.20
    max_gripper_velocity_mps: float = 0.02
    max_joint_delta_rad: float = 0.02          # per policy-rate command
    workspace_bounds: np.ndarray | None = None  # (3, 2) xyz low/high in base frame
    joint_limit_margin_rad: float = 0.01
    ik_max_error: float = 5e-3                  # rad-equivalent log6 norm
    state_max_age_s: float = 0.2
    camera_max_age_s: float = 0.5
    command_timeout_s: float = 0.25


def sanitize_normalized_action(action) -> np.ndarray:
    """7D normalized action -> finite, clamped to [-1, 1]."""
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.shape[0] != 7:
        raise SafetyError(f"Expected 7D action, got shape {a.shape}")
    if not np.isfinite(a).all():
        logger.warning("Action contains NaN/Inf; replacing with zeros: %s", a)
        a = np.where(np.isfinite(a), a, 0.0)
    return np.clip(a, -1.0, 1.0)


def clamp_position_to_workspace(pos: np.ndarray, bounds: np.ndarray | None) -> np.ndarray:
    if bounds is None:
        return pos
    bounds = np.asarray(bounds, dtype=np.float64)
    return np.clip(pos, bounds[:, 0], bounds[:, 1])


def filter_joint_target(
    q_target: np.ndarray,
    q_reference: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    limits: SafetyLimits,
) -> np.ndarray:
    """Clamp a joint target to limits and to a max delta from the reference
    (the previously commanded target), so one command can never jump."""
    q_target = np.clip(q_target, lower + limits.joint_limit_margin_rad, upper - limits.joint_limit_margin_rad)
    delta = np.clip(q_target - q_reference, -limits.max_joint_delta_rad, limits.max_joint_delta_rad)
    return q_reference + delta


class CanLock:
    """Exclusive, advisory OS lock on a CAN interface (e.g. can_left).

    Only one process may own an interface; the actor refuses to start when
    the lock is held (e.g. ROS/HoloBrain teleop is still running).
    """

    def __init__(self, can_name: str, lock_dir: str = "/tmp"):
        self.can_name = can_name
        self.path = os.path.join(lock_dir, f"expo_ft_piper_{can_name}.lock")
        self._fd: int | None = None

    def acquire(self):
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise SafetyError(
                f"CAN interface {self.can_name} is locked by another process "
                f"({self.path}). Stop ROS/HoloBrain/Piper publishers on this bus first."
            )
        os.truncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._fd = fd
        logger.info("acquired CAN lock %s", self.path)

    def release(self):
        if self._fd is not None:
            try:
                fcntl.lockf(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            finally:
                self._fd = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
