"""ALOHA-style joint-space mirroring for demonstration collection.

The follower copies the master's joint positions at servo rate — crisp,
1:1 teleoperation, speed set by the human hand. Safety comes from a
per-tick joint-speed clamp and the follower's joint limits, which also
double as the engage ramp: on start, the follower converges to the master
pose at the capped speed instead of snapping.

Recording is decoupled: the 10 Hz collection loop derives the normalized
Cartesian-velocity action (the Pi0.5 training contract) from the motion
the follower actually executed — see executed_action_from_obs().
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
import pinocchio as pin

from piper_client.control.safety import SafetyLimits
from piper_client.hardware.kinematics import PiperKinematics, rpy_to_matrix

logger = logging.getLogger(__name__)


class MirrorController:
    def __init__(
        self,
        master,                      # MasterArm
        servo,                       # ServoLoop (owned by the env)
        kin: PiperKinematics,
        panel=None,                  # OperatorPanel; pause freezes mirroring
        mirror_hz: float = 50.0,
        max_joint_speed_rads: float = 2.0,
        gripper_scale: float = 1.0,  # master opening -> follower opening
    ):
        self.master = master
        self.servo = servo
        self.kin = kin
        self.panel = panel
        self.dt = 1.0 / float(mirror_hz)
        self.max_step = float(max_joint_speed_rads) * self.dt
        self.gripper_scale = float(gripper_scale)
        self._q_cmd: np.ndarray | None = None
        self._stop = threading.Event()
        self._active = threading.Event()   # cleared = suspended (e.g. during env.reset)
        self._thread: threading.Thread | None = None

    def start(self):
        """Start the mirror thread in the SUSPENDED state; call resume()
        with the follower's current joints to begin mirroring."""
        self._stop.clear()
        self._active.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="mirror")
        self._thread.start()
        logger.info("mirror thread up (rate cap %.1f rad/s), suspended", self.max_step / self.dt)

    def resume(self, q_current: np.ndarray):
        """Begin mirroring from the follower's current joints. The rate
        clamp makes the initial master/follower offset close smoothly at
        max_joint_speed instead of snapping."""
        self._q_cmd = np.asarray(q_current, dtype=np.float64).copy()
        self._active.set()

    def suspend(self):
        """Stop commanding (e.g. while env.reset() owns the servo)."""
        self._active.clear()

    def _loop(self):
        next_tick = time.monotonic()
        while not self._stop.is_set():
            try:
                if not self._active.is_set() or (self.panel is not None and self.panel.paused):
                    pass  # hold: servo keeps re-sending its last point
                else:
                    q_master, grip_master = self.master.read_raw()
                    q_target = self.kin.clamp_to_limits(q_master)
                    step = np.clip(q_target - self._q_cmd, -self.max_step, self.max_step)
                    self._q_cmd = self._q_cmd + step
                    grip = float(np.clip(grip_master * self.gripper_scale, 0.0, None))
                    self.servo.set_target(self._q_cmd, grip, duration_s=self.dt)
            except Exception:
                logger.exception("mirror tick failed — holding")
            next_tick += self.dt
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def executed_action_from_obs(prev_obs, obs, dt: float, limits: SafetyLimits,
                             gripper_max_open_m: float):
    """Normalized 7D action for the motion between two observations.

    Returns (action[7] clipped to [-1,1], saturated: bool, raw[7]).
    Saturation means the true motion exceeded the contract limits — the
    recorded action understates it. `raw` is the unclipped normalized
    velocity, for choosing limits from real demo statistics.
    """
    p0 = np.asarray(prev_obs["cartesian_position"], dtype=np.float64)
    p1 = np.asarray(obs["cartesian_position"], dtype=np.float64)
    v_lin = (p1[:3] - p0[:3]) / dt
    dR = rpy_to_matrix(p1[3:6]) @ rpy_to_matrix(p0[3:6]).T
    v_ang = np.asarray(pin.log3(dR)).ravel() / dt

    g0 = float(np.asarray(prev_obs["gripper_position"]).ravel()[0])
    g1 = float(np.asarray(obs["gripper_position"]).ravel()[0])
    # gripper_position: 0=open, 1=closed; positive action closes.
    v_close = (g1 - g0) * gripper_max_open_m / dt

    raw = np.concatenate([
        v_lin / limits.max_linear_velocity_mps,
        v_ang / limits.max_angular_velocity_radps,
        [v_close / limits.max_gripper_velocity_mps],
    ])
    saturated = bool(np.abs(raw).max() > 1.0)
    return np.clip(raw, -1.0, 1.0), saturated, raw
