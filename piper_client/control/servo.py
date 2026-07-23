"""50 Hz joint interpolation/streaming thread.

The policy sets a joint + gripper target at ~10 Hz; this loop linearly
interpolates toward it and streams JointCtrl/GripperCtrl at servo rate.
If no fresh target arrives within `command_timeout_s`, the loop holds the
last interpolated position (keeps re-sending it) instead of extrapolating.

In dry-run mode the loop runs the full interpolation but never sends.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from piper_client.hardware.arm import PiperArm

logger = logging.getLogger(__name__)


class ServoLoop:
    def __init__(
        self,
        arm: PiperArm,
        servo_hz: float = 50.0,
        command_timeout_s: float = 0.25,
        gripper_effort_sdk: int = 1000,
        dry_run: bool = True,
    ):
        self.arm = arm
        self.dt = 1.0 / float(servo_hz)
        self.command_timeout_s = float(command_timeout_s)
        self.gripper_effort_sdk = int(gripper_effort_sdk)
        self.dry_run = dry_run

        self._lock = threading.Lock()
        self._target_q: np.ndarray | None = None
        self._target_grip_m: float | None = None
        self._target_time: float = 0.0
        self._target_duration: float = 0.1  # seconds to reach target (policy period)

        self._q_sent: np.ndarray | None = None
        self._grip_sent: float | None = None
        self._q_at_target_set: np.ndarray | None = None
        self._holding = False

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, q_initial: np.ndarray, grip_initial_m: float):
        """Start streaming from the given (measured) state."""
        self._q_sent = np.asarray(q_initial, dtype=np.float64).reshape(6).copy()
        self._grip_sent = float(grip_initial_m)
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"servo-{self.arm.can_name}")
        self._stop.clear()
        self._thread.start()

    def set_target(self, q_target: np.ndarray, grip_target_m: float, duration_s: float = 0.1):
        with self._lock:
            self._target_q = np.asarray(q_target, dtype=np.float64).reshape(6).copy()
            self._target_grip_m = float(grip_target_m)
            self._target_time = time.monotonic()
            self._target_duration = max(float(duration_s), self.dt)
            self._q_at_target_set = self._q_sent.copy() if self._q_sent is not None else self._target_q.copy()

    def move_to_blocking(self, q_target: np.ndarray, grip_target_m: float, duration_s: float):
        """Slow interpolated move (e.g. reset motion); blocks until reached."""
        self.set_target(q_target, grip_target_m, duration_s)
        deadline = time.monotonic() + duration_s + 1.0
        while time.monotonic() < deadline:
            with self._lock:
                q_sent, tq = self._q_sent, self._target_q
            if q_sent is not None and tq is not None and np.abs(q_sent - tq).max() < 1e-6:
                return
            time.sleep(self.dt)
        logger.warning("move_to_blocking: did not settle within %.1fs", duration_s + 1.0)

    def _loop(self):
        next_tick = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            with self._lock:
                target_q = self._target_q
                target_grip = self._target_grip_m
                t_set = self._target_time
                dur = self._target_duration
                q_from = self._q_at_target_set

                if target_q is not None:
                    stale = (now - t_set) > (dur + self.command_timeout_s)
                    if stale:
                        if not self._holding:
                            # Reaching the target and then hearing nothing is the
                            # normal end of a blocking move — only warn if we were
                            # starved mid-interpolation.
                            reached = self._q_sent is not None and np.abs(self._q_sent - target_q).max() < 1e-9
                            if not reached:
                                logger.warning("servo: command timeout — holding position")
                            self._holding = True
                        # hold: keep re-sending the current interpolated point
                    else:
                        self._holding = False
                        alpha = np.clip((now - t_set) / dur, 0.0, 1.0)
                        self._q_sent = q_from + alpha * (target_q - q_from)
                        if target_grip is not None:
                            self._grip_sent = target_grip
                q_send, grip_send = self._q_sent, self._grip_sent

            if q_send is not None and not self.dry_run:
                try:
                    self._send(q_send, grip_send)
                except Exception:
                    logger.exception("servo: send failed")

            next_tick += self.dt
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()

    def _send(self, q_send, grip_send):
        """Transmit one setpoint. Subclasses override (e.g. MIT mode)."""
        self.arm.command_joints(q_send)
        if grip_send is not None:
            self.arm.command_gripper(grip_send, self.gripper_effort_sdk)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
