"""PiperEnv: the actor-side environment for one Piper follower arm.

Implements the same five operations the EXPO-FT learner drives over the
WebSocket protocol (create_env/reset/step/get_observation/get_info_for_step)
and returns the DROID-shaped observation dict that the existing Pi0.5
wrapper expects.

Safety modes (mutually exclusive, motion off by default):
- dry_run=True: read sensors, run the full action pipeline, never command.
- enable_motion=True: commands allowed (operator confirmed at launch).
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import yaml

from piper_client.control.action_adapter import ActionAdapter
from piper_client.control.safety import CanLock, SafetyError, SafetyLimits
from piper_client.control.servo import ServoLoop
from piper_client.control.watchdog import Watchdog
from piper_client.hardware.arm import PiperArm
from piper_client.hardware.cameras import RealSenseCamera, resize_for_obs
from piper_client.hardware.gripper import GripperModel
from piper_client.hardware.kinematics import PiperKinematics
from piper_client.success.manual import success_detector_manual
from piper_client.teleop.intervention import InterventionDetector
from piper_client.teleop.master_arm import MasterArm

logger = logging.getLogger(__name__)


class PiperEnv:
    def __init__(
        self,
        hardware_config: str = "configs/hardware/gail8_left.yaml",
        language_instruction: str = "",
        bounds=None,                    # (3,2) xyz workspace, base frame
        reset_joints=None,              # (6,) rad
        reset_gripper_opening_m: float | None = None,  # None -> fully open
        image_size=(180, 320),
        control_hz: float = 10,
        auto_reset_steps: int = 80,
        ignore_auto_reset: bool = False,
        video_dir: str = "",
        dry_run: bool = True,
        enable_motion: bool = False,
        use_master_intervention: bool = False,
        reset_duration_s: float = 4.0,
        **kwargs,                       # tolerate extra task-config keys
    ):
        if enable_motion and dry_run:
            raise SafetyError("dry_run and enable_motion are mutually exclusive")
        self.dry_run = not enable_motion  # motion only with explicit opt-in
        self.language_instruction = language_instruction
        self.image_size = tuple(image_size)
        self.control_hz = float(control_hz)
        self.auto_reset_steps = int(auto_reset_steps)
        self.ignore_auto_reset = ignore_auto_reset
        self.video_dir = video_dir or ""
        self.bounds = None if bounds is None else np.asarray(bounds, dtype=np.float64)
        self.reset_joints = None if reset_joints is None else np.asarray(reset_joints, dtype=np.float64)
        self.reset_duration_s = float(reset_duration_s)

        with open(hardware_config) as f:
            self.hw = yaml.safe_load(f)

        ctrl = self.hw["control"]
        self.limits = SafetyLimits(
            max_linear_velocity_mps=ctrl["max_linear_velocity_mps"],
            max_angular_velocity_radps=ctrl["max_angular_velocity_radps"],
            max_gripper_velocity_mps=ctrl.get("max_gripper_velocity_mps", 0.02),
            max_joint_delta_rad=ctrl["max_joint_delta_rad"],
            workspace_bounds=self.bounds,
            command_timeout_s=ctrl["command_timeout_s"],
        )

        kin_cfg = self.hw["kinematics"]
        self.kin = PiperKinematics(
            urdf_path=kin_cfg["urdf"],
            base_frame=kin_cfg["base_frame"],
            ee_frame=kin_cfg["ee_frame"],
            side=self.hw["robot"]["side"],
        )
        self.gripper = GripperModel(max_open_m=self.hw["robot"].get("gripper_max_open_m", 0.07))
        self.adapter = ActionAdapter(self.kin, self.gripper, self.limits, policy_hz=self.control_hz)
        self.watchdog = Watchdog()

        # --- exclusive CAN ownership, then hardware ---
        follower_can = self.hw["robot"]["follower_can"]
        self._can_lock = CanLock(follower_can)
        self._can_lock.acquire()

        self.arm = PiperArm(follower_can)
        self.arm.connect()
        if not self.dry_run:
            self.arm.enable()
            self.arm.set_joint_motion_mode(speed_rate=ctrl.get("move_speed_rate", 100))

        cams = self.hw["cameras"]
        cap_w, cap_h = cams.get("capture_width", 640), cams.get("capture_height", 360)
        self.static_cam = RealSenseCamera(cams["static"]["serial"], cap_w, cap_h, cams["fps"], name="static")
        self.wrist_cam = RealSenseCamera(cams["wrist"]["serial"], cap_w, cap_h, cams["fps"], name="wrist")
        self.static_cam.start()
        self.wrist_cam.start()

        q0, _ = self.arm.read_joints()
        grip0, _ = self.arm.read_gripper()
        self.servo = ServoLoop(
            self.arm,
            servo_hz=ctrl["servo_hz"],
            command_timeout_s=ctrl["command_timeout_s"],
            gripper_effort_sdk=ctrl.get("gripper_effort_sdk", 1000),
            dry_run=self.dry_run,
        )
        self.servo.start(q0, grip0)
        self.adapter.reset(q0, grip0)
        self.reset_gripper_opening_m = (
            self.gripper.max_open_m if reset_gripper_opening_m is None else float(reset_gripper_opening_m)
        )

        # --- optional master-arm intervention (online RL / demos) ---
        self.intervention: InterventionDetector | None = None
        if use_master_intervention:
            master_can = self.hw["robot"]["master_can"]
            master_lock = CanLock(master_can)
            master_lock.acquire()
            self._master_lock = master_lock
            self.master = MasterArm(master_can, self.kin, self.gripper, self.limits)
            self.master.connect()
            self.intervention = InterventionDetector(self.master)

        # episode bookkeeping
        self._steps_since_reset = 0
        self.done, self.success, self.reward = False, False, 0.0
        self.prev_obs: dict | None = None
        self._frame_buffer: list[np.ndarray] = []
        self._ep_count = 0

        logger.info(
            "PiperEnv up on %s (dry_run=%s, motion=%s)",
            follower_can, self.dry_run, not self.dry_run,
        )

    # ---------------- five-operation protocol ----------------

    def reset(self):
        self._before_reset()
        self._steps_since_reset = 0
        self._frame_buffer = []
        self.done, self.success, self.reward = False, False, 0.0

        if self.reset_joints is not None and not self.dry_run:
            self.servo.move_to_blocking(
                self.reset_joints, self.reset_gripper_opening_m, self.reset_duration_s
            )
            time.sleep(0.3)
        q, _ = self.arm.read_joints()
        grip, _ = self.arm.read_gripper()
        self.adapter.reset(q, grip)
        return self.get_observation()

    def _before_reset(self):
        """Override in subclasses (e.g. open gripper / detector reset)."""
        pass

    def get_observation(self):
        static_rgb, t_static = self.static_cam.get_latest()
        wrist_rgb, t_wrist = self.wrist_cam.get_latest()
        if static_rgb is None or wrist_rgb is None:
            raise SafetyError("Camera frame not available (static or wrist)")
        self.watchdog.beat("static_cam", t_static)
        self.watchdog.beat("wrist_cam", t_wrist)

        q, _ = self.arm.read_joints()
        grip_m, _ = self.arm.read_gripper()
        self.watchdog.beat("arm_state")

        h, w = self.image_size
        static_img = resize_for_obs(static_rgb, h, w)
        wrist_img = resize_for_obs(wrist_rgb, h, w)
        ee_pose = self.kin.fk(q).astype(np.float32)
        grip_norm = np.array([self.gripper.opening_to_normalized(grip_m)], dtype=np.float32)

        obs = {
            "exterior_image_1_left": static_img,
            "exterior_image_2_left": static_img,   # deliberate duplicate (DROID contract)
            "wrist_image_left": wrist_img,
            "cartesian_position": ee_pose,
            "gripper_position": grip_norm,
            "prompt": self.language_instruction,
        }
        self.prev_obs = obs
        if self.video_dir:
            self._frame_buffer.append(np.concatenate([static_img, wrist_img], axis=1))
        return obs

    def step(self, action):
        self._steps_since_reset += 1

        stale = self.watchdog.check_fresh({
            "arm_state": self.limits.state_max_age_s,
            "static_cam": self.limits.camera_max_age_s,
            "wrist_cam": self.limits.camera_max_age_s,
        })
        if stale:
            logger.warning("stale signals %s — holding position", stale)
            return {"executed_action": np.zeros(7)}

        result = self.adapter.compute(action)
        if not self.dry_run:
            self.servo.set_target(
                result.q_target, result.gripper_opening_m, duration_s=1.0 / self.control_hz
            )
        return {"executed_action": result.executed_action}

    def get_info_for_step(self):
        obs = self.prev_obs
        time_stop = self.auto_reset_due()
        reached_boundary = self.reached_boundary(obs)

        manual = success_detector_manual()
        if manual == "success":
            done, success = True, True
        elif manual == "reset":
            done, success = True, False
        else:
            success, terminate = self.detect(obs)
            done = bool(success or terminate or time_stop or reached_boundary)

        if done:
            logger.info("Done! success=%s time_stop=%s manual=%s boundary=%s",
                        success, time_stop, manual, reached_boundary)
            self._save_episode_video()
        self.done, self.success = done, success
        reward = 1.0 if success else 0.0
        self.reward = reward
        mask = 0.0 if done else 1.0
        return done, success, reward, mask

    # ---------------- helpers ----------------

    def detect(self, obs) -> tuple[bool, bool]:
        """(success, terminate). Base env: manual labeling only."""
        return False, False

    def get_human_override_action(self):
        """(action_7d or None, is_human) from the master arm, if configured."""
        if self.intervention is None:
            return None, False
        return self.intervention.get_human_action()

    def auto_reset_due(self):
        if self.ignore_auto_reset:
            return False
        due = self._steps_since_reset >= self.auto_reset_steps
        if due:
            logger.info("Timeout...")
        return due

    def reached_boundary(self, obs):
        if self.bounds is None or obs is None:
            return False
        pos = np.asarray(obs["cartesian_position"][:3], dtype=np.float64)
        return bool((pos <= self.bounds[:, 0]).any() or (pos >= self.bounds[:, 1]).any())

    def _save_episode_video(self):
        if not (self.video_dir and self._frame_buffer):
            return
        try:
            import imageio

            os.makedirs(self.video_dir, exist_ok=True)
            path = os.path.join(self.video_dir, f"episode_{self._ep_count}.mp4")
            with imageio.get_writer(path, fps=int(self.control_hz), format="ffmpeg",
                                    codec="libx264", output_params=["-preset", "ultrafast", "-crf", "28"]) as w:
                for f in self._frame_buffer:
                    w.append_data(f)
            logger.info("saved episode video %s", path)
        except Exception:
            logger.exception("failed to save episode video")
        self._frame_buffer = []
        self._ep_count += 1

    @property
    def steps_since_reset(self):
        return self._steps_since_reset

    def close(self):
        try:
            self.servo.stop()
        except Exception:
            pass
        for cam in (getattr(self, "static_cam", None), getattr(self, "wrist_cam", None)):
            if cam is not None:
                cam.stop()
        if getattr(self, "intervention", None) is not None:
            self.master.close()
            self._master_lock.release()
        self.arm.disconnect()
        self._can_lock.release()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
