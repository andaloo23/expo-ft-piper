"""BimanualPiperEnv: both Piper followers + three cameras, for two-arm
teleop demonstration collection.

Observation is a SUPERSET of the single-arm contract — left-arm keys keep
their exact single-arm names so existing single-arm training runs on
bimanual files unchanged; right-arm keys get a `_right` suffix. Joint
positions are recorded for both arms so a future joint-action (ALOHA-style)
policy can be trained from the same data.

    exterior_image_1_left / exterior_image_2_left   (static, duplicated)
    wrist_image_left / wrist_image_right
    cartesian_position / cartesian_position_right   (6 each)
    gripper_position / gripper_position_right       (1 each)
    joint_position / joint_position_right           (6 each)
    prompt

The policy step() path is NOT implemented — this env is for collection
(and later bimanual rollout work); run_client still uses the single-arm
PiperEnv.
"""

from __future__ import annotations

import logging
import os
import threading
import time

import numpy as np
import yaml

from piper_client.control.safety import CanLock, SafetyError, SafetyLimits
from piper_client.control.watchdog import Watchdog
from piper_client.envs.base import build_servo
from piper_client.hardware.arm import PiperArm
from piper_client.hardware.cameras import RealSenseCamera, resize_for_obs
from piper_client.hardware.gripper import GripperModel
from piper_client.hardware.kinematics import PiperKinematics
from piper_client.success.manual import success_detector_manual
from piper_client.success.operator_panel import OperatorPanel

logger = logging.getLogger(__name__)


class _ArmUnit:
    """One follower arm bundle (lock, sdk, kinematics, gripper, servo)."""

    def __init__(self, side: str, suffix: str, robot_cfg: dict, kin_cfg: dict,
                 ctrl: dict, urdf: str, dry_run: bool, mit_cfg: dict | None = None):
        self.side = side
        self.suffix = suffix                     # "" for left, "_right" for right
        self.master_can = robot_cfg["master_can"]
        self.kin = PiperKinematics(urdf, kin_cfg["base_frame"], kin_cfg["ee_frame"], side)
        self.gripper = GripperModel(robot_cfg.get("gripper_max_open_m", 0.07))
        self.lock = CanLock(robot_cfg["follower_can"])
        self.lock.acquire()
        self.arm = PiperArm(robot_cfg["follower_can"])
        self.arm.connect()
        if not dry_run:
            self.arm.enable()
            self.arm.set_joint_motion_mode(speed_rate=ctrl.get("move_speed_rate", 100))
        q0, _ = self.arm.read_joints()
        grip0, _ = self.arm.read_gripper()
        self.servo = build_servo(
            self.arm, self.kin, ctrl, mit_cfg or {},
            gravity_scale=robot_cfg.get("gravity_scale"),
            dry_run=dry_run,
        )
        self.servo.start(q0, grip0)

    def close(self):
        try:
            self.servo.stop()
        except Exception:
            pass
        self.arm.disconnect()
        self.lock.release()


class BimanualPiperEnv:
    def __init__(
        self,
        hardware_config: str = "configs/hardware/gail8_dual.yaml",
        language_instruction: str = "",
        bounds=None,                       # left_base_link frame; engage gate + logging only
        reset_joints=None,                 # (6,) left arm
        reset_joints_right=None,           # (6,) right arm
        image_size=(180, 320),
        control_hz: float = 10,
        auto_reset_steps: int = 80,
        ignore_auto_reset: bool = True,
        video_dir: str = "",
        dry_run: bool = True,
        enable_motion: bool = False,
        use_operator_panel: bool = True,
        reset_duration_s: float = 4.0,
        **kwargs,
    ):
        if enable_motion and dry_run:
            raise SafetyError("dry_run and enable_motion are mutually exclusive")
        self.dry_run = not enable_motion
        self.language_instruction = language_instruction
        self.image_size = tuple(image_size)
        self.control_hz = float(control_hz)
        self.auto_reset_steps = int(auto_reset_steps)
        self.ignore_auto_reset = ignore_auto_reset
        self.video_dir = video_dir or ""
        self.bounds = None if bounds is None else np.asarray(bounds, dtype=np.float64)
        self.reset_joints = None if reset_joints is None else np.asarray(reset_joints, dtype=np.float64)
        self.reset_joints_right = (
            None if reset_joints_right is None else np.asarray(reset_joints_right, dtype=np.float64)
        )
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
        urdf = self.hw["kinematics"]["urdf"]
        mit_cfg = self.hw.get("mit", {})
        self._units = [
            _ArmUnit("left", "", self.hw["robot"]["left"], self.hw["kinematics"]["left"],
                     ctrl, urdf, self.dry_run, mit_cfg),
            _ArmUnit("right", "_right", self.hw["robot"]["right"], self.hw["kinematics"]["right"],
                     ctrl, urdf, self.dry_run, mit_cfg),
        ]
        self.watchdog = Watchdog()

        cams = self.hw["cameras"]
        cap_w, cap_h = cams.get("capture_width", 640), cams.get("capture_height", 360)
        self.static_cam = RealSenseCamera(cams["static"]["serial"], cap_w, cap_h, cams["fps"], name="static")
        self.wrist_left_cam = RealSenseCamera(cams["wrist_left"]["serial"], cap_w, cap_h, cams["fps"], name="wrist_left")
        self.wrist_right_cam = RealSenseCamera(cams["wrist_right"]["serial"], cap_w, cap_h, cams["fps"], name="wrist_right")
        for cam in (self.static_cam, self.wrist_left_cam, self.wrist_right_cam):
            cam.start()

        self.panel: OperatorPanel | None = None
        if use_operator_panel:
            panel = OperatorPanel()
            if panel.start():
                self.panel = panel

        self._steps_since_reset = 0
        self.done, self.success, self.reward = False, False, 0.0
        self.prev_obs: dict | None = None
        self._frame_buffer: list[np.ndarray] = []
        self._ep_count = 0
        logger.info("BimanualPiperEnv up on %s + %s (dry_run=%s)",
                    self.hw["robot"]["left"]["follower_can"],
                    self.hw["robot"]["right"]["follower_can"], self.dry_run)

    # ---- units: uniform accessor used by collect_data (PiperEnv has it too) ----

    @property
    def units(self) -> list[dict]:
        return [
            {"suffix": u.suffix, "arm": u.arm, "kin": u.kin, "servo": u.servo,
             "gripper": u.gripper, "master_can": u.master_can}
            for u in self._units
        ]

    # ---------------- episode interface ----------------

    def reset(self, move: bool = True):
        self._steps_since_reset = 0
        self._frame_buffer = []
        self.done, self.success, self.reward = False, False, 0.0
        if self.panel is not None:
            self.panel.reset_episode()

        if move and not self.dry_run:
            threads = []
            for unit, q_reset in zip(self._units, (self.reset_joints, self.reset_joints_right)):
                if q_reset is None:
                    continue
                t = threading.Thread(
                    target=unit.servo.move_to_blocking,
                    args=(q_reset, unit.gripper.max_open_m, self.reset_duration_s),
                    daemon=True,
                )
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
            time.sleep(0.3)
        return self.get_observation()

    def reset_episode_clock(self):
        self._steps_since_reset = 0
        self.done, self.success, self.reward = False, False, 0.0
        if self.panel is not None:
            self.panel.reset_episode()

    def get_observation(self):
        static_rgb, t_s = self.static_cam.get_latest()
        wl_rgb, t_l = self.wrist_left_cam.get_latest()
        wr_rgb, t_r = self.wrist_right_cam.get_latest()
        if static_rgb is None or wl_rgb is None or wr_rgb is None:
            raise SafetyError("Camera frame not available (static or wrist)")
        for name, t in (("static_cam", t_s), ("wrist_left_cam", t_l), ("wrist_right_cam", t_r)):
            self.watchdog.beat(name, t)

        h, w = self.image_size
        static_img = resize_for_obs(static_rgb, h, w)
        obs = {
            "exterior_image_1_left": static_img,
            "exterior_image_2_left": static_img,
            "wrist_image_left": resize_for_obs(wl_rgb, h, w),
            "wrist_image_right": resize_for_obs(wr_rgb, h, w),
            "prompt": self.language_instruction,
        }
        for unit in self._units:
            q, _ = unit.arm.read_joints()
            grip_m, _ = unit.arm.read_gripper()
            obs[f"cartesian_position{unit.suffix}"] = unit.kin.fk(q).astype(np.float32)
            obs[f"gripper_position{unit.suffix}"] = np.array(
                [unit.gripper.opening_to_normalized(grip_m)], dtype=np.float32)
            obs[f"joint_position{unit.suffix}"] = q.astype(np.float32)
        self.watchdog.beat("arm_state")

        self.prev_obs = obs
        if self.video_dir:
            self._frame_buffer.append(np.concatenate(
                [obs["wrist_image_left"], static_img, obs["wrist_image_right"]], axis=1))
        return obs

    def step(self, action):
        raise NotImplementedError(
            "BimanualPiperEnv is collection-only for now; the policy rollout "
            "path (run_client) uses the single-arm PiperEnv."
        )

    def get_info_for_step(self):
        time_stop = (not self.ignore_auto_reset) and self._steps_since_reset >= self.auto_reset_steps
        if self.panel is not None:
            manual = self.panel.consume_label() or "keep_going"
        else:
            manual = success_detector_manual()
        if manual == "success":
            done, success = True, True
        elif manual == "reset":
            done, success = True, False
        else:
            done, success = bool(time_stop), False
        if done:
            logger.info("Done! success=%s time_stop=%s manual=%s", success, time_stop, manual)
            self._save_episode_video()
        self.done, self.success = done, success
        reward = 1.0 if success else 0.0
        self.reward = reward
        return done, success, reward, 0.0 if done else 1.0

    def _save_episode_video(self):
        if not (self.video_dir and self._frame_buffer):
            return
        try:
            import imageio

            os.makedirs(self.video_dir, exist_ok=True)
            path = os.path.join(self.video_dir, f"episode_{self._ep_count}.mp4")
            with imageio.get_writer(path, fps=int(self.control_hz), format="ffmpeg",
                                    codec="libx264", output_params=["-preset", "ultrafast", "-crf", "28"]) as wr:
                for f in self._frame_buffer:
                    wr.append_data(f)
        except Exception:
            logger.exception("failed to save episode video")
        self._frame_buffer = []
        self._ep_count += 1

    def close(self):
        if getattr(self, "panel", None) is not None:
            self.panel.stop()
        for cam in (getattr(self, "static_cam", None),
                    getattr(self, "wrist_left_cam", None),
                    getattr(self, "wrist_right_cam", None)):
            if cam is not None:
                cam.stop()
        for unit in getattr(self, "_units", []):
            unit.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = ["BimanualPiperEnv"]
