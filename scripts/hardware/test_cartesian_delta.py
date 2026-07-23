"""Step 6 of bring-up: a 1-2 cm Cartesian delta through IK and JointCtrl.

Drives the full action pipeline (normalized action -> adapter -> IK ->
servo) with a constant small velocity for a fixed distance, then reverses.
Run without --enable-motion first to check the IK solution offline.

    python scripts/hardware/test_cartesian_delta.py --task configs/task/piper_pick.py --axis z --distance_m 0.02
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import tyro
import yaml

from piper_client.control.action_adapter import ActionAdapter
from piper_client.control.safety import CanLock, SafetyLimits
from piper_client.hardware.arm import PiperArm
from piper_client.hardware.gripper import GripperModel
from piper_client.hardware.kinematics import PiperKinematics
from piper_client.run_client import load_task_config


def main(task: str = "configs/task/piper_pick.py", axis: str = "z",
         distance_m: float = 0.02, enable_motion: bool = False):
    cfg = load_task_config(task)
    with open(cfg.hardware_config) as f:
        hw = yaml.safe_load(f)
    ctrl = hw["control"]
    limits = SafetyLimits(
        max_linear_velocity_mps=ctrl["max_linear_velocity_mps"],
        max_angular_velocity_radps=ctrl["max_angular_velocity_radps"],
        max_gripper_velocity_mps=ctrl.get("max_gripper_velocity_mps", 0.02),
        max_joint_delta_rad=ctrl["max_joint_delta_rad"],
        workspace_bounds=np.asarray(cfg.bounds) if cfg.bounds is not None else None,
        command_timeout_s=ctrl["command_timeout_s"],
    )
    kin = PiperKinematics(hw["kinematics"]["urdf"], hw["kinematics"]["base_frame"],
                          hw["kinematics"]["ee_frame"], hw["robot"]["side"])
    gripper = GripperModel(hw["robot"].get("gripper_max_open_m", 0.07))
    adapter = ActionAdapter(kin, gripper, limits, policy_hz=cfg.control_hz)

    ax = {"x": 0, "y": 1, "z": 2}[axis]
    dt = 1.0 / cfg.control_hz
    v = limits.max_linear_velocity_mps
    n_steps = max(1, int(round(distance_m / (v * dt))))

    can = hw["robot"]["follower_can"]
    with CanLock(can):
        arm = PiperArm(can)
        arm.connect()
        q0, _ = arm.read_joints()
        grip0, _ = arm.read_gripper()
        adapter.reset(q0, grip0)

        np.set_printoptions(precision=4, suppress=True)
        print(f"start pose: {adapter.commanded_pose6}")
        print(f"{n_steps} steps of +{axis} at {v} m/s, then reverse")

        servo = None
        if enable_motion:
            if input("Arm WILL move. Type 'yes' to continue: ").strip().lower() != "yes":
                raise SystemExit("Not confirmed.")
            arm.enable()
            arm.set_joint_motion_mode(ctrl.get("move_speed_rate", 100))
            from piper_client.envs.base import build_servo
            servo = build_servo(arm, kin, ctrl, hw.get("mit", {}),
                                gravity_scale=hw["robot"].get("gravity_scale"), dry_run=False)
            servo.start(q0, grip0)

        try:
            for direction in (+1.0, -1.0):
                for i in range(n_steps):
                    action = np.zeros(7)
                    action[ax] = direction
                    res = adapter.compute(action)
                    if not res.ik_converged:
                        print(f"  step {i}: IK rejected (err={res.ik_error:.4f}) — held")
                    if servo is not None:
                        servo.set_target(res.q_target, res.gripper_opening_m, dt)
                        time.sleep(dt)
                print(f"after {'+' if direction > 0 else '-'}{axis}: pose {adapter.commanded_pose6}")
        finally:
            if servo is not None:
                servo.stop()
            arm.disconnect()


if __name__ == "__main__":
    tyro.cli(main)
