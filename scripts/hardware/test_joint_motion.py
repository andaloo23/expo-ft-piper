"""Step 5 of bring-up: interpolated joint reset motion.

Moves the arm slowly from its current pose to the task reset_joints via
the 50 Hz servo interpolation path. Preview first: run without
--enable-motion to print the planned start/end joints and FK poses.

    python scripts/hardware/test_joint_motion.py --task configs/task/piper_pick.py            # preview
    python scripts/hardware/test_joint_motion.py --task configs/task/piper_pick.py --enable-motion
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import tyro
import yaml

from piper_client.control.safety import CanLock
from piper_client.hardware.arm import PiperArm
from piper_client.hardware.kinematics import PiperKinematics
from piper_client.run_client import load_task_config


def main(task: str = "configs/task/piper_pick.py", enable_motion: bool = False,
         duration_s: float = 6.0):
    cfg = load_task_config(task)
    with open(cfg.hardware_config) as f:
        hw = yaml.safe_load(f)
    kin = PiperKinematics(hw["kinematics"]["urdf"], hw["kinematics"]["base_frame"],
                          hw["kinematics"]["ee_frame"], hw["robot"]["side"])
    q_target = np.asarray(cfg.reset_joints, dtype=np.float64)

    can = hw["robot"]["follower_can"]
    with CanLock(can):
        arm = PiperArm(can)
        arm.connect()
        q0, _ = arm.read_joints()
        grip0, _ = arm.read_gripper()

        np.set_printoptions(precision=4, suppress=True)
        print(f"current q: {q0}\n  FK: {kin.fk(q0)}")
        print(f"target  q: {q_target}\n  FK: {kin.fk(q_target)}")
        print(f"max joint delta: {np.abs(q_target - q0).max():.3f} rad over {duration_s}s")

        if not enable_motion:
            print("Preview only. Pass --enable-motion to execute.")
            arm.disconnect()
            return
        if input("Arm WILL move to target. Type 'yes' to continue: ").strip().lower() != "yes":
            raise SystemExit("Not confirmed.")

        arm.enable()
        arm.set_joint_motion_mode(hw["control"].get("move_speed_rate", 100))
        from piper_client.envs.base import build_servo
        servo = build_servo(arm, kin, hw["control"], hw.get("mit", {}),
                            gravity_scale=hw["robot"].get("gravity_scale"), dry_run=False)
        servo.start(q0, grip0)
        try:
            servo.move_to_blocking(q_target, grip0, duration_s)
            time.sleep(0.5)
            qf, _ = arm.read_joints()
            print(f"final q: {qf}  (err {np.abs(qf - q_target).max():.4f} rad)")
        finally:
            servo.stop()
            arm.disconnect()


if __name__ == "__main__":
    tyro.cli(main)
