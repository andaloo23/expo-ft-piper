"""Step 1 of bring-up: read-only Piper state through piper_sdk.

Prints joint angles (rad), gripper opening (m), and FK EE pose at ~5 Hz.
Never enables or commands the arm.

    python scripts/hardware/read_state.py --can can_left --hardware configs/hardware/gail8_left.yaml
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import tyro
import yaml

from piper_client.hardware.arm import PiperArm
from piper_client.hardware.kinematics import PiperKinematics


def main(can: str = "can_left", hardware: str = "configs/hardware/gail8_left.yaml", hz: float = 5.0):
    with open(hardware) as f:
        hw = yaml.safe_load(f)
    kin = PiperKinematics(
        hw["kinematics"]["urdf"], hw["kinematics"]["base_frame"],
        hw["kinematics"]["ee_frame"], hw["robot"]["side"],
    )
    arm = PiperArm(can, read_only=True)
    arm.connect()
    print(f"reading {can} (read-only, Ctrl+C to stop)")
    try:
        while True:
            q, tq = arm.read_joints()
            grip, tg = arm.read_gripper()
            pose = kin.fk(q)
            np.set_printoptions(precision=4, suppress=True)
            print(f"q(rad)={q}  grip={grip*1000:.1f}mm  ee[xyz]={pose[:3]}  ee[rpy]={pose[3:]}  t={tq:.3f}")
            time.sleep(1.0 / hz)
    except KeyboardInterrupt:
        pass
    finally:
        arm.disconnect()


if __name__ == "__main__":
    tyro.cli(main)
