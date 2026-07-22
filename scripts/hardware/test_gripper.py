"""Step 4 of bring-up: gripper-only hardware test (arm does not move).

Opens and closes the gripper a few times through the SDK path. First
motion test — verify e-stop works before running.

    python scripts/hardware/test_gripper.py --can can_left --enable-motion
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tyro

from piper_client.control.safety import CanLock
from piper_client.hardware.arm import PiperArm


def main(can: str = "can_left", enable_motion: bool = False, cycles: int = 3,
         open_m: float = 0.06, closed_m: float = 0.01, effort_sdk: int = 1000):
    if not enable_motion:
        raise SystemExit("Pass --enable-motion to run (this test moves the gripper).")
    if input("Gripper will open/close. Type 'yes' to continue: ").strip().lower() != "yes":
        raise SystemExit("Not confirmed.")

    with CanLock(can):
        arm = PiperArm(can)
        arm.connect()
        arm.enable()
        try:
            for i in range(cycles):
                print(f"cycle {i+1}/{cycles}: open {open_m*1000:.0f}mm")
                arm.command_gripper(open_m, effort_sdk)
                time.sleep(2.0)
                print(f"  feedback: {arm.read_gripper()[0]*1000:.1f}mm")
                print(f"cycle {i+1}/{cycles}: close {closed_m*1000:.0f}mm")
                arm.command_gripper(closed_m, effort_sdk)
                time.sleep(2.0)
                print(f"  feedback: {arm.read_gripper()[0]*1000:.1f}mm")
            arm.command_gripper(open_m, effort_sdk)
            time.sleep(1.0)
        finally:
            arm.disconnect()


if __name__ == "__main__":
    tyro.cli(main)
