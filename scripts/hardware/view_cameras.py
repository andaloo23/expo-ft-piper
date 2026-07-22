"""Step 2 of bring-up: RealSense capture by serial number.

Lists connected devices, then shows a live window per serial so you can
identify which D405 is the left wrist camera. Update the `wrist.serial`
field in configs/hardware/gail8_left.yaml accordingly. Never configure
cameras by /dev/video* index — those change across reboots.

    python scripts/hardware/view_cameras.py                # all detected devices
    python scripts/hardware/view_cameras.py --serials 327323070980,402323072095
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import tyro

from piper_client.hardware.cameras import RealSenseCamera, list_connected_serials


def main(serials: str = "", width: int = 640, height: int = 360, fps: int = 30):
    devices = list_connected_serials()
    print("Connected RealSense devices:")
    for d in devices:
        print(f"  {d['serial']}  {d['name']}")
    wanted = [s.strip() for s in serials.split(",") if s.strip()] or [d["serial"] for d in devices]

    cams = []
    for s in wanted:
        cam = RealSenseCamera(s, width, height, fps, name=s)
        cam.start()
        cams.append(cam)

    print("Press q to quit. Wave a hand in front of each camera to identify it.")
    try:
        while True:
            for cam in cams:
                frame, _ = cam.get_latest()
                if frame is not None:
                    cv2.imshow(f"{cam.serial}", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    finally:
        for cam in cams:
            cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tyro.cli(main)
