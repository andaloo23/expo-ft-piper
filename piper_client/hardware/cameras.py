"""Direct pyrealsense2 capture, addressed by serial number (never /dev/video*).

Each camera runs a background thread that keeps only the latest RGB frame
plus its wall-clock receive time, so observation reads never block on the
camera pipeline.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)


def resize_for_obs(img_rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize with pad to (height, width, 3) uint8 — same transform openpi uses."""
    from openpi_client import image_tools

    return np.asarray(image_tools.resize_with_pad(img_rgb, height, width), dtype=np.uint8)


class RealSenseCamera:
    """Latest-frame RGB capture from one RealSense, selected by serial."""

    def __init__(self, serial: str, width: int = 640, height: int = 360, fps: int = 30, name: str = ""):
        self.serial = str(serial)
        self.name = name or self.serial
        self.width, self.height, self.fps = width, height, fps
        self._frame: np.ndarray | None = None
        self._frame_time: float = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pipeline = None

    def start(self, first_frame_timeout_s: float = 4.0):
        """Open the pipeline and verify frames actually flow. RealSense
        devices (D455 especially) can wedge after rapid open/close cycles —
        pipeline.start() succeeds but no frames ever arrive. On a silent
        camera we issue a hardware reset and retry once before failing."""
        import pyrealsense2 as rs

        for attempt in (1, 2):
            self._pipeline = self._open_pipeline()
            if self._first_frame_arrives(first_frame_timeout_s):
                break
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
            if attempt == 2:
                raise RuntimeError(f"camera {self.serial}: no frames after hardware reset")
            logger.warning("%s: pipeline silent — issuing hardware reset", self.serial)
            for dev in rs.context().query_devices():
                if dev.get_info(rs.camera_info.serial_number) == self.serial:
                    dev.hardware_reset()
            time.sleep(6.0)  # give the device time to re-enumerate

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"rs-{self.name}")
        self._thread.start()
        logger.info("camera %s (%s) started", self.name, self.serial)

    def _open_pipeline(self):
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        try:
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
            pipeline.start(config)
        except RuntimeError:
            # Not every model supports every color profile (e.g. D405 vs D455);
            # fall back to the device default and resize downstream.
            logger.warning("%s: %dx%d@%d color unsupported, falling back to default profile",
                           self.serial, self.width, self.height, self.fps)
            config = rs.config()
            config.enable_device(self.serial)
            config.enable_stream(rs.stream.color, rs.format.rgb8, self.fps)
            pipeline.start(config)
        return pipeline

    def _first_frame_arrives(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                if frames.get_color_frame():
                    return True
            except RuntimeError:
                pass
        return False

    def _loop(self):
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=2000)
                color = frames.get_color_frame()
                if not color:
                    continue
                img = np.asanyarray(color.get_data())  # already RGB (rs.format.rgb8)
                with self._lock:
                    self._frame = img
                    self._frame_time = time.monotonic()
            except RuntimeError as e:
                logger.warning("%s: frame wait failed: %s", self.serial, e)

    def get_latest(self) -> tuple[np.ndarray | None, float]:
        """(latest RGB frame or None, monotonic receive time)."""
        with self._lock:
            return (None, 0.0) if self._frame is None else (self._frame.copy(), self._frame_time)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None


def list_connected_serials() -> list[dict]:
    """[{serial, name}] for all connected RealSense devices."""
    import pyrealsense2 as rs

    out = []
    for dev in rs.context().query_devices():
        out.append({
            "serial": dev.get_info(rs.camera_info.serial_number),
            "name": dev.get_info(rs.camera_info.name),
        })
    return out
