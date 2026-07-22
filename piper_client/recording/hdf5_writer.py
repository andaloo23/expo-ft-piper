"""Streaming HDF5 trajectory writer in the EXPO-FT raw layout.

Structure (read by expo_ft.env.robot_dataset.process_robot_hdf5_dataset):

    saved_observation/
        exterior_image_1_left   uint8  [T, H, W, 3]
        exterior_image_2_left   uint8  [T, H, W, 3]
        wrist_image_left        uint8  [T, H, W, 3]
        cartesian_position      float32 [T, 6]
        gripper_position        float32 [T, 1] (or [T])
        prompt                  bytes   [T]
    action/
        cartesian_velocity      float32 [T, 6]
        gripper_velocity        float32 [T]
"""

from __future__ import annotations

import numpy as np

import h5py


class HDF5TrajWriter:
    def __init__(self, filepath: str):
        self._file = h5py.File(filepath, "w")
        self._datasets: dict[str, h5py.Dataset] = {}
        self._count = 0

    def _append(self, path: str, value):
        if isinstance(value, str):
            value = np.bytes_(value.encode("utf-8"))
        arr = np.asarray(value)
        if path not in self._datasets:
            maxshape = (None,) + arr.shape
            self._datasets[path] = self._file.create_dataset(
                path,
                shape=(0,) + arr.shape,
                maxshape=maxshape,
                dtype=arr.dtype,
                chunks=(1,) + arr.shape if arr.ndim else True,
            )
        ds = self._datasets[path]
        ds.resize(ds.shape[0] + 1, axis=0)
        ds[-1] = arr

    def write_timestep(self, timestep: dict):
        """timestep = {"saved_observation": {...}, "action": {...}} (nested dicts ok)."""

        def walk(prefix, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    walk(f"{prefix}/{k}", v)
                else:
                    self._append(f"{prefix}/{k}", v)

        for top, group in timestep.items():
            walk(top, group)
        self._count += 1

    @property
    def num_steps(self) -> int:
        return self._count

    def close(self):
        try:
            self._file.flush()
        finally:
            self._file.close()
