"""Backward-compatible shim: the loader moved to expo_ft.env.robot_dataset."""

from expo_ft.env.robot_dataset import (  # noqa: F401
    _discover_episode_dirs,
    process_droid_dataset,
    process_robot_hdf5_dataset,
)
