"""The HDF5 the actor writes must round-trip through the learner's loader."""

import types

import numpy as np

from expo_ft.env.robot_dataset import process_robot_hdf5_dataset
from piper_client.recording.hdf5_writer import HDF5TrajWriter


def _write_episode(path, T=6):
    writer = HDF5TrajWriter(str(path))
    img = np.zeros((180, 320, 3), dtype=np.uint8)
    for t in range(T):
        writer.write_timestep({
            "saved_observation": {
                "exterior_image_1_left": img,
                "exterior_image_2_left": img,
                "wrist_image_left": img,
                "cartesian_position": np.full(6, t, dtype=np.float32),
                "gripper_position": np.array([0.5], dtype=np.float32),
                "prompt": "pick up the cube",
            },
            "action": {
                "cartesian_velocity": np.full(6, 0.1 * t, dtype=np.float32),
                "gripper_velocity": np.float32(0.2),
            },
        })
    writer.close()


def test_writer_output_loads_in_learner(tmp_path):
    T = 6
    for ep in range(2):
        ep_dir = tmp_path / str(ep)
        ep_dir.mkdir()
        _write_episode(ep_dir / "traj.hdf5", T=T)
    # non-numeric dirs must be skipped
    (tmp_path / "lerobot").mkdir()

    task_config = types.SimpleNamespace(
        action_space="cartesian_velocity",
        gripper_action_space="velocity",
    )
    data = process_robot_hdf5_dataset(str(tmp_path), task_config)

    assert len(data) == 2 * T
    first, last = data[0], data[T - 1]

    # 7D action = 6D cartesian velocity + gripper velocity
    assert first["actions"].shape == (7,)
    assert first["actions"][6] == np.float32(0.2)

    obs = first["observations"]
    assert obs["exterior_image_1_left"].shape == (180, 320, 3)
    assert obs["exterior_image_1_left"].dtype == np.uint8
    assert obs["wrist_image_left"].shape == (180, 320, 3)
    assert obs["cartesian_position"].shape == (6,)
    assert obs["gripper_position"].shape == (1,)
    assert str(obs["prompt"]) == "pick up the cube"

    # Terminal-reward structure: reward/done only on the last step of an episode
    assert first["rewards"] == 0.0 and first["dones"] == 0.0 and first["masks"] == 1.0
    assert last["rewards"] == 1.0 and last["dones"] == 1.0 and last["masks"] == 0.0
