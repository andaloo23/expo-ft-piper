"""Task config that builds a FakeEnv — used by the WebSocket round-trip test."""

import ml_collections
import numpy as np

from tests.fakes.fake_env import FakeEnv


def get_config():
    config = ml_collections.ConfigDict()
    config.env_type = "piper"
    config.env = FakeEnv
    config.env_name = "fake_pick"
    config.language_instruction = "pick up the cube"
    config.action_space = "cartesian_velocity"
    config.gripper_action_space = "velocity"
    config.control_hz = 10
    config.auto_reset_steps = 5
    config.example_action = np.array([[0.0] * 7])
    config.residual_action_xyzg = False
    return config
