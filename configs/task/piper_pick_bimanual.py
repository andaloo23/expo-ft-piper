"""Bimanual demo collection: both Piper arms, both masters, three cameras."""

import numpy as np

from configs.task import piper_base

try:
    from piper_client.envs.bimanual import BimanualPiperEnv
except Exception:
    print("Not importing bimanual piper env [module]")


def get_config():
    config = piper_base.get_config()

    try:
        config.env = BimanualPiperEnv
    except Exception:
        print("Not importing bimanual piper env [env]")

    config.env_type = "piper_bimanual"
    config.env_name = "piper_pick_bimanual"
    config.language_instruction = "pick up the cube"
    config.hardware_config = "configs/hardware/gail8_dual.yaml"

    # Wide envelope (left_base_link frame): engage gate + logging only.
    config.bounds = np.array([
        [0.02, 0.65],
        [-0.40, 0.40],
        [0.01, 0.60],
    ])

    # Per-arm reset poses. Right arm is mirrored hardware — VERIFY with a
    # preview (scripts/hardware/test_joint_motion.py on the right yaml)
    # before first motion.
    config.reset_joints = np.array([0.0, 1.35, -0.65, 0.0, 0.0, 0.0])
    config.reset_joints_right = np.array([0.0, 1.35, -0.65, 0.0, 0.0, 0.0])

    config.ignore_auto_reset = True
    config.use_master_intervention = False

    return config
