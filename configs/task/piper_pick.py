"""Pick task: Piper real robot (left follower arm), pick up the cube."""

import numpy as np

from configs.task import piper_base

try:
    from piper_client.envs.pick import PiperPickEnv
except Exception:
    print("Not importing piper env [module]")


def get_config():
    config = piper_base.get_config()

    try:
        config.env = PiperPickEnv
    except Exception:
        print("Not importing piper env [env]")

    config.env_name = "piper_pick"
    config.language_instruction = "pick up the cube"
    config.hardware_config = "configs/hardware/gail8_left.yaml"

    # Workspace over the table in the left_base_link frame. PLACEHOLDER —
    # verify against the physical rig (table at base plane z=0; keep the
    # gripper above z≈0.05) before enabling motion.
    config.bounds = np.array([
        [0.15, 0.55],    # x
        [-0.25, 0.25],   # y
        [0.05, 0.45],    # z
    ])

    # Nominal manipulation-ready pose (6, rad): FK puts the EE at
    # [0.30, 0.00, 0.20] in left_base_link, gripper pitched forward-down.
    # PLACEHOLDER — preview with scripts/hardware/test_joint_motion.py
    # before enabling motion.
    config.reset_joints = np.array([0.0, 1.35, -0.65, 0.0, 0.0, 0.0])

    config.auto_reset_steps = 80

    # Pick task: no rotation needed; residual only on xyz and gripper
    config.residual_action_xyzg = True

    # Success labeling: manual (terminal 1/2) until the pipeline works end-to-end.
    config.use_auto_detector = False
    config.pick_lift_z_m = 0.20

    return config
