"""Base task config for Piper real-robot tasks (mirrors real_base.py for DROID)."""

import ml_collections
import numpy as np

def get_config():
    config = ml_collections.ConfigDict()

    config.env_type = "piper"

    # Action stream names used when loading and converting datasets.
    config.action_space = "cartesian_velocity"
    config.gripper_action_space = "velocity"

    # Which arm / cameras / kinematics to use (see configs/hardware/).
    config.hardware_config = "configs/hardware/gail8_left.yaml"

    # Cartesian workspace bounds (3x2, base frame) and joint reset pose (6, rad);
    # override per task. VERIFY on the physical rig before enabling motion.
    config.bounds = None
    config.reset_joints = None

    # Observation image resize and control loop frequency.
    config.image_size = (180, 320)
    config.control_hz = 10

    # Episode step budget before auto-reset.
    config.auto_reset_steps = 80
    config.ignore_auto_reset = False

    # Master-arm human override during online RL.
    config.use_master_intervention = True

    # Safety: motion off unless the actor is launched with --enable-motion.
    config.dry_run = True
    config.enable_motion = False

    # Action-shaped zero used for env/replay-buffer initialization.
    config.example_action = np.array([[0., 0., 0., 0., 0., 0., 0.]])

    # Residual policy: when True, only apply residual to xyz and gripper (not rotation).
    config.residual_action_xyzg = False

    return config
