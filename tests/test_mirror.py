import numpy as np
import pytest

from piper_client.control.safety import SafetyLimits
from piper_client.hardware.kinematics import PiperKinematics
from piper_client.teleop.mirror import executed_action_from_obs

URDF = "assets/piper_x/urdf/piper_x_description_dualarm_v2.urdf"
LIMITS = SafetyLimits(
    max_linear_velocity_mps=0.10,
    max_angular_velocity_radps=0.50,
    max_gripper_velocity_mps=0.05,
)


def obs(cart, grip):
    return {
        "cartesian_position": np.asarray(cart, dtype=np.float32),
        "gripper_position": np.asarray([grip], dtype=np.float32),
    }


def test_pure_translation_recovers_velocity():
    o0 = obs([0.30, 0.0, 0.20, -2.18, 0.0, -1.57], 0.0)
    o1 = obs([0.30, 0.0, 0.205, -2.18, 0.0, -1.57], 0.0)  # +5 mm z in 0.1 s = 0.05 m/s
    a, sat, _ = executed_action_from_obs(o0, o1, 0.1, LIMITS, 0.07)
    assert not sat
    assert a[2] == pytest.approx(0.05 / 0.10, abs=1e-4)
    np.testing.assert_allclose(a[[0, 1, 3, 4, 5, 6]], 0.0, atol=1e-4)


def test_saturation_flagged_and_clipped():
    o0 = obs([0.30, 0.0, 0.20, -2.18, 0.0, -1.57], 0.0)
    o1 = obs([0.35, 0.0, 0.20, -2.18, 0.0, -1.57], 0.0)  # 0.5 m/s >> limit
    a, sat, _ = executed_action_from_obs(o0, o1, 0.1, LIMITS, 0.07)
    assert sat
    assert a[0] == 1.0


def test_gripper_close_direction():
    o0 = obs([0.3, 0, 0.2, -2.18, 0, -1.57], 0.0)   # open
    o1 = obs([0.3, 0, 0.2, -2.18, 0, -1.57], 0.5)   # half closed after 0.1 s
    a, _, _ = executed_action_from_obs(o0, o1, 0.1, LIMITS, 0.07)
    assert a[6] > 0  # positive closes


def test_roundtrip_with_fk():
    # FK poses from two nearby joint configs must produce a small, finite action.
    kin = PiperKinematics(URDF, "left_base_link", "left_gripper_base", "left")
    q0 = np.array([0.0, 1.35, -0.65, 0.0, 0.0, 0.0])
    q1 = q0 + np.array([0.002, 0.002, -0.002, 0.0, 0.002, 0.0])
    o0, o1 = obs(kin.fk(q0), 0.2), obs(kin.fk(q1), 0.2)
    a, sat, _ = executed_action_from_obs(o0, o1, 0.1, LIMITS, 0.07)
    assert np.isfinite(a).all() and not sat
