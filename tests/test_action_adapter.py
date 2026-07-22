import numpy as np
import pytest

from piper_client.control.action_adapter import ActionAdapter
from piper_client.control.safety import SafetyLimits
from piper_client.hardware.gripper import GripperModel
from piper_client.hardware.kinematics import PiperKinematics

URDF = "assets/piper_x/urdf/piper_x_description_dualarm_v2.urdf"
POLICY_HZ = 10.0
# A mid-workspace elbow-folded configuration (away from limits/singularities).
Q_HOME = np.array([0.0, 0.6, -1.4, 0.0, 0.8, 0.0])


@pytest.fixture(scope="module")
def kin():
    return PiperKinematics(URDF, "left_base_link", "left_gripper_base", "left")


def make_adapter(kin, bounds=None):
    limits = SafetyLimits(
        max_linear_velocity_mps=0.03,
        max_angular_velocity_radps=0.20,
        max_gripper_velocity_mps=0.02,
        max_joint_delta_rad=0.05,
        workspace_bounds=bounds,
    )
    adapter = ActionAdapter(kin, GripperModel(0.07), limits, policy_hz=POLICY_HZ)
    adapter.reset(Q_HOME, 0.07)
    return adapter


def test_fk_roundtrip_via_ik(kin):
    pose = kin.fk(Q_HOME)
    q, converged, err = kin.ik(pose, Q_HOME + 0.01)
    assert converged, f"IK failed, err={err}"
    np.testing.assert_allclose(kin.fk(q), pose, atol=1e-3)


def test_zero_action_holds_pose(kin):
    adapter = make_adapter(kin)
    pose0 = adapter.commanded_pose6
    res = adapter.compute(np.zeros(7))
    np.testing.assert_allclose(res.target_pose6[:3], pose0[:3], atol=1e-4)
    np.testing.assert_allclose(res.executed_action, 0.0, atol=0.05)


def test_translation_moves_expected_distance(kin):
    adapter = make_adapter(kin)
    pose0 = adapter.commanded_pose6
    res = adapter.compute([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # +z at full scale
    assert res.ik_converged
    dz = res.target_pose6[2] - pose0[2]
    # 0.03 m/s over 0.1 s = 3 mm
    assert dz == pytest.approx(0.003, abs=5e-4)
    assert res.executed_action[2] == pytest.approx(1.0, abs=0.1)


def test_workspace_bound_blocks_motion(kin):
    pose0 = PiperKinematics(URDF, "left_base_link", "left_gripper_base", "left").fk(Q_HOME)
    # Bounds that end exactly at the current x: +x motion must be blocked.
    bounds = np.array([
        [pose0[0] - 0.2, pose0[0]],
        [pose0[1] - 0.2, pose0[1] + 0.2],
        [pose0[2] - 0.2, pose0[2] + 0.2],
    ])
    adapter = make_adapter(kin, bounds=bounds)
    res = adapter.compute([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert res.target_pose6[0] <= pose0[0] + 1e-9
    assert res.executed_action[0] == pytest.approx(0.0, abs=0.05)


def test_nan_action_is_neutralized(kin):
    adapter = make_adapter(kin)
    pose0 = adapter.commanded_pose6
    res = adapter.compute([np.nan, np.inf, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(res.target_pose6[:3], pose0[:3], atol=1e-4)


def test_joint_delta_never_exceeded(kin):
    adapter = make_adapter(kin)
    q_prev = Q_HOME.copy()
    rng = np.random.default_rng(0)
    for _ in range(20):
        res = adapter.compute(rng.uniform(-1, 1, size=7))
        assert np.abs(res.q_target - q_prev).max() <= adapter.limits.max_joint_delta_rad + 1e-9
        q_prev = res.q_target


def test_gripper_integration(kin):
    adapter = make_adapter(kin)
    res = adapter.compute([0, 0, 0, 0, 0, 0, 1.0])  # close at full speed
    # 0.02 m/s * 0.1 s = 2 mm of closing from fully open (0.07)
    assert res.gripper_opening_m == pytest.approx(0.068, abs=1e-6)
    assert res.executed_action[6] == pytest.approx(1.0, abs=1e-6)
    # Opening beyond max clamps and reports reduced executed action
    adapter.reset(Q_HOME, 0.07)
    res = adapter.compute([0, 0, 0, 0, 0, 0, -1.0])
    assert res.gripper_opening_m == pytest.approx(0.07, abs=1e-9)
    assert res.executed_action[6] == pytest.approx(0.0, abs=1e-6)
