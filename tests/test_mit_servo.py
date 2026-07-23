import numpy as np
import pytest

from piper_client.control.mit_servo import MitServoLoop
from piper_client.hardware.kinematics import PiperKinematics

URDF = "assets/piper_x/urdf/piper_x_description_dualarm_v2.urdf"


class FakeMitArm:
    can_name = "fake"
    read_only = False

    def __init__(self):
        self.mit_calls = []
        self.mode_calls = []
        self.gripper_calls = []

    def set_mit_motion_mode(self):
        self.mode_calls.append("mit")

    def set_joint_motion_mode(self, speed_rate=100):
        self.mode_calls.append("pos")

    def command_joints_mit(self, q, v, kp, kd, t):
        self.mit_calls.append((np.array(q), np.array(v), kp, kd, np.array(t)))

    def command_joints(self, q):
        pass

    def command_gripper(self, opening_m, effort):
        self.gripper_calls.append(opening_m)

    def read_joints(self):
        return np.zeros(6), 0.0


@pytest.fixture(scope="module")
def kin():
    return PiperKinematics(URDF, "left_base_link", "left_gripper_base", "left")


def test_gravity_torques_shape_and_dominance(kin):
    # Elbow-folded reach pose: shoulder pitch (J2) carries the most load.
    tau = kin.gravity_torques(np.array([0.0, 1.35, -0.65, 0.0, 0.0, 0.0]))
    assert tau.shape == (6,)
    assert np.isfinite(tau).all()
    assert np.argmax(np.abs(tau)) in (1, 2)  # shoulder/elbow carry the load
    # Wrist joints see near-zero gravity torque in this pose.
    assert np.abs(tau[[4, 5]]).max() < 0.5


def test_mit_send_clamps_and_feeds_forward(kin):
    arm = FakeMitArm()
    servo = MitServoLoop(arm, kin, servo_hz=100, dry_run=False,
                         kp=25.0, kd=0.8, t_ref_clamp_nm=8.0,
                         gravity_scale=[1.0, 1.09, 1.06, 0.88, 1.0, 1.0])
    q0 = np.array([0.0, 1.35, -0.65, 0.0, 0.0, 0.0])
    servo._send(q0, 0.07)
    q1 = q0 + 0.01
    servo._send(q1, 0.07)

    assert arm.mode_calls[:2] == ["mit", "mit"]
    (qa, va, kp, kd, ta) = arm.mit_calls[0]
    assert kp == 25.0 and kd == 0.8
    np.testing.assert_allclose(qa, q0)
    np.testing.assert_allclose(va, 0.0)          # first tick: no velocity yet
    assert np.abs(ta).max() <= 8.0 + 1e-9        # clamp mandatory (aliasing)

    (_, vb, _, _, tb) = arm.mit_calls[1]
    np.testing.assert_allclose(vb, 0.01 / servo.dt, atol=1e-9)  # 1 rad/s ff
    assert np.abs(tb).max() <= 8.0 + 1e-9
    # Gravity feedforward is actually present (J2 holds the arm up).
    assert abs(tb[1]) > 1.0
    assert arm.gripper_calls == [0.07, 0.07]


def test_stop_reverts_to_position_mode(kin):
    arm = FakeMitArm()
    servo = MitServoLoop(arm, kin, servo_hz=100, dry_run=False)
    servo.start(np.zeros(6), 0.07)
    servo.stop()
    assert "pos" in arm.mode_calls  # reverted, not left soft
