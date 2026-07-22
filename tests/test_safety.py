import multiprocessing
import os

import numpy as np
import pytest

from piper_client.control.safety import (
    CanLock,
    SafetyError,
    SafetyLimits,
    clamp_position_to_workspace,
    filter_joint_target,
    sanitize_normalized_action,
)


def test_sanitize_clamps_and_replaces_nonfinite():
    a = sanitize_normalized_action([2.0, -3.0, np.nan, np.inf, 0.5, -0.5, 0.0])
    assert a.shape == (7,)
    assert np.all(a <= 1.0) and np.all(a >= -1.0)
    assert a[2] == 0.0 and a[3] == 0.0
    assert a[0] == 1.0 and a[1] == -1.0


def test_sanitize_rejects_wrong_dim():
    with pytest.raises(SafetyError):
        sanitize_normalized_action([0.0] * 6)


def test_workspace_clamp():
    bounds = np.array([[0.1, 0.5], [-0.2, 0.2], [0.05, 0.4]])
    pos = clamp_position_to_workspace(np.array([0.9, -0.9, 0.2]), bounds)
    np.testing.assert_allclose(pos, [0.5, -0.2, 0.2])
    # No bounds -> passthrough
    np.testing.assert_allclose(clamp_position_to_workspace(np.array([9.0, 9.0, 9.0]), None), 9.0)


def test_filter_joint_target_delta_and_limits():
    limits = SafetyLimits(max_joint_delta_rad=0.02, joint_limit_margin_rad=0.01)
    lower, upper = -np.ones(6), np.ones(6)
    q_ref = np.zeros(6)

    # Large jump is clamped to max delta
    q = filter_joint_target(np.full(6, 0.5), q_ref, lower, upper, limits)
    np.testing.assert_allclose(q, 0.02)

    # Target beyond joint limit is clamped inside limit (then delta-limited)
    q_ref2 = np.full(6, 0.985)
    q2 = filter_joint_target(np.full(6, 2.0), q_ref2, lower, upper, limits)
    assert np.all(q2 <= 0.99 + 1e-12)


def test_can_lock_is_exclusive(tmp_path):
    lock = CanLock("test_can", lock_dir=str(tmp_path))
    lock.acquire()
    try:
        # A second process must be refused while we hold the lock.
        ctx = multiprocessing.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_try_lock_path, args=(str(tmp_path), q))
        p.start()
        p.join(timeout=10)
        assert q.get(timeout=5) == "blocked"
    finally:
        lock.release()

    # After release it can be acquired again.
    lock2 = CanLock("test_can", lock_dir=str(tmp_path))
    lock2.acquire()
    lock2.release()


def _try_lock_path(lock_dir, q):
    try:
        lock = CanLock("test_can", lock_dir=lock_dir)
        lock.acquire()
        q.put("acquired")
        lock.release()
    except SafetyError:
        q.put("blocked")
