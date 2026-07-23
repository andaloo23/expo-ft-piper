import numpy as np
import pytest

from piper_client.control.calibration import (
    CalibrationError,
    deflection_from_torque,
    load_deflection_calibration,
    split_gravity_torque,
)

STORE = "assets/piper_x/calibration/deflection_calibrations.json"


def test_load_kp25_both_sides():
    for side in ("left", "right"):
        stiffness, tables = load_deflection_calibration(STORE, side, 25)
        assert stiffness.shape == (6,)
        assert stiffness[1] > 100 and stiffness[2] > 100  # J2/J3 calibrated
        assert tables[1] is not None                      # J2 table present
        assert all(t is None for t in (tables[0], tables[4], tables[5]))


def test_uncalibrated_kp_rejected():
    with pytest.raises(CalibrationError, match="calibrated kp values"):
        load_deflection_calibration(STORE, "left", 99)


def test_split_no_overflow_passthrough():
    t_ref, offset = split_gravity_torque(5.0, 8.0, kp=25.0, offset_stiffness=232.0)
    assert t_ref == 5.0 and offset == 0.0


def test_split_overflow_to_stiffness_offset():
    t_ref, offset = split_gravity_torque(10.0, 8.0, kp=25.0, offset_stiffness=200.0)
    assert t_ref == 8.0
    assert offset == pytest.approx(2.0 / 200.0)
    # Negative torque mirrors.
    t_ref, offset = split_gravity_torque(-10.0, 8.0, kp=25.0, offset_stiffness=200.0)
    assert t_ref == -8.0 and offset == pytest.approx(-0.01)


def test_split_overflow_prefers_table():
    _, tables = load_deflection_calibration(STORE, "left", 25)
    t_ref, offset = split_gravity_torque(10.0, 8.0, kp=25.0,
                                         offset_stiffness=232.0,
                                         deflection_table=tables[1])
    assert t_ref == 8.0
    expected = deflection_from_torque(tables[1], 2.0)
    assert offset == pytest.approx(expected)
    assert 0.0 < offset < 0.05  # a few-mrad correction, not garbage


def test_deflection_interp_monotone():
    _, tables = load_deflection_calibration(STORE, "left", 25)
    t = np.linspace(0.1, 12.0, 40)
    d = np.array([deflection_from_torque(tables[1], x) for x in t])
    assert (np.diff(d) >= -1e-12).all()
    assert deflection_from_torque(tables[1], -3.0) == -deflection_from_torque(tables[1], 3.0)
