"""Loader for the Piper MIT deflection-calibration store.

Same store format as roboorchard-dev (deflection_calibrations.json v2,
the rig's single source of truth for the kp-offset compensation path):
side -> str(kp) -> {offset_stiffness: [6], deflection_table: {joint_no:
[[|torque|, |deflection|], ...]}}. Entries are only valid at their own
kp (k_eff depends on kp); an uncalibrated kp is rejected, matching the
upstream controllers' behavior.
"""

from __future__ import annotations

import json

import numpy as np


class CalibrationError(RuntimeError):
    pass


def load_deflection_calibration(path: str, side: str, kp: float):
    """Returns (offset_stiffness[6], tables) where tables is a list of 6
    entries: None or a sorted [(|torque|, |deflection|), ...] list."""
    with open(path) as f:
        store = json.load(f)
    side_entry = store.get(side)
    if not isinstance(side_entry, dict):
        raise CalibrationError(f"No '{side}' section in {path}")
    key = None
    for k in side_entry:
        try:
            if float(k) == float(kp):
                key = k
                break
        except ValueError:
            continue
    if key is None:
        available = sorted(k for k in side_entry if k.replace('.', '', 1).isdigit())
        raise CalibrationError(
            f"No deflection calibration for kp={kp:g} (side {side}) in {path}; "
            f"calibrated kp values: {available}. Re-measure or use one of those."
        )
    entry = side_entry[key]
    stiffness = np.asarray(entry.get("offset_stiffness", [0.0] * 6), dtype=np.float64)
    if stiffness.shape != (6,):
        raise CalibrationError(f"{path}: {side}/{key} offset_stiffness must have 6 values")
    tables: list = [None] * 6
    for joint_no, knots in entry.get("deflection_table", {}).items():
        idx = int(joint_no) - 1  # store keys are 1-based joint numbers
        if not 0 <= idx < 6:
            raise CalibrationError(f"{path}: bad deflection_table joint {joint_no}")
        pts = sorted((float(t), float(d)) for t, d in knots)
        if len(pts) >= 2:
            tables[idx] = pts
    return stiffness, tables


def deflection_from_torque(table, torque: float) -> float:
    """|torque| -> |deflection| via linear interpolation over the calibrated
    knots (last-segment slope extrapolation), sign following the torque.
    Upstream uses a monotone hermite spline; linear interp over the same
    knots differs by well under a millimeter of EE motion."""
    a = abs(float(torque))
    sign = 1.0 if torque >= 0.0 else -1.0
    xs = [p[0] for p in table]
    ys = [p[1] for p in table]
    if a >= xs[-1]:
        slope = (ys[-1] - ys[-2]) / max(xs[-1] - xs[-2], 1e-9)
        return sign * (ys[-1] + (a - xs[-1]) * slope)
    return sign * float(np.interp(a, xs, ys))


def split_gravity_torque(
    desired_torque: float,
    max_abs_t_ref: float,
    kp: float,
    offset_stiffness: float = 0.0,
    deflection_table=None,
) -> tuple[float, float]:
    """Port of upstream _split_gravity_torque (ros_bridge.py): clamp the
    torque-channel feedforward, convert ONLY the clamped-off overflow into
    a position offset (deflection table > offset_stiffness > kp).

    Returns (t_ref_to_send, position_offset_rad).
    """
    t_ref = float(np.clip(desired_torque, -max_abs_t_ref, max_abs_t_ref))
    overflow = float(desired_torque) - t_ref
    if overflow == 0.0:
        return t_ref, 0.0
    if deflection_table:
        offset = deflection_from_torque(deflection_table, overflow)
    else:
        stiffness = offset_stiffness if offset_stiffness > 0.0 else kp
        offset = overflow / stiffness if stiffness > 0.0 else 0.0
    return t_ref, offset
