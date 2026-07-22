# Working notes for agents

## What this repo is

Fork of pd-perry/expo-ft (remote `upstream`, branch `piper-port`) with the
DROID/Polymetis/ZED layer replaced by AgileX Piper + RealSense + Pinocchio.
Read README.md first — architecture, contracts, and bring-up order live there.

## Hard rules

- **Never** commit anything Piper-specific into `expo_ft/` — the learner must
  stay diffable against upstream. Robot decisions go in `piper_client/`,
  `configs/hardware/`, or `scripts/convert_piper_data_to_lerobot.py`.
- **Never** weaken safety defaults: motion is opt-in (`--enable-motion` +
  typed confirmation), CAN lockfiles are mandatory, the servo holds position
  on timeout. Any change to `piper_client/control/` needs the tests in
  `tests/` passing and a hardware re-validation note.
- Unit conversions (0.001° joints, 0.001 mm gripper) exist **only** in
  `piper_client/hardware/arm.py`. Everything else is rad / m / s.
- Cameras are addressed by RealSense serial, never /dev/video*.
- First version is firmware position mode + JointCtrl only: no MIT mode, no
  gravity feedforward, no EndPoseCtrl, no joint-space VLA actions.

## Environments

- Learner: `.venv` at repo root (`uv sync`; GPU, jax/openpi).
- Actor: `piper_client/.venv` (`cd piper_client && uv sync --extra test`;
  CPU only — launch the actor with `CUDA_VISIBLE_DEVICES=`).
- Tests (no hardware): `piper_client/.venv/bin/python -m pytest tests/`.

## Rig facts (gail8)

- Followers `can_left`/`can_right`, masters `can_left_mst`/`can_right_mst`.
  One process per bus; stop the ROS/HoloBrain teleop stack before running
  the actor (it owns all four buses).
- piper_sdk 0.6.1 vendored in `third_party/piper_sdk` (interface 2,
  protocol 2, firmware S-V1.8-5). Don't depend on ../piper_sdk.
- URDF: `assets/piper_x/urdf/piper_x_description_dualarm_v2.urdf`; the left
  arm base is at z=0 pointing up, table at the base plane. `bounds` and
  `reset_joints` in `configs/task/piper_pick.py` are placeholders until
  previewed on the rig (`scripts/hardware/test_joint_motion.py`).
- Static camera D455 serial 327743060187; the two D405s are 327323070980 and
  402323072095 — identify the wrist one with `scripts/hardware/view_cameras.py`
  and update `configs/hardware/gail8_left.yaml`.
- Bulk data: `/data2/expo-ft-piper/` (raw, lerobot, checkpoints, runs, videos).

## Current status / next steps

- Code + tests complete through bring-up step 3 (contract tests pass).
- Not yet done on hardware: wrist-camera identification, reset-pose /
  workspace-bounds preview, gripper test, first motion, demo collection.
- `origin` remote not set yet — create a GitHub repo and add it; keep
  `upstream` for fixes.
