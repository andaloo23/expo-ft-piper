# EXPO-FT-Piper

Standalone port of [EXPO-FT](https://github.com/pd-perry/expo-ft) (sample-efficient RL
finetuning for VLAs) from the DROID/Franka platform to **AgileX Piper** arms
(dual-arm ALOHA rig on gail8). The Pi0.5 integration, EXPO learner, critics,
replay buffer, training loops, and WebSocket protocol are preserved from
upstream; only the robot-facing layer is replaced:

| DROID upstream        | This port                           |
|-----------------------|-------------------------------------|
| Polymetis / Franka    | piper_sdk (vendored, 0.6.1)         |
| ZED stereo cameras    | RealSense (pyrealsense2, by serial) |
| DROID `RobotEnv`      | `piper_client/` (envs, IK, servo, safety) |
| Spacemouse teleop     | Master Piper arms (can_*_mst)       |

The `upstream` git remote points at the original EXPO-FT for pulling fixes.
The rule: **expo_ft/ stays generic; every Piper-specific decision lives under
`piper_client/`, `configs/hardware/`, and the Piper dataset converter.**

## Architecture

```
RTX 5090 learner (.venv, GPU)
  Pi0.5 + EXPO-FT + replay buffer          train_pi_robo.py / eval_piper_policy.py
              │ WebSocket 127.0.0.1:8102 (same machine, no tunnel)
              ▼
Piper actor (piper_client/.venv, CPU only — CUDA_VISIBLE_DEVICES=)
  RealSense → observations                 piper_client/run_client.py
  piper_sdk → feedback & JointCtrl
  master Piper → demos / intervention
```

The actor serves the same five operations as upstream (`create_env`, `reset`,
`step`, `get_observation`, `get_info_for_step`), so the learner is nearly
unchanged (only the dataset dispatch gained a `piper` branch and the loader
was generalized to `expo_ft/env/robot_dataset.py`).

### Observation / action contract (single left arm)

Observations mirror the DROID env expected by the Pi0.5 wrapper (RGB, the
static image deliberately duplicated):

```python
{
  "exterior_image_1_left": uint8 [180,320,3],
  "exterior_image_2_left": uint8 [180,320,3],   # duplicate of 1
  "wrist_image_left":      uint8 [180,320,3],
  "cartesian_position":    float32 [6],          # xyz + rpy, left_base_link frame
  "gripper_position":      float32 [1],          # 0=open, 1=closed
  "prompt":                str,
}
```

Action: `[vx, vy, vz, wx, wy, wz, gripper_velocity]`, normalized [-1, 1].
Physical limits (configs/hardware/*.yaml): ±0.03 m/s translation, ±0.20 rad/s
rotation, ±0.02 m/s gripper; policy 10 Hz, servo 50 Hz. Demonstrations are
recorded with this same normalized contract so OpenPI normalization
statistics learn the Piper action distribution. `step()` returns the
**executed** (safety-filtered) action, which EXPO-FT stores in replay.

Control path (first version — firmware position mode + JointCtrl; no MIT
torque mode, no gravity feedforward, no joint-space VLA actions, no
EndPoseCtrl):

```
action → clamp → Cartesian velocity → integrate 0.1 s → workspace filter
       → Pinocchio IK → joint limit/delta filter → 50 Hz interpolation → JointCtrl
```

Unit conversions live only in `piper_client/hardware/arm.py`:
`sdk_joint = round(q_rad * 180/π * 1000)` (0.001°),
`sdk_gripper = round(opening_m * 1e6)` (0.001 mm).

## Setup

Two independent environments (same split as upstream):

```bash
# Learner (repo root; needs the openpi fork cloned first)
git clone -b expo_ft https://github.com/pd-perry/openpi.git expo_ft/agents/vla/openpi
uv sync

# Actor (CPU): uses vendored third_party/piper_sdk + openpi-client
cd piper_client && uv sync --extra test
```

Bulk data lives outside git in `/data2/expo-ft-piper/{raw,lerobot,checkpoints,runs,videos}`.

Raw demonstrations: `/data2/expo-ft-piper/raw/pick/success/<id>/traj.hdf5` with
`saved_observation/*` (the observation dict above) and
`action/{cartesian_velocity [T,6], gripper_velocity [T]}`.

## Safety

- Motion is **off by default**. `--dry-run` reads sensors and runs the full
  action pipeline without commanding; `--enable-motion` requires typing `yes`
  at launch.
- Exclusive CAN ownership via `/tmp/expo_ft_piper_<can>.lock` — the actor
  refuses to start if another process holds the bus. **Stop the
  ROS/HoloBrain teleop stack first** (it owns all four arms' CAN).
- NaN rejection, workspace bounds, joint limit + per-command delta clamps,
  IK convergence gating, sensor/command freshness watchdog, hold-position on
  command timeout. See `piper_client/control/safety.py`.
- Keep the physical e-stop reachable whenever motion is enabled.

## Bring-up order

1. `scripts/hardware/read_state.py` — read-only SDK state (never enables).
2. `scripts/hardware/view_cameras.py` — identify D405 serials, fill
   `wrist.serial` in `configs/hardware/gail8_left.yaml`.
3. `piper_client/.venv/bin/python -m pytest tests/` — fake-arm contract tests.
4. `scripts/hardware/test_gripper.py --enable-motion` — gripper only.
5. `scripts/hardware/test_joint_motion.py` — preview, then `--enable-motion`.
6. `scripts/hardware/test_cartesian_delta.py` — 1–2 cm through IK + JointCtrl.
7. `scripts/piper_pick/collect_data.sh` — master-arm demonstrations.
8. `scripts/piper_pick/convert_data.sh` + `calculate_norm.sh` — LeRobot + norms.
9. `scripts/piper_pick/finetune_pi05.sh` — Pi0.5-DROID init → Piper SFT.
10. Shadow eval: `run_client.sh` (dry-run) + `eval_policy.sh`.
11. Real eval: `run_client.sh --enable-motion` + `eval_policy.sh`.
12. Online RL: `run_client.sh --enable-motion` + `run_server.sh`
    (synchronous; batch 16–32 on the 5090; actor runs with the GPU hidden).

Only after all of that works: automatic resets, auto success detection
(`success/pick_detector.py`), or bimanual control.

## Repo layout

- `expo_ft/` — upstream learner, unchanged apart from
  `env/robot_dataset.py` (generalized HDF5 loader; `droid_utils.py` is a shim).
- `piper_client/` — the actor: `hardware/` (arm, cameras, Pinocchio
  kinematics, gripper), `control/` (action adapter, 50 Hz servo, safety,
  watchdog), `teleop/` (master arm, intervention), `envs/`, `recording/`,
  `success/`, `run_client.py`, `collect_data.py`.
- `configs/hardware/gail8_{left,right}.yaml`, `configs/task/piper_*.py`,
  `configs/model/` (unchanged upstream algorithm configs).
- `assets/piper_x/urdf/` — piper_x dual-arm URDF (frames
  `left_base_link`/`left_gripper_base`, joints `left_joint1..6`).
- `third_party/piper_sdk/` — vendored SDK 0.6.1 (interface 2, protocol 2,
  firmware S-V1.8-5); see `VENDORED_VERSION.txt`.
- `tests/` — adapter, safety, dataset-contract, and WebSocket round-trip
  tests against fakes (no hardware needed).
- Legacy DROID files (`client/`, `eval_droid_policy.py`,
  `configs/task/{pick,light2,real_base}.py`) are kept for upstream diffing.

## ROS

Not used, by design: direct piper_sdk for CAN, direct pyrealsense2, Pinocchio
for FK/IK, WebSocket to the learner, HDF5/LeRobot for data, files/W&B for
logging. If Foxglove visualization is wanted later, add a *read-only*
diagnostics process that is never in the command path and never owns a CAN
interface.
