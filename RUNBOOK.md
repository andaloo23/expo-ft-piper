# Runbook: zero → online RL on the left Piper arm

All commands from the repo root (`~/roboorchard-work/expo-ft-piper`).
Two environments: **actor** = `source piper_client/.venv/bin/activate`,
**learner** = `source .venv/bin/activate`. Every motion command requires
`--enable-motion` + typing `yes`; keep the e-stop in reach.

## A. Free the rig (once per session)

```bash
# 1. Confirm nobody is using the teleop stack, move followers to a low safe
#    pose (they go limp when controllers die), then:
docker exec holobrain bash -lc "pkill -f piper_dagger_compat.launch.py; pkill -f single_ctrl"

# 2. Sanity: CAN up, nothing else owns the buses
ip -br link show | grep can          # 4 interfaces UP
pgrep -af single_ctrl                # should print nothing
```

After a reboot instead: `sudo scripts/hardware/setup_can.sh` (names + raises the buses).

## B. Hardware checks (actor venv, ~15 min, first time only)

```bash
source piper_client/.venv/bin/activate

python scripts/hardware/read_state.py --can can_left        # joints/FK stream, Ctrl+C
python scripts/hardware/view_cameras.py                     # all 3 cameras stream? q to quit
python scripts/hardware/test_gripper.py --can can_left --enable-motion
python scripts/hardware/test_joint_motion.py --task configs/task/piper_pick.py          # preview
python scripts/hardware/test_joint_motion.py --task configs/task/piper_pick.py --enable-motion
python scripts/hardware/test_cartesian_delta.py --axis z --distance_m 0.02 --enable-motion
```

Gate: motion smooth, no faults → proceed. If the reset pose or workspace
bounds look wrong, fix `configs/task/piper_pick.py` before continuing.

## C. Collect demonstrations (actor venv)

```bash
bash scripts/piper_pick/collect_data.sh    # 15 successes by default
```

Drive the follower with the **left master arm**. Keys (no Enter):
`space` pause/resume · `t` takeover on/off · `s` success (saves) · `f` failure (discards).
Episodes land in `/data2/expo-ft-piper/raw/pick/success/<id>/traj.hdf5`.
Start with 10–15 good episodes.

## D. Offline: convert + SFT (learner venv, no rig needed)

```bash
source .venv/bin/activate
bash scripts/piper_pick/convert_data.sh      # HDF5 → LeRobot
bash scripts/piper_pick/calculate_norm.sh    # OpenPI norm stats
bash scripts/piper_pick/finetune_pi05.sh     # Pi0.5 SFT on the 5090 (hours)
```

Checkpoints: `/data2/expo-ft-piper/checkpoints/.../piper_pick_cube_10_lora_sft/<step>/params`.
If your episode count ≠ 10, edit `MAX_EPISODES`/`DATA_ID` in these scripts
and the `pi05_weight_loader_path` step number in `run_server.sh`/`eval_policy.sh`.

## E. Evaluate (two terminals)

```bash
# T1 (actor) — shadow first: full pipeline, arm never moves
bash scripts/piper_pick/run_client.sh                      # dry-run default
# T2 (learner)
bash scripts/piper_pick/eval_policy.sh
```

Shadow looks sane (no IK rejections spamming, actions plausible) →
restart T1 with motion for the real eval:

```bash
bash scripts/piper_pick/run_client.sh --enable-motion      # type yes
```

During eval: grab the master arm any time to take over; `space` to pause;
`s`/`f` to end + label an episode.

## F. Online RL (two terminals)

```bash
# T1 (actor, motion on)
bash scripts/piper_pick/run_client.sh --enable-motion
# T2 (learner) — synchronous EXPO-FT, batch 16
bash scripts/piper_pick/run_server.sh
```

Your interventions are stored as human actions (DAgger-style); label every
episode end with `s`/`f` — rewards come only from those labels.
Updates begin after 10 episodes are in the buffer. Resume any time: the
scripts pass `--resume`.

## Recovery

- Anything weird → `space` (hold) or e-stop; the servo also auto-holds if
  commands stop for 0.25 s.
- Actor refuses to start: another process owns the CAN lock
  (`/tmp/expo_ft_piper_can_left.lock`) — check `pgrep -af single_ctrl`.
- Give the rig back to HoloBrain: stop the actor (frees locks), then
  `bash ~/roboorchard-work/restart_piper_gravity.sh`.
