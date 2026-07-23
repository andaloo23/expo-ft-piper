"""Master-arm demonstration collection for one Piper follower arm.

Teleop is ALOHA-style joint mirroring at servo rate (teleop/mirror.py):
the follower copies the master's joints 1:1, speed set by your hand,
rate-clamped for safety. Recording stays in the training contract: at
10 Hz the executed normalized Cartesian-velocity action is derived from
the follower's actual motion and written with the observation to
traj.hdf5 in the EXPO-FT raw layout. Successful episodes are routed to
<save_root>/success/<id>/traj.hdf5; failures are discarded.

Keys (no Enter): space=pause  s=success  f=failure.

Usage:
    python -m piper_client.collect_data \
        --save_root /data2/expo-ft-piper/raw/pick \
        --num_episodes 15 \
        --task_config configs/task/piper_pick.py \
        --enable_motion
"""

from __future__ import annotations

import os
import shutil
import time

import numpy as np
from absl import app, flags
from ml_collections import config_flags

from piper_client.recording.hdf5_writer import HDF5TrajWriter
from piper_client.teleop.master_arm import MasterArm
from piper_client.teleop.mirror import MirrorController, executed_action_from_obs

FLAGS = flags.FLAGS

flags.DEFINE_string("save_root", "/data2/expo-ft-piper/raw/pick",
                    "Root directory where collected trajectories will be stored.")
flags.DEFINE_integer("num_episodes", 0,
                     "Number of successful trajectories to collect. 0 = run indefinitely.")
config_flags.DEFINE_config_file("task_config", "configs/task/piper_pick.py",
                                "File path to the task configuration.", lock_config=False)
flags.DEFINE_boolean("enable_motion", False,
                     "Actually move the follower (operator confirmation required).")


def smallest_missing_id(dir_path: str) -> int:
    os.makedirs(dir_path, exist_ok=True)
    ids = {int(n) for n in os.listdir(dir_path)
           if n.isdigit() and os.path.isdir(os.path.join(dir_path, n))}
    i = 0
    while i in ids:
        i += 1
    return i


def _suffixed_obs(obs, suffix):
    """View of one arm's fields under the single-arm key names."""
    return {
        "cartesian_position": obs[f"cartesian_position{suffix}"],
        "gripper_position": obs[f"gripper_position{suffix}"],
    }


def wait_for_engage(env, units, mirrors, tol_rad: float = 0.05):
    """Block until every follower has converged to its master AND the left
    EE is inside the task workspace. The episode (recording, clock) starts
    only after this returns, so the engage ramp is never part of the data."""
    print("[engage] move the master(s) into the workspace; episode starts when "
          "the follower(s) have converged...")
    last_msg = 0.0
    while True:
        why = []
        for u, mc in zip(units, mirrors):
            q_f, _ = u["arm"].read_joints()
            q_m, _ = mc.master.read_raw()
            gap = np.abs(q_f - u["kin"].clamp_to_limits(q_m)).max()
            if gap >= tol_rad:
                why.append(f"{u['suffix'] or '_left'} joint gap {gap:.2f} rad")
            if u["suffix"] == "" and env.bounds is not None:
                pose = u["kin"].fk(q_f)
                in_bounds = bool((pose[:3] > env.bounds[:, 0]).all()
                                 and (pose[:3] < env.bounds[:, 1]).all())
                if not in_bounds:
                    why.append(f"left EE {np.round(pose[:3], 2)} outside workspace")
        if not why:
            print("[engage] engaged — recording.")
            return
        now = time.monotonic()
        if now - last_msg > 3.0:
            print(f"[engage] waiting: {'; '.join(why)}")
            last_msg = now
        time.sleep(0.1)


def collect_trajectory(env, units, mirrors, save_filepath: str | None):
    """One episode of mirrored teleop; returns {"success": bool}."""
    writer = HDF5TrajWriter(save_filepath) if save_filepath else None
    # No reset motion: the master defines the pose, and mirroring stays
    # engaged across episodes (bookkeeping-only reset).
    env.reset(move=False)
    wait_for_engage(env, units, mirrors)
    env.reset_episode_clock()
    dt = 1.0 / env.control_hz
    next_tick = time.monotonic()
    success = False
    prev_obs = None
    n_sat = 0
    raw_history: list[np.ndarray] = []
    try:
        while True:
            obs = env.get_observation()
            done, success, _, _ = env.get_info_for_step()
            if done:
                return {"success": bool(success)}

            if prev_obs is not None and writer is not None:
                # Actions = each follower's actual motion between the last
                # two observations, in the normalized training contract.
                action_group = {}
                any_sat = False
                for u in units:
                    executed, saturated, raw = executed_action_from_obs(
                        _suffixed_obs(prev_obs, u["suffix"]),
                        _suffixed_obs(obs, u["suffix"]),
                        dt, env.limits, u["gripper"].max_open_m,
                    )
                    any_sat = any_sat or saturated
                    raw_history.append(np.abs(raw))
                    executed = executed.astype(np.float32)
                    action_group[f"cartesian_velocity{u['suffix']}"] = executed[:6]
                    action_group[f"gripper_velocity{u['suffix']}"] = np.float32(executed[6])
                n_sat += int(any_sat)
                writer.write_timestep({
                    "saved_observation": prev_obs,
                    "action": action_group,
                })
            prev_obs = obs

            next_tick += dt
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()
    finally:
        if writer is not None:
            steps = writer.num_steps
            writer.close()
            print(f"episode finished: {steps} steps, success={success}")
            if steps and n_sat:
                pct = 100.0 * n_sat / steps
                print(f"WARNING: {n_sat}/{steps} steps ({pct:.0f}%) saturated the "
                      f"velocity limits — recorded actions understate the motion. "
                      f"Move slower or raise the limits in the hardware yaml.")
            if raw_history:
                r = np.stack(raw_history)  # |raw|, 1.0 == at the limit
                p95 = np.percentile(r, 95, axis=0)
                mx = r.max(axis=0)
                names = ["vx", "vy", "vz", "wx", "wy", "wz", "grip"]
                print("  velocity usage (1.0 = limit): "
                      + "  ".join(f"{n} p95={p:.2f} max={m:.2f}" for n, p, m in zip(names, p95, mx)))


def run_and_route_one(env, units, mirrors, base_dir: str):
    tmp_root = os.path.join(base_dir, "tmp")
    os.makedirs(tmp_root, exist_ok=True)
    tmp_dir = os.path.join(tmp_root, f"session_{int(time.time())}")
    os.makedirs(tmp_dir, exist_ok=True)
    save_filepath = os.path.join(tmp_dir, "traj.hdf5")

    print("Start collecting ->", save_filepath)
    result = collect_trajectory(env, units, mirrors, save_filepath)

    if not result["success"]:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("Failure — discarded temp data.")
        return {"dest_dir": None, "result": result}

    outcome_root = os.path.join(base_dir, "success")
    new_id = smallest_missing_id(outcome_root)
    dest_dir = os.path.join(outcome_root, str(new_id))
    shutil.move(tmp_dir, dest_dir)
    print("Saved:", os.path.join(dest_dir, "traj.hdf5"))
    return {"dest_dir": dest_dir, "result": result}


def main(_):
    base_dir = FLAGS.save_root
    os.makedirs(base_dir, exist_ok=True)

    task_config = FLAGS.task_config
    env_kwargs = dict(task_config)
    env_kwargs["dry_run"] = not FLAGS.enable_motion
    env_kwargs["enable_motion"] = FLAGS.enable_motion
    env_kwargs["use_master_intervention"] = False  # we read the master directly here

    if FLAGS.enable_motion:
        answer = input("Motion ENABLED — follower will mirror the master (hold the master "
                       "near a neutral pose; the follower ramps to it on start). "
                       "Type 'yes' to continue: ")
        if answer.strip().lower() != "yes":
            raise SystemExit("Motion not confirmed; exiting.")

    env = task_config.env(**env_kwargs)
    env.ignore_auto_reset = True  # teleop: end episodes manually, not on step budget

    units = env.units             # one entry per arm (single or bimanual env)
    mirror_cfg = env.hw.get("mirror", {})
    masters, mirrors = [], []
    for u in units:
        master = MasterArm(u["master_can"], u["kin"], u["gripper"], env.limits)
        master.connect()
        mirror = MirrorController(
            master, u["servo"], u["kin"], panel=env.panel,
            mirror_hz=mirror_cfg.get("hz", 50),
            max_joint_speed_rads=mirror_cfg.get("max_joint_speed_rads", 2.0),
            gripper_scale=mirror_cfg.get("gripper_scale", 1.0),
        )
        mirror.start()
        q_now, _ = u["arm"].read_joints()
        mirror.resume(q_now)  # one engage ramp per session; stays coupled after
        masters.append(master)
        mirrors.append(mirror)

    episode, successful = 0, 0
    try:
        while True:
            episode += 1
            print(f"\n=== Episode {episode} (successful {successful}/"
                  f"{FLAGS.num_episodes if FLAGS.num_episodes > 0 else '∞'}) ===")
            result = run_and_route_one(env, units, mirrors, base_dir)
            if result["result"]["success"]:
                successful += 1
            if FLAGS.num_episodes > 0 and successful >= FLAGS.num_episodes:
                print(f"Reached target of {FLAGS.num_episodes} successful episodes.")
                break
    finally:
        for mirror in mirrors:
            mirror.stop()
        for master in masters:
            master.close()
        env.close()


if __name__ == "__main__":
    app.run(main)
