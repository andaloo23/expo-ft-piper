"""Master-arm demonstration collection for one Piper follower arm.

Loop at policy rate: read the master arm as a normalized 7D action, execute
it on the follower through the same ActionAdapter/servo path the policy
will use, and record (observation, executed action) pairs to traj.hdf5 in
the EXPO-FT raw layout. Successful episodes are routed to
<save_root>/success/<id>/traj.hdf5; failures are discarded.

End an episode from the terminal: 1 = success, 2 = reset (failure).

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


def collect_trajectory(env, master: MasterArm, save_filepath: str | None):
    """One episode; returns {"success": bool}."""
    master.reset_state()
    writer = HDF5TrajWriter(save_filepath) if save_filepath else None
    env.reset()
    dt = 1.0 / env.control_hz
    step_start = time.monotonic()
    success = False
    try:
        while True:
            obs = env.get_observation()
            done, success, _, _ = env.get_info_for_step()
            if done:
                return {"success": bool(success)}

            sample = master.read()
            action = sample.action

            elapsed = time.monotonic() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
            step_result = env.step(action)
            step_start = time.monotonic()
            executed = np.asarray(step_result["executed_action"], dtype=np.float32)

            if writer is not None:
                writer.write_timestep({
                    "saved_observation": obs,
                    "action": {
                        "cartesian_velocity": executed[:6],
                        "gripper_velocity": np.float32(executed[6]),
                    },
                })
    finally:
        if writer is not None:
            steps = writer.num_steps
            writer.close()
            print(f"episode finished: {steps} steps, success={success}")


def run_and_route_one(env, master: MasterArm, base_dir: str):
    tmp_root = os.path.join(base_dir, "tmp")
    os.makedirs(tmp_root, exist_ok=True)
    tmp_dir = os.path.join(tmp_root, f"session_{int(time.time())}")
    os.makedirs(tmp_dir, exist_ok=True)
    save_filepath = os.path.join(tmp_dir, "traj.hdf5")

    print("Start collecting ->", save_filepath)
    result = collect_trajectory(env, master, save_filepath)

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
        answer = input("Motion ENABLED — follower will mirror the master. Type 'yes' to continue: ")
        if answer.strip().lower() != "yes":
            raise SystemExit("Motion not confirmed; exiting.")

    env = task_config.env(**env_kwargs)
    env.ignore_auto_reset = True  # teleop: end episodes manually, not on step budget

    master = MasterArm(env.hw["robot"]["master_can"], env.kin, env.gripper, env.limits)
    master.connect()

    episode, successful = 0, 0
    try:
        while True:
            episode += 1
            print(f"\n=== Episode {episode} (successful {successful}/"
                  f"{FLAGS.num_episodes if FLAGS.num_episodes > 0 else '∞'}) ===")
            result = run_and_route_one(env, master, base_dir)
            if result["result"]["success"]:
                successful += 1
            if FLAGS.num_episodes > 0 and successful >= FLAGS.num_episodes:
                print(f"Reached target of {FLAGS.num_episodes} successful episodes.")
                break
    finally:
        master.close()
        env.close()


if __name__ == "__main__":
    app.run(main)
