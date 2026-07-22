"""Convert Piper raw HDF5 demonstrations to LeRobot format.

Reads traj.hdf5 files written by piper_client (saved_observation/* +
action/*) and writes a LeRobot dataset for Pi0.5 normalization statistics
and supervised fine-tuning. Language instruction comes from the task
config (config.language_instruction).

Usage:
uv run scripts/convert_piper_data_to_lerobot.py \\
    --data_dir /data2/expo-ft-piper/raw/pick/success \\
    --repo_name expo_ft/piper_pick_cube --task_config configs/task/piper_pick.py \\
    --use_cartesian_state --no-push-to-hub

The resulting dataset is saved under the $LEROBOT_HOME directory.
"""

import shutil
import sys
from pathlib import Path

# Add project root so configs.task.* can be imported when run as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from tqdm import tqdm
import tyro


def resize_image(image, size):
    image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    return np.array(image.resize(size, resample=Image.BICUBIC))


def load_task_config(config_path: str):
    if "/" in config_path or ".py" in config_path:
        config_path = config_path.replace(".py", "").replace("/", ".")
    module = __import__(config_path, fromlist=["get_config"])
    return module.get_config()


def main(
    data_dir: str,
    *,
    repo_name: str,
    task_config: str,
    max_episodes: int | None = None,
    push_to_hub: bool = False,
    use_cartesian_state: bool = True,
    fps: int = 10,
):
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    data_dir = Path(data_dir)
    task_cfg = load_task_config(task_config)
    language_instruction = task_cfg.language_instruction
    action_key = task_cfg.action_space
    gripper_key = f"gripper_{task_cfg.gripper_action_space}"
    action_dim = 7 if action_key == "cartesian_velocity" else 8

    state_feature = (
        {"cartesian_position": {"dtype": "float32", "shape": (6,), "names": ["cartesian_position"]}}
        if use_cartesian_state
        else {"joint_position": {"dtype": "float32", "shape": (7,), "names": ["joint_position"]}}
    )
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="piper",
        fps=fps,  # Piper data is recorded at the policy rate (10 Hz)
        features={
            # DROID RLDS naming so the existing Pi0.5-DROID pipeline transfers
            "exterior_image_1_left": {"dtype": "image", "shape": (180, 320, 3),
                                      "names": ["height", "width", "channel"]},
            "exterior_image_2_left": {"dtype": "image", "shape": (180, 320, 3),
                                      "names": ["height", "width", "channel"]},
            "wrist_image_left": {"dtype": "image", "shape": (180, 320, 3),
                                 "names": ["height", "width", "channel"]},
            **state_feature,
            "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
            "actions": {"dtype": "float32", "shape": (action_dim,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    def _episode_sort_key(p):
        name = p.parent.name
        return (int(name),) if name.isdigit() else (float("inf"), name)

    episode_paths = sorted(data_dir.glob("**/traj.hdf5"), key=_episode_sort_key)
    if max_episodes is not None:
        episode_paths = episode_paths[:max_episodes]
        print(f"Using {len(episode_paths)} episodes (max_episodes={max_episodes})")
    else:
        print(f"Found {len(episode_paths)} episodes for conversion")

    state_key = "cartesian_position" if use_cartesian_state else "joint_position"

    for episode_path in tqdm(episode_paths, desc="Converting episodes"):
        with h5py.File(episode_path, "r") as f:
            obs = f["saved_observation"]
            act = f["action"]
            T = len(act[gripper_key])
            if T == 0:
                print(f"Skipping empty trajectory: {episode_path}")
                continue
            arm_actions = np.asarray(act[action_key], dtype=np.float32)
            grip_actions = np.atleast_2d(np.asarray(act[gripper_key], dtype=np.float32).reshape(T, -1))
            actions = np.concatenate([arm_actions, grip_actions], axis=-1)

            for t in range(T):
                dataset.add_frame({
                    "exterior_image_1_left": resize_image(obs["exterior_image_1_left"][t], (320, 180)),
                    "exterior_image_2_left": resize_image(obs["exterior_image_2_left"][t], (320, 180)),
                    "wrist_image_left": resize_image(obs["wrist_image_left"][t], (320, 180)),
                    state_key: np.asarray(obs[state_key][t], dtype=np.float32),
                    "gripper_position": np.atleast_1d(
                        np.asarray(obs["gripper_position"][t], dtype=np.float32)).reshape(1),
                    "actions": actions[t],
                    "task": language_instruction,
                })
        dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["piper", "expo-ft"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
