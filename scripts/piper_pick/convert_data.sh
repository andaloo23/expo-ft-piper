#!/usr/bin/env bash
# Convert raw Piper HDF5 episodes to LeRobot (learner env).
source .venv/bin/activate

MAX_EPISODES=10
TASK_CONFIG="configs/task/piper_pick.py"
DATA_DIR="/data2/expo-ft-piper/raw/pick/success"
REPO_NAME="expo_ft/piper_pick_cube_${MAX_EPISODES}"

uv run scripts/convert_piper_data_to_lerobot.py \
    --data_dir="$DATA_DIR" \
    --repo_name="$REPO_NAME" \
    --task_config="$TASK_CONFIG" \
    --max_episodes="$MAX_EPISODES" \
    --use_cartesian_state \
    --no-push-to-hub
