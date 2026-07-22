#!/usr/bin/env bash
# Master-arm demonstration collection (actor env).
source piper_client/.venv/bin/activate

NUM_EPISODES=15

python -m piper_client.collect_data \
    --save_root /data2/expo-ft-piper/raw/pick \
    --num_episodes $NUM_EPISODES \
    --task_config configs/task/piper_pick.py \
    --enable_motion
