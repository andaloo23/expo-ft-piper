#!/usr/bin/env bash
# Piper actor: rollout server on the robot side (CPU only — hide the GPU so
# the actor cannot consume learner VRAM). Robot and learner are both on
# gail8, so the learner connects to 127.0.0.1:8102; no SSH tunnel.
source piper_client/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=

# Default is dry-run. Add --enable-motion (and confirm at the prompt) for real motion.
python -m piper_client.run_client \
    --server_host=127.0.0.1 \
    --server_port=8102 \
    --config_task_path=configs/task/piper_pick.py \
    "$@"
