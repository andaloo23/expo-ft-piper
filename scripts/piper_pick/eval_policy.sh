#!/usr/bin/env bash
# Guarded policy evaluation (learner env). Start the actor first with
# run_client.sh (shadow eval: leave it in --dry-run; real eval: --enable-motion).
source .venv/bin/activate
export OPENPI_DATA_HOME=/data/cache/openpi
CLIENT_IP=127.0.0.1

export CUDA_VISIBLE_DEVICES=0

python eval_piper_policy.py \
    --config_task=configs/task/piper_pick.py \
    --config=configs/model/expo_ft_pi_config.py \
    --dataset_path=/data2/expo-ft-piper/raw/pick/success \
    --num_data=1 \
    --client_host="$CLIENT_IP" \
    --client_port=8102 \
    --config.N=8 \
    --config.n_edit_samples=8 \
    --config.edit_scale=0.2 \
    --config.pi05_config_name=expo_pi05_droid_lora_finetune_sft_cartesian_state \
    --config.pi05_weight_loader_path="/data2/expo-ft-piper/checkpoints/expo_pi05_droid_lora_finetune_sft_cartesian_state/piper_pick_cube_10_lora_sft/4000/params" \
    --config.pi05_assets_dir="./assets/expo_pi05_droid_lora_finetune_sft_cartesian_state" \
    --config.pi05_asset_id="expo_ft/piper_pick_cube_10" \
    --checkpoint_dir=/data2/expo-ft-piper/checkpoints/pick/expo_piper_pick_example/checkpoints \
    --num_episodes=10
