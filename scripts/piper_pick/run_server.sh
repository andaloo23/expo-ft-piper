#!/usr/bin/env bash
# Synchronous EXPO-FT online training (learner env, RTX 5090).
source .venv/bin/activate

CLIENT_IP=127.0.0.1   # actor runs on the same machine (gail8)

export CUDA_VISIBLE_DEVICES=0

python train_pi_robo.py \
    --config_task=configs/task/piper_pick.py \
    --dataset_path=/data2/expo-ft-piper/raw/pick/success \
    --num_data=10 \
    --update_type=episode \
    --num_updates=3 \
    --offline_ratio=0 \
    --batch_size=16 \
    --config=configs/model/expo_ft_pi_config.py \
    --config.N=8 \
    --config.n_edit_samples=8 \
    --config.edit_scale=0.2 \
    --config.pi05_config_name=expo_pi05_droid_lora_finetune_sft_cartesian_state \
    --config.pi05_weight_loader_path="/data2/expo-ft-piper/checkpoints/expo_pi05_droid_lora_finetune_sft_cartesian_state/piper_pick_cube_10_lora_sft/4000/params" \
    --config.pi05_assets_dir="./assets/expo_pi05_droid_lora_finetune_sft_cartesian_state" \
    --config.pi05_asset_id="expo_ft/piper_pick_cube_10" \
    --project_name=expo_ft_piper_pick \
    --output_dir=/data2/expo-ft-piper/checkpoints/pick \
    --client_host="$CLIENT_IP" \
    --client_port=8102 \
    --fsdp_devices=1 \
    --resume \
    --checkpoint_model \
    --checkpoint_buffer \
    --checkpoint_interval=2000 \
    --run_name=expo_piper_pick_example
