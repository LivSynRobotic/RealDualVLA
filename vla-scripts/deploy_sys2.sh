#!/bin/bash

SYS1_IP="http://0.0.0.0:8777"

export PYTHONPATH=$(pwd):$PYTHONPATH
CUDA_VISIBLE_DEVICES=3 python vla-scripts/deploy_sys2.py \
  --host 0.0.0.0 \
  --port 8778 \
  --action_head_server_url "${SYS1_IP}/update_hidden_states" \
  --pretrained_checkpoint /root/data0/rd_vla/real_dual_vla/checkpoint/lerobot_single_tasks/ar_1104_save \
  --use_film True \
  --num_images_in_input 2 \
  --use_proprio True \
  --center_crop True \
  --unnorm_key lerobot_single_put_red_apple_in_green_plate_1019_rlds