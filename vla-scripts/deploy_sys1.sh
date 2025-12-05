#!/bin/bash


# 建立反向隧道
# autossh -M 0 -f -N -R 8777:127.0.0.1:8777 user@YOUR_SERVER_IP

python vla-scripts/deploy_sys1.py \
  --host 0.0.0.0 \
  --port 8777 \
  --rdt_config_path /home/ls01/Music/openvla-oft/rdt_train/base.yaml \
  --pretrained_rdt_path /home/ls01/Music/openvla-oft/checkpoint/lerobot_single_one_red_apple_1021_rlds/130m_norm/rdt_action_head_step_50000 \
  --pretrained_vision_encoder_name_or_path /home/ls01/RoboticsDiffusionTransformer/google/siglip-so400m-patch14-384 \
  --num_images_in_input 2 \
  --unnorm_key lerobot_single_one_red_apple_1021_rlds