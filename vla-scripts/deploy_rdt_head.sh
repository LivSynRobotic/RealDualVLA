#!/bin/bash


# /home/ls01/RoboticsDiffusionTransformer/google/siglip-so400m-patch14-384
# /home/ls01/Music/siglip-so400m-patch14-224

# lerobot_single_smart_grab_fruit_0715_rlds
# lerobot_dual_fold_clothes_3_step_20250904_rlds
# lerobot_single_106_smart_grab_fruit_0916_rlds
# lerobot_single_106_grab_block_0928_rlds
# lerobot_single_jq_grab_block_1015_rlds
# lerobot_single_jq_smart_grab_fruit_1016_rlds
# lerobot_single_put_green_apple_in_green_plate_1019_rlds
# lerobot_single_put_red_apple_in_green_plate_1019_rlds
# lerobot_single_one_red_apple_1021_rlds

# 检查反向隧道是否存在，不存在则建立
if ! pgrep -f "autossh.*8777.*root@159.75.78.6" > /dev/null; then
    echo "建立反向隧道..."
    autossh -M 0 -f -N -R 8777:127.0.0.1:8777 root@159.75.78.6
fi

python vla-scripts/deploy_rdt_head.py \
  --host 0.0.0.0 \
  --port 8777 \
  --rdt_config_path /home/ls01/Music/openvla-oft/rdt_train/base.yaml \
  --pretrained_rdt_path /home/ls01/Music/openvla-oft/checkpoint/lerobot_single_one_red_apple_1021_rlds/130m_norm/rdt_action_head_step_50000 \
  --pretrained_vision_encoder_name_or_path /home/ls01/RoboticsDiffusionTransformer/google/siglip-so400m-patch14-384 \
  --num_images_in_input 2 \
  --unnorm_key lerobot_single_one_red_apple_1021_rlds