#!/bin/bash

# source ~/.bashrc
# conda activate openvla-oft
# cd /vla/ym_test/openvla-oft/
  #--pretrained_checkpoint /root/data0/rd_vla/openvla-oft/checkpoint/0818_l1/openvla-7b+lerobot_single_smart_grab_fruit_0710_rlds+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--bin_rdt \
  #--pretrained_checkpoint /root/data0/rd_vla/openvla-oft/checkpoint/0818_diff/openvla-7b+lerobot_single_smart_grab_fruit_0710_rlds+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--bin_rdt \

export PYTHONPATH=$(pwd):$PYTHONPATH 
CUDA_VISIBLE_DEVICES=0 python vla-scripts/deploy.py \
  --pretrained_checkpoint /home/ls01/Music/openvla-oft/checkpoint/0818_diff/openvla-7b+smart_grab_fruit_0710_rlds+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--bin_rdt \
  --use_l1_regression False \
  --use_diffusion True \
  --use_film True \
  --num_images_in_input 2 \
  --use_proprio True \
  --center_crop True \
  --unnorm_key lerobot_single_smart_grab_fruit_0710_rlds
