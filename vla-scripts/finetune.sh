#!/bin/bash

# --pretrained_rdt_path /vla/ym_test/RoboticsDiffusionTransformer/checkpoints/rdt-1b \
# /root/data0/rd_vla/siglip-so400m-patch14-224
# /root/data0/rd_vla/RoboticsDiffusionTransformer/google/siglip-so400m-patch14-384

# nohup ./vla-scripts/finetune.sh > ar_1205.log 2>&1 &

# lerobot_single_smart_grab_fruit_0715_rlds
# lerobot_single_smart_grab_fruit_avoid_poke_0707_rlds
# lerobot_single_smart_grab_fruit_pi0_611s_rlds
# lerobot_single_tasks
# lerobot_single_106_smart_grab_fruit_0916_rlds
# lerobot_single_eef_106_smart_grab_fruit_0920_rlds
# lerobot_single_106_grab_block_0928_rlds
# lerobot_single_jq_grab_block_1015_rlds
# lerobot_single_jq_smart_grab_fruit_1016_rlds/
# lerobot_single_put_green_apple_in_green_plate_1019_rlds
# lerobot_single_put_red_apple_in_green_plate_1019_rlds
# lerobot_single_one_red_apple_1021_rlds/
# --pretrained_rdt_path /root/data0/rd_vla/real_dual_vla/checkpoint/lerobot_single_tasks/fm_t1_1016/rdt_action_head_step_40000 \


export PYTHONPATH=$(pwd):$PYTHONPATH
CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path /root/data0/rd_vla/openvla-7b \
  --data_root_dir /root/data0/rd_vla/ \
  --dataset_name lerobot_single_tasks \
  --run_root_dir /root/data0/rd_vla/real_dual_vla/checkpoint/lerobot_single_tasks \
  --use_l1_regression False \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 10000005 \
  --use_val_set False \
  --val_freq 10000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only True \
  --image_aug True \
  --lora_rank 32 \
  --pretrained_vision_encoder_name_or_path /root/data0/rd_vla/siglip-so400m-patch14-224 \
  --use_rdt False \
  --rdt_config_path /root/data0/rd_vla/real_dual_vla/fast_models/base.yaml \
  --run_id_override=test \
  --slow_train_mode True \

