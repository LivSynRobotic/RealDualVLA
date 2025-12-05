#!/bin/bash

# source ~/.bashrc
# conda activate openvla-oft
# cd /vla/ym_test/openvla-oft/
  # --pretrained_rdt_path /vla/ym_test/RoboticsDiffusionTransformer/checkpoints/rdt-1b \

# nohup ./vla-scripts/finetune_dual.sh > lerobot_dual_tasks_64s_130m_unnorm_0925.log 2>&1 &

# lerobot_dual_fold_0825_rlds
# lerobot_single_clean_dish_pi0_0620_rlds
# lerobot_dual_fold_clothes_3_step_20250904_rlds
# lerobot_single_smart_grab_fruit_0710_rlds
# lerobot_single_smart_grab_fruit_0715_rlds
# lerobot_single_smart_grab_fruit_avoid_poke_0707_rlds
# lerobot_dual_tasks

export PYTHONPATH=$(pwd):$PYTHONPATH
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path /root/data0/rd_vla/openvla-7b \
  --data_root_dir /root/data0/rd_vla/ \
  --dataset_name lerobot_dual_tasks \
  --run_root_dir /root/data0/rd_vla/openvla-oft/checkpoint/lerobot_dual_tasks \
  --use_l1_regression False \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --batch_size 32 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 100005 \
  --use_val_set False \
  --val_freq 10000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only True \
  --image_aug False \
  --lora_rank 32 \
  --pretrained_vision_encoder_name_or_path /root/data0/rd_vla/RoboticsDiffusionTransformer/google/siglip-so400m-patch14-384 \
  --use_rdt True \
  --rdt_config_path /root/data0/rd_vla/openvla-oft/rdt_train/base_dual.yaml \
  --run_id_override=64s_130m_unnorm_0925