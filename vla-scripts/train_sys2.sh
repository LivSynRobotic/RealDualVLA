#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH
CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path /root/data0/rd_vla/openvla-7b \
  --data_root_dir /root/data0/rd_vla/ \
  --dataset_name lerobot_single_tasks \
  --run_root_dir /root/data0/rd_vla/real_dual_vla/checkpoint/lerobot_single_tasks \
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

