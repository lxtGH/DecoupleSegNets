#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")
EXP_DIR=./body_edge/ResNet50_FCN_m4_decouple_ft_175_e
mkdir -p ${EXP_DIR}
# Example on Cityscapes by resnet50-deeplabv3+ as baseline
python -m torch.distributed.launch --nproc_per_node=8 train.py \
  --dataset cityscapes \
  --cv 0 \
  --arch network.deepv3_decouple.DeepR50FCN_m1_deeply \
  --class_uniform_pct 0.5 \
  --class_uniform_tile 1024 \
  --max_cu_epoch 150 \
  --lr 0.01 \
  --lr_schedule poly \
  --poly_exp 1.0 \
  --repoly 1.5  \
  --rescale 1.0 \
  --syncbn \
  --sgd \
  --crop_size 832 \
  --scale_min 0.5 \
  --scale_max 2.0 \
  --color_aug 0.25 \
  --gblur \
  --max_epoch 100 \
  --ohem \
  --coarse_boost_classes 14,15,16,3,12,17,4 \
  --ohem \
  --wt_bound 1.0 \
  --bs_mult 1 \
  --apex \
  --exp cityscapes_ft \
  --ckpt ${EXP_DIR}/ \
  --tb_path ${EXP_DIR}/ \
  2>&1 | tee  ${EXP_DIR}/log_${now}.txt &
