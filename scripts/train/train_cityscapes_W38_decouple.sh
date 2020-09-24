#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")
EXP_DIR=./WiderResNet38_map_city_decouple_edge_attention
mkdir ${EXP_DIR}
#Example on Cityscapes
python -m torch.distributed.launch --nproc_per_node=8 train.py \
  --dataset cityscapes \
  --cv 2 \
  --arch network.deepv3_decouple.DeepWV3PlusDecoupleEdgeBody \
  --snapshot /path/wider_38_deeplabv3+.pth \
  --class_uniform_pct 0.5 \
  --class_uniform_tile 1024 \
  --max_cu_epoch 150 \
  --lr 0.005 \
  --lr_schedule poly \
  --poly_exp 1.0 \
  --repoly 1.5  \
  --rescale 1.0 \
  --syncbn \
  --sgd \
  --crop_size 896 \
  --scale_min 0.5 \
  --scale_max 2.0 \
  --color_aug 0.25 \
  --gblur \
  --eval_epoch 1 \
  --max_epoch 175 \
  --coarse_boost_classes 14,15,16,3,12,17,4 \
  --jointwtborder \
  --joint_edgeseg_loss \
  --wt_bound 1.0 \
  --bs_mult 2 \
  --apex \
  --exp cityscapes_ft \
  --ckpt ${EXP_DIR}/ \
  --tb_path ${EXP_DIR}/ \
  2>&1 | tee  ${EXP_DIR}/log_$now.txt &