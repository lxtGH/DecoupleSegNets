#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")
EXP_DIR=./body_edge/BDD_large_ft_map_w_38_deepv3
mkdir -p ${EXP_DIR}
# Example on Camvid,  fine tune from Cityscapes

python -m torch.distributed.launch --nproc_per_node=8 train.py \
  --dataset bdd \
  --cv 2 \
  --arch network.deepv3.DeepWV3Plus \
  --snapshot ~/pretrained/mapillary_DeepWV3PlusS_50.03.pth \
  --class_uniform_pct 0.5 \
  --class_uniform_tile 720 \
  --lr 0.001 \
  --lr_schedule poly \
  --poly_exp 1.0 \
  --syncbn \
  --sgd \
  --crop_size 720 \
  --scale_min 1.0 \
  --scale_max 2.0 \
  --color_aug 0.25 \
  --max_epoch 100 \
  --jointwtborder \
  --joint_edgeseg_loss \
  --wt_bound 1.0 \
  --bs_mult 2 \
  --apex \
  --exp kitti_ft \
  --ckpt ${EXP_DIR}/ \
  --tb_path ${EXP_DIR}/ \
  2>&1 | tee  ${EXP_DIR}/log_${now}.txt &

