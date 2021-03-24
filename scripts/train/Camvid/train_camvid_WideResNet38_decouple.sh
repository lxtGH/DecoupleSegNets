#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")
EXP_DIR=./body_edge/camvid_large_ft_best_city_decouple_837
mkdir -p ${EXP_DIR}
# Example on Camvid,  fine tune from Cityscapes

python -m torch.distributed.launch --nproc_per_node=8 train.py \
  --dataset camvid \
  --cv 2 \
  --arch network.deepv3_decouple.DeepWV3PlusDecoupleEdgeBody \
  --snapshot ~/pretrained_models/best_city_decouple_837.pth \
  --class_uniform_pct 0.5 \
  --class_uniform_tile 640 \
  --lr 0.001 \
  --lr_schedule poly \
  --poly_exp 1.0 \
  --syncbn \
  --sgd \
  --crop_size 640 \
  --scale_min 1.0 \
  --scale_max 2.0 \
  --color_aug 0.25 \
  --max_epoch 90 \
  --jointwtborder \
  --joint_edgeseg_loss \
  --wt_bound 1.0 \
  --bs_mult 2 \
  --apex \
  --exp kitti_ft \
  --ckpt ${EXP_DIR}/ \
  --tb_path ${EXP_DIR}/ \
  2>&1 | tee  ${EXP_DIR}/log_${now}.txt &

