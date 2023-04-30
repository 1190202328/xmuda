#!/bin/bash

ROOT=../..

bash /nfs/volume-902-16/tangwenbo/ofs-1.sh

cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/MS-MMDA/experiments/source_only_source_combined && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python -m torch.distributed.launch \
  --nproc_per_node=$1 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port=$2 \
  $ROOT/train.py --config=config.yaml --seed 2 --port $2
