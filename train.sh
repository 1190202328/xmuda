#!/bin/bash

bash /nfs/volume-902-16/tangwenbo/s3_all.sh

cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/xmuda && pip install -ve . -i https://pypi.mirrors.ustc.edu.cn/simple/

cd /nfs/ofs-902-1/object-detection/jiangjing/experiments/xmuda && CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python \
  xmuda/train_xmuda.py --cfg=configs/nuscenes/usa_singapore/xmuda.yaml
