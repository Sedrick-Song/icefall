#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PATH=/hpc_stor03/sjtu_home/jianheng.zhuo/miniconda3/envs/k2/bin
export LD_LIBRARY_PATH=/hpc_stor03/sjtu_home/jianheng.zhuo/miniconda3/envs/k2/lib

#training process
./zipformer/train.py \
    --world-size 4 \
    --num-epochs 15 \
    --start-epoch 1 \
    --exp-dir zipformer/exp_nst_withoutdrama \
    --max-duration 600 \
    --base-lr 0.045 \
    --use-fp16 0 \
    --master-port 12234


