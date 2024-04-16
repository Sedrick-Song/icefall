#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

#training process
./zipformer/train.py \
    --world-size 4 \
    --lang-dir data/lang_char_all \
    --num-epochs 60 \
    --start-epoch 1 \
    --exp-dir zipformer/exp_all_withoutmusan \
    --max-duration 300 \
    --base-lr 0.045 \
    --use-fp16 0 \
    --context-size 1 \
    --enable-musan 0 \
    --master-port 12111


