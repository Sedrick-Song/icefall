#!/usr/bin/env bash

#decoding using modified beam search
:<<!
./zipformer/decode.py \
    --epoch 25 \
    --avg 10 \
    --exp-dir ./zipformer/exp_all \
    --lang-dir data/lang_char_all \
    --max-duration 600 \
    --use-averaged-model True \
    --context-size 1 \
    --decoding-method modified_beam_search \
    --beam-size 4
!

for ((i=41; i<=60; i++)); do
    for ((j=10; j<i-10; j++)); do
        echo "epoch: $i, avg: $j"
        ./zipformer/decode.py \
            --epoch $i \
            --avg $j \
            --exp-dir ./zipformer/exp_all_withoutmusan \
            --lang-dir data/lang_char_all \
            --max-duration 1000 \
            --use-averaged-model True \
            --context-size 1 \
            --decoding-method modified_beam_search \
            --beam-size 4
    done
done