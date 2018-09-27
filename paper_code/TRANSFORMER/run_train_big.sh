#!/bin/sh
nohup python3 -u main.py \
    -pth_tr ../../data/translation/de_en_big.txt \
    -pth_va ../../data/translation/de_en_small_valid.txt \
    -exts src-trg \
    -stp 10 \
    -bs 32 \
    -cuda \
    -nl 3 \
    -nh 6 \
    -dk 32 \
    -dv 32 \
    -dm 192 \
    -df 576 \
    -drop 0.1 \
    -lws \
    -warm 4000 \
    -save \
    -svp ./model/ > ./transformer.log &