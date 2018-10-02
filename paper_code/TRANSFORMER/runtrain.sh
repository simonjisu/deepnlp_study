#!/bin/sh
nohup python3 -u main.py \
    -pth_tr ../../data/translation/de_en_small.txt \
    -pth_va ../../data/translation/de_en_small_valid.txt \
    -exts src-trg \
    -stp 10 \
    -bs 32 \
    -cuda \
    -nl 3 \
    -nh 5 \
    -dk 32 \
    -dv 32 \
    -dm 160 \
    -df 320 \
    -drop 0.1 \
    -lws \
    -warm 4000 \
    -save \
    -svp ./model/transformer > ./transformer.log &