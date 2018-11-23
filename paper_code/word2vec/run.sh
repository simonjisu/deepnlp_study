#!/bin/sh
nohup python3 -u main.py -pth ../../data/wiki/enwik9_text \
		 	 -svp ./model/word2vec_wiki.model \
		   	 -bat 1024 \
		   	 -win 2 \
		   	 -z 1e-5 \
		   	 -emd 300 \
		   	 -stp 20 \
		   	 -lr 0.001 \
		   	 -ee 1 \
		   	 -n 10000000 > ./wiki.log &

