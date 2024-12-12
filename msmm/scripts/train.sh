#!/bin/bash

DIR=..

GPU_IDX=${2:-0}

cd /home/jjiang/experiments/m2can/msmm/scripts/ && CUDA_VISIBLE_DEVICES=$GPU_IDX python3 \
  $DIR/train.py --config=$1
