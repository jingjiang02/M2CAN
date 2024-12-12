#!/bin/bash

ROOT=../..

GPU_IDX=${2:-0}

CUDA_VISIBLE_DEVICES=$GPU_IDX python3 \
  $ROOT/test.py --pth_file $1 --config=$1/config.yaml --model_name msmm
