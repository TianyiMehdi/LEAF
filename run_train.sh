#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bmpc
cd "$(dirname "$0")/leaf"
export CUDA_VISIBLE_DEVICES=0

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

# TASK=humanoid_h1hand-pole-v0
TASK=humanoid-run
SEED=1
STEPS=1_000_000

EXP_NAME=leaf

python train.py task=${TASK} seed=${SEED} flow=true exp_name=${EXP_NAME} steps=${STEPS}
