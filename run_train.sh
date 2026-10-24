#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bmpc
cd "$(dirname "$0")/leaf"
export CUDA_VISIBLE_DEVICES=0

# 集群上无本地代理时取消代理，避免 wandb 连 127.0.0.1:7897 报错
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

# 直接改这里即可，会覆盖 config.yaml 里的默认值
# TASK=humanoid_h1hand-pole-v0
TASK=humanoid-run
SEED=1
STEPS=1_000_000

EXP_NAME=leaf

python train.py task=${TASK} seed=${SEED} flow=true exp_name=${EXP_NAME} steps=${STEPS}
