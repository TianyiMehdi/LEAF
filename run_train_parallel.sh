#!/bin/bash
# 并行启动多 task × 多 seed（模仿 FLAC/scripts/train_parallel.sh + 本仓库 run_train.sh）

set -uo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bmpc || { echo "conda env bmpc not found"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/leaf"

# 集群上无本地代理时取消代理，避免 wandb 连 127.0.0.1:7897 报错
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

# ========== 按需修改 ==========
# DMC 示例：humanoid-run / dog-stand；H1 示例：humanoid_h1hand-sit_simple-v0
tasks=(
  humanoid_h1hand-sit_simple-v0
  humanoid_h1hand-sit_hard-v0
  humanoid_h1hand-balance_simple-v0
  humanoid_h1hand-bookshelf_simple-v0
  humanoid_h1hand-door-v0
  humanoid_h1hand-spoon-v0
  humanoid_h1hand-pole-v0
)

seeds=(1 2 3)

# 与 run_train.sh 一致，使用 LEAF。
STEPS=1_001_000

# 日志目录（相对仓库根）
log_dir="${SCRIPT_DIR}/parallel_train_logs"
mkdir -p "${log_dir}"

# GPU：总卡数 × 每卡最多并行作业数 = 最大并发
NUM_GPUS=8
MAX_TASKS_PER_GPU=2
MAX_CONCURRENT=$((NUM_GPUS * MAX_TASKS_PER_GPU))
# ==============================

EXP_NAME=leaf

current_jobs() {
  jobs -rp | wc -l
}

task_id=0
for task in "${tasks[@]}"; do
  for seed in "${seeds[@]}"; do
    gpu=$(( (task_id / MAX_TASKS_PER_GPU) % NUM_GPUS ))
    while [ "$(current_jobs)" -ge "${MAX_CONCURRENT}" ]; do
      sleep 1
    done
    safe_task="${task//\//_}"
    log_file="${log_dir}/train_${safe_task}_seed${seed}.log"
    echo "Launching task=${task} seed=${seed} exp_name=${EXP_NAME} on GPU ${gpu} -> ${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
      "task=${task}" \
      "seed=${seed}" \
      "flow=true" \
      "exp_name=${EXP_NAME}" \
      "steps=${STEPS}" \
      > "${log_file}" 2>&1 &
    task_id=$((task_id + 1))
  done
done

wait
echo "All parallel training jobs finished."
