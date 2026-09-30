#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/data/cosmos/hf_cache
export VLLM_CONFIG_ROOT=/data/david/.config/vllm
export VLLM_CACHE_ROOT=/data/david/.cache/vllm

MODEL="${1:-nvidia/Cosmos-Reason1-7B}"
source "/data/cosmos/cosmos-transfer2.5/prompt_processor/.venv/bin/activate"

# Replace this Bash process with vLLM.
# This makes signal handling and cleanup simpler.
exec vllm serve "${MODEL}" \
  --port 8000 \
  --runner generate \
  --gpu-memory-utilization 0.80 \
  --max-model-len 72000 \
  --tensor-parallel-size 2 \
  --enforce-eager \
  --max-num-seqs 1 \
  --max-num-batched-tokens 72000 \
  --allowed-local-media-path /data/cosmos/ \
  --media-io-kwargs '{"video": {"backend": "torchcodec", "seek_mode": "exact", "frame_recovery": false}}'
