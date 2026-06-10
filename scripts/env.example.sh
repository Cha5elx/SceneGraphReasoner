# Example local run environment.
# Usage:
#   RUN_ENV=scripts/env.5090.local.sh bash scripts/run.sh
#   RUN_ENV=scripts/env.4090.local.sh bash scripts/run.sh

export CUDA_VISIBLE_DEVICES=0
export SGR_ANNO_ROOT="/path/to/SGR/annotations"
export SGR_LLM_PATH="/path/to/Qwen3.5-2B"

gpu_num=1
batch_size=8
gradient_accumulation_steps=1
lr=5e-6
other_info="sgr"
