# 直接跑（前台）
# CUDA_VISIBLE_DEVICES=1 bash scripts/run.sh

# 后台跑
# CUDA_VISIBLE_DEVICES=1 setsid bash scripts/run.sh &
# CUDA_VISIBLE_DEVICES=1 setsid bash scripts/run.sh > /dev/null 2> output.log &

# 看日志
# tail -f outputs/20260513_*/train.log

# killall -9 -u mic_lcx
# pkill -9 -u mic_lcx

which_python=$(which python)
export PYTHONPATH=${PYTHONPATH}:${which_python}:.
echo "PYTHONPATH: ${PYTHONPATH}"

export MASTER_PORT=$((54000 + $RANDOM % 10000))
export MASTER_ADDR=localhost

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

load_run_env() {
    local env_file="$1"
    if [ ! -f "$env_file" ] && [ -f "${PROJECT_ROOT}/${env_file}" ]; then
        env_file="${PROJECT_ROOT}/${env_file}"
    fi
    if [ ! -f "$env_file" ]; then
        echo "Run env file not found: $1" >&2
        exit 1
    fi
    echo "Loading run env: ${env_file}"
    set -a
    source "$env_file"
    set +a
}

if [ -n "${RUN_ENV:-}" ]; then
    load_run_env "$RUN_ENV"
elif [ -f "${SCRIPT_DIR}/env.local.sh" ]; then
    load_run_env "${SCRIPT_DIR}/env.local.sh"
fi

# ========================= 训练参数（直接在这里改） =========================
epoch=${epoch:-1}
batch_size=${batch_size:-8}
gradient_accumulation_steps=${gradient_accumulation_steps:-1}
lr=${lr:-5e-6}

# ===== Qwen3.5-2B + CLASP =====
export SGR_ANNO_ROOT="${SGR_ANNO_ROOT:-/data/ZXMIC/mic_lcx/ScaneGraphReasoning/SGR/annotations}"
export SGR_LLM_PATH="${SGR_LLM_PATH:-/data/ZXMIC/mic_lcx/ScaneGraphReasoning/llm/Qwen3.5-2B}"
llama_model_path="${llama_model_path:-$SGR_LLM_PATH}"
input_dim=${input_dim:-128}
max_obj_num=${max_obj_num:-100}
pc_encoder="${pc_encoder:-clasp}"
segmentor="${segmentor:-clasp}"

# ===== SGR query-conditioned residual graph =====
use_scene_graph="${use_scene_graph:-True}"
sg_candidate_k=${sg_candidate_k:-6}
sg_effective_k=${sg_effective_k:-2}
sg_hidden_dim=${sg_hidden_dim:-128}
sg_hard_prune_eval=${sg_hard_prune_eval:-False}
sg_residual_scale=${sg_residual_scale:-0.1}
sg_use_query_gating=${sg_use_query_gating:-True}
seq_len_cap=${seq_len_cap:-1280}

# ===== LoRA =====
lora_r=${lora_r:-64}
lora_alpha=${lora_alpha:-16}

# ===== 模块开关 =====
train_emb=${train_emb:-True}
train_img_proj=${train_img_proj:-False}
add_img_token=${add_img_token:-True}
add_scene_token=${add_scene_token:-False}
no_obj=${no_obj:-False}
bidirection=${bidirection:-False}
add_pos_emb=${add_pos_emb:-False}
feat_fusion=${feat_fusion:-False}
fuse_with_id=${fuse_with_id:-False}
use_location_token=${use_location_token:-False}
different_lr=${different_lr:-False}

# ===== 训练数据 =====
# train_tag="scanqa"
# val_tag="scanqa"
# train_tag="scanrefer#obj_align#nr3d_caption#scan2cap#scanqa#sqa3d#multi3dref"
# val_tag="scanrefer#scanqa#sqa3d#multi3dref#scan2cap"
train_tag="${train_tag:-scanrefer#obj_align#nr3d_caption#scanqa}"
val_tag="${val_tag:-scanrefer#scanqa}"

# ===== 其他 =====
max_grad_norm=${max_grad_norm:-5}
seed=${seed:-42}
evaluate=${evaluate:-False}
pretrained_path="${pretrained_path:-}"

# ===== debug / 正常模式 =====
debug=${debug:-False}
if [ "$debug" = "True" ]; then
    enable_wandb=${enable_wandb:-False}
    gpu_num=${gpu_num:-1}
    do_save=${do_save:-False}
    other_info="${other_info:-debug}"
else
    enable_wandb=${enable_wandb:-False}
    gpu_num=${gpu_num:-1}
    do_save=${do_save:-True}
    other_info="${other_info:-sgr}"
fi
# ========================================================================

tag="${train_tag}__${val_tag}__${other_info}"
effective_batch_size=$((batch_size * gpu_num * gradient_accumulation_steps))
OUTPUT_DIR=outputs/"$(date +"%Y%m%d_%H%M%S")"_lr"$lr"_bs"$batch_size"_acc"$gradient_accumulation_steps"_ep"$epoch"_"$tag"
mkdir -p ${OUTPUT_DIR}

# 把 stdout+stderr 都记录到 OUTPUT_DIR
# exec > >(tee -a "${OUTPUT_DIR}/train.log") 2>&1
echo "Logging to: ${OUTPUT_DIR}/train.log"
echo "SGR_ANNO_ROOT: ${SGR_ANNO_ROOT}"
echo "SGR_LLM_PATH: ${SGR_LLM_PATH}"
echo "batch_size=${batch_size}, gradient_accumulation_steps=${gradient_accumulation_steps}, effective_batch_size=${effective_batch_size}"

ARGS=(
    "scripts/config.py"
    output_dir "$OUTPUT_DIR"
    scheduler.epochs "$epoch"
    optimizer.lr "$lr"
    model.add_scene_token "$add_scene_token"
    model.add_img_token "$add_img_token"
    pretrained_path "$pretrained_path"
    evaluate "$evaluate"
    wandb.enable "$enable_wandb"
    gpu_num "$gpu_num"
    do_save "$do_save"
    batch_size "$batch_size"
    optimizer.gradient_accumulation_steps "$gradient_accumulation_steps"
    model.train_emb "$train_emb"
    model.train_img_proj "$train_img_proj"
    train_tag "$train_tag"
    val_tag "$val_tag"
    model.no_obj "$no_obj"
    segmentor "$segmentor"
    pc_encoder "$pc_encoder"
    model.input_dim "$input_dim"
    model.bidirection "$bidirection"
    optimizer.different_lr.enable "$different_lr"
    model.max_obj_num "$max_obj_num"
    lora.lora_r "$lora_r"
    lora.lora_alpha "$lora_alpha"
    model.add_pos_emb "$add_pos_emb"
    model.feat_fusion "$feat_fusion"
    optimizer.max_grad_norm "$max_grad_norm"
    seed "$seed"
    model.fuse_with_id "$fuse_with_id"
    model.llama_model_path "$llama_model_path"
    model.use_location_token "$use_location_token"
    model.use_scene_graph "$use_scene_graph"
    model.sg_candidate_k "$sg_candidate_k"
    model.sg_effective_k "$sg_effective_k"
    model.sg_hidden_dim "$sg_hidden_dim"
    model.sg_hard_prune_eval "$sg_hard_prune_eval"
    model.sg_residual_scale "$sg_residual_scale"
    model.sg_use_query_gating "$sg_use_query_gating"
    model.seq_len_cap "$seq_len_cap"
)

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [ "$gpu_num" -gt 1 ]; then
        devices=$(seq -s, 0 $(($gpu_num - 1)))
        export CUDA_VISIBLE_DEVICES=$devices
        echo "Running on $gpu_num GPUs (CUDA_VISIBLE_DEVICES=$devices) with torchrun..."
        torchrun --nproc_per_node=${gpu_num} --master_port=${MASTER_PORT} tasks/train.py "${ARGS[@]}"
    else
        if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
            export CUDA_VISIBLE_DEVICES=0
            echo "Running on single GPU (CUDA_VISIBLE_DEVICES=0)..."
        else
            echo "Running on single GPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)..."
        fi
        python tasks/train.py "${ARGS[@]}"
    fi
fi
