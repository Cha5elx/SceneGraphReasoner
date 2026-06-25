# ========================= data ==========================
anno_root = __import__("os").environ.get(
    "SGR_ANNO_ROOT",
    "/data/ZXMIC/mic_lcx/ScaneGraphReasoning/SGR/annotations",
)
pc_encoder = "clasp"       # CLASP 编码器
segmentor = "clasp"        # CLASP 分割器
version = ""

# CLASP 特征和属性文件
feat_file           = f"{anno_root}/scannet_clasp_clasp_feats.pt"
img_feat_file       = f"{anno_root}/scannet_clasp_videofeats.pt"
train_attr_file     = f"{anno_root}/scannet_clasp_train_attributes.pt"
val_attr_file       = f"{anno_root}/scannet_clasp_val_attributes.pt"

# 兼容旧变量名
seg_feat_file       = feat_file
seg_img_feat_file   = img_feat_file
seg_train_attr_file = train_attr_file
seg_val_attr_file   = val_attr_file

train_tag = 'scanqa'
val_tag = 'scanqa'

train_file_dict = {
    # === CLASP 预提取数据可用的任务 ===
    'scanrefer': [
        feat_file,
        img_feat_file,
        train_attr_file,
        f"{anno_root}/scanrefer_clasp_train.json"
    ],
    'scanrefer_cot': [  # 带 CoT 推理链的 ScanRefer
        feat_file,
        img_feat_file,
        train_attr_file,
        f"{anno_root}/scanrefer_cot_clasp_train.json"
    ],
    'scan2cap': [
        feat_file,
        img_feat_file,
        train_attr_file,
        f"{anno_root}/scan2cap_clasp_train.json"
    ],
    'nr3d_caption': [
        feat_file,
        img_feat_file,
        train_attr_file,
        f"{anno_root}/nr3d_caption_clasp_train.json"
    ],
    'obj_align': [
        feat_file,
        img_feat_file,
        train_attr_file,
        f"{anno_root}/obj_align_clasp_train.json"
    ],
    'multi3dref': [
        feat_file,
        img_feat_file,
        train_attr_file,
        f"{anno_root}/multi3dref_clasp_train.json"
    ],
    'scanqa': [
        feat_file,
        img_feat_file,
        train_attr_file,
        f"{anno_root}/scanqa_train.json"
    ],
    'scanqa_cot': [  # 带 CoT 推理链的 ScanQA
        feat_file,
        img_feat_file,
        train_attr_file,
        f"{anno_root}/scanqa_cot_clasp_train.json"
    ],
    'sqa3d': [
        feat_file,
        img_feat_file,
        train_attr_file,
        f"{anno_root}/sqa3d_train.json"
    ],
    'scannet_caption': [
        feat_file,
        img_feat_file,
        train_attr_file,
        f"{anno_root}/scannet_caption_clasp_train.json"
    ],
    'scannet_region_caption': [
        feat_file,
        img_feat_file,
        train_attr_file,
        f"{anno_root}/scannet_region_caption_clasp_train.json"
    ],
}

val_file_dict = {
    'scanrefer': [
        feat_file,
        img_feat_file,
        val_attr_file,
        f"{anno_root}/scanrefer_clasp_val.json"
    ],
    'scanrefer_cot': [
        feat_file,
        img_feat_file,
        val_attr_file,
        f"{anno_root}/scanrefer_cot_clasp_val.json"
    ],
    'scan2cap': [
        feat_file,
        img_feat_file,
        val_attr_file,
        f"{anno_root}/scan2cap_clasp_val.json"
    ],
    'multi3dref': [
        feat_file,
        img_feat_file,
        val_attr_file,
        f"{anno_root}/multi3dref_clasp_val.json"
    ],
    'scanqa': [
        feat_file,
        img_feat_file,
        val_attr_file,
        f"{anno_root}/scanqa_val.json"
    ],
    'scanqa_cot': [
        feat_file,
        img_feat_file,
        val_attr_file,
        f"{anno_root}/scanqa_cot_clasp_val.json"
    ],
    'sqa3d': [
        feat_file,
        img_feat_file,
        val_attr_file,
        f"{anno_root}/sqa3d_val.json"
    ],
}


num_workers = 32
batch_size = 32


# ========================= model ==========================
model = dict(
    # ===== LLM 基座 =====
    llama_model_path=__import__("os").environ.get(
        "SGR_LLM_PATH",
        "/data/ZXMIC/mic_lcx/ScaneGraphReasoning/llm/Qwen3.5-2B",
    ),
    low_resource=False,
    system_path="prompts/system.txt",
    instruction_path="prompts/instruction.txt",
    max_txt_len=128,  # 训练时的最大文本长度（推理链需要更长输出）
    seq_len_cap=1280,  # LLM 输入序列最大长度
    gen_max_txt_len=32,  # 评估时 generate 的最大 token 数（短答案32够用，正式CoT评估用128）
    num_beams=1,       # 验证阶段 beam 数（1=greedy快速验证，正式评估用5）
    end_sym="</s>",
    role=("USER", "ASSISTANT"),

    # ===== 特征维度 =====
    input_dim=128,       # CLASP 3D 特征 (原 Uni3D 为 1024)
    img_input_dim=1024,  # DINOv2 2D 特征
    attr_dim=512,        # 关系特征 (VL-SAT / MLP)
    scene_dim=256,       # 场景级嵌入
    pos_dim=128,         # 位置编码维度

    # ===== Query-conditioned residual graph reasoning (SGR) =====
    use_scene_graph=False,    # scripts/run.sh 中为图推理实验显式开启
    sg_candidate_k=6,         # 每个节点的几何候选邻居数
    sg_effective_k=2,         # 推理时按 query edge score 保留的邻居数
    sg_hidden_dim=128,
    sg_bbox_eps=1e-6,         # 零尺寸 bbox 不参与图构建
    sg_hard_prune_eval=True,  # 训练 soft gating，评估 hard top-k
    sg_residual_scale=1.0,    # scale graph residual before adding to object tokens
    sg_use_query_gating=True, # False = fixed GNN ablation
    sg_diagnostics=True,      # log query sensitivity and edge ranking diagnostics
    sg_aux_object_loss_weight=0.0, # target-aware object relevance auxiliary loss
    sg_object_topm=0,         # 0 disables hard top-M residual masking
    sg_use_object_gate=False, # gate graph residuals with query-object relevance

    # ===== 模块开关 =====
    add_scene_token=False,    # SGR使用场景图替代原Transformer场景token
    add_img_token=True,
    use_lora=True,
    train_emb=True,
    train_img_proj=False,
    train_graph_only=False,   # True = freeze everything except query_graph_reasoner
    no_obj=False,
    max_obj_num=100,          # CLASP 固定 100 个物体
    bidirection=False,
    add_pos_emb=False,
    feat_fusion=False,
    fuse_with_id=False,
    use_objid=True,
    use_location_token=False,
    encoder_num_layers=3,     # 未使用 (add_scene_token=False)
)

lora = dict(
    lora_target_modules=[
      "q_proj",
      "v_proj",
      "k_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj"
    ],
    lora_r=64,
    lora_alpha=16,
    lora_dropout=0.05
)

optimizer = dict(
    opt="adamW",
    lr=5e-6,  # Base LR; scaled by batch_size * gpu_num * gradient_accumulation_steps
    opt_betas=[0.9, 0.999],  # default
    weight_decay=0.02,
    scaler_enable=False,
    gradient_accumulation_steps=1,
    max_grad_norm=5,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(
        enable=False,
        module_names=["model.embed_tokens"],
        lr=[5e-4],
        wd=[0.02]
    ),
)

scheduler = dict(sched="cosine", epochs=3, min_lr_multi=0.0, warmup_epochs=0.1)

evaluate = False

# ========================= wandb ==========================
wandb = dict(
    enable=False,
    entity="huanghaifeng",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="Scene-LLM",
)
dist_url = "env://"
device = "cuda"

# ========================= others ==========================
output_dir = "outputs/tmp"  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 20
eval_freq = 0  # 0表示仅在epoch结束时评估一次
# eval_freq = 500
seed = 42

save_latest = False
do_save = True
auto_resume = True
pretrained_path = ""
img_projector_path = ""

debug=False
gpu_num=1
