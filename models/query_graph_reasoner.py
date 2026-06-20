"""面向 CLASP 物体特征的查询条件化残差图推理模块。

该模块把图计算放在 LLM token 序列之外：在线构建一个小型几何候选图，
用当前问题对边进行路由和加权，执行一轮消息传播，最后返回可加到
3D 物体特征上的残差。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryEncoder(nn.Module):
    """将冻结的 LLM token embedding 池化成一个轻量级 query 向量。"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.token_proj = nn.Linear(input_dim, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)

    def forward(self, token_embeds, token_mask):
        """
        参数:
            token_embeds: [B, T, D_text] 冻结的 LLM token embedding。
            token_mask:   [B, T] 有效文本 token 的 mask。

        返回:
            query_embed: [B, D_graph]。
        """
        token_feats = torch.tanh(self.token_proj(token_embeds))
        scores = self.attn_score(token_feats).squeeze(-1)
        scores = scores.masked_fill(~token_mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * token_mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.sum(token_feats * weights.unsqueeze(-1), dim=1)


class QueryGraphReasoner(nn.Module):
    """用针对当前查询的稀疏证据图增强 3D 物体特征。

    输入:
        scene_feat:         [B, N, D_clasp] 已归一化的 CLASP 物体特征。
        scene_locs:         [B, N, 6] box 信息，前 3 维为中心点 xyz，后 3 维为尺寸 whl。
        scene_mask:         [B, N] baseline 使用的物体有效性 mask。
        query_token_embeds: [B, T, D_text] 冻结的 LLM token embedding。
        query_token_mask:   [B, T] 有效 query token 的 mask。

    输出:
        graph_residual: [B, N, output_dim] 投影前加到物体特征上的缩放残差。
        graph_info:     用于训练监控和后续图可视化的张量字典。

    零尺寸 box 仍然保留给 baseline 的 LLM 物体序列使用，但会被排除在图构建之外，
    因为它们不携带可靠的几何信息。
    """

    EDGE_GEOMETRY_NAMES = (
        "delta_x",
        "delta_y",
        "delta_z",
        "distance",
        "size_ratio_w",
        "size_ratio_h",
        "size_ratio_l",
        "iou_3d",
        "overlap_xy",
        "gap_z",
        "clasp_cosine",
    )
    EDGE_GEOMETRY_DIM = len(EDGE_GEOMETRY_NAMES)

    def __init__(
        self,
        feat_dim,
        query_input_dim,
        output_dim,
        hidden_dim=128,
        candidate_k=6,
        effective_k=2,
        bbox_eps=1e-6,
        hard_prune_eval=True,
        residual_scale=1.0,
        use_query_gating=True,
    ):
        super().__init__()
        if candidate_k < 1:
            raise ValueError("candidate_k must be positive")
        if effective_k < 1 or effective_k > candidate_k:
            raise ValueError("effective_k must be in [1, candidate_k]")

        self.candidate_k = candidate_k
        self.effective_k = effective_k
        self.bbox_eps = bbox_eps
        self.hard_prune_eval = hard_prune_eval
        self.residual_scale = float(residual_scale)
        self.use_query_gating = use_query_gating

        self.query_encoder = QueryEncoder(query_input_dim, hidden_dim)
        self.node_encoder = nn.Sequential(
            nn.Linear(feat_dim + 6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_router = nn.Sequential(
            nn.Linear(hidden_dim * 3 + self.EDGE_GEOMETRY_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim + self.EDGE_GEOMETRY_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message_out = nn.Linear(hidden_dim, hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.residual_proj = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)

    def forward(
        self,
        scene_feat,
        scene_locs,
        scene_mask,
        query_token_embeds,
        query_token_mask,
        hard_prune=None,
    ):
        """执行一次 query-conditioned 图推理，并返回可加到物体特征上的残差。

        整体思路:
        1. 先根据 bbox 几何有效性筛掉不能进图的节点；
        2. 把物体 CLASP 特征和归一化 bbox 编码成图节点特征；
        3. 用问题文本的 LLM token embedding 池化出一个 query 向量；
        4. 按空间距离为每个物体找 candidate_k 个候选邻居；
        5. 对每条候选边计算 query-aware edge score；
        6. 用边分数加权聚合邻居 message，得到 refined node feature；
        7. 将 refined feature 投回 CLASP 维度，作为 residual 返回给 Chat3D。
        """
        # 只让 baseline 中有效、且 bbox 尺寸非零的物体参与图构建。
        # 注意: 被过滤的节点仍可进入后续 LLM object token 序列，只是不参与图消息传递。
        graph_node_mask = self._build_graph_node_mask(scene_locs, scene_mask)

        # 将每个物体的中心点和尺寸按当前场景尺度归一化。
        # normalized_boxes: [B, N, 6]，scene_scale: [B, 1]，后续边几何也复用该尺度。
        normalized_boxes, scene_scale = self._normalize_boxes(
            scene_locs, graph_node_mask
        )

        # 节点输入由两部分组成:
        # 1) L2 归一化后的 CLASP 物体语义特征；
        # 2) 归一化后的 3D box 几何特征。
        node_input = torch.cat(
            [F.normalize(scene_feat, dim=-1), normalized_boxes], dim=-1
        )

        # node_encoder 将 [128 + 6] 维输入映射到图推理隐藏维 hidden_dim。
        # node_feats: [B, N, H]，是后续边打分和消息传递的节点表示。
        node_feats = self.node_encoder(node_input)

        # 把 query 的多个 LLM token embedding 池化成一个图路由向量。
        # query_embed: [B, H]，每个 batch 样本一个 query 表示。
        query_embed = self.query_encoder(query_token_embeds, query_token_mask)

        # 基于物体中心点距离构造候选边。
        # target_indices: [B, N, K]，表示每个源节点的 K 个候选目标节点编号。
        # candidate_mask: [B, N, K]，标记该候选边是否真实有效，避免 topk 中的 inf 占位边参与计算。
        target_indices, candidate_mask = self._build_candidate_edges(
            scene_locs[..., :3], graph_node_mask
        )

        # 根据 target_indices 把目标邻居节点特征取出来。
        # neighbor_feats[b, i, k] 就是 batch b 中源节点 i 的第 k 个候选邻居的图特征。
        neighbor_feats = self._gather_neighbors(node_feats, target_indices)

        # 为每条候选边计算 11 维边几何/语义特征:
        # 相对位移、距离、尺寸比例、3D IoU、XY 重叠、Z 间隔、CLASP 余弦相似度。
        edge_geometry = self._compute_edge_geometry(
            scene_feat, scene_locs, target_indices, scene_scale
        )

        # 从 target_indices 读出 batch 大小、节点数和每个节点的候选边数 K。
        batch_size, obj_num, edge_num = target_indices.shape

        # 将源节点特征扩展到边维度，方便和每条候选边的目标节点特征拼接。
        # source_feats: [B, N, K, H]，第 K 维上复制同一个源节点特征。
        source_feats = node_feats.unsqueeze(2).expand(-1, -1, edge_num, -1)

        # 将 query 向量扩展到每条边上，使每条边的打分都能感知当前问题。
        # edge_query: [B, N, K, H]。
        edge_query = query_embed[:, None, None, :].expand(
            -1, obj_num, edge_num, -1
        )

        # edge_router 的输入 = 源节点 + 目标节点 + 边几何 + query。
        # 这一步体现“查询条件图”: 同一场景、同一对物体，在不同 query 下可以得到不同边权重。
        router_input = torch.cat(
            [source_feats, neighbor_feats, edge_geometry, edge_query], dim=-1
        )

        # 正常 SGR 模式: edge_router 输出每条候选边的 logit，再经 sigmoid 得到 [0,1] 权重。
        if self.use_query_gating:
            edge_scores = torch.sigmoid(self.edge_router(router_input).squeeze(-1))

            # 将无效候选边的分数归零，确保 padding/无效几何边不会传递消息。
            edge_scores = edge_scores * candidate_mask.to(edge_scores.dtype)
        else:
            # 消融实验模式: 不使用 query-aware gating，所有候选边统一权重为 1。
            # 这等价于固定几何邻接图上的普通 message passing。
            edge_scores = candidate_mask.to(node_feats.dtype)

        # hard_prune 默认策略:
        # - 训练阶段: False，保留所有候选边，用 soft edge score 学习；
        # - 评估阶段: 若 hard_prune_eval=True，则每个源节点只保留 top effective_k 条边。
        if hard_prune is None:
            hard_prune = self.hard_prune_eval and not self.training

        # active_edge_mask 标记最终参与聚合的边。
        # 训练 soft 模式下它等于 candidate_mask；hard 模式下它是每个源节点的 top-k 边。
        active_edge_mask = self._select_active_edges(
            edge_scores, candidate_mask, hard_prune
        )

        # 最终边权重 = query-aware 分数 * active mask。
        # active mask 为 False 的边权重为 0，不影响后续加权求和。
        edge_weights = edge_scores * active_edge_mask.to(edge_scores.dtype)

        # 为每条候选边生成 message。
        # message 的来源是目标邻居节点特征和边几何，不直接拼 query；
        # query 已经通过 edge_scores 控制“哪条边重要、重要多少”。
        messages = self.message_mlp(
            torch.cat([neighbor_feats, edge_geometry], dim=-1)
        )

        # 对每个源节点的所有激活边权重求和。
        # clamp_min 避免某个节点没有有效边时除以 0。
        weight_sum = edge_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        # 按边权重对邻居 message 做加权求和，dim=2 是候选邻居维度 K。
        aggregated = torch.sum(messages * edge_weights.unsqueeze(-1), dim=2)

        # 将加权和除以权重和，得到每个源节点的平均邻居证据。
        aggregated = aggregated / weight_sum

        # 残差式节点更新:
        # 原节点表示 node_feats 加上聚合后的邻居信息，再做 LayerNorm 稳定尺度。
        refined_feats = self.node_norm(node_feats + self.message_out(aggregated))

        # 无效图节点的 refined feature 清零，避免后续 residual_proj 为它们生成残差。
        refined_feats = refined_feats * graph_node_mask.unsqueeze(-1).to(
            refined_feats.dtype
        )

        # 将 refined graph hidden feature 先 L2 归一化，再投回 output_dim。
        # residual_proj 在 __init__ 中零初始化，因此训练初始时 graph_residual 为 0，
        # 模型从 baseline 行为开始，逐渐学习如何使用图信息。
        raw_graph_residual = self.residual_proj(F.normalize(refined_feats, dim=-1))

        # 再次 mask 无效节点，保证无效几何节点不会向 Chat3D 的 object feature 注入残差。
        raw_graph_residual = raw_graph_residual * graph_node_mask.unsqueeze(-1).to(
            raw_graph_residual.dtype
        )

        # residual_scale 控制图残差注入强度，run.sh 中通常设得较小以稳定训练。
        graph_residual = raw_graph_residual * self.residual_scale

        # 构造显式 source index，用于和 target_indices 组合成边索引。
        # 这里每个源节点 i 都对应 K 个 target，所以 source_indices 形状为 [B, N, K]。
        source_indices = torch.arange(
            obj_num, device=target_indices.device, dtype=torch.long
        )
        source_indices = source_indices.view(1, obj_num, 1).expand(
            batch_size, -1, edge_num
        )

        # edge_indices[b, i, k] = [source_i, target_j]，主要用于日志和后续可视化。
        edge_indices = torch.stack([source_indices, target_indices], dim=-1)

        # 每个样本的候选边数量，统计时按 batch 平均。
        candidate_edge_count = candidate_mask.sum(dim=(1, 2)).to(edge_scores.dtype)

        # 每个样本最终激活边数量，soft 训练时通常等于候选边数量。
        active_edge_count = active_edge_mask.sum(dim=(1, 2)).to(edge_scores.dtype)

        # 每个样本的平均 edge score，只在候选边范围内归一化。
        mean_edge_score_per_sample = edge_scores.sum(dim=(1, 2)).div(
            candidate_edge_count.clamp_min(1)
        )

        # batch 级平均边分数，用于训练日志观察 query gating 是否塌缩。
        mean_edge_score = mean_edge_score_per_sample.mean()

        # 计算每个样本 edge score 的平方均值，为方差/标准差做准备。
        edge_score_sq_mean = edge_scores.pow(2).sum(dim=(1, 2)).div(
            candidate_edge_count.clamp_min(1)
        )

        # E[x^2] - E[x]^2 得到方差；clamp_min 防止数值误差导致负数开方。
        edge_score_std = (
            edge_score_sq_mean - mean_edge_score_per_sample.pow(2)
        ).clamp_min(0).sqrt().mean()

        # graph_info 不参与主前向输出的 object feature 加法，主要用于:
        # 1) 训练日志监控；
        # 2) 分析/可视化 query-conditioned 边选择；
        # 3) debug 图节点、候选边和 residual 的统计。
        graph_info = {
            # 有效图节点 mask: [B, N]。
            "node_mask": graph_node_mask,
            # 显式边索引: [B, N, K, 2]，最后一维是 [source, target]。
            "edge_indices": edge_indices,
            # 几何 KNN 得到的候选边 mask: [B, N, K]。
            "candidate_edge_mask": candidate_mask,
            # 实际参与消息聚合的边 mask: [B, N, K]。
            "active_edge_mask": active_edge_mask,
            # query-aware 边分数: [B, N, K]。
            "edge_scores": edge_scores,
            # 最终消息聚合权重: [B, N, K]。
            "edge_weights": edge_weights,
            # 池化后的 query 表示: [B, H]。
            "query_embed": query_embed,
            # 缩放后 residual 的平均范数。
            "residual_norm": graph_residual.norm(dim=-1).mean(),
            # 缩放前 residual 的平均范数，用于观察 residual_scale 的影响。
            "raw_residual_norm": raw_graph_residual.norm(dim=-1).mean(),
            # 当前残差缩放系数，以 tensor 形式放入日志系统。
            "residual_scale": torch.tensor(
                self.residual_scale,
                dtype=graph_residual.dtype,
                device=graph_residual.device,
            ),
            # 平均有效图节点数。
            "valid_node_count": graph_node_mask.sum(dim=1).float().mean(),
            # 平均候选边数。
            "candidate_edge_count": candidate_edge_count.mean(),
            # 平均激活边数。
            "active_edge_count": active_edge_count.mean(),
            # 平均边分数。
            "mean_edge_score": mean_edge_score,
            # 边分数标准差。
            "edge_score_std": edge_score_std,
        }

        # 返回给 Chat3D 的只有 graph_residual；graph_info 作为日志/可视化辅助信息。
        return graph_residual, graph_info

    def _build_graph_node_mask(self, scene_locs, scene_mask):
        """筛选真正可以参与图构建的节点。

        scene_mask 表示数据层面的有效物体；valid_geometry 进一步要求 bbox 的
        w/h/l 都大于 bbox_eps。这样可以避免零尺寸 box 在距离、IoU、尺寸比例中
        产生不稳定的几何特征。
        """
        # scene_locs[..., 3:] 是 bbox 尺寸 whl；三个维度都有效才算几何有效。
        valid_geometry = (scene_locs[..., 3:] > self.bbox_eps).all(dim=-1)

        # 图节点必须同时满足: 数据 mask 有效 + 几何尺寸有效。
        return scene_mask.bool() & valid_geometry

    def _normalize_boxes(self, scene_locs, graph_node_mask):
        """按场景尺度归一化 box 中心和尺寸。

        归一化的目的: 不同 ScanNet 场景的绝对坐标范围不同，如果直接把原始坐标
        拼到节点特征里，图模块会更难学习可迁移的空间关系。
        """
        # centers: [B, N, 3]，每个物体 bbox 中心点。
        centers = scene_locs[..., :3]

        # sizes: [B, N, 3]，每个物体 bbox 尺寸 whl。
        sizes = scene_locs[..., 3:]

        # 扩展 mask 到最后一维，使它能和 centers/sizes 做逐元素选择。
        valid = graph_node_mask.unsqueeze(-1)

        # 对无效节点填 +inf，这样 amin 时它们不会影响场景最小坐标。
        mins = torch.where(valid, centers, torch.full_like(centers, float("inf")))

        # 对无效节点填 -inf，这样 amax 时它们不会影响场景最大坐标。
        maxs = torch.where(
            valid, centers, torch.full_like(centers, float("-inf"))
        )

        # 每个 batch 样本内，所有有效节点中心点的最小 xyz。
        mins = mins.amin(dim=1)

        # 每个 batch 样本内，所有有效节点中心点的最大 xyz。
        maxs = maxs.amax(dim=1)

        # 标记该样本是否至少有一个有效图节点。
        has_valid_node = graph_node_mask.any(dim=1, keepdim=True)

        # 如果整张图没有有效节点，则用 0 兜底，避免 inf 进入后续计算。
        mins = torch.where(has_valid_node, mins, torch.zeros_like(mins))
        maxs = torch.where(has_valid_node, maxs, torch.zeros_like(maxs))

        # scene_scale 是场景对角线长度，作为归一化尺度；clamp 防止除以 0。
        scene_scale = (maxs - mins).norm(dim=-1, keepdim=True).clamp_min(
            self.bbox_eps
        )

        # 场景中心点，用于把坐标平移到以场景为中心的局部坐标系。
        scene_center = (mins + maxs) / 2

        # 中心点归一化: 先减场景中心，再除场景尺度。
        normalized_centers = (centers - scene_center.unsqueeze(1)) / (
            scene_scale.unsqueeze(1)
        )

        # 尺寸归一化: 直接除以场景尺度，保留相对大小。
        normalized_sizes = sizes / scene_scale.unsqueeze(1)

        # 返回 [normalized xyz; normalized whl] 以及 scene_scale。
        return torch.cat([normalized_centers, normalized_sizes], dim=-1), scene_scale

    def _build_candidate_edges(self, centers, graph_node_mask):
        """用中心点 KNN 为每个源节点构造候选邻居边。"""
        # obj_num 是当前 batch padding 后的最大物体数 N。
        _, obj_num, _ = centers.shape

        # 每个节点最多连 candidate_k 条边；如果节点数不足，则不能超过 N-1。
        # max(..., 1) 保证 topk 的 k 至少为 1，即使 N=1 也能保持张量形状稳定。
        edge_num = min(self.candidate_k, max(obj_num - 1, 1))

        # 计算所有物体中心之间的欧氏距离: [B, N, N]。
        pairwise_dist = torch.cdist(centers, centers)

        # 只有 source 和 target 都是有效图节点时，这对节点才允许成边。
        valid_pairs = graph_node_mask.unsqueeze(2) & graph_node_mask.unsqueeze(1)

        # 构造单位矩阵 mask，用于去掉自环 i -> i。
        eye = torch.eye(obj_num, dtype=torch.bool, device=centers.device)

        # 去掉自环后，valid_pairs[b, i, j] 表示 i -> j 是否可作为候选边。
        valid_pairs = valid_pairs & ~eye.unsqueeze(0)

        # 无效 pair 的距离设为 inf，这样 topk 最近邻不会优先选到它们。
        pairwise_dist = pairwise_dist.masked_fill(~valid_pairs, float("inf"))

        # 对每个源节点取距离最近的 edge_num 个目标节点。
        # largest=False 表示取最小距离，也就是 KNN。
        distances, target_indices = torch.topk(
            pairwise_dist, k=edge_num, dim=-1, largest=False
        )

        # 如果某个 topk 结果距离是 inf，说明它只是占位，不是真实候选边。
        return target_indices, torch.isfinite(distances)

    def _compute_edge_geometry(
        self, scene_feat, scene_locs, target_indices, scene_scale
    ):
        """计算每条候选边的显式几何和语义关系特征。

        输出维度对应 EDGE_GEOMETRY_NAMES:
        [delta_x, delta_y, delta_z, distance,
         size_ratio_w, size_ratio_h, size_ratio_l,
         iou_3d, overlap_xy, gap_z, clasp_cosine]
        """
        # edge_num = K，每个源节点的候选邻居数量。
        edge_num = target_indices.shape[-1]

        # 将源节点 box 复制到候选边维度: [B, N, K, 6]。
        source_boxes = scene_locs.unsqueeze(2).expand(-1, -1, edge_num, -1)

        # 根据 target_indices 取目标邻居 box: [B, N, K, 6]。
        target_boxes = self._gather_neighbors(scene_locs, target_indices)

        # 将源节点 CLASP 特征复制到候选边维度: [B, N, K, D]。
        source_feat = scene_feat.unsqueeze(2).expand(-1, -1, edge_num, -1)

        # 根据 target_indices 取目标邻居 CLASP 特征: [B, N, K, D]。
        target_feat = self._gather_neighbors(scene_feat, target_indices)

        # 拆分源 box 的中心点和尺寸。
        source_center, source_size = source_boxes[..., :3], source_boxes[..., 3:]

        # 拆分目标 box 的中心点和尺寸。
        target_center, target_size = target_boxes[..., :3], target_boxes[..., 3:]

        # scene_scale 扩展成 [B, 1, 1, 1]，方便广播到每条边。
        scale = scene_scale[:, None, None, :]

        # 目标中心相对源中心的位移，并按场景尺度归一化。
        delta = (target_center - source_center) / scale

        # 归一化相对位移的 L2 范数，即源到目标的相对距离。
        distance = delta.norm(dim=-1, keepdim=True)

        # 目标 box 尺寸 / 源 box 尺寸，再取 log。
        # log ratio 比直接 ratio 更对称: target 比 source 大/小时分别为正/负。
        size_ratio = torch.log(
            (target_size + self.bbox_eps) / (source_size + self.bbox_eps)
        )

        # 源 box 的最小/最大角点坐标。
        source_min = source_center - source_size / 2
        source_max = source_center + source_size / 2

        # 目标 box 的最小/最大角点坐标。
        target_min = target_center - target_size / 2
        target_max = target_center + target_size / 2

        # 两个 box 在 xyz 三个方向上的交集尺寸；若不相交则 clamp 到 0。
        intersection_size = (
            torch.minimum(source_max, target_max)
            - torch.maximum(source_min, target_min)
        ).clamp_min(0)

        # 3D 交集体积。
        intersection_volume = intersection_size.prod(dim=-1, keepdim=True)

        # 源/目标 box 体积；clamp_min 防止异常负尺寸污染体积。
        source_volume = source_size.clamp_min(0).prod(dim=-1, keepdim=True)
        target_volume = target_size.clamp_min(0).prod(dim=-1, keepdim=True)

        # 3D IoU = intersection / union。
        iou = intersection_volume / (
            source_volume + target_volume - intersection_volume + self.bbox_eps
        )

        # XY 平面交集面积，用于描述俯视图上的重叠程度。
        intersection_xy = intersection_size[..., :2].prod(dim=-1, keepdim=True)

        # 源/目标在 XY 平面的面积。
        source_xy = source_size[..., :2].clamp_min(0).prod(dim=-1, keepdim=True)
        target_xy = target_size[..., :2].clamp_min(0).prod(dim=-1, keepdim=True)

        # XY overlap 使用较小 XY 面积归一化，强调“投影是否覆盖/贴近”。
        overlap_xy = intersection_xy / (
            torch.minimum(source_xy, target_xy) + self.bbox_eps
        )

        # 目标 box 底部到源 box 顶部的 z 方向间隔；正值通常表示目标在源上方。
        gap_z = (target_min[..., 2:3] - source_max[..., 2:3]) / scale

        # CLASP 语义特征余弦相似度，为边提供语义相近性信息。
        cosine = F.cosine_similarity(source_feat, target_feat, dim=-1).unsqueeze(-1)

        # 拼成固定 11 维 edge geometry，供 edge_router 和 message_mlp 使用。
        return torch.cat(
            [delta, distance, size_ratio, iou, overlap_xy, gap_z, cosine], dim=-1
        )

    def _select_active_edges(self, edge_scores, candidate_mask, hard_prune):
        """根据是否 hard prune 选择最终参与消息传递的边。"""
        # soft 模式: 所有候选边都参与，只是权重由 edge_scores 控制。
        if not hard_prune:
            return candidate_mask

        # hard 模式: 每个源节点最多保留 effective_k 条分数最高的边。
        keep_num = min(self.effective_k, edge_scores.shape[-1])

        # 无效候选边分数设为 -inf，防止 topk 选中 padding/无效边。
        masked_scores = edge_scores.masked_fill(~candidate_mask, float("-inf"))

        # 在每个源节点的 K 条候选边中取 top effective_k。
        top_indices = torch.topk(masked_scores, k=keep_num, dim=-1).indices

        # 初始化全 False mask，然后把 topk 位置 scatter 成 True。
        active_mask = torch.zeros_like(candidate_mask)
        active_mask.scatter_(-1, top_indices, True)

        # 再与 candidate_mask 相与，避免全 inf 场景中 topk 的占位 index 被误当有效边。
        return active_mask & candidate_mask

    @staticmethod
    def _gather_neighbors(features, target_indices):
        """按 target_indices 从 features 中批量取邻居特征。

        features: [B, N, D]
        target_indices: [B, N, K]
        return: [B, N, K, D]
        """
        # 从 target_indices 读取 batch、源节点数 N、候选邻居数 K。
        batch_size, obj_num, edge_num = target_indices.shape

        # 最后一维特征维度 D，可以是节点 hidden_dim、scene_locs 的 6，或 CLASP 的 128。
        feat_dim = features.shape[-1]

        # 把 features 扩展成 [B, N_source, N_target, D]。
        # 第 2 维的 N_source 表示“为每个源节点都准备一份完整 target 表”。
        expanded_features = features.unsqueeze(1).expand(-1, obj_num, -1, -1)

        # 将 target_indices 扩展到特征维，作为 torch.gather 的 index。
        expanded_indices = target_indices.unsqueeze(-1).expand(
            batch_size, obj_num, edge_num, feat_dim
        )

        # 在 target 维 dim=2 上 gather，得到每个源节点对应 K 个目标邻居的特征。
        return torch.gather(expanded_features, dim=2, index=expanded_indices)
