"""场景图构建模块（MLP-based，替换 VL-SAT）

设计思路：
1. 基于物体中心坐标的 KNN 构建拓扑结构（不需要训练）
2. 用 MLP + 空间编码从 CLASP 128维特征中提取显式关系特征（需要训练）
3. CLASP 特征已隐式包含场景上下文，MLP 的作用是解耦 + 精确化 + 结构化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightRelationExtractor(nn.Module):
    """轻量级关系特征提取器。

    输入: 两个物体的 CLASP 特征 + 空间编码
    输出: 显式的关系特征向量

    与 3DGraphLLM 的区别：
    - 3DGraphLLM 使用 VL-SAT（预训练模型，512维关系特征，需离线预计算）
    - SGR 使用 MLP（从零学习，256维关系特征，在线计算）
    - 得益于 CLASP 的场景级上下文，物体特征已带隐式关系信息，MLP 足以解耦
    """

    def __init__(self, feat_dim=128, hidden_dim=512, rel_dim=256, spatial_dim=128):
        super().__init__()
        # 空间关系编码: [dx, dy, dz, dist, angle_h, angle_v] → spatial_dim
        self.spatial_encoder = nn.Sequential(
            nn.Linear(6, spatial_dim),
            nn.GELU(),
            nn.Linear(spatial_dim, spatial_dim),
        )
        # 关系特征提取: [feat_i; feat_j; spatial_enc] → rel_dim
        self.relation_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2 + spatial_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, rel_dim),
        )

    def compute_spatial_features(self, pos_i, pos_j):
        """计算两个物体中心点之间的空间特征。

        pos_i, pos_j: [..., 3] 物体中心 (x, y, z)

        返回: [..., 6] = [dx, dy, dz, dist, angle_xy, angle_z]
        """
        delta = pos_j - pos_i
        dist = torch.norm(delta, dim=-1, keepdim=True)
        angle_xy = torch.atan2(delta[..., 1], delta[..., 0]).unsqueeze(-1)
        angle_z = torch.atan2(
            delta[..., 2],
            torch.sqrt(delta[..., 0] ** 2 + delta[..., 1] ** 2 + 1e-8),
        ).unsqueeze(-1)
        return torch.cat([delta, dist, angle_xy, angle_z], dim=-1)

    def forward(self, feat_src, feat_tgt, pos_src, pos_tgt):
        """
        feat_src, feat_tgt: [..., 128] CLASP 3D 特征
        pos_src, pos_tgt:   [..., 3]   物体中心坐标

        返回: [..., 256] 关系特征
        """
        spatial = self.compute_spatial_features(pos_src, pos_tgt)
        spatial_enc = self.spatial_encoder(spatial)
        combined = torch.cat([feat_src, feat_tgt, spatial_enc], dim=-1)
        return self.relation_mlp(combined)


class SceneGraphBuilder(nn.Module):
    """场景图构建器。

    流程:
    1. 基于物体中心坐标做 KNN，确定图拓扑
    2. 对每条边调用 LightweightRelationExtractor，得到边特征
    3. 输出场景图: 关系特征 + 边索引
    """

    def __init__(
        self, feat_dim=128, rel_dim=256, spatial_dim=128, hidden_dim=512, k=2
    ):
        super().__init__()
        self.k = k
        self.rel_dim = rel_dim
        self.relation_extractor = LightweightRelationExtractor(
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            rel_dim=rel_dim,
            spatial_dim=spatial_dim,
        )

    def _build_knn_edges(self, obj_positions, scene_mask, k):
        """基于物体中心坐标构建 KNN 边。

        obj_positions: [N, 3] 物体中心 (x,y,z)
        scene_mask:    [N]   有效物体 mask

        返回: edge_index [E, 2] (src, tgt)
        """
        N = obj_positions.shape[0]
        device = obj_positions.device

        # 成对距离
        pairwise_dists = torch.cdist(
            obj_positions, obj_positions
        )  # [N, N]

        # 排除自身和过近物体
        pairwise_dists.fill_diagonal_(float("inf"))
        pairwise_dists[pairwise_dists < 0.01] = float("inf")

        # 只从有效物体出发选邻居
        valid_indices = torch.where(scene_mask)[0]
        if len(valid_indices) == 0:
            return torch.empty((0, 2), dtype=torch.long, device=device)

        k_actual = min(k, N - 1)
        edges = []
        for i in valid_indices:
            topk = torch.topk(
                pairwise_dists[i], k_actual, largest=False
            )
            for j in topk.indices:
                if scene_mask[j]:
                    edges.append([int(i), int(j)])

        if len(edges) == 0:
            return torch.empty((0, 2), dtype=torch.long, device=device)
        return torch.tensor(edges, dtype=torch.long, device=device)

    def forward(self, scene_feat, scene_locs, scene_mask):
        """
        scene_feat:  [B, N, 128] CLASP 3D 特征
        scene_locs: [B, N, 6]   物体位置 (前3维 = 中心 xyz)
        scene_mask: [B, N]       有效物体 mask

        返回:
          rel_feats:    [B, E_max, 256] 关系特征（padding 到 batch 内最大边数）
          edge_indices: [B, E_max, 2]   边索引
          edge_masks:   [B, E_max]       有效边 mask
        """
        B, N, _ = scene_feat.shape
        device = scene_feat.device
        obj_positions = scene_locs[..., :3]  # [B, N, 3]

        batch_rel_feats = []
        batch_edge_indices = []
        max_edges = 0

        for b in range(B):
            mask = scene_mask[b]  # [N]
            pos = obj_positions[b]  # [N, 3]
            feat = scene_feat[b]  # [N, 128]

            # 1. KNN 拓扑
            edge_idx = self._build_knn_edges(pos, mask, self.k)  # [E, 2]
            E = edge_idx.shape[0]

            if E == 0:
                rel_feat = torch.empty(
                    (0, self.rel_dim), dtype=feat.dtype, device=device
                )
            else:
                # 2. 收集每条边的特征
                src = edge_idx[:, 0]
                tgt = edge_idx[:, 1]
                feat_src = feat[src]  # [E, 128]
                feat_tgt = feat[tgt]  # [E, 128]
                pos_src = pos[src]  # [E, 3]
                pos_tgt = pos[tgt]  # [E, 3]

                # 3. 关系特征提取
                rel_feat = self.relation_extractor(
                    feat_src, feat_tgt, pos_src, pos_tgt
                )  # [E, 256]

            batch_rel_feats.append(rel_feat)
            batch_edge_indices.append(edge_idx)
            max_edges = max(max_edges, E)

        # Padding 到 batch 内统一尺寸
        padded_rel_feats = torch.zeros(
            B, max_edges, self.rel_dim, dtype=scene_feat.dtype, device=device
        )
        padded_edge_indices = torch.zeros(
            B, max_edges, 2, dtype=torch.long, device=device
        )
        edge_masks = torch.zeros(B, max_edges, dtype=torch.bool, device=device)

        for b in range(B):
            E = batch_rel_feats[b].shape[0]
            if E > 0:
                padded_rel_feats[b, :E] = batch_rel_feats[b]
                padded_edge_indices[b, :E] = batch_edge_indices[b]
                edge_masks[b, :E] = True

        return padded_rel_feats, padded_edge_indices, edge_masks


class SceneGraphTokenBuilder(nn.Module):
    """将场景图边特征投影为 LLM 可用的 token 嵌入序列。

    输入场景图（关系特征 + 边索引），输出拼接在物体 token 后面的关系 token 序列。
    每条约的格式: [OBJ_src, REL_feat, OBJ_tgt]
    """

    def __init__(self, llama_dim=2048, rel_dim=256):
        super().__init__()
        self.relation_proj = nn.Sequential(
            nn.Linear(rel_dim, llama_dim),
            nn.GELU(),
            nn.Linear(llama_dim, llama_dim),
        )

    def forward(self, rel_feats, edge_indices, edge_masks, objid_embeds):
        """
        rel_feats:    [B, E, 256]  关系特征
        edge_indices: [B, E, 2]    边 (src, tgt)
        edge_masks:   [B, E]        有效边 mask
        objid_embeds: [max_obj, llama_dim]  Object Identifier 嵌入表

        返回:
          sg_tokens: [B, E*3, llama_dim] 场景图 token 序列
          sg_mask:   [B, E*3]            有效 token mask
        """
        B, E, _ = rel_feats.shape
        llama_dim = objid_embeds.shape[-1]
        device = rel_feats.device

        proj_rel = self.relation_proj(rel_feats)  # [B, E, llama_dim]

        sg_tokens = torch.zeros(
            B, E * 3, llama_dim, dtype=rel_feats.dtype, device=device
        )
        sg_mask = torch.zeros(B, E * 3, dtype=torch.bool, device=device)

        for b in range(B):
            valid_e = edge_masks[b].sum().item()
            for e in range(valid_e):
                src, tgt = edge_indices[b, e]
                offset = e * 3
                sg_tokens[b, offset] = objid_embeds[src]
                sg_tokens[b, offset + 1] = proj_rel[b, e]
                sg_tokens[b, offset + 2] = objid_embeds[tgt]
                sg_mask[b, offset : offset + 3] = True

        return sg_tokens, sg_mask


# ---------------------------------------------------------------------------
# 测试入口：验证场景图构建是否正常
# 用法: cd SGR && python -m models.scene_graph
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os

    # 数据路径：Chat-Scene++_data 在 SGR 的同级目录
    project_root = os.path.join(os.path.dirname(__file__), "..", "..")
    data_dir = os.path.abspath(os.path.join(project_root, "Chat-Scene++_data"))

    print("=" * 60)
    print("场景图构建模块测试")
    print(f"数据目录: {data_dir}")
    print("=" * 60)

    # 1. 加载 CLASP 数据
    attr_path = os.path.join(data_dir, "scannet_clasp_train_attributes.pt")
    feat_path = os.path.join(data_dir, "scannet_clasp_clasp_feats.pt")

    if not os.path.exists(attr_path) or not os.path.exists(feat_path):
        print(f"错误: 数据文件不存在，请确认 {data_dir} 中有 .pt 文件")
        sys.exit(1)

    attributes = torch.load(attr_path, map_location="cpu", weights_only=False)
    feats = torch.load(feat_path, map_location="cpu", weights_only=False)

    # 取第一个场景
    scene_id = list(attributes.keys())[0]
    scene_attr = attributes[scene_id]
    locs = scene_attr["locs"]  # [N, 6]
    objects = scene_attr["objects"]  # list of N strings
    N = locs.shape[0]

    # 组装该场景的 CLASP 特征
    scene_feat_list = []
    for i in range(N):
        item_id = f"{scene_id}_{i:02d}"
        if item_id in feats:
            scene_feat_list.append(feats[item_id])
        else:
            scene_feat_list.append(torch.zeros(128))
    scene_feat = torch.stack(scene_feat_list, dim=0)  # [N, 128]

    print(f"\n场景: {scene_id}")
    print(f"物体数: {N}")
    print(f"locs 形状: {locs.shape}")
    print(f"CLASP 特征形状: {scene_feat.shape}")
    print(f"前 10 个物体类别: {objects[:10]}")

    # 2. 构建场景图（拓扑 + 关系特征）
    sg_builder = SceneGraphBuilder(feat_dim=128, rel_dim=256, k=2)
    sg_builder.eval()  # 未训练，只是验证形状

    scene_mask = torch.ones(N, dtype=torch.bool)  # 全部物体都有效

    with torch.no_grad():
        rel_feats, edge_indices, edge_masks = sg_builder(
            scene_feat.unsqueeze(0),  # [B=1, N, 128]
            locs.unsqueeze(0),  # [B=1, N, 6]
            scene_mask.unsqueeze(0),  # [B=1, N]
        )

    E = edge_masks.sum().item()
    print(f"\n--- 场景图拓扑 ---")
    print(f"总边数: {E} (KNN k=2, 理论最大 = {N * 2})")
    print(f"关系特征形状: {rel_feats.shape}  (B=1, E_padded={rel_feats.shape[1]}, 256)")

    # 打印部分边样例
    print(f"\n边样例 (前 15 条):")
    print(f"{'src':>4s}  {'tgt':>4s}  {'src_cat':>12s}  {'tgt_cat':>12s}  {'dist(m)':>8s}")
    print("-" * 52)
    for e in range(min(15, E)):
        src = int(edge_indices[0, e, 0])
        tgt = int(edge_indices[0, e, 1])
        pos_src = locs[src, :3]
        pos_tgt = locs[tgt, :3]
        dist = torch.norm(pos_tgt - pos_src).item()
        cat_src = objects[src] if src < len(objects) else "?"
        cat_tgt = objects[tgt] if tgt < len(objects) else "?"
        print(f"{src:4d}  {tgt:4d}  {cat_src:>12s}  {cat_tgt:>12s}  {dist:8.3f}")

    # 3. 简单统计
    all_src = edge_indices[0, :E, 0].tolist()
    from collections import Counter

    degree = Counter(all_src)
    print(f"\n--- 度分布统计 ---")
    print(f"平均度数: {sum(degree.values()) / N:.2f}")
    print(f"最大度数: {max(degree.values())}")
    print(f"最小度数: {min(degree.values())}")
    print(f"孤立物体 (度=0): {N - len(degree)}")

    pairwise = torch.cdist(locs[:, :3], locs[:, :3])
    pairwise[pairwise < 0.01] = float("inf")
    min_dist = pairwise.min(dim=1)[0]
    print(f"\n--- 空间分布 ---")
    print(f"最近邻距离: 均值={min_dist.mean():.3f}m, 中位数={min_dist.median():.3f}m")

    # 4. 测试关系特征质量
    rel_norm = rel_feats[0, :E].norm(dim=-1)
    print(f"\n--- 关系特征（未训练，随机初始化） ---")
    print(f"特征范数: 均值={rel_norm.mean():.3f}, 标准差={rel_norm.std():.3f}")

    print(f"\n{'=' * 60}")
    print("测试通过！模块可正常运行。")
    print("注意: 关系特征是随机初始化的，需要训练后才有意义。")
    print("下一步: 将 SceneGraphBuilder 集成到 chat3d.py 的 forward_train 中。")
    print("=" * 60)
