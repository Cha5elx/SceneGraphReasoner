"""Query-conditioned residual graph reasoning for CLASP object features.

The module keeps graph computation outside the LLM token sequence. It builds a
small geometric candidate graph online, routes edges with the current query,
propagates one round of messages, and returns a residual for the projected 3D
object tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryEncoder(nn.Module):
    """Pool frozen LLM token embeddings into one lightweight query vector."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.token_proj = nn.Linear(input_dim, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)

    def forward(self, token_embeds, token_mask):
        """
        Args:
            token_embeds: [B, T, D_text] frozen LLM token embeddings.
            token_mask:   [B, T] valid text token mask.

        Returns:
            query_embed: [B, D_graph].
        """
        token_feats = torch.tanh(self.token_proj(token_embeds))
        scores = self.attn_score(token_feats).squeeze(-1)
        scores = scores.masked_fill(~token_mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * token_mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.sum(token_feats * weights.unsqueeze(-1), dim=1)


class QueryGraphReasoner(nn.Module):
    """Enhance 3D object tokens with a query-specific sparse evidence graph.

    Inputs:
        scene_feat:         [B, N, D_clasp] normalized CLASP object features.
        scene_locs:         [B, N, 6] boxes as center xyz and size whl.
        scene_mask:         [B, N] baseline object mask.
        query_token_embeds: [B, T, D_text] frozen LLM token embeddings.
        query_token_mask:   [B, T] valid query token mask.

    Outputs:
        graph_residual: [B, N, D_llm] residual added to projected 3D tokens.
        graph_info:     tensors for monitoring and later graph visualization.

    Zero-size boxes stay available to the baseline LLM object sequence but are
    excluded from graph construction because they do not carry valid geometry.
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
        graph_node_mask = self._build_graph_node_mask(scene_locs, scene_mask)
        normalized_boxes, scene_scale = self._normalize_boxes(
            scene_locs, graph_node_mask
        )
        node_input = torch.cat(
            [F.normalize(scene_feat, dim=-1), normalized_boxes], dim=-1
        )
        node_feats = self.node_encoder(node_input)
        query_embed = self.query_encoder(query_token_embeds, query_token_mask)

        target_indices, candidate_mask = self._build_candidate_edges(
            scene_locs[..., :3], graph_node_mask
        )
        neighbor_feats = self._gather_neighbors(node_feats, target_indices)
        edge_geometry = self._compute_edge_geometry(
            scene_feat, scene_locs, target_indices, scene_scale
        )

        batch_size, obj_num, edge_num = target_indices.shape
        source_feats = node_feats.unsqueeze(2).expand(-1, -1, edge_num, -1)
        edge_query = query_embed[:, None, None, :].expand(
            -1, obj_num, edge_num, -1
        )
        router_input = torch.cat(
            [source_feats, neighbor_feats, edge_geometry, edge_query], dim=-1
        )
        edge_scores = torch.sigmoid(self.edge_router(router_input).squeeze(-1))
        edge_scores = edge_scores * candidate_mask.to(edge_scores.dtype)

        if hard_prune is None:
            hard_prune = self.hard_prune_eval and not self.training
        active_edge_mask = self._select_active_edges(
            edge_scores, candidate_mask, hard_prune
        )
        edge_weights = edge_scores * active_edge_mask.to(edge_scores.dtype)

        messages = self.message_mlp(
            torch.cat([neighbor_feats, edge_geometry], dim=-1)
        )
        weight_sum = edge_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        aggregated = torch.sum(messages * edge_weights.unsqueeze(-1), dim=2)
        aggregated = aggregated / weight_sum
        refined_feats = self.node_norm(node_feats + self.message_out(aggregated))
        refined_feats = refined_feats * graph_node_mask.unsqueeze(-1).to(
            refined_feats.dtype
        )

        graph_residual = self.residual_proj(F.normalize(refined_feats, dim=-1))
        graph_residual = graph_residual * graph_node_mask.unsqueeze(-1).to(
            graph_residual.dtype
        )

        source_indices = torch.arange(
            obj_num, device=target_indices.device, dtype=torch.long
        )
        source_indices = source_indices.view(1, obj_num, 1).expand(
            batch_size, -1, edge_num
        )
        edge_indices = torch.stack([source_indices, target_indices], dim=-1)
        candidate_edge_count = candidate_mask.sum(dim=(1, 2)).to(edge_scores.dtype)
        active_edge_count = active_edge_mask.sum(dim=(1, 2)).to(edge_scores.dtype)
        graph_info = {
            "node_mask": graph_node_mask,
            "edge_indices": edge_indices,
            "candidate_edge_mask": candidate_mask,
            "active_edge_mask": active_edge_mask,
            "edge_scores": edge_scores,
            "edge_weights": edge_weights,
            "query_embed": query_embed,
            "residual_norm": graph_residual.norm(dim=-1).mean(),
            "valid_node_count": graph_node_mask.sum(dim=1).float().mean(),
            "candidate_edge_count": candidate_edge_count.mean(),
            "active_edge_count": active_edge_count.mean(),
            "mean_edge_score": edge_scores.sum(dim=(1, 2)).div(
                candidate_edge_count.clamp_min(1)
            ).mean(),
        }
        return graph_residual, graph_info

    def _build_graph_node_mask(self, scene_locs, scene_mask):
        valid_geometry = (scene_locs[..., 3:] > self.bbox_eps).all(dim=-1)
        return scene_mask.bool() & valid_geometry

    def _normalize_boxes(self, scene_locs, graph_node_mask):
        centers = scene_locs[..., :3]
        sizes = scene_locs[..., 3:]
        valid = graph_node_mask.unsqueeze(-1)
        mins = torch.where(valid, centers, torch.full_like(centers, float("inf")))
        maxs = torch.where(
            valid, centers, torch.full_like(centers, float("-inf"))
        )
        mins = mins.amin(dim=1)
        maxs = maxs.amax(dim=1)
        has_valid_node = graph_node_mask.any(dim=1, keepdim=True)
        mins = torch.where(has_valid_node, mins, torch.zeros_like(mins))
        maxs = torch.where(has_valid_node, maxs, torch.zeros_like(maxs))
        scene_scale = (maxs - mins).norm(dim=-1, keepdim=True).clamp_min(
            self.bbox_eps
        )
        scene_center = (mins + maxs) / 2
        normalized_centers = (centers - scene_center.unsqueeze(1)) / (
            scene_scale.unsqueeze(1)
        )
        normalized_sizes = sizes / scene_scale.unsqueeze(1)
        return torch.cat([normalized_centers, normalized_sizes], dim=-1), scene_scale

    def _build_candidate_edges(self, centers, graph_node_mask):
        _, obj_num, _ = centers.shape
        edge_num = min(self.candidate_k, max(obj_num - 1, 1))
        pairwise_dist = torch.cdist(centers, centers)
        valid_pairs = graph_node_mask.unsqueeze(2) & graph_node_mask.unsqueeze(1)
        eye = torch.eye(obj_num, dtype=torch.bool, device=centers.device)
        valid_pairs = valid_pairs & ~eye.unsqueeze(0)
        pairwise_dist = pairwise_dist.masked_fill(~valid_pairs, float("inf"))
        distances, target_indices = torch.topk(
            pairwise_dist, k=edge_num, dim=-1, largest=False
        )
        return target_indices, torch.isfinite(distances)

    def _compute_edge_geometry(
        self, scene_feat, scene_locs, target_indices, scene_scale
    ):
        edge_num = target_indices.shape[-1]
        source_boxes = scene_locs.unsqueeze(2).expand(-1, -1, edge_num, -1)
        target_boxes = self._gather_neighbors(scene_locs, target_indices)
        source_feat = scene_feat.unsqueeze(2).expand(-1, -1, edge_num, -1)
        target_feat = self._gather_neighbors(scene_feat, target_indices)

        source_center, source_size = source_boxes[..., :3], source_boxes[..., 3:]
        target_center, target_size = target_boxes[..., :3], target_boxes[..., 3:]
        scale = scene_scale[:, None, None, :]
        delta = (target_center - source_center) / scale
        distance = delta.norm(dim=-1, keepdim=True)
        size_ratio = torch.log(
            (target_size + self.bbox_eps) / (source_size + self.bbox_eps)
        )

        source_min = source_center - source_size / 2
        source_max = source_center + source_size / 2
        target_min = target_center - target_size / 2
        target_max = target_center + target_size / 2
        intersection_size = (
            torch.minimum(source_max, target_max)
            - torch.maximum(source_min, target_min)
        ).clamp_min(0)
        intersection_volume = intersection_size.prod(dim=-1, keepdim=True)
        source_volume = source_size.clamp_min(0).prod(dim=-1, keepdim=True)
        target_volume = target_size.clamp_min(0).prod(dim=-1, keepdim=True)
        iou = intersection_volume / (
            source_volume + target_volume - intersection_volume + self.bbox_eps
        )

        intersection_xy = intersection_size[..., :2].prod(dim=-1, keepdim=True)
        source_xy = source_size[..., :2].clamp_min(0).prod(dim=-1, keepdim=True)
        target_xy = target_size[..., :2].clamp_min(0).prod(dim=-1, keepdim=True)
        overlap_xy = intersection_xy / (
            torch.minimum(source_xy, target_xy) + self.bbox_eps
        )
        gap_z = (target_min[..., 2:3] - source_max[..., 2:3]) / scale
        cosine = F.cosine_similarity(source_feat, target_feat, dim=-1).unsqueeze(-1)
        return torch.cat(
            [delta, distance, size_ratio, iou, overlap_xy, gap_z, cosine], dim=-1
        )

    def _select_active_edges(self, edge_scores, candidate_mask, hard_prune):
        if not hard_prune:
            return candidate_mask
        keep_num = min(self.effective_k, edge_scores.shape[-1])
        masked_scores = edge_scores.masked_fill(~candidate_mask, float("-inf"))
        top_indices = torch.topk(masked_scores, k=keep_num, dim=-1).indices
        active_mask = torch.zeros_like(candidate_mask)
        active_mask.scatter_(-1, top_indices, True)
        return active_mask & candidate_mask

    @staticmethod
    def _gather_neighbors(features, target_indices):
        batch_size, obj_num, edge_num = target_indices.shape
        feat_dim = features.shape[-1]
        expanded_features = features.unsqueeze(1).expand(-1, obj_num, -1, -1)
        expanded_indices = target_indices.unsqueeze(-1).expand(
            batch_size, obj_num, edge_num, feat_dim
        )
        return torch.gather(expanded_features, dim=2, index=expanded_indices)
