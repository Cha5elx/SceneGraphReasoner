import torch
from torch import nn

from models.chat3d import Chat3D


class ZeroResidualReasoner(nn.Module):
    def forward(self, scene_feat, **kwargs):
        return torch.zeros_like(scene_feat), {"object_loss": scene_feat.sum() * 0}


def test_scale_zero_preserves_object_features_with_auxiliary_loss_enabled():
    model = Chat3D.__new__(Chat3D)
    nn.Module.__init__(model)
    model.use_scene_graph = True
    model.sg_residual_scale = 0
    model.sg_aux_object_loss_weight = 0.5
    model.query_graph_reasoner = ZeroResidualReasoner()
    model.get_query_token_embeds = lambda texts, device: (
        torch.zeros(len(texts), 1, 4, device=device),
        torch.ones(len(texts), 1, dtype=torch.bool, device=device),
    )

    object_embed = torch.tensor([[[0.1, 0.2, 0.3]]])
    output, graph_info = model.apply_query_graph_reasoning(
        object_embed=object_embed,
        scene_locs=torch.ones(1, 1, 6),
        scene_mask=torch.ones(1, 1, dtype=torch.bool),
        queries=["find the object"],
        target_obj_ids=torch.tensor([0]),
        target_obj_mask=torch.tensor([True]),
    )

    assert torch.equal(output, object_embed)
    assert graph_info is not None
