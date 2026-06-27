import torch

from models.query_graph_reasoner import QueryGraphReasoner


def build_inputs():
    torch.manual_seed(0)
    scene_feat = torch.randn(2, 4, 8)
    scene_locs = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [2.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 2.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 3.0, 0.0, 1.0, 1.0, 1.0],
            ],
        ]
    )
    scene_mask = torch.ones(2, 4, dtype=torch.bool)
    query_token_embeds = torch.randn(2, 3, 12)
    query_token_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    return (
        scene_feat,
        scene_locs,
        scene_mask,
        query_token_embeds,
        query_token_mask,
    )


def test_zero_init_residual_preserves_baseline():
    reasoner = QueryGraphReasoner(
        feat_dim=8,
        query_input_dim=12,
        output_dim=16,
        hidden_dim=8,
        candidate_k=3,
        effective_k=1,
    )
    residual, graph_info = reasoner(*build_inputs())

    assert residual.shape == (2, 4, 16)
    assert torch.count_nonzero(residual) == 0
    assert not graph_info["node_mask"][0, 3]
    assert graph_info["edge_indices"].shape == (2, 4, 3, 2)
    assert graph_info["valid_node_count"] == 3.5
    assert graph_info["candidate_edge_count"] == 9.0


def test_eval_keeps_per_source_top_k_edges():
    reasoner = QueryGraphReasoner(
        feat_dim=8,
        query_input_dim=12,
        output_dim=16,
        hidden_dim=8,
        candidate_k=3,
        effective_k=1,
    )
    reasoner.eval()
    _, graph_info = reasoner(*build_inputs())

    active_per_source = graph_info["active_edge_mask"].sum(dim=-1)
    assert torch.all(active_per_source <= 1)
    assert active_per_source[0, 3] == 0
    assert torch.all(active_per_source[1] == 1)
    assert graph_info["active_edge_count"] == 3.5


def test_training_uses_soft_candidate_edges():
    reasoner = QueryGraphReasoner(
        feat_dim=8,
        query_input_dim=12,
        output_dim=16,
        hidden_dim=8,
        candidate_k=3,
        effective_k=1,
    )
    reasoner.train()
    _, graph_info = reasoner(*build_inputs())

    assert torch.equal(
        graph_info["active_edge_mask"], graph_info["candidate_edge_mask"]
    )


def test_residual_scale_controls_output_magnitude():
    reasoner = QueryGraphReasoner(
        feat_dim=8,
        query_input_dim=12,
        output_dim=16,
        hidden_dim=8,
        candidate_k=3,
        effective_k=1,
        residual_scale=0.25,
    )
    with torch.no_grad():
        reasoner.residual_proj.bias.fill_(1.0)

    residual, graph_info = reasoner(*build_inputs())

    valid_nodes = graph_info["node_mask"]
    assert torch.allclose(
        residual[valid_nodes],
        torch.full_like(residual[valid_nodes], 0.25),
    )
    assert torch.count_nonzero(residual[~valid_nodes]) == 0
    assert torch.isclose(graph_info["residual_scale"], torch.tensor(0.25))
    assert torch.allclose(
        graph_info["residual_norm"],
        graph_info["raw_residual_norm"] * 0.25,
    )


def test_fixed_gnn_ablation_uses_uniform_candidate_scores():
    reasoner = QueryGraphReasoner(
        feat_dim=8,
        query_input_dim=12,
        output_dim=16,
        hidden_dim=8,
        candidate_k=3,
        effective_k=1,
        use_query_gating=False,
    )
    reasoner.train()
    _, graph_info = reasoner(*build_inputs())

    assert torch.equal(
        graph_info["edge_scores"],
        graph_info["candidate_edge_mask"].to(graph_info["edge_scores"].dtype),
    )
    assert torch.isclose(graph_info["mean_edge_score"], torch.tensor(1.0))
    assert torch.isclose(graph_info["edge_score_std"], torch.tensor(0.0))


def test_cosine_object_selector_prefers_aligned_node():
    reasoner = QueryGraphReasoner(
        feat_dim=4,
        query_input_dim=4,
        output_dim=4,
        hidden_dim=4,
        candidate_k=2,
        effective_k=1,
        object_temperature=5.0,
    )
    with torch.no_grad():
        reasoner.object_key_proj.weight.copy_(torch.eye(4))
        reasoner.query_key_proj.weight.copy_(torch.eye(4))

    node_feats = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]
    )
    query_embed = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    node_mask = torch.ones(1, 2, dtype=torch.bool)

    logits, _, _ = reasoner._score_objects(
        node_feats, query_embed, node_mask
    )

    assert torch.isclose(logits[0, 0], torch.tensor(5.0))
    assert torch.isclose(logits[0, 1], torch.tensor(0.0))


def test_object_relevance_modulates_destination_edges():
    reasoner = QueryGraphReasoner(
        feat_dim=4,
        query_input_dim=4,
        output_dim=4,
        hidden_dim=4,
        candidate_k=2,
        effective_k=1,
    )
    base_scores = torch.ones(1, 1, 2)
    object_logits = torch.tensor([[-2.0, 2.0, 0.0]])
    target_indices = torch.tensor([[[0, 1]]])
    candidate_mask = torch.ones(1, 1, 2, dtype=torch.bool)

    edge_scores = reasoner._modulate_edges_with_object_relevance(
        base_edge_scores=base_scores,
        object_logits=object_logits,
        target_indices=target_indices,
        candidate_mask=candidate_mask,
    )

    assert edge_scores[0, 0, 1] > edge_scores[0, 0, 0]
    assert torch.allclose(
        edge_scores[0, 0], torch.sigmoid(torch.tensor([-2.0, 2.0]))
    )


def test_target_object_auxiliary_loss_reports_rank_metrics():
    reasoner = QueryGraphReasoner(
        feat_dim=8,
        query_input_dim=12,
        output_dim=16,
        hidden_dim=8,
        candidate_k=3,
        effective_k=1,
        object_topm=2,
        use_object_gate=True,
    )
    target_obj_ids = torch.tensor([1, 2])
    target_obj_mask = torch.tensor([True, False])

    _, graph_info = reasoner(
        *build_inputs(),
        target_obj_ids=target_obj_ids,
        target_obj_mask=target_obj_mask,
    )

    assert graph_info["object_loss"] > 0
    assert torch.isclose(graph_info["object_target_count"], torch.tensor(1.0))
    assert graph_info["object_target_rank"] >= 1
    assert torch.isclose(
        graph_info["object_loss_sum"], graph_info["object_loss"]
    )
    assert torch.isclose(
        graph_info["object_target_rank_sum"],
        graph_info["object_target_rank"],
    )
    assert torch.isclose(
        graph_info["object_reciprocal_rank_sum"],
        graph_info["object_target_rank"].reciprocal(),
    )
    assert 0 <= graph_info["object_top1_acc"] <= 1
    assert 0 <= graph_info["object_top5_acc"] <= 1
    assert 0 <= graph_info["object_topm_recall"] <= 1
    assert 0 <= graph_info["object_gate_mean"] <= 1
    assert torch.isclose(
        graph_info["object_shuffled_target_count"], torch.tensor(1.0)
    )
    assert 0 <= graph_info["object_query_top1_change"] <= 1


def test_object_auxiliary_loss_trains_cosine_projections():
    reasoner = QueryGraphReasoner(
        feat_dim=8,
        query_input_dim=12,
        output_dim=16,
        hidden_dim=8,
        candidate_k=3,
        effective_k=1,
    )

    _, graph_info = reasoner(
        *build_inputs(),
        target_obj_ids=torch.tensor([1, 2]),
        target_obj_mask=torch.tensor([True, True]),
    )
    graph_info["object_loss"].backward()

    assert reasoner.object_key_proj.weight.grad is not None
    assert reasoner.query_key_proj.weight.grad is not None
    assert reasoner.query_encoder.token_proj.weight.grad is not None


def test_empty_target_batch_keeps_differentiable_zero_loss():
    reasoner = QueryGraphReasoner(
        feat_dim=8,
        query_input_dim=12,
        output_dim=16,
        hidden_dim=8,
        candidate_k=3,
        effective_k=1,
    )

    _, graph_info = reasoner(
        *build_inputs(),
        target_obj_ids=torch.tensor([1, 2]),
        target_obj_mask=torch.tensor([False, False]),
    )

    assert graph_info["object_loss"].requires_grad
    assert torch.isclose(graph_info["object_loss"], torch.tensor(0.0))
