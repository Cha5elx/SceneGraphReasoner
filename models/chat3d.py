import random
import logging
from abc import ABC

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F

from .modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from models.position_embedding import PositionEmbeddingCoordsSine
from models.query_graph_reasoner import QueryGraphReasoner
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence

import contextlib
from dataset.base_dataset import update_caption, recover_caption

logger = logging.getLogger(__name__)


def nclamp(input, min, max):
    return input.clamp(min=min, max=max).detach() + input - input.detach()


def print_grad_status(model):
    """Call this function after losses.backward()
    and it will find out all variables without grad, which
    means that the varaible is not in the graph.
    """
    for name, p in model.named_parameters():
        print('{:80s}{:20s}{:20s}{}'.format(name,
            '(Trainable)' if p.requires_grad else '(Fixed)',
            '(Has grad):' if p.grad is not None else '(No grad backward):',
            list(p.shape)))


class Chat3D(nn.Module):
    """
    VideoChat model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        llama_model_path = config.model.llama_model_path
        self.low_resource = config.model.low_resource
        self.max_txt_len = config.model.max_txt_len
        self.gen_max_txt_len = getattr(config.model, 'gen_max_txt_len', config.model.max_txt_len)
        self.num_beams = getattr(config.model, 'num_beams', 1)
        self.seq_len_cap = getattr(config.model, 'seq_len_cap', 768)
        self.end_sym = config.model.end_sym
        self.system_path = config.model.system_path
        self.instruction_path = config.model.instruction_path
        self.role = config.model.role
        self.no_obj = config.model.no_obj
        self.add_scene_token = config.model.add_scene_token
        self.add_img_token = config.model.add_img_token
        self.train_emb = config.model.train_emb
        self.train_img_proj = config.model.train_img_proj
        self.train_graph_only = getattr(config.model, 'train_graph_only', False)
        self.input_dim = config.model.input_dim    # CLASP: 128
        self.img_input_dim = config.model.img_input_dim
        self.attr_dim = config.model.attr_dim
        self.scene_dim = config.model.scene_dim
        self.pos_dim = config.model.pos_dim
        self.max_obj_num = config.model.max_obj_num  # CLASP: 100
        self.bidirection = config.model.bidirection
        self.add_pos_emb = config.model.add_pos_emb
        self.feat_fusion = config.model.feat_fusion
        self.fuse_with_id = config.model.fuse_with_id
        self.use_location_token = config.model.use_location_token

        # SGR 场景图配置
        self.use_scene_graph = getattr(config.model, 'use_scene_graph', False)
        self.sg_candidate_k = getattr(config.model, 'sg_candidate_k', 6)
        self.sg_effective_k = getattr(config.model, 'sg_effective_k', 2)
        self.sg_hidden_dim = getattr(config.model, 'sg_hidden_dim', 128)
        self.sg_bbox_eps = getattr(config.model, 'sg_bbox_eps', 1e-6)
        self.sg_hard_prune_eval = getattr(config.model, 'sg_hard_prune_eval', True)
        self.sg_residual_scale = getattr(config.model, 'sg_residual_scale', 1.0)
        self.sg_use_query_gating = getattr(config.model, 'sg_use_query_gating', True)
        self.sg_diagnostics = getattr(config.model, 'sg_diagnostics', True)
        self.sg_aux_object_loss_weight = getattr(config.model, 'sg_aux_object_loss_weight', 0.0)
        self.sg_object_topm = getattr(config.model, 'sg_object_topm', 0)
        self.sg_use_object_gate = getattr(config.model, 'sg_use_object_gate', False)
        self.sg_object_temperature = getattr(config.model, 'sg_object_temperature', 5.0)
        self.sg_selector_only = getattr(config.model, 'sg_selector_only', False)
        if self.train_graph_only and not self.use_scene_graph:
            raise ValueError("model.train_graph_only=True requires model.use_scene_graph=True")
        if self.sg_selector_only:
            if not self.train_graph_only:
                raise ValueError("model.sg_selector_only=True requires model.train_graph_only=True")
            if self.sg_residual_scale != 0:
                raise ValueError("model.sg_selector_only=True requires model.sg_residual_scale=0")
            if self.sg_aux_object_loss_weight <= 0:
                raise ValueError(
                    "model.sg_selector_only=True requires "
                    "model.sg_aux_object_loss_weight>0"
                )

        self.debug = config.debug
        if not self.debug:
            logger.info(f'Loading model from {llama_model_path}')
            self.is_vicuna = "vicuna" in llama_model_path
            if self.is_vicuna:
                self.llama_tokenizer = LlamaTokenizer.from_pretrained(
                    llama_model_path, use_fast=False, legacy=False
                )
            else:
                self.llama_tokenizer = AutoTokenizer.from_pretrained(
                    llama_model_path, use_fast=False, trust_remote_code=True
                )
                if self.llama_tokenizer.pad_token is None:
                    self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            if self.low_resource:
                self.llama_model = AutoModelForCausalLM.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.bfloat16,
                    load_in_8bit=True,
                    device_map="auto",
                    attn_implementation="flash_attention_2"
                )
            elif self.is_vicuna:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2"
                )
            else:
                self.llama_model = AutoModelForCausalLM.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2"
                )
            # print(torch.cuda.memory_allocated(device="cuda:0")/1e9)
            # self.llama_model = self.llama_model.to("cuda")
            # print(torch.cuda.memory_allocated(device="cuda:0")/1e9)
            # breakpoint()
            logger.info("freeze LLAMA")
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False

            if config.model.use_lora:
                def find_linear_layers(model, lora_target_modules):
                    cls = torch.nn.Linear
                    lora_module_names = set()
                    for name, module in model.named_modules():
                        if (
                            isinstance(module, cls)
                            and all(
                                [
                                    x not in name
                                    for x in [
                                        "instance2embed",
                                        "hidden_state2query"
                                    ]
                                ]
                            )
                            and any([x in name for x in lora_target_modules])
                        ):
                            lora_module_names.add(name)
                    return sorted(list(lora_module_names))
            
                lora_target_modules = find_linear_layers(self.llama_model, config.lora.lora_target_modules)

                lora_config = LoraConfig(
                    r=config.lora.lora_r,
                    lora_alpha=config.lora.lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=config.lora.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.llama_model = get_peft_model(self.llama_model, lora_config)
                self.llama_model.print_trainable_parameters()
                self.llama_model.model.lm_head.weight.requires_grad = True
                self.llama_model.model.lm_head.weight.data = self.llama_model.model.lm_head.weight.data.float()
                self.llama_model.print_trainable_parameters()
                self.llama_model.model.model.embed_tokens.weight.requires_grad = True
                self.llama_model.model.model.embed_tokens.weight.data = self.llama_model.model.model.embed_tokens.weight.data.float()
                self.llama_model.print_trainable_parameters()
            else:
                self.llama_model.lm_head.weight.requires_grad = True
                self.llama_model.lm_head.weight.data = self.llama_model.lm_head.weight.data.float()
                self.llama_model.model.embed_tokens.weight.requires_grad = True
                self.llama_model.model.embed_tokens.weight.data = self.llama_model.model.embed_tokens.weight.data.float()
            
            self.llama_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
            objid_tokens = []
            for i in range(self.max_obj_num):
                objid_tokens.append(f"<OBJ{i:03}>")
            self.objid_start_idx = self.ori_vocab_size = len(self.llama_tokenizer)
            self.llama_tokenizer.add_tokens(objid_tokens, special_tokens=True)
            self.objid_end_idx = len(self.llama_tokenizer)
            self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
            
            # if self.use_location_token:
            #     location_tokens = ["<LOCATION>", "</LOCATION>"]
            #     for i in range(1000):
            #         location_tokens.append(f"<LOC{i:03}>")
            #     self.llama_tokenizer.add_tokens(location_tokens, special_tokens=True)
            #     self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))

            self.llama_dim = self.llama_model.config.hidden_size
            logger.info('Loading LLAMA Done')

        else:
            self.llama_model = None
            self.llama_dim = 4096

        
        self.object_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.llama_dim),
            nn.GELU(),
            nn.Linear(self.llama_dim, self.llama_dim)
        )
        self.object_img_proj = nn.Sequential(
            nn.Linear(self.img_input_dim, self.llama_dim),
            nn.GELU(),
            nn.Linear(self.llama_dim, self.llama_dim)
        )
        if self.use_scene_graph:
            self.query_graph_reasoner = QueryGraphReasoner(
                feat_dim=self.input_dim,
                query_input_dim=self.llama_dim,
                output_dim=self.input_dim,
                hidden_dim=self.sg_hidden_dim,
                candidate_k=self.sg_candidate_k,
                effective_k=self.sg_effective_k,
                bbox_eps=self.sg_bbox_eps,
                hard_prune_eval=self.sg_hard_prune_eval,
                residual_scale=self.sg_residual_scale,
                use_query_gating=self.sg_use_query_gating,
                diagnostics=self.sg_diagnostics,
                object_topm=self.sg_object_topm,
                use_object_gate=self.sg_use_object_gate,
                object_temperature=self.sg_object_temperature,
            )
            logger.info(
                'QueryGraphReasoner initialized: candidate_k=%s, effective_k=%s, hidden_dim=%s, residual_scale=%s, use_query_gating=%s, aux_object_loss_weight=%s, object_topm=%s, use_object_gate=%s, object_temperature=%s',
                self.sg_candidate_k,
                self.sg_effective_k,
                self.sg_hidden_dim,
                self.sg_residual_scale,
                self.sg_use_query_gating,
                self.sg_aux_object_loss_weight,
                self.sg_object_topm,
                self.sg_use_object_gate,
                self.sg_object_temperature,
            )
        if not self.train_img_proj:
            for p in self.object_img_proj.parameters():
                p.requires_grad = False
        self.pos_embedding = PositionEmbeddingCoordsSine(d_pos=self.pos_dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(self.pos_dim, self.llama_dim)
        )
        if self.train_graph_only:
            self.freeze_non_graph_parameters()
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.scene_dim, nhead=8, dim_feedforward=2048, dropout=0.05, norm_first=True, batch_first=True)
        # self.relation_module = nn.TransformerEncoder(self.encoder_layer, num_layers=config.model.encoder_num_layers)
        # self.scene_init_proj = nn.Sequential(
        #     nn.Linear(self.input_dim, self.scene_dim)
        # )
        # self.scene_proj = nn.Sequential(
        #     nn.Linear(self.scene_dim, self.llama_dim),
        #     # nn.GELU(),
        #     # nn.Linear(self.llama_dim, self.llama_dim)
        # )
        
        # if not self.add_scene_token:
        #     for p in self.relation_module.parameters():
        #         p.requires_grad = False
        #     for p in self.scene_init_proj.parameters():
        #         p.requires_grad = False
        #     for p in self.scene_proj.parameters():
        #         p.requires_grad = False
                

        with open(self.system_path, "r") as f:
            self.system = "\n".join([x.strip() for x in f.readlines()])
        with open(self.instruction_path, "r") as f:
            self.instruction = "\n".join([x.strip() for x in f.readlines()])

        if not self.debug:
            self.p_0_embed, self.p_1_embed = self.prepare_fixed_embed()
        self.last_embed = None
        
        # print_grad_status(self)

    def freeze_non_graph_parameters(self):
        if not self.use_scene_graph or not hasattr(self, "query_graph_reasoner"):
            raise ValueError("train_graph_only requires an initialized query_graph_reasoner")
        for param in self.parameters():
            param.requires_grad = False
        for param in self.query_graph_reasoner.parameters():
            param.requires_grad = True

        trainable_params = sum(
            p.numel() for p in self.query_graph_reasoner.parameters() if p.requires_grad
        )
        logger.info(
            "train_graph_only=True: froze all parameters except query_graph_reasoner "
            "(trainable_params=%s)",
            trainable_params,
        )

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, "train_graph_only", False):
            # Keep the frozen LLM in train mode: Qwen only applies gradient
            # checkpointing when module.training is True, and graph-only still
            # needs gradients through the LLM back to object tokens.
            self.object_proj.eval()
            self.object_img_proj.eval()
            self.pos_embedding.eval()
            self.pos_proj.eval()
            if hasattr(self, "query_graph_reasoner"):
                self.query_graph_reasoner.train(True)
        return self

    def get_objid_embeds(self):
        if self.config.model.use_lora:
            objid_embeds = self.llama_model.model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx] # max_obj_num * 4096
        else:
            objid_embeds = self.llama_model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx]
        return objid_embeds
    
    def llama_embed_tokens(self, token_ids):
        if self.config.model.use_lora:
            return self.llama_model.model.model.embed_tokens(token_ids)
        else:
            return self.llama_model.model.embed_tokens(token_ids)

    def prepare_fixed_embed(self):
        prompt = self.system + " " + self.instruction + " " + self.role[0] + ": " 
        p_0, p_1 = prompt.split("<REPLACE>")
        p_0_token = self.llama_tokenizer(p_0, return_tensors="pt", add_special_tokens=True)
        p_1_token = self.llama_tokenizer(p_1, return_tensors="pt", add_special_tokens=False)
        p_0_embed = self.llama_embed_tokens(p_0_token.input_ids).squeeze(0).detach()
        p_1_embed = self.llama_embed_tokens(p_1_token.input_ids).squeeze(0).detach()
        return p_0_embed, p_1_embed

    def get_text_emb(self, text, device="cpu"):
        text_tokens = self.llama_tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        embeds = self.llama_embed_tokens(text_tokens.input_ids)
        if self.train_emb:
            indices = text_tokens.input_ids >= self.ori_vocab_size
            indices = (indices * 1).unsqueeze(-1)
            embeds = (1 - indices) * embeds.detach() + indices * embeds
        else:
            embeds = embeds.detach()
        return embeds

    def get_query_token_embeds(self, texts, device):
        """Tokenize queries once and reuse frozen LLM embeddings for graph routing."""
        text_tokens = self.llama_tokenizer(
            list(texts),
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(device)
        token_embeds = self.llama_embed_tokens(text_tokens.input_ids).detach()
        return token_embeds, text_tokens.attention_mask.to(torch.bool)

    def apply_query_graph_reasoning(
        self,
        object_embed,
        scene_locs,
        scene_mask,
        queries,
        target_obj_ids=None,
        target_obj_mask=None,
    ):
        """Add query-specific graph evidence before projecting object tokens."""
        if not self.use_scene_graph:
            return object_embed, None
        if self.sg_residual_scale == 0 and self.sg_aux_object_loss_weight <= 0:
            return object_embed, None
        query_token_embeds, query_token_mask = self.get_query_token_embeds(
            queries, device=object_embed.device
        )
        graph_residual, graph_info = self.query_graph_reasoner(
            scene_feat=object_embed,
            scene_locs=scene_locs,
            scene_mask=scene_mask,
            query_token_embeds=query_token_embeds,
            query_token_mask=query_token_mask,
            target_obj_ids=target_obj_ids,
            target_obj_mask=target_obj_mask,
        )
        if self.sg_residual_scale != 0:
            object_embed = torch.nn.functional.normalize(
                object_embed + graph_residual, dim=-1
            )
        return object_embed, graph_info

    @staticmethod
    def _get_graph_metrics(graph_info):
        metric_keys = {
            "graph_object_target_count": "object_target_count",
            "graph_object_target_rank": "object_target_rank",
            "graph_object_top1_acc": "object_top1_acc",
            "graph_object_top5_acc": "object_top5_acc",
            "graph_object_top10_acc": "object_top10_acc",
            "graph_object_query_score_delta": "object_query_score_delta",
            "graph_object_query_top1_change": "object_query_top1_change",
            "graph_object_shuffled_rank": "object_shuffled_target_rank",
            "graph_object_shuffled_top10_acc": "object_shuffled_top10_acc",
            "graph_object_topm_recall": "object_topm_recall",
            "graph_object_gate": "object_gate_mean",
            "scene_norm": "residual_norm",
            "raw_scene_norm": "raw_residual_norm",
            "graph_residual_scale": "residual_scale",
            "graph_nodes": "valid_node_count",
            "graph_candidate_edges": "candidate_edge_count",
            "graph_active_edges": "active_edge_count",
            "graph_edge_score": "mean_edge_score",
            "graph_edge_score_std": "edge_score_std",
            "graph_edge_score_min": "edge_score_min",
            "graph_edge_score_max": "edge_score_max",
            "graph_edge_score_range": "edge_score_range",
            "graph_edge_top1_margin": "edge_top1_margin",
            "graph_query_score_delta": "query_score_delta",
            "graph_query_top1_change": "query_top1_change",
            "graph_query_topk_overlap": "query_topk_overlap",
        }
        if graph_info is None:
            return {name: 0.0 for name in metric_keys}
        return {
            name: graph_info[key].detach().cpu()
            for name, key in metric_keys.items()
        }

    def encode_object_feat(self, feat, img_feat, locs):
        feat = torch.nn.functional.normalize(feat, dim=-1)
        img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
        return feat, img_feat
    
    @staticmethod
    def get_dist_attention(pos, dist_exp=1):
        # pos (bs, obj_num, 3)
        dist = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = torch.sum(dist.abs()**dist_exp, dim=-1)
        dist_attn = torch.nn.functional.softmax(-dist, dim=-1)
        return dist_attn

    def get_object_list_embed(self, embed_obj, embed_img, embed_scene, scene_mask, obj_id, assigned_ids):
        valid_ids = torch.where(scene_mask)[0].tolist()
        # object_list_embed = []
        # object_list_embed.append(embed_obj[obj_id])
        # object_list_embed = torch.stack(object_list_embed, dim=0)
        # return object_list_embed
        if self.config.model.use_lora:
            objid_embeds = self.llama_model.model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx] # max_obj_num * 4096
        else:
            objid_embeds = self.llama_model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx]
        # if len(valid_ids) == 1:
        #     object_list_embed = []
        #     object_list_embed.append(objid_embeds[obj_id])
        #     if not self.no_obj:
        #         object_list_embed.append(embed_obj[valid_ids[0]])
        #     # if embed_scene is not None:
        #     #     object_list_embed.append(embed_scene[valid_ids[0]])
        #     # if embed_img is not None:
        #     #     object_list_embed.append(embed_img[valid_ids[0]])
        #     object_list_embed = torch.stack(object_list_embed, dim=0)
        #     return object_list_embed
        # random.shuffle(valid_ids)

        assigned_ids = assigned_ids[valid_ids]
        if not self.train_emb:
            objid_embeds = objid_embeds.detach()
        selected_objid_embeds = objid_embeds[valid_ids]
        if self.use_location_token:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 2, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::2, :] += embed_obj[assigned_ids]
            object_list_embed[1::2, :] += embed_img[assigned_ids]
            return object_list_embed
        if self.fuse_with_id:
            object_list_embed = selected_objid_embeds
            if not self.no_obj:
                object_list_embed += embed_obj[assigned_ids]
            if self.add_img_token:
                object_list_embed += embed_img[assigned_ids]
            return object_list_embed
        if self.feat_fusion:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 2, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::2, :] = selected_objid_embeds
            if not self.no_obj:
                object_list_embed[1::2, :] += embed_obj[assigned_ids]
            if self.add_img_token:
                object_list_embed[1::2, :] += embed_img[assigned_ids]
            return object_list_embed
        if self.no_obj:
            # if embed_img is None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 2, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::2, :] = selected_objid_embeds
            object_list_embed[1::2, :] = embed_img[assigned_ids]
            # else:
            #     object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 3, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            #     object_list_embed[0::3, :] = selected_objid_embeds
            #     object_list_embed[1::3, :] = embed_scene[assigned_ids]
            #     object_list_embed[2::3, :] = embed_img[assigned_ids]
            return object_list_embed
        if embed_img is None and embed_scene is None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 2, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::2, :] = selected_objid_embeds
            object_list_embed[1::2, :] = embed_obj[assigned_ids]
            return object_list_embed
            # object_list_embed = selected_objid_embeds + embed_obj[assigned_ids]
        if embed_img is None and embed_scene is not None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 3, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::3, :] = selected_objid_embeds
            object_list_embed[1::3, :] = embed_obj[assigned_ids]
            object_list_embed[2::3, :] = embed_scene[assigned_ids]
            return object_list_embed
        if embed_img is not None and embed_scene is None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 3, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::3, :] = selected_objid_embeds
            object_list_embed[1::3, :] = embed_obj[assigned_ids]
            object_list_embed[2::3, :] = embed_img[assigned_ids]
            return object_list_embed
        if embed_img is not None and embed_scene is not None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 4, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::4, :] = selected_objid_embeds
            object_list_embed[1::4, :] = embed_obj[assigned_ids]
            object_list_embed[2::4, :] = embed_scene[assigned_ids]
            object_list_embed[3::4, :] = embed_img[assigned_ids]
            return object_list_embed
        return object_list_embed

    def get_min_max_coord(self, xyz, scene_mask):
        scene_mask = scene_mask.unsqueeze(-1).expand_as(xyz)
        masked_xyz_min = torch.where(scene_mask, xyz, torch.full_like(xyz, float('inf')))
        masked_xyz_max = torch.where(scene_mask, xyz, torch.full_like(xyz, float('-inf')))
        mins = masked_xyz_min.min(dim=1)[0]
        maxs = masked_xyz_max.max(dim=1)[0]
        return mins, maxs

    def forward_train(self, scene_feat, scene_img_feat, scene_locs, scene_mask, obj_ids, assigned_ids, questions, answers, is_eval=False, obj_id_mask=None, **kwargs):
        object_embed, object_img_embed = self.encode_object_feat(scene_feat, scene_img_feat, scene_locs)
        device = object_embed.device
        batch_size = object_embed.shape[0]
        object_embed, graph_info = self.apply_query_graph_reasoning(
            object_embed,
            scene_locs,
            scene_mask,
            questions,
            target_obj_ids=obj_ids,
            target_obj_mask=obj_id_mask,
        )
        if self.sg_selector_only:
            graph_object_loss = graph_info["object_loss"]
            graph_aux_loss = graph_object_loss * self.sg_aux_object_loss_weight
            return dict(
                loss=graph_aux_loss,
                lm_loss=0.0,
                graph_aux_loss=graph_aux_loss.detach().cpu(),
                graph_object_loss=graph_object_loss.detach().cpu(),
                **self._get_graph_metrics(graph_info),
                max_seq_len=0,
            )

        proj_object_embed = self.object_proj(object_embed)
        proj_object_img_embed = self.object_img_proj(object_img_embed)
        if self.add_pos_emb:
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
            pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs]) / 10
            proj_pos_embed = self.pos_proj(pos_embed)
            proj_object_embed = proj_object_embed + proj_pos_embed
            proj_object_img_embed = proj_object_img_embed + proj_pos_embed

        proj_scene_embed = None
        if self.add_scene_token:  # remember to change the evaluate
            # if self.add_img_token:
            #     object_embed = object_embed + object_img_embed
            obj_embed = self.scene_init_proj(object_embed)
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
            pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs])
            pos_embed = self.pos_proj(pos_embed)
            scene_embed = obj_embed + pos_embed
            scene_embed = self.relation_module(scene_embed, src_key_padding_mask=~scene_mask)
            proj_scene_embed = self.scene_proj(scene_embed)

        input_embed_list, attn_list, target_list = [], [], []
        max_seq_len = 0
        p_0_embed = self.p_0_embed.to(device)
        p_1_embed = self.p_1_embed.to(device)
        object_list_intervals = []

        for i, question in enumerate(questions):
            prompt = f"{question} {self.role[1]}: "
            prompt_embed = self.get_text_emb(prompt, device=device).squeeze(0)
            object_list_embed = self.get_object_list_embed(
                proj_object_embed[i], 
                proj_object_img_embed[i] if self.add_img_token else None, 
                proj_scene_embed[i] if self.add_scene_token else None, 
                scene_mask[i],
                obj_ids[i],
                assigned_ids[i]
            )
            # object_list_embed = nclamp(object_list_embed, min=-0.05, max=0.05)
            object_list_intervals.append((p_0_embed.shape[0], p_0_embed.shape[0] + object_list_embed.shape[0]))
            wrapped_embed = torch.cat([p_0_embed, object_list_embed, p_1_embed, prompt_embed], dim=0)
            wrapped_attn = torch.ones(wrapped_embed.size()[:-1], dtype=torch.long).to(wrapped_embed.device)
            empty_target = (
                torch.ones(wrapped_attn.shape[0], dtype=torch.long).to(device).fill_(-100)
            )

            answer = answers[i] + self.end_sym
            to_regress_token = self.llama_tokenizer(answer, return_tensors="pt", add_special_tokens=False).to(device)
            # breakpoint()
            answer_target = to_regress_token.input_ids.masked_fill(
                to_regress_token.input_ids == self.llama_tokenizer.pad_token_id, -100
            ).squeeze(0)
            # to_regress_embed = self.llama_model.model.embed_tokens(to_regress_token.input_ids).squeeze(0).detach()
            to_regress_embed = self.get_text_emb(answer, device=device).squeeze(0)

            target = torch.cat([empty_target, answer_target], dim=0)
            input_embed = torch.cat([wrapped_embed, to_regress_embed], dim=0)
            attn = torch.cat([wrapped_attn, to_regress_token.attention_mask[0]], dim=0)
            input_embed_list.append(input_embed)
            attn_list.append(attn)
            target_list.append(target)
            max_seq_len = max(max_seq_len, target.shape[0])
        
        max_seq_len = min(self.seq_len_cap, max_seq_len)

        def pad_and_trim(tensor_list, max_len, batch_first=True, padding_value=0):
            padded = pad_sequence(tensor_list, batch_first=batch_first, padding_value=padding_value)
            if padded.shape[1] > max_len:
                return padded[:, :max_len]
            return padded
        
        input_embeds = pad_and_trim(input_embed_list, max_seq_len, batch_first=True, padding_value=0).to(device)
        targets = pad_and_trim(target_list, max_seq_len, batch_first=True, padding_value=-100).to(device)
        attention_mask = pad_and_trim(attn_list, max_seq_len, batch_first=True, padding_value=0).to(device)
        if self.bidirection:
            input_dtype = input_embeds.dtype
            causal_mask = torch.ones((max_seq_len, max_seq_len), dtype=input_dtype, device=device)
            causal_mask = torch.tril(causal_mask, diagonal=0)
            causal_mask = causal_mask[None, None, :, :].expand(input_embeds.shape[0], 1, -1, -1).clone()
            padding_mask = causal_mask[..., :].eq(1.0) * attention_mask[:, None, None, :].eq(0.0)
            causal_mask[..., :] = causal_mask[..., :].masked_fill(padding_mask, 0.0)
            for i in range(causal_mask.shape[0]):
                st, ed = object_list_intervals[i]
                causal_mask[i, :, st:ed, st:ed] = 1.0
            attention_mask = causal_mask
        
        # label_weights = torch.ones(self.llama_model.config.vocab_size, device=device)
        # label_weights[self.objid_start_idx:self.objid_end_idx] = 10

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                # label_weights=label_weights
            )

        graph_object_loss = (
            graph_info["object_loss"]
            if graph_info is not None
            else outputs.loss.new_tensor(0.0)
        )
        graph_aux_loss = graph_object_loss * self.sg_aux_object_loss_weight
        total_loss = outputs.loss + graph_aux_loss

        return dict(
            loss=total_loss,
            lm_loss=outputs.loss.detach().cpu(),
            graph_aux_loss=graph_aux_loss.detach().cpu(),
            graph_object_loss=graph_object_loss.detach().cpu(),
            **self._get_graph_metrics(graph_info),
            obj_norm=proj_object_embed.norm(dim=-1).mean().detach().cpu(),
            obj_img_norm=proj_object_img_embed.norm(dim=-1).mean().detach().cpu(),
            objid_norm=self.get_objid_embeds().norm(dim=-1).mean().detach().cpu(),
            max_seq_len=max_seq_len
        )

    @torch.no_grad()
    def evaluate_graph_selector(
        self,
        scene_feat,
        scene_img_feat,
        scene_locs,
        scene_mask,
        custom_prompt,
        obj_ids,
        assigned_ids,
        obj_id_mask=None,
        **kwargs,
    ):
        """Evaluate query-object ranking without running the LLM."""
        object_embed, _ = self.encode_object_feat(
            scene_feat, scene_img_feat, scene_locs
        )
        graph_queries = [
            update_caption(custom_prompt[i], assigned_ids[i])
            for i in range(object_embed.shape[0])
        ]
        query_token_embeds, query_token_mask = self.get_query_token_embeds(
            graph_queries, device=object_embed.device
        )
        _, graph_info = self.query_graph_reasoner(
            scene_feat=object_embed,
            scene_locs=scene_locs,
            scene_mask=scene_mask,
            query_token_embeds=query_token_embeds,
            query_token_mask=query_token_mask,
            target_obj_ids=obj_ids,
            target_obj_mask=obj_id_mask,
        )
        return graph_info

    def evaluate(self, scene_feat, scene_img_feat, scene_locs, scene_mask, custom_prompt, obj_ids, assigned_ids, is_eval=True, **kwargs):
        object_embed, object_img_embed = self.encode_object_feat(scene_feat, scene_img_feat, scene_locs)
        device = object_embed.device
        batch_size, obj_num = object_embed.shape[:2]
        graph_queries = [
            update_caption(custom_prompt[i], assigned_ids[i])
            for i in range(batch_size)
        ]
        object_embed, _ = self.apply_query_graph_reasoning(
            object_embed, scene_locs, scene_mask, graph_queries
        )
        proj_object_embed = self.object_proj(object_embed)
        proj_object_img_embed = self.object_img_proj(object_img_embed)
        if self.add_pos_emb:
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
            pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs]) / 10
            proj_pos_embed = self.pos_proj(pos_embed)
            proj_object_embed = proj_object_embed + proj_pos_embed
            proj_object_img_embed = proj_object_img_embed + proj_pos_embed
        if self.add_scene_token:
            # if self.add_img_token:
            #     object_embed = object_embed + object_img_embed
            obj_embed = self.scene_init_proj(object_embed)
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
            pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs])
            pos_embed = self.pos_proj(pos_embed)
            scene_embed = obj_embed + pos_embed
            scene_embed = self.relation_module(scene_embed, src_key_padding_mask=~scene_mask)
            proj_scene_embed = self.scene_proj(scene_embed)

        output_texts = []
        p_0_embed = self.p_0_embed.to(device).unsqueeze(0)
        p_1_embed = self.p_1_embed.to(device).unsqueeze(0)
        for i in range(batch_size):
            tmp_prompt = f" {custom_prompt[i]} {self.role[1]}: "
            tmp_prompt = update_caption(tmp_prompt, assigned_ids[i])
            prompt_embed = self.get_text_emb(tmp_prompt, device=device)
            object_list_embed = self.get_object_list_embed(
                proj_object_embed[i],
                proj_object_img_embed[i] if self.add_img_token else None,
                proj_scene_embed[i] if self.add_scene_token else None,
                scene_mask[i],
                obj_ids[i],
                assigned_ids[i]
            )
            object_list_embed = object_list_embed.unsqueeze(0)
            wrapped_embed = torch.cat([p_0_embed, object_list_embed, p_1_embed, prompt_embed], dim=1)
            attention_mask=None
            if self.bidirection:
                seq_len = wrapped_embed.shape[1]
                attention_mask = torch.ones((seq_len, seq_len), dtype=wrapped_embed.dtype, device=device)
                attention_mask = torch.tril(attention_mask, diagonal=0)
                attention_mask = attention_mask[None, None, :, :].expand(1, 1, -1, -1).clone()
                st, ed = p_0_embed.shape[1], p_0_embed.shape[1] + object_list_embed.shape[1]
                attention_mask[:, :, st:ed, st:ed] = 1.0
            
            with self.maybe_autocast():
                gen_kwargs = dict(
                    inputs_embeds=wrapped_embed,
                    max_new_tokens=self.gen_max_txt_len,
                    num_beams=self.num_beams,
                    min_length=1,
                    repetition_penalty=3.0,
                    length_penalty=1,
                    temperature=1.0 if self.num_beams > 1 else None,
                    do_sample=False,
                    pad_token_id=self.llama_tokenizer.pad_token_id,
                    eos_token_id=self.llama_tokenizer.eos_token_id,
                )
                if self.is_vicuna:
                    gen_kwargs["customized_mask"] = attention_mask
                elif attention_mask is not None:
                    gen_kwargs["attention_mask"] = attention_mask
                outputs = self.llama_model.generate(**gen_kwargs)
            output_token = outputs[0]
            output_text = self.llama_tokenizer.decode(output_token)
            output_text = output_text.split(self.end_sym)[0]
            output_text = output_text.replace('  ', ' ').replace(' .', '.').strip()
            output_text = recover_caption(output_text, assigned_ids[i].tolist())
            output_texts.append(output_text)
        return output_texts

    def forward(self, **kwargs):
        if kwargs.pop("selector_eval", False):
            return self.evaluate_graph_selector(**kwargs)
        if "answers" in kwargs:
            return self.forward_train(**kwargs)
        if "custom_prompt" in kwargs:
            return self.evaluate(**kwargs)
        return None

    def _get_text_len(self, text):
        return self.llama_tokenizer(text, return_tensors="pt").input_ids.shape[1]

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @property
    def device(self):
        return list(self.parameters())[0].device
