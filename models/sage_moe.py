"""
SAGE MoE Integration Layer.

Integrates Teacher-Student routing from routing_components.py into the MoE-LLaVA
forward pass. Provides evidence-biased routing and distillation loss computation.

Core flow:
    Training:  image+question -> SAGE Router (Teacher with s_answer, Student without)
    Inference: image+question -> Student Router only (no cue)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

sys_path_inserted = False

logger = logging.getLogger(__name__)


def _ensure_imports():
    """Lazily import from CoGR-MoE models directory."""
    global sys_path_inserted
    if not sys_path_inserted:
        import sys
        import os
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models"
        )
        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)
        sys_path_inserted = True


@dataclass
class SAGEConfig:
    """Configuration for SAGE routing."""
    hidden_dim: int = 4096
    num_experts: int = 4
    top_k: int = 2
    lambda_param: float = 1.0
    beta: float = 0.5
    delta: float = 2.0
    distill_weight: float = 0.1
    load_balance_weight: float = 0.01
    pure_student_softmax_z: bool = False
    lambda_option_reweight: float = 0.0


class SAGERouter(nn.Module):
    def __init__(self, config: SAGEConfig):
        super().__init__()
        self.config = config
        _ensure_imports()

        from routing_components import TeacherStudentRouter

        self.router = TeacherStudentRouter(
            hidden_dim=config.hidden_dim,
            num_experts=config.num_experts,
            lambda_param=config.lambda_param,
            beta=config.beta,
            delta=config.delta,
        )
        self.last_z_teacher_logits: Optional[torch.Tensor] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        s_question: Optional[torch.Tensor] = None,
        s_answer: Optional[torch.Tensor] = None,
        unc_vis: float = 0.0,
        is_training: bool = True,
        s_option: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        hidden_mean = hidden_states.mean(dim=1)
        z_base = self.router.router(hidden_mean)

        distill_loss = torch.tensor(0.0, device=hidden_states.device)
        self.last_z_teacher_logits = None

        if is_training and s_answer is not None and s_question is not None:
            g_teacher, _topk_np, z_final = self.router.compute_teacher_routing(
                s_answer=s_answer,
                s_question=s_question,
                unc_vis_teacher=unc_vis,
                z_base=z_base,
                return_grad=True,
            )
            self.last_z_teacher_logits = z_final

            if self.config.pure_student_softmax_z:
                g_student = self.router.compute_student_softmax_z_base(z_base=z_base)
            else:
                g_student = self.router.compute_student_routing(
                    s_question=s_question,
                    z_base=z_base,
                )

            distill_loss = self.router.compute_distill_loss(g_teacher, g_student)

            if self.config.lambda_option_reweight > 0.0 and s_option is not None:
                b_opt = self.router.compute_expert_bias_clipped(s_option)
                z_moe = z_final + self.config.lambda_option_reweight * b_opt
                g_moe = F.softmax(z_moe, dim=-1)
            else:
                g_moe = g_teacher

            routing_probs = g_moe.unsqueeze(1).expand(-1, seq_len, -1)
        elif s_question is not None:
            g_student, _ = self.router.compute_student_routing_with_topk(
                s_question=s_question,
                z_base=z_base,
                top_k=self.config.top_k,
            )
            routing_probs = g_student.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            routing_probs = F.softmax(z_base, dim=-1).unsqueeze(1).expand(-1, seq_len, -1)

        top_k_weights, top_k_indices = torch.topk(routing_probs, self.config.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        return routing_probs, top_k_indices, distill_loss


class SAGEMoELayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 4096,
        intermediate_dim: int = 11008,
        num_experts: int = 4,
        top_k: int = 2,
        sage_config: Optional[SAGEConfig] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = hidden_dim

        from moe_ffn import ExpertFFN
        self.experts = nn.ModuleList([ExpertFFN(hidden_dim, intermediate_dim) for _ in range(num_experts)])

        if sage_config is None:
            sage_config = SAGEConfig(hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k)
        self.sage_router = SAGERouter(sage_config)

        self.simple_router = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        s_question: Optional[torch.Tensor] = None,
        s_answer: Optional[torch.Tensor] = None,
        unc_vis: float = 0.0,
        is_training: bool = True,
        s_option: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_dim)

        if s_question is not None:
            routing_probs, _top_k_indices, distill_loss = self.sage_router(
                hidden_states,
                s_question,
                s_answer,
                unc_vis,
                is_training,
                s_option=s_option,
            )
            routing_probs_flat = routing_probs.reshape(-1, self.num_experts)
        else:
            router_logits = self.simple_router(flat_hidden)
            routing_probs_flat = F.softmax(router_logits, dim=-1)
            distill_loss = torch.tensor(0.0, device=hidden_states.device)

        top_k_weights, top_k_indices_flat = torch.topk(routing_probs_flat, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(flat_hidden)
        for expert_idx in range(self.num_experts):
            expert_mask = (top_k_indices_flat == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue

            expert_weight = torch.zeros(
                flat_hidden.shape[0], device=flat_hidden.device, dtype=flat_hidden.dtype
            )
            for k_idx in range(self.top_k):
                mask_k = top_k_indices_flat[:, k_idx] == expert_idx
                w = top_k_weights[mask_k, k_idx].to(
                    dtype=expert_weight.dtype, device=expert_weight.device
                )
                expert_weight[mask_k] = w

            selected = flat_hidden[expert_mask]
            expert_out = self.experts[expert_idx](selected)
            output[expert_mask] += expert_weight[expert_mask].unsqueeze(-1) * expert_out

        with torch.no_grad():
            expert_mask_all = (top_k_indices_flat.unsqueeze(-1) == torch.arange(
                self.num_experts, device=hidden_states.device
            )).any(dim=1).float()
            tokens_per_expert = expert_mask_all.mean(dim=0)
        router_prob_per_expert = routing_probs_flat.mean(dim=0)
        load_balance_loss = (
            self.num_experts
            * torch.sum(tokens_per_expert * router_prob_per_expert)
            * 0.01
        )

        output = output.view(batch_size, seq_len, hidden_dim)
        return output, load_balance_loss, distill_loss
