#!/usr/bin/env python3
"""
Routing Components
职责：图像-问题编码、语义投影、Teacher/Student路由计算
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)


def encode_image_question_pair(image, question, qwen3_model, qwen3_processor, device):
    """
    编码图像-问题对，获得语义表示 s_question
    
    Args:
        image: PIL.Image
        question: str
        qwen3_model: Qwen3模型
        qwen3_processor: Qwen3处理器
        device: 设备
    
    Returns:
        s_question: [1, hidden_dim] tensor
    """
    try:
        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            if hasattr(qwen3_processor, 'apply_chat_template'):
                text = qwen3_processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,  # 编码时不需要generation prompt
                )
            else:
                text = question
            
            inputs = qwen3_processor(
                text=[text],
                images=[image],
                return_tensors="pt",
            )
        else:
            inputs = qwen3_processor(
                text=[question],
                return_tensors="pt",
            )
        
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        with torch.no_grad():
            outputs = qwen3_model(**inputs)
            
            if hasattr(outputs, 'last_hidden_state'):
                s_question = outputs.last_hidden_state.mean(dim=1)  # [1, hidden_dim]
            elif hasattr(outputs, 'pooler_output'):
                s_question = outputs.pooler_output
            else:
                s_question = outputs.logits.mean(dim=1)
        
        logger.debug(f" 图像-问题编码完成: s_question.shape={s_question.shape}")
        return s_question
        
    except Exception as e:
        logger.error(f" 图像-问题编码失败: {e}")
        import traceback
        logger.error(f"完整错误堆栈:\n{traceback.format_exc()}")
        raise


class SemanticProjector(nn.Module):
    """语义投影层：计算选项的语义向量"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_sem = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.normal_(self.W_sem.weight, std=0.02)
        
        logger.info(f" SemanticProjector 初始化: hidden_dim={hidden_dim}")
    
    def compute_s_answer(self, p_must_have: torch.Tensor, p_must_not: torch.Tensor) -> torch.Tensor:
        """
        计算正确选项的语义方向
        s_answer = W_sem(p_must_have - p_must_not)
        
        Args:
            p_must_have: [1, hidden_dim] 正向探针编码
            p_must_not: [1, hidden_dim] 负向探针编码
        
        Returns:
            s_answer: [1, hidden_dim]
        """
        s_answer = self.W_sem(p_must_have - p_must_not)
        logger.debug(f" s_answer 计算完成: {s_answer.shape}")
        return s_answer
    
    def compute_s_option(self, option_embedding: torch.Tensor) -> torch.Tensor:
        """
        为每个选项计算语义向量
        s_a = W_sem(option_embedding)
        
        Args:
            option_embedding: [1, hidden_dim] 选项的嵌入向量
        
        Returns:
            s_a: [1, hidden_dim]
        """
        s_a = self.W_sem(option_embedding)
        return s_a


class TeacherStudentRouter(nn.Module):
    """Teacher-Student路由计算器"""
    
    def __init__(self, 
                 hidden_dim: int,
                 num_experts: int,
                 lambda_param: float = 1.0,
                 beta: float = 0.5,
                 delta: float = 2.0,
                 eps: float = 1e-8):
        """
        Args:
            hidden_dim: 隐藏维度
            num_experts: 专家数量
            lambda_param: 偏置强度系数 λ
            beta: Teacher routing中s_answer和s_question的融合权重
            delta: Clip函数的边界
            eps: 数值稳定性常数
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.lambda_param = lambda_param
        self.beta = beta
        self.delta = delta
        self.eps = eps
        
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        
        logger.info(f" TeacherStudentRouter 初始化: hidden_dim={hidden_dim}, num_experts={num_experts}")

    def compute_expert_bias_clipped(self, s_vec: torch.Tensor) -> torch.Tensor:
        """
        b = Clip((W s - mean) / (||W s|| + eps), -delta, delta)，与 Teacher/Student 偏置同形。
        s_vec: [batch, hidden_dim]
        returns: [batch, num_experts]
        """
        s_proj = self.router(s_vec)
        s_mean = s_proj.mean(dim=-1, keepdim=True)
        s_norm = s_proj.norm(p=2, dim=-1, keepdim=True) + self.eps
        normalized = (s_proj - s_mean) / s_norm
        return torch.clamp(normalized, -self.delta, self.delta)

    def compute_teacher_routing(self,
                               s_answer: torch.Tensor,
                               s_question: torch.Tensor,
                               unc_vis_teacher: float,
                               z_base: Optional[torch.Tensor] = None,
                               hidden_states: Optional[torch.Tensor] = None,
                               return_grad: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算Teacher路由（SAGE路由）
        
        Args:
            s_answer: [1, hidden_dim] 正确选项的语义方向
            s_question: [1, hidden_dim] 问题的语义表示
            unc_vis_teacher: float Teacher的视觉不确定性（不再使用）
            z_base: [1, num_experts] 基础路由logits（可选）
            hidden*.states: [1, seq_len, hidden_dim] 如果未提供z_base，用此计算
            return_grad: 是否返回可计算梯度的g_teacher（用于L_align）
        
        Returns:
            g_teacher: [1, num_experts] Teacher gating分布
                       - 如果return_grad=False: detached（用于路由选择）
                       - 如果return_grad=True: 不detached（用于L_align计算）
            TopK: [top_k] Teacher选择的TopK专家索引
            z_final: [1, num_experts] Teacher 路由 logits（softmax 前）
        """
        s_teacher = self.beta * s_answer + (1.0 - self.beta) * s_question  # [1, hidden_dim]
        
        c_teacher = 1.0  # 不再使用unc_vis_teacher
        
        if z_base is None:
            if hidden_states is None:
                raise ValueError("必须提供z_base或hidden_states")
            hidden_mean = hidden_states.mean(dim=1)  # [1, hidden_dim]
            z_base = self.router(hidden_mean)  # [1, num_experts]
        else:
            if z_base.dim() == 3:
                z_base = z_base.mean(dim=1)  # [batch, seq, num_experts] -> [batch, num_experts]
        
        s_sage_proj = self.router(s_teacher)  # [1, num_experts]
        s_sage_mean = s_sage_proj.mean(dim=-1, keepdim=True)  # [1, 1]
        s_sage_norm = s_sage_proj.norm(p=2, dim=-1, keepdim=True) + self.eps  # [1, 1]
        
        normalized_s_sage = (s_sage_proj - s_sage_mean) / s_sage_norm  # [1, num_experts]
        b_teacher = torch.clamp(normalized_s_sage, -self.delta, self.delta)  # [1, num_experts]
        
        z_final = z_base + self.lambda_param * b_teacher
        g_teacher = F.softmax(z_final, dim=-1)  # [1, num_experts]
        
        g_teacher_detached = g_teacher.detach()
        top_k = min(2, self.num_experts)
        _, TopK = torch.topk(g_teacher_detached, top_k, dim=-1)
        TopK = TopK.squeeze(0).cpu().numpy()  # [top_k]
        
        if not return_grad:
            g_teacher = g_teacher.detach()  # 阻止梯度（用于路由选择）
        
        logger.debug(f" Teacher routing完成: TopK={TopK}, g_teacher_sum={g_teacher.sum().item():.4f}, return_grad={return_grad}")
        return g_teacher, TopK, z_final

    def compute_student_softmax_z_base(
        self,
        z_base: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """g^S = softmax(z_base)，仅依赖图文隐状态，不注入选项语义。"""
        if z_base is None:
            if hidden_states is None:
                raise ValueError("z_base or hidden_states required")
            hidden_mean = hidden_states.mean(dim=1)
            z_base = self.router(hidden_mean)
        else:
            if z_base.dim() == 3:
                z_base = z_base.mean(dim=1)
        return F.softmax(z_base, dim=-1)

    def compute_student_routing(self,
                               s_question: torch.Tensor,
                               z_base: Optional[torch.Tensor] = None,
                               hidden_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算Student路由（无cue，仅使用图像-问题语义表示）
        
        公式: g^(S) = softmax(z_base + λ * b_student)
        其中 b_student 由 s_question 或其投影生成，不包含任何 cue 信息
        
        Args:
            s_question: [1, hidden_dim] 问题的语义表示（图像-问题对编码）
            z_base: [1, num_experts] 基础路由logits（可选）
            hidden_states: [1, seq_len, hidden_dim] 如果未提供z_base，用此计算
        
        Returns:
            g_student: [1, num_experts] Student gating分布（可训练）
        """
        if z_base is None:
            if hidden_states is None:
                raise ValueError("必须提供z_base或hidden_states")
            hidden_mean = hidden_states.mean(dim=1)  # [1, hidden_dim]
            z_base = self.router(hidden_mean)  # [1, num_experts]
        else:
            if z_base.dim() == 3:
                z_base = z_base.mean(dim=1)  # [batch, seq, num_experts] -> [batch, num_experts]
        
        s_question_proj = self.router(s_question)  # [1, num_experts]
        
        s_question_mean = s_question_proj.mean(dim=-1, keepdim=True)  # [1, 1]
        s_question_norm = s_question_proj.norm(p=2, dim=-1, keepdim=True) + self.eps  # [1, 1]
        
        normalized = (s_question_proj - s_question_mean) / s_question_norm  # [1, num_experts]
        b_student = torch.clamp(normalized, -self.delta, self.delta)  # [1, num_experts]
        
        z_final = z_base + self.lambda_param * b_student
        g_student = F.softmax(z_final, dim=-1)  # [1, num_experts]
        
        logger.debug(f" Student routing完成: g_student_sum={g_student.sum().item():.4f}")
        return g_student
    
    def compute_student_routing_with_topk(self,
                                         s_question: torch.Tensor,
                                         z_base: Optional[torch.Tensor] = None,
                                         hidden_states: Optional[torch.Tensor] = None,
                                         top_k: int = 2) -> Tuple[torch.Tensor, np.ndarray]:
        """
        计算Student路由并返回TopK专家索引（用于推理）
        
        Args:
            s_question: [1, hidden_dim] 问题的语义表示（图像-问题对编码）
            z_base: [1, num_experts] 基础路由logits（可选）
            hidden_states: [1, seq_len, hidden_dim] 如果未提供z_base，用此计算
            top_k: TopK专家数量
        
        Returns:
            g_student: [1, num_experts] Student gating分布
            TopK: [top_k] Student选择的TopK专家索引
        """
        g_student = self.compute_student_routing(s_question, z_base, hidden_states)
        
        g_student_detached = g_student.detach()
        top_k = min(top_k, self.num_experts)
        _, TopK = torch.topk(g_student_detached, top_k, dim=-1)
        TopK = TopK.squeeze(0).cpu().numpy()  # [top_k]
        
        logger.debug(f" Student routing with TopK: TopK={TopK}")
        return g_student, TopK
    
    def compute_distill_loss(self,
                             g_teacher: torch.Tensor,
                             g_student: torch.Tensor) -> torch.Tensor:
        """
        计算Teacher → Student蒸馏损失
        
        公式: L_distill = KL(g^(T) || g^(S))
        使Student在无cue条件下复现Teacher的语义路由结构
        
        Args:
            g_teacher: [batch_size, num_experts] 或 [1, num_experts] Teacher gating分布
            g_student: [batch_size, num_experts] 或 [1, num_experts] Student gating分布
        
        Returns:
            distill_loss: scalar tensor KL散度损失
        """
        if g_teacher.dim() == 1:
            g_teacher = g_teacher.unsqueeze(0)  # [1, num_experts]
        if g_student.dim() == 1:
            g_student = g_student.unsqueeze(0)  # [1, num_experts]
        
        eps = 1e-8
        g_teacher = g_teacher + eps
        g_student = g_student + eps
        
        g_teacher = g_teacher / g_teacher.sum(dim=-1, keepdim=True)
        g_student = g_student / g_student.sum(dim=-1, keepdim=True)
        
        log_q = torch.log(g_student + eps)
        kl_div = F.kl_div(
            log_q,
            g_teacher,
            reduction='batchmean',
            log_target=False,
        )
        
        logger.debug(f" 蒸馏损失计算完成: KL={kl_div.item():.6f}")
        return kl_div

