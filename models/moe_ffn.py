"""MoE FFN building blocks (Llama-style SiLU gate)."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    """Single expert: SwiGLU block matching Llama MLP shape."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEFFNConfig:
    """Legacy stub for imports."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MoEFFNLayer(nn.Module):
    """Legacy stub."""

    pass
