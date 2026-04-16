"""Lightweight GPU utility helpers for training scripts."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def log_device_info() -> None:
    """Log available CUDA devices if torch is present."""
    try:
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            logger.info("GPU available: %s device(s)", n)
            for i in range(n):
                try:
                    name = torch.cuda.get_device_name(i)
                except Exception:
                    name = "<unknown>"
                logger.info("  - cuda:%s -> %s", i, name)
        else:
            logger.info("CUDA not available")
    except Exception:
        logger.info("torch unavailable; skip GPU info")


def format_device_string(gpu_spec: Any) -> str:
    """Normalize device spec for logging."""
    if isinstance(gpu_spec, str):
        return gpu_spec
    if isinstance(gpu_spec, (list, tuple)):
        return ",".join(str(x) for x in gpu_spec)
    try:
        return str(int(gpu_spec))
    except Exception:
        return str(gpu_spec)


def get_physical_gpu_id() -> int:
    """Return current CUDA device index, or -1."""
    try:
        import torch

        if torch.cuda.is_available():
            try:
                return int(torch.cuda.current_device())
            except Exception:
                return 0
    except Exception:
        pass
    return -1


__all__ = ["log_device_info", "format_device_string", "get_physical_gpu_id"]
