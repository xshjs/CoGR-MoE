#!/usr/bin/env python3
"""
Unified VMC staged training entry:
- stage1: LoRA on dev split
- stage2: CoGR on dev split (requires LoRA adapter)
- both: stage1 then stage2

This script provides staged VMC training on dev split.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path as _Path

_core = _Path(__file__).resolve()
_core = _core.parent.parent if _core.parent.name == "vmc" else _core.parent
if str(_core) not in sys.path:
    sys.path.insert(0, str(_core))
from repo_bootstrap import bootstrap

bootstrap(__file__, chdir=True, include_models_pkg=True, include_llm=True)

import train_cogr as core  # noqa: E402
from training_probe_openai import add_training_probe_args  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--train_stage", choices=("stage1", "stage2", "both"), default="both")
    p.add_argument("--base_model_path", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--data_path", default="../../VMCBench")
    p.add_argument(
        "--output_dir",
        default="./vmc_staged_out",
        help="Unified output root; stage1/2 outputs are derived from this unless explicitly set.",
    )
    p.add_argument("--stage1_output_dir", default="")
    p.add_argument("--stage2_output_dir", default="")
    p.add_argument("--adapter_path", default="")
    p.add_argument("--resume_cogr_pt", default=None)

    p.add_argument("--categories", default="VQAv2,GQA,VizWiz,ScienceQA,MMVet,MMStar")
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--shuffle_train", action="store_true", default=True)
    p.add_argument("--no_shuffle_train", action="store_true")
    p.add_argument("--max_train_samples", type=int, default=-1)

    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--stage1_num_epochs", type=int, default=2)
    p.add_argument("--stage1_learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--stage1_warmup_steps", type=int, default=500)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--no_bf16", action="store_true")

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    p.add_argument("--train_lm_head", action="store_true", default=True)
    p.add_argument("--no_train_lm_head", action="store_true")
    p.add_argument("--freeze_lora", action="store_true")
    p.add_argument("--detach_cue_embeddings", action="store_true")

    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--lambda_router", type=float, default=0.5)
    p.add_argument("--beta_teacher", type=float, default=0.5)
    p.add_argument("--delta_clip", type=float, default=2.0)
    p.add_argument("--distill_weight", type=float, default=0.1)
    p.add_argument("--gamma_distill", type=float, default=0.1)
    p.add_argument("--beta_load_balance", type=float, default=0.01)
    p.add_argument("--legacy_student_routing", action="store_true")
    p.add_argument("--lambda_option_reweight", type=float, default=0.5)
    p.add_argument("--gamma_contrast", type=float, default=0.3)
    p.add_argument("--contrast_tau", type=float, default=0.07)

    add_training_probe_args(p)
    return p


def _normalize_paths(args: argparse.Namespace) -> None:
    root = args.output_dir
    if not args.stage1_output_dir:
        args.stage1_output_dir = os.path.join(root, "stage1_lora")
    if not args.stage2_output_dir:
        args.stage2_output_dir = os.path.join(root, "stage2_cogr")
    args.output_dir = args.stage2_output_dir
    if args.no_train_lm_head:
        args.train_lm_head = False
    if args.no_shuffle_train:
        args.shuffle_train = False


def run_main(legacy_mode: str | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _normalize_paths(args)

    if legacy_mode == "baseline":
        args.train_stage = "stage1"
    if args.train_stage == "stage1":
        core.train_stage1_lora(args)
        return

    if args.train_stage == "both":
        stage1_adapter_dir = core.train_stage1_lora(args)
        args.adapter_path = stage1_adapter_dir
        core.logger.info("Stage2 will use Stage1 adapter: %s", args.adapter_path)

    if args.train_stage == "stage2" and not args.adapter_path:
        raise ValueError("stage2 requires --adapter_path (or use --train_stage both)")

    core.train_stage2(args)


if __name__ == "__main__":
    run_main()

