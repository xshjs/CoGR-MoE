#!/usr/bin/env python3
"""
MRAG: stage1 LLaVA+LoRA, stage2 CoGR/SAGE (Yes-logit 4-way selection).

Default data is `../../MRAG-Bench/mrag_20_for_vmc` (split_20_80 train_indices, about 20%).
Run `scripts/prepare_mrag_20_tsv.py` first. Default runs both stages.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_core = _Path(__file__).resolve()
_core = _core.parent.parent if _core.parent.name == "vmc" else _core.parent
if str(_core) not in sys.path:
    sys.path.insert(0, str(_core))
from repo_bootstrap import bootstrap

bootstrap(__file__, chdir=True, include_models_pkg=True, include_llm=False)

import argparse
import base64
import io
import json
import logging
import math
import os
from typing import List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import LoraConfig, PeftModel, get_peft_model

from training_probe_openai import add_training_probe_args, ensure_training_probes_ready

from routing_components import SemanticProjector  # noqa: E402
from sage_moe import SAGEConfig, SAGEMoELayer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_categories(s: str) -> Optional[List[str]]:
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def load_split_df(data_path: str, split: str, categories: Optional[List[str]]) -> pd.DataFrame:
    tsv = os.path.join(data_path, "data", "tsv", f"VMCBench_{split.upper()}.tsv")
    df = pd.read_csv(tsv, sep="\t")
    if not categories:
        return df
    if "category" not in df.columns:
        logger.warning("TSV has no category column; skipping category filter")
        return df
    n0 = len(df)
    df = df[df["category"].isin(categories)].reset_index(drop=True)
    logger.info("Filter %s: %d -> %d", split, n0, len(df))
    return df


class VMCBenchDataFrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_samples: int = -1):
        self.df = df if max_samples <= 0 else df.head(max_samples).copy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image = None
        img_data = row.get("image", None)
        if img_data is not None and isinstance(img_data, str) and len(img_data) > 100:
            try:
                raw = base64.b64decode(img_data)
                image = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                image = None
        return {
            "question": str(row.get("question", "")),
            "options": {
                "A": str(row.get("A", "")),
                "B": str(row.get("B", "")),
                "C": str(row.get("C", "")),
                "D": str(row.get("D", "")),
            },
            "answer": str(row.get("answer", "A")),
            "image": image,
            "category": str(row.get("category", "")),
        }


def yes_token_id(processor) -> int:
    ids = processor.tokenizer("Yes", add_special_tokens=False).input_ids
    return int(ids[0]) if ids else 3869


@torch.no_grad()
def validate(
    model: LlavaForConditionalGeneration,
    processor,
    val_df: pd.DataFrame,
    device: str,
    amp_dtype: torch.dtype,
    yid: int,
) -> Tuple[float, float]:
    model.eval()
    tot_loss = 0.0
    correct = 0
    total = 0
    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Validating", leave=False):
        q = str(row.get("question", ""))
        opts = {
            "A": str(row.get("A", "")),
            "B": str(row.get("B", "")),
            "C": str(row.get("C", "")),
            "D": str(row.get("D", "")),
        }
        ans = str(row.get("answer", "A"))
        img_data = row.get("image", None)
        image = None
        if img_data is not None and isinstance(img_data, str) and len(img_data) > 100:
            try:
                raw = base64.b64decode(img_data)
                image = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                image = None
        correct_key = ans.upper() if len(ans) == 1 else "A"
        label_idx = ["A", "B", "C", "D"].index(correct_key)
        scores = []
        for k in ["A", "B", "C", "D"]:
            image_tag = "<image>\n" if image is not None else ""
            prompt = (
                f"USER: {image_tag}{q}\nOption: {opts[k]}\n"
                f"Is this option correct?\nASSISTANT: Yes"
            )
            if image is not None:
                inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
            else:
                inputs = processor(text=prompt, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=str(device).startswith("cuda")):
                out = model(**inputs)
            s = out.logits[:, -1, yid].float().clamp(-100.0, 100.0).item()
            scores.append(s)
        logits = torch.tensor(scores, dtype=torch.float32).unsqueeze(0)
        li = label_idx
        tot_loss += F.cross_entropy(logits, torch.tensor([li], device=logits.device, dtype=torch.long)).item()
        total += 1
        if logits.argmax(dim=-1).item() == li:
            correct += 1
    model.train()
    return tot_loss / max(total, 1), correct / max(total, 1)

def _find_llama_layers(lm_module: nn.Module) -> nn.ModuleList:
    """Locate LlamaDecoderLayer stack inside Peft-wrapped language_model."""

    def recur(m: nn.Module, depth: int = 0):
        if depth > 16:
            return None
        if hasattr(m, "layers") and isinstance(m.layers, nn.ModuleList) and len(m.layers) > 0:
            L0 = m.layers[0]
            if hasattr(L0, "self_attn") and hasattr(L0, "mlp"):
                return m.layers
        for c in m.children():
            r = recur(c, depth + 1)
            if r is not None:
                return r
        return None

    out = recur(lm_module)
    if out is None:
        raise RuntimeError("Cannot find Llama layers")
    return out


def _get_embed_tokens(model: LlavaForConditionalGeneration) -> nn.Module:
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None:
            return emb
    lm = model.model.language_model
    cur = lm
    for _ in range(8):
        if hasattr(cur, "get_base_model"):
            cur = cur.get_base_model()
        elif hasattr(cur, "base_model"):
            cur = cur.base_model
        else:
            break
    if hasattr(cur, "model") and hasattr(cur.model, "embed_tokens"):
        return cur.model.embed_tokens
    raise RuntimeError("Cannot find embed_tokens")


class AdditiveCogrMlp(nn.Module):
    """Last FFN: original LlamaMLP (with LoRA) plus parallel SAGEMoE and scalar mix."""

    def __init__(self, builtin_mlp: nn.Module, sage_moe: SAGEMoELayer):
        super().__init__()
        self.builtin_mlp = builtin_mlp
        self.sage_moe = sage_moe
        self.mix = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self._sq = None
        self._sa = None
        self._s_option = None
        self._unc: float = 0.0
        self.last_lb = torch.tensor(0.0)
        self.last_distill = torch.tensor(0.0)

    def set_cues(
        self,
        s_question: torch.Tensor,
        s_answer: torch.Tensor,
        unc: float = 0.0,
        s_option: torch.Tensor | None = None,
    ):
        self._sq = s_question
        self._sa = s_answer
        self._unc = unc
        self._s_option = s_option

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.builtin_mlp(x)
        ys, lb, dl = self.sage_moe(
            x,
            self._sq,
            self._sa,
            self._unc,
            self.training,
            s_option=self._s_option,
        )
        self.last_lb = lb
        self.last_distill = dl
        alpha = self.mix.to(dtype=y0.dtype, device=y0.device)
        return y0 + alpha * ys


@torch.no_grad()
def _encode_text_mean(
    embed_tokens: nn.Module,
    tokenizer,
    text: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    emb = embed_tokens(ids)
    if emb.dtype != dtype:
        emb = emb.to(dtype)
    return emb.mean(dim=1)


def train_stage2(args: argparse.Namespace) -> None:
    categories = _parse_categories(args.categories)
    os.makedirs(args.output_dir, exist_ok=True)

    train_df = load_split_df(args.data_path, "dev", categories)
    if args.shuffle_train:
        train_df = train_df.sample(frac=1.0, random_state=args.split_seed).reset_index(drop=True)

    meta = {
        "train_split": "dev",
        "n_train": len(train_df),
        "script": "train_mrag.py",
    }
    with open(os.path.join(args.output_dir, "split_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    ensure_training_probes_ready(args, train_df, args.output_dir)

    use_bf16 = not args.no_bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    amp_dtype = dtype

    logger.info("Loading LLaVA + LoRA from %s", args.adapter_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model_path,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(args.base_model_path)
    model.model.language_model = PeftModel.from_pretrained(
        model.model.language_model,
        args.adapter_path,
    )

    cfg = model.config
    text_cfg = getattr(cfg, "text_config", None) or cfg
    hidden_dim = int(getattr(text_cfg, "hidden_size", 4096))
    intermediate_dim = int(getattr(text_cfg, "intermediate_size", 11008))

    layers = _find_llama_layers(model.model.language_model)
    target_layer = layers[-1]
    old_mlp = target_layer.mlp
    layer_dev = next(target_layer.parameters()).device

    sage_cfg = SAGEConfig(
        hidden_dim=hidden_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
        lambda_param=args.lambda_router,
        beta=args.beta_teacher,
        delta=args.delta_clip,
        distill_weight=args.distill_weight,
        pure_student_softmax_z=not args.legacy_student_routing,
        lambda_option_reweight=args.lambda_option_reweight,
    )
    sage_moe = SAGEMoELayer(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
        sage_config=sage_cfg,
    ).to(device=layer_dev, dtype=dtype)

    wrapped = AdditiveCogrMlp(old_mlp, sage_moe).to(device=layer_dev)
    target_layer.mlp = wrapped

    semantic_proj = SemanticProjector(hidden_dim).to(device=layer_dev, dtype=dtype)

    if getattr(args, "resume_cogr_pt", None):
        ck_path = args.resume_cogr_pt
        if ck_path and os.path.isfile(ck_path):
            logger.info("Loading CoGR modules from %s", ck_path)
            ck = torch.load(ck_path, map_location="cpu")
            semantic_proj.load_state_dict(ck["semantic_projector"], strict=True)
            miss, unexp = wrapped.load_state_dict(ck["additive_cogr_mlp"], strict=False)
            if miss:
                logger.warning("additive_cogr_mlp missing (show up to 8): %s", list(miss)[:8])
            if unexp:
                logger.warning("additive_cogr_mlp unexpected (show up to 8): %s", list(unexp)[:8])

    embed_tokens = _get_embed_tokens(model)
    cue_device = next(semantic_proj.parameters()).device
    tokenizer = processor.tokenizer

    yid = yes_token_id(processor)
    device = torch.device(str(next(model.parameters()).device))

    train_lora = not args.freeze_lora
    for n, p in model.named_parameters():
        if "sage_moe" in n or "semantic_proj" in n or "mix" in n:
            p.requires_grad = True
        elif "lora_" in n.lower() or "lora." in n:
            p.requires_grad = train_lora
        elif "lm_head" in n:
            p.requires_grad = args.train_lm_head
        else:
            p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable += list(semantic_proj.parameters())
    logger.info(
        "Trainable (incl. semantic_proj): %s",
        f"{sum(p.numel() for p in trainable):,}",
    )

    train_ds = VMCBenchDataFrameDataset(train_df, max_samples=args.max_train_samples)
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda b: {k: [d[k] for d in b] for k in b[0]},
    )

    params = [p for p in model.parameters() if p.requires_grad] + list(semantic_proj.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    opt_steps = max(1, math.ceil(len(train_loader) / max(args.gradient_accumulation_steps, 1)))
    total_opt_steps = max(1, opt_steps * args.num_epochs)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return float(step + 1) / float(max(1, args.warmup_steps))
        prog = float(step - args.warmup_steps) / float(max(1, total_opt_steps - args.warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    global_step = 0

    model.train()
    semantic_proj.train()

    for epoch in range(args.num_epochs):
        epoch_ce = 0.0
        epoch_aux = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"CoGR Epoch {epoch + 1}/{args.num_epochs}")

        for step, batch in enumerate(pbar):
            questions = batch["question"]
            options = batch["options"]
            answers = batch["answer"]
            images = batch.get("image")

            for i in range(len(questions)):
                q = questions[i]
                opts = options[i] if isinstance(options[i], dict) else options[0]
                ans = answers[i]
                img = images[i] if images is not None else None
                if img is not None and not isinstance(img, Image.Image):
                    try:
                        img = Image.open(img).convert("RGB") if isinstance(img, str) else img
                    except Exception:
                        img = None

                correct_key = ans.upper() if len(ans) == 1 else "A"
                wrong_keys = [k for k in ["A", "B", "C", "D"] if k != correct_key]

                pos_text = opts[correct_key]
                neg_text = " ".join(opts[k] for k in wrong_keys)
                s_question = _encode_text_mean(embed_tokens, tokenizer, q, device, amp_dtype).to(cue_device)
                p_pos = _encode_text_mean(embed_tokens, tokenizer, pos_text, device, amp_dtype).to(cue_device)
                p_neg = _encode_text_mean(embed_tokens, tokenizer, neg_text, device, amp_dtype).to(cue_device)
                s_answer = semantic_proj.compute_s_answer(p_pos, p_neg)

                mlp_wrap = layers[-1].mlp
                assert isinstance(mlp_wrap, AdditiveCogrMlp)
                base_sq = s_question.detach() if args.detach_cue_embeddings else s_question

                emb_opts = []
                for opt_key in ["A", "B", "C", "D"]:
                    ot = opts[opt_key]
                    emb_opts.append(_encode_text_mean(embed_tokens, tokenizer, ot, device, amp_dtype).to(cue_device))
                emb_stack = torch.stack(emb_opts, dim=0)

                option_scores = []
                aux_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
                h_correct: torch.Tensor | None = None
                for ok_i, opt_key in enumerate(["A", "B", "C", "D"]):
                    opt_text = opts[opt_key]
                    s_opt_raw = emb_opts[ok_i]
                    s_option_vec = semantic_proj.compute_s_option(s_opt_raw)
                    mlp_wrap.set_cues(
                        base_sq,
                        s_answer,
                        0.0,
                        s_option=s_option_vec,
                    )
                    image_tag = "<image>\n" if img else ""
                    prompt = (
                        f"USER: {image_tag}{q}\nOption: {opt_text}\n"
                        f"Is this option correct?\nASSISTANT: Yes"
                    )
                    if img is not None:
                        inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)
                    else:
                        inputs = processor(text=prompt, return_tensors="pt").to(device)

                    use_hs = args.gamma_contrast > 0.0
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                        out = model(
                            **inputs,
                            output_hidden_states=use_hs,
                        )
                    score = out.logits[:, -1, yid].float().clamp(-100.0, 100.0)
                    option_scores.append(score)
                    if use_hs and out.hidden_states is not None:
                        hl = out.hidden_states[-1][:, -1, :].float()
                        label_idx_pre = ["A", "B", "C", "D"].index(correct_key)
                        if ok_i == label_idx_pre:
                            h_correct = hl.squeeze(0)

                    d = mlp_wrap.last_distill.float().to(device)
                    lb = mlp_wrap.last_lb.float().to(device)
                    aux_sum = aux_sum + args.gamma_distill * d + args.beta_load_balance * lb

                logits = torch.stack([s.reshape(-1) for s in option_scores], dim=1)
                label_idx = ["A", "B", "C", "D"].index(correct_key)
                label = torch.full((logits.size(0),), label_idx, device=logits.device, dtype=torch.long)
                ce = F.cross_entropy(logits, label)
                contrast_term = torch.tensor(0.0, device=device, dtype=torch.float32)
                if args.gamma_contrast > 0.0 and h_correct is not None:
                    tau = max(args.contrast_tau, 1e-6)
                    h_n = F.normalize(h_correct, dim=-1)
                    e_n = F.normalize(emb_stack.float(), dim=-1)
                    logits_c = (h_n.unsqueeze(0) @ e_n.T).squeeze(0) / tau
                    contrast_term = F.cross_entropy(
                        logits_c.unsqueeze(0),
                        torch.tensor([label_idx], device=device, dtype=torch.long),
                    )

                loss = (ce + aux_sum / 4.0 + args.gamma_contrast * contrast_term) / args.gradient_accumulation_steps

                if torch.isfinite(loss):
                    loss.backward()
                    epoch_ce += ce.item()
                    epoch_aux += (aux_sum / 4.0).item()
                else:
                    logger.warning("Non-finite loss, skip step")

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            pbar.set_postfix(
                {
                    "ce": f"{epoch_ce / max(step + 1, 1):.4f}",
                    "aux": f"{epoch_aux / max(step + 1, 1):.4f}",
                }
            )

        logger.info(
            "Epoch %d done: avg_ce≈%.4f avg_aux≈%.4f",
            epoch + 1,
            epoch_ce / max(len(train_loader), 1),
            epoch_aux / max(len(train_loader), 1),
        )

    save_lora = os.path.join(args.output_dir, "best_lora")
    os.makedirs(save_lora, exist_ok=True)
    model.model.language_model.save_pretrained(save_lora)
    processor.save_pretrained(save_lora)

    mlp_wrap = layers[-1].mlp
    torch.save(
        {
            "semantic_projector": semantic_proj.state_dict(),
            "additive_cogr_mlp": mlp_wrap.state_dict(),
            "sage_moe": mlp_wrap.sage_moe.state_dict(),
            "additive_mix": mlp_wrap.mix.detach().cpu(),
            "hidden_dim": hidden_dim,
            "intermediate_dim": intermediate_dim,
            "num_experts": args.num_experts,
            "top_k": args.top_k,
        },
        os.path.join(args.output_dir, "cogr_modules.pt"),
    )

    meta = {
        "adapter_path": args.adapter_path,
        "resume_cogr_pt": getattr(args, "resume_cogr_pt", None),
        "train_epochs": args.num_epochs,
        "inject": "last_layer_additive_sage_moe",
        "loss": "CE + gamma*KL_distill + beta*load_balance + eta*L_contrast (optional)",
    }
    with open(os.path.join(save_lora, "cogr_train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info("Saved LoRA -> %s | CoGR modules -> %s", save_lora, os.path.join(args.output_dir, "cogr_modules.pt"))


def train_stage1_mrag_lora(args: argparse.Namespace) -> str:
    """Stage1: train LoRA strictly on DEV only."""
    categories = _parse_categories(args.categories)
    os.makedirs(args.lora_output_dir, exist_ok=True)

    df_dev = load_split_df(args.data_path, "dev", categories)
    meta = {
        "mode": "mrag_dev_only",
        "categories": categories,
        "split_seed": args.split_seed,
        "n_dev": len(df_dev),
        "n_train_total": len(df_dev),
    }

    train_df = df_dev.copy()
    if args.shuffle_train:
        train_df = train_df.sample(frac=1.0, random_state=args.split_seed).reset_index(drop=True)

    with open(os.path.join(args.lora_output_dir, "split_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info("Split summary: %s", json.dumps(meta, ensure_ascii=False))

    ensure_training_probes_ready(args, train_df, args.lora_output_dir)

    use_bf16 = not args.no_bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    amp_dtype = dtype

    logger.info("Loading LLaVA from %s", args.base_model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model_path,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(args.base_model_path)
    yid = yes_token_id(processor)

    if args.resume_adapter:
        logger.info("Resume training from existing LoRA: %s", args.resume_adapter)
        model.model.language_model = PeftModel.from_pretrained(model.model.language_model, args.resume_adapter)
    else:
        target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=None,
            target_modules=target_modules,
        )
        model.model.language_model = get_peft_model(model.model.language_model, lora_config)

    for p in model.model.vision_tower.parameters():
        p.requires_grad = False
    for p in model.model.multi_modal_projector.parameters():
        p.requires_grad = False
    if args.train_lm_head:
        for p in model.lm_head.parameters():
            p.requires_grad = True
    else:
        for p in model.lm_head.parameters():
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{trainable:,}")

    device = str(next(model.parameters()).device)

    train_ds = VMCBenchDataFrameDataset(train_df, max_samples=args.max_train_samples)
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda b: {k: [d[k] for d in b] for k in b[0]},
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.stage1_learning_rate, weight_decay=args.weight_decay)
    opt_steps_per_epoch = max(1, math.ceil(len(train_loader) / max(args.gradient_accumulation_steps, 1)))
    total_opt_steps = max(1, opt_steps_per_epoch * args.stage1_num_epochs)

    def lr_lambda(step: int) -> float:
        if step < args.stage1_warmup_steps:
            return float(step + 1) / float(max(1, args.stage1_warmup_steps))
        prog = float(step - args.stage1_warmup_steps) / float(max(1, total_opt_steps - args.stage1_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    for epoch in range(args.stage1_num_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Stage1 Epoch {epoch + 1}/{args.stage1_num_epochs}")
        for step, batch in enumerate(pbar):
            questions = batch["question"]
            options = batch["options"]
            answers = batch["answer"]
            images = batch.get("image")
            for i in range(len(questions)):
                q = questions[i]
                opts = options[i] if isinstance(options[i], dict) else options[0]
                ans = answers[i]
                img = images[i] if images is not None else None
                if img is not None and not isinstance(img, Image.Image):
                    try:
                        img = Image.open(img).convert("RGB") if isinstance(img, str) else img
                    except Exception:
                        img = None
                correct_key = ans.upper() if len(ans) == 1 else "A"
                option_scores = []
                for opt_key in ["A", "B", "C", "D"]:
                    opt_text = opts[opt_key]
                    image_tag = "<image>\n" if img is not None else ""
                    prompt = (
                        f"USER: {image_tag}{q}\nOption: {opt_text}\n"
                        f"Is this option correct?\nASSISTANT: Yes"
                    )
                    if img is not None:
                        inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)
                    else:
                        inputs = processor(text=prompt, return_tensors="pt").to(device)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.startswith("cuda")):
                        out = model(**inputs)
                    score = out.logits[:, -1, yid].float().clamp(-100.0, 100.0)
                    option_scores.append(score)
                logits = torch.stack([s.reshape(-1) for s in option_scores], dim=1)
                label_idx = ["A", "B", "C", "D"].index(correct_key)
                label = torch.full((logits.size(0),), label_idx, device=logits.device, dtype=torch.long)
                loss = F.cross_entropy(logits, label) / args.gradient_accumulation_steps
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                epoch_loss += loss.item() * args.gradient_accumulation_steps

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({"avg_ce": f"{epoch_loss / max(step + 1, 1):.4f}"})

        avg_loss = epoch_loss / max(len(train_loader), 1)
        logger.info(
            "Stage1 Epoch %d: avg_loss=%.4f lr=%s",
            epoch + 1,
            avg_loss,
            scheduler.get_last_lr(),
        )

    save_dir = os.path.join(args.lora_output_dir, "best_lora")
    os.makedirs(save_dir, exist_ok=True)
    model.model.language_model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    meta_save = {
        "yes_token_id": yid,
        "split_seed": args.split_seed,
        "num_epochs": args.stage1_num_epochs,
        "resume_adapter": args.resume_adapter,
        "train_split": "dev",
    }
    with open(os.path.join(save_dir, "baseline_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_save, f, indent=2, ensure_ascii=False)
    logger.info("Saved stage1 LoRA -> %s", save_dir)

    out_lora = os.path.join(args.lora_output_dir, "best_lora")
    logger.info("Stage1 completed on DEV-only training split.")
    return out_lora


def main() -> None:
    p = argparse.ArgumentParser(description="MRAG: stage1 LoRA, stage2 CoGR+LoRA")
    p.add_argument(
        "--train_stage",
        choices=("stage1", "stage2", "both"),
        default="both",
        help="stage1: LoRA only; stage2: CoGR only (requires existing LoRA); both: LoRA then CoGR",
    )
    p.add_argument("--base_model_path", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument(
        "--data_path",
        default="../../MRAG-Bench/mrag_20_for_vmc",
        help="MRAG 20%% TSV root (contains data/tsv/VMCBench_DEV.tsv)",
    )
    p.add_argument(
        "--lora_output_dir",
        default="./mrag20_lora_out",
        help="Stage1 LoRA output directory",
    )
    p.add_argument(
        "--adapter_path",
        default="./mrag20_lora_out/best_lora",
        help="LoRA loaded by stage2 (defaults to stage1 output)",
    )
    p.add_argument("--cogr_output_dir", default="./mrag20_cogr_out", help="Stage2 CoGR output directory")
    p.add_argument(
        "--categories",
        default="",
        help="Comma-separated categories; empty means no filtering",
    )
    p.add_argument("--resume_adapter", type=str, default=None)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--shuffle_train", action="store_true", default=True)
    p.add_argument("--no_shuffle_train", action="store_true")
    p.add_argument("--max_train_samples", type=int, default=-1)
    p.add_argument("--stage1_num_epochs", type=int, default=2)
    p.add_argument("--stage1_learning_rate", type=float, default=2e-4)
    p.add_argument("--stage1_warmup_steps", type=int, default=500)
    p.add_argument("--num_epochs", type=int, default=1, help="Number of stage2 CoGR epochs")
    p.add_argument("--learning_rate", type=float, default=1e-4, help="Stage2 learning rate")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=100)
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
    p.add_argument(
        "--legacy_student_routing",
        action="store_true",
        help="Use legacy student routing with s_question bias; default uses softmax(z_base)",
    )
    p.add_argument(
        "--lambda_option_reweight",
        type=float,
        default=0.0,
        help="Strength of option-semantic bias on teacher z^T in MoE routing; 0 means original g_teacher-only behavior",
    )
    p.add_argument("--gamma_contrast", type=float, default=0.0, help="Enable contrastive term between final hidden states and option embeddings when >0")
    p.add_argument("--contrast_tau", type=float, default=0.07)
    p.add_argument(
        "--resume_cogr_pt",
        default=None,
        help="Resume SemanticProjector and additive MLP (including SAGE) from existing cogr_modules.pt",
    )
    add_training_probe_args(p)
    args = p.parse_args()
    if args.no_train_lm_head:
        args.train_lm_head = False
    if args.no_shuffle_train:
        args.shuffle_train = False

    if args.train_stage == "stage1":
        train_stage1_mrag_lora(args)
        return

    if args.train_stage == "both":
        adapter_dir = train_stage1_mrag_lora(args)
        args.adapter_path = adapter_dir
        logger.info("Stage2 will use stage1 LoRA: %s", args.adapter_path)

    if args.train_stage in ("stage2", "both"):
        args.output_dir = args.cogr_output_dir
        train_stage2(args)
        return

    raise RuntimeError("unreachable")


if __name__ == "__main__":
    main()
