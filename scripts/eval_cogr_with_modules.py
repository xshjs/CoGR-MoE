#!/usr/bin/env python3
"""Evaluate on test TSV: LLaVA + LoRA + injected final-layer CoGR (SemanticProjector + SAGEMoE)."""

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
import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from routing_components import SemanticProjector  # noqa: E402

from sage_moe import SAGEConfig, SAGEMoELayer  # noqa: E402
from vmc_data_utils import (  # noqa: E402
    VMCBenchDataFrameDataset,
    _parse_categories,
    load_split_df,
    yes_token_id,
)
from train_cogr import (  # noqa: E402
    AdditiveCogrMlp,
    _encode_text_mean,
    _find_llama_layers,
    _get_embed_tokens,
)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_path", required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--cogr_modules_path", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--categories", default="", help="Comma-separated categories; empty means no filtering")
    p.add_argument("--use_fp16", action="store_true")
    p.add_argument("--detach_cue_embeddings", action="store_true")
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--lambda_router", type=float, default=0.5)
    p.add_argument("--beta_teacher", type=float, default=0.5)
    p.add_argument("--delta_clip", type=float, default=2.0)
    p.add_argument("--distill_weight", type=float, default=0.1)
    p.add_argument("--legacy_student_routing", action="store_true")
    p.add_argument("--lambda_option_reweight", type=float, default=0.0)
    p.add_argument("--gamma_contrast", type=float, default=0.0)
    p.add_argument("--contrast_tau", type=float, default=0.07)
    p.add_argument(
        "--output_json",
        default="",
        help="If non-empty, writes grouped and per-category stats to JSON",
    )
    args = p.parse_args()

    perspective_cats = frozenset({"Angle", "Partial", "Scope", "Obstruction"})
    transformative_cats = frozenset({"Temporal", "Deformation", "Incomplete", "Biological"})
    others_cats = frozenset({"Others"})

    cats = _parse_categories(args.categories) if args.categories.strip() else None
    df = load_split_df(args.data_path, "test", cats)

    dtype = torch.float16 if args.use_fp16 else torch.bfloat16
    amp_dtype = dtype

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

    ck = torch.load(args.cogr_modules_path, map_location="cpu")
    num_experts = int(ck.get("num_experts", args.num_experts))
    top_k = int(ck.get("top_k", args.top_k))

    sage_cfg = SAGEConfig(
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
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
        num_experts=num_experts,
        top_k=top_k,
        sage_config=sage_cfg,
    ).to(device=layer_dev, dtype=dtype)

    wrapped = AdditiveCogrMlp(old_mlp, sage_moe).to(device=layer_dev)
    target_layer.mlp = wrapped

    semantic_proj = SemanticProjector(hidden_dim).to(device=layer_dev, dtype=dtype)
    semantic_proj.load_state_dict(ck["semantic_projector"], strict=True)
    miss, unexp = wrapped.load_state_dict(ck["additive_cogr_mlp"], strict=False)
    if miss:
        print("warning: additive_cogr_mlp missing keys (up to 8):", list(miss)[:8])
    if unexp:
        print("warning: additive_cogr_mlp unexpected keys (up to 8):", list(unexp)[:8])

    embed_tokens = _get_embed_tokens(model)
    cue_device = next(semantic_proj.parameters()).device
    tokenizer = processor.tokenizer
    yid = yes_token_id(processor)
    device = torch.device(str(next(model.parameters()).device))

    model.eval()
    semantic_proj.eval()
    wrapped.eval()

    ds = VMCBenchDataFrameDataset(df, max_samples=-1)
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: {k: [d[k] for d in b] for k in b[0]},
    )

    tot_ce = 0.0
    correct = 0
    n = 0
    per_cat_correct: dict[str, int] = defaultdict(int)
    per_cat_total: dict[str, int] = defaultdict(int)
    group_correct: dict[str, int] = defaultdict(int)
    group_total: dict[str, int] = defaultdict(int)

    def _bump(cat: str, hit: bool) -> None:
        per_cat_total[cat] += 1
        if hit:
            per_cat_correct[cat] += 1
        g = ""
        if cat in perspective_cats:
            g = "Perspective"
        elif cat in transformative_cats:
            g = "Transformative"
        elif cat in others_cats:
            g = "Others"
        if g:
            group_total[g] += 1
            if hit:
                group_correct[g] += 1

    for batch in tqdm(loader, desc="Validating"):
        questions = batch["question"]
        options = batch["options"]
        answers = batch["answer"]
        images = batch.get("image")
        categories = batch.get("category", [""] * len(questions))
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
            h_correct = None
            for ok_i, opt_key in enumerate(["A", "B", "C", "D"]):
                opt_text = opts[opt_key]
                s_opt_raw = emb_opts[ok_i]
                s_option_vec = semantic_proj.compute_s_option(s_opt_raw)
                mlp_wrap.set_cues(base_sq, s_answer, 0.0, s_option=s_option_vec)
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
                    out = model(**inputs, output_hidden_states=use_hs)
                score = out.logits[:, -1, yid].float().clamp(-100.0, 100.0)
                option_scores.append(score)
                if use_hs and out.hidden_states is not None:
                    hl = out.hidden_states[-1][:, -1, :].float()
                    label_idx_pre = ["A", "B", "C", "D"].index(correct_key)
                    if ok_i == label_idx_pre:
                        h_correct = hl.squeeze(0)

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
            loss = ce + args.gamma_contrast * contrast_term

            tot_ce += loss.item()
            pred = logits.argmax(dim=-1).item()
            hit = pred == label_idx
            if hit:
                correct += 1
            n += 1
            raw_cat = categories[i] if i < len(categories) else ""
            raw_cat = str(raw_cat).strip() if raw_cat is not None else ""
            if raw_cat:
                _bump(raw_cat, hit)

    vacc = correct / max(n, 1)
    vloss = tot_ce / max(n, 1)
    print(f"eval_samples={n}")
    print(f"val_loss={vloss:.6f}")
    print(f"val_acc={vacc:.6f}")

    def _acc(c: int, t: int) -> float:
        return (c / t) if t > 0 else float("nan")

    print("\n--- Super-groups ---")
    for g in ("Perspective", "Transformative", "Others"):
        t = group_total[g]
        c = group_correct[g]
        print(f"{g}: n={t} acc={_acc(c, t):.6f}")

    print("\n--- Fine categories (MRAG scenario) ---")
    order = sorted(
        set(per_cat_total.keys()),
        key=lambda x: (
            0
            if x in perspective_cats
            else (1 if x in transformative_cats else (2 if x in others_cats else 3)),
            x,
        ),
    )
    for cat in order:
        t = per_cat_total[cat]
        c = per_cat_correct[cat]
        print(f"{cat}: n={t} acc={_acc(c, t):.6f}")

    if args.output_json.strip():
        out = {
            "eval_samples": n,
            "val_loss": vloss,
            "val_acc": vacc,
            "super_groups": {
                g: {
                    "n": group_total[g],
                    "correct": group_correct[g],
                    "acc": _acc(group_correct[g], group_total[g]),
                }
                for g in ("Perspective", "Transformative", "Others")
            },
            "per_category": {
                cat: {
                    "n": per_cat_total[cat],
                    "correct": per_cat_correct[cat],
                    "acc": _acc(per_cat_correct[cat], per_cat_total[cat]),
                }
                for cat in order
            },
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.output_json}")


if __name__ == "__main__":
    main()
