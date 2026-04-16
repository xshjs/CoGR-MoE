#!/usr/bin/env python3
"""Compute MC accuracy by VMCBench category (Yes-logit protocol, consistent with Stage-A training)."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_core = _Path(__file__).resolve()
_core = _core.parent.parent if _core.parent.name == "vmc" else _core.parent
if str(_core) not in sys.path:
    sys.path.insert(0, str(_core))
from repo_bootstrap import bootstrap

bootstrap(__file__, chdir=True, include_models_pkg=False, include_llm=False)

import argparse
import base64
import io
import json
import os
from collections import defaultdict

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from peft import PeftModel


def decode_image(cell) -> Image.Image | None:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None
    s = str(cell)
    if len(s) < 100:
        return None
    try:
        raw = base64.b64decode(s)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_path", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--adapter_path", default="./vmc_lora_baseline_out/best_lora")
    p.add_argument("--data_path", default="../../VMCBench")
    p.add_argument("--split", default="test", choices=("test", "dev"))
    p.add_argument(
        "--categories",
        default="VQAv2,GQA,VizWiz,ScienceQA,MMVet,MMStar",
    )
    p.add_argument("--output_json", default="./vmc_per_category_acc.json")
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--max_samples", type=int, default=-1, help="-1 means full set (global head, not per-class)")
    p.add_argument(
        "--max_per_category",
        type=int,
        default=-1,
        help="Maximum evaluated samples per category (first N in TSV order); takes precedence over max_samples when >0",
    )
    args = p.parse_args()

    cats = [x.strip() for x in args.categories.split(",") if x.strip()]
    tsv = os.path.join(
        args.data_path, "data", "tsv", f"VMCBench_{args.split.upper()}.tsv"
    )
    df = pd.read_csv(tsv, sep="\t")
    df = df[df["category"].isin(cats)].reset_index(drop=True)
    if args.max_per_category > 0:
        parts = [df[df["category"] == c].head(args.max_per_category) for c in cats]
        df = pd.concat(parts, ignore_index=True)
    elif args.max_samples > 0:
        df = df.head(args.max_samples)

    use_bf16 = not args.no_bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    amp_dtype = dtype

    print(f"Loading base from {args.base_model_path} ...")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model_path,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(args.base_model_path)
    if args.adapter_path:
        print(f"Loading LoRA from {args.adapter_path} ...")
        model.model.language_model = PeftModel.from_pretrained(
            model.model.language_model,
            args.adapter_path,
        )
    model.eval()

    yid = processor.tokenizer.encode("Yes", add_special_tokens=False)
    yes_token_id = yid[0] if yid else 3869
    device = str(next(model.parameters()).device)

    correct = defaultdict(int)
    total = defaultdict(int)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="eval"):
        cat = str(row.get("category", ""))
        q = str(row.get("question", ""))
        opts = {
            "A": str(row.get("A", "")),
            "B": str(row.get("B", "")),
            "C": str(row.get("C", "")),
            "D": str(row.get("D", "")),
        }
        ans = str(row.get("answer", "A"))
        img = decode_image(row.get("image"))
        correct_key = ans.upper() if len(ans) == 1 else "A"
        label_idx = ["A", "B", "C", "D"].index(correct_key)

        scores = []
        for k in ["A", "B", "C", "D"]:
            image_tag = "<image>\n" if img else ""
            prompt = (
                f"USER: {image_tag}{q}\nOption: {opts[k]}\n"
                f"Is this option correct?\nASSISTANT: Yes"
            )
            if img is not None:
                inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)
            else:
                inputs = processor(text=prompt, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.startswith("cuda")):
                out = model(**inputs)
            s = out.logits[:, -1, yes_token_id].float().clamp(-100.0, 100.0).item()
            scores.append(s)

        logits = torch.tensor([scores], dtype=torch.float32)
        pred = logits.argmax(dim=-1).item()
        total[cat] += 1
        if pred == label_idx:
            correct[cat] += 1

    rows = []
    overall_c, overall_n = 0, 0
    for c in sorted(cats):
        n = total.get(c, 0)
        k = correct.get(c, 0)
        row = {"category": c, "n": int(n), "correct": int(k)}
        row["acc"] = (k / n) if n else None
        rows.append(row)
        overall_c += k
        overall_n += n

    overall_acc = overall_c / overall_n if overall_n else 0.0
    out = {
        "split": args.split,
        "adapter_path": args.adapter_path,
        "max_per_category": args.max_per_category,
        "max_samples": args.max_samples,
        "n_total": overall_n,
        "acc_overall": overall_acc,
        "per_category": rows,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
