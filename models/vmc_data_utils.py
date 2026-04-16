#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import logging
import os
from typing import List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration
from torch.utils.data import Dataset

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
        logger.warning("TSV has no category column, skip filtering")
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
        tot_loss += F.cross_entropy(
            logits, torch.tensor([label_idx], device=logits.device, dtype=torch.long)
        ).item()
        total += 1
        if logits.argmax(dim=-1).item() == label_idx:
            correct += 1

    model.train()
    return tot_loss / max(total, 1), correct / max(total, 1)
