#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
MRAG_DIR = str(REPO_ROOT.parent / "MRAG-Bench")
SPLIT_JSON = os.path.join(MRAG_DIR, "data", "split_20_80", "split_indices.json")
OUT_DIR = os.path.join(MRAG_DIR, "mrag_80_for_vmc")


def _img_to_b64(img_obj) -> str:
    if isinstance(img_obj, dict):
        if img_obj.get("bytes") is not None:
            raw = img_obj["bytes"]
        elif img_obj.get("path"):
            with open(img_obj["path"], "rb") as f:
                raw = f.read()
        else:
            return ""
    elif isinstance(img_obj, bytes):
        raw = img_obj
    elif isinstance(img_obj, Image.Image):
        buf = BytesIO()
        img_obj.save(buf, format="PNG")
        raw = buf.getvalue()
    else:
        return ""
    return base64.b64encode(raw).decode("utf-8")


def main() -> None:
    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        split = json.load(f)
    use_indices = split["test_indices"]  # 80%

    data_dir = os.path.join(MRAG_DIR, "data")
    files = sorted([x for x in os.listdir(data_dir) if x.endswith(".parquet")])
    df = pd.concat([pd.read_parquet(os.path.join(data_dir, x)) for x in files], ignore_index=True)
    sub = df.iloc[use_indices].reset_index(drop=True)

    out = pd.DataFrame(
        {
            "question": sub["question"].astype(str),
            "A": sub["A"].astype(str),
            "B": sub["B"].astype(str),
            "C": sub["C"].astype(str),
            "D": sub["D"].astype(str),
            "answer": sub["answer_choice"].astype(str),
            "category": sub.get("scenario", pd.Series(["MRAG"] * len(sub))).astype(str),
            "image": sub["image"].apply(_img_to_b64),
        }
    )

    tsv_dir = os.path.join(OUT_DIR, "data", "tsv")
    os.makedirs(tsv_dir, exist_ok=True)
    dev_path = os.path.join(tsv_dir, "VMCBench_DEV.tsv")
    test_path = os.path.join(tsv_dir, "VMCBench_TEST.tsv")
    out.to_csv(dev_path, sep="\t", index=False)
    out.to_csv(test_path, sep="\t", index=False)
    meta = {
        "mode": "eval_only_80_pool",
        "source": "test_indices",
        "n_samples": len(out),
        "note": "DEV/TEST are mirrored for evaluation compatibility; do not use as training data.",
    }
    with open(os.path.join(OUT_DIR, "split_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"saved: {dev_path}")
    print(f"saved: {test_path}")
    print(f"samples: {len(out)}")


if __name__ == "__main__":
    main()
