#!/usr/bin/env python3
"""
Pre-generate probes for MRAG-Bench using Qwen3-VL-30B.
Output: probes_mrag_test.jsonl (one line per sample)

Usage (single process):
    CUDA_VISIBLE_DEVICES=4,5,6,7 python pregen_mrag_probes.py

Usage (4-process parallel, 1 GPU each):
    bash run_pregen_mrag.sh
"""

import os
import sys
from pathlib import Path as _Path

_core = _Path(__file__).resolve()
_core = _core.parent.parent if _core.parent.name == "vmc" else _core.parent
if str(_core) not in sys.path:
    sys.path.insert(0, str(_core))
from repo_bootstrap import bootstrap

bootstrap(__file__, chdir=True, include_models_pkg=False, include_llm=False)

import json
import argparse
import logging
import io
import base64
from tqdm import tqdm

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = _Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = str(REPO_ROOT / "qwen3-vl-30b-instruct")
DEFAULT_MRAG_DATA_DIR = str(REPO_ROOT.parent / "MRAG-Bench" / "data")

PROBE_PROMPT = """You are a visual evidence aligner. Given an image-question pair with candidate options (correct answer: {answer}), generate must_have and must_not probes for each option.

[Input]
- Question: '{question}'
- Options: {options_text}
- Correct answer: '{answer}'

For each option, list visual features that MUST appear (must_have) or MUST NOT appear (must_not) in the image to support/contradict this option.

[Output format - strict JSON only]
{{
  "per_option_probes": [
    {{"option": "<text>", "group": "correct" or "incorrect", "must_have": ["<feature>"], "must_not": ["<feature>"]}}
  ]
}}"""


def decode_image(img_data):
    """Decode image from various formats."""
    if img_data is None:
        return None
    if isinstance(img_data, dict):
        if 'bytes' in img_data and img_data['bytes'] is not None:
            return Image.open(io.BytesIO(img_data['bytes'])).convert("RGB")
        if 'path' in img_data and img_data['path']:
            return Image.open(img_data['path']).convert("RGB")
    if isinstance(img_data, bytes):
        return Image.open(io.BytesIO(img_data)).convert("RGB")
    if isinstance(img_data, str) and len(img_data) > 100:
        try:
            return Image.open(io.BytesIO(base64.b64decode(img_data))).convert("RGB")
        except Exception:
            return None
    if isinstance(img_data, Image.Image):
        return img_data
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--output", default="probes_mrag_test.jsonl")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_MRAG_DATA_DIR)
    args = parser.parse_args()

    import pandas as pd
    files = sorted(f for f in os.listdir(args.data_dir) if f.endswith(".parquet"))
    dfs = [pd.read_parquet(os.path.join(args.data_dir, f)) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    if args.num_shards > 1:
        df = df.iloc[args.shard_id::args.num_shards].reset_index(drop=True)
        logger.info(f"Shard {args.shard_id}/{args.num_shards}: {len(df)} samples")

    total = len(df)
    logger.info(f"Total samples: {total}")

    logger.info(f"Loading Qwen3-VL-30B from {args.model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    logger.info("Model loaded")

    done = 0
    mode = "a" if args.num_shards == 1 and os.path.exists(args.output) else "w"

    with open(args.output, mode) as f:
        for idx in tqdm(range(total), desc="Generating probes"):
            row = df.iloc[idx]
            question = str(row.get("question", ""))
            options = {
                "A": str(row.get("A", "")),
                "B": str(row.get("B", "")),
                "C": str(row.get("C", "")),
                "D": str(row.get("D", "")),
            }
            answer_key = str(row.get("answer_choice", "A"))
            answer_text = str(row.get("answer", options.get(answer_key, "")))
            image = decode_image(row.get("image", None))

            option_list = [options[k] for k in ["A", "B", "C", "D"]]
            options_text = ", ".join(f"{k}. {v}" for k, v in options.items())
            prompt_text = PROBE_PROMPT.format(question=question, options_text=options_text, answer=answer_text)

            try:
                if image is not None:
                    messages = [{"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text},
                    ]}]
                else:
                    messages = [{"role": "user", "content": [
                        {"type": "text", "text": prompt_text},
                    ]}]

                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if image is not None:
                    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
                else:
                    inputs = processor(text=[text], return_tensors="pt").to(model.device)

                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)

                input_len = inputs["input_ids"].shape[1]
                response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)

                import re
                json_match = re.search(r'\{[\s\S]*\}', response)
                probes = json.loads(json_match.group(0)) if json_match else {"per_option_probes": []}

            except Exception as e:
                logger.warning(f"Sample {idx} failed: {e}")
                probes = {
                    "per_option_probes": [
                        {"option": opt, "group": "correct" if opt == answer_text else "incorrect",
                         "must_have": [], "must_not": []}
                        for opt in option_list
                    ]
                }

            line = {
                "index": int(row.get("id", idx)),
                "question": question,
                "options": options,
                "answer_key": answer_key,
                "answer": answer_text,
                "probes": probes,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            f.flush()
            done += 1

            if done % 50 == 0:
                logger.info(f"Progress: {done}/{total}")

    logger.info(f"Done! {done} probes saved to {args.output}")


if __name__ == "__main__":
    main()
