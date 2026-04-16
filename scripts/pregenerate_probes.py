#!/usr/bin/env python3
"""
Pre-generate probes for all VMCBench samples (OpenAI, aligned with training_probe_openai).
Saves to a JSONL file: one line per sample with probes.
Run once; training scripts then read this file (without API key, a complete JSONL must already exist).

Usage:
    python pregenerate_probes.py \
        --data_path ../../VMCBench --split test --output probes_test.jsonl
"""

import argparse
import logging
import os
import sys
from pathlib import Path as _Path

_core = _Path(__file__).resolve()
_core = _core.parent.parent if _core.parent.name == "vmc" else _core.parent
if str(_core) not in sys.path:
    sys.path.insert(0, str(_core))
from repo_bootstrap import bootstrap

bootstrap(__file__, chdir=True, include_models_pkg=True, include_llm=False)

import pandas as pd

from training_probe_openai import add_training_probe_args, generate_probes_openai_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(data_path: str, split: str) -> pd.DataFrame:
    tsv_path = os.path.join(data_path, "data", "tsv", f"VMCBench_{split.upper()}.tsv")
    if os.path.exists(tsv_path):
        return pd.read_csv(tsv_path, sep="\t")
    raise FileNotFoundError(f"No data at {tsv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../../VMCBench")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="probes_test.jsonl", help="Output JSONL path")
    parser.add_argument("--start", type=int, default=0, help="Start index (for resuming)")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    add_training_probe_args(parser)
    args = parser.parse_args()

    df = load_dataset(args.data_path, args.split)

    if args.num_shards > 1:
        df = df.iloc[args.shard_id :: args.num_shards].reset_index(drop=True)
        logger.info("Shard %s/%s: %s samples", args.shard_id, args.num_shards, len(df))

    if args.start > 0:
        df = df.iloc[args.start :].reset_index(drop=True)
        logger.info("After --start=%s: %s samples", args.start, len(df))

    out_path = args.output or args.probe_output_path
    if not out_path:
        raise SystemExit("Please provide --output or --probe_output_path")

    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY or --openai_api_key is required for online probe generation."
        )

    resume = bool(args.probe_resume) or os.path.isfile(out_path)
    generate_probes_openai_jsonl(
        df,
        out_path,
        api_key=api_key,
        model=args.probe_openai_model,
        resume=resume,
        sleep_s=float(args.probe_sleep_s),
    )
    logger.info("Done! probes -> %s", out_path)


if __name__ == "__main__":
    main()
