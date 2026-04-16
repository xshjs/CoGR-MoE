#!/usr/bin/env python3
"""Backward-compatible entry: baseline stage1 now routes to unified staged trainer."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_core = _Path(__file__).resolve()
_core = _core.parent.parent if _core.parent.name == "vmc" else _core.parent
if str(_core) not in sys.path:
    sys.path.insert(0, str(_core))

from train_vmc_staged import run_main  # noqa: E402


if __name__ == "__main__":
    run_main(legacy_mode="baseline")

