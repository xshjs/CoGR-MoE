#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


def _add_path(p: Path) -> None:
    s = str(p.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


def bootstrap(
    script_path: str,
    *,
    chdir: bool = True,
    include_models_pkg: bool = True,
    include_llm: bool = False,
) -> None:
    script_dir = Path(script_path).resolve().parent
    repo_root = script_dir.parent
    llm_dir = repo_root / "models" / "moe-llava"

    if include_models_pkg:
        _add_path(repo_root / "models")
    if include_llm:
        _add_path(llm_dir)

    if chdir:
        os.chdir(script_dir)

