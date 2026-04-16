"""
Training prerequisite: provide a probe JSONL covering current train_df, or use OpenAI key for generation.

Prompt and parser follow four-option must_have/must_not structure.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    import pandas as pd
    from PIL import Image

logger = logging.getLogger(__name__)


def add_training_probe_args(p: Any) -> None:
    p.add_argument(
        "--openai_api_key",
        default=None,
        help="OpenAI API key; if omitted, reads OPENAI_API_KEY env var (training is blocked without key and complete probe file).",
    )
    p.add_argument(
        "--probe_openai_model",
        default="gpt-4o-mini",
        help="OpenAI model name for probe generation (must support vision when images are used).",
    )
    p.add_argument(
        "--probe_output_path",
        default=None,
        help="Probe JSONL path; defaults to <output_dir>/probes_train.jsonl (output_dir provided by caller).",
    )
    p.add_argument(
        "--probe_resume",
        action="store_true",
        help="If output exists, skip already-written rows by index.",
    )
    p.add_argument(
        "--probe_sleep_s",
        type=float,
        default=0.0,
        help="Sleep seconds after each request for rate limiting.",
    )


def _decode_image_from_row(row: Any) -> Optional["Image.Image"]:
    import base64 as b64mod
    import io as iomod

    from PIL import Image

    img_data = row.get("image", None)
    if img_data is not None and isinstance(img_data, str) and len(img_data) > 100:
        try:
            raw = b64mod.b64decode(img_data)
            return Image.open(iomod.BytesIO(raw)).convert("RGB")
        except Exception:
            return None
    return None


def _option_dict(row: Any) -> Dict[str, str]:
    return {k: str(row.get(k, "")) for k in ("A", "B", "C", "D")}


def _answer_letter(row: Any) -> str:
    a = str(row.get("answer", "A")).strip().upper()
    if len(a) == 1 and a in "ABCD":
        return a
    return "A"


def _openai_client(api_key: str):
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("Probe generation requires: pip install openai") from e
    return OpenAI(api_key=api_key)


def _openai_chat_once(
    client: Any,
    model: str,
    prompt_text: str,
    image: Optional["Image.Image"],
) -> str:
    content: List[Dict[str, Any]] = []
    if image is not None:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )
    content.append({"type": "text", "text": prompt_text})
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=1024,
        temperature=0.3,
    )
    choice = resp.choices[0].message
    return (choice.content or "").strip()


def _build_enhanced_prompt(question: str, options: List[str], answer: str) -> str:
    """Build the enhanced probe prompt in English."""
    options_text = "\n".join([f"- {opt}" for opt in options])
    answer_str = str(answer)
    prompt_parts = [
        "You are a visual evidence aligner. Given an image-question pair (I, Q) and candidate options "
        f"(the correct answer is {answer_str}), generate must_have and must_not probes for each option.\n",
        "We score each option with consensus_score(a) = alpha * score_must_have(a) + (1-alpha) * score_must_not(a).\n\n",
        "Input:\n",
        "- Image I (visible)\n",
        f"- Question Q: '{question}'\n",
        f"- Candidate options:\n{options_text}\n",
        f"- Correct answer: '{answer}'\n\n",
        "Task-type self-classification (multi-label allowed):\n",
        "Identify which of these categories are involved and give confidence in [0,1]:\n",
        "- Perception\n",
        "- Counting\n",
        "- Spatial\n",
        "- OCR / Text\n",
        "- Commonsense & Knowledge\n",
        "- Reasoning\n\n",
        "Heuristics (examples only, do not copy literally):\n",
        "- Counting cues: how many, number of, most, least, more/less than\n",
        "- Spatial cues: left/right/up/down/next to/between/distance/position/direction/angle\n",
        "- OCR cues: text, number, sign, label, read/write, sequence, plate, receipt, display\n",
        "- Perception cues: color, material, texture, category, existence\n",
        "- Knowledge cues: likely place, flag, profession, usage, commonsense\n",
        "- Reasoning cues: why, causality, multi-step inference, contradiction, entailment, complex constraints\n\n",
        "Dynamic resource allocation by category:\n",
        "- Perception: k_have=3, k_not=2, alpha=0.7\n",
        "- Counting: k_have=2, k_not=2, alpha=0.6\n",
        "- Spatial: k_have=3, k_not=2, alpha=0.65\n",
        "- OCR/Text: k_have=3, k_not=2, alpha=0.85\n",
        "- Commonsense/Knowledge: k_have=2, k_not=2, alpha=0.6\n",
        "- Reasoning: k_have=3, k_not=2, alpha=0.75\n\n",
        "If multiple categories apply, merge budgets with caps k_have<=4 and k_not<=3, and compute alpha as a confidence-weighted average.\n\n",
        "Grouping policy:\n",
        "- Correct option (a = answer):\n",
        "  - must_have: minimum verifiable evidence that supports a.\n",
        "  - must_not: evidence that would weaken a if true.\n",
        "- Incorrect option (a != answer):\n",
        "  - must_have: features likely absent or conflicting with the image.\n",
        "  - must_not: strong visible evidence that contradicts the option.\n\n",
        "OCR constraints (if OCR is triggered):\n",
        '- Each OCR must_have should include "text_span", "context_left/right" (<=5 chars), and "rough_location".\n',
        '- Do not hallucinate text. If uncertain, use empty text_span with low certainty.\n',
        "- For must_not, prefer missing strings or near-miss confusions (edit distance ~1) and state the difference.\n\n",
        "Reasoning constraints (if Reasoning is triggered):\n",
        '- Use "premises" (2-3 observable facts) + one-sentence "rule".\n',
        "- Only use visible facts or stated question facts; avoid external facts as decisive evidence.\n\n",
        "Quality rules:\n",
        "- Evidence must be concrete, localizable, and verifiable.\n",
        "- Avoid subjective or non-observable attributes.\n",
        '- If evidence is weak, lower "certainty" instead of inventing details.\n\n',
        "Output format (strict JSON only, no extra explanation):\n",
        "{\n",
        '  "task_types": [{"type": "<OneOf[Perception,Counting,Spatial,OCR,Knowledge,Reasoning]>", "confidence": 0.0~1.0}, ...],\n',
        '  "alpha": <0.0~1.0>,\n',
        '  "per_option_probes": [\n',
        "    {\n",
        '      "option": "<text of a>",\n',
        '      "group": "correct" | "incorrect",\n',
        '      "must_have": [\n',
        '        { "cue": "<phrase/evidence>", "why": "<one verifiable sentence>", "certainty": 0.0~1.0,\n',
        '          "ocr": {"text_span": "<verbatim or empty>", "context_left": "<=5 or empty>", "context_right": "<=5 or empty>", "rough_location": "<optional>"},\n',
        '          "reasoning": {"premises": ["<observable fact>"], "rule": "<one sentence>", "used": true|false}\n',
        "        }\n",
        "      ],\n",
        '      "must_not": [\n',
        '        { "cue": "<visible evidence contradicting the option>", "why": "<one verifiable sentence>", "certainty": 0.0~1.0 }\n',
        "      ]\n",
        "    },\n",
        "    ...\n",
        "  ]\n",
        "}\n",
    ]
    return "".join(prompt_parts)


def _fallback_enhanced_probes(options: List[str], answer_text: str) -> Dict[str, Any]:
    return {
        "task_types": [],
        "alpha": 0.6,
        "per_option_probes": [
            {
                "option": opt,
                "group": "correct" if opt == answer_text else "incorrect",
                "must_have": [],
                "must_not": [],
            }
            for opt in options
        ],
    }


def _parse_enhanced_response(response: str, options: List[str], answer_text: str) -> Dict[str, Any]:
    if not response or not response.strip():
        return _fallback_enhanced_probes(options, answer_text)
    try:
        content = response.strip()
        match = re.search(r"\{[\s\S]*\}", content)
        json_str = match.group(0) if match else content
        parsed = json.loads(json_str)
        if not isinstance(parsed, dict):
            return _fallback_enhanced_probes(options, answer_text)
    except Exception:
        return _fallback_enhanced_probes(options, answer_text)

    parsed.setdefault("task_types", [])
    try:
        parsed["alpha"] = float(parsed.get("alpha", 0.6))
    except Exception:
        parsed["alpha"] = 0.6
    parsed.setdefault("per_option_probes", [])

    normalized = []
    seen = set()
    for item in parsed.get("per_option_probes", []):
        if not isinstance(item, dict):
            continue
        opt = str(item.get("option", "")).strip()
        if not opt:
            continue
        seen.add(opt)
        mh = item.get("must_have", [])
        mn = item.get("must_not", [])
        mh = mh if isinstance(mh, list) else []
        mn = mn if isinstance(mn, list) else []
        normalized.append(
            {
                "option": opt,
                "group": item.get("group", "correct" if opt == answer_text else "incorrect"),
                "must_have": mh,
                "must_not": mn,
            }
        )
    for opt in options:
        if opt not in seen:
            normalized.append(
                {
                    "option": opt,
                    "group": "correct" if opt == answer_text else "incorrect",
                    "must_have": [],
                    "must_not": [],
                }
            )
    parsed["per_option_probes"] = normalized
    return parsed


def _probe_indices_covered(path: str, n: int) -> bool:
    """Requires JSONL to contain indexes 0..n-1 with per_option_probes."""
    if n <= 0:
        return True
    if not os.path.isfile(path):
        return False
    found: Set[int] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                idx = int(obj["index"])
                pr = obj.get("probes") or {}
                pops = pr.get("per_option_probes") or []
                if not pops:
                    continue
                found.add(idx)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return found >= set(range(n))


def _load_existing_indices(path: str) -> Set[int]:
    done: Set[int] = set()
    if not os.path.isfile(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["index"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return done


def generate_probes_openai_jsonl(
    train_df: "pd.DataFrame",
    output_path: str,
    *,
    api_key: str,
    model: str = "gpt-4o-mini",
    resume: bool = False,
    sleep_s: float = 0.0,
) -> None:
    from tqdm import tqdm

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    existing = _load_existing_indices(output_path) if resume else set()
    mode = "a" if resume and existing and os.path.isfile(output_path) else "w"

    client = _openai_client(api_key)
    n = len(train_df)

    with open(output_path, mode, encoding="utf-8") as fout:
        for idx in tqdm(range(n), desc="OpenAI probes"):
            if idx in existing:
                continue
            row = train_df.iloc[idx]
            question = str(row.get("question", ""))
            od = _option_dict(row)
            option_list = [od[k] for k in ("A", "B", "C", "D")]
            letter = _answer_letter(row)
            correct_text = od[letter]
            prompt_text = _build_enhanced_prompt(question, option_list, correct_text)

            image = _decode_image_from_row(row)
            try:
                resp_text = _openai_chat_once(client, model, prompt_text, image)
                parsed = _parse_enhanced_response(resp_text, option_list, correct_text)
            except Exception as e:
                logger.warning("probe row %s failed: %s", idx, e)
                parsed = _fallback_enhanced_probes(option_list, correct_text)

            rec = {
                "index": idx,
                "question": question,
                "options": od,
                "answer": str(row.get("answer", letter)),
                "probes": parsed,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if sleep_s > 0:
                time.sleep(sleep_s)

    logger.info("Probes JSONL written: %s", output_path)


def ensure_training_probes_ready(
    args: Any,
    train_df: "pd.DataFrame",
    output_dir: str,
    default_filename: str = "probes_train.jsonl",
) -> str:
    """
    Called before loading the large model: probes must fully cover train_df, otherwise fail immediately.
    Returns the probe JSONL path actually used.
    """
    n = len(train_df)
    probe_path = getattr(args, "probe_output_path", None) or os.path.join(output_dir, default_filename)
    if n <= 0:
        logger.warning("train_df is empty; skip probe validation")
        return probe_path

    if _probe_indices_covered(probe_path, n):
        logger.info("probes are ready: %s (n=%d)", probe_path, n)
        return probe_path

    raise ValueError(
        f"Cannot start training: probe file is incomplete or missing ({probe_path})."
        "Strict mode is enabled: auto-generation during training is disabled. Prepare a complete offline JSONL (indexes 0..n-1 with per_option_probes)."
    )
