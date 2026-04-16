#!/usr/bin/env python3
"""Common utilities for probe generation and verification.

Provides image/text helpers and parsing/score-extraction shared by probe_generator
and qwen3_evidence_verifier to avoid duplication.
"""

import io
import os
import re
import json
from typing import List, Dict, Optional, Any
from PIL import Image, ImageFilter, ImageEnhance
import base64
import numpy as np


def image_to_data_url(image: Any) -> Optional[str]:
    if image is None:
        return None
    try:
        if isinstance(image, str) and os.path.exists(image):
            im = Image.open(image).convert('RGB')
        elif hasattr(image, 'convert'):
            im = image.convert('RGB')
        else:
            return None
        buf = io.BytesIO()
        im.save(buf, format='JPEG', quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def apply_image_perturbations(image: Image.Image, num_perturbations: int = 5) -> List[Image.Image]:
    out = [image]
    try:
        w, h = image.width, image.height
        crop = image.crop((w // 8, h // 8, w * 7 // 8, h * 7 // 8)).resize((w, h))
        out += [
            crop,
            image.filter(ImageFilter.GaussianBlur(radius=2)),
            ImageEnhance.Brightness(image).enhance(1.3),
            ImageEnhance.Contrast(image).enhance(1.2),
            image.rotate(5, expand=False),
        ]
    except Exception:
        pass
    return out[:num_perturbations]


def generate_text_paraphrases(probe_text: str, num_paraphrases: int = 3) -> List[str]:
    paraphrases = [probe_text]
    synonym_map = {
        'must': ['should', 'need to', 'required to'],
        'have': ['contain', 'include', 'possess'],
        'not': ['cannot', 'should not', 'avoid'],
        'visible': ['apparent', 'observable', 'clear'],
        'present': ['shown', 'displayed', 'exhibited'],
        'color': ['hue', 'shade', 'tint'],
        'shape': ['form', 'structure', 'contour'],
        'size': ['dimension', 'scale', 'magnitude']
    }
    try:
        for i in range(num_paraphrases - 1):
            s = probe_text.lower()
            for orig, syns in synonym_map.items():
                if orig in s:
                    s = s.replace(orig, syns[i % len(syns)])
            paraphrases.append(s)
    except Exception:
        pass
    return paraphrases[:num_paraphrases]


def extract_items_from_line(line: str) -> List[str]:
    body = line.split(':', 1)[1] if ':' in line else line
    items = re.findall(r'\[([^\]]+)\]', body) or [x.strip() for x in body.split(',') if x.strip()]
    return [i.strip() for i in items]


def parse_llm_probes(response: str, options: Optional[List[str]] = None) -> Dict[str, Dict[str, List[str]]]:
    if not response:
        return {}
    try:
        jm = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if jm:
            data = json.loads(jm.group(0))
            parsed = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    parsed[str(k)] = {
                        'must_have': v.get('must_have', []),
                        'must_not': v.get('must_not', []),
                    }
            return _align_options(parsed, options)
    except Exception:
        pass

    parsed = {}
    pat = (
        r'([^\n:]+?)(?:\s*\([^)]+\))?:\s*\n[^\n]*must_have:\s*(\[[^\]]*\]|[^\n]*)\s*\n'
        r'[^\n]*must_not:\s*(\[[^\]]*\]|[^\n]*)'
    )
    for name, mh, mn in re.findall(pat, response, flags=re.IGNORECASE):
        mh_items = re.findall(r'\[([^\]]+)\]', mh) or [x.strip() for x in mh.split(',') if x.strip()]
        mn_items = re.findall(r'\[([^\]]+)\]', mn) or [x.strip() for x in mn.split(',') if x.strip()]
        parsed[name.strip()] = {'must_have': [s.strip() for s in mh_items], 'must_not': [s.strip() for s in mn_items]}

    if not parsed:
        current = None
        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue
            if ':' in line and not line.lower().startswith(('must_have', 'must_not')):
                current = line.split(':')[0].strip()
                current = re.sub(r'\s*\([^)]+\)', '', current).strip()
                if current:
                    parsed.setdefault(current, {'must_have': [], 'must_not': []})
                continue
            if current and 'must_have' in line.lower():
                parsed[current]['must_have'] = extract_items_from_line(line)
            if current and 'must_not' in line.lower():
                parsed[current]['must_not'] = extract_items_from_line(line)

    return _align_options(parsed, options)


def extract_score_from_response(response: str) -> float:
    try:
        import re
        numbers = re.findall(r'0\.\d+|1\.0|0|1', response)
        if numbers:
            scores = [float(num) for num in numbers]
            score = scores[-1]
            return max(0.0, min(1.0, score))
        response_lower = response.lower()
        if any(word in response_lower for word in ['非常明显', '清晰可见', '很明显']):
            return 0.9
        if any(word in response_lower for word in ['比较明显', '可见']):
            return 0.7
        if any(word in response_lower for word in ['模糊', '不太明显']):
            return 0.4
        if any(word in response_lower for word in ['没有', '不包含', '无']):
            return 0.1
        return 0.5
    except Exception:
        return 0.5


def _align_options(parsed: Dict[str, Dict[str, List[str]]], options: Optional[List[str]]):
    if options is None:
        return parsed
    aligned = {}
    for opt in options:
        aligned[opt] = parsed.get(opt, {'must_have': [], 'must_not': []})
    return aligned
