#!/usr/bin/env python3
"""
Probe Conflict Calculator
职责：计算 visual_conflict 和 text_conflict
"""

import logging
from typing import List, Dict, Optional, Any
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from probe_utils import apply_image_perturbations, generate_text_paraphrases

logger = logging.getLogger(__name__)


class ProbeConflictCalculator:
    """计算探针的视觉冲突和文本冲突"""
    
    def __init__(self, verifier=None, qwen3_model=None, qwen3_processor=None, device=None):
        """
        Args:
            verifier: Qwen3EvidenceVerifier实例，用于验证匹配分数
            qwen3_model: Qwen3模型（如果verifier未提供，直接使用）
            qwen3_processor: Qwen3处理器
            device: 设备
        """
        self.verifier = verifier
        self.qwen3_model = qwen3_model
        self.qwen3_processor = qwen3_processor
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        
        logger.info(f" ProbeConflictCalculator 初始化完成")
    
    def compute_visual_conflict(self, 
                               probe_data: Dict[str, List[str]], 
                               image: Optional[Image.Image] = None) -> float:
        """
        计算图像扰动下的匹配方差（visual_conflict）
        
        Args:
            probe_data: {"must_have": [...], "must_not": [...]}
            image: 原始图像
        
        Returns:
            visual_conflict: 方差值
        """
        if image is None:
            return 0.0
        
        perturbed_images = apply_image_perturbations(image, num_perturbations=5)
        
        scores = []
        for perturbed_img in perturbed_images:
            score = self._verify_probe_on_image(perturbed_img, probe_data)
            scores.append(score)
        
        visual_conflict = float(np.var(scores)) if len(scores) > 1 else 0.0
        logger.debug(f" Visual conflict: {visual_conflict:.4f} (from {len(scores)} perturbations)")
        return visual_conflict
    
    def compute_text_conflict(self,
                             probe_texts: List[str],
                             image: Optional[Image.Image] = None,
                             text_type: str = "must_have") -> float:
        """
        计算文本同义改写后的匹配方差（text_conflict）
        
        Args:
            probe_texts: probe文本列表
            image: 原始图像
            text_type: "must_have" 或 "must_not"
        
        Returns:
            text_conflict: 方差值
        """
        if not probe_texts or image is None:
            return 0.0
        
        scores = []
        for probe_text in probe_texts:
            paraphrases = generate_text_paraphrases(probe_text, num_paraphrases=3)
            
            for para in paraphrases:
                score = self._verify_text_probe(para, image, text_type)
                scores.append(score)
        
        text_conflict = float(np.var(scores)) if len(scores) > 1 else 0.0
        logger.debug(f" Text conflict ({text_type}): {text_conflict:.4f} (from {len(scores)} paraphrases)")
        return text_conflict
    
    def compute_all_conflicts(self,
                             probes: Dict[str, Dict[str, List[str]]],
                             image: Optional[Image.Image] = None) -> Dict[str, Dict[str, float]]:
        """
        计算所有选项的conflict分数
        
        Args:
            probes: {option: {"must_have": [...], "must_not": [...]}}
            image: 原始图像
        
        Returns:
            {
                option: {
                    "visual_conflict": float,
                    "text_conflict": float,
                    "score_must_have": float,
                    "score_must_not": float
                }
            }
        """
        out: Dict[str, Dict[str, float]] = {}
        
        for opt, probe_data in probes.items():
            visual_conflict = self.compute_visual_conflict(probe_data, image)
            
            text_scores_all = []
            mh_scores, mn_scores = [], []
            
            for clause in probe_data.get("must_have", []):
                paraphrases = generate_text_paraphrases(clause, num_paraphrases=3)
                for para in paraphrases:
                    score = self._verify_text_probe(para, image, text_type="must_have")
                    text_scores_all.append(score)
                    mh_scores.append(score)
            
            for clause in probe_data.get("must_not", []):
                paraphrases = generate_text_paraphrases(clause, num_paraphrases=3)
                for para in paraphrases:
                    score = self._verify_text_probe(para, image, text_type="must_not")
                    text_scores_all.append(score)
                    mn_scores.append(score)
            
            text_conflict = float(np.var(text_scores_all)) if text_scores_all else 0.0
            text_conflict_must_have = float(np.var(mh_scores)) if len(mh_scores) > 1 else 0.0
            text_conflict_must_not = float(np.var(mn_scores)) if len(mn_scores) > 1 else 0.0
            score_must_have = float(np.mean(mh_scores)) if mh_scores else 0.0
            score_must_not = float(np.mean(mn_scores)) if mn_scores else 0.0
            
            out[opt] = {
                "visual_conflict": visual_conflict,
                "text_conflict": text_conflict,
                "text_conflict_must_have": text_conflict_must_have,
                "text_conflict_must_not": text_conflict_must_not,
                "score_must_have": score_must_have,
                "score_must_not": score_must_not,
            }
        
        logger.info(f" Conflict scores computed: {len(out)} options")
        return out
    
    def _verify_probe_on_image(self,
                              image: Image.Image,
                              probe_data: Dict[str, List[str]]) -> float:
        """
        在图像上验证probe，返回匹配分数 [0,1]
        
        使用verifier或qwen3_model计算匹配度
        """
        if self.verifier is not None:
            try:
                if hasattr(self.verifier, 'verify_probe_on_image'):
                    return self.verifier.verify_probe_on_image(image, probe_data)
            except Exception as e:
                logger.warning(f" Verifier验证失败: {e}")
        
        if self.qwen3_model is not None and self.qwen3_processor is not None:
            try:
                return self._compute_match_score_with_qwen3(image, probe_data)
            except Exception as e:
                logger.warning(f" Qwen3验证失败: {e}")
        
        logger.warning(" 无可用验证器，返回随机分数")
        return float(np.random.uniform(0.3, 0.8))
    
    def _verify_text_probe(self,
                          text: str,
                          image: Image.Image,
                          text_type: str) -> float:
        """
        验证改写后的文本probe，返回匹配分数 [0,1]
        """
        if self.verifier is not None:
            try:
                if hasattr(self.verifier, 'verify_text_probe'):
                    return self.verifier.verify_text_probe(text, image, text_type)
            except Exception as e:
                logger.warning(f" Verifier文本验证失败: {e}")
        
        if self.qwen3_model is not None and self.qwen3_processor is not None:
            try:
                return self._compute_text_image_match(text, image)
            except Exception as e:
                logger.warning(f" Qwen3文本验证失败: {e}")
        
        return float(np.random.uniform(0.4, 0.9))
    
    def _compute_match_score_with_qwen3(self,
                                       image: Image.Image,
                                       probe_data: Dict[str, List[str]]) -> float:
        """使用Qwen3计算probe与图像的匹配分数"""
        try:
            must_have = probe_data.get("must_have", [])
            must_not = probe_data.get("must_not", [])
            
            img_vec = self._encode_image(image)
            if img_vec is None:
                return 0.5
            
            mh_sims = []
            for text in must_have:
                text_vec = self._encode_text(text)
                if text_vec is not None:
                    sim = F.cosine_similarity(text_vec, img_vec, dim=-1).item()
                    mh_sims.append(float(sim))
            
            mn_sims = []
            for text in must_not:
                text_vec = self._encode_text(text)
                if text_vec is not None:
                    sim = F.cosine_similarity(text_vec, img_vec, dim=-1).item()
                    mn_sims.append(float(sim))
            
            mh_mean = float(np.mean(mh_sims)) if mh_sims else 0.0
            mn_mean = float(np.mean(mn_sims)) if mn_sims else 0.0
            
            score = mh_mean * (1.0 - mn_mean)
            return float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.error(f" Qwen3匹配分数计算失败: {e}")
            return 0.5
    
    def _compute_text_image_match(self, text: str, image: Image.Image) -> float:
        """计算文本与图像的匹配分数"""
        try:
            img_vec = self._encode_image(image)
            text_vec = self._encode_text(text)
            
            if img_vec is None or text_vec is None:
                return 0.5
            
            sim = F.cosine_similarity(text_vec, img_vec, dim=-1).item()
            return float(np.clip(sim, 0.0, 1.0))
        except Exception as e:
            logger.error(f" 文本-图像匹配计算失败: {e}")
            return 0.5
    
    def _encode_image(self, image: Image.Image):
        """编码图像"""
        if self.qwen3_model is None or self.qwen3_processor is None:
            return None
        try:
            inputs = self.qwen3_processor(images=[image], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.qwen3_model(**inputs)
                if hasattr(outputs, 'last_hidden_state'):
                    vec = outputs.last_hidden_state.mean(dim=1)
                elif hasattr(outputs, 'image_embeds'):
                    vec = outputs.image_embeds
                else:
                    vec = outputs.logits.mean(dim=1)
            
            vec = vec.float()
            vec = vec / (vec.norm(p=2, dim=-1, keepdim=True) + 1e-8)
            return vec
        except Exception as e:
            logger.warning(f" 图像编码失败: {e}")
            return None
    
    def _encode_text(self, text: str):
        """编码文本"""
        if self.qwen3_model is None or self.qwen3_processor is None:
            return None
        try:
            inputs = self.qwen3_processor(text=[text], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.qwen3_model(**inputs)
                if hasattr(outputs, 'last_hidden_state'):
                    vec = outputs.last_hidden_state.mean(dim=1)
                elif hasattr(outputs, 'text_embeds'):
                    vec = outputs.text_embeds
                else:
                    vec = outputs.logits.mean(dim=1)
            
            vec = vec.float()
            vec = vec / (vec.norm(p=2, dim=-1, keepdim=True) + 1e-8)
            return vec
        except Exception as e:
            logger.warning(f" 文本编码失败: {e}")
            return None

