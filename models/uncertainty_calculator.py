#!/usr/bin/env python3
"""
Uncertainty Calculator
职责：根据consensus_score和conflict计算不确定性
"""

import logging
from typing import Dict
import numpy as np

logger = logging.getLogger(__name__)


class UncertaintyCalculator:
    """计算探针的不确定性（用于路由和训练）"""
    
    def __init__(self):
        logger.info(" UncertaintyCalculator 初始化完成")
    
    def compute_uncertainties(self,
                             consensus_scores: Dict[str, float],
                             conflict_scores: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        计算所有选项的不确定性
        
        Args:
            consensus_scores: {option: consensus_score}
            conflict_scores: {
                option: {
                    "visual_conflict": float,
                    "text_conflict": float
                }
            }
        
        Returns:
            {
                option: {
                    "unc_vis": float,    # 用于路由的视觉不确定性
                    "unc_text": float    # 用于训练的文本不确定性
                }
            }
        """
        uncertainties = {}
        
        for option in consensus_scores.keys():
            consensus = consensus_scores.get(option, 0.0)
            visual_conflict = conflict_scores.get(option, {}).get("visual_conflict", 0.0)
            text_conflict = conflict_scores.get(option, {}).get("text_conflict", 0.0)
            
            unc_vis = visual_conflict / (1.0 + consensus + 1e-8)
            unc_text = text_conflict / (1.0 + consensus + 1e-8)
            
            unc_vis = float(np.clip(unc_vis, 0.0, 10.0))  # 允许较大的unc_vis用于路由
            unc_text = float(np.clip(unc_text, 0.0, 1.0))  # unc_text用于训练，限制在[0,1]
            
            uncertainties[option] = {
                "unc_vis": unc_vis,
                "unc_text": unc_text
            }
        
        logger.info(f" 不确定性计算完成: {len(uncertainties)} 个选项")
        return uncertainties

