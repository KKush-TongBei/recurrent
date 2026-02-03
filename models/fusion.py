"""
自适应跨模态特征融合模块
按照论文公式20和21：根据歧义性分数自适应加权并拼接
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class AdaptiveFusionModule(nn.Module):
    """自适应特征融合模块
    按照论文公式20和21：
    - 公式20：加权单模态和跨模态特征
    - 公式21：拼接所有加权特征
    """
    
    def __init__(
        self,
        hidden_size: int = 812,  # 论文d_f=812（默认值，实际从config传入）
        dropout: float = 0.1
    ):
        """
        初始化自适应融合模块
        
        Args:
            hidden_size: 特征维度（论文d_f=812）
            dropout: Dropout比率
        """
        super(AdaptiveFusionModule, self).__init__()
        self.hidden_size = hidden_size
        
        # 论文算法1第9-10步：直接拼接F_t, F_v, F_{tv}, F_{vt}，不做额外压缩
        # 注意：去掉final_fusion层，严格按照论文流程
    
    def forward(
        self,
        text_single_feat: torch.Tensor,
        image_single_feat: torch.Tensor,
        text_cross_feat: torch.Tensor,
        image_cross_feat: torch.Tensor,
        ambiguity_results: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        前向传播（论文公式20和21）
        
        Args:
            text_single_feat: 文本单模态融合特征f_t（论文公式20中的f_{tuni}） [batch_size, hidden_size]
            image_single_feat: 图像单模态融合特征f_v（论文公式20中的f_{vuni}） [batch_size, hidden_size]
            text_cross_feat: 文本跨模态特征f_{tv}（图像增强的文本） [batch_size, hidden_size]
            image_cross_feat: 图像跨模态特征f_{vt}（文本增强的图像） [batch_size, hidden_size]
            ambiguity_results: 歧义性分析结果字典，包含权重
        
        Returns:
            final_feat: 最终融合特征 [batch_size, hidden_size * 4]
        """
        # 获取权重（论文公式20）
        alpha = ambiguity_results['ambiguity']  # 歧义性分数
        single_modal_weight = ambiguity_results['single_modal_weight']  # 1 - α
        cross_modal_weight = ambiguity_results['cross_modal_weight']  # α
        
        # 论文公式20：加权特征
        # 当歧义性大(α接近1)时，加大跨模态特征权重
        # 当歧义性小(α接近0)时，加大单模态特征权重
        weighted_text_single = text_single_feat * single_modal_weight  # (1-α) * f_t^s
        weighted_image_single = image_single_feat * single_modal_weight  # (1-α) * f_i^s
        weighted_text_cross = text_cross_feat * cross_modal_weight  # α * f_t^c
        weighted_image_cross = image_cross_feat * cross_modal_weight  # α * f_i^c
        
        # 论文算法1第9-10步：拼接所有加权特征，直接返回（不做额外压缩）
        # 论文公式21：拼接F_t, F_v, F_{tv}, F_{vt}
        concatenated = torch.cat([
            weighted_text_single,
            weighted_image_single,
            weighted_text_cross,
            weighted_image_cross
        ], dim=1)  # [batch_size, hidden_size * 4]
        
        # 注意：论文算法1要求直接拼接，不做final_fusion压缩
        # 分类器将直接接收4*d_f维度的特征（4 * 812 = 3248）
        
        return concatenated
