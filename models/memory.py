"""
可学习内存模块
按照论文：可学习内存信息的长度设置为50
用于跨模态融合阶段，作为额外的K/V参与注意力计算
"""

import torch
import torch.nn as nn
from typing import Optional


class LearnableMemory(nn.Module):
    """可学习内存模块
    论文明确提到"可学习内存信息的长度设置为50"
    在跨模态融合阶段，将memory作为额外的K/V参与注意力计算
    """
    
    def __init__(
        self,
        memory_length: int = 50,
        hidden_size: int = 812  # 论文d_f=812（默认值，实际从config传入）
    ):
        """
        初始化可学习内存模块
        
        Args:
            memory_length: 内存长度（论文设置为50）
            hidden_size: 特征维度（论文d_f=812）
        """
        super(LearnableMemory, self).__init__()
        self.memory_length = memory_length
        self.hidden_size = hidden_size
        
        # 可学习内存参数矩阵 [memory_length, hidden_size]
        self.memory = nn.Parameter(
            torch.randn(memory_length, hidden_size) * 0.02
        )
    
    def forward(
        self,
        batch_size: int
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            batch_size: 批次大小
        
        Returns:
            memory_tokens: 内存tokens [batch_size, memory_length, hidden_size]
        """
        # 扩展内存到batch维度
        # [memory_length, hidden_size] -> [batch_size, memory_length, hidden_size]
        memory_tokens = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        
        return memory_tokens
    
    def get_memory(self) -> torch.Tensor:
        """
        获取内存矩阵（用于作为K/V）
        
        Returns:
            memory: 内存矩阵 [memory_length, hidden_size]
        """
        return self.memory
