"""
注意力增强模块
包含自注意力（用于单模态特征增强）和交叉注意力（用于跨模态特征增强）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def _attention_dim(hidden_size: int, num_heads: int) -> int:
    """用于注意力的可整除维度。当 hidden_size 不能被 num_heads 整除时，取最小 >= hidden_size 且能被 num_heads 整除的值（仅注意力内部使用，对外仍保持论文 d_f）。"""
    if hidden_size % num_heads == 0:
        return hidden_size
    return ((hidden_size + num_heads - 1) // num_heads) * num_heads


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        初始化多头注意力
        
        Args:
            hidden_size: 隐藏层大小
            num_heads: 注意力头数
            dropout: Dropout比率
        """
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Q, K, V投影层
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        
        # 输出投影层
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query: Query张量 [batch_size, seq_len_q, hidden_size]
            key: Key张量 [batch_size, seq_len_k, hidden_size]
            value: Value张量 [batch_size, seq_len_v, hidden_size]
            mask: 注意力掩码 [batch_size, seq_len_q, seq_len_k]
        
        Returns:
            output: 注意力输出 [batch_size, seq_len_q, hidden_size]
        """
        batch_size = query.size(0)
        
        # 投影到Q, K, V
        Q = self.query_proj(query)  # [batch_size, seq_len_q, hidden_size]
        K = self.key_proj(key)       # [batch_size, seq_len_k, hidden_size]
        V = self.value_proj(value)   # [batch_size, seq_len_v, hidden_size]
        
        # 重塑为多头形式
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch_size, num_heads, seq_len, head_dim]
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # [batch_size, num_heads, seq_len_q, seq_len_k]
        
        # 应用掩码
        if mask is not None:
            # 扩展掩码维度以匹配多头
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len_k]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)
        # [batch_size, num_heads, seq_len_q, head_dim]
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.hidden_size)
        # [batch_size, seq_len_q, hidden_size]
        
        # 输出投影
        output = self.output_proj(attn_output)
        
        return output


class SelfAttention(nn.Module):
    """自注意力模块（用于单模态特征增强）
    按照论文公式(11)：𝑓̂𝑡 = 𝑀𝐻𝐴(𝑓𝑡, 𝑓𝑡, 𝑓𝑡) = (𝑀𝑡 + 𝑓𝑡) + 𝐹𝑁𝑁(𝑀𝑡 + 𝑓𝑡)
    包含：MHA + 残差 + FFN + 第二次残差（完整Transformer block结构）
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        初始化自注意力模块
        
        Args:
            hidden_size: 隐藏层大小
            num_heads: 注意力头数
            dropout: Dropout比率
        """
        super(SelfAttention, self).__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # 论文公式(11)：FFN由两个全连接（FC）线性层和ReLU激活函数组成
        # Linear(d, 4d) -> ReLU -> Dropout -> Linear(4d, d) -> Dropout
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（论文公式11）
        𝑓̂𝑡 = 𝑀𝐻𝐴(𝑓𝑡, 𝑓𝑡, 𝑓𝑡) = (𝑀𝑡 + 𝑓𝑡) + 𝐹𝑁𝑁(𝑀𝑡 + 𝑓𝑡)
        
        Args:
            x: 输入特征 [batch_size, hidden_size] 或 [batch_size, seq_len, hidden_size]
        
        Returns:
            enhanced_feat: 增强后的特征
        """
        # 如果输入是2D，添加序列维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # 第一步：MHA + 第一次残差连接
        # 𝑀𝑡 = MHA(𝑓𝑡, 𝑓𝑡, 𝑓𝑡)
        attn_output = self.attention(x, x, x)
        # (𝑀𝑡 + 𝑓𝑡)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 第二步：FFN + 第二次残差连接
        # 𝐹𝑁𝑁(𝑀𝑡 + 𝑓𝑡)
        ffn_output = self.ffn(x)
        # (𝑀𝑡 + 𝑓𝑡) + 𝐹𝑁𝑁(𝑀𝑡 + 𝑓𝑡)
        output = self.norm2(x + ffn_output)
        
        # 如果原来是2D，移除序列维度
        if output.size(1) == 1:
            output = output.squeeze(1)
        
        return output


class CrossAttention(nn.Module):
    """交叉注意力模块（用于跨模态特征增强）"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        初始化交叉注意力模块
        
        Args:
            hidden_size: 隐藏层大小
            num_heads: 注意力头数
            dropout: Dropout比率
        """
        super(CrossAttention, self).__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query: Query特征 [batch_size, hidden_size] 或 [batch_size, seq_len_q, hidden_size]
            key_value: Key-Value特征 [batch_size, hidden_size] 或 [batch_size, seq_len_kv, hidden_size]
        
        Returns:
            enhanced_feat: 增强后的特征
        """
        # 如果输入是2D，添加序列维度
        if query.dim() == 2:
            query = query.unsqueeze(1)
        if key_value.dim() == 2:
            key_value = key_value.unsqueeze(1)
        
        # 交叉注意力：query来自一个模态，key和value来自另一个模态
        attn_output = self.attention(query, key_value, key_value)
        
        # 残差连接和层归一化
        output = self.norm(query + self.dropout(attn_output))
        
        # 如果原来是2D，移除序列维度
        if output.size(1) == 1:
            output = output.squeeze(1)
        
        return output


class CoAttentionLayer(nn.Module):
    """单层Co-Attention Transformer Block
    包含：双向交叉注意力 + FFN + 残差连接
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        初始化单层Co-Attention
        
        Args:
            hidden_size: 隐藏层大小
            num_heads: 注意力头数
            dropout: Dropout比率
        """
        super(CoAttentionLayer, self).__init__()
        
        # 双向交叉注意力
        self.text_to_image_attn = CrossAttention(hidden_size, num_heads, dropout)
        self.image_to_text_attn = CrossAttention(hidden_size, num_heads, dropout)
        
        # FFN（前馈网络）
        self.text_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        self.image_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
        # LayerNorm
        self.text_norm = nn.LayerNorm(hidden_size)
        self.image_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        text_feat: torch.Tensor,
        image_feat: torch.Tensor,
        memory_tokens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            text_feat: 文本特征 [batch_size, hidden_size]
            image_feat: 图像特征 [batch_size, hidden_size]
            memory_tokens: 可学习内存tokens [batch_size, memory_length, hidden_size]（可选）
        
        Returns:
            text_enhanced: 增强后的文本特征 [batch_size, hidden_size]
            image_enhanced: 增强后的图像特征 [batch_size, hidden_size]
        """
        # 双向交叉注意力
        # 如果提供memory，将其作为额外的K/V参与注意力计算
        if memory_tokens is not None:
            # 文本作为Query：K,V = concat(image_feat, memory_tokens)
            image_with_memory = torch.cat([
                image_feat.unsqueeze(1),  # [batch_size, 1, hidden_size]
                memory_tokens  # [batch_size, memory_length, hidden_size]
            ], dim=1)  # [batch_size, 1+memory_length, hidden_size]
            
            # 图像作为Query：K,V = concat(text_feat, memory_tokens)
            text_with_memory = torch.cat([
                text_feat.unsqueeze(1),  # [batch_size, 1, hidden_size]
                memory_tokens  # [batch_size, memory_length, hidden_size]
            ], dim=1)  # [batch_size, 1+memory_length, hidden_size]
            
            text_attn = self.text_to_image_attn(text_feat, image_with_memory)
            image_attn = self.image_to_text_attn(image_feat, text_with_memory)
        else:
            # 不使用memory的原始逻辑
            text_attn = self.text_to_image_attn(text_feat, image_feat)
            image_attn = self.image_to_text_attn(image_feat, text_feat)
        
        # FFN + 残差连接
        text_enhanced = self.text_norm(text_attn + self.text_ffn(text_attn))
        image_enhanced = self.image_norm(image_attn + self.image_ffn(image_attn))
        
        return text_enhanced, image_enhanced


class CoAttention(nn.Module):
    """协同注意力（Co-Attention）
    论文中：双层联合注意力/两个Co-Attention Transformer。
    当 hidden_size 不能被 num_heads 整除时，内部使用「仅用于注意力的投影维度」：
    812 -> Linear -> d_attn(816) -> Co-Attention(head=8) -> Linear -> 812，对外仍为论文 d_f。
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 2
    ):
        """
        初始化协同注意力
        
        Args:
            hidden_size: 隐藏层大小（论文 d_f=812）
            num_heads: 注意力头数
            dropout: Dropout比率
            num_layers: Co-Attention层数（论文设置为2）
        """
        super(CoAttention, self).__init__()
        self.hidden_size = hidden_size
        d_attn = _attention_dim(hidden_size, num_heads)
        self._use_proj = d_attn != hidden_size
        if self._use_proj:
            self.proj_in = nn.Linear(hidden_size, d_attn)
            self.proj_out = nn.Linear(d_attn, hidden_size)
        # 堆叠多层Co-Attention Transformer，在 d_attn 维上满足 head 整除
        self.layers = nn.ModuleList([
            CoAttentionLayer(d_attn, num_heads, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        text_feat: torch.Tensor,
        image_feat: torch.Tensor,
        memory_tokens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            text_feat: 文本特征 [batch_size, hidden_size]
            image_feat: 图像特征 [batch_size, hidden_size]
            memory_tokens: 可学习内存tokens [batch_size, memory_length, hidden_size]（可选）
        
        Returns:
            text_enhanced: 图像增强的文本特征 [batch_size, hidden_size]
            image_enhanced: 文本增强的图像特征 [batch_size, hidden_size]
        """
        if self._use_proj:
            text_feat = self.proj_in(text_feat)
            image_feat = self.proj_in(image_feat)
            if memory_tokens is not None:
                memory_tokens = self.proj_in(memory_tokens)
        for layer in self.layers:
            text_feat, image_feat = layer(text_feat, image_feat, memory_tokens)
        if self._use_proj:
            text_feat = self.proj_out(text_feat)
            image_feat = self.proj_out(image_feat)
        return text_feat, image_feat


class FeatureEnhancementModule(nn.Module):
    """特征增强模块
    按照论文：使用多头自注意力（MHSA）处理单模态特征。
    当 hidden_size 不能被 num_heads 整除时，内部使用「仅用于注意力的投影维度」：
    812 -> Linear -> d_attn(816) -> MHSA(head=8) -> Linear -> 812，对外仍为论文 d_f。
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        初始化特征增强模块
        
        Args:
            hidden_size: 隐藏层大小（论文 d_f=812）
            num_heads: 注意力头数（论文中Head=8）
            dropout: Dropout比率
        """
        super(FeatureEnhancementModule, self).__init__()
        self.hidden_size = hidden_size
        d_attn = _attention_dim(hidden_size, num_heads)
        self._use_proj = d_attn != hidden_size
        if self._use_proj:
            self.proj_in = nn.Linear(hidden_size, d_attn)
            self.proj_out = nn.Linear(d_attn, hidden_size)
        # 单模态特征增强（多头自注意力MHSA），在 d_attn 维上满足 head 整除
        self.text_self_attn = SelfAttention(d_attn, num_heads, dropout)
        self.image_self_attn = SelfAttention(d_attn, num_heads, dropout)
    
    def forward(
        self,
        text_feat: torch.Tensor,
        image_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            text_feat: 文本特征 [batch_size, hidden_size]
            image_feat: 图像特征 [batch_size, hidden_size]
        
        Returns:
            enhanced_text: 增强后的文本特征 [batch_size, hidden_size]
            enhanced_image: 增强后的图像特征 [batch_size, hidden_size]
        """
        if self._use_proj:
            text_feat = self.proj_in(text_feat)
            image_feat = self.proj_in(image_feat)
        enhanced_text = self.text_self_attn(text_feat)
        enhanced_image = self.image_self_attn(image_feat)
        if self._use_proj:
            enhanced_text = self.proj_out(enhanced_text)
            enhanced_image = self.proj_out(enhanced_image)
        return enhanced_text, enhanced_image
