"""
模态间歧义性分析模块
按照论文3.3节：使用VAE学习特征分布，通过KL散度计算歧义性
这是论文的核心创新点
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class VAEEncoder(nn.Module):
    """VAE编码器
    将特征编码为高斯分布的均值和方差
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 256):
        """
        初始化VAE编码器
        
        Args:
            input_dim: 输入特征维度
            latent_dim: 潜在空间维度
        """
        super(VAEEncoder, self).__init__()
        
        # 编码器：输入 -> 均值和方差
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, input_dim]
        
        Returns:
            mu: 均值 [batch_size, latent_dim]
            logvar: 对数方差 [batch_size, latent_dim]
        """
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class AmbiguityAnalyzer(nn.Module):
    """模态间歧义性分析器
    按照论文公式：使用VAE学习分布，计算双向KL散度，得到歧义性分数
    """
    
    def __init__(self, hidden_size: int = 512, latent_dim: int = 256):
        """
        初始化歧义性分析器
        
        Args:
            hidden_size: 特征维度
            latent_dim: VAE潜在空间维度
        """
        super(AmbiguityAnalyzer, self).__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        
        # 文本特征VAE编码器
        self.text_vae_encoder = VAEEncoder(hidden_size, latent_dim)
        
        # 图像特征VAE编码器
        self.image_vae_encoder = VAEEncoder(hidden_size, latent_dim)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧（Reparameterization Trick）
        
        Args:
            mu: 均值 [batch_size, latent_dim]
            logvar: 对数方差 [batch_size, latent_dim]
        
        Returns:
            z: 采样得到的潜在向量 [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def kl_divergence(
        self,
        mu1: torch.Tensor,
        logvar1: torch.Tensor,
        mu2: torch.Tensor,
        logvar2: torch.Tensor
    ) -> torch.Tensor:
        """
        计算两个高斯分布之间的KL散度
        KL(N(μ1,σ1) || N(μ2,σ2))
        
        Args:
            mu1: 分布1的均值 [batch_size, latent_dim]
            logvar1: 分布1的对数方差 [batch_size, latent_dim]
            mu2: 分布2的均值 [batch_size, latent_dim]
            logvar2: 分布2的对数方差 [batch_size, latent_dim]
        
        Returns:
            kl: KL散度 [batch_size, 1]
        """
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)
        
        # KL散度公式
        kl = 0.5 * (
            logvar2 - logvar1
            + (var1 + (mu1 - mu2) ** 2) / var2
            - 1
        )
        
        # 对潜在维度求和，得到标量
        kl = kl.sum(dim=1, keepdim=True)
        
        return kl
    
    def compute_ambiguity(
        self,
        text_feat: torch.Tensor,
        image_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        计算模态间歧义性（论文公式19）
        
        步骤：
        1. 将特征输入VAE编码器，得到分布参数
        2. 计算双向KL散度：KL(text||image) 和 KL(image||text)
        3. 通过Sigmoid得到歧义性分数
        
        Args:
            text_feat: 文本特征 [batch_size, hidden_size]
            image_feat: 图像特征 [batch_size, hidden_size]
        
        Returns:
            ambiguity: 歧义性分数 [batch_size, 1]，范围[0, 1]
        """
        # 1. 通过VAE编码器得到分布参数
        text_mu, text_logvar = self.text_vae_encoder(text_feat)
        image_mu, image_logvar = self.image_vae_encoder(image_feat)
        
        # 2. 计算双向KL散度（论文公式）
        kl_text_to_image = self.kl_divergence(
            text_mu, text_logvar, image_mu, image_logvar
        )
        kl_image_to_text = self.kl_divergence(
            image_mu, image_logvar, text_mu, text_logvar
        )
        
        # 3. 双向KL散度的平均（论文公式）
        kl_avg = (kl_text_to_image + kl_image_to_text) / 2.0
        
        # 4. 通过Sigmoid得到歧义性分数（论文公式19）
        # 论文公式19：直接使用 sigmoid(kl_avg)，没有除以维度
        # 严格按论文公式实现
        ambiguity = torch.sigmoid(kl_avg)
        
        return ambiguity
    
    def forward(
        self,
        text_feat: torch.Tensor,
        image_feat: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            text_feat: 文本特征 [batch_size, hidden_size]
            image_feat: 图像特征 [batch_size, hidden_size]
        
        Returns:
            results: 包含歧义性分数的字典
        """
        # 计算歧义性
        ambiguity = self.compute_ambiguity(text_feat, image_feat)
        
        # 根据歧义性生成权重（论文公式20）
        # 当歧义性大(α接近1)时，加大跨模态特征权重
        # 当歧义性小(α接近0)时，加大单模态特征权重
        cross_modal_weight = ambiguity  # α
        single_modal_weight = 1.0 - ambiguity  # 1 - α
        
        return {
            'ambiguity': ambiguity,
            'cross_modal_weight': cross_modal_weight,
            'single_modal_weight': single_modal_weight
        }
