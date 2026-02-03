"""
特征提取模块
包含文本编码器（BERT）、图像编码器（ResNet）和CLIP编码器
按照论文：微观特征（BERT/ResNet）+ 宏观特征（CLIP）
"""

import torch
import torch.nn as nn
from transformers import BertModel
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
import torch.nn.functional as F
from typing import Tuple
try:
    import clip
except ImportError:
    print("Warning: CLIP not installed. Please install with: pip install git+https://github.com/openai/CLIP.git")


class TextEncoder(nn.Module):
    """基于BERT的文本编码器（微观特征）
    论文中：微观特征关注细节，使用BERT的last_hidden_state序列
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        hidden_size: int = 768,
        dropout: float = 0.1
    ):
        """
        初始化文本编码器
        
        Args:
            model_name: BERT模型名称
            hidden_size: 隐藏层大小
            dropout: Dropout比率
        """
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(model_name, local_files_only=True)
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        
        # 投影层：将BERT特征投影到统一维度
        self.proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
        
        Returns:
            text_feat: 文本特征 [batch_size, seq_len, hidden_size]
        """
        # 获取BERT输出（论文公式1：序列特征）
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 使用last_hidden_state（序列特征），后续会进行Token级池化
        # [batch_size, seq_len, hidden_size]
        sequence_output = outputs.last_hidden_state
        
        # 投影到统一维度
        text_feat = self.proj(sequence_output)
        text_feat = self.dropout(text_feat)
        
        return text_feat


class ImageEncoder(nn.Module):
    """基于ResNet的图像编码器（微观特征）
    论文中：微观特征关注细节，使用ResNet去掉FC层后的特征（区域序列）
    """
    
    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        hidden_size: int = 512,
        dropout: float = 0.1
    ):
        """
        初始化图像编码器
        
        Args:
            model_name: ResNet模型名称
            pretrained: 是否使用预训练权重
            hidden_size: 输出隐藏层大小
            dropout: Dropout比率
        """
        super(ImageEncoder, self).__init__()
        model_name_lower = (model_name or "resnet50").lower()
        # 加载 ResNet（支持 resnet18 提速、resnet50 与论文一致）
        if model_name_lower == "resnet18":
            if pretrained:
                self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.resnet = resnet18(weights=None)
            resnet_feat_size = 512  # ResNet18 最后一层 conv 输出 512
        else:
            if pretrained:
                self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.resnet = resnet50(weights=None)
            resnet_feat_size = 2048  # ResNet50 最后一层 conv 输出 2048
        # 移除 ResNet 的分类层和全局池化层，保留特征图
        self.resnet_backbone = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # 投影层：将ResNet特征映射到目标隐藏层大小
        # 论文3.1.4节：明确提到"将池化的图像特征..."
        # ResNet输出是区域序列（特征图），需要池化为向量
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化（论文3.1.4节）
            nn.Flatten(),
            nn.Linear(resnet_feat_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            images: 输入图像 [batch_size, 3, H, W]
        
        Returns:
            image_feat: 图像特征 [batch_size, hidden_size]
        """
        # 通过ResNet提取特征图（论文公式2：区域序列）
        # [batch_size, 2048, H, W]
        feature_map = self.resnet_backbone(images)
        
        # 投影到统一维度（论文中会进行区域级池化）
        image_feat = self.proj(feature_map)  # [batch_size, hidden_size]
        
        return image_feat


class CLIPEncoder(nn.Module):
    """CLIP编码器（宏观特征）
    论文中：宏观特征关注全局语义对齐，使用CLIP Text Encoder和Image Encoder
    """
    
    def __init__(self, clip_model_name: str = "ViT-B/16"):
        """
        初始化CLIP编码器
        
        Args:
            clip_model_name: CLIP模型名称
        """
        super(CLIPEncoder, self).__init__()
        self.clip_model_name = clip_model_name
        self.clip_model = None
        self.clip_tokenizer = None
        self.clip_preprocess = None  # CLIP官方图像预处理
        self.clip_dim = 512  # CLIP ViT-B/16的输出维度是512
        
        try:
            import clip as clip_lib
            # 延迟加载，在实际使用时再加载到正确的设备
            self._clip_lib = clip_lib
            self._model_loaded = False
        except ImportError:
            print("Warning: CLIP not installed. Please install with: pip install git+https://github.com/openai/CLIP.git")
            self._clip_lib = None
            self._model_loaded = False
    
    def _load_model(self, device):
        """延迟加载CLIP模型到指定设备"""
        if self._model_loaded or self._clip_lib is None:
            return
        
        try:
            self.clip_model, self.clip_preprocess = self._clip_lib.load(
                self.clip_model_name, device=device
            )
            # 获取tokenizer（clip库的tokenize函数）
            self.clip_tokenizer = self._clip_lib.tokenize
            self.clip_model = self.clip_model.to(device)
            self.clip_model = self.clip_model.float()  # 强制 fp32，避免 MPS 上 fp16/fp32 混算触发 dtype assert
            self.clip_dim = self.clip_model.visual.output_dim
            self._model_loaded = True
        except Exception as e:
            print(f"Warning: CLIP loading failed: {e}")
            self._model_loaded = False
    
    def encode_text(self, text: str, device: torch.device) -> torch.Tensor:
        """
        编码文本（CLIP Text Encoder）
        按照论文4.3.2节：文本被截断或填充为77个token
        
        Args:
            text: 文本字符串（CLIP tokenizer会自动处理为77 tokens）
            device: 设备
        
        Returns:
            text_feat: 文本特征 [batch_size, clip_dim]
        """
        if self._clip_lib is None:
            # 如果CLIP未安装，返回零向量
            batch_size = 1 if isinstance(text, str) else text.size(0)
            return torch.zeros(batch_size, self.clip_dim, device=device)
        
        self._load_model(device)
        
        if self.clip_model is None:
            batch_size = 1 if isinstance(text, str) else text.size(0)
            return torch.zeros(batch_size, self.clip_dim, device=device)
        
        # CLIP tokenizer会自动将文本截断或填充为77个token（论文4.3.2节）
        if isinstance(text, str):
            text_tokens = self.clip_tokenizer(text).to(device)  # [1, 77]
        else:
            # 如果已经是tokens，直接使用
            text_tokens = text.to(device)
        
        with torch.no_grad():
            text_feat = self.clip_model.encode_text(text_tokens)
        return text_feat.float()
    
    def encode_text_batch(self, texts: list, device: torch.device) -> torch.Tensor:
        """
        批量编码文本（CLIP Text Encoder），整批 tokenize + 一次 forward，避免逐条循环。
        
        Args:
            texts: 文本字符串列表 [batch_size]
            device: 设备
        
        Returns:
            text_feat: 文本特征 [batch_size, clip_dim]
        """
        if self._clip_lib is None:
            return torch.zeros(len(texts), self.clip_dim, device=device)
        
        self._load_model(device)
        
        if self.clip_model is None:
            return torch.zeros(len(texts), self.clip_dim, device=device)
        
        # 整批 tokenize（CLIP 截断/填充为 77）
        text_tokens = self.clip_tokenizer(texts, truncate=True).to(device)
        with torch.no_grad():
            text_feat = self.clip_model.encode_text(text_tokens)
        return text_feat.float()
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图像（CLIP Image Encoder - ViT-B/16）
        纯 tensor 批归一化（整批一次做完），避免 PIL 循环打满 CPU、提升 MPS 吞吐。
        论文要求：CLIP 的 mean/std 与官方一致；输入已是 224x224 时仅做 CLIP normalize。
        
        Args:
            images: 输入图像 [batch_size, 3, H, W]（通常已 224x224，可能 ImageNet normalize）
        
        Returns:
            image_feat: 图像特征 [batch_size, clip_dim]
        """
        device = images.device
        
        if self._clip_lib is None:
            return torch.zeros(images.size(0), self.clip_dim, device=device)
        
        self._load_model(device)
        
        if self.clip_model is None:
            return torch.zeros(images.size(0), self.clip_dim, device=device)
        
        # CLIP ViT-B/16 官方归一化参数（与 openai/CLIP 一致）
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device, dtype=images.dtype).view(1, 3, 1, 1)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device, dtype=images.dtype).view(1, 3, 1, 1)
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=images.dtype).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=images.dtype).view(1, 3, 1, 1)
        
        # 若已是 ImageNet 归一化（有负值），先反归一化到 [0,1]
        if images.min() < 0:
            images = images * imagenet_std + imagenet_mean
        images = torch.clamp(images, 0.0, 1.0)
        # 整批做 CLIP normalize（与官方 preprocess 一致）
        images = (images - clip_mean) / clip_std
        
        with torch.no_grad():
            image_feat = self.clip_model.encode_image(images)
        return image_feat.float()


class FeatureFusionModule(nn.Module):
    """特征融合模块
    按照论文公式5和6：将微观特征（BERT/ResNet）和宏观特征（CLIP）进行拼接
    论文3.1.4节：𝑑𝑓 = 𝑑𝑐 + 𝑑𝑙，𝑑𝑙是文本/图像模态通过线性层映射到相同维度
    """
    
    def __init__(
        self,
        text_micro_dim: int,
        image_micro_dim: int,
        macro_dim: int,
        micro_proj_dim: int = 300,  # 论文4.3.2节：d_l=300（词嵌入维度）
        dropout: float = 0.1
    ):
        """
        初始化特征融合模块
        
        Args:
            text_micro_dim: 文本微观特征维度（BERT，通常是768）
            image_micro_dim: 图像微观特征维度（ResNet，通常是512）
            macro_dim: 宏观特征维度（CLIP维度，通常是512）
            micro_proj_dim: 微观特征投影维度d_l（论文4.3.2节：300维词嵌入）
            dropout: Dropout比率
        """
        super(FeatureFusionModule, self).__init__()
        
        # 论文3.1.4节：先将文本/图像模态通过线性层映射到相同维度d_l
        self.text_micro_proj = nn.Linear(text_micro_dim, micro_proj_dim)
        self.image_micro_proj = nn.Linear(image_micro_dim, micro_proj_dim)
        
        # 论文公式5和6：𝑓𝑡 = concat(𝑓𝑡𝑢, 𝑓𝑡𝑐)，𝑓𝑣 = concat(𝑓𝑣𝑢, 𝑓𝑣𝑐)
        # 融合后维度：d_f = d_l + d_c = micro_proj_dim + macro_dim
        self.fusion_dim = micro_proj_dim + macro_dim
        
        # Dropout（论文未明确，但通常需要）
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        text_micro: torch.Tensor,
        text_macro: torch.Tensor,
        image_micro: torch.Tensor,
        image_macro: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播（论文公式5和6）
        论文3.1.4节：先将micro映射到d_l，再与CLIP(d_c)拼接得到d_f=d_l+d_c
        
        Args:
            text_micro: 文本微观特征 [batch_size, seq_len, text_micro_dim] 或 [batch_size, text_micro_dim]
            text_macro: 文本宏观特征 [batch_size, macro_dim]
            image_micro: 图像微观特征 [batch_size, image_micro_dim]
            image_macro: 图像宏观特征 [batch_size, macro_dim]
        
        Returns:
            text_fused: 融合后的文本特征 [batch_size, seq_len, fusion_dim] 或 [batch_size, fusion_dim]
            image_fused: 融合后的图像特征 [batch_size, fusion_dim]
        """
        # 论文3.1.4节：先将文本模态特征池化成一个特征向量（如果是序列）
        if text_micro.dim() == 3:
            # [batch_size, seq_len, text_micro_dim] -> [batch_size, text_micro_dim]
            # 使用平均池化（论文3.1.4节：Token级池化）
            text_micro_pooled = text_micro.mean(dim=1)
        else:
            text_micro_pooled = text_micro
        
        # 论文3.1.4节：通过线性层将文本/图像模态映射到相同维度d_l
        text_micro_proj = self.text_micro_proj(text_micro_pooled)  # [batch_size, micro_proj_dim]
        image_micro_proj = self.image_micro_proj(image_micro)  # [batch_size, micro_proj_dim]
        
        # 论文公式5：𝑓𝑡 = concat(𝑓𝑡𝑢, 𝑓𝑡𝑐)，维度d_f = d_l + d_c
        text_fused = torch.cat([text_micro_proj, text_macro], dim=1)  # [batch_size, micro_proj_dim + macro_dim]
        
        # 论文公式6：𝑓𝑣 = concat(𝑓𝑣𝑢, 𝑓𝑣𝑐)，维度d_f = d_l + d_c
        image_fused = torch.cat([image_micro_proj, image_macro], dim=1)  # [batch_size, micro_proj_dim + macro_dim]
        
        text_fused = self.dropout(text_fused)
        image_fused = self.dropout(image_fused)
        
        return text_fused, image_fused
