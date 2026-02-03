"""
CMFFA主模型
跨模态特征融合与对齐的虚假信息检测模型
按照论文算法1的完整流程实现
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa

from .feature_extractors import TextEncoder, ImageEncoder, CLIPEncoder, FeatureFusionModule
from .attention import FeatureEnhancementModule, CoAttention
from .ambiguity_analysis import AmbiguityAnalyzer
from .fusion import AdaptiveFusionModule
from .memory import LearnableMemory


class CMFFA(nn.Module):
    """跨模态特征融合与对齐的虚假信息检测模型
    按照论文算法1的流程：
    1. 特征提取（微观BERT/ResNet + 宏观CLIP）
    2. 初级融合（Concat + 投影）
    3. 特征增强（MHSA）
    4. 双流处理（Co-Attention + VAE/KL）
    5. 加权与拼接
    6. 分类
    """
    
    def __init__(
        self,
        text_encoder_config: Dict,
        image_encoder_config: Dict,
        clip_config: Dict,
        fusion_config: Dict,
        attention_config: Dict,
        classifier_config: Dict
    ):
        """
        初始化CMFFA模型
        
        Args:
            text_encoder_config: 文本编码器配置（BERT）
            image_encoder_config: 图像编码器配置（ResNet）
            clip_config: CLIP编码器配置
            fusion_config: 特征融合配置
            attention_config: 注意力模块配置
            classifier_config: 分类器配置
        """
        super(CMFFA, self).__init__()
        
        # 1. 特征提取器（微观特征）
        self.text_encoder = TextEncoder(
            model_name=text_encoder_config.get('model_name', 'bert-base-uncased'),
            hidden_size=text_encoder_config.get('hidden_size', 768),
            dropout=text_encoder_config.get('dropout', 0.1)
        )
        
        self.image_encoder = ImageEncoder(
            model_name=image_encoder_config.get('model_name', 'resnet50'),
            pretrained=image_encoder_config.get('pretrained', True),
            hidden_size=image_encoder_config.get('hidden_size', 512),
            dropout=image_encoder_config.get('dropout', 0.1)
        )
        
        # CLIP编码器（宏观特征）
        self.clip_encoder = CLIPEncoder(
            clip_model_name=clip_config.get('model_name', 'ViT-B/16')
        )
        clip_dim = self.clip_encoder.clip_dim  # 通常是512
        
        # 2. 特征融合模块（微观+宏观）
        text_micro_dim = text_encoder_config.get('hidden_size', 768)  # BERT: 768
        image_micro_dim = image_encoder_config.get('hidden_size', 512)  # ResNet: 512
        micro_proj_dim = fusion_config.get('micro_proj_dim', 300)  # 论文4.3.2节：d_l=300（词嵌入维度）
        
        # 论文3.1.4节：d_f = d_l + d_c = micro_proj_dim + clip_dim
        fusion_dim = micro_proj_dim + clip_dim  # 300 + 512 = 812
        
        self.feature_fusion = FeatureFusionModule(
            text_micro_dim=text_micro_dim,
            image_micro_dim=image_micro_dim,
            macro_dim=clip_dim,
            micro_proj_dim=micro_proj_dim,  # 论文4.3.2节：300维
            dropout=fusion_config.get('dropout', 0.1)
        )
        
        # 3. 特征增强模块（MHSA）
        # 论文3.2.1节：输入是融合特征f_t, f_v，维度为d_f=812
        # 论文公式(7-11)：MHSA在d_f维上操作，输出\hat f_t, \hat f_v维度不变（仍为d_f）
        self.feature_enhancement = FeatureEnhancementModule(
            hidden_size=fusion_dim,  # 论文d_f=812，不使用对齐层
            num_heads=attention_config.get('num_heads', 8),
            dropout=attention_config.get('dropout', 0.1)
        )
        
        # 可学习内存模块（论文：可学习内存信息的长度设置为50）
        memory_length = fusion_config.get('memory_length', 50)
        self.memory = LearnableMemory(
            memory_length=memory_length,
            hidden_size=fusion_dim  # 论文d_f=812
        )
        
        # 4. 双流处理
        # 分支A：协同注意力（Co-Attention）
        # 论文强调：双层联合注意力/两个Co-Attention Transformer
        # 论文公式(12-14)：输入是增强后的特征\hat f_t, \hat f_v，维度为d_f=812
        self.co_attention = CoAttention(
            hidden_size=fusion_dim,  # 论文d_f=812
            num_heads=attention_config.get('num_heads', 8),
            dropout=attention_config.get('dropout', 0.1),
            num_layers=2  # 论文明确：两个Co-Attention Transformer
        )
        
        # 分支B：歧义性分析（VAE + KL散度）
        # 论文公式(15-19)：输入维度应为d_f（如果按文字描述用增强特征）或d_c（如果按公式17-18用CLIP特征）
        # 本实现按公式(17)(18)使用CLIP特征（d_c=512），但VAE编码器维度设为d_f以保持一致性
        self.ambiguity_analyzer = AmbiguityAnalyzer(
            hidden_size=clip_dim,  # 论文公式(17)(18)使用CLIP特征，维度d_c=512
            latent_dim=fusion_config.get('latent_dim', 256)
        )
        
        # 5. 自适应融合模块
        # 论文公式(20-21)：输入是融合特征和跨模态特征，维度为d_f=812
        self.fusion_module = AdaptiveFusionModule(
            hidden_size=fusion_dim,  # 论文d_f=812
            dropout=fusion_config.get('dropout', 0.1)
        )
        
        # 6. 分类器（论文3.4节）
        # 论文3.4节："除最后一层外，由五个具有ReLU激活函数的全连接层组成"
        # 这意味着：前5层FC（带ReLU）+ 最后1层FC（无激活）= 总共6层FC
        # 论文公式(22)：𝑦̂ = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝐹𝐶𝑠(𝐹))，输出2 logits用于softmax
        # 输入维度：4 * d_f = 4 * 812 = 3248（论文公式21：拼接F_t, F_v, F_{tv}, F_{vt}）
        input_dim = fusion_dim * 4  # 4个特征拼接后的维度：4 * 812 = 3248
        dropout = classifier_config.get('dropout', 0.3)
        num_classes = classifier_config.get('num_classes', 2)
        
        # 6层FC：前5层带ReLU，最后1层无激活（论文3.4节）
        # 第一层输入：3248维（论文d_f * 4）
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, fusion_dim),  # 第1层：4*d_f -> d_f (3248 -> 812)
            nn.ReLU(),  # 论文要求：ReLU激活函数
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),  # 第2层：812 -> 406
            nn.ReLU(),  # 论文要求：ReLU激活函数
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),  # 第3层：406 -> 203
            nn.ReLU(),  # 论文要求：ReLU激活函数
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 4, fusion_dim // 8),  # 第4层：203 -> 101
            nn.ReLU(),  # 论文要求：ReLU激活函数
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 8, fusion_dim // 16),  # 第5层：101 -> 50
            nn.ReLU(),  # 论文要求：ReLU激活函数
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 16, num_classes)  # 第6层（最后一层）：50 -> 2（无激活，用于softmax）
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
        text_strings: Optional[list] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播（按照论文算法1）
        
        Args:
            input_ids: 文本token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            images: 图像 [batch_size, 3, H, W]
            text_strings: 原始文本字符串列表 [batch_size]（用于CLIP编码，可选）
            return_features: 是否返回中间特征（用于分析）
        
        Returns:
            output: 包含logits和可选特征的字典
        """
        # 1. 特征提取
        # 微观特征
        text_micro = self.text_encoder(input_ids, attention_mask)  # [batch_size, seq_len, 768]
        image_micro = self.image_encoder(images)  # [batch_size, 512]
        
        # 宏观特征（CLIP）
        # CLIP需要原始文本字符串
        if text_strings is not None:
            text_macro = self.clip_encoder.encode_text_batch(text_strings, images.device)  # [batch_size, 512]
        else:
            # 如果没有提供文本，使用零向量
            text_macro = torch.zeros(images.size(0), self.clip_encoder.clip_dim, device=images.device)
        
        image_macro = self.clip_encoder.encode_image(images)  # [batch_size, 512]
        
        # 2. 初级融合（论文公式5和6：Concat）
        # 论文3.1.4节：d_f = d_l + d_c = 300 + 512 = 812
        text_fused, image_fused = self.feature_fusion(
            text_micro, text_macro, image_micro, image_macro
        )  # [batch_size, fusion_dim=812]
        
        # 3. 特征增强（MHSA）
        # 论文公式(7-11)：输入f_t, f_v维度为d_f=812，输出\hat f_t, \hat f_v维度不变（仍为812）
        text_enhanced, image_enhanced = self.feature_enhancement(
            text_fused, image_fused
        )  # [batch_size, fusion_dim=812]
        
        # 获取可学习内存tokens（论文：可学习内存信息的长度设置为50）
        batch_size = text_enhanced.size(0)
        memory_tokens = self.memory(batch_size)  # [batch_size, memory_length, hidden_size]
        
        # 4. 双流处理（并行）
        # 分支A：协同注意力（Co-Attention）
        # 论文公式(12-14)：输入\hat f_t, \hat f_v维度为d_f=812，输出f_{tv}, f_{vt}维度不变（仍为812）
        # 论文4.3.2节：可学习内存信息长度设置为50，作为额外的K/V参与注意力计算
        text_cross, image_cross = self.co_attention(
            text_enhanced, image_enhanced, memory_tokens
        )  # [batch_size, fusion_dim=812]
        
        # 分支B：歧义性分析（VAE + KL散度）
        # 论文符号矛盾：
        # - 文字描述（公式15-16）：q(z_t|\hat f_t), q(z_v|\hat f_v)（使用增强特征）
        # - 公式(17)(18)：q(z_t|f_t^c), q(z_v|f_v^c)（使用CLIP特征）
        # 本实现按公式(17)(18)使用CLIP特征，与公式保持一致
        ambiguity_results = self.ambiguity_analyzer(
            text_macro, image_macro  # 使用CLIP特征f_t^c, f_v^c（按公式17-18）
        )
        
        # 5. 加权与拼接（论文公式20和21）
        # 论文算法步骤：先得到单模态融合特征f_t, f_v，再得到增强后的\hat f_t, \hat f_v
        # 公式(20)：F_t=(1-a)f_{tuni}，F_v=(1-a)f_{vuni}
        # 这里f_{tuni}和f_{vuni}指的是"单模态融合特征"，即融合后的text_fused/image_fused
        # 所有特征维度均为d_f=812
        fused_feat = self.fusion_module(
            text_fused,  # 文本单模态融合特征f_t（论文公式20中的f_{tuni}）[batch_size, 812]
            image_fused,  # 图像单模态融合特征f_v（论文公式20中的f_{vuni}）[batch_size, 812]
            text_cross,  # 图像增强的文本特征f_{tv}（论文公式20中的F_{tv}）[batch_size, 812]
            image_cross,  # 文本增强的图像特征f_{vt}（论文公式20中的F_{vt}）[batch_size, 812]
            ambiguity_results
        )  # [batch_size, fusion_dim * 4 = 3248]
        
        # 6. 分类（论文公式22：𝑦̂ = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝐹𝐶𝑠(𝐹))）
        logits = self.classifier(fused_feat)  # [batch_size, 2]（2 logits用于softmax）
        
        # 构建输出
        output = {
            'logits': logits,
            'probs': torch.softmax(logits, dim=-1)  # 论文公式(22)：softmax输出
        }
        
        if return_features:
            output.update({
                'text_enhanced': text_enhanced,
                'image_enhanced': image_enhanced,
                'text_cross': text_cross,
                'image_cross': image_cross,
                'fused_feat': fused_feat,
                'ambiguity': ambiguity_results
            })
        
        return output
    
    def forward_from_partial_encoder(
        self,
        bert_after_layer10: torch.Tensor,
        resnet_after_layer3: torch.Tensor,
        text_macro: torch.Tensor,
        image_macro: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        从 partial 缓存前向：只跑 BERT layer 11 + proj、ResNet layer4 + proj，再走融合→分类器。
        bert_after_layer10: [batch_size, seq_len, 768]，resnet_after_layer3: [batch_size, C, H, W]
        text_macro/image_macro: [batch_size, 512]。attention_mask: [batch_size, seq_len]，用于 BERT layer 11。
        """
        # BERT layer 11 + proj -> text_micro
        layer_11 = self.text_encoder.bert.encoder.layer[-1]
        if attention_mask is None:
            attention_mask = torch.ones(
                bert_after_layer10.size(0), bert_after_layer10.size(1),
                dtype=torch.long, device=bert_after_layer10.device
            )
        # BERT layer 期望 4D attention mask（SDPA 格式），与 encoder 一致
        extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
            attention_mask, bert_after_layer10.dtype, tgt_len=bert_after_layer10.size(1)
        )
        layer_outputs = layer_11(bert_after_layer10, attention_mask=extended_attention_mask)
        layer_output = layer_outputs[0]  # (batch, seq_len, 768)
        text_micro = self.text_encoder.dropout(self.text_encoder.proj(layer_output))
        # ResNet layer4 + proj -> image_micro
        backbone_children = list(self.image_encoder.resnet_backbone.children())
        layer4 = backbone_children[7]
        feature_map = layer4(resnet_after_layer3)
        image_micro = self.image_encoder.proj(feature_map)
        # 与 forward_from_features 一致：融合 → 分类
        text_fused, image_fused = self.feature_fusion(
            text_micro, text_macro, image_micro, image_macro
        )
        text_enhanced, image_enhanced = self.feature_enhancement(text_fused, image_fused)
        batch_size = text_enhanced.size(0)
        memory_tokens = self.memory(batch_size)
        text_cross, image_cross = self.co_attention(
            text_enhanced, image_enhanced, memory_tokens
        )
        ambiguity_results = self.ambiguity_analyzer(text_macro, image_macro)
        fused_feat = self.fusion_module(
            text_fused, image_fused, text_cross, image_cross, ambiguity_results
        )
        logits = self.classifier(fused_feat)
        output = {
            'logits': logits,
            'probs': torch.softmax(logits, dim=-1)
        }
        if return_features:
            output.update({
                'text_enhanced': text_enhanced,
                'image_enhanced': image_enhanced,
                'text_cross': text_cross,
                'image_cross': image_cross,
                'fused_feat': fused_feat,
                'ambiguity': ambiguity_results
            })
        return output
    
    def forward_from_features(
        self,
        text_micro: torch.Tensor,
        image_micro: torch.Tensor,
        text_macro: torch.Tensor,
        image_macro: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        从预计算的 encoder 特征前向（训练/推理用缓存时调用，不再跑 BERT/ResNet/CLIP）
        text_micro: [batch_size, seq_len, text_micro_dim]，image_micro: [batch_size, image_micro_dim]
        text_macro/image_macro: [batch_size, 512]
        """
        # 2. 初级融合
        text_fused, image_fused = self.feature_fusion(
            text_micro, text_macro, image_micro, image_macro
        )
        # 3. 特征增强
        text_enhanced, image_enhanced = self.feature_enhancement(text_fused, image_fused)
        batch_size = text_enhanced.size(0)
        memory_tokens = self.memory(batch_size)
        # 4. 双流
        text_cross, image_cross = self.co_attention(
            text_enhanced, image_enhanced, memory_tokens
        )
        ambiguity_results = self.ambiguity_analyzer(text_macro, image_macro)
        # 5. 加权与拼接
        fused_feat = self.fusion_module(
            text_fused, image_fused, text_cross, image_cross, ambiguity_results
        )
        # 6. 分类
        logits = self.classifier(fused_feat)
        output = {
            'logits': logits,
            'probs': torch.softmax(logits, dim=-1)
        }
        if return_features:
            output.update({
                'text_enhanced': text_enhanced,
                'image_enhanced': image_enhanced,
                'text_cross': text_cross,
                'image_cross': image_cross,
                'fused_feat': fused_feat,
                'ambiguity': ambiguity_results
            })
        return output
    
    def predict(self, input_ids, attention_mask, images, text_strings=None):
        """
        预测函数（用于推理）
        
        Args:
            input_ids: 文本token IDs
            attention_mask: 注意力掩码
            images: 图像
            text_strings: 原始文本字符串列表（可选）
        
        Returns:
            predictions: 预测类别（0或1）
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask, images, text_strings)
            # 论文公式(22)：softmax输出，argmax得到预测类别
            logits = output['logits']  # [batch_size, 2]
            predictions = torch.argmax(logits, dim=-1)
        return predictions
