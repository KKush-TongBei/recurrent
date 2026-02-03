"""
数据加载模块
支持JSON格式的多模态虚假信息检测数据集
论文要求：中文按字切分，英文WordPiece切分
"""

import json
import os
import re
import io
from typing import Dict, List, Tuple, Optional
from urllib.request import urlopen, Request
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer
import numpy as np


def is_chinese_text(text: str) -> bool:
    """
    检测文本是否主要为中文
    论文要求：中文按字切分，英文WordPiece切分
    
    Args:
        text: 文本字符串
    
    Returns:
        is_chinese: 是否主要为中文
    """
    if not text:
        return False
    
    # 统计中文字符数量
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_chars = len(text.replace(' ', ''))
    
    # 如果中文字符占比超过30%，认为是中文文本
    if total_chars > 0:
        return chinese_chars / total_chars > 0.3
    return False


class CharacterLevelTokenizer:
    """
    字符级tokenizer（用于中文按字切分）
    论文要求：中文按字切分，每个中文字符对应一个token
    
    这个类包装BERT tokenizer，但对于中文文本，确保每个字符都被切分为独立的token。
    对于bert-base-chinese，虽然它本身是按字训练的，但为了严格遵循论文要求，
    我们显式地按字符进行tokenization。
    """
    
    def __init__(self, base_tokenizer: BertTokenizer):
        """
        初始化字符级tokenizer
        
        Args:
            base_tokenizer: 基础的BERT tokenizer（bert-base-chinese）
        """
        self.base_tokenizer = base_tokenizer
        self.vocab = base_tokenizer.get_vocab()
        self.unk_token_id = base_tokenizer.unk_token_id
        self.cls_token_id = base_tokenizer.cls_token_id
        self.sep_token_id = base_tokenizer.sep_token_id
        self.pad_token_id = base_tokenizer.pad_token_id
        self.mask_token_id = base_tokenizer.mask_token_id
        
    def _char_to_token_id(self, char: str) -> int:
        """
        将单个字符转换为token ID
        
        Args:
            char: 单个字符
        
        Returns:
            token_id: token ID
        """
        # 对于bert-base-chinese，大多数中文字符都在vocab中
        # 如果字符在vocab中，直接返回对应的ID
        if char in self.vocab:
            return self.vocab[char]
        # 如果不在vocab中，尝试使用tokenizer的encode方法
        # 但为了确保按字切分，我们使用字符本身作为token
        encoded = self.base_tokenizer.encode(char, add_special_tokens=False)
        if len(encoded) > 0:
            return encoded[0]
        return self.unk_token_id
    
    def encode_chinese_text(self, text: str, max_length: int = 512, 
                           padding: str = 'max_length', 
                           truncation: bool = True) -> Dict[str, torch.Tensor]:
        """
        对中文文本进行字符级编码（严格按字切分）
        论文要求：中文按字切分，每个字符一个token
        
        Args:
            text: 中文文本
            max_length: 最大长度
            padding: 填充方式
            truncation: 是否截断
        
        Returns:
            encoded: 包含input_ids和attention_mask的字典
        """
        # 移除首尾空格
        text = text.strip()
        
        # 按字符切分：每个字符（包括中文字符、英文、数字、标点）都作为独立的token
        # 但为了与BERT兼容，我们保留空格的处理
        chars = []
        for char in text:
            # 跳过空格（BERT tokenizer通常忽略空格）
            if char == ' ':
                continue
            chars.append(char)
        
        # 转换为token IDs（每个字符一个token）
        token_ids = [self.cls_token_id]  # [CLS]
        for char in chars:
            token_id = self._char_to_token_id(char)
            token_ids.append(token_id)
        token_ids.append(self.sep_token_id)  # [SEP]
        
        # 截断
        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length-1] + [self.sep_token_id]
        
        # 填充
        attention_mask = [1] * len(token_ids)
        if padding == 'max_length':
            while len(token_ids) < max_length:
                token_ids.append(self.pad_token_id)
                attention_mask.append(0)
        
        return {
            'input_ids': torch.tensor([token_ids], dtype=torch.long),
            'attention_mask': torch.tensor([attention_mask], dtype=torch.long)
        }


def preprocess_chinese_text(text: str) -> str:
    """
    对中文文本进行字符级切分（按字切分）
    论文要求：中文按字切分
    
    注意：这个函数现在主要用于CLIP编码，BERT编码使用CharacterLevelTokenizer
    
    Args:
        text: 原始中文文本
    
    Returns:
        processed_text: 按字符切分后的文本（字符之间用空格分隔）
    """
    # 按字符切分：每个字符之间用空格分隔
    # 保留英文单词和数字的完整性，只对中文字符进行切分
    processed_chars = []
    current_word = []
    
    for char in text:
        # 中文字符：单独切分（按字切分）
        if '\u4e00' <= char <= '\u9fff':
            if current_word:
                processed_chars.append(''.join(current_word))
                current_word = []
            processed_chars.append(char)
        # 英文、数字、标点：保持连续（不切分）
        else:
            if char == ' ':
                if current_word:
                    processed_chars.append(''.join(current_word))
                    current_word = []
                # 保留空格
                if processed_chars and processed_chars[-1] != ' ':
                    processed_chars.append(' ')
            else:
                current_word.append(char)
    
    if current_word:
        processed_chars.append(''.join(current_word))
    
    # 用空格连接所有字符/单词（确保中文字符之间用空格分隔，实现按字切分）
    result = ' '.join(processed_chars)
    # 清理多余空格（保留单个空格）
    result = re.sub(r'\s+', ' ', result).strip()
    return result


class FakeNewsDataset(Dataset):
    """虚假信息检测数据集"""
    
    def __init__(
        self,
        data_path: str,
        image_dir: str,
        tokenizer: BertTokenizer,
        max_text_length: int = 512,
        image_size: int = 224,
        is_training: bool = True,
        strict_paper_mode: bool = False
    ):
        """
        初始化数据集
        
        Args:
            data_path: JSON数据文件路径
            image_dir: 图像文件夹路径
            tokenizer: BERT tokenizer
            max_text_length: 文本最大长度
            image_size: 图像尺寸
            is_training: 是否为训练模式（用于数据增强）
            strict_paper_mode: 是否严格遵循论文设置（论文只做resize，无增强）
        """
        self.data_path = data_path
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.is_training = is_training
        self.strict_paper_mode = strict_paper_mode
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 判断是否为中文数据集（用于选择tokenizer策略）
        self.is_chinese_dataset = 'weibo' in data_path.lower()
        
        # 如果是中文数据集，创建字符级tokenizer（严格按字切分）
        if self.is_chinese_dataset:
            self.char_tokenizer = CharacterLevelTokenizer(tokenizer)
        else:
            self.char_tokenizer = None
        
        # URL 图片缓存：同一 URL 只请求一次，避免每个 step 触发网络（解决 47s/step 的 I/O 瓶颈）
        self._url_image_cache: Dict[str, Image.Image] = {}
        # 图像预处理
        # 论文4.3.1节：图像统一调整为224×224，不进行数据增强
        # strict_paper_mode=True时，只做resize和归一化，无任何增强
        if strict_paper_mode or not is_training:
            # 严格遵循论文：只resize，无增强
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            # 非严格模式：可以添加一些增强（但论文未提及，所以默认不添加）
            # 为了与论文一致，这里也只用resize
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        期望JSON格式:
        {
            "text": "新闻文本内容",
            "image_path": "image.jpg",  # 相对于image_dir的路径
            "label": 0  # 0: 真实, 1: 虚假
        }
        """
        item = self.data[idx]
        
        # 处理文本
        text = item.get('text', '')
        
        # 论文要求：中文按字切分，英文WordPiece切分
        if self.is_chinese_dataset and self.char_tokenizer is not None:
            # 中文数据集：使用字符级tokenizer（严格按字切分）
            # 每个中文字符对应一个token
            text_encoded = self.char_tokenizer.encode_chinese_text(
                text,
                max_length=self.max_text_length,
                padding='max_length',
                truncation=True
            )
        else:
            # 英文数据集：使用WordPiece切分（BERT默认方式）
            text_encoded = self.tokenizer(
                text,
                max_length=self.max_text_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        # 对于CLIP编码，保留原始文本（中文需要预处理）
        text_for_clip = text
        if self.is_chinese_dataset:
            text_for_clip = preprocess_chinese_text(text)
        
        # 处理图像（支持本地路径 或 image_url / 以 http 开头的 image_path，带缓存避免每 step 请求网络）
        image_path = item.get('image_path', '') or item.get('image_url', '')
        if image_path:
            if image_path.startswith(('http://', 'https://')):
                # URL：使用缓存，同一 URL 不重复请求
                if image_path not in self._url_image_cache:
                    try:
                        req = Request(image_path, headers={'User-Agent': 'Mozilla/5.0'})
                        with urlopen(req, timeout=15) as r:
                            raw = r.read()
                        self._url_image_cache[image_path] = Image.open(io.BytesIO(raw)).convert('RGB')
                    except Exception:
                        self._url_image_cache[image_path] = None
                pil_img = self._url_image_cache[image_path]
                if pil_img is not None:
                    image = self.image_transform(pil_img)
                else:
                    image = torch.zeros(3, self.image_transform.transforms[0].size[0],
                                        self.image_transform.transforms[0].size[0])
            else:
                full_image_path = os.path.join(self.image_dir, image_path)
                if os.path.exists(full_image_path):
                    image = Image.open(full_image_path).convert('RGB')
                    image = self.image_transform(image)
                else:
                    image = torch.zeros(3, self.image_transform.transforms[0].size[0],
                                        self.image_transform.transforms[0].size[0])
        else:
            image = torch.zeros(3, self.image_transform.transforms[0].size[0],
                                self.image_transform.transforms[0].size[0])
        
        # 标签
        label = item.get('label', 0)
        label = torch.tensor(label, dtype=torch.long)
        
        return {
            'input_ids': text_encoded['input_ids'].squeeze(0),
            'attention_mask': text_encoded['attention_mask'].squeeze(0),
            'image': image,
            'label': label,
            'text': text_for_clip,  # 保留处理后的文本用于CLIP编码
            'image_path': image_path
        }


class CachedFeaturesDataset(Dataset):
    """
    从 partial 预计算缓存加载（BERT 0–10、ResNet 0–3、CLIP 输出）。
    缓存 .pt 为 dict：bert_after_layer10 [N,seq,768], resnet_after_layer3 [N,C,H,W],
    text_macro [N,512], image_macro [N,512], label [N], attention_mask [N,seq]。
    """
    
    def __init__(self, cache_path: str):
        """
        Args:
            cache_path: 单分片缓存路径，如 data/gossipcop/features_cache/train.pt
        """
        self.data = torch.load(cache_path, map_location='cpu')
        self.n = self.data['label'].size(0)
    
    def __len__(self) -> int:
        return self.n
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        out = {
            'bert_after_layer10': self.data['bert_after_layer10'][idx],
            'resnet_after_layer3': self.data['resnet_after_layer3'][idx],
            'text_macro': self.data['text_macro'][idx],
            'image_macro': self.data['image_macro'][idx],
            'label': self.data['label'][idx]
        }
        if 'attention_mask' in self.data:
            out['attention_mask'] = self.data['attention_mask'][idx]
        return out


def create_cached_dataloader(
    cache_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    is_training: bool = True,
    shuffle: bool = True,
    device: Optional[torch.device] = None
) -> DataLoader:
    """
    使用 partial 预计算缓存的 DataLoader（训练时跑 BERT 11 + ResNet 4 + 融合→分类器）。
    """
    dataset = CachedFeaturesDataset(cache_path)
    use_pin_memory = device is not None and device.type == 'cuda'
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=is_training
    )


def create_dataloader(
    data_path: str,
    image_dir: str,
    tokenizer: BertTokenizer,
    max_text_length: int = 512,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    is_training: bool = True,
    shuffle: bool = True,
    strict_paper_mode: bool = False,
    device: Optional[torch.device] = None
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        data_path: JSON数据文件路径
        image_dir: 图像文件夹路径
        tokenizer: BERT tokenizer
        max_text_length: 文本最大长度
        image_size: 图像尺寸
        batch_size: 批次大小
        num_workers: 数据加载线程数
        is_training: 是否为训练模式
        shuffle: 是否打乱数据
        strict_paper_mode: 是否严格遵循论文设置（论文只做resize，无增强）
        device: 训练/推理设备；仅 CUDA 时 pin_memory=True，MPS/CPU 为 False（避免 MPS 警告与额外开销）
    
    Returns:
        DataLoader对象
    """
    dataset = FakeNewsDataset(
        data_path=data_path,
        image_dir=image_dir,
        tokenizer=tokenizer,
        max_text_length=max_text_length,
        image_size=image_size,
        is_training=is_training,
        strict_paper_mode=strict_paper_mode
    )
    # MPS/CPU 上 pin_memory 无益且会触发警告；仅 CUDA 时开启
    use_pin_memory = device is not None and device.type == 'cuda'
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=is_training
    )
    return dataloader
