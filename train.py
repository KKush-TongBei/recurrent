"""
训练脚本
用于训练CMFFA模型
"""

import os
import argparse
import yaml
import random
import torch

# 限制 CPU 线程，避免风扇起飞/CPU 满载（不改模型逻辑）
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(4)

import torch.nn as nn
# 论文4.3.2节：使用 Adam 优化器（非 AdamW，论文未提及 weight_decay/AdamW）
from torch.optim import Adam
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
from pathlib import Path

from models.cmffa import CMFFA
from data.dataset import create_dataloader, create_cached_dataloader
from utils.logger import setup_logger
from utils.visualization import plot_training_curves


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_balanced_class_weights(train_path: str, num_classes: int = 2) -> torch.Tensor:
    """
    从训练集 JSON 统计各类别数量，返回平衡类别权重的张量 [num_classes]。
    weight_i = n_samples / (num_classes * n_i)，少数类权重大，缓解全预测多数类。
    """
    import json
    with open(train_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        return None
    counts = [0] * num_classes
    for item in data:
        label = int(item.get('label', 0))
        if 0 <= label < num_classes:
            counts[label] += 1
    n = sum(counts)
    if n == 0 or min(counts) == 0:
        return None
    # weight_i = n / (num_classes * n_i)
    weights = [n / (num_classes * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def get_bert_model_name(data_path: str, config: dict) -> str:
    """
    根据数据集路径自动选择BERT模型
    论文要求：中文"按字切分"、英文"WordPiece切分"
    
    Args:
        data_path: 数据文件路径（如 "data/weibo/train.json"）
        config: 配置字典
    
    Returns:
        bert_model_name: BERT模型名称
    """
    # 检查config中是否明确指定了模型（且不是"auto"）
    if 'model' in config and 'text_encoder' in config['model']:
        explicit_model = config['model']['text_encoder'].get('model_name')
        # 只有当显式指定且不是"auto"或空字符串时，才使用显式值
        if explicit_model and str(explicit_model).lower() not in ['auto', '']:
            return str(explicit_model)
    
    # 根据数据路径自动判断（论文要求：Weibo用中文BERT，Pheme/Gossipcop用英文BERT）
    data_path_lower = data_path.lower()
    if 'weibo' in data_path_lower:
        return 'bert-base-chinese'  # 中文数据集：按字切分
    elif 'gossipcop' in data_path_lower or 'pheme' in data_path_lower:
        return 'bert-base-uncased'  # 英文数据集：WordPiece切分
    else:
        # 默认使用英文BERT（向后兼容）
        return 'bert-base-uncased'


def get_learning_rate(data_path: str, config: dict) -> float:
    """
    根据数据集自动选择学习率
    论文4.3.2节：Weibo lr=0.001, Pheme lr=0.002
    
    Args:
        data_path: 数据文件路径
        config: 配置字典
    
    Returns:
        learning_rate: 学习率
    """
    # 检查config中是否明确指定了学习率（且不是"auto"）
    explicit_lr = config.get('training', {}).get('learning_rate')
    # 只有当显式指定且不是"auto"或None时，才使用显式值
    if explicit_lr is not None:
        explicit_lr_str = str(explicit_lr).lower()
        if explicit_lr_str not in ['auto', '']:
            try:
                return float(explicit_lr)
            except (ValueError, TypeError):
                # 如果无法转换为float，继续自动选择
                pass
    
    # 根据数据路径自动判断（论文4.3.2节明确说明）
    data_path_lower = data_path.lower()
    if 'weibo' in data_path_lower:
        return 0.001  # 论文4.3.2节：Weibo学习率
    elif 'pheme' in data_path_lower:
        return 0.002  # 论文4.3.2节：Pheme学习率
    else:
        # Gossipcop或其他数据集，默认使用0.001（论文未明确说明Gossipcop的学习率）
        return 0.001


def pgd_attack(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    text_strings: list,
    criterion: nn.Module,
    epsilon: float = 0.01,
    alpha: float = 0.003,
    num_steps: int = 3
) -> torch.Tensor:
    """
    PGD对抗攻击：对BERT文本嵌入添加扰动
    按照论文要求：对文本嵌入做PGD对抗训练来增强鲁棒性
    
    Args:
        model: CMFFA模型
        input_ids: 文本token IDs
        attention_mask: 注意力掩码
        images: 图像
        labels: 标签
        text_strings: 原始文本字符串列表（用于CLIP）
        criterion: 损失函数
        epsilon: 扰动上限
        alpha: 每次迭代步长
        num_steps: PGD迭代次数
    
    Returns:
        perturbed_embeddings: 扰动后的嵌入 [batch_size, seq_len, hidden_size]
    """
    # 获取BERT的embedding层
    text_encoder = model.text_encoder
    bert_model = text_encoder.bert
    embeddings = bert_model.embeddings
    
    # 获取完整的原始embeddings（包含word + position + token_type）
    # 通过一次前向传播获取，确保包含所有embedding组件
    with torch.no_grad():
        # 临时hook获取完整embeddings输出（embeddings层的最终输出）
        full_embeddings_list = []
        def get_full_embeddings(module, input, output):
            full_embeddings_list.append(output.clone())
            return output
        
        handle_temp = embeddings.register_forward_hook(get_full_embeddings)
        try:
            # 使用token_type_ids=0（单句任务）
            token_type_ids = torch.zeros_like(input_ids, device=input_ids.device)
            _ = bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        finally:
            handle_temp.remove()
        
        if len(full_embeddings_list) > 0:
            original_embeddings = full_embeddings_list[0]  # [batch_size, seq_len, hidden_size]
        else:
            # 如果hook未触发，回退到word_embeddings（向后兼容）
            original_embeddings = embeddings.word_embeddings(input_ids)
    
    # 初始化扰动（随机初始化）
    delta = torch.zeros_like(original_embeddings, requires_grad=True)
    
    # PGD迭代攻击
    for step in range(num_steps):
        # 计算当前嵌入（原始 + 扰动）
        perturbed_embeddings = original_embeddings + delta
        
        # 创建hook来替换embedding层的最终输出（包含position/token_type之后）
        # 这确保扰动作用在完整的embeddings上，与BERT真实embedding管线一致
        def embedding_hook(module, input, output):
            # 返回扰动后的完整embeddings（包含所有组件）
            return perturbed_embeddings
        
        # 注册hook
        handle = embeddings.register_forward_hook(embedding_hook)
        
        try:
            # 前向传播计算损失
            output = model(input_ids, attention_mask, images, text_strings=text_strings)
            logits = output['logits']  # [batch_size, 2]
            loss = criterion(logits, labels)  # CrossEntropyLoss需要long类型的labels
            
            # 只对delta求梯度，不污染模型参数梯度（论文规范的PGD实现）
            # 使用torch.autograd.grad只计算delta的梯度
            delta_grad = torch.autograd.grad(
                outputs=loss,
                inputs=delta,
                retain_graph=False,
                create_graph=False,
                only_inputs=True
            )[0]
            
            # 更新扰动（梯度上升）
            if delta_grad is not None:
                delta.data = delta.data + alpha * delta_grad.sign()
                # 投影到epsilon球内
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)
                delta.grad = None  # 清除梯度
        finally:
            # 移除hook
            handle.remove()
    
    return delta.detach()


def create_model(config: dict) -> CMFFA:
    """创建模型"""
    model = CMFFA(
        text_encoder_config=config['model']['text_encoder'],
        image_encoder_config=config['model']['image_encoder'],
        clip_config=config['model']['clip'],
        fusion_config=config['model']['fusion'],
        attention_config=config['model']['attention'],
        classifier_config=config['model']['classifier']
    )
    return model


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    logger,
    pgd_config: dict = None,
    strict_paper_mode: bool = False,
    max_grad_norm: float = 1.0,
    grad_accum_steps: int = 1,
    use_cached_features: bool = False
) -> tuple:
    """训练一个epoch。grad_accum_steps>1 时等效大 batch；use_cached_features 时用预计算特征、不跑 encoder、禁用 PGD。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # PGD 仅在原始输入下可用（缓存特征时无 BERT 嵌入可扰动）
    use_pgd = not use_cached_features and pgd_config is not None and pgd_config.get('enabled', False)
    pgd_epsilon = pgd_config.get('epsilon', 0.01) if use_pgd else 0.01
    pgd_alpha = pgd_config.get('alpha', 0.003) if use_pgd else 0.003
    pgd_steps = pgd_config.get('steps', 3) if use_pgd else 3
    
    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(pbar):
        labels = batch['label'].to(device)
        if use_cached_features:
            bert_after = batch['bert_after_layer10'].to(device)
            resnet_after = batch['resnet_after_layer3'].to(device)
            text_macro = batch['text_macro'].to(device)
            image_macro = batch['image_macro'].to(device)
            attn_mask = batch.get('attention_mask')
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)
            output = model.forward_from_partial_encoder(bert_after, resnet_after, text_macro, image_macro, attention_mask=attn_mask)
        else:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            text_strings = batch.get('text', None)
            if text_strings is not None and isinstance(text_strings, str):
                text_strings = [text_strings]
            output = model(input_ids, attention_mask, images, text_strings=text_strings)
        logits = output['logits']  # [batch_size, 2]
        loss = criterion(logits, labels) / grad_accum_steps  # 梯度累积时按步缩放
        
        # PGD对抗训练（仅原始输入时：对文本嵌入做PGD；缓存特征时跳过）
        if use_pgd:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            text_strings = batch.get('text', None)
            if isinstance(text_strings, str):
                text_strings = [text_strings]
            delta = pgd_attack(
                model, input_ids, attention_mask, images, labels, text_strings,
                criterion, pgd_epsilon, pgd_alpha, pgd_steps
            )
            text_encoder = model.text_encoder
            bert_model = text_encoder.bert
            embeddings = bert_model.embeddings
            with torch.no_grad():
                full_embeddings_list = []
                def get_full_embeddings(module, input, output):
                    full_embeddings_list.append(output.clone())
                    return output
                handle_get = embeddings.register_forward_hook(get_full_embeddings)
                try:
                    token_type_ids = torch.zeros_like(input_ids, device=input_ids.device)
                    _ = bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                finally:
                    handle_get.remove()
                full_embeddings = full_embeddings_list[0]
            perturbed_embeddings = full_embeddings + delta
            def embedding_hook(module, input, output):
                return perturbed_embeddings
            handle = embeddings.register_forward_hook(embedding_hook)
            try:
                adv_output = model(input_ids, attention_mask, images, text_strings=text_strings)
                adv_logits = adv_output['logits']
                adv_loss = criterion(adv_logits, labels) / grad_accum_steps
                total_loss_batch = loss + adv_loss
            finally:
                handle.remove()
        else:
            total_loss_batch = loss
        
        # 反向传播（梯度累积）
        total_loss_batch.backward()
        
        # 每 grad_accum_steps 步或最后一步才更新参数
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            if not strict_paper_mode:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=max_grad_norm
                )
            optimizer.step()
            optimizer.zero_grad()
        
        # 统计（按未缩放 loss 显示）
        total_loss += total_loss_batch.item() * grad_accum_steps
        # 论文公式(22)：softmax输出，argmax得到预测类别
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{total_loss_batch.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    logger,
    use_cached_features: bool = False
) -> tuple:
    """验证。use_cached_features 时用 forward_from_partial_encoder。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for batch in pbar:
            labels = batch['label'].to(device)
            if use_cached_features:
                bert_after = batch['bert_after_layer10'].to(device)
                resnet_after = batch['resnet_after_layer3'].to(device)
                text_macro = batch['text_macro'].to(device)
                image_macro = batch['image_macro'].to(device)
                attn_mask = batch.get('attention_mask')
                if attn_mask is not None:
                    attn_mask = attn_mask.to(device)
                output = model.forward_from_partial_encoder(bert_after, resnet_after, text_macro, image_macro, attention_mask=attn_mask)
            else:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                images = batch['image'].to(device)
                text_strings = batch.get('text', None)
                if text_strings is not None and isinstance(text_strings, str):
                    text_strings = [text_strings]
                output = model(input_ids, attention_mask, images, text_strings=text_strings)
            logits = output['logits']  # [batch_size, 2]
            
            # 计算损失
            loss = criterion(logits, labels)  # CrossEntropyLoss需要long类型的labels
            
            # 统计
            total_loss += loss.item()
            # 论文公式(22)：softmax输出，argmax得到预测类别
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str
):
    """保存检查点"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    print(f"Checkpoint saved to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train CMFFA model')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, metavar='PATH', help='从该 checkpoint 继续训练（如 checkpoints/best_model.pt）')
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = getattr(args, 'config', 'config.yaml')
    config = load_config(config_path)
    
    # 设置设备：CUDA > MPS（Apple Silicon）> CPU，MPS 下保持论文超参且减轻 CPU 负载
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device(config['training'].get('device', 'cpu'))
    print(f"Using device: {device}")
    
    # 固定随机种子（论文4.3.2节：seed=42，保证训练全程可复现）
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True  # 可复现，略慢
    torch.backends.cudnn.benchmark = False
    
    # 设置日志
    logger = setup_logger(
        'train',
        log_dir=config['training']['log_dir'],
        log_to_file=True
    )
    
    # 根据数据集自动选择BERT模型（论文要求：中文按字切分，英文WordPiece切分）
    bert_model_name = get_bert_model_name(config['data']['train_path'], config)
    # 更新config中的模型名称，确保模型和tokenizer使用相同的BERT
    config['model']['text_encoder']['model_name'] = bert_model_name
    logger.info(f"Using BERT model: {bert_model_name} (auto-selected based on dataset)")
    
    # 加载tokenizer（优先本地缓存，避免无网时反复重试 huggingface.co）
    logger.info("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(bert_model_name, local_files_only=True)
    logger.info("Tokenizer loaded.")
    
    # 获取strict_paper_mode配置（严格遵循论文设置）
    strict_paper_mode = config.get('training', {}).get('strict_paper_mode', False)
    if strict_paper_mode:
        logger.info("="*50)
        logger.info("STRICT PAPER MODE ENABLED")
        logger.info("Following paper settings exactly:")
        logger.info("- Chinese text: character-level tokenization (按字切分)")
        logger.info("- Image preprocessing: resize only (224x224), no augmentation")
        logger.info("- Training: no weight_decay, no warmup, no grad_clip (paper not mentioned)")
        logger.info("="*50)
    
    # 若配置了特征缓存且存在 train.pt，则用预计算特征（不再跑 BERT/ResNet/CLIP，单步快一个数量级）
    data_cfg = config['data']
    feature_cache_dir = data_cfg.get('feature_cache_dir', '')
    train_cache_path = os.path.join(feature_cache_dir, 'train.pt') if feature_cache_dir else ''
    use_cached_features = bool(feature_cache_dir and os.path.exists(train_cache_path))
    if use_cached_features:
        logger.info(f"Using partial encoder cache from {feature_cache_dir} (BERT 0–10, ResNet 0–3 + CLIP); training runs BERT 11 + ResNet 4 + fusion.")
        val_cache_path = os.path.join(feature_cache_dir, 'val.pt')
        logger.info("Loading train cache (train.pt)...")
        train_loader = create_cached_dataloader(
            train_cache_path,
            batch_size=data_cfg['batch_size'],
            num_workers=data_cfg.get('num_workers', 4),
            is_training=True,
            shuffle=True,
            device=device
        )
        logger.info("Train cache loaded.")
        logger.info("Loading val cache (val.pt)...")
        val_loader = create_cached_dataloader(
            val_cache_path,
            batch_size=data_cfg['batch_size'],
            num_workers=data_cfg.get('num_workers', 4),
            is_training=False,
            shuffle=False,
            device=device
        ) if os.path.exists(val_cache_path) else None
        if val_loader is not None:
            logger.info("Val cache loaded.")
        if val_loader is None:
            logger.warning(f"Val cache not found at {val_cache_path}; validation will be skipped.")
    else:
        train_loader = create_dataloader(
            data_path=data_cfg['train_path'],
            image_dir=data_cfg['image_dir'],
            tokenizer=tokenizer,
            max_text_length=data_cfg['max_text_length'],
            image_size=data_cfg['image_size'],
            batch_size=data_cfg['batch_size'],
            num_workers=data_cfg.get('num_workers', 4),
            is_training=True,
            shuffle=True,
            strict_paper_mode=strict_paper_mode,
            device=device
        )
        val_loader = create_dataloader(
            data_path=data_cfg['val_path'],
            image_dir=data_cfg['image_dir'],
            tokenizer=tokenizer,
            max_text_length=data_cfg['max_text_length'],
            image_size=data_cfg['image_size'],
            batch_size=data_cfg['batch_size'],
            num_workers=data_cfg.get('num_workers', 4),
            is_training=False,
            shuffle=False,
            strict_paper_mode=strict_paper_mode,
            device=device
        )
    
    # 创建模型（含 BERT/ResNet/CLIP，首次或加载权重可能较慢）
    logger.info("Creating model...")
    model = create_model(config)
    logger.info("Model created, moving to device...")
    model = model.to(device)
    logger.info("Model ready.")
    
    # 可选：冻结 BERT + ResNet，只训练融合/注意力/VAE/分类器（提速明显，思路不变）
    freeze_encoders = config.get('training', {}).get('freeze_encoders', False)
    if freeze_encoders:
        for p in model.text_encoder.parameters():
            p.requires_grad = False
        for p in model.image_encoder.parameters():
            p.requires_grad = False
        logger.info("Frozen text_encoder (BERT) and image_encoder (ResNet); only fusion/attention/VAE/classifier are trained.")

    def unfreeze_encoder_last_layers(model):
        """解冻 BERT 最后一层 + ResNet 最后一个 block，返回新可训练参数列表（用于加入优化器）。"""
        newly_trainable = []
        if hasattr(model.text_encoder, 'bert') and hasattr(model.text_encoder.bert, 'encoder'):
            last_layer = model.text_encoder.bert.encoder.layer[-1]
            for p in last_layer.parameters():
                p.requires_grad = True
                newly_trainable.append(p)
        if hasattr(model.image_encoder, 'resnet_backbone'):
            children = list(model.image_encoder.resnet_backbone.children())
            if children:
                last_block = children[-1]
                for p in last_block.parameters():
                    p.requires_grad = True
                    newly_trainable.append(p)
        return newly_trainable
    
    # 损失函数（论文公式23：交叉熵损失）
    # 论文公式(22)：𝑦̂ = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝐹𝐶𝑠(𝐹))
    # 论文公式(23)：ℒcls = 𝑦𝑙𝑜𝑔(𝑦̂ ) + (1 − 𝑦)𝑙𝑜𝑔(1 − 𝑦̂ )
    # 可选：类别权重，让模型更在意少数类（Fake），缓解全预测 Real）
    num_classes = config['model']['classifier'].get('num_classes', 2)
    class_weight = config.get('training', {}).get('class_weight', None)
    if class_weight == 'balanced':
        train_path = data_cfg['train_path']
        if not os.path.isabs(train_path):
            train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), train_path)
        weight_tensor = get_balanced_class_weights(train_path, num_classes)
        if weight_tensor is not None:
            weight_tensor = weight_tensor.to(device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            logger.info(f"Using CrossEntropyLoss with balanced class_weight: {weight_tensor.tolist()}")
        else:
            criterion = nn.CrossEntropyLoss()
            logger.info("Using CrossEntropyLoss (balanced weight computation failed, using no weight)")
    elif isinstance(class_weight, (list, tuple)) and len(class_weight) == num_classes:
        weight_tensor = torch.tensor(class_weight, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        logger.info(f"Using CrossEntropyLoss with class_weight: {class_weight}")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Using CrossEntropyLoss (no class weight)")
    
    # 根据数据集自动选择学习率（论文4.3.2节：Weibo 0.001, Pheme 0.002）
    learning_rate = get_learning_rate(config['data']['train_path'], config)
    logger.info(f"Using learning rate: {learning_rate} (auto-selected based on dataset)")
    
    # 优化器：论文4.3.2节明确为 Adam（非 AdamW）
    # 论文未提及 weight_decay、warmup；strict_paper_mode 下不使用任何学习率调度器（无 warmup）
    if strict_paper_mode:
        weight_decay = 0.0
        logger.info("Strict paper mode: Adam, weight_decay=0, no warmup (paper not mentioned)")
    else:
        weight_decay = config['training'].get('weight_decay', 1e-4)
        if isinstance(weight_decay, str):
            weight_decay = float(weight_decay)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay
    )
    if freeze_encoders:
        logger.info(f"Optimizer only updates {len(trainable_params)} trainable parameter tensors (encoders frozen).")
    
    # 从 checkpoint 恢复（可选）：只加载模型权重，从下一 epoch 继续；optimizer 不加载（避免解冻后 param group 不一致）
    start_epoch = 1
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')
    resume_from = config.get('training', {}).get('resume_from') or getattr(args, 'resume', None)
    if resume_from:
        resume_path = resume_from if os.path.isabs(resume_from) else os.path.join(os.path.dirname(os.path.abspath(__file__)), resume_from)
        if os.path.exists(resume_path):
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'], strict=True)
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val_loss = ckpt.get('loss', float('inf'))
            logger.info(f"Resumed from {resume_path}, starting at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        else:
            logger.warning(f"resume_from path not found: {resume_path}, training from scratch.")
    
    # 训练循环
    num_epochs = config['training']['num_epochs']
    save_dir = config['training']['save_dir']
    grad_accum_steps = config['training'].get('grad_accum_steps', 1)
    if grad_accum_steps > 1:
        logger.info(f"Gradient accumulation: {grad_accum_steps} steps (effective batch = {config['data']['batch_size']} * {grad_accum_steps})")
    # 折中方案：最后 1–2 个 epoch 只解冻 BERT 最后一层 + ResNet 最后一 block，提升 Fake recall，速度影响小
    unfreeze_last_layers = config.get('training', {}).get('unfreeze_last_layers', False)
    unfreeze_last_epochs = config.get('training', {}).get('unfreeze_last_epochs', 2)
    light_unfreeze_lr = config.get('training', {}).get('light_unfreeze_lr', 1e-5)
    light_unfreeze_lr = float(light_unfreeze_lr)  # YAML 可能解析为 str，确保 optimizer 收到 float
    unfreeze_start_epoch = max(1, num_epochs - unfreeze_last_epochs + 1) if unfreeze_last_layers else num_epochs + 1
    if unfreeze_last_layers and freeze_encoders:
        logger.info(f"Light unfreeze: from epoch {unfreeze_start_epoch}, BERT last layer + ResNet last block will be unfrozen for {unfreeze_last_epochs} epoch(s), lr={light_unfreeze_lr}")
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, num_epochs + 1):
        # 进入“最后几轮”时解冻 BERT 最后一层 + ResNet 最后一 block，并加入优化器（partial 缓存时前向已含 BERT 11 + ResNet 4，解冻生效）
        if unfreeze_last_layers and freeze_encoders and epoch == unfreeze_start_epoch:
            newly_trainable = unfreeze_encoder_last_layers(model)
            if newly_trainable:
                optimizer.add_param_group({'params': newly_trainable, 'lr': light_unfreeze_lr})
                logger.info(f"Unfrozen BERT last layer + ResNet last block for final {unfreeze_last_epochs} epoch(s) (lr={light_unfreeze_lr}), added {len(newly_trainable)} param tensors to optimizer.")
        
        logger.info(f"\nEpoch {epoch}/{num_epochs}")
        
        # 训练
        pgd_config = config['training'].get('pgd', {})
        max_grad_norm = config['training'].get('max_grad_norm', 1.0) if not strict_paper_mode else None
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, logger,
            pgd_config, strict_paper_mode, max_grad_norm, grad_accum_steps,
            use_cached_features=use_cached_features
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证（无 val_loader 时跳过，如仅生成了 train.pt 缓存）
        if val_loader is not None:
            val_loss, val_acc = validate(
                model, val_loader, criterion, device, logger,
                use_cached_features=use_cached_features
            )
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            logger.info(
                f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    os.path.join(save_dir, 'best_model.pt')
                )
            if epoch % config['training']['save_every'] == 0:
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
                )
        else:
            val_losses.append(0.0)
            val_accs.append(0.0)
            val_loss = train_loss  # 无 val 时用 train_loss 占位，供下面 save_checkpoint 用
            logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} (no val)")
        
        # 定期保存检查点（有 val 时上面已按 best 保存过 best_model）
        if epoch % config['training']['save_every'] == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
            )
        
        # 绘制训练曲线
        if epoch % config['training']['eval_every'] == 0:
            plot_training_curves(
                train_losses, val_losses, train_accs, val_accs,
                save_path=os.path.join(config['training']['log_dir'], 'training_curves.png')
            )
    
    logger.info("Training completed!")
    
    # 保存最终模型
    save_checkpoint(
        model, optimizer, num_epochs, val_loss,
        os.path.join(save_dir, 'final_model.pt')
    )
    
    # 绘制最终训练曲线
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        save_path=os.path.join(config['training']['log_dir'], 'final_training_curves.png')
    )


if __name__ == '__main__':
    main()
