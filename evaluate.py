"""
评估脚本
用于评估CMFFA模型的性能，计算F1、精确率、召回率等指标
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import json

from models.cmffa import CMFFA
from data.dataset import create_dataloader
from utils.logger import setup_logger
from utils.visualization import plot_confusion_matrix, plot_metrics_bar


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


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


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint.get('epoch', 0)


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    logger,
    save_predictions: bool = False,
    fake_threshold: float = None
) -> dict:
    """评估模型。fake_threshold: 若给定，则 prob(Fake)>fake_threshold 判为 Fake，否则用 argmax。"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            # 移动到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播（从batch中获取原始文本用于CLIP）
            text_strings = batch.get('text', None)
            if text_strings is not None:
                if isinstance(text_strings, str):
                    text_strings = [text_strings]
            output = model(input_ids, attention_mask, images, text_strings=text_strings)
            logits = output['logits']
            probs = output['probs']
            
            # 获取预测：可选用 Fake 阈值，否则 argmax
            if fake_threshold is not None:
                predictions = (probs[:, 1] > fake_threshold).long()
            else:
                predictions = torch.argmax(logits, dim=-1)
            
            # 收集结果
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    # 二分类的详细指标
    precision_binary = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
    recall_binary = recall_score(all_labels, all_predictions, average='binary', zero_division=0)
    f1_binary = f1_score(all_labels, all_predictions, average='binary', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'precision_binary': precision_binary,
        'recall_binary': recall_binary,
        'f1_binary': f1_binary
    }
    
    # 分类报告
    class_names = ['Real', 'Fake']
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        output_dict=True
    )
    
    logger.info("\n" + "="*50)
    logger.info("Evaluation Results")
    logger.info("="*50)
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision_binary:.4f}")
    logger.info(f"Recall:    {recall_binary:.4f}")
    logger.info(f"F1 Score:  {f1_binary:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # 保存预测结果
    if save_predictions:
        predictions_dict = {
            'labels': all_labels.tolist(),
            'predictions': all_predictions.tolist(),
            'probabilities': all_probs.tolist(),
            'metrics': metrics,
            'classification_report': report
        }
        return metrics, predictions_dict, all_labels, all_predictions
    
    return metrics, None, all_labels, all_predictions


def main(args=None):
    # 加载配置
    config_path = getattr(args, 'config', 'config.yaml') if args is not None else 'config.yaml'
    config = load_config(config_path)
    
    # 设置设备
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 设置日志
    logger = setup_logger(
        'evaluate',
        log_dir=config['training']['log_dir'],
        log_to_file=True
    )
    
    # 根据数据集自动选择BERT模型（论文要求：中文按字切分，英文WordPiece切分）
    bert_model_name = get_bert_model_name(config['data']['test_path'], config)
    # 更新config中的模型名称，确保模型和tokenizer使用相同的BERT
    config['model']['text_encoder']['model_name'] = bert_model_name
    logger.info(f"Using BERT model: {bert_model_name} (auto-selected based on dataset)")
    
    # 加载tokenizer（优先本地缓存，避免无网时反复重试）
    tokenizer = BertTokenizer.from_pretrained(bert_model_name, local_files_only=True)
    
    # 获取strict_paper_mode配置（严格遵循论文设置）
    strict_paper_mode = config.get('training', {}).get('strict_paper_mode', False)
    
    # 创建测试数据加载器（论文4.3.2节：测试时batch_size=50）
    test_batch_size = config['data'].get('test_batch_size', config['data']['batch_size'])
    test_loader = create_dataloader(
        data_path=config['data']['test_path'],
        image_dir=config['data']['image_dir'],
        tokenizer=tokenizer,
        max_text_length=config['data']['max_text_length'],
        image_size=config['data']['image_size'],
        batch_size=test_batch_size,
        num_workers=config['data']['num_workers'],
        is_training=False,
        shuffle=False,
        strict_paper_mode=strict_paper_mode,
        device=device
    )
    
    # 创建模型
    model = create_model(config)
    model = model.to(device)
    
    # 加载检查点（默认 best_model.pt；可用 --checkpoint 指定如 final_model.pt、checkpoint_epoch_10.pt）
    checkpoint_name = getattr(args, 'checkpoint', 'best_model.pt') if args is not None else 'best_model.pt'
    checkpoint_path = os.path.join(
        config['training']['save_dir'],
        checkpoint_name
    )
    
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found at {checkpoint_path}")
        logger.info("Using untrained model for evaluation")
    else:
        load_checkpoint(model, checkpoint_path, device)
        logger.info(f"Evaluating checkpoint: {checkpoint_name}")
    
    # 评估
    save_predictions = config['evaluation'].get('save_predictions', True)
    fake_threshold = getattr(args, 'threshold', None)
    if fake_threshold is not None:
        logger.info(f"Using Fake threshold: prob(Fake) > {fake_threshold}")
    metrics, predictions_dict, y_true, y_pred = evaluate(
        model, test_loader, device, logger, save_predictions, fake_threshold=fake_threshold
    )
    
    # 保存结果
    output_dir = config['evaluation'].get('output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存指标
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # 保存预测结果
    if predictions_dict is not None:
        predictions_path = os.path.join(output_dir, 'predictions.json')
        with open(predictions_path, 'w', encoding='utf-8') as f:
            json.dump(predictions_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Predictions saved to {predictions_path}")
    
    # 绘制混淆矩阵
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(
        y_true, y_pred,
        class_names=['Real', 'Fake'],
        save_path=cm_path
    )
    
    # 绘制指标柱状图
    metrics_plot = {
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision_binary'],
        'Recall': metrics['recall_binary'],
        'F1 Score': metrics['f1_binary']
    }
    metrics_path_plot = os.path.join(output_dir, 'metrics_bar.png')
    plot_metrics_bar(metrics_plot, save_path=metrics_path_plot)
    
    logger.info("Evaluation completed!")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Evaluate CMFFA model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='best_model.pt',
        help='Checkpoint 文件名，如 best_model.pt、final_model.pt、checkpoint_epoch_10.pt（默认: best_model.pt）'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='配置文件路径（默认: config.yaml）'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        metavar='T',
        help='Fake 类决策阈值：prob(Fake)>T 则判为 Fake，不设则用 argmax；当前 checkpoint 全预测 Real 时可试 0.3 或 0.35'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
