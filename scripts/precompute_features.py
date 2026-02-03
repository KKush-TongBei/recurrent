#!/usr/bin/env python3
"""
预计算 partial 特征（BERT 0–10、ResNet 0–3、CLIP）并写入磁盘缓存。
训练时用 feature_cache_dir 后每步只跑 BERT layer 11 + ResNet layer4 + 融合→分类器，
最后 2 个 epoch 可解冻上述层；需在 freeze_encoders: true 且 encoder 配置与训练一致时使用。
新格式会覆盖旧 train.pt/val.pt；若之前用过旧全量缓存，请重跑本脚本后再训练。

用法:
  python scripts/precompute_features.py
  python scripts/precompute_features.py --config config.yaml --cache-dir data/gossipcop/features_cache
"""
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import yaml
from tqdm import tqdm

from models.cmffa import CMFFA
from data.dataset import create_dataloader


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_bert_model_name(data_path: str, config: dict) -> str:
    if 'model' in config and 'text_encoder' in config['model']:
        explicit = config['model']['text_encoder'].get('model_name')
        if explicit and str(explicit).lower() not in ['auto', '']:
            return str(explicit)
    path_lower = (data_path or '').lower()
    if 'weibo' in path_lower:
        return 'bert-base-chinese'
    return 'bert-base-uncased'


def create_model(config: dict) -> CMFFA:
    return CMFFA(
        text_encoder_config=config['model']['text_encoder'],
        image_encoder_config=config['model']['image_encoder'],
        clip_config=config['model']['clip'],
        fusion_config=config['model']['fusion'],
        attention_config=config['model']['attention'],
        classifier_config=config['model']['classifier']
    )


def precompute_split(model: CMFFA, dataloader, device: torch.device, split_name: str):
    """对一个 split 跑 partial encoder 前向：BERT 0–10、ResNet 0–3、CLIP；收集并返回 dict（CPU 张量）。"""
    model.eval()
    bert_after_list = []
    resnet_after_list = []
    text_macro_list = []
    image_macro_list = []
    label_list = []
    attention_mask_list = []
    backbone_children = list(model.image_encoder.resnet_backbone.children())
    backbone_before_layer4 = backbone_children[:7]  # 0–6：到 layer3 为止

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Precompute {split_name}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label']
            text_strings = batch.get('text', None)
            if isinstance(text_strings, str):
                text_strings = [text_strings]

            # BERT：只取 layer 10 的输出（hidden_states[11]）
            bert_out = model.text_encoder.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
            # hidden_states: (embedding, layer0, ..., layer11)，共 13 个；取 [11] = 第 10 层输出
            bert_after_layer10 = bert_out.hidden_states[11]  # [B, seq, 768]
            bert_after_list.append(bert_after_layer10.cpu())
            attention_mask_list.append(attention_mask.cpu())

            # ResNet：只跑前 7 个子模块（到 layer3）
            feat = images
            for child in backbone_before_layer4:
                feat = child(feat)
            resnet_after_list.append(feat.cpu())

            # CLIP
            if text_strings is not None:
                text_macro = model.clip_encoder.encode_text_batch(text_strings, device)
            else:
                text_macro = torch.zeros(images.size(0), model.clip_encoder.clip_dim, device=device)
            image_macro = model.clip_encoder.encode_image(images)
            text_macro_list.append(text_macro.cpu())
            image_macro_list.append(image_macro.cpu())
            label_list.append(labels)

    return {
        'bert_after_layer10': torch.cat(bert_after_list, dim=0),
        'resnet_after_layer3': torch.cat(resnet_after_list, dim=0),
        'text_macro': torch.cat(text_macro_list, dim=0),
        'image_macro': torch.cat(image_macro_list, dim=0),
        'label': torch.cat(label_list, dim=0),
        'attention_mask': torch.cat(attention_mask_list, dim=0)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(ROOT, 'config.yaml'))
    ap.add_argument('--cache-dir', default=None, help='特征缓存目录，默认 <data_dir>/features_cache')
    args = ap.parse_args()
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_path = config['data']['train_path']
    bert_model_name = get_bert_model_name(train_path, config)
    config['model']['text_encoder']['model_name'] = bert_model_name
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model_name, local_files_only=True)
    strict_paper_mode = config.get('training', {}).get('strict_paper_mode', False)
    data_cfg = config['data']
    image_dir = data_cfg['image_dir']
    max_text_length = data_cfg['max_text_length']
    image_size = data_cfg['image_size']
    batch_size = data_cfg.get('batch_size', 32)
    num_workers = min(2, data_cfg.get('num_workers', 2))

    cache_dir = args.cache_dir
    if not cache_dir:
        base = os.path.dirname(os.path.join(ROOT, train_path))
        cache_dir = os.path.join(base, 'features_cache')
    cache_dir = os.path.join(ROOT, cache_dir) if not os.path.isabs(cache_dir) else cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache dir: {cache_dir}")
    print("Partial cache format: bert_after_layer10, resnet_after_layer3, text_macro, image_macro, label, attention_mask. Old full-cache files will be overwritten.")

    model = create_model(config)
    model = model.to(device)
    model.eval()

    splits = [
        ('train', data_cfg['train_path'], True, True),
        ('val', data_cfg['val_path'], False, False),
        ('test', data_cfg.get('test_path'), False, False)
    ]
    for name, path, shuffle, drop_last in splits:
        if not path or not os.path.exists(os.path.join(ROOT, path) if not os.path.isabs(path) else path):
            continue
        abs_path = os.path.join(ROOT, path) if not os.path.isabs(path) else path
        loader = create_dataloader(
            data_path=abs_path,
            image_dir=os.path.join(ROOT, image_dir) if not os.path.isabs(image_dir) else image_dir,
            tokenizer=tokenizer,
            max_text_length=max_text_length,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            is_training=shuffle,
            shuffle=shuffle,
            strict_paper_mode=strict_paper_mode,
            device=device
        )
        data = precompute_split(model, loader, device, name)
        out_path = os.path.join(cache_dir, f'{name}.pt')
        torch.save(data, out_path)
        print(f"Saved {out_path} (N={data['label'].size(0)})")
    print("Done. Set config data.feature_cache_dir to this cache dir and run train.py (partial cache: BERT 11 + ResNet 4 + fusion; light unfreeze will run in last 2 epochs).")


if __name__ == '__main__':
    main()
