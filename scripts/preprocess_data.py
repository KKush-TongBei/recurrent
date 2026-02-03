#!/usr/bin/env python3
"""
Preprocess and split datasets per paper: filter, dedupe, split 7:1:2 with seed 42.
Paper 4.1: train:val:test = 7:1:2. Paper 4.3.1: filter articles without images (optional).
"""

import json
import re
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLIT_RATIO = (0.7, 0.1, 0.2)  # train, val, test
SEED = 42


def normalize_text(s):
    if not s or not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_json(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def preprocess(records, filter_no_image=False):
    out = []
    seen = set()
    for r in records:
        text = normalize_text(r.get("text", ""))
        if not text:
            continue
        if filter_no_image and not (r.get("image_path") or r.get("image")):
            continue
        key = (text[:200], r.get("label"))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "text": text[:5000],
            "image_path": r.get("image_path") or r.get("image") or "",
            "label": int(r.get("label", 0)),
        })
    return out


def split(records, train_ratio, val_ratio, test_ratio, seed=42):
    random.seed(seed)
    indices = list(range(len(records)))
    random.shuffle(indices)
    n = len(indices)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    train = [records[i] for i in indices[:t1]]
    val = [records[i] for i in indices[t1:t2]]
    test = [records[i] for i in indices[t2:]]
    return train, val, test


def process_dataset(name, filter_no_image=False):
    base = DATA_DIR / name
    all_path = base / "all_data.json"
    if not all_path.exists():
        print(f"Skip {name}: {all_path} not found")
        return
    records = load_json(all_path)
    records = preprocess(records, filter_no_image=filter_no_image)
    if not records:
        print(f"Skip {name}: no records after preprocess")
        return
    train, val, test = split(records, *SPLIT_RATIO, seed=SEED)
    save_json(base / "train.json", train)
    save_json(base / "val.json", val)
    save_json(base / "test.json", test)
    print(f"{name}: {len(records)} -> train {len(train)} val {len(val)} test {len(test)}")


def main():
    for name in ("gossipcop", "pheme", "weibo"):
        process_dataset(name, filter_no_image=False)
    print("Done. Images: resize to 224x224 in data loader (config).")


if __name__ == "__main__":
    main()
