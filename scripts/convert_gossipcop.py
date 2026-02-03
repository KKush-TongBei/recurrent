#!/usr/bin/env python3
"""
Convert Gossipcop (FakeNewsNet) CSV + optional news content to project JSON.
Expects data/gossipcop/gossipcop_fake.csv and gossipcop_real.csv.
If news content is available (news content.json per article), use it; else use title from CSV.
"""

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOSSIPCOP_DIR = ROOT / "data" / "gossipcop"
IMAGES_DIR = GOSSIPCOP_DIR / "images"
OUT_JSON = GOSSIPCOP_DIR / "all_data.json"


def load_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def normalize_text(s):
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    return s[:5000]


def convert():
    GOSSIPCOP_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for name, label in [("gossipcop_fake", 1), ("gossipcop_real", 0)]:
        path = GOSSIPCOP_DIR / f"{name}.csv"
        if not path.exists():
            print(f"Skip: {path} not found. Run download_datasets.py first.")
            continue
        rows = load_csv(path)
        for i, row in enumerate(rows):
            news_id = row.get("id", "").strip() or f"{name}_{i}"
            title = normalize_text(row.get("title", ""))
            url = row.get("url", "").strip()
            text = title
            if not text and url:
                text = url
            if not text:
                text = "No title"
            image_path = ""
            records.append({
                "id": news_id,
                "text": text,
                "image_path": image_path,
                "label": label,
                "url": url,
            })
        print(f"Loaded {len(rows)} rows from {name}.csv")
    if not records:
        print("No records. Ensure data/gossipcop/gossipcop_*.csv exist.")
        return
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records)} samples to {OUT_JSON}")


if __name__ == "__main__":
    convert()
