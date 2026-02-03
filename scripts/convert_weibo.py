#!/usr/bin/env python3
"""
Convert Weibo dataset to project JSON.
Expects data/weibo/ with some structure (e.g. weibo_all.json or train/val/test + images).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIBO_DIR = ROOT / "data" / "weibo"
OUT_JSON = WEIBO_DIR / "all_data.json"


def convert():
    WEIBO_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ("weibo_all.json", "all_data.json", "data.json", "weibo_sample.json"):
        path = WEIBO_DIR / name
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        text = (item.get("text") or item.get("content") or item.get("title") or "")[:5000]
                        label = item.get("label", item.get("rumor", 0))
                        if isinstance(label, bool):
                            label = 1 if label else 0
                        image_path = item.get("image_path") or item.get("image") or ""
                        records.append({
                            "id": item.get("id", str(i)),
                            "text": text or "No text",
                            "image_path": image_path,
                            "label": int(label),
                        })
                    break
            except Exception as e:
                print(f"Error reading {path}: {e}")
    if not records:
        print("No Weibo data found. Place weibo JSON in data/weibo/ (see README_WEIBO.txt).")
        return
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records)} samples to {OUT_JSON}")


if __name__ == "__main__":
    convert()
