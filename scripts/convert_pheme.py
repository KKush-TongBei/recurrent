#!/usr/bin/env python3
"""
Convert PHEME dataset to project JSON.
Expects data/pheme/ with event folders (rumours / non-rumours) and annotation.json.
Or CSV files if using GitHub preprocessed version.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHEME_DIR = ROOT / "data" / "pheme"
OUT_JSON = PHEME_DIR / "all_data.json"


def find_source_tweet_text(event_dir):
    """Get text from source-tweet if available."""
    for sub in event_dir.iterdir():
        if not sub.is_dir():
            continue
        st_dir = sub / "source-tweet"
        if not st_dir.exists():
            continue
        for f in st_dir.glob("*.json"):
            try:
                d = json.loads(f.read_text(encoding="utf-8", errors="replace"))
                return d.get("text", "")
            except Exception:
                pass
    return ""


def convert_from_structure():
    """Convert from PHEME folder structure (rumours / non-rumours)."""
    records = []
    for label_name, label in [("rumours", 1), ("non-rumours", 0)]:
        base = PHEME_DIR / label_name
        if not base.exists():
            for d in PHEME_DIR.iterdir():
                if d.is_dir():
                    sub = d / label_name
                    if sub.exists():
                        base = sub
                        break
        if not base.exists():
            continue
        for event_dir in base.iterdir():
            if not event_dir.is_dir():
                continue
            ann_path = event_dir / "annotation.json"
            text = find_source_tweet_text(event_dir)
            if not text:
                text = event_dir.name
            veracity = None
            if ann_path.exists():
                try:
                    ann = json.loads(ann_path.read_text(encoding="utf-8", errors="replace"))
                    veracity = ann.get("veracity")
                except Exception:
                    pass
            records.append({
                "id": event_dir.name,
                "text": text[:5000],
                "image_path": "",
                "label": label,
            })
    return records


def convert_from_csv():
    """Convert from GitHub CSV if present."""
    records = []
    for f in PHEME_DIR.rglob("*.csv"):
        try:
            import csv
            with open(f, "r", encoding="utf-8", errors="replace") as fp:
                r = csv.DictReader(fp)
                for row in r:
                    text = (row.get("text") or row.get(" tweet") or "").strip()[:5000]
                    if not text:
                        continue
                    label_str = (row.get("label") or row.get(" label") or "").strip().lower()
                    label = 1 if label_str in ("rumour", "rumor", "1", "true") else 0
                    records.append({
                        "id": row.get("id", str(len(records))),
                        "text": text,
                        "image_path": "",
                        "label": label,
                    })
            if records:
                return records
        except Exception:
            continue
    return []


def convert():
    PHEME_DIR.mkdir(parents=True, exist_ok=True)
    records = convert_from_csv()
    if not records:
        records = convert_from_structure()
    if not records:
        print("No PHEME data found. Run download_datasets.py and ensure data/pheme/ has content.")
        return
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records)} samples to {OUT_JSON}")


if __name__ == "__main__":
    convert()
