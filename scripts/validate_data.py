#!/usr/bin/env python3
"""
Validate dataset counts vs paper (4.1).
Paper: Weibo fake 3630 real 3479; Pheme fake 590 real 1563; Gossipcop fake 4547 real 10126.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

PAPER_STATS = {
    "weibo": {"fake": 3630, "real": 3479, "images": 6844},
    "pheme": {"fake": 590, "real": 1563, "images": 2018},
    "gossipcop": {"fake": 4547, "real": 10126, "images": 7542},
}


def load_json(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def validate_dataset(name):
    base = DATA_DIR / name
    all_path = base / "all_data.json"
    train_path = base / "train.json"
    val_path = base / "val.json"
    test_path = base / "test.json"
    out = {"name": name, "paper": PAPER_STATS.get(name, {}), "actual": {}}
    if all_path.exists():
        data = load_json(all_path)
        fake = sum(1 for r in data if r.get("label") == 1)
        real = sum(1 for r in data if r.get("label") == 0)
        out["actual"]["total"] = len(data)
        out["actual"]["fake"] = fake
        out["actual"]["real"] = real
    for split_name, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
        if path.exists():
            data = load_json(path)
            out["actual"][split_name] = len(data)
    return out


def main():
    for name in ("gossipcop", "pheme", "weibo"):
        v = validate_dataset(name)
        print(f"\n{name.upper()}")
        print(f"  Paper: fake={v['paper'].get('fake')} real={v['paper'].get('real')}")
        print(f"  Actual: {v['actual']}")
        if v["actual"]:
            total = v["actual"].get("total", 0)
            train = v["actual"].get("train", 0)
            val = v["actual"].get("val", 0)
            test = v["actual"].get("test", 0)
            if total and (train + val + test) == total:
                r = (train, val, test)
                expected = (0.7, 0.1, 0.2)
                ok = all(abs(r[i] / total - expected[i]) < 0.02 for i in range(3))
                print(f"  Split 7:1:2 ok: {ok} (train={train} val={val} test={test})")
    print("\nDone. Replace placeholder data with full datasets and re-run download/convert for paper counts.")


if __name__ == "__main__":
    main()
