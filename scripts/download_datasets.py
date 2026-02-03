#!/usr/bin/env python3
"""
Download datasets per paper (4.1): Weibo, Pheme, Gossipcop.
Paper: 跨模态特征融合与对齐的虚假信息检测模型 (Yang et al.)
Run with network enabled: python3 scripts/download_datasets.py
"""

import os
import json
import urllib.request
import shutil
import zipfile
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def download_gossipcop():
    """Download Gossipcop (FakeNewsNet). Fetches CSV from GitHub."""
    out_dir = DATA_DIR / "gossipcop"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_base = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset"
    ok = True
    for name in ("gossipcop_fake", "gossipcop_real"):
        url = f"{raw_base}/{name}.csv"
        path = out_dir / f"{name}.csv"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                path.write_bytes(r.read())
            print(f"Downloaded {name}.csv -> {path}")
        except Exception as e:
            print(f"Failed {url}: {e}")
            ok = False
    meta = {"source": "FakeNewsNet", "paper_split": "7:1:2", "seed": 42}
    (out_dir / "download_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if not ok:
        print("Tip: Clone manually: git clone https://github.com/KaiDMML/FakeNewsNet.git")
        print("Then copy dataset/gossipcop_*.csv to data/gossipcop/")
    return ok


def download_pheme():
    """Download Pheme dataset (Figshare/GitHub preprocessed)."""
    out_dir = DATA_DIR / "pheme"
    out_dir.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/aung-st/PHEME-Data/archive/refs/heads/master.zip"
    zip_path = out_dir / "pheme.zip"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            zip_path.write_bytes(r.read())
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)
        zip_path.unlink()
        extracted = out_dir / "PHEME-Data-master"
        if extracted.exists():
            for f in extracted.iterdir():
                dest = out_dir / f.name
                if dest.exists():
                    (shutil.rmtree if dest.is_dir() else dest.unlink)(dest)
                shutil.move(str(f), str(out_dir))
            extracted.rmdir()
        print("Downloaded PHEME-Data to data/pheme/")
        return True
    except Exception as e:
        print(f"Pheme download failed: {e}")
        (out_dir / "download_meta.json").write_text(
            json.dumps({"source": "PHEME", "note": "Manual: https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078"}, indent=2),
            encoding="utf-8",
        )
        return False


def download_weibo():
    """Weibo: 曹娟团队 (ref [14]). Document; manual acquisition often required."""
    out_dir = DATA_DIR / "weibo"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README_WEIBO.txt").write_text(
        "Weibo dataset (paper ref [14]: Jin et al., MM 2017, Cao Juan team).\n"
        "Obtain from paper authors or supplementary. Place JSON + images here after conversion.\n",
        encoding="utf-8",
    )
    (out_dir / "download_meta.json").write_text(
        json.dumps({"source": "Weibo", "ref": "[14] Jin et al. MM 2017"}, indent=2),
        encoding="utf-8",
    )
    print("Weibo: README written; dataset may require manual acquisition.")
    return True


def main():
    os.chdir(ROOT)
    download_gossipcop()
    download_pheme()
    download_weibo()
    print("\nDone. If any download failed, run with network or follow tips above.")


if __name__ == "__main__":
    main()
