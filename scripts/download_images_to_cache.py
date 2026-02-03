#!/usr/bin/env python3
"""
方案 A：训练前把数据里的 image_url / 以 http 开头的 image_path 下载到本地缓存，
训练时 Dataset 只读本地文件，彻底避免每个 step 触发网络请求（解决 47s/step 的 I/O 瓶颈）。

用法:
  python scripts/download_images_to_cache.py --config config.yaml
  或指定 JSON 与缓存目录:
  python scripts/download_images_to_cache.py --train data/gossipcop/train.json --val data/gossipcop/val.json --test data/gossipcop/test.json --cache-dir data/gossipcop/images_cache --image-dir data/gossipcop/images
"""
import argparse
import hashlib
import json
import os
import sys
from urllib.request import urlopen, Request

# 项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def download_url_to_path(url: str, out_path: str, timeout: int = 15) -> bool:
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout) as r:
            raw = r.read()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(raw)
        return True
    except Exception as e:
        print(f"  [skip] {url[:60]}... -> {e}")
        return False


def is_url(s: str) -> bool:
    return (s or "").strip().startswith(("http://", "https://"))


def url_to_cache_filename(url: str) -> str:
    h = hashlib.sha256(url.encode()).hexdigest()[:16]
    ext = ".jpg"
    for e in [".png", ".gif", ".webp", ".jpeg"]:
        if e in url.lower():
            ext = e
            break
    return f"{h}{ext}"


def process_json_path(
    json_path: str,
    cache_dir: str,
    image_dir: str,
    key_path: str = "image_path",
    key_url: str = "image_url",
) -> int:
    """把 JSON 里所有 URL 下载到 cache_dir，并把条目改为相对 image_dir 的本地路径。返回下载数量。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"  {json_path}: 根节点不是 list，跳过")
        return 0
    downloaded = 0
    for i, item in enumerate(data):
        url = (item.get(key_url) or item.get(key_path) or "").strip()
        if not is_url(url):
            continue
        cache_name = url_to_cache_filename(url)
        # 缓存目录：默认 image_dir/images_cache，这样 path = "images_cache/xxx.jpg"，Dataset 用 join(image_dir, path) 即可
        local_full = os.path.join(cache_dir, cache_name)
        if not os.path.exists(local_full):
            if download_url_to_path(url, local_full):
                downloaded += 1
        # 相对 image_dir 的路径（image_dir 不变时：images_cache/xxx.jpg）
        item[key_path] = os.path.join("images_cache", cache_name)
        if key_url in item:
            del item[key_url]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return downloaded


def main():
    ap = argparse.ArgumentParser(description="Pre-download image URLs to local cache for training.")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"), help="config.yaml path")
    ap.add_argument("--train", help="train.json path (overrides config)")
    ap.add_argument("--val", help="val.json path")
    ap.add_argument("--test", help="test.json path")
    ap.add_argument("--cache-dir", help="Directory to save downloaded images (default: <image_dir>/images_cache)")
    ap.add_argument("--image-dir", help="image_dir from config (used as base when cache-dir not set)")
    args = ap.parse_args()

    train_path = args.train
    val_path = args.val
    test_path = args.test
    image_dir = args.image_dir
    cache_dir = args.cache_dir

    if not train_path or not image_dir:
        try:
            import yaml
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            data_cfg = cfg.get("data", {})
            if not train_path:
                train_path = data_cfg.get("train_path")
            if not val_path:
                val_path = data_cfg.get("val_path")
            if not test_path:
                test_path = data_cfg.get("test_path")
            if not image_dir:
                image_dir = data_cfg.get("image_dir", "data/gossipcop/images")
        except Exception as e:
            print("Need --train and --image-dir, or a valid --config. Error:", e)
            sys.exit(1)

    if not os.path.isabs(image_dir):
        image_dir = os.path.join(ROOT, image_dir)
    if not cache_dir:
        cache_dir = os.path.join(image_dir, "images_cache")
    elif not os.path.isabs(cache_dir):
        cache_dir = os.path.join(ROOT, cache_dir)

    os.makedirs(cache_dir, exist_ok=True)
    total = 0
    for name, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
        if not path or not os.path.exists(path):
            continue
        abs_path = path if os.path.isabs(path) else os.path.join(ROOT, path)
        n = process_json_path(abs_path, cache_dir, image_dir)
        total += n
        print(f"{name}: {abs_path} -> downloaded {n} new images, cache_dir={cache_dir}")
    print("Done. Set config data.image_dir to the directory that contains 'images_cache' (or set image_dir to cache_dir) so Dataset finds local files.")


if __name__ == "__main__":
    main()
