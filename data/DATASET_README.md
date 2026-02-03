# 数据集说明（与论文4.1节一致）

论文使用 3 个数据集：**Weibo**（中文）、**Pheme**（英文 Twitter）、**Gossipcop**（英文 FakeNewsNet），划分比例 **7:1:2**（训练:验证:测试），随机种子 **42**。

## 1. 获取完整数据

在项目根目录执行（需联网）：

```bash
python3 scripts/download_datasets.py
```

- **Gossipcop**：从 FakeNewsNet 仓库下载 `gossipcop_fake.csv`、`gossipcop_real.csv`。若失败可手动执行：`git clone https://github.com/KaiDMML/FakeNewsNet.git`，再将 `dataset/gossipcop_*.csv` 复制到 `data/gossipcop/`。
- **Pheme**：从 GitHub 下载 PHEME-Data。或从 Figshare：https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078
- **Weibo**：需从论文作者或补充材料获取（参考文献 [14]），放入 `data/weibo/` 后按下方步骤转换。

## 2. 转换与划分

```bash
python3 scripts/convert_gossipcop.py
python3 scripts/convert_pheme.py
python3 scripts/convert_weibo.py    # 需先有 Weibo 原始数据
python3 scripts/preprocess_data.py  # 清洗、去重、7:1:2 划分（seed=42）
python3 scripts/validate_data.py    # 检查数量是否与论文一致
```

## 3. 使用某一数据集训练

修改 `config.yaml` 中数据路径，例如使用 Gossipcop：

```yaml
data:
  train_path: "data/gossipcop/train.json"
  val_path: "data/gossipcop/val.json"
  test_path: "data/gossipcop/test.json"
  image_dir: "data/gossipcop/images"
```

## 4. 论文统计（表 1）

| 数据集   | 虚假  | 真实   | 图像  |
|----------|-------|--------|-------|
| Weibo    | 3630  | 3479   | 6844  |
| Pheme    | 590   | 1563   | 2018  |
| Gossipcop| 4547  | 10126  | 7542  |

当前仓库内为占位小样本，用于跑通流程。获取完整数据并重新执行步骤 1–2 后，`validate_data.py` 会对照上表校验数量。
