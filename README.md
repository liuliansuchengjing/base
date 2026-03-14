# RecBole 序列推荐模型实验

本项目使用 RecBole 框架在两个数据集上运行序列推荐模型。

## 数据集

- **MOOPer**: 5000 用户, 3571 物品, 267604 交互
- **Assist09**: 3244 用户, 26635 物品, 520940 交互

## 数据格式转换

原始数据格式：
```
item_id timestamp correctness, item_id timestamp correctness, ...
```

已转换为 RecBole 标准格式：
- `dataset/mooper/mooper.inter` - 交互文件
- `dataset/mooper/mooper.user` - 用户文件  
- `dataset/mooper/mooper.item` - 物品文件

## 运行方式

### 1. 数据转换（已完成）
```bash
python convert_data.py
```

### 2. 运行推荐模型

#### 方式一：传统算法（无需 PyTorch）
```bash
python run_traditional.py
```

包含：
- Popularity（流行度推荐）
- ItemCF（物品协同过滤）

#### 方式二：深度学习模型（需要 PyTorch）
```bash
# 先修复 PyTorch DLL 问题
powershell -ExecutionPolicy Bypass -File fix_pytorch.ps1

# 运行模型
python run_simple.py
```

包含：
- **SASRec**: Self-Attentive Sequential Recommendation
- **GRU4Rec**: Session-based Recommendations with RNNs
- **SRGNN**: Session-based Recommendation with Graph Neural Networks

## 评估指标

- **Hit@K**: 命中率 (K=5, 10, 20)
- **NDCG@K**: 归一化折损累计增益 (K=5, 10, 20)
- **MRR@K**: 平均倒数排名 (K=5, 10, 20)

## 项目文件

```
.
├── cascades_MOOPer.txt          # 原始 MOOPer 数据
├── cascades_Assist09.txt        # 原始 Assist09 数据
├── convert_data.py              # 数据转换脚本
├── run_traditional.py           # 传统算法（无需 PyTorch）
├── run_simple.py                # RecBole 简化版
├── run_baseline.py              # RecBole 详细版
├── fix_pytorch.ps1              # PyTorch 修复脚本
├── requirements.txt             # 依赖列表
└── dataset/                     # 转换后的数据
    ├── mooper/
    │   ├── mooper.inter
    │   ├── mooper.user
    │   └── mooper.item
    └── assist09/
        ├── assist09.inter
        ├── assist09.user
        └── assist09.item
```

## 依赖安装

```bash
pip install recbole pandas numpy scipy torch
```

如果遇到 PyTorch DLL 问题，运行：
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 结果

运行完成后，结果将保存在 `results.pkl` 文件中，并打印汇总表格。

## 注意事项

1. 深度学习模型需要 PyTorch，Windows 系统可能遇到 DLL 加载问题
2. 推荐优先使用传统算法（run_traditional.py）获得基线结果
3. 所有模型默认使用 CPU 训练，如有 GPU 可修改配置中的 device 参数
