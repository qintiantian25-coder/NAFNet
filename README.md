# NAFNet 红外盲元图像修复

基于 Megvii Research 的 [NAFNet](https://github.com/megvii-research/NAFNet) (ECCV 2022 "Simple Baselines for Image Restoration")，针对**红外盲元连续帧图像修复**任务进行适配和优化。

## 项目结构

```
NAFNet/
├── basicsr/                  # 核心库（基于 BasicSR 修改）
│   ├── models/archs/         # 网络架构（NAFNet, Baseline, NAFSSR）
│   ├── models/losses/        # 损失函数（PSNRLoss, L1, MSE）
│   ├── data/                 # 数据集类（PairedImageDataset）
│   ├── metrics/              # 评估指标（PSNR, SSIM）
│   └── train.py, test.py     # 基础训练/测试入口
├── tools/
│   └── test_nafnet_blind.py  # 盲元评估测试脚本
├── main.py                   # 项目入口（训练/测试调度）
├── train_real.py             # 真实数据续训脚本
├── evaluate.py               # 离线指标计算（不需要模型）
├── experiment.cfg            # 仿真数据训练配置
├── experiment_real.cfg       # 真实数据续训配置
├── data_new/                 # 仿真红外盲元数据集
├── real_image/               # 真实红外盲元数据集
└── experiments/              # 仿真数据训练输出
```

## 环境安装

```bash
# Python 3.9+, PyTorch 1.11+, CUDA 11.3
pip install -r requirements.txt
pip uninstall basicsr -y          # 卸载系统安装的 BasicSR
python setup.py develop --no_cuda_ext  # 以开发模式安装项目的 BasicSR
```

如 `python setup.py develop` 失败（新版 pip/setuptools 兼容性问题），可直接将项目根目录添加到 PYTHONPATH：

```bash
export PYTHONPATH="/home/student_server/Qtt/NAFNet:$PYTHONPATH"
```

## 模型架构

**NAFNetLocal** — U-Net 编码器-解码器，使用 SimpleGate（通道分裂 + 逐元素乘法）替代所有激活函数：

- 宽度: 64
- 编码器块: [1, 1, 1, 28]（最深一层 28 个 NAFBlock）
- 中间块: 1
- 解码器块: [1, 1, 1, 1]
- 全局残差连接
- `Local_Base` mixin 支持任意尺寸输入（瓦片式推理）

参数量: ~67.9M

## 训练

### 1. 仿真数据训练（已完成）

```bash
python main.py --train --config_path experiment.cfg
# 从 checkpoint 恢复:
python main.py --train --config_path experiment.cfg --resume_state experiments/models/training_states/XXXX.state
```

训练配置：AdamW (lr=2e-4), CosineAnnealing, PSNRLoss, 200k iter, 256×256 patch

输出目录: `experiments/`

### 2. 真实数据续训

在仿真数据训练的 `best_model.pt` 基础上，使用 `real_image/` 数据集续训 100 轮：

```bash
python train_real.py
```

续训配置：AdamW (lr=1e-4), CosineAnnealing (T_max=7600), PSNRLoss, 256×256 patch

可以通过命令行调整参数：

```bash
python train_real.py --epochs 200 --device cuda          # 续训 200 轮
python train_real.py --resume                             # 从 latest.pt 恢复续训
python train_real.py --config experiment_real.cfg --epochs 50
```

输出目录: `experiments_real/`

```
experiments_real/
├── models/
│   ├── best_model.pt       # 最佳模型权重（验证集 PSNR 最高时更新）
│   └── latest.pt           # 最新训练状态（模型+优化器+调度器，每轮保存）
└── logs/
    ├── train.txt            # 每轮训练损失
    └── val.txt              # 每轮验证指标（PSNR, SSIM）
```

**断点续训机制：** 训练意外中断后，运行 `python train_real.py --resume` 从 `latest.pt` 恢复训练状态继续。

## 测试

### 模型推理 + 盲元评估

```bash
python main.py --test --config_path experiment.cfg
```

读取 `experiment.cfg` 的 `test_runner` 配置段，调用 `tools/test_nafnet_blind.py` 进行：
- 全图 PSNR / SSIM 评估
- 盲元位置 MAE / RMSE / PSNR 评估（合并静态盲元 + 逐帧闪光盲元）
- 输出三合一对比图（输入 | 输出 | 真值）和逐帧 CSV 指标

### 续训模型测试

修改 `experiment.cfg` 中 `test_runner.checkpoint` 为 `experiments_real/models/best_model.pt`，然后运行上述测试命令。

### 离线指标计算（不需要模型）

编辑 `evaluate.py` 顶部的路径配置区域，然后：

```bash
python evaluate.py
```

从已保存的输出 PNG 重新计算 PSNR / SSIM / 盲元指标。

## 数据集结构

### 仿真数据 (`data_new/`)

```
data_new/
├── train_blur/     # 训练集模糊图像 (按序列分文件夹 001/, 002/, ...)
├── train_sharp/    # 训练集真值图像
├── train_mask/     # 训练集盲元坐标 CSV
├── val_blur/       # 验证集
├── val_sharp/
├── val_mask/
├── test_blur/      # 测试集
├── test_sharp/
└── test_mask/
```

### 真实数据 (`real_image/`)

```
real_image/
├── train_blur/      # 6 个序列 (001–006), 301 帧 512×640 灰度 PNG
├── train_sharp/     # 训练集真值（无盲元原始帧）
├── train_mask/      # 盲元标注 (blind_pixel_coords.csv)
├── val_blur/        # 3 个序列 (001–003), 152 帧
├── val_sharp/
├── val_mask/
├── test_blur/       # 3 个序列 (001–003), 151 帧
├── test_sharp/
└── test_mask/
```

每帧的 `*_mask/` 目录中包含 `blind_pixel_coords.csv`（静态盲元坐标）和 `flash_pixel_coords.csv`（逐帧闪光盲元坐标，可选）。

## 关键技术点

- **SimpleGate**: NAFBlock 的核心非线性操作，将特征通道一分为二并逐元素相乘 (`x1 * x2`)，替代传统的 ReLU/GELU
- **Simplified Channel Attention (SCA)**: 仅使用一个 Conv1x1 实现通道注意力（无降维/升维，无 Sigmoid）
- **LayerNorm2d**: 自定义 CUDA 优化的 2D LayerNorm（对每个空间位置沿通道维度归一化）
- **Tile-based Inference**: `Local_Base` mixin 替换 `AdaptiveAvgPool2d` 为支持任意输入尺寸的自定义平均池化
