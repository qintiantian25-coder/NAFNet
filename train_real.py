#!/usr/bin/env python3
"""
真实红外图像盲元数据集 - 续训脚本
从仿真数据训练的 best_model.pt 加载权重，在 real_image 数据集上续训 100 轮。

保存结构:
  experiments_real/
    models/
      best_model.pt      # 最佳模型权重（验证集 PSNR 最高时更新）
      latest.pt           # 最佳模型训练状态（权重+优化器+调度器，和 best_model.pt 同步更新）
    logs/
      train.txt           # 每轮训练损失
      val.txt             # 每轮验证指标

用法:
  python train_real.py                          # 从 config 指定的预训练权重开始
  python train_real.py --resume                 # 从 experiments_real/models/latest.pt 续训（最佳模型状态）
  python train_real.py --config experiment_real.cfg
"""

import argparse
import math
import os
import sys
import time
import logging
import yaml
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 确保项目根目录在 sys.path 中
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from basicsr.data import create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher
from basicsr.models.archs.NAFNet_arch import NAFNetLocal
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from basicsr.utils import imfrombytes, img2tensor
from basicsr.utils.options import ordered_yaml


# ===========================================================================
# 配置加载
# ===========================================================================

def load_config(config_path):
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    # 扩展路径
    for phase, dataset in opt.get('datasets', {}).items():
        dataset['phase'] = phase.split('_')[0]
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt'):
            dataset['dataroot_gt'] = os.path.expanduser(dataset['dataroot_gt'])
        if dataset.get('dataroot_lq'):
            dataset['dataroot_lq'] = os.path.expanduser(dataset['dataroot_lq'])

    if opt['path'].get('pretrain_network_g'):
        opt['path']['pretrain_network_g'] = os.path.expanduser(
            opt['path']['pretrain_network_g'])

    return opt


# ===========================================================================
# 日志工具
# ===========================================================================

def setup_logging(save_dir):
    """设置日志：控制台 + 文件"""
    os.makedirs(save_dir, exist_ok=True)

    log_format = '%(asctime)s | %(levelname)s | %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    logger = logging.getLogger('train_real')
    return logger


class EpochLogger:
    """记录每轮训练损失和验证指标到文件"""

    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.train_path = os.path.join(log_dir, 'train.txt')
        self.val_path = os.path.join(log_dir, 'val.txt')

    def log_train(self, line):
        with open(self.train_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    def log_val(self, line):
        with open(self.val_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')


# ===========================================================================
# 数据集构建
# ===========================================================================

def build_dataloaders(opt):
    """构建训练集和验证集 DataLoader"""
    logger = logging.getLogger('train_real')

    # 训练集
    train_opt = opt['datasets']['train']
    train_opt['phase'] = 'train'
    if 'scale' in opt:
        train_opt['scale'] = opt['scale']
    train_set = create_dataset(train_opt)

    train_sampler = EnlargedSampler(
        train_set, 1, 0,
        train_opt.get('dataset_enlarge_ratio', 1)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=train_opt['batch_size_per_gpu'],
        shuffle=False,
        num_workers=train_opt.get('num_worker_per_gpu', 4),
        pin_memory=True,
        sampler=train_sampler,
        drop_last=False,
    )

    # 验证集
    val_opt = opt['datasets']['val']
    val_opt['phase'] = 'val'
    if 'scale' in opt:
        val_opt['scale'] = opt['scale']
    val_set = create_dataset(val_opt)
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    num_iters_per_epoch = math.ceil(
        len(train_set) * train_opt.get('dataset_enlarge_ratio', 1) /
        train_opt['batch_size_per_gpu']
    )

    logger.info(f'训练集图像数: {len(train_set)}')
    logger.info(f'验证集图像数: {len(val_set)}')
    logger.info(f'每轮迭代数: {num_iters_per_epoch}')
    logger.info(f'Batch Size: {train_opt["batch_size_per_gpu"]}')

    return train_loader, train_sampler, val_loader, num_iters_per_epoch


# ===========================================================================
# 模型构建 + 权重加载
# ===========================================================================

def build_model(opt, device):
    """构建 NAFNetLocal 并加载预训练权重"""
    logger = logging.getLogger('train_real')

    net_opt = opt['network_g']
    model = NAFNetLocal(
        img_channel=3,
        width=net_opt['width'],
        enc_blk_nums=list(net_opt['enc_blk_nums']),
        middle_blk_num=net_opt['middle_blk_num'],
        dec_blk_nums=list(net_opt['dec_blk_nums']),
        train_size=(1, 3, 256, 256),
    )

    # 加载预训练权重
    pretrain_path = opt['path'].get('pretrain_network_g')
    if pretrain_path and os.path.exists(pretrain_path):
        logger.info(f'加载预训练权重: {pretrain_path}')
        ckpt = torch.load(pretrain_path, map_location='cpu')
        # 兼容不同格式
        state = ckpt.get('params', ckpt.get('model', ckpt))
        if isinstance(state, dict) and 'params' not in state:
            # 可能是直接的 state_dict
            pass
        else:
            state = state

        # 去掉 module. 前缀
        if isinstance(state, dict):
            new_state = OrderedDict()
            for k, v in state.items():
                new_k = k[7:] if k.startswith('module.') else k
                new_state[new_k] = v
            state = new_state

        model.load_state_dict(state, strict=opt['path'].get('strict_load_g', True))
        logger.info('预训练权重加载成功')
    else:
        logger.warning(f'未找到预训练权重: {pretrain_path}，将随机初始化')

    model = model.to(device)
    return model


# ===========================================================================
# 优化器 + 调度器
# ===========================================================================

def build_optimizer_and_scheduler(model, opt):
    """构建优化器和学习率调度器"""
    train_opt = opt['train']
    optim_opt = train_opt['optim_g'].copy()

    optim_type = optim_opt.pop('type')
    if optim_type == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), **optim_opt)
    elif optim_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), **optim_opt)
    else:
        raise ValueError(f'不支持的优化器: {optim_type}')

    sched_opt = train_opt['scheduler'].copy()
    sched_type = sched_opt.pop('type')
    if sched_type == 'TrueCosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **sched_opt)
    else:
        raise ValueError(f'不支持的调度器: {sched_type}')

    return optimizer, scheduler


# ===========================================================================
# 损失函数
# ===========================================================================

class PSNRLoss(torch.nn.Module):
    """PSNR Loss = MSE（最小化 MSE 等同于最大化 PSNR）"""

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        if self.reduction == 'mean':
            loss = F.mse_loss(pred, target)
        else:
            loss = F.mse_loss(pred, target, reduction=self.reduction)
        return self.loss_weight * loss


# ===========================================================================
# 验证
# ===========================================================================

@torch.no_grad()
def validate(model, val_loader, device):
    """在验证集上计算 PSNR 和 SSIM"""
    model.eval()

    psnr_vals = []
    ssim_vals = []

    for val_data in tqdm(val_loader, desc='验证', leave=False):
        lq = val_data['lq'].to(device)
        gt = val_data['gt'].to(device)

        # 前向推理
        out = model(lq)
        out = out.clamp(0, 1)

        # 转为 numpy (H, W, C) uint8 用于指标计算
        out_np = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        gt_np = gt.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # 归一化到 [0, 255]
        out_np = (out_np * 255.0).round().clip(0, 255).astype(np.uint8)
        gt_np = (gt_np * 255.0).round().clip(0, 255).astype(np.uint8)

        try:
            psnr = calculate_psnr(out_np, gt_np, crop_border=0, input_order='HWC')
            ssim = calculate_ssim(out_np, gt_np, crop_border=0, input_order='HWC')
            psnr_vals.append(float(psnr))
            ssim_vals.append(float(ssim))
        except Exception:
            continue

    model.train()

    if len(psnr_vals) == 0:
        return 0.0, 0.0

    avg_psnr = np.mean(psnr_vals)
    avg_ssim = np.mean(ssim_vals)
    return avg_psnr, avg_ssim


# ===========================================================================
# 检查点保存 / 恢复
# ===========================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, best_psnr, save_dir):
    """保存完整训练状态到 latest.pt"""
    os.makedirs(save_dir, exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_psnr': best_psnr,
    }
    path = os.path.join(save_dir, 'latest.pt')
    torch.save(state, path)


def save_best_model(model, save_dir):
    """保存最佳模型权重到 best_model.pt"""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'best_model.pt')
    torch.save({'params': model.state_dict()}, path)


def load_checkpoint(model, optimizer, scheduler, save_dir, device):
    """从 latest.pt 恢复训练状态，返回 (epoch, best_psnr)"""
    path = os.path.join(save_dir, 'latest.pt')
    if not os.path.exists(path):
        return 0, float('-inf')

    logger = logging.getLogger('train_real')
    logger.info(f'从 {path} 恢复训练状态')
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    epoch = ckpt['epoch']
    best_psnr = ckpt.get('best_psnr', float('-inf'))

    logger.info(f'恢复: epoch={epoch}, best_psnr={best_psnr:.4f}')
    return epoch, best_psnr


# ===========================================================================
# 训练主循环
# ===========================================================================

def train_one_epoch(model, train_loader, train_sampler, optimizer, scheduler,
                    criterion, device, epoch, total_epochs, epoch_logger):
    """训练一轮，返回平均损失。scheduler 每 iter 步进一次。"""
    logger = logging.getLogger('train_real')
    model.train()

    train_sampler.set_epoch(epoch)

    total_loss = 0.0
    num_batches = 0
    epoch_start = time.time()

    prefetcher = CPUPrefetcher(train_loader)
    train_data = prefetcher.next()

    while train_data is not None:
        lq = train_data['lq'].to(device)
        gt = train_data['gt'].to(device)

        optimizer.zero_grad()
        pred = model(lq)
        if isinstance(pred, list):
            pred = pred[-1]
        loss = criterion(pred, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        optimizer.step()
        scheduler.step()  # CosineAnnealingLR 每 iter 步进

        total_loss += loss.item()
        num_batches += 1
        train_data = prefetcher.next()

    avg_loss = total_loss / max(num_batches, 1)
    elapsed = time.time() - epoch_start

    # 记录到 train.txt
    current_lr = optimizer.param_groups[0]['lr']
    log_line = (f'第 {epoch:3d}/{total_epochs} 轮 | '
                f'训练损失: {avg_loss:.8f} | '
                f'学习率: {current_lr:.2e} | '
                f'耗时: {elapsed:.1f}s')
    logger.info(log_line)
    epoch_logger.log_train(log_line)

    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='真实红外图像盲元数据集续训')
    parser.add_argument('--config', type=str, default='experiment_real.cfg',
                        help='配置文件路径')
    parser.add_argument('--resume', action='store_true',
                        help='从 latest.pt 续训')
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备')
    parser.add_argument('--epochs', type=int, default=100,
                        help='续训轮数')
    args = parser.parse_args()

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 加载配置
    config_path = os.path.join(REPO_ROOT, args.config)
    if not os.path.isabs(args.config):
        config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        print(f'配置文件不存在: {config_path}')
        sys.exit(1)

    opt = load_config(config_path)

    # 输出目录
    save_root = os.path.join(REPO_ROOT, 'experiments_real')
    models_dir = os.path.join(save_root, 'models')
    logs_dir = os.path.join(save_root, 'logs')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # 日志
    logger = setup_logging(logs_dir)
    epoch_logger = EpochLogger(logs_dir)
    logger.info(f'配置: {config_path}')
    logger.info(f'设备: {device}')
    logger.info(f'输出目录: {save_root}')
    logger.info(f'续训轮数: {args.epochs}')

    # 数据集
    train_loader, train_sampler, val_loader, num_iters_per_epoch = build_dataloaders(opt)

    # 模型
    model = build_model(opt, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'模型参数量: {total_params:,} (可训练: {trainable_params:,})')

    # 优化器 & 调度器
    optimizer, scheduler = build_optimizer_and_scheduler(model, opt)

    # 损失函数
    criterion = PSNRLoss(loss_weight=1.0, reduction='mean')

    # 续训 / 首次训练
    best_psnr = float('-inf')
    start_epoch = 0

    if args.resume:
        start_epoch, best_psnr = load_checkpoint(model, optimizer, scheduler,
                                                  models_dir, device)
        logger.info(f'续训模式: 从第 {start_epoch + 1} 轮继续')
    else:
        logger.info('首次训练: 从预训练权重开始')

    total_epochs = args.epochs

    # 写入日志表头
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    epoch_logger.log_train(f'# 续训开始: {now} | 总轮数: {total_epochs} | '
                           f'训练集: {len(train_loader.dataset)} 张 | '
                           f'图像尺寸: 512×640')
    epoch_logger.log_val(f'# 续训开始: {now} | 总轮数: {total_epochs} | '
                         f'验证集: {len(val_loader.dataset)} 张 | '
                         f'监控指标: PSNR')

    # =======================================================================
    # 训练循环
    # =======================================================================
    logger.info(f'===== 开始续训，共 {total_epochs} 轮 =====')
    total_start = time.time()

    for epoch in range(start_epoch + 1, total_epochs + 1):
        # 训练一轮（scheduler 在每 iter 步进）
        train_loss = train_one_epoch(
            model, train_loader, train_sampler, optimizer, scheduler,
            criterion, device, epoch, total_epochs, epoch_logger
        )

        # 验证
        logger.info(f'第 {epoch}/{total_epochs} 轮验证中 ...')
        val_start = time.time()
        val_psnr, val_ssim = validate(model, val_loader, device)
        val_time = time.time() - val_start

        # 记录验证结果
        is_best = val_psnr > best_psnr
        best_marker = ' (*) 新最佳!' if is_best else ''

        val_line = (f'第 {epoch:3d}/{total_epochs} 轮 | '
                    f'PSNR: {val_psnr:.4f} dB | '
                    f'SSIM: {val_ssim:.6f} | '
                    f'最佳 PSNR: {max(val_psnr, best_psnr):.4f} dB | '
                    f'验证耗时: {val_time:.1f}s{best_marker}')
        logger.info(val_line)
        epoch_logger.log_val(val_line)

        # 更新最佳模型 + 保存训练状态（只保留最佳模型的完整状态）
        if is_best:
            best_psnr = val_psnr
            save_best_model(model, models_dir)
            save_checkpoint(model, optimizer, scheduler, epoch, best_psnr, models_dir)
            logger.info(f'  -> 已保存最佳模型: best_model.pt (PSNR={best_psnr:.4f} dB)')
            logger.info(f'  -> 已更新训练状态: latest.pt（用于断点续训）')

    # =======================================================================
    # 训练结束
    # =======================================================================
    total_time = time.time() - total_start
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    logger.info(f'===== 续训完成! 总耗时: {hours}h {minutes}m {seconds}s =====')
    logger.info(f'最佳 PSNR: {best_psnr:.4f} dB')
    logger.info(f'最佳模型: {models_dir}/best_model.pt')
    logger.info(f'训练状态: {models_dir}/latest.pt')

    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    epoch_logger.log_train(f'# 续训结束: {end_time} | 最佳 PSNR: {best_psnr:.4f} dB')
    epoch_logger.log_val(f'# 续训结束: {end_time} | 最佳 PSNR: {best_psnr:.4f} dB')


if __name__ == '__main__':
    main()
