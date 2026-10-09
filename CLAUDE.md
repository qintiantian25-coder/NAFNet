# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NAFNet (Nonlinear Activation Free Network) — ECCV 2022 paper "Simple Baselines for Image Restoration" by Megvii Research. Replaces all activation functions (ReLU, GELU, sigmoid) with multiplication via **SimpleGate** (channel split + element-wise multiply: `x1 * x2`) plus Simplified Channel Attention. Built on a forked BasicSR framework.

This fork is adapted for **blind-pixel image restoration** on grayscale imagery with custom evaluation metrics.

## Install

```bash
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

Python 3.9.5, PyTorch 1.11.0, CUDA 11.3.

## Commands

### Training

```bash
python main.py --train --config_path experiment.cfg
# Resume from checkpoint:
python main.py --train --config_path experiment.cfg --resume_state experiments/models/training_states/XXX.state
# Multi-GPU:
python main.py --train --config_path experiment.cfg --nproc_per_node 4
```

This dispatches to `torch.distributed.launch basicsr/train.py -opt experiment.cfg`. The config drives everything: model architecture, dataset paths, optimizer, scheduler, validation cadence.

### Testing (inference + metrics)

```bash
python main.py --test --config_path experiment.cfg
```

Reads `test_runner` section from `experiment.cfg` and launches `tools/test_nafnet_blind.py` with the appropriate arguments. This runs the model on test images, saves output PNGs and triple-comparison images (input|output|GT), and computes per-image and per-group blind-pixel metrics.

### Post-hoc evaluation (no model needed)

```bash
python evaluate.py
```

Edit the paths in the config block at the top of the script first. Recomputes PSNR, SSIM, and blind-pixel MAE/RMSE/PSNR from saved output PNGs against GT — does not load the model.

### Real-data fine-tuning (standalone — does NOT go through main.py)

```bash
python train_real.py                          # 100 epochs, starts from experiments/models/best_model.pt
python train_real.py --epochs 200 --device cuda
python train_real.py --resume                 # resume from experiments_real/models/latest.pt
python train_real.py --config experiment_real.cfg
```

`train_real.py` has its own training loop (no main.py/BasicSR framework). It loads the sim-trained `best_model.pt` into `NAFNetLocal` with `train_size=(1,3,256,256)`, trains on `real_image/` with AdamW (lr=1e-4), `CosineAnnealingLR` (T_max=7600, stepped per iter), PSNRLoss (MSE), grad-clip 0.01, 256×256 crops. Key checkpoint semantics: on improving val PSNR, *both* `best_model.pt` and `latest.pt` are overwritten, so `latest.pt` = full state of the best epoch (model+optimizer+scheduler) — `--resume` continues from best, not last epoch. Logs: `experiments_real/logs/{train,val}.txt`, one line per epoch.

### No-reference evaluation (no GT, no masks)

```bash
python evaluate_nr.py
```

Pure math on saved output PNGs versus input (edit paths at top: `OUTPUT_DIR`, `INPUT_DIR`, `SAVE_DIR`). Per-image/per-sequence metrics with `*_out` / `*_in` columns, written to `nr_metrics.csv`: `residual_{5,10,20,30,50}` (% pixels deviating >X gray levels from 5×5 median, ↓), `localstd` (mean 5×5 local std, ↓), `estsnr` (Immerkaer noise-estimate SNR, ↑). Tuned for grayscale infrared — useful when GT/mask CSVs are unavailable (e.g. on `LWIR_test`).

### Direct test script invocation

```bash
python tools/test_nafnet_blind.py \
    --data_root /path/to/data_new \
    --checkpoint experiments/models/best_model.pt \
    --in_chans 3 \
    --save_dir results/NAFNet_test \
    --test_mask_csv /path/to/test_mask \
    --width 64 --enc_blk_nums 1,1,1,28 --middle_blk_num 1 --dec_blk_nums 1,1,1,1
```

## Architecture

### Core model: `basicsr/models/archs/NAFNet_arch.py`

U-Net encoder-decoder with NAFBlocks as the building block. Each NAFBlock contains two sub-blocks, each: `LayerNorm2d → Conv1x1(expand) → Conv3x3(depthwise) → SimpleGate(split+multiply) → SCA → Conv1x1(project)` with a learned residual scale (beta/gamma).

- **SimpleGate**: splits channels in half along dim=1 and returns `x1 * x2` — the only nonlinearity.
- **Simplified Channel Attention (SCA)**: `AdaptiveAvgPool2d(1) → Conv1x1` only, no reduction ratio, no sigmoid. Output multiplied element-wise with features.
- **NAFNetLocal** (used here): NAFNet wrapped with `Local_Base` mixin from `local_arch.py` that replaces `AdaptiveAvgPool2d` with a custom `AvgPool2d` supporting arbitrary input sizes for tile-based inference.

The config uses: width=64, encoder blocks [1,1,1,28] (deepest at bottom), middle=1, decoder [1,1,1,1].

### Model architecture classes hierarchy

- `NAFNet` — pure network, fixed input sizes
- `NAFNetLocal(NAFNet, Local_Base)` — tile-friendly variant used in this project
- `Baseline` — comparison architecture with GELU + standard SE attention (in `Baseline_arch.py`)

### Training framework: `basicsr/models/image_restoration_model.py`

`ImageRestorationModel` extends `BaseModel`. Key methods:
- `optimize_parameters()` — one training step (forward + loss + backward)
- `validation()` — PSNR/SSIM on val set, triggers `best_model.pt` save
- `test()` — inference with optional grid-based tiling for large images

The loss is `PSNRLoss` which is actually MSE (minimizing MSE ≡ maximizing PSNR). The optimizer is AdamW with CosineAnnealingLR.

### Dataset: `basicsr/data/paired_image_dataset.py`

`PairedImageDataset` loads LQ/GT pairs from disk folders with grouped subfolder structure (e.g., `train_blur/001/...`, `train_sharp/001/...`). Training applies random 256×256 crops; testing reads full images.

### Data flow for this project

```
data_new/
  train_blur/     → LQ (blurred/noisy input)
  train_sharp/    → GT (clean target)
  train_mask/     → CSV masks (blind_pixel_coords.csv, flash_pixel_coords.csv)
  val_blur/       → validation LQ
  val_sharp/      → validation GT
  test_blur/      → test LQ
  test_sharp/     → test GT
  test_mask/      → per-group CSVs for blind/flash pixel evaluation
real_image/         → real-infrared data for fine-tuning (same grouped layout, 512×640 grayscale PNGs)
LWIR_test/          → raw inputs for the no-reference evaluator (evaluate_nr.py); only test_blur/
real_image_test/    → real test sequences (001, 002, 003) for final evaluation
```

Images are organized in numbered subfolders (001/, 002/, ...) representing sequences. Mask CSVs contain pixel coordinates of known defective (blind) pixels and per-frame flash pixels.

### Custom LayerNorm2d: `basicsr/models/archs/arch_util.py`

CUDA-optimized LayerNorm over channel dimension using `torch.autograd.Function`. Unlike standard LayerNorm (which normalizes over last dims), this normalizes over C for each spatial position.

### Evaluation pipeline

`tools/test_nafnet_blind.py` loads the model directly (bypasses BasicSR's model system), instantiates `NAFNetLocal`, runs inference, and computes:
- **Global**: PSNR, SSIM per image
- **Blind-pixel**: MAE, RMSE, PSNR at known blind/flash pixel coordinates (merged from static blind CSV + per-frame flash CSV)
- **Input baseline**: same blind-pixel metrics on the input image for gain calculation

`evaluate.py` does the same metrics but operates on already-saved output PNGs — no model loaded.

## Configuration

`experiment.cfg` is the single source of truth. It uses BasicSR's YAML schema with these project-specific additions:
- `test_runner` section — configures which test script and arguments `main.py --test` launches
- `train_schedule` section — documents epoch estimates (informational only)
- Blind-pixel mask paths are resolved relative to `data_root` automatically

## Key differences from upstream NAFNet

1. **Custom entry point**: `main.py` wraps training/testing dispatch instead of calling `basicsr/train.py` directly
2. **Blind-pixel evaluation**: `tools/test_nafnet_blind.py` and `evaluate.py` are project-specific additions
3. **Grayscale focus**: model input is 3-channel (BGR from OpenCV) but output is reduced to grayscale for metrics
4. **Grouped dataset structure**: images organized in sequence subfolders with per-sequence mask CSVs
5. **No perceptual loss or GAN**: training uses only PSNRLoss (MSE)
6. **Real-data fine-tuning**: `train_real.py` + `experiment_real.cfg` form a standalone second-stage pipeline (outputs to `experiments_real/`), separate from the BasicSR-driven `main.py` workflow (outputs to `experiments/`)
