# DATASET MANIFEST —— `data_block_v1`（Block-v1 正式数据）

> 2026-10-08 用户批准正式生成。生成器 **FINAL FROZEN**，本数据集由冻结入口产出，
> 生成后**未做任何人工筛选**。审计见 `audit_formal_dataset.json`（全部 PASS）。

## 1. 来源与版本

| | |
|---|---|
| generator | `simulation/block_macrocompact_generator.py` |
| — SHA256 | `c4ab35bab7580c1f04c787d0d671a114c7b895a8924b4f8a7c242d992f0bae56` |
| 正式入口 | `simulation/add_block_macrocompact_dataset.py` |
| — SHA256 | `48e68ef8dee83b662a2f32a76dfbd02a59bfb4f231731560b2d4fac299387d96` |
| 审计脚本 | `diagnostics/audit_block_v1_dataset.py` |
| — SHA256 | `c824ac9a4b79728d2a528f17e70dafc22ed5cafcafaf7c92339ff5049782237b` |
| git HEAD | `cb214c253ad4b923d36c38cee6ee054e7f791e62` |
| 权威外观清单 | `results/block_macrocompact_crop_check/SELECTED_PROTOTYPE_MANIFEST.md` |
| 生成时间 | 2026-10-08 |

## 2. Split seed scheme（**禁止复用同一 mask**）

```
SEED_BASE = 20261011
SPLIT_NAMESPACE = {'train': 0, 'val': 100000, 'test': 200000}
seed(split, seq) = SEED_BASE + SPLIT_NAMESPACE[split] + int(seq)
```
生成器侧：geometry / response 各由 `np.random.default_rng(seed)` 派生出的独立流驱动；
`random.seed(seed)` 一次用于 `get_mostly_black_color()`。
三个 split 的 seed 取值两两不相交；已断言 **15 个 sequence 的 mask SHA256 互不相同**
（`checks.no_duplicate_mask_across_splits`）。

## 3. 每个 sequence 的帧数与 component 数

| split/seq | 帧数 | components | S/M/L | 盲元像素 | 盲元比 |
|---|---|---|---|---|---|
| test/001 | 270 | 30 | 10/10/10 | 10783 | 3.29% |
| test/002 | 54 | 30 | 10/10/10 | 10753 | 3.28% |
| test/003 | 47 | 30 | 10/10/10 | 11171 | 3.41% |
| test/004 | 42 | 30 | 10/10/10 | 11182 | 3.41% |
| test/005 | 39 | 30 | 10/10/10 | 10916 | 3.33% |
| test/006 | 33 | 30 | 10/10/10 | 11980 | 3.66% |
| train/001 | 270 | 30 | 10/10/10 | 10298 | 3.14% |
| train/002 | 270 | 30 | 10/10/10 | 10368 | 3.16% |
| train/003 | 270 | 30 | 10/10/10 | 10418 | 3.18% |
| train/004 | 79 | 30 | 10/10/10 | 10759 | 3.28% |
| train/005 | 70 | 30 | 10/10/10 | 10563 | 3.22% |
| train/006 | 67 | 30 | 10/10/10 | 10808 | 3.30% |
| train/007 | 59 | 30 | 10/10/10 | 11291 | 3.45% |
| val/001 | 270 | 30 | 10/10/10 | 11187 | 3.41% |
| val/002 | 54 | 30 | 10/10/10 | 11208 | 3.42% |

**train / val 严格 30 components = 10 S + 10 M + 10 L（1:1:1）**；
test 同为 30，其 4 个 severity 子集见 §6。

## 4. Block-S / M / L 定义与冻结约束

| | |
|---|---|
| `Block-S` | `64 <= area <= 168` |
| `Block-M` | `169 <= area <= 440` |
| `Block-L` | `441 <= area <= 900` |
| 尺度量 | `equivalent_side = sqrt(area)` |
| `MAX_ELONGATION` | **1.50** |
| `bbox_diag / sqrt(area)` | `<= 2.0` |
| `compactness` | `>= 0.50` |
| `max_depth` | `>= 2` |
| 连通性 | `connectivity = 8`，单一组件，`hole_count == 0` |
| depth 口径 | `cv2.distanceTransform(mask, cv2.DIST_L2, 5)`（historical 5×5 L2 approximation） |

## 5. Radiometric 冻结参数

```
亮异常 = 255
暗异常 = get_mostly_black_color()   (0-15)
f_dark ~ U(0.3, 0.4)
暗簇尺寸 = (2, max(3, round(0.3 * sqrt(area))))
种子最小间距 = max(3, round(0.13 * sqrt(area)))  （超限时 +2 重试，最多 3 次）
单个暗簇 <= 0.12 * area
边缘抖动 RIM_DITHER = 0.18 （重试时 0.12 / 0.06）
相关场相关长度 CORR_LEN = 3.0 px
小于 MIN_DARK_CLUSTER = 2 的暗连通域翻回亮
```

**跨帧固定**：同 sequence 内 geometry 坐标、bright/dark 标签、模拟灰度值全部固定
（`checks.cross_frame_mask_and_response_fixed` 15/15 PASS）。

## 6. Quantity severity 嵌套定义（**训练期间不得修改**）

```
Q06 = 2S + 2M + 2L   ⊂  Q12 = 4S + 4M + 4L  ⊂  Q18 = 6S + 6M + 6L  ⊂  Q30 = 10S + 10M + 10L
```
切法：在 Q30 的 `placed` 顺序中，**每个尺度档内取前 k 个** component（k = 2/4/6/10）。
⇒ 天然嵌套，共享 component 的 geometry / response 标签 / 模拟灰度值 / metadata 逐位相同，
**绝不重抽**。目录 `data_block_v1/test_severity/<level>/test_{blur,mask}/<seq>/`。

| 档 | n_blocks | S/M/L | 盲元像素(均值) | blind_ratio | sequences |
|---|---|---|---|---|---|
| **Q06** | 6 | 2/2/2 | 2204 | 0.67% | 6 |
| **Q12** | 12 | 4/4/4 | 4410 | 1.35% | 6 |
| **Q18** | 18 | 6/6/6 | 6635 | 2.02% | 6 |
| **Q30** | 30 | 10/10/10 | 11130 | 3.40% | 6 |

## 7. 形态族核验（只做分布核验，未据此调整 generator）

| 指标 | 选定原型 seed 20261008 (n=15) | final check seed 20261012 (n=120) | **data_block_v1 (n=450)** |
|---|---|---|---|
| `compactness` | 0.700 (0.570–0.752) | 0.708 (0.562–0.797) | **0.719** (0.506–0.851) |
| `solidity` | 0.930 (0.841–0.968) | 0.936 (0.772–0.973) | **0.940** (0.736–0.978) |
| `circularity` | 0.752 (0.588–0.809) | 0.763 (0.487–0.862) | **0.777** (0.472–0.882) |
| `perimeter_over_hull` | 1.077 (1.031–1.123) | 1.062 (1.024–1.161) | **1.061** (1.010–1.181) |
| `elongation` | 1.167 (1.000–1.412) | 1.133 (1.000–1.500) | **1.096** (1.000–1.500) |
| `max_depth` | 9.000 (4.394–13.197) | 7.991 (3.597–15.000) | **8.197** (3.197–15.591) |
| `dark_ratio` | 0.192 (0.121–0.256) | 0.190 (0.089–0.296) | **0.177** (0.086–0.441) |

`dark_ratio`: mean **0.1971** / median
0.1769 / p05 0.1216
/ p95 0.3580

## 8. 监督覆盖率（供下一阶段 loss / training audit 用）

```
|M_block_GT| / |GTblind|              = 0.7719
|GTblind ∧ ¬M_block_GT| / |GTblind|   = 0.2281
    （M_block_GT = 1[area>=21] · 1[depth>=2]；Block-only 组件全部 area>=64 ⇒ 实际由 depth>=2 决定）
128x128 random crops (train, n=5000, seed 20261013):
    P(|M|=0)            = 0.2248
    interior/crop  mean 672.2  median 464
                   p05 0  p95 2072
    boundary/crop       = 198.3
    nonblind/crop       = 15513.5
```

## 9. 审计结论

`audit_formal_dataset.json` —— **19 项检查全部 PASS**（逐 component 450、逐 sequence 15、
severity 嵌套 18 组）。本数据集**训练期间不得修改**；quantity severity 档不得重选。

**未创建 trainer、未启动训练。**
