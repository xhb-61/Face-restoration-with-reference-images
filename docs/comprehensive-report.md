# 参考图像引导的人脸恢复项目综合报告

本文汇总以下两个实验记录，并整理成适合项目展示和后续复现的综合报告：

- `C:\Users\11654\Desktop\华为人脸恢复问题研究.md`
- `E:\concert\1\ref-face\FACEME_REF_GENERATION_EXPERIMENT_REPORT.md`

## 1. 项目目标

项目围绕华为 `concert` 数据集中的低质量人脸恢复问题展开。数据集中共有 6 类身份，每类包含低质量输入图像 `LR` 和同身份参考图 `ref`。目标是在参考图辅助下恢复低质量人脸，使输出同时满足：

- 人脸结构稳定，不出现明显五官错位、脸型拉伸或缝合伪影。
- 身份特征尽量接近参考人脸。
- 对舞台光照、遮挡、运动模糊、极低分辨率等退化具备一定鲁棒性。
- 形成可复现实验流程，便于后续筛选最优参数和定位失败来源。

## 2. 数据与主要困难

原始记录中先对数据进行了人脸裁剪：`LR` 主要由人工裁剪，`ref` 主要通过 `face_recognition` 裁剪。实验发现，直接把原图送入 Ref-LDM 效果很差；即使裁剪后输入，极度模糊的 LR 图仍容易导致结构错位和伪影。

主要困难包括：

1. 部分 LR 图本身过糊，五官位置和脸部轮廓已经不可靠。
2. 参考图数量有限，姿态、表情、光照覆盖不足。
3. 原始 Ref-LDM noise2img 流程从随机 latent 开始生成，结构约束较弱。
4. 直接把非正方形脸图 resize 到 `512x512` 会改变脸部比例。
5. 过高的参考图引导强度会强化身份相似度，但也可能引入结构冲突和缝合伪影。

## 3. Ref-LDM 实验路线

### 3.1 初始实验

`2026-04-09`，直接将原图输入 Ref-LDM，结果保存在 `result_concert_i_saved`。由于未做人脸裁剪，背景和非对齐人脸严重干扰模型，结果较差。

`2026-04-12`，对整个数据集重新做人脸裁剪，并将裁剪后的人脸图作为 Ref-LDM 输入。结果相比原图输入更稳定，但对于 `1`、`2`、`6` 等本身很糊的 LR 图，仍出现结构错位和伪影。

### 3.2 CFG Scale 分析

实验对 `cfg_scale` 进行了重点分析。该参数控制模型对参考图的依赖程度：

- `cfg_scale` 较高时，模型更倾向于参考图身份特征，但容易在 LR 结构不足时产生错位、伪影或过度锐化。
- `cfg_scale` 较低时，结构崩坏风险下降，但结果可能更模糊，身份相似度不足。

因此，CFG 搜索的核心目标不是单纯追求最大值，而是在“身份保持”和“结构稳定”之间寻找临界平衡点。

### 3.3 两阶段策略

针对极低质量 LR 图，报告中提出并验证了两阶段策略：

1. **Stage 1: GFPGAN/CodeFormer 结构重建**  
   先用无参考盲修复模型恢复较稳定的人脸结构。该阶段不一定能保持目标身份，但通常能减少严重五官错位。
2. **Stage 2: Ref-LDM 身份注入与细节精修**  
   将 Stage 1 输出作为 `lq_image`，配合同身份 `ref_image`，由 Ref-LDM 进一步恢复身份特征和细节。

原始记录中还尝试了对 LR 做 Gaussian blur / anti-aliasing，但结论是收益有限，因为根本问题是部分 LR 输入的人脸结构已经发生不可逆退化。

### 3.4 Img2Img / SDEdit 改造

Ref-LDM 原始流程接近 noise2img：从随机噪声 latent 开始生成。对于极低质量 LR 图，这会给模型过大的自由度，导致脸型漂移。

后续实验将扩散输入改为更接近 img2img / SDEdit 的方式：先编码 `lq_image` latent，再加噪到中间步并去噪。这样能保留更多输入结构，减少自由生成带来的错位。

### 3.5 最终 Ref-LDM 改动

在 `recovered_gfpgan_two_stage_img2img` 流程中完成了以下修改：

1. 输入预处理从直接 `512x512` resize 改为等比缩放加居中补边，减少几何形变。
2. 将 `configs/refldm.yaml` 中的 `max_num_refs` 从 `5` 改为 `10`。
3. 使用 `CUDA_VISIBLE_DEVICES=2,3` 运行全量实验。
4. 对每张输入保存 `*_compare_stage1_cfgs.jpg`，横向对比 `stage1` 和 `cfg1.0` 到 `cfg10.0`。
5. 保留不同 CFG 下的全部输出，支持后续人工筛选最佳结果。

最终运行结果目录：

```text
/home/hbxu/local/ref-ldm-main/results_ref1_6_generated_gfpgan_refldm_img2img_split10_aspectpad_ref10_gpu23_20260519_133740
```

运行状态：已完成。完成时间：`2026-05-19 14:11:12`。

| 输出类型 | 数量 |
|---|---:|
| GFPGAN Stage 1 中间图 | 66 |
| Ref-LDM Stage 2 CFG 结果图 | 660 |
| Stage1/CFG 对比图 | 66 |

### 3.6 参考图数量上限测试

当前实现会将多张参考图按宽度拼接后送入 VQGAN encoder，因此显存随参考图数量增长较快。

| 参考图数量 | 结果 |
|---:|---|
| 10 | 通过 |
| 12 | 通过 |
| 13 | 通过 |
| 14 | OOM |

结论：在当前代码和显存占用下，13 张参考图接近上限。批量实验建议使用 10 张参考图，稳定性更高。

## 4. FaceMe 参考图扩增路线

### 4.1 实验目的

FaceMe 方向参考论文 **FaceMe: Robust Blind Face Restoration with Personal Identification** 中构造同身份、多姿态、多表情参考人脸数据的思路。由于原始参考图数量较少，实验目标是为 `data/ref-1` 至 `data/ref-6` 的每个身份生成更多同身份参考图，再作为 FaceMe 或 Ref-LDM 的参考集合。

### 4.2 方法概述

FaceMe 论文中的参考图构造思路包括：

1. 构建 pose/expression reference pool。
2. 将 pose 和 expression 聚类成若干子集。
3. 对每个身份从不同 pose/expression 子集中采样条件图。
4. 使用 Arc2Face + ControlNet 生成保持目标身份、迁移参考姿态和表情的图像。
5. 使用 ArcFace 计算身份相似度，低于阈值则丢弃并重试。

本实验沿用 `c1 = 3`、`c2 = 3`，即每个身份构造 9 个 pose/expression 桶，每桶生成 3 张，通过 ArcFace 阈值过滤后，每个身份保留 27 张参考图。

由于 `lf2` 环境尚未完整安装 EMOCA/PyTorch3D 渲染链，实际使用 `generate_faceme_refs.py` 的 `--pose-backend raw-image` 后端：用其它身份参考图作为 ControlNet 条件图，同时保留 Arc2Face 身份控制和 ArcFace 相似度过滤。脚本中保留 `--pose-backend emoca`，后续补齐依赖后可切回 normal-map pose control 流程。

### 4.3 实验环境

```bash
cd /home/hbxu/local/FaceMe-main
conda activate faceme
```

硬件：`NVIDIA GeForce RTX 4090`

主要模型和权重路径：

```text
FaceMe ControlNet: ./weights/controlnet
FaceMe Mix:        ./weights/mix
InsightFace:       ./models/antelopev2
Arc2Face source:   ./external/Arc2Face
Arc2Face weights:  ./external/Arc2Face/models
```

### 4.4 脚本改动

新增参考图生成脚本：

```text
/home/hbxu/local/FaceMe-main/generate_faceme_refs.py
```

主要功能：

1. 从 `--identity-dir` 提取目标身份 ArcFace embedding。
2. 从 `--pose-pool` 采样 pose/expression 条件图。
3. 按 `3 x 3` 桶生成参考图。
4. 调用 Arc2Face + ControlNet 生成同身份候选图。
5. 使用 ArcFace cosine similarity 过滤。
6. 输出 `metadata.csv`，记录 pose 来源、随机种子和相似度。
7. 对过紧裁剪的人脸图增加黑边扩展后重试检测，解决部分样本无法被 InsightFace 检测的问题。
8. 支持 `--pose-backend emoca/raw-image/latent` 三种后端。

同时对 `infer_concert1.py` 至 `infer_concert6.py` 加入参考图过滤逻辑：

- 只读取顶层图片文件：`.jpg/.jpeg/.png/.bmp/.webp`。
- 跳过 `metadata.csv`。
- 跳过 `_controls` 子目录。
- 跳过 `contact_sheet*.jpg`，避免预览图被误作为参考图。

### 4.5 参考图生成命令

`ref-1` 示例：

```bash
python generate_faceme_refs.py \
  --identity-dir ./data/ref-1 \
  --pose-pool ./data/pose_pool_refs_2_6 \
  --output-dir ./data/ref-1_generated \
  --arc2face-root ./external/Arc2Face \
  --models-dir ./external/Arc2Face/models \
  --models-root ./external/Arc2Face \
  --pose-backend raw-image \
  --num-per-bucket 3 \
  --images-per-attempt 1 \
  --max-attempts-per-accept 3 \
  --id-threshold 0.35 \
  --steps 15 \
  --guidance-scale 3.0 \
  --save-controls
```

`ref-2` 至 `ref-6` 由以下脚本批量运行：

```text
/home/hbxu/local/FaceMe-main/run_ref2_6_generation_and_infer.sh
```

### 4.6 FaceMe 结果统计

| 身份目录 | 生成目录 | 生成图片数 | metadata 行数 | ArcFace 相似度最小值 | ArcFace 相似度最大值 |
|---|---|---:|---:|---:|---:|
| `data/ref-1` | `data/ref-1_generated` | 27 | 27 | 0.428570 | 0.814600 |
| `data/ref-2` | `data/ref-2_generated` | 27 | 27 | 0.642068 | 0.848243 |
| `data/ref-3` | `data/ref-3_generated` | 27 | 27 | 0.445721 | 0.797131 |
| `data/ref-4` | `data/ref-4_generated` | 27 | 27 | 0.497504 | 0.736867 |
| `data/ref-5` | `data/ref-5_generated` | 27 | 27 | 0.482107 | 0.779534 |
| `data/ref-6` | `data/ref-6_generated` | 27 | 27 | 0.511796 | 0.760006 |

总计生成参考图：`162` 张。

FaceMe 推理输出：

| 推理脚本 | ref_dir | result_dir | 输出图片数 |
|---|---|---|---:|
| `infer_concert1.py` | `./data/ref-1_generated` | `./results_1_ref-1_generated` | 2 |
| `infer_concert2.py` | `./data/ref-2_generated` | `./results_2_ref-2_generated` | 3 |
| `infer_concert3.py` | `./data/ref-3_generated` | `./results_3_ref-3_generated` | 5 |
| `infer_concert4.py` | `./data/ref-4_generated` | `./results_4_ref-4_generated` | 4 |
| `infer_concert5.py` | `./data/ref-5_generated` | `./results_5_ref-5_generated` | 7 |
| `infer_concert6.py` | `./data/ref-6_generated` | `./results_6_ref-6_generated` | 1 |

总计恢复输出：`22` 张。

## 5. 综合结论

1. 直接输入原图给 Ref-LDM 效果较差，必须先做人脸裁剪和对齐。
2. 对极低质量 LR 图，仅靠调节 CFG 或轻微平滑无法根治结构缺失问题。
3. 两阶段策略更合理：先用 GFPGAN 恢复基础结构，再用 Ref-LDM 结合参考图恢复身份与细节。
4. Img2img/SDEdit 风格的 latent 初始化比 noise2img 更能保留 LR 输入结构。
5. 非正方形图像直接 resize 会引入脸型变形，等比缩放加居中补边是必要修正。
6. 参考图数量增加能提供更多身份信息，但会带来显存压力和参考质量筛选问题。
7. FaceMe-style 参考图扩增可以缓解原始参考图不足，但生成参考图仍需 ArcFace 过滤和人工复核。
8. 当前最有效的工程流程是：人脸裁剪/对齐、参考图扩增与过滤、GFPGAN Stage 1、Ref-LDM img2img Stage 2、多 CFG 输出、人工筛选。

## 6. 代码位置

| 模块 | 路径 | 说明 |
|---|---|---|
| Ref-LDM 主目录 | `/home/hbxu/local/ref-ldm-main` | Ref-LDM 实验代码、配置和结果 |
| Ref-LDM 两阶段 img2img | `/home/hbxu/local/ref-ldm-main/recovered_gfpgan_two_stage_img2img` | GFPGAN + Ref-LDM img2img 推理脚本 |
| Ref-LDM 配置 | `/home/hbxu/local/ref-ldm-main/configs/refldm.yaml` | 修改 `max_num_refs` 等配置 |
| FaceMe 主目录 | `/home/hbxu/local/FaceMe-main` | FaceMe 推理和参考图生成代码 |
| FaceMe 参考图生成 | `/home/hbxu/local/FaceMe-main/generate_faceme_refs.py` | Arc2Face/ControlNet/ArcFace 参考图扩增 |
| FaceMe 批处理 | `/home/hbxu/local/FaceMe-main/run_ref2_6_generation_and_infer.sh` | `ref-2` 至 `ref-6` 批量生成和推理 |

## 7. 后续计划

1. 基于 `*_compare_stage1_cfgs.jpg` 筛选不同 CFG 下的最佳输出。
2. 统计不同 CFG 对身份保持、清晰度和形变程度的影响。
3. 对生成参考图做更严格的质量过滤，优先保留高相似度、高质量样本。
4. 比较 5、10、13 张参考图对恢复质量和稳定性的影响。
5. 补齐 EMOCA/PyTorch3D 环境，切换到 FaceMe 论文中更精确的 normal-map pose control 流程。
6. 如果继续提升质量，可考虑训练阶段加入更强退化模型或额外结构条件，如 landmarks、face parsing 或 ControlNet 分支。

