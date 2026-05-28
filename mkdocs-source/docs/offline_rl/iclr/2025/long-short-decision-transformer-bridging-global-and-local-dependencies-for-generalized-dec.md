---
title: "Long-Short Decision Transformer Bridging Global and Local Dependencies for Generalized Decision-Making"
description: "> **论文标题**: Long-Short Decision Transformer: Bridging Global and Local Dependencies for Generalized Decision-Making"
tags:
  - Offline RL
---

# Long-Short Decision Transformer Bridging Global and Local Dependencies for Generalized Decision-Making

<div class="paper-hero">
<p class="paper-meta">详见笔记 · ICLR 2025 · ICLR / 2025</p>
<div class="tag-row"><span>Offline RL</span></div>
</div>

<article class="note-body" markdown>

﻿# Long-Short Decision Transformer (LSDT) 深度阅读笔记

> **论文标题**: Long-Short Decision Transformer: Bridging Global and Local Dependencies for Generalized Decision-Making
> **会议**: ICLR 2025 (已接受)
> **作者**: Jincheng Wang (UCL), Penny Karanasou (Cambridge), Pengyuan Wei (UCL), Elia Gatti (UCL), Diego Martinez Plasencia (UCL), Dimitrios Kanoulas (UCL)
> **机构**: University College London, University of Cambridge
> **资源**: [OpenReview](https://openreview.net/forum?id=NHMuM84tRT) · arXiv: 未见公开预印本 · 代码: 未确认官方代码链接
> **阅读日期**: 2026-05-27

---

## 0. 一句话总结

LSDT 提出**双分支并行架构**——一条 self-attention 分支捕获全局长程依赖，一条卷积（DynamicConv）分支捕获局部马尔可夫依赖——通过在离线 RL 的不同环境中分配不同的通道维度比例，统一了 Decision Transformer (DT) 和 Decision ConvFormer (DC) 的优势，在 D4RL 上达到 SOTA。

---

## 1. 定位 (Positioning)

本文属于 **offline RL 中序列建模（sequence modeling）方向**，具体位于 Decision Transformer 架构改进这条子线。

**在序列建模 offline RL 版图中的位置**:
- **DT (Decision Transformer)**: 用 causal Transformer 做条件序列建模——擅长全局长程依赖，但忽略局部马尔可夫性质
- **DC (Decision ConvFormer)**: 用因果卷积（causal convolution）替换 self-attention——擅长局部依赖，但在需要全局推理的任务（如 Maze2d）上性能差
- **Trajectory Transformer**: 离散化+Transformer——同时捕捉局部和全局，但离散化开销大
- **CDT (Contrastive DT)**: 引入对比学习——优化方向不同

**LSDT 的差异化**:
1. 首次在 DT 框架中系统地融合 self-attention 和 convolution，形成"长-短"双分支
2. 通过可调节的通道维度比例 ξ（dimension ratio）实现环境自适应
3. 将 LSDT 的架构思想泛化到 LSDC（Long-Short Decision ConvFormer），证明该双分支设计具有通用性

---

## 2. 第一作者简介

**Jincheng Wang** (王金程)，University College London (UCL)，Dimitrios Kanoulas 课题组成员。

DBLP 记录 (pid: 85/5973) 显示其研究领域较广，涉及 RL、机器人、声悬浮、电路设计等。与本文相关的近期 RL 论文：

| 年份 | 论文 | 会议/期刊 |
|------|------|-----------|
| 2025 | **LSDT: Long-Short Decision Transformer** | ICLR 2025 |
| 2026 | AAC: Acoustic Actor-Critic for Multi-Particle Levitation Displays | CHI 2026 |
| 2026 | AcoustoReinforce: Acoustophoretic Path Planning with Deep RL | AAAI 2026 |

> 来源: DBLP，Jincheng Wang 为常见姓名，DBLP 有大量同名记录。此处仅列出与 RL/UCL 团队相关的最近发表。团队导师 Diego Martinez Plasencia 和 Dimitrios Kanoulas 分别从事人机交互和机器人研究，LSDT 是该组首次出现在 offline RL 顶会上。

---

## 3. 核心问题

### 问题陈述
Decision Transformer 将 RL 建模为**条件序列预测**（给定 return-to-go、状态、动作的历史，预测下一个动作），绕过了传统 RL 的 bootstrapping（自举）误差问题。但 DT 面临一个架构层面的张力：

- **Self-attention 擅长全局长程依赖**（如 Maze2d 中需要记住遥远过去的转弯），但**忽略局部马尔可夫性质**（相邻时间步有直接的因果关系）
- **卷积擅长局部依赖**（相邻帧/动作的细粒度关联），但**无法捕获全局依赖**——DC 在 Maze2d 上的极差性能（Figure 2）印证了这一点

**为什么这很重要**：Offline RL 数据集的特性往往混合——有些来自马尔可夫策略（相邻步独立的专家数据），有些是非马尔可夫的（轨迹级规划产生的数据）。单一架构无法同时适配。

### 现有方法的局限

| 方法 | 优势 | 局限 |
|------|------|------|
| DT (Transformer) | 全局长程依赖 | 忽略局部马尔可夫性质 |
| DC (ConvFormer) | 局部依赖，参数效率高 | 全局依赖能力弱（Maze2d 失败） |
| S4/序列状态空间 | 长程建模 | 非并行因果结构，离线 RL 场景未充分验证 |

**核心矛盾**: 如何在同一个架构中同时高效捕获全局长程依赖和局部细粒度依赖，且能适配不同类型的数据集？

### 论据来源
- Figure 2: DC 在 Maze2d 上完全失败（回报接近0），而 DT 表现良好——说明全局依赖不可或缺
- DC 在 Locomotion 任务（马尔可夫性强）上表现优秀——说明局部建模有独特价值
- Kim et al. (2023) 的论文指出：纯 Transformer 在 D4RL locomotion 上不如卷积

---

## 4. 前置知识

### 4.1 Decision Transformer (DT) 原理
DT 将 RL 问题重新表述为**自回归序列建模**：

$$\tau = (\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, ..., \hat{R}_T, s_T, a_T)$$

其中 $\hat{R}_t = \sum_{t'=t}^T r_{t'}$ 是 return-to-go (RTG)。DT 使用 causal Transformer 预测 $a_t$，训练时最小化动作预测误差。**核心优势**: 无需价值函数，无 bootstrapping，通过模仿数据中的高回报轨迹学习。

### 4.2 自注意力 (Self-Attention) 的全局性
标准 self-attention 计算：

$$\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

由于每个位置的 query 与所有位置的 key 做点积，self-attention 天然具有**全局感受野**——位置 t 可以直接"看到"位置 1。这使其擅长捕获长程依赖（如"在迷宫起点转左会影响终点的路径"）。

**局限**: Self-attention 是置换等变的（在位置编码加入之前），对局部邻域的细粒度模式不加区分。在高度马尔可夫的数据中，这种全局性反而引入噪声。

### 4.3 因果卷积 (Causal Convolution) 的局部性
因果卷积只在当前及历史时间步上做卷积，确保"不看未来"：

$$y_t = \sum_{i=0}^{k-1} w_i \cdot x_{t-i}$$

局部感受野大小 = kernel_size，天然适合马尔可夫过程（当前状态只与前一步相关），且参数效率高。

**局限**: 固定的局部感受野无法捕获超出 kernel_size 的依赖。在 Maze2d 这种需要跨多步推理的任务中失效。

### 4.4 动态卷积 (Dynamic Convolution)
标准卷积的权重是固定的（与输入无关），而动态卷积的核权重**由输入条件生成**：

$$W = f_\theta(x)$$

这使得卷积能够根据不同的上下文（如不同的轨迹/状态）自适应调整滤波模式。**在 LSDT 中的作用**: 相比 DC 中的标准卷积，DynamicConv 提供了在局部依赖建模上的更大表达力。

### 4.5 通道维度比例 (Channel Dimension Ratio ξ)
LSDT 的核心创新点之一。输入序列的 embedding 维度为 C，LSDT 将其按比例 ξ 分成两部分：
- ξ × C 维度 → 全局分支（self-attention）
- (1-ξ) × C 维度 → 局部分支（convolution）

不同的 ξ 值反映了对不同依赖类型的侧重。ξ=0 退化为纯卷积（DC），ξ=100% 退化为纯 attention（DT）。

---

## 5. 思想与方法

### 5.1 核心思想

LSDT 的核心洞察：**全局和局部依赖不是互斥的，而是互补的——可以通过在架构层面将它们并行化，然后自适应地分配模型容量来实现统一**。

**设计哲学**:
1. **分而治之 (Divide)**: 将输入表示按维度比例 ξ 分给两个专用分支
2. **各自专业 (Specialize)**: Self-attention 分支专注于全局依赖；DynamicConv 分支专注于局部依赖
3. **融合 (Merge)**: 两个分支的输出通过 HFM (Hierarchical Fusion Module) 融合

### 5.2 架构细节 (Figure 1)

LSDT 由 3 层 stacked block 组成，每层包含：

```
输入序列 X (T × C)
    |
    ├── 全局分支 (ξ×C dims): Self-Attention (因果mask) → FFN
    ├── 局部分支 ((1-ξ)×C dims): DynamicConv (因果) → FFN
    |
    └── HFM 融合模块 → 输出
```

**DynamicConv** 使用权重共享策略减少参数量（kernel 数 = 卷积分支维度 ÷ 4）。

**HFM (Hierarchical Fusion Module)**: 在维度拼接后，通过一层 FFN 重新混合全局和局部特征。论文附录 B.3 中讨论了几种融合策略（简单拼接、加权和、HFM），HFM 表现最好。

### 5.3 关键设计选择及动机

**(1) 为什么用 DynamicConv 而非标准 Conv？**
- 标准卷积权重对所有输入相同，无法适应不同轨迹/状态
- 动态卷积的核由输入条件生成，使其能在不同上下文中自适应——在非平稳的 offline RL 数据中更重要

**(2) 为什么比例 ξ 需要手动调而非学习？**
- 论文承认这是局限性（Section F），也需要手动为不同任务设置不同的 ξ
- 附录 B.10 探索了自动学习 ξ 的方案，但实验性较强
- 直觉上：马尔可夫性强的任务（locomotion）偏重卷积（ξ 较小），需全局推理的任务（Maze2d, AntMaze）偏重 attention（ξ 较大）

**(3) 为什么还需要 FFN？**
- Self-attention 本身没有非线性激活（仅 softmax），DynamicConv 也是线性运算
- FFN 引入非线性变换能力，且在两个分支中独立使用 FFN，使它们学习不同的特征变换模式

### 5.4 从 DT 和 DC 到 LSDT 的演进逻辑

| | DT | DC | LSDT |
|---|---|---|---|
| 全局依赖 | ✅ 强 | ❌ 弱 | ✅ 强 |
| 局部依赖 | ❌ 弱 | ✅ 强 | ✅ 强 |
| 参数效率 | 中等 | 高 | 中等偏上 |
| 环境适配 | 一律全局 | 一律局部 | 自适应 ξ |
| 优势场景 | Maze2d, AntMaze | Locomotion | **全部** |
| FLOPs | 17.99M | 11.01M | 17.08M |

### 5.5 为什么能同时适配两类环境？

**机制链条**:
1. 在 **Locomotion**（马尔可夫性强）: ξ 较小（如 Walker2d-medium: 12.5%），模型主要依赖 DynamicConv 分支——因为相邻步的动作关系足够且最重要
2. 在 **AntMaze/Maze2d**（需全局推理）: ξ 较大（如 AntMaze-medium: 75%），模型主要依赖 Self-Attention 分支——因为迷宫需要跨多步推理路径规划
3. **HFM 融合**确保了即使某个分支权重较小，其信息也不会丢失——通过跨维度信息交互，全局和局部信号互相增强

**证据强度**: 实验验证充分（Table 1 全面比较，Table 14 的 ξ 消融），架构设计的合理性有 Figure 2 的 Maze2d 可视化支撑。但缺乏严格理论（如 attention/convolution 互补性的 formal analysis）。

---

## 6. 算法与伪代码

### 算法名称: Long-Short Decision Transformer (LSDT)

### 架构概述

```
算法: LSDT 训练与推理

输入: 离线数据集 D, 上下文长度 K, 通道维度比率 ξ, 总维度 C

模型结构:
  - 3 层 LSDT Block，每层含:
    - 全局分支: [GELU, Causal Self-Attention, Dropout, FFN]
    - 局部分支: [GELU, DynamicConv (kernel_size=3或6), Dropout, FFN]
    - HFM 融合: 拼接 + 线性投影 + 残差连接
  - 最终动作预测头: Linear(C → act_dim)

训练:
  for each minibatch from D:
    采样长度为 K 的子序列 τ = [RTG_t, s_t, a_t, ..., RTG_{t+K}, s_{t+K}, a_{t+K}]
    
    // 输入嵌入
    embed = [TokenEmb(RTG_t,s_t), ..., TokenEmb(RTG_{t+K},s_{t+K})]
    
    // 按比例 ξ 拆分维度
    embed_global = embed[:, :, :ξ*C]
    embed_local  = embed[:, :, ξ*C:]
    
    // 3层 LSDT Block (因果mask确保不看到未来)
    for each layer:
      global_out = SelfAttention(embed_global, causal_mask=True) + FFN(...)
      local_out  = DynamicConv(embed_local, causal=True) + FFN(...)
      combined   = HFM(global_out, local_out)
    
    // 预测动作
    pred_actions = ActionHead(combined)
    
    // 损失: MSE(预测动作, 真实动作)
    loss = MSE(pred_actions, ground_truth_actions)
    
    梯度更新

推理 (zero-shot):
  给定初始状态 s_0 和目标 RTG (如 1.0 × target_return):
  for t = 0 to max_steps:
    构建当前序列 (含历史+当前状态+目标RTG)
    通过 LSDT 预测动作 a_t
    执行 a_t，获取 s_{t+1}, r_{t+1}
    RTG_{t+1} = RTG_t - r_{t+1}
```

### 关键超参数:
| 参数 | 含义 | 典型值 |
|------|------|--------|
| ξ | 全局分支通道比例 | 12.5%-100%，按任务选择 (Table 14) |
| K | 上下文长度 | 8 (locomotion), 变化值 (AntMaze/Adroit) |
| T | 层数 (堆叠的 blocks) | 3 |
| kernel_size | 动态卷积核大小 | 3 (Maze2d), 6 (其他) |
| emb_dim | 总嵌入维度 C | 128 |
| batch_size | 批次大小 | 64 (locomotion), 256 (AntMaze/Adroit/Maze2d) |
| lr | 学习率 | 1×10⁻⁴ 或 2×10⁻⁴ |
| dropout | Self-Attention | 0.1 (DT标准), 0.25 (LSDT) |
| dropout_conv | DynamicConv | 0.2 |

### LSDC 变体:
论文证明该 Long-Short 架构具有通用性——将 DynamicConv 替换为 DC 中的标准因果卷积，得到 **LSDC** (Long-Short Decision ConvFormer)。LSDC 在训练时间上比 LSDT 更快（473s vs 713s），说明双分支设计的价值不限于特定卷积类型。

---

## 7. 实验与消融

### 7.1 实验设置
- **基准**: D4RL (Fu et al., 2020)
  - **Locomotion**: halfcheetah/hopper/walker2d × medium/medium-replay/medium-expert (9 tasks)
  - **AntMaze**: umaze/umaze-diverse/medium-play/medium-diverse/large-play/large-diverse (6 tasks)
  - **Maze2d**: umaze/medium/large (3 tasks)
  - **Adroit**: pen-human/pen-cloned/door-expert/hammer-expert/relocate-expert (5 tasks)
- **Baselines**: DT, DC + 变体 (DC-A, DCh), %BC, CQL, IQL, TT (Trajectory Transformer), CDT
- **评估**: normalized score, 5 seeds

### 7.2 主要结果 (Table 1-4)

**Locomotion (Table 1)**:
| 方法 | halfcheetah-m | hopper-m | walker2d-m | half-m-r | hopper-m-r | walk-m-r | 总计 |
|------|-------------|----------|-----------|---------|----------|---------|------|
| DT | 42.6 | 69.4 | 74.6 | 35.7 | 83.1 | 65.7 | 568.1 |
| DC | 43.0 | **111.9** | 79.3 | 40.9 | 95.6 | 75.3 | 618.0 |
| **LSDT** | **47.2** | 111.8 | **84.6** | **46.2** | **104.9** | **82.1** | **662.9** |

- LSDT 在 9 个 locomotion 任务上平均分最高（662.9 vs DC 618.0 vs DT 568.1）
- LSDT 在大部分任务上超越 DT 和 DC，且稳定性更好（标准差更小）

**AntMaze (Table 2)**: LSDT 在 6 个迷宫任务中 5 个为最优，超越了 IQL（IQL 通常是 AntMaze 上的强 baseline）。尤其 large-play (79.2 → LSDT 70.0 vs IQL 39.6) 和 large-diverse (70.0 → LSDT 60.0 vs IQL 47.5)。

**Maze2d (Table 3, Figure 2)**: DC 在这些任务上完全失败（回报接近0），而 DT 表现良好。LSDT 进一步将 DC 的性能从接近 0 提升到显著水平（通过 DC + LSDT → LSDC 的对比，Figure 2 展示），超越了纯 DT。

**Adroit (Table 4)**: LSDT 在 reloc-expert 上达到 102.6（DT: 98.1, DC: 98.1），在 pen-human 上达到 73.3（DT: 73.2），整体稳定领先。

### 7.3 消融研究

**(1) 维度比例 ξ 的影响 (Figure 3-6)**:
- 马尔可夫环境（locomotion）: ξ 较小时（12.5%-50%）性能最优 → 局部分支更重要
- 非马尔可夫环境（Maze2d, AntMaze）: ξ 较大时（50%-100%）性能最优 → 全局分支更重要
- ξ = 0% 的 conv-only 在 Maze2d 上回到 DC 的失败模式
- ξ = 100% 的 attn-only 回到 DT 的 baseline

**(2) LSDC 验证框架通用性 (Section 4.4, Figure 2)**:
- 将 LSDT 的 DynamicConv 换成 DC 中的标准卷积 → LSDC
- LSDC 在 Maze2d 上大幅超越纯 DC，证明了 Long-Short 架构本身的价值
- LSDC 训练比 LSDT 快 33%（473s vs 713s）

**(3) 融合策略比较 (Table 7, Appendix)**:
- 简单拼接 < 加权和 < HFM
- HFM 通过跨维度投影实现信息重组，对性能贡献显著

**(4) 计算效率 (Table 15-16)**:
- LSDT FLOPs: 17.08M MMAC，接近 DT (17.99)，远超 DC (11.01) 但换取了更全面的能力
- 训练时间: LSDT 713s vs DT 381s vs DC 364s（1.87× DT, 1.96× DC）
- 参数总量: LSDT 1.09M vs DT 1.13M → 参数更少，因为卷积分支比 attention 更参数高效
- GPU 内存: 0.11-0.14GB（取决于 ξ），非常轻量

### 7.4 实验结果解读

**LSDT 最强的地方**:
1. 跨域泛化——同一个架构在 locomotion、AntMaze、Maze2d、Adroit 上都达到 SOTA 或接近 SOTA
2. 在马尔可夫和非马尔可夫任务上都有大幅提升——证明了"长+短"互补的普适性

**LSDT 仍有差距的地方**:
1. AntMaze-large 的某些配置上仍弱于 IQL（IQL 擅长稀疏奖励的探索策略）
2. Adroit 的某些任务（door/hammer）略弱于最先进的 DT 变体

---

## 8. 展望

### 研究启发

1. **"长+短"架构互补是普适原则**：LSDT 的核心贡献不在于某个新模块，而在于展示 self-attention（全局）和 convolution（局部）可以并行工作并互补。这一发现对 CV 和 NLP 中的架构设计也有启示。

2. **维度比例 ξ 是环境特性的代理指标**：ξ 的最佳值实际上反映了环境的马尔可夫性/非马尔可夫性——提供了一种从架构选择反推环境特性的方法。

3. **双分支设计的通用性**：LSDT → LSDC 的转换（仅替换卷积类型）证明了 Long-Short 思想是通用的架构设计模式，可移植到其它序列建模框架。

4. **序列建模在 offline RL 中的潜力尚未耗尽**：DT 提出已近 4 年，但 LSDT 通过架构层面的创新仍在多任务上实现了显著提升（远超调参级别的改进），说明序列建模路线的架构优化仍有很大空间。

5. **参数效率的惊喜**：LSDT 的参数量（1.09M）竟然比 DT（1.13M）更少，因为卷积分支的参数共享设计抵消了双分支的开销。这暗示"专业化模块"可以比"通用模块"更参数高效。

### 局限性

1. **ξ 需手动调节**：论文承认（Section F）不同任务需要不同的 ξ，这是实际部署的障碍。附录 B.10 的自动方案仍不成熟。

2. **训练时间较长**：LSDT 训练时间是 DT 的约 1.87 倍，DynamicConv 是主要瓶颈。LSDC 提供了缓解方案但失去了 DynamicConv 的自适应能力。

3. **DynamicConv 的 kernel 共享策略**：权重共享（kernel 数 = 卷积分支维度 ÷4）虽然减少了参数，但可能限制了局部依赖的表达力上限。

4. **实验限于 D4RL 标准基准**：未在图像观察（Atari, DMControl）、更复杂环境或在线微调场景中验证。

5. **缺乏理论分析**：论文未提供 attention+convolution 互补性的任何理论证明，如对全局/局部依赖的形式化定义或泛化界分析。

### 后续研究方向

1. **自适应 ξ 选择**：设计一个轻量级的元控制器/路由器，根据环境的在线反馈动态调整 ξ，使 LSDT 能在部署时自适应。

2. **将 LSDT 扩展到在线 RL**：在线探索场景中，全局/局部依赖的混合可能有助于更好的信用分配（credit assignment），特别是在部分可观测环境中。

3. **Long-Short 思想应用于 diffusion-based RL 方法**：将双分支概念从 DT 系列扩展到近期的扩散模型 offline RL 方法（如 DAC、Diffusion Q-learning），在 score function 建模中引入多尺度依赖。

---

## 链接

- **OpenReview**: [https://openreview.net/forum?id=NHMuM84tRT](https://openreview.net/forum?id=NHMuM84tRT)
- **arXiv**: 未能从公开来源确认预印本链接
- **代码**: 未能从公开来源确认官方代码链接
- **DBLP**: [https://dblp.org/rec/conf/iclr/WangKWGPK25](https://dblp.org/rec/conf/iclr/WangKWGPK25)

</article>
