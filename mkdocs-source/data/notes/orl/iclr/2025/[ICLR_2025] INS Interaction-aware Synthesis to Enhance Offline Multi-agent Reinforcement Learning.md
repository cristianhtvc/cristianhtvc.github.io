---
title: "INS: Interaction-aware Synthesis to Enhance Offline Multi-agent Reinforcement Learning"
authors: "Yuqian Fu, Yuanheng Zhu, Jian Zhao, Jiajun Chai, Dongbin Zhao"
venue: "ICLR 2025 Poster"
date: 2025
tags: [Offline MARL, Data Synthesis, Diffusion Models, Sparse Attention, Multi-agent Interaction]
openreview: "https://openreview.net/forum?id=kxD2LlPr40"
code: "论文提供了源代码链接（见论文摘要）"
---

# INS 深度阅读笔记

## 一句话总结

INS (INteraction-aware Synthesis) 是一种利用扩散模型为离线 MARL **合成高质量多智能体数据集**的方法，通过**稀疏注意力**捕获智能体间交互、**bit action 模块**兼容离散/连续动作空间、**select 机制**优先保留高价值转移，实验证明 INS 合成的数据集在四个下游离线 MARL 算法上均带来一致且显著的性能提升，且仅用 10% 原始数据即可保持效果。

## 定位

本文处于 **离线多智能体强化学习 × 数据增强/合成** 的交叉领域。不同于已有离线 MARL 方法（如 OMIGA、ComaDICE、OMAR）在**算法层面**（策略约束、保守估计、分布正则化）解决数据稀疏问题，INS 走的是**数据层面**的路线：用生成模型扩充 offline 数据集。这条路线在单智能体 offline RL 中已有成功先例（SynthER、GTA、Policy-guided Diffusion），但在多智能体场景下存在两大独特挑战：

1. **智能体间交互**：多智能体系统中转移 (transition) 涉及多个互相影响的智能体，独立处理会破坏交互信
2. **动作空间离散性**：MA 环境（如 SMAC）常用离散动作，而主流扩散模型面向连续数据

INS 是首个针对这两大挑战给出完整解决方案的工作。

## 第一作者简介

**Yuqian Fu（付雨倩）**，中国科学院自动化研究所博士生（2022年入学），导师为 Yuanheng Zhu（朱圆恒）和 Dongbin Zhao（赵冬斌）。研究方向为多智能体强化学习、扩散模型在 RL 中的应用。合作者：

- **Yuanheng Zhu（朱圆恒）**：中科院自动化所，通讯作者，研究方向包括深度强化学习、多智能体系统
- **Dongbin Zhao（赵冬斌）**：中科院自动化所研究员，通讯作者，研究方向包括智能控制、强化学习
- **Jian Zhao（赵鉴）**：Polixir（南栖仙策），工业界合作者
- **Jiajun Chai（柴嘉骏）**：中科院自动化所/国科大，合作者，有 MARL 相关发表（AAMAS 2024）

（以上信息来源于 OpenReview 和论文署名。）

## 前置知识

### 1. 离线 MARL 中的数据稀疏问题

离线 MARL 面临的核心困难：
- **分布偏移 (distribution shift)**：学习策略偏离行为策略导致外推误差
- **数据稀疏 (data scarcity)**：多智能体联合动作空间 $\prod_i |A_i|$ 随智能体数量指数增长，offline 数据覆盖率极低

现有解决方法主要分两类：
- **算法层面**：策略约束 (OMAR)、保守 Q 学习 (MA-CQL)、占优正则化 (ComaDICE)、隐式全局-局部正则化 (OMIGA)
- **数据层面**：数据增强/合成——这是 INS 属于的路线

### 2. 扩散模型做数据合成

扩散模型可以通过学习数据分布 $p(x)$，然后从噪声中采样生成新的数据点。在离线 RL 中，这意味着可以从有限的数据集中学到一个"数据生成器"，产出更多样的转移样本。

**已有工作的局限**：
- **SynthER** (Lu et al., 2024)：为单智能体 offline RL 设计，直接用 MLP 做 denoiser，不建模智能体间交互
- **MADiff** (Zhu et al., 2023)：将扩散模型用于多智能体策略表达（policy parameterization），而非数据合成
- **GTA** (Lee et al., 2024)：用引导扩散做 trajectory augmentation，但在单智能体场景

### 3. EDM（Elucidated Diffusion Model）

INS 采用 Karras et al. (2022) 提出的 EDM 架构。EDM 将扩散模型统一为概率流 ODE：

$$dx = -\dot{\sigma}(k)\sigma(k)\nabla_x \log p(x; \sigma(k)) dk$$

其中 $\sigma(k)$ 是时间步 $k$ 的噪声水平，$\nabla_x \log p(x; \sigma)$ 是分数函数（score function）。通过去噪分数匹配（denoising score matching），可以用一个去噪网络 $D_\theta(x; \sigma)$ 来近似分数函数：$\nabla_x \log p(x; \sigma) = (D_\theta(x; \sigma) - x)/\sigma^2$。

EDM 的优势：连续噪声调度 + 概率流 ODE 采样，比 DDPM 更高效灵活。

### 4. Sparsemax vs Softmax

传统 Transformer 的 Self-Attention 使用 Softmax 归一化注意力权重，产生的权重向量是稠密的（所有位置都有非零权重）。**Sparsemax** (Martins & Astudillo, 2016) 将 Softmax 替换为 Euclidean 投影到单纯形（simplex）上：

$$\text{Sparsemax}(z) = \arg\min_{p\in\Delta^{n-1}} \|p - z\|_2$$

结果是**稀疏的概率向量**——只有部分分量非零。这在多智能体场景中非常有用：智能体通常只与少数邻近智能体交互，Sparsemax 天然将不相关智能体的注意力权重置零。

### 5. Bit Diffusion

Bit Diffusion (Chen et al., 2023) 是一种将离散数据表示为连续 bit 向量的方法。对于有 $M$ 个取值的离散变量，用 $L = \lceil\log_2(M)\rceil$ 个二进制位表示，每位取 $\{0.0, 1.0\}$。这使得扩散模型可以在连续空间中处理离散变量。

---

## 问题分析

### 论文诊断的核心问题

**离线 MARL 中数据合成的三大挑战**：

1. **智能体间交互被忽视导致数据失真**：将每个智能体的数据独立处理、独立合成，会破坏多智能体转移样本中的交互信息（碰撞、协作、信息传递等），合成数据无法还原真实的智能体动力学。附录 A 给出了一个 motivating example 来展示这一现象。

2. **合成数据集质量受限于原始数据集**：扩散模型学习分布 $p(x)$，但如果 $p(x)$ 中高质量转移占比低，合成数据中高质量样本的占比也不会提升。需要一种机制来**有偏地**提升合成数据中高价值转移的比例。

3. **离散动作空间不兼容**：主流扩散模型面向连续变量（如图像像素 $-1$ 到 $1$），但许多 MARL benchmark（如 SMAC）使用离散动作。需要让扩散模型生成离散动作而又不损失精度。

### 现有方法的不足

| 方法 | 交互建模 | 质量选择 | 离散动作 | 本质问题 |
|------|:---:|:---:|:---:|------|
| MA-SynthER (SynthER 多智能体版) | ✗ (MLP, 独立处理) | ✗ | ✗ | 忽视交互 → 数据失真 |
| MADiff | ✓ (扩散策略) | ✗ | ✗ | 做策略参数化而非数据合成 |
| Additive Noise | ✗ | ✗ | N/A | 简单加噪，多样性差 |
| VAE Augmented | ✗ | ✗ | N/A | VAE 生成质量远不如扩散模型 |

### 论文论证路径

1. **Motivation**：通过附录 A 的示例展示忽视交互的后果——独立合成的转移样本中智能体"穿越"彼此，违背物理规律
2. **Phase A**：设计带稀疏注意力的 Transformer 扩散模型，学习多智能体转移的联合分布
3. **Phase B**：引入 select 机制——训练一个 value estimator $V_\phi$，根据估计状态价值优先选择高价值转移
4. **Phase C**：用合成数据集训练任意离线 MARL 算法
5. **实验验证**：MPE（连续动作）+ SMAC（离散动作）上 4 种离线 MARL 算法 + 多指标数据集质量评估

---

## 思想与方法

### 核心思想：分三阶段——建模交互、提升质量、训练策略

INS 的核心直觉简洁而有力：

> **与其在算法层面约束策略来对抗数据稀疏，不如直接合成更多、更好的数据。关键是合成时必须尊重多智能体交互结构——哪些智能体之间确有交互、哪些转移更"值钱"。**

### 方法名称与组件

**INS** = **IN**teraction-aware **S**ynthesis

三大阶段 × 三个关键设计：

#### Phase A：训练交互感知扩散模型

**三大设计**：

**设计 1：Sparse Attention（稀疏注意力）**
- 使用 Sparsemax 替代 Softmax 计算多头注意力权重
- 效果：不相关的智能体对之间注意力权重为 0，模型专注于真实的交互关系
- 消融验证：去掉 sparse attention（改用 dense attention）后，合成数据准确性和策略性能都下降（Figure 4）

**设计 2：Transformer Encoder 架构**
- 使用 Transformer encoder（而非 U-Net）作为 denoising backbone
- 智能体维度上的 attention 天然适合建模 agent-agent 交互
- 相比 MLP（各智能体独立处理），attention 可以捕获联合转移中的交互模式

**设计 3：Bit Action 模块**
- 离散动作 $a \in \{1,...,M\}$ → 二进制位向量 $b \in \{0,1\}^L$，$L = \lceil\log_2 M\rceil$
- 前向过程中：bit 向量与连续变量拼接，一起参与扩散
- 反向采样后：量化（threshold at 0.5）→ 映射回离散动作
- 优势：(1) 维度从 $M$ 降到 $\lceil\log_2 M\rceil$；(2) 使 INS 同时兼容连续和离散动作空间

#### Phase B：合成 + 选择

1. **采样**：从噪声 $x_K$ 开始，通过 EDM 的 ODE solver 逐步去噪，得到合成转移 $x_0 = (o, a, o', r)$
2. **Select 机制**：
   - 训练一个 value estimator $V_\phi$（公式 8）：$L_V(o) = \mathbb{E}[(r + \gamma V_{\bar{\phi}}(o') - V_\phi(o))^2]$
   - 对合成转移计算估计价值 $V_\phi(o)$
   - 用 Softmax 权重 $\propto \exp(V_\phi(o)/\tau)$ 概率性地选择转移（$\eta$ 为保留比例）
   - **关键tradeoff**：$\eta$ 越大 → 越偏向高价值，但多样性越低。实验发现 $\eta=0.8$ 在多数任务上平衡最好

#### Phase C：下游离线 MARL 训练

将合成数据集 $D_{syn}$ 送入任意离线 MARL 算法（MA-ICQ, MA-CQL, OMAR, MA-BCQ）训练策略。INS 与下游算法**完全正交**（orthogonal），即插即用。

### 为什么能解决问题

| 挑战 | INS 的解决方案 | 机制 |
|------|-------------|------|
| 智能体间交互被忽视 | Sparse attention + Transformer encoder | Sparsemax 只关注确有交互的智能体对，去噪时保留交互结构 |
| 合成数据质量受限 | Select 机制（value estimator + Softmax 选择）| 优先保留高价值转移，提升合成数据中"好样本"密度 |
| 离散动作不兼容 | Bit action 模块 | 将离散动作映射为连续 bit 向量，在 bit 空间做扩散 |
| 数据多样性不足 | 扩散模型天然泛化 + 覆盖增强 | 扩散模型在分布支撑集内 interpolation，t-SNE 显示可填补分布间隙 |

---

## 算法与伪代码

### 算法名称：INS (INteraction-aware Synthesis)

### 三阶段流程

**Phase A：训练扩散模型**

1. 构建 denoising 网络 $D_\theta$：
   - Backbone：Transformer encoder，agent 维度做 sparse attention (Sparsemax)
   - 输入：噪声转移 $x = (o_1, a_1, ..., o_n, a_n, o'_1, ..., o'_n, r)$
   - 离散动作经 bit action 映射为连续 bit 向量后拼接
2. 优化 EDM 去噪损失（公式 2）：
   $$L_{dm}(\theta) = \mathbb{E}_{x\sim p(x), \epsilon\sim\mathcal{N}(0,\sigma^2 I)} \|D_\theta(x+\epsilon, \sigma) - x\|^2$$
3. 同时训练 value estimator $V_\phi$（公式 8）

**Phase B：合成 + 选择**

4. 采样 $K$ 步 ODE（公式 1），从噪声 $x_K$ 生成合成转移 $x_0$
5. 用 $V_\phi$ 计算每个合成转移的估计价值
6. Select：
   - 计算 softmax 概率：$p_i \propto \exp(V_\phi(o_i)/\tau)$
   - 按 $p_i$ 采样，保留 $\eta$ 比例的转移
   - 形成最终合成数据集 $D_{syn}$

**Phase C：训练策略**

7. 将 $D_{syn}$ 输入任意离线 MARL 算法（MA-ICQ/MA-CQL/OMAR/MA-BCQ）

### 关键超参数

- $\eta$：select 保留比例，0.8 在大多数任务中最优
- $\tau$：select 的 temperature（Softmax 策略的平滑度）
- 合成数据集大小：5M 转移最佳（Figure 6，更大趋于饱和）
- 扩散步数 $K$：EDM 的 ODE 积分步数
- $\sigma_{max}$：最大噪声水平
- Bit 长度 $L = \lceil\log_2 M\rceil$

### 消融验证（Figure 4）

| 变体 | 相对 INS | 说明 |
|------|:---:|------|
| w/o Sparse | 下降 | Dense attention 引入无关交互干扰 |
| w/o Select | 下降 | 不筛选，合成数据中低质量转移占比高 |
| w/o Attention | 大降 | MLP 替代 attention，各智能体独立合成 |

---

## 实验与消融

### 环境与数据集

| 环境 | 动作空间 | 任务数 | 数据质量 | 数据集来源 |
|------|:---:|:---:|------|------|
| **MPE** | 连续 | 4 (Spread, Tag, World, 另有一个) | expert / medium / medium-replay / random | 标准离线 MARL 基准 |
| **SMAC** | 离散 | 3 (3m, 5m_vs_6m, 8m) | good / medium / poor | 标准 SMAC offline 基准 |

### Baselines

- **Original**：直接用原始数据集训练（无数据合成）
- **MA-SynthER**：SynthER (Lu et al., 2024) 的多智能体版本——MLP-based 扩散模型，不做交互建模
- 排除：Additive Noise / VAE Augmented（初步实验性能太差）

### 下游算法（4 个）

- **MA-ICQ** (Yang et al., 2021)：重要性采样约束
- **MA-CQL** (Kumar et al., 2020)：保守 Q 学习
- **OMAR** (Pan et al., 2022)：CQL + actor rectification
- **MA-BCQ** (Wu et al., 2019)：批量约束 Q 学习

### 主要结果 (Table 1)

**MPE 环境 (连续动作，总计分)**：

| 下游算法 | Original | MA-SynthER | INS |
|------|:---:|:---:|:---:|
| MA-ICQ | 550.0 | 561.9 | **573.0** |
| MA-CQL | 520.6 | 510.3 | **529.1** |
| OMAR | 708.0 | 689.3 | **725.9** |

- INS 在 3/3 算法上全面超越 Original 和 MA-SynthER
- 注意：MA-SynthER + OMAR 组合反而**低于** Original（689.3 < 708.0），说明忽视交互的数据合成可能起反作用

**SMAC 环境 (离散动作，总计分)**：

| 下游算法 | Original | MA-SynthER | INS |
|------|:---:|:---:|:---:|
| MA-ICQ | 133.6 | 137.8 | **143.3** |
| MA-CQL | 109.4 | 105.9 | **120.0** |
| MA-BCQ | 33.5 | 34.3 | **35.8** |

- INS 在全部 3/3 算法上胜出
- SMAC 上的提升幅度比 MPE 更显著（MA-CQL: +10.6 分, +9.7%）

### 数据集质量分析 (Figure 3)

5 个评估指标：

| 指标 | 含义 | INS vs MA-SynthER |
|------|------|:---:|
| Similarity | 合成数据与原始数据分布距离 | INS 更近 |
| Correlation | 统计相关性 | INS 更高 |
| Oracle Reward | 真实环境奖励 | INS 更高（select 机制的效果）|
| Dynamic MSE | 动力学一致性 | INS 更低（交互建模的效果）|
| Novelty | 覆盖度/新颖性 | INS 更高 |

### 消融分析 (Figure 4 & 5)

| 消融 | 结论 |
|------|------|
| 模块消融：w/o Sparse / w/o Select / w/o Attention | 三个模块对性能都有显著贡献；attention 机制（含 sparse）贡献最大 |
| $\eta$ 选择比例 | 0.8 在多数任务上最优；不同环境偏好不同（Spread 喜欢高 $\eta$，5m_vs_6m 需要更多样性）|
| Select 类型：Softmax vs Top-K | Softmax 选择策略始终优于 Top-K（证明概率性选择比硬截断更好平衡多样性和质量）|
| 合成数据集大小 | 0.1M→5M 持续提升，5M→10M 趋于饱和 |

### 小数据集实验 (Table 2)

| 数据集 | Original(100%) | 10% INS | 50% INS | 100% INS |
|------|:---:|:---:|:---:|:---:|
| Spread | 102.8 | 102.3 | 106.3 | **107.0** |
| 5m_vs_6m | 16.3 | 16.3 | 16.8 | **17.2** |

- 仅用 10% 原始数据合成的数据集，就能达到与 100% 原始数据相当的性能
- 50% INS 已经显著超越 100% Original

### 可视化（Figures 7 & 8）
- **t-SNE (Figure 7)**：INS 合成数据覆盖了原始数据的分布间隙（interpolation 效应），同时向外扩展了覆盖范围
- **Attention Weights (Figure 8)**：稀疏注意力权重与智能体间距离的倒数（$1/d(i,j)$）高度一致，dense 注意力则无视距离

### 实验结论的可靠性

- **强点**：多环境/多算法/多指标/消融全面的评估体系；小数据集实验具有很强的实用价值
- **弱点**：(1) 未在更大规模环境（如 SMACv2、MaMuJoCo）评估；(2) 扩散模型训练和采样开销较大，未与更轻量的增强方法（如 random perturbation）系统对比效率；(3) select 机制依赖 value estimator 质量，后者本身面临离线 OOD 估计问题

---

## 展望

### 研究启发

1. **"数据增强"路线在离线 MARL 中被严重低估**：绝大多数离线 MARL 工作集中在算法层面（约束、保守估计、正则化），INS 有力地证明了数据合成是一条同样有效且正交的路线——合成数据可以与任何算法叠加使用。
2. **Sparsemax 在 MARL 中有更广阔的应用场景**：除了数据合成中的 attention，Sparsemax 也可以用于策略网络中的通信选择、信用分配中的相关性学习等需要筛选"真正相关的智能体"的场景。
3. **Bit action 是连接离散动作和扩散模型的简洁方案**：与 Gumbel-Softmax 或 one-hot 扩散相比，bit action 在维度压缩和精度之间取得了好的平衡，适用于各类需要扩散模型处理离散变量的 MARL 场景。
4. **Select 机制 = value-weighted sampling，可迁移到其他数据增强方法**：用 value function 评估转移质量并据此加权/筛选，这一思路不限于扩散模型——GAN、VAE 甚至 heuristic augmentation 都可以引入类似机制。
5. **10% 数据 → 100% 效果的能力具有重大实际意义**：真实的机器人多智能体数据收集极其昂贵，INS 的小数据集实验直接指向了最有应用价值的场景。

### 局限与开放问题

1. **扩散模型的训练和采样计算开销**：EDM 的 ODE 积分需要多步（通常 $>32$），对于大规模 MARL 系统可能成为瓶颈
2. **可变数量智能体的场景**：INS 的 Transformer 架构假设固定数量的智能体，对于智能体数量动态变化的环境（如某些 ad hoc teamwork 场景）不直接适用
3. **未做 guided generation**：论文选择不做引导（guidance）以平衡多样性和保真度，但在安全关键型 MARL 中，引导合成可能格外重要
4. **Value estimator 的准确性是 select 机制的瓶颈**：离线训练的 value function 本身面临 OOD 过估计，如果 $V_\phi$ 估计不准，select 可能反而选到"虚高"的转移
5. **合成数据的"真实性"保证**：没有理论保证合成转移一定满足环境动力学（虽有 Dynamic MSE 评估），极端情况下可能合成物理上不可能的转移

### 可跟进的研究方向

1. **Guided Data Synthesis for Safety-Critical MARL**：为 INS 引入 safety-guided 扩散（类似 classifier guidance），确保合成转移避开危险区域。动机：自动驾驶/无人机编队场景需要合成数据既多样又安全。
2. **INS + Offline-to-Online Fine-tuning**：用 INS 合成的数据做 offline 预训练，再在 online 环境中 fine-tune。动机：合成数据填补了 offline 数据的覆盖空白，可能为 online fine-tuning 提供更好的初始化。
3. **轻量化 INS**：用 consistency models 或蒸馏技术加速扩散采样，降低 INS 的计算开销。动机：当前 5M 合成样本需要昂贵采样，轻量化后可推广到实时/大规模场景。

---

## 链接

- **OpenReview（ICLR 2025 正式版）**：[openreview.net/forum?id=kxD2LlPr40](https://openreview.net/forum?id=kxD2LlPr40)
- **代码**：论文摘要声明提供了源代码链接（here），具体仓库地址需查看论文原文
