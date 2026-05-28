---
title: "RTDiff: Reverse Trajectory Synthesis via Diffusion for Offline Reinforcement Learning"
description: "RTDiff 通过反向合成轨迹进行离线数据增强，降低 forward synthesis 从已知区域走向 OOD 区域带来的过估计风险。"
tags:
  - Offline RL
  - Diffusion Models
  - Data Augmentation
  - Trajectory Synthesis
  - Distribution Shift
---

# RTDiff: Reverse Trajectory Synthesis via Diffusion for Offline Reinforcement Learning

<div class="paper-hero">
<p class="paper-meta">Qianlan Yang, Yu-Xiong Wang · ICLR 2025 · ICLR / 2025</p>
<div class="tag-row"><span>Offline RL</span><span>Diffusion Models</span><span>Data Augmentation</span><span>Trajectory Synthesis</span><span>Distribution Shift</span></div>
</div>

<article class="note-body" markdown>

| 字段 | 内容 |
|---|---|
| Title | RTDiff: Reverse Trajectory Synthesis via Diffusion for Offline Reinforcement Learning |
| Year | 2025 |
| Source | ICLR 2025 Poster；OpenReview 显示 Published 22 Jan 2025，Last Modified 02 Mar 2025 |
| Authors | Qianlan Yang, Yu-Xiong Wang |
| Affiliations | University of Illinois Urbana-Champaign |
| Tags | Offline RL；Diffusion Data Augmentation；Reverse Trajectory Synthesis；OOD Detection；D4RL |

**一句话概括**：RTDiff 用扩散模型从已知状态向过去反向生成轨迹，把数据增强的方向从“走向未知”改成“追溯如何到达已知”，从而减少合成 OOD 转移导致的价值过估计。

**我对这篇论文的定位**：这是 offline RL 数据增强方向的扩散模型论文。它的关键不是更复杂的 diffusion backbone，而是方向性的改变：forward synthesis 可能生成从数据支持区通向未知区的危险轨迹，reverse synthesis 则让合成轨迹终点落在已知数据状态附近，更符合离线安全直觉。

## 1. 第一作者相关信息

Qianlan Yang 来自 UIUC，合作者 Yu-Xiong Wang 也来自 UIUC。项目页为 `https://yanqval.github.io/RTDiff`，OpenReview 与 ICLR proceedings 均收录该文。Qianlan Yang 相关研究包括 ATraDiff 与 RTDiff，集中在 diffusion trajectory synthesis 与 RL 数据增强。

| 年份 | 论文 | Venue/source | 主题标签 | 与本文关系 |
|---|---|---|---|---|
| 2025 | RTDiff: Reverse Trajectory Synthesis via Diffusion for Offline RL | ICLR 2025 Poster | Reverse trajectory synthesis | 本文主体 |
| 2024 | ATraDiff | ICML/相关工作 | Forward trajectory synthesis | 本文比较和动机来源 |
| 2025 | RTDiff project page | Project/GitHub page | Code and demos | 论文资源入口 |

作者研究线很清晰：用生成模型补充 RL 数据，但重点从“多生成”转向“生成方向和安全性”。

## 2. 研究问题

离线 RL 的分布偏移会导致 OOD 动作或状态价值被高估。保守算法通过惩罚 OOD 区域缓解这个问题，但可能过度保守，限制泛化。生成式数据增强试图合成更多轨迹来扩大数据覆盖，但 forward synthesis 会产生新的风险：从已知状态向未来生成时，模型可能把轨迹带到数据未覆盖区域，并错误赋予高价值。

RTDiff 的问题是：如何利用扩散模型生成更长、更有信息量的轨迹，同时不把 agent 引向未知区域？答案是 reverse trajectory synthesis：给定离线数据中的已知状态 $s_0$，生成一段通向该状态的过去轨迹。

形式上，合成轨迹可以表示为：

$$
\tau^- = (s_{-L},a_{-L},r_{-L},\ldots,s_{-1},a_{-1},r_{-1},s_0),
$$

其中 $s_0$ 来自真实离线数据。这样即使合成的早期部分有误，策略部署时也不会从 $s_0$ 主动走向未知区域。

## 3. 背景知识

Offline RL 中的过估计通常来自“数据里没有但 Q 值很高”的区域。数据增强如果生成了 OOD 转移，就可能让算法更相信这些虚假高价值样本。

Forward synthesis 从当前已知状态向未来生成。这和真实部署方向一致，但离线环境无法验证未来是否安全或可达。Reverse synthesis 反过来问“哪些过去状态可能到达这个已知状态”，它把合成片段锚定到真实数据终点。

扩散模型适合多步轨迹合成，因为它可以生成连续、多样且长 horizon 的序列。RTDiff 使用 EDM 风格扩散模型和 MLP backbone，处理低维 proprioceptive 状态；视觉任务也在 Meta-World 中验证。

| 术语 | 直观含义 | 在本文中的作用 |
|---|---|---|
| Forward synthesis | 从已知状态向未来生成 | 主要风险对照 |
| Reverse synthesis | 从已知状态向过去生成 | RTDiff 核心 |
| In2Out transition | 从数据内状态转到 OOD 状态 | 过估计来源 |
| OOD detector | 判断状态是否离开数据支持 | 控制生成长度 |
| Noise control | 控制扩散噪声多样性 | 减少冗余合成 |
| Dataset augmentation | 合成轨迹并加入离线数据 | 最终用途 |

为什么动机成立：离线 RL 不能试错验证合成未来；因此“向未知走”的错误比“从未知回到已知”的错误更危险。RTDiff 利用方向不对称性，把生成风险转化为更不容易误导部署策略的形式。

## 4. 问题分析

Figure 1 给出 RTDiff 主框架和直观示例：红色 forward 轨迹可能穿过危险或未知区域，绿色 reverse 轨迹则从未知方向回到已知轨迹附近。Figure 2 进一步可视化正常/反向合成的差异。

Table 8 是核心定量证据：在 Maze2D-large 上，normal/forward synthesis 产生明显更多 In2Out 转移，而 reverse synthesis 的 In2Out 比例显著更低。旧笔记记录的数字为 forward 约 11.2%、reverse 约 2.7%；论文还指出即使用 OOD detector 修剪 forward synthesis，仍可能保留较高 In2Out 风险。

问题本质不是扩散模型生成质量不够，而是生成方向与离线部署风险不匹配。RTDiff 将方向作为安全约束的一部分。

## 5. 思想与方法

RTDiff 的指导原则是：让合成轨迹的可靠端点来自真实数据。给定数据集中的状态作为锚点，扩散模型生成其反向历史；再把真实数据和合成数据混合训练 CQL、DT 等 offline RL 算法。

方法有三项关键组件：反向轨迹扩散模型、OOD detector 进行自适应长度控制、noise management 提高生成多样性和效率。

OOD detector 采用类似 SSD/Mahalanobis 距离的方式：在离线状态特征空间聚类，若生成状态离最近簇过远，则停止继续延长轨迹。这样生成长度不是固定的，而是由数据支持边界决定。

Noise control 的作用是避免从同一锚点生成大量高度相似的轨迹，提高每次合成的边际信息量。

## 6. 算法与伪代码

算法名：RTDiff（论文 Algorithm 1: Augment Offline Dataset D with RTDiff）。

1. 输入离线数据集 $D$。
2. 将每条真实轨迹反向排列，构造反向轨迹训练样本。
3. 训练条件扩散模型 $p_\theta(\tau^-\mid s_0)$，给定真实状态 $s_0$ 生成过去轨迹。
4. 训练 OOD detector，用于判断生成状态是否超出数据支持。
5. 从 $D$ 中采样锚点状态 $s_0$。
6. 用扩散模型生成反向轨迹候选。
7. 沿生成序列检查 OOD detector；一旦超过阈值，截断该轨迹。
8. 应用 noise control，减少重复或过近样本。
9. 将合成反向轨迹加入原数据集，形成 $D' = D \cup D_{syn}$。
10. 用 CQL、Decision Transformer 等 offline RL 算法在 $D'$ 上训练策略。

扩散目标可概括为标准噪声预测：

$$
\mathcal{L}_{diff}=\mathbb{E}_{\tau,\epsilon,t}\left[\|\epsilon - \epsilon_\theta(\alpha_t \tau + \sigma_t\epsilon, t, s_0)\|^2\right].
$$

含义：模型学习在给定锚点状态 $s_0$ 时，从噪声中恢复反向轨迹。

## 7. 实验与消融

实验覆盖 D4RL Maze2D、AntMaze、Kitchen、Locomotion，以及 Meta-World visual RL。Table 1-2 报告主结果：RTDiff 加到多种 offline RL 算法上通常带来稳定提升。Table 6 与 Table 7 进一步验证 Maze2D 和视觉 Meta-World。

baseline 包括 SynthER、ATraDiff、S4RL、ROMI，以及原始 CQL/DT 等。RTDiff 的比较重点是：同样用生成数据增强，反向生成比正向生成更少诱导过估计。

消融包括：Table 3 比较 reverse synthesis 与 normal synthesis；Table 4 分析生成长度；Table 5 分析 noise control；Table 8 分析不同转移类型比例；Table 13-15 讨论 OOD threshold 和 forward 方法加 OOD detector 后的表现。

实验能证明的是：在 D4RL 长时程导航和控制任务中，反向合成作为数据增强更稳。尚需验证的是：真实机器人中反向轨迹是否总是可逆、非可逆动力学和接触丰富任务中反向生成是否会产生物理不一致。

## 8. 展望

启发：第一，生成式数据增强不仅要看样本质量，还要看生成方向。第二，离线 RL 中“从已知到未知”的合成最危险，OOD detector 应该参与生成过程而不是只做后处理。第三，反向轨迹可为稀疏奖励导航提供更丰富的到达路径。

局限：反向轨迹不一定对应真实可逆动力学；OOD detector 阈值敏感；扩散生成成本较高；在高维视觉和真实物理环境中，合成轨迹的可执行性仍需额外检查。

后续方向：1. 结合动力学一致性约束，过滤不可逆反向轨迹。2. 将 RTDiff 与 uncertainty-aware Q-learning 结合，合成样本按置信度加权。3. 在机器人 manipulation 中研究接触事件的反向合成是否可靠。

## Links

- Paper page: https://openreview.net/forum?id=0FK6tzqV76
- ICLR proceedings: https://proceedings.iclr.cc/paper_files/paper/2025/hash/45d4924460c37853d57885d8af0b8d5c-Abstract-Conference.html
- Project / Code: https://yanqval.github.io/RTDiff
- DBLP: https://dblp.org/rec/conf/iclr/YangW25
- 本地 PDF: `[ICLR_2025] RTDiff Reverse Trajectory Synthesis via Diffusion for Offline Reinforcement Learning.pdf`

</article>
