---
title: "Semantic Temporal Abstraction via Vision-Language Model Guidance for Efficient Reinforcement Learning"
description: "VanTA 用 VLM 知识引导 VQ-VAE 技能发现，让离线层级 RL 的时序抽象更有语义且更高效。"
tags:
  - Offline RL
  - Temporal Abstraction
  - Vision-Language Models
  - Hierarchical RL
  - Skill Discovery
---

# Semantic Temporal Abstraction via Vision-Language Model Guidance for Efficient Reinforcement Learning

<div class="paper-hero">
<p class="paper-meta">Tian-Shuo Liu, Xu-Hui Liu, Ruifeng Chen, Lixuan Jin, Pengyuan Wang, Zhilong Zhang, Yang Yu · ICLR 2025 · ICLR / 2025</p>
<div class="tag-row"><span>Offline RL</span><span>Temporal Abstraction</span><span>Vision-Language Models</span><span>Hierarchical RL</span><span>Skill Discovery</span></div>
</div>

<article class="note-body" markdown>

| 字段 | 内容 |
|---|---|
| Title | Semantic Temporal Abstraction via Vision-Language Model Guidance for Efficient Reinforcement Learning |
| Year | 2025 |
| Source | ICLR 2025 Poster；OpenReview 显示 Published 22 Jan 2025，Last Modified 16 Mar 2025 |
| Authors | Tian-Shuo Liu, Xu-Hui Liu, Ruifeng Chen, Lixuan Jin, Pengyuan Wang, Zhilong Zhang, Yang Yu |
| Affiliations | National Key Laboratory for Novel Software Technology, Nanjing University；School of Artificial Intelligence, Nanjing University；Polixir Technologies |
| Tags | Temporal Abstraction；VLM；VQ-VAE；Offline RL；Hierarchical RL |

**一句话概括**：VanTA 让 VLM 参与离线轨迹的技能重标注，把原本无监督、碎片化的 VQ-VAE 时序抽象拉向语义一致的子任务，从而提升长时程稀疏奖励任务中的离线层级 RL。

**我对这篇论文的定位**：这篇论文处在 temporal abstraction、offline RL 和 foundation model guidance 的交界。它不是让 VLM 直接控制机器人，而是把 VLM 当作“语义裁判”，帮助离线技能发现形成更像人类子任务的分段。对 ORL 研究者来说，VanTA 的价值在于提供了一种低人工标注成本的语义技能发现方式。

## 1. 第一作者相关信息

Tian-Shuo Liu、Xu-Hui Liu、Ruifeng Chen 为共同一作，均与南京大学 LAMDA/人工智能学院及 Polixir Technologies 相关；Yang Yu 为通讯作者。Tian-Shuo Liu 近期工作与离线/在线 RL、扩散模型辅助采样和时序抽象高度相关。

| 年份 | 论文 | Venue/source | 主题标签 | 与本文关系 |
|---|---|---|---|---|
| 2025 | Semantic Temporal Abstraction via VLM Guidance for Efficient RL | ICLR 2025 Poster | VLM-guided temporal abstraction | 本文主体 |
| 2024 | Energy-Guided Diffusion Sampling for Offline-to-Online RL | ICML 2024 | Diffusion, offline-to-online RL | 同团队关于生成模型辅助 RL 的工作 |
| 2025 | LAMDA/Polixir 离线 RL 与层级 RL 相关工作 | OpenReview/团队论文 | Offline RL, skill learning | 方法背景与实验实现相关 |

作者团队长期关注离线 RL、扩散模型、层级 RL 与理论分析。VanTA 可看作这一研究线从“生成轨迹/状态”转向“生成或修正决策抽象”的一步。

## 2. 研究问题

长时程稀疏奖励任务很难直接用低层动作学习，因为信用分配长、探索空间大。时序抽象希望把轨迹切成技能或子任务，例如“打开微波炉”“移动水壶”。问题是：纯无监督技能发现常产生碎片化、无语义的 code；人工标注又昂贵且不可扩展。

VanTA 的设置是：给定离线轨迹数据，先学习离散技能 code，再训练高层策略选择技能、低层策略执行技能。它既不是在线探索方法，也不是直接用语言奖励训练策略，而是用 VLM 对轨迹片段进行语义重标注，改造技能 latent space。

形式上，轨迹片段被编码为离散技能 $z$，高层策略学习 $\pi_H(z\mid s)$，低层策略学习 $\pi_L(a\mid s,z)$。目标是让 $z$ 同时满足可执行性和语义一致性。

## 3. 背景知识

Temporal abstraction 的核心是用一个高层动作代表多个低层时间步。这样原本 $H$ 步的决策可被压缩为 $H/K$ 个技能选择，信用分配也从单步奖励变成片段级贡献。

VQ-VAE 用 codebook 把连续表示量化成离散 code。普通 VQ-VAE 只会根据重建或动作预测误差组织 latent space，不知道“打开微波炉”和“把手移动一点点”哪一个更像子任务。因此它可能频繁切换 code，产生没有语义边界的分段。

VLM 的作用是提供外部语义知识。VanTA 把技能片段的关键帧或首尾帧交给 VLM，让它判断该片段属于哪个语义技能，再用这个重标注信号逐步更新 codebook。

| 术语 | 直观含义 | 在本文中的作用 |
|---|---|---|
| Temporal abstraction | 用多步技能替代单步动作 | 降低长时程决策难度 |
| Skill / option | 一段可复用行为片段 | 高层策略选择对象 |
| VQ-VAE | 把连续表示映射到离散 code | 初始技能发现工具 |
| VLM relabeling | 用视觉语言模型给片段赋语义标签 | VanTA 的关键引导信号 |
| Codebook shaping | 调整离散技能向量空间 | 让技能更语义化、更稳定 |
| Concentrability | 目标策略分布相对离线数据覆盖的难度 | 理论上解释抽象为何降低学习难度 |

为什么动机成立：无监督抽象能压缩空间，但不保证抽象对任务有用；VLM 有语义知识，但直接控制动作不可靠。VanTA 把两者分工：VQ-VAE 负责从动作数据中找到可执行片段，VLM 负责把片段边界和 code 推向人类可解释的子任务。

## 4. 问题分析

Figure 2 展示无 VLM guidance 时的技能抽取：code 切换频繁，片段缺少清晰语义。Figure 3 展示 VanTA 后的分段：技能更像自然子任务，例如按顺序打开微波炉、移动物体等。

论文认为问题不只是可解释性，而是策略学习效率。若 code 频繁切换，高层策略的动作空间虽然离散，但仍然碎片化；低层策略也要学习语义不一致的行为分布。VanTA 增强同一子任务内部的顺序相关性，使每个技能更稳定。

理论部分使用已有 offline RL/行为克隆分析说明：更强的技能内部相关性和更小的有效策略空间会降低下游学习的 suboptimality。Theorem 5.3 与 Theorem 5.6 支撑这种“好的抽象降低学习误差”的方向性结论。

## 5. 思想与方法

VanTA 的核心思想是渐进式语义塑形：不要一次性让 VLM 决定所有动作，而是在 VQ-VAE 学到初始技能后，让 VLM 反复重标注片段，再用重标注结果以 EMA 方式更新 codebook。

方法组件包括：状态/动作编码器、VQ codebook、policy decoder、VLM relabeler、高层离散 Q-learning、低层加权行为克隆。与普通 VQ-VAE 相比，VanTA 的 decoder 更关注动作可执行性，而不是图像或状态重建。

训练损失可以概括为：动作预测损失 + VQ commitment/codebook 损失 + VLM guidance 损失 + 平滑损失。直观上，前两者保证 code 能解释数据，VLM 项保证 code 有语义，平滑项减少无意义抖动。

该方法的新意在于把 VLM 放在技能发现的中间层，而不是作为 reward model 或 policy prompt。它借用 foundation model 的语义，但最终策略仍由离线 RL/BC 训练得到。

## 6. 算法与伪代码

算法名：VanTA (Vision-language model guided Temporal Abstraction)。

1. 输入离线轨迹数据 $D$。
2. 训练 VQ-VAE 式技能编码器，把连续轨迹片段映射到离散 code $z$。
3. 根据连续相同 code 的区间形成初始技能片段。
4. 对每个片段抽取视觉观测或首尾帧，构造 VLM 查询。
5. VLM 判断片段语义类别或最匹配的技能描述。
6. 用 VLM 重标注结果更新 codebook 目标向量，通常采用 EMA 平滑。
7. 重复 VQ 训练与 VLM relabeling，直到技能分段稳定。
8. 用最终技能标签训练高层策略 $\pi_H(z\mid s)$。
9. 用加权 BC/IQL 风格目标训练低层策略 $\pi_L(a\mid s,z)$。
10. 测试时高层选择技能，低层在技能条件下输出原始动作。

高层 Bellman 目标可简化为：

$$
Q(s_t,z_t) \leftarrow r_{t:t+K-1} + \gamma^K \max_z Q(s_{t+K},z).
$$

含义：一个技能 $z_t$ 覆盖 $K$ 个底层时间步，因此折扣因子变成 $\gamma^K$，奖励也聚合为片段奖励。

## 7. 实验与消融

实验覆盖 Franka Kitchen、MiniGrid、Crafter，包含本体状态和视觉观测任务。Table 1 报告主结果：VanTA 在多类长时程任务上超过 OPAL、LDCQ、RvS/GCSL 等无监督或启发式抽象 baseline，也优于常规 offline RL 方法。

Figure 2-3 是定性可视化，说明 VanTA 产生更清晰的语义技能。Table 2 显示在减少离线数据时，VanTA 的优势更明显，说明 VLM guidance 对小数据技能发现有帮助。Table 3 比较低层策略空间，支持“更好的抽象降低策略复杂度”的说法。Figure 4 对比有无 VLM guidance，显示错误或破碎 codebook 会伤害下游回报。

实验较强的证据是：语义抽象确实改善了长期任务表现，尤其在数据有限时。较弱之处是：VLM 的选择、prompt、候选技能描述集合会影响结果；真实机器人中 VLM 对视觉帧的误判可能引入系统性偏差。

## 8. 展望

启发：第一，foundation model 可以作为离线 RL 中间表示的语义约束，而不必直接输出动作。第二，技能发现的可解释性不仅是展示价值，也会影响高层策略学习。第三，VLM guidance 可以降低人工分段成本，适合长时程机器人数据。

局限：VLM 依赖视觉可见性，隐藏状态或触觉任务较难；候选语义集合如果设计不好，会限制技能空间；多模态 VLM 的错误会被写入 codebook；理论分析仍较抽象，和具体 VLM 误差之间的关系未闭合。

后续方向：1. 引入 uncertainty-aware VLM relabeling，只在 VLM 置信度高时更新 codebook。2. 将语言指令与 VanTA 技能 code 对齐，支持可组合任务。3. 在真实机器人长轨迹上评估 VLM 误标注对 offline HRL 的影响。

## Links

- Paper page: https://openreview.net/forum?id=zY37C8d6bS
- OpenReview PDF: https://openreview.net/pdf?id=zY37C8d6bS
- Related ICML 2024 work: https://proceedings.mlr.press/v235/liu24ao.html
- Code: 论文说明代码在 supplementary materials；未能从公开页面确认独立官方仓库。
- 本地 PDF: `[ICLR_2025] Semantic Temporal Abstraction via Vision-Language Model Guidance for Efficient Reinforcement Learning.pdf`

</article>
