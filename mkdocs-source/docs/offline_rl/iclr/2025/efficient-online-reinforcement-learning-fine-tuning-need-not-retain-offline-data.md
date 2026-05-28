---
title: "Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data"
description: "一篇系统研究 offline-to-online RL 中无离线数据保留微调问题，并提出 Warm-start RL 的论文。"
tags:
  - Offline-to-Online RL
  - Fine-tuning
  - No-retention RL
  - Catastrophic Forgetting
  - Warm-start RL
---

# Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data

<div class="paper-hero">
<p class="paper-meta">Zhiyuan Zhou, Andy Peng, Qiyang Li, Sergey Levine, Aviral Kumar · ICLR 2025 · ICLR / 2025</p>
<div class="tag-row"><span>Offline-to-Online RL</span><span>Fine-tuning</span><span>No-retention RL</span><span>Catastrophic Forgetting</span><span>Warm-start RL</span></div>
</div>

<article class="note-body" markdown>

| Field | 内容 |
|---|---|
| Title | Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data |
| Year | 2025 |
| Source | ICLR 2025 conference paper |
| Authors | Zhiyuan Zhou, Andy Peng, Qiyang Li, Sergey Levine, Aviral Kumar |
| Affiliations | UC Berkeley；Carnegie Mellon University |
| Tags | Offline-to-online RL, fine-tuning, no data retention, warmup, SAC, catastrophic forgetting |

**一句话概括**：本文指出 offline-to-online RL 微调阶段并不一定要继续保留离线数据，只要先用冻结的预训练策略收集少量 warmup rollout 来校准 Q 函数，再用高 UTD 的标准在线 SAC 微调，就能避免灾难性遗忘并获得更快、更高的在线性能。

**我对这篇论文的定位**：这篇论文把“离线预训练 + 在线微调”重新对齐到机器学习中常见的预训练范式：预训练数据在 fine-tuning 阶段可以被丢弃。它的贡献不是复杂算法，而是对 offline-to-online RL 失败机制的诊断：保留离线数据主要是在 fine-tuning 初期防止 Q 函数因分布偏移而发散，但长期会拖慢在线优化。WSRL 的价值在于用一个极简 warmup 机制替代离线数据保留。

## 1. 第一作者相关信息

本文由 **Zhiyuan Zhou** 和 **Andy Peng** 共同一作完成。论文署名显示 Zhiyuan Zhou、Andy Peng、Qiyang Li、Sergey Levine 来自 UC Berkeley，Aviral Kumar 来自 Carnegie Mellon University。

Zhiyuan Zhou 的公开研究轨迹与机器人学习、强化学习预训练/微调和大规模决策模型相关；Andy Peng 也与 Berkeley 机器人学习/强化学习生态高度相关。共同作者 Sergey Levine 和 Aviral Kumar 是离线 RL 与机器人学习领域的重要研究者：CQL、CalQL、IQL 相关研究均与这个研究网络紧密相关。因此本文很像一篇“对自己领域常用范式的反思”：既分析 CQL/IQL/CalQL 在 no-retention fine-tuning 中为什么失败，也提出一个更贴近大规模预训练实践的替代方案。

| Year | Title | Venue/Source | Topic tag | Relation to this paper |
|---|---|---|---|---|
| 2020 | Conservative Q-Learning | NeurIPS | Conservative Offline RL | 本文分析 CQL 无保留微调时的遗忘问题 |
| 2021/2022 | Offline RL with Implicit Q-Learning | ICLR | In-sample Offline RL | 本文将 IQL 作为 no-retention baseline |
| 2023 | RLPD | ICLR / related online RL line | High-UTD Online RL | WSRL 的在线微调实现借鉴其 SAC、Q ensemble、LayerNorm 设计 |
| 2024 | CalQL | NeurIPS / arXiv line | Offline-to-online RL | 本文重点比较并指出 CalQL 仍依赖离线数据保留 |
| 2025 | Efficient Online RL Fine-Tuning Need Not Retain Offline Data | ICLR | No-retention fine-tuning | 本文提出 WSRL |

**研究方向综合**：这篇论文的作者群体关注“如何让 RL 像监督学习/大模型一样预训练后高效微调”。它把问题从“如何设计更强离线 RL 正则”转到“预训练初始化怎样被在线 RL 正确接住”。

**信息来源**：论文首页署名、论文附录代码链接、作者机构信息；近期相关工作来自论文引用与公开论文记录。作者完整发表记录需以个人主页或 DBLP 为准。

## 2. 研究问题

本文研究的是 **no-retention online fine-tuning**：给定通过离线 RL 预训练得到的策略 $\pi_{\psi}^{pre}$ 和 Q 函数 $Q_{\theta}^{pre}$，在线微调阶段不能继续使用原始离线数据集 $\mathcal{D}_{off}$，只能使用在线交互数据。

形式化地，MDP 为

$$
\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma,\rho),
$$

目标是最大化在线策略的折扣回报：

$$
J(\pi)=
\mathbb{E}_{s_0\sim \rho,\ a_t\sim \pi(\cdot|s_t),\ s_{t+1}\sim P(\cdot|s_t,a_t)}
\left[\sum_{t=0}^{\infty}\gamma^t r(s_t,a_t)\right].
$$

关键限制是：微调时只能访问 $\pi_{\psi}^{pre}$ 和 $Q_{\theta}^{pre}$，不能再从 $\mathcal{D}_{off}$ 中采样。

为什么这个问题重要：

1. 大规模离线数据集越来越大，微调时继续反复训练离线数据会非常慢。
2. 离线数据上的保守约束可能限制在线阶段继续提升的上限。
3. 如果预训练范式成立，离线数据应像监督预训练数据一样在 fine-tuning 后可被丢弃。

已有方法的失败模式是：直接丢弃离线数据后，CQL、IQL、CalQL 等会在 fine-tuning 初期发生 unlearning，甚至 catastrophic forgetting，导致预训练优势被破坏，后续在线训练难以恢复。

## 3. 背景知识

**Offline-to-online RL**：先用固定离线数据训练一个策略和价值函数，再让智能体与环境交互进行在线微调。这一范式希望结合离线数据的先验知识和在线交互的适应能力。

**数据保留 fine-tuning**：传统做法是在在线阶段继续把离线数据放进 replay buffer，或者每个 batch 混合离线数据和在线数据。这可以稳定 Q 函数，但会带来计算成本和保守性。

**No-retention fine-tuning**：本文强调的新设置。离线数据只用于预训练，之后完全丢弃；在线阶段的 replay buffer 初始为空或只由在线数据构成。

**Unlearning vs. Forgetting**：

- Unlearning：微调初期性能下降，但后续可以恢复。
- Forgetting：预训练得到的有用价值函数/策略结构被破坏，后续很难恢复。

本文认为 no-retention 设置中一定程度的 unlearning 可能不可避免，但必须避免 forgetting。

**Q 函数校准与分布偏移**：离线预训练得到的 $Q^{pre}$ 在离线数据分布上有意义，但在线 rollout 产生的状态动作分布不同。若只用少量在线数据做 TD 更新，Q 函数会迅速重新校准；这个过程中若 TD target 过低或过保守，可能导致 Q 值进入 downward spiral。

**高 UTD 在线 RL**：UTD（updates-to-data ratio）表示每收集一步环境数据做多少次梯度更新。RLPD/REDQ 风格的高 UTD SAC 用 Q ensemble、LayerNorm、较高更新频率提升样本效率。WSRL 的在线阶段采用类似机制。

| Term | 朴素含义 | 在本文中的作用 |
|---|---|---|
| $\mathcal{D}_{off}$ | 离线预训练数据 | 微调阶段不允许使用 |
| $\pi^{pre}$ | 离线预训练策略 | warmup 数据收集和策略初始化 |
| $Q^{pre}$ | 离线预训练 Q 函数 | critic 初始化，帮助快速在线学习 |
| Retention | 微调时继续用离线数据 | 稳定但慢，可能限制性能 |
| Warmup | 冻结预训练策略收集少量在线数据 | 替代离线数据保留，桥接分布 |
| Downward spiral | Q 值低估和 TD target 低估互相放大 | no-retention 失败机制 |

**为什么这篇文章的动机成立**：如果微调必须反复使用离线数据，那么离线 RL 预训练与大规模 ML 预训练范式并不一致。论文通过 Section 3 的诊断证明，离线数据保留并非为了长期性能本身，而主要是在微调初期防止 Q 函数崩掉。因此可以寻找一个更便宜、更在线分布一致的替代物。

## 4. 问题分析

论文的分析从三个实验现象展开。

**现象一：直接 no-retention 微调会失败**  
Figure 2 显示，在 kitchen-partial 上，IQL、CQL、CalQL 若不保留离线数据，fine-tuning 开始后性能显著下降。CQL 和 IQL 近似归零且难以恢复；CalQL 稍稳定但无法明显超过预训练水平。

**现象二：离线分布上的 Q 拟合被破坏**  
Figure 3 逐步减少 batch 中离线数据比例。离线数据越少，离线分布上的 Q 值和 TD error 越容易发散；但在线分布上的 TD error 仍可以较小。这说明问题不是“在线数据学不了”，而是“在线学习破坏了离线预训练知识”。

**现象三：Q 值出现 downward spiral**  
Figure 4 和 Figure 5 展示 Q 值在 no-retention 微调初期会过度低估。因离线算法通常学习到保守 Q，在线 rollout 到新的状态动作后，TD target 可能偏低；低估又被 Bellman 更新继续传播，最终策略已遗忘预训练知识。

论文总结出三个 takeaway：

1. Offline/online 分布偏移会破坏 Q 函数在离线分布上的拟合。
2. Q 函数重新校准会造成过度低估，形成 downward spiral。
3. 保留离线数据能防止 early forgetting，但会损害渐近在线性能和效率。

Figure 6 支持第三点：保留离线数据的 fine-tuning 可能被从头在线训练的高效方法 RLPD 超过，说明离线数据在后期并非总是资产。

## 5. 思想与方法

WSRL 的核心思想非常简单：**用预训练策略在真实在线环境中先收集少量 warmup 数据，让 replay buffer 从一开始就包含接近预训练策略分布且属于在线环境的数据，然后丢弃离线数据，用标准在线 RL 快速微调**。

这个想法如何对应问题分析：

- 分布偏移问题：warmup 数据由 $\pi^{pre}$ 在在线环境中产生，既接近预训练策略，又来自真实在线分布。
- Q 函数校准问题：critic 在 warmup 后有足够初始在线 buffer，不会只基于极少量新数据剧烈发散。
- 渐近性能问题：warmup 之后不再运行 CQL/CalQL 这种 pessimistic offline RL，而是使用标准 SAC，因此不会被离线数据保守项持续束缚。

WSRL 的组件：

1. **初始化**：用离线 RL 得到的 $\pi^{pre}$ 和 $Q^{pre}$ 初始化在线策略和 critic。
2. **Warmup phase**：冻结 $\pi^{pre}$，在线交互 $K$ 步，把 transition 存入 replay buffer。
3. **Online fine-tuning**：使用高 UTD SAC，配合 Q ensemble 和 LayerNorm。

新颖性在于问题设定和机制解释，而不是算法复杂度。作者自己也强调 WSRL 不是“clever algorithm”，它更像是正确工程原则的复现：先让在线 replay buffer 有合适分布，再用高效在线 RL 接管。

证据强度：

- 机制证据：Figure 3-5 和 Figure 11 分析 Q 值、TD error、离线/在线分布。
- 性能证据：Figure 7、9、14、15 等在 AntMaze、Kitchen、Adroit、MuJoCo 上比较。
- 消融证据：Figure 10-13、16-17、22-24 分析 warmup、策略初始化、Q 初始化、warmup 长度、是否保留离线数据等。

## 6. 算法与伪代码

算法名：**WSRL: Warm Start Reinforcement Learning**。

**中文伪代码（对应 Algorithm 1）**：

1. 输入离线预训练策略 $\pi_{\psi}^{pre}$、离线预训练 Q 函数 $Q_{\theta}^{pre}$、在线 RL 算法 $\mathcal{A}_{on}$、UTD 比例 $M$、warmup 步数 $K$。
2. 初始化：
   1. 设置 $Q_\theta \leftarrow Q_{\theta}^{pre}$。
   2. 设置 $\pi_\psi \leftarrow \pi_{\psi}^{pre}$。
   3. 初始化空 replay buffer $\mathcal{R}$。
3. 对每个在线 step：
   1. 如果 step $\le K$，用冻结的 $\pi_{\psi}^{pre}$ 与环境交互，收集 $(s,a,s',r)$。
   2. 否则，用当前策略 $\pi_\psi$ 与环境交互，收集 $(s,a,s',r)$。
   3. 将 transition 加入 $\mathcal{R}$。
   4. 如果 step $> K$：
      1. 从 $\mathcal{R}$ 中采样 $M$ 个 batch。
      2. 对 critic 做 $M$ 次标准 TD update。
      3. 用 policy gradient 更新 actor，actor update delay 为 $M$。
4. 输出微调后的策略 $\pi_\psi$。

**在线 RL 实现**：

- 使用 SAC 作为在线算法。
- 使用 10 个 Q 函数 ensemble。
- 每次随机取 2 个 Q 函数并取 min 来估计 Q。
- 使用 LayerNorm 稳定高 UTD 训练。
- 默认 UTD=4，batch size=256。
- 默认 warmup steps $K=5000$。
- actor learning rate $10^{-4}$，critic learning rate $3\times 10^{-4}$，temperature learning rate $10^{-4}$。
- 保留预训练网络的 optimizer state。

**训练/推理区分**：训练阶段需要在线交互和 replay buffer 更新；推理阶段只执行最终策略。WSRL 的关键不是部署时的动作选择，而是 fine-tuning 初期 replay buffer 的构成方式。

## 7. 实验与消融

**实验任务**：

- D4RL AntMaze、Kitchen、Adroit。
- Gym MuJoCo locomotion，包括 HalfCheetah、Hopper、Walker2d 等不同数据质量。
- 预训练通常使用 CalQL，但 Appendix H 显示 IQL/CQL/CalQL 的 Q 初始化也可工作。

**Baselines**：

- CQL、IQL、CalQL：代表可 fine-tune 的离线 RL 方法。
- JSRL：使用预训练策略 roll-in 的近邻方法。
- SO2：尝试平衡 reward maximization 和 distribution matching。
- RLPD / SAC(fast)：高 UTD 在线 RL，从头或混合数据学习。

**主要结果**：

- Figure 7：在 no-retention 设置下，WSRL 在七个任务上显著优于 CQL/IQL/CalQL 等；后者常常无法从初始性能下降中恢复。
- Figure 8：WSRL 的策略和 Q 函数相对预训练初始化保持稳定，说明它可以 unlearn 一部分不相关内容，但没有 catastrophic forgetting。
- Figure 9：即使与保留全部离线数据的 prior methods 比较，WSRL 也通常更快或更高，说明保留离线数据并非必要。
- Appendix Figure 14：在所有八个 D4RL AntMaze 环境上，WSRL 显著优于 baselines。
- Appendix Figure 15：在九个 MuJoCo locomotion 环境上，WSRL 也有稳定增益。

**关键消融**：

- Figure 10：去掉 5000 步 warmup 后，WSRL 显著变差或方差升高。
- Figure 11：warmup 能防止 Q 值过度悲观和 TD error 在离线分布上发散。
- Figure 12：Q 函数初始化在高覆盖任务如 AntMaze 中尤其有用；fine-tuning 用标准 SAC 比继续用 CalQL 更好。
- Figure 13：策略初始化对 Kitchen 等任务很重要。
- Figure 16：随机动作 warmup 不如预训练策略 warmup，说明 warmup 的价值不是简单增加 buffer 大小。
- Figure 17：1k warmup 有时不足，20k warmup 有时过长；默认 5k 是经验折中。
- Figure 23：用离线数据替代 warmup 不如真实在线 warmup，尤其在 AntMaze 上明显。
- Figure 24：让 WSRL 也保留离线数据并无明显优势，甚至后期可能更慢。

**实验解释**：WSRL 证明的是“no-retention fine-tuning 可行”，而不是所有任务都不需要离线数据。如果离线预训练本身失败，Appendix Figure 28 显示 WSRL 也会表现不好。它依赖一个至少能产生有意义 warmup rollout 的 $\pi^{pre}$。

## 8. 展望

**对 ORL 研究者的启发**：

1. Offline-to-online RL 的关键不只是离线算法强弱，而是在线初期如何接住预训练初始化。
2. 离线数据保留可以被看作一种 forgetting mitigation，而不是必需的长期训练资源。
3. 高 UTD 在线 RL 与离线预训练结合时，需要重新考虑 replay buffer 初始化。
4. 分布偏移诊断应同时看在线分布和离线分布上的 Q/TD error。
5. “简单 warmup”可能是复杂 fine-tuning 正则的有力替代。

**局限与开放问题**：

1. 需要环境允许在线 warmup，纯离线场景不适用。
2. 如果预训练策略很差，warmup 数据也会差，WSRL 难以弥补。
3. 默认 5000 步 warmup 是经验设置，不同任务可能需要自适应。
4. 理论分析较少，主要是经验诊断和机制解释。
5. 在真实机器人或高风险系统中，warmup 本身的安全性仍需约束。

**后续研究想法**：

1. **自适应 warmup 终止准则**：用 Q/TD error 稳定性判断何时结束 warmup，减少人工选择 $K$。
2. **安全约束 warmup**：结合 safe RL 或 shield，让预训练策略 warmup 在真实系统中更可靠。
3. **多任务预训练后的 no-retention 微调**：研究大规模混合离线数据预训练后，WSRL 是否能在新任务中快速过滤无关经验。

## Links

- Paper page: https://openreview.net/forum?id=HN0CYZbAPw
- Code: https://github.com/zhouzypaul/wsrl
- arXiv: 未能从论文 PDF 正文确认 arXiv 链接
- Related project/code note: PDF Appendix I 明确写明 “Code for WSRL is released at https://github.com/zhouzypaul/wsrl.”

</article>
