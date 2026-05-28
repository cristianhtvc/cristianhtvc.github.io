---
title: "Fewer May Be Better: Enhancing Offline Reinforcement Learning with Reduced Dataset"
description: "ReDOR 将离线 RL 数据子集选择建模为梯度近似与弱次模优化，用更少数据提升训练效率和策略表现。"
tags:
  - Offline RL
  - Dataset Selection
  - Coreset
  - Submodular Optimization
  - D4RL
---

# Fewer May Be Better: Enhancing Offline Reinforcement Learning with Reduced Dataset

<div class="paper-hero">
<p class="paper-meta">Yiqin Yang, Quanwei Wang, Chenghao Li, Hao Hu, Chengjie Wu, Yuhua Jiang, Dianyu Zhong, Ziyou Zhang, Qianchuan Zhao, Chongjie Zhang, Bo Xu · ICLR 2025 · ICLR / 2025</p>
<div class="tag-row"><span>Offline RL</span><span>Dataset Selection</span><span>Coreset</span><span>Submodular Optimization</span><span>D4RL</span></div>
</div>

<article class="note-body" markdown>

| 字段 | 内容 |
|---|---|
| Title | Fewer May Be Better: Enhancing Offline Reinforcement Learning with Reduced Dataset |
| Year | 2025 |
| Source | ICLR 2025 Conference Paper |
| Authors | Yiqin Yang, Quanwei Wang, Chenghao Li, Hao Hu, Chengjie Wu, Yuhua Jiang, Dianyu Zhong, Ziyou Zhang, Qianchuan Zhao, Chongjie Zhang, Bo Xu |
| Affiliations | Institute of Automation, Chinese Academy of Sciences; Tsinghua University; Washington University in St. Louis |
| Tags | Offline RL; data reduction; gradient approximation; OMP; submodularity |

**一句话概括：** 论文提出 ReDOR，把 offline RL 的数据集缩减问题转化为近似完整数据梯度的 coreset 选择问题，并用改造后的 OMP 选出更小但更有用的数据子集。

**我对这篇论文的定位：** 这篇论文属于 offline RL 的数据中心视角：不是改算法本身，而是问“哪些离线数据真正有助于训练”。它和 dataset pruning、coreset selection、D4RL data quality、trajectory stitching 等问题相关。

## 1. 第一作者相关信息

第一作者 **Yiqin Yang**。论文首页显示其来自 Institute of Automation, Chinese Academy of Sciences。公开检索中可核验到本文 ICLR 2025 信息；本次未完成第一作者近五年完整 publication profile。

| year | title | venue/source | topic tag | relation to this paper |
|---|---|---|---|---|
| 2025 | Fewer May Be Better: Enhancing Offline Reinforcement Learning with Reduced Dataset | ICLR 2025 | Offline RL, data selection | 本文 |

从本文看，作者关注 offline RL 中“数据质量和数据规模”的关系：更多数据不一定更好，尤其当额外数据是低质量、次优或会加剧 distribution shift 的样本时。

## 2. 研究问题

offline RL 常默认“数据越多越好”，但现实中大数据集会带来计算成本，且混入低质量数据会恶化 distribution shift。论文提出的问题是：

> 如何选择离线数据集的一个子集，使训练更快、性能不降甚至提升？

形式上，给定完整数据集 $D$，希望选出 $S\subset D$ 和权重 $w$，让在 $S$ 上训练得到的 Q 函数近似在 $D$ 上训练的 Q 函数。论文把目标写成梯度近似：

$$
\mathrm{Err}(w,S,L,\theta)
=\left\|\sum_{i\in S}w_i\nabla_\theta L_i(\theta)-\nabla_\theta L(\theta)\right\|.
$$

含义：如果加权子集的梯度方向能逼近完整数据集梯度，那么在子集上训练应能模拟完整训练过程。

## 3. 背景知识

| term | plain Chinese meaning | role in this paper |
|---|---|---|
| Dataset reduction | 从大数据集中选较小子集 | 论文主任务 |
| Coreset | 能代表完整数据集的小集合 | ReDOR 的目标产物 |
| TD loss | Q-learning 的 Bellman 误差损失 | 用于构造梯度 |
| Gradient matching | 子集梯度近似全数据梯度 | 数据选择准则 |
| OMP | 逐步选基向量降低残差的方法 | ReDOR 的求解器 |
| Weak submodularity | 贪心选择有近似保证的性质 | 理论支撑 |

监督学习中的 coreset 通常看样本 loss 或分类边界；offline RL 不同，因为学习目标会随 policy 和 target Q 更新而变化。一个 transition 当前 TD loss 高，不代表它一定能提升最终策略；反而可能是噪声或低质量行为带来的误导。ReDOR 因此选择看“梯度能否代表完整训练信号”。

## 4. 问题分析

论文指出 RL 数据选择和监督学习有两个核心差异：

1. 监督学习 loss 可直接反映样本难度，但 RL 中 TD loss 与最终 policy performance 不直接对应。
2. offline RL 的目标值随训练动态变化，导致梯度不稳定。

为解决第 2 点，ReDOR 不直接用瞬时 TD target，而用 trajectory empirical return 平滑目标：

$$
y=\sum_{k=0}^{H-t}\gamma^k r(s_{t+k},a_{t+k}).
$$

这样选择数据时看到的是更稳定的回报信号，而不是不断漂移的 bootstrapped target。理论上，论文证明 actor-critic 框架下 critic 和 actor 对应的选择目标都是 weakly submodular，可用 OMP 贪心近似求解。

## 5. 思想与方法

ReDOR 的核心思想是：**不要按“样本看起来难不难”选数据，而按“它的梯度是否能代表完整数据集训练方向”选数据。**

方法有三层：

- **梯度近似目标**：让子集加权梯度逼近完整数据梯度。
- **OMP 逐步选择**：每轮选择与当前 residual 最相关的 trajectory，把它加入子集。
- **针对 offline RL 的稳定化修改**：用 empirical return 代替标准 TD target，并做多轮选择，因为训练目标随 policy 变化。

这与 Prioritized Experience Replay 的差别很大：PER 偏向高 TD error，ReDOR 偏向能覆盖完整梯度结构、对 actor-critic 更新有代表性的样本。

## 6. 算法与伪代码

算法名：**ReDOR (Reduced Datasets for Offline RL)**。

1. 输入完整离线数据集 $D$。
2. 初始化用于数据选择的 offline agent，包括 critic $Q_\theta$ 和 actor $\pi_\phi$。
3. 对多个选择轮次 $t=1,\dots,T$：
4. 读取当前 critic 参数 $\theta_t$。
5. 基于 empirical return 计算完整数据梯度和 trajectory-level 梯度。
6. 调用 OMP：选出与 residual 最相关的轨迹，并更新样本权重 $w$。
7. 合并各轮选出的轨迹形成 reduced dataset $S$。
8. 重新初始化 offline agent。
9. 在 $S$ 和权重 $w$ 上训练下游 offline RL 算法。

论文的 OMP 子过程是 residual-driven：每次找与当前误差方向内积最大的 trajectory，然后重新求解权重以最小化正则化 residual。

## 7. 实验与消融

实验使用 D4RL，并额外构造 D4RL (Hard)：向数据中加入低质量数据模拟真实任务噪声。下游算法主要用 TD3+BC；AntMaze 中用 IQL 作为 backbone。对比包括 Complete Dataset、Random、Prioritized，以及 KRLS、LogDet、BlockGreedy 等监督学习数据缩减方法。

关键结果：

- Figure 1/2：ReDOR 选择的子集在 MuJoCo 上通常比完整数据训练更快、更好。
- Table 1：D4RL (Hard) 上，ReDOR 在 Hopper/Walker2d/HalfCheetah 多个任务上优于 KRLS、LogDet、BlockGreedy。
- Table 2：AntMaze 上，ReDOR 优于 Random、Prioritized 和 Complete Dataset；这说明它没有简单删掉 trajectory stitching 所需的关键片段。
- Appendix Figure 4：去掉多轮选择或用标准 Q target 替代 empirical return 都会显著变差，walker2d-medium 和 walker2d-expert 上退化尤其明显。
- 计算开销：论文称即使百万级数据，开销也在数分钟级，主要得益于 trajectory-level selection 和正则化约束。

实验解释上要注意：ReDOR 的收益部分来自“去除坏数据”，在数据已经非常干净且覆盖充分时，子集选择未必总能超越完整数据。

## 8. 展望

启发：

- offline RL benchmark 不应只报告算法，还应报告数据选择和数据质量机制。
- “更少数据更好”在 offline RL 中合理，因为次优数据会改变行为分布和 Bellman target。
- trajectory-level coreset 比 transition-level random pruning 更适合 AntMaze 等 stitching 任务。

局限：

- 需要额外的数据选择阶段。
- 梯度近似依赖 backbone 选择，换算法后最优子集可能变化。
- 真实机器人/多模态数据上的有效性仍需验证。

后续方向：

- 将 ReDOR 用于 offline-to-online fine-tuning 的 replay buffer pruning。
- 研究与数据估值、influence function 的结合。
- 在视觉机器人数据中用 representation-level gradient matching 做大规模筛选。

## Links

- Paper page: https://proceedings.iclr.cc/paper_files/paper/2025/hash/07bc83d986470c6c4d83179958a54e6d-Abstract-Conference.html
- DBLP: https://dblp.org/rec/conf/iclr/YangWLHWJZZZX25
- Code: 未能从论文和公开检索中确认官方代码链接

</article>
