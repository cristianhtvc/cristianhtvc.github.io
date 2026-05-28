---
title: "Scalable Decision-Making in Stochastic Environments through Learned Temporal Abstraction"
description: "L-MAP 通过 VQ-VAE 学习宏动作 latent，再用 MCTS 在离散隐空间中高效处理随机连续控制规划。"
tags:
  - Offline RL
  - Model-Based Planning
  - Temporal Abstraction
  - MCTS
  - VQ-VAE
---

# Scalable Decision-Making in Stochastic Environments through Learned Temporal Abstraction

| 字段 | 内容 |
|---|---|
| Title | Scalable Decision-Making in Stochastic Environments through Learned Temporal Abstraction |
| Year | 2025 |
| Source | ICLR 2025 Spotlight；OpenReview 显示 Published 22 Jan 2025，Last Modified 28 Feb 2025 |
| Authors | Baiting Luo, Ava Pettet, Aron Laszka, Abhishek Dubey, Ayan Mukhopadhyay |
| Affiliations | Vanderbilt University；Pennsylvania State University；Nissan Advanced Technology Center |
| Tags | Offline RL；Latent Macro Action Planner；Temporal Abstraction；MCTS；Stochastic Environments |

**一句话概括**：L-MAP 学习离散 latent macro-action，把高维连续动作和长 horizon 同时压缩，再用 MCTS 在隐空间中规划，从而在随机环境中保持低延迟和高回报。

**我对这篇论文的定位**：这是 model-based offline RL 中“可扩展规划”的论文。它关注的不是如何学一个更保守的 Q 函数，而是如何让规划在高维连续动作、长时间跨度和随机动态下仍然可计算。对 ORL 来说，它把 temporal abstraction、latent action model 和 MCTS 组合成一个较完整的 planning agent。

## 1. 第一作者相关信息

Baiting Luo 是 Vanderbilt University 博士生，OpenReview profile 显示研究方向包括 Reinforcement Learning、Planning、Embodied AI Systems 和 Cyber Physical Systems，导师关系包括 Abhishek Dubey 与 Ayan Mukhopadhyay。本文为 ICLR 2025 Spotlight，另有 FMEA @ CVPR 2025 版本。

| 年份 | 论文 | Venue/source | 主题标签 | 与本文关系 |
|---|---|---|---|---|
| 2025 | Scalable Decision-Making in Stochastic Environments through Learned Temporal Abstraction | ICLR 2025 Spotlight | Offline RL, MCTS, temporal abstraction | 本文主体 |
| 2025 | NS-Gym: A Comprehensive and Open-Source Simulation Framework for Non-Stationary MDPs | NeurIPS 2025 Datasets/Benchmarks | Non-stationary MDP | 同一作者的环境/决策方向 |
| 2024 | Act as You Learn: Adaptive Decision-Making in Non-Stationary MDPs | AAMAS 2024 | Adaptive decision-making | 与随机/非平稳决策相关 |
| 2024 | Decision Making in Non-Stationary Environments with Policy-Augmented Search | AAMAS 2024 | Search, planning | 与本文搜索规划思想相邻 |

作者研究轨迹从非平稳 MDP、自动驾驶/智能体安全逐步延伸到高维随机环境中的实时规划，这也解释了本文对 latency 和 stochasticity 的重视。

## 2. 研究问题

高维连续动作空间中的规划会遇到两个指数级困难：动作维度高导致可选动作巨大，规划 horizon 长导致搜索树爆炸。随机环境又让同一个状态-动作可能产生多个下一状态，确定性模型很难稳定评估。

论文研究传统 offline RL 设置：智能体只有由随机行为策略收集的数据，需要在不在线探索的情况下学习可规划模型。目标是在连续控制、机器人手和 AntMaze 等任务中，用低延迟搜索得到高期望回报。

L-MAP 的关键设定是固定长度 macro-action：

$$
m_t = (a_t, a_{t+1}, \ldots, a_{t+L-1}).
$$

一个宏动作覆盖多个原始动作步，因此同时缩短规划深度，并通过 VQ-VAE 把宏动作离散化，降低动作分支数。

## 3. 背景知识

Macro-action 是时序抽象的一种：高层只选择一个技能或动作片段，低层连续执行。这样原本 $H$ 步规划变为约 $H/L$ 步规划，缓解 curse of history。

VQ-VAE 可把连续输入压到有限 codebook。L-MAP 输入包含状态和宏动作，学习离散 latent code；decoder 可以从 code 还原宏动作。这样 MCTS 不再在原始连续动作上搜索，而是在离散 latent action 上搜索。

MCTS 用模拟展开估计每个动作的价值，适合随机环境，因为它可以通过多次 rollout 估计期望。Progressive widening 用于连续或大动作空间：随着访问次数增加逐步扩展新动作，避免一开始展开过多分支。

| 术语 | 直观含义 | 在本文中的作用 |
|---|---|---|
| Macro-action | 多个低层动作组成的片段 | 时间压缩 |
| Latent action | VQ-VAE code 表示的动作片段 | 动作空间压缩 |
| Prior model | 隐空间中的转移/候选生成模型 | 提供可行 latent action |
| MCTS | 蒙特卡洛树搜索 | 处理随机规划 |
| Progressive widening | 逐步增加分支 | 控制大动作空间搜索复杂度 |
| Behavior policy stochasticity | 离线数据来自随机策略 | 需要模型能覆盖多样动作 |

为什么动机成立：Trajectory Transformer 类方法在原始 token 空间规划，动作维和 horizon 一大就慢；TAP 只做单步 latent action，没压缩时间；model-free baseline 虽强，但缺乏显式搜索的实时适应性。L-MAP 正是把空间和时间两个维度同时压缩。

## 4. 问题分析

Figure 1 展示预构造 latent 搜索空间后，L-MAP 相比 vanilla MCTS 在相同延迟下能取得更高性能。论文的诊断是：瓶颈不是 MCTS 本身，而是直接在连续原始动作上扩展搜索树。

Figure 2 描述 VQ-VAE 如何离散化 state-macro-action 序列。随机环境中直接把回报和状态都编码进同一个 code 可能导致碎片化，因此 L-MAP 使用 masked/full 双通道思想，让 latent 更关注可执行动作结构，同时保留随机结果建模。

Figure 3-4 展示预构造 latent search space 与 MCTS 过程：先学习可行宏动作集合和 prior，再在规划时根据当前状态扩展、模拟、回传价值。

## 5. 思想与方法

L-MAP 的指导原则是：不要让搜索直接面对原始连续动作，而是先从离线数据中学一个“可行且短时稳定”的宏动作字典。规划时搜索这个字典中的 latent code，再解码为连续动作序列。

方法由三块组成：state-conditional VQ-VAE 学宏动作；prior/transition model 学 latent 空间动态和可行动作分布；MCTS 在 latent macro-action 上做规划。随机性由 MCTS 的多次采样和 value backup 处理。

它的新意是三者组合。单独用 VQ-VAE 只能得到离散动作；单独用 temporal abstraction 仍可能搜索困难；单独 MCTS 在连续空间太慢。L-MAP 把它们拼在一起，使 search branching 和 horizon 都下降。

## 6. 算法与伪代码

算法名：Latent Macro Action Planner (L-MAP)。

1. 从离线数据中切出长度为 $L$ 的动作片段 $m_t$。
2. 训练 state-conditional VQ-VAE，将 $(s_t,m_t)$ 编码为离散 latent code $z_t$，并解码重建宏动作。
3. 训练 prior/latent transition model，给定当前状态或历史，预测可行 latent code 与下一状态分布。
4. 测试时以当前状态作为 MCTS 根节点。
5. 通过 prior model 采样或扩展候选 latent macro-actions。
6. 用 decoder 将 latent code 还原为连续宏动作，并用 learned dynamics 模拟结果。
7. MCTS 根据 rollout 回报回传更新节点价值。
8. 使用 progressive widening 控制每个节点扩展的候选数。
9. 选择根节点下价值最高的 latent macro-action。
10. 执行对应宏动作的第一步或整个片段，之后重新规划。

VQ-VAE 目标可概括为：

$$
\mathcal{L}_{VQ} = \|m-\hat m\|^2 + \|\mathrm{sg}[z_e]-e\|^2 + \beta\|z_e-\mathrm{sg}[e]\|^2.
$$

含义：既要能重建宏动作，又要让 encoder 输出稳定贴近 codebook。

## 7. 实验与消融

实验覆盖 Stochastic MuJoCo、D4RL MuJoCo、Adroit robotic hand、AntMaze。Table 1 显示在 Stochastic MuJoCo 中 L-MAP 显著优于 model-based baselines；Table 2 报告 D4RL MuJoCo-v2，L-MAP 与强 model-free actor-critic baseline 接近或竞争；Table 3 展示 Adroit 高维机器人手任务，Table 4 展示 AntMaze 稀疏奖励任务。

baseline 包括 Trajectory Transformer、TAP、1R2R、CQL/IQL/TD3+BC 等。论文强调 L-MAP 在动作维增加时仍保持较低 decision latency，这是 model-based planning 能否部署的关键。

实验支持 L-MAP 对随机性和高维动作的处理有效，但也有边界：learned dynamics 和 latent prior 的误差会影响 MCTS；macro length 固定可能不适合所有任务；在极长 horizon 或强视觉部分可观测环境中还需要扩展。

## 8. 展望

启发：第一，离线模型规划可以通过“学动作抽象”而不是只学更准 dynamics 来扩展；第二，随机环境下 MCTS 仍有价值，但必须先压缩动作空间；第三，latent macro-action 是连接 model-based planning 和 offline dataset support 的好接口。

局限：VQ codebook 容量和宏动作长度敏感；learned model 的误差会在 rollout 中累积；MCTS 仍有计算成本；模型主要在仿真 benchmark 上验证。

后续方向：1. 学习可变长度 macro-action，而不是固定 $L$。2. 将 uncertainty penalty 加入 MCTS backup，避免搜索偏向模型不确定区域。3. 把 L-MAP 与语言/任务条件结合，形成可解释技能规划器。

## Links

- Paper page: https://openreview.net/forum?id=pQsllTesiE
- OpenReview PDF: https://openreview.net/pdf?id=pQsllTesiE
- arXiv: https://arxiv.org/abs/2502.21186
- DBLP: https://dblp.org/rec/conf/iclr/LuoPLDM25
- Code: 未能从论文与 OpenReview 页面确认官方公开代码链接。
- 本地 PDF: `[ICLR_2025] Scalable Decision-Making in Stochastic Environments through Learned Temporal Abstraction.pdf`
