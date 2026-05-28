---
title: "What Makes a Good Diffusion Planner for Decision Making?"
description: "一项系统拆解扩散规划器关键设计的 ICLR 2025 Spotlight 经验研究，并给出强基线 Diffusion Veteran。"
tags:
  - Offline RL
  - Diffusion Planning
  - Decision Making
  - Empirical Study
  - Diffusion Veteran
---

# What Makes a Good Diffusion Planner for Decision Making?

| 字段 | 内容 |
|---|---|
| Title | What Makes a Good Diffusion Planner for Decision Making? |
| Year | 2025 |
| Source | ICLR 2025 Spotlight；OpenReview 显示 Published 22 Jan 2025，Last Modified 06 Mar 2025 |
| Authors | Haofei Lu, Dongqi Han, Yifei Shen, Dongsheng Li |
| Affiliations | Tsinghua University；Microsoft Research Asia。论文脚注说明 Haofei Lu 在 Microsoft Research Asia 实习期间完成本工作，Dongqi Han 为通讯作者。 |
| Tags | Offline RL；Diffusion Planning；Guided Sampling；Transformer Planner；D4RL |

**一句话概括**：本文训练和评估 6000 多个扩散模型，系统回答“什么样的 diffusion planner 才好”，并把结论凝练成强基线 Diffusion Veteran (DV)。

**我对这篇论文的定位**：这是一篇扩散规划领域的“设计空间地图”论文。它把 Diffuser、Decision Diffuser、AdaptDiffuser、HD 等方法中纠缠在一起的选择拆开，分别研究 guided sampling、denoising backbone、action generation 与 planning stride 的影响。对 ORL 研究者来说，它提醒我们：扩散模型的生成能力本身不等于决策能力，规划器的采样、筛选和执行方式才是性能差异的核心来源。

## 1. 第一作者相关信息

Haofei Lu 是清华大学学生，本工作在 Microsoft Research Asia 实习期间完成；Dongqi Han、Yifei Shen、Dongsheng Li 来自 Microsoft Research Asia，其中 Dongqi Han 为通讯作者。OpenReview 页面列出该文为 ICLR 2025 Spotlight，并给出官方代码 `https://github.com/Josh00-Lu/DiffusionVeteran`。

| 年份 | 论文 | Venue/source | 主题标签 | 与本文关系 |
|---|---|---|---|---|
| 2025 | What Makes a Good Diffusion Planner for Decision Making? | ICLR 2025 Spotlight | Diffusion planning, offline RL | 本文主体，系统经验研究与 DV 基线 |
| 2025 | Habitizing Diffusion Planning for Efficient and Effective Decision Making | arXiv/后续相关工作 | Diffusion planning efficiency | 与本文同一研究线，关注规划效率与习惯化 |
| 2025 | Diffusion Veteran code/release | GitHub | Reproducibility | 本文方法实现与复现实验入口 |

这条研究线集中在“生成模型怎样真正服务决策”：不是只把轨迹当图像一样生成，而是追问生成轨迹如何被评估、如何执行、如何避免高回报但不可行的幻觉。作者团队的贡献偏工程经验和系统归因，很适合做后续扩散 ORL 方法的参考基线。

## 2. 研究问题

论文要解决的具体问题是：在离线强化学习中，已有 diffusion planning 方法的设计选择高度不一致，导致我们很难判断性能来自扩散模型本身、guidance 技巧、网络结构，还是 action/trajectory 生成方式。它关心的不是“扩散规划能不能做决策”，而是“哪些组件真正让扩散规划器好用”。

设离线数据集为 $D=\{\tau_i\}$，每条轨迹包含状态、动作和奖励。扩散规划器学习轨迹或状态序列分布 $p_\theta(\tau)$，在测试时从当前状态出发生成未来候选计划，再执行其中的一步或几步。形式上，规划器希望在数据支持范围内生成高回报轨迹：

$$
\tau^\star = \arg\max_\tau \hat{Q}(\tau) \quad \text{s.t. } \tau \sim p_\theta(\tau \mid s_t).
$$

含义：扩散模型负责提出候选轨迹，critic 或 reward estimator 负责给候选打分；真正的决策质量取决于“提出候选”和“选择候选”的组合，而不是单独某个模块。

论文假设的是标准 offline RL 设置：训练阶段只能访问静态数据集，测试阶段在环境中执行策略；评估覆盖 D4RL MuJoCo locomotion、Kitchen、Maze2D 等连续控制与长期规划任务。

## 3. 背景知识

扩散模型通过前向加噪和反向去噪学习数据分布。在轨迹生成中，数据点不再是图片，而是长度为 $H$ 的状态或状态-动作序列：

$$
\tau_t = [(s_t,a_t),(s_{t+1},a_{t+1}),\ldots,(s_{t+H-1},a_{t+H-1})].
$$

训练时模型学习从噪声中恢复轨迹；推理时从高斯噪声出发迭代去噪，得到候选未来。

离线 RL 的核心困难是分布偏移：测试策略可能选择数据集中很少出现的动作或轨迹，critic 对这些 OOD 区域容易高估。扩散规划表面上缓解了这个问题，因为生成模型学习的是数据分布；但如果 guidance 过强、critic 打分错误，模型仍可能被推向不可行或虚高回报轨迹。

扩散 guidance 有三类常见做法。Classifier Guidance 训练一个外部分数模型，用梯度把样本推向高回报；Classifier-Free Guidance 同时训练条件与无条件分支，用二者差值控制生成方向；MCSS 则先无条件生成多个候选，再用 critic 选择最优。本文的重要发现是：在决策中，“先生成再筛选”的 MCSS 往往比梯度 guidance 更稳。

| 术语 | 直观含义 | 在本文中的作用 |
|---|---|---|
| Diffusion planner | 用扩散模型生成未来计划的决策器 | 研究对象 |
| Guided sampling | 采样时用价值/奖励信号引导生成 | 被系统比较，发现未必最优 |
| MCSS | 多次无条件采样后用 critic 选择 | DV 的关键选择之一 |
| Transformer denoiser | 用自注意力做去噪网络 | 实验中优于 U-Net |
| Joint action generation | 直接生成状态-动作序列 | 与逆动力学分离式动作生成比较 |
| Planning stride | 每次执行计划中的几步 | 控制 replanning 频率和效率 |

为什么这篇文章的动机成立：扩散规划领域常把来自图像生成的直觉直接搬到 RL，例如“guidance 应该越强越能产生高回报样本”“U-Net 是扩散模型默认骨干”。但 RL 里的轨迹既有动力学约束又有数据支持约束，过强 guidance 可能破坏可行性，U-Net 的局部卷积归纳偏置也未必适合长程依赖。因此需要大规模控制变量实验。

## 4. 问题分析

论文把 diffusion planner 拆成四个设计维度。

第一是 action generation。论文比较“联合生成状态和动作”与“只生成状态，再用 inverse dynamics 产生动作”。Figure 3 显示不同任务中二者各有优势：在高维动作或逆动力学更可靠的场景，分离式动作生成更稳；但在需要动作-状态联合一致性的任务中，联合生成也有价值。

第二是 planning stride。Dense-step planning 每一步都重规划，计算开销大但反馈频繁；jump-step planning 一次执行多个计划步，效率高但可能累积误差。Figure 4 显示 stride 存在任务相关的甜点区间，过长会让计划脱离实际状态。

第三是 denoising architecture。Figure 5 和 Figure 6 表明 Transformer backbone 在多类任务上优于常见 U-Net，深度增加也并非无限收益。这说明轨迹规划更需要建模全局时间依赖，而不是图像式的局部空间结构。

第四是 guided sampling。Figure 7 对比 CG、CFG、MCSS 等采样方式，结论相当反直觉：无条件生成加选择经常优于 gradient guidance。原因是 critic 梯度在噪声轨迹上不稳定，容易把样本推离数据流形；而 MCSS 保持生成分布不被破坏，只在候选集中做选择。

## 5. 思想与方法

本文的核心思想是：好的扩散规划器不应该过度“改造”生成过程，而应该让生成模型忠实覆盖数据分布，再用外部选择机制挑出高价值计划。这个思想直接导向 Diffusion Veteran (DV)。

DV 的主要组件是：Transformer denoising network、无条件 trajectory generation、Monte Carlo Sampling with Selection、任务相关的 action generation 选择，以及合适的 planning stride。它的新意不是引入复杂目标，而是把大规模实验中稳定有效的选择组合成一个简单、可复现的强基线。

因果机制可以这样理解：Transformer 改善长程依赖建模；无条件采样避免 guidance 把样本推离离线数据流形；MCSS 用 critic 做后验筛选，降低过估计样本被直接执行的概率；planning stride 在计算成本和闭环纠偏之间折中。

证据强度主要来自实验和消融，而不是理论证明。论文的结论适合作为经验法则，但不应被理解成所有环境中 MCSS 必然优于 CFG；当 critic 更可靠或任务条件更清晰时，guided sampling 仍可能有优势。

## 6. 算法与伪代码

算法名：Diffusion Veteran (DV)。论文 Algorithm 1 给出简化流程。

1. 从离线数据集 $D$ 构造长度为 $H$ 的轨迹片段。
2. 训练 Transformer 去噪网络 $\epsilon_\theta(x_t,t)$，预测扩散噪声。
3. 训练或复用 critic/reward estimator，用于评价生成轨迹。
4. 测试时给定当前状态 $s_t$，以该状态作为条件，进行 $N$ 次无条件扩散采样，得到候选计划 $\{\tau_i\}_{i=1}^N$。
5. 对每个候选计划计算价值分数，例如累计预测奖励或 critic score。
6. 选择得分最高的计划 $\tau^\star$。
7. 根据 action generation 设定取出动作：若计划含动作则直接执行；若只含状态则用 inverse dynamics 得到动作。
8. 执行前 $k$ 步，之后回到第 4 步重规划。

核心采样-选择目标可写作：

$$
\tau^\star = \arg\max_{\tau_i \sim p_\theta(\tau \mid s_t),\ i=1,\ldots,N} S(\tau_i),
$$

其中 $S(\tau_i)$ 是 critic 或奖励模型给出的计划分数。直观上，扩散模型负责保证候选“像数据”，选择器负责保证候选“有用”。

重要实现点包括：候选数 $N$、规划长度 $H$、stride、Transformer 深度、是否使用 inverse dynamics、critic 的训练质量。论文附录 Table 2 提供配置；Table 1 报告 DV 在标准 benchmark 上的总体表现。

## 7. 实验与消融

数据集和环境包括 D4RL 的 MuJoCo locomotion、Kitchen、Maze2D 等任务，并在附录验证 Adroit Hand。评价指标是 D4RL normalized score，结果以多随机种子平均报告。

主要 baseline 包括 Diffuser、Decision Diffuser、HD、diffusion policy 类方法，以及 CQL/IQL/TD3+BC 等传统 offline RL 方法。Table 1 显示 DV 在总体平均上优于此前 diffusion planning 与 diffusion policy 方法，是本文把消融发现合成强基线的主要证据。

消融结论非常清晰：Figure 3 讨论动作生成；Figure 4 讨论 stride；Figure 5-6 讨论 Transformer vs U-Net 与网络深度；Figure 7 讨论 guided sampling；Figure 8 按任务类型总结 diffusion planner 和 diffusion policy 的适用性。整体上，长期规划与多峰轨迹任务更适合 diffusion planner；单步动作拟合足够的 locomotion 任务中，diffusion policy 也很强。

实验能证明的是：在作者覆盖的 D4RL 设计空间内，若想建立强 diffusion planning baseline，Transformer + MCSS + 合理 stride 是非常稳的组合。实验尚不能证明的是：这些结论在高维视觉机器人、真实世界随机动力学或人类偏好奖励下仍然无条件成立。

## 8. 展望

对 ORL 研究者的启发：第一，扩散规划的核心竞争力来自候选覆盖和选择机制的配合；第二，图像生成中的默认架构不一定适合轨迹；第三，guidance 需要与 critic 可靠性一起评估；第四，规划频率是性能和部署成本之间的关键旋钮。

局限与开放问题：本文偏经验归因，缺少关于 guidance 破坏数据流形的理论刻画；实验主要在 D4RL，真实机器人和视觉输入覆盖有限；MCSS 依赖候选数，计算成本随任务复杂度上升；critic 错误仍可能导致选择坏计划。

可能的后续方向：1. 为 MCSS 建立覆盖-选择误差界，解释候选数和 critic 误差如何影响性能。2. 研究视觉世界模型中的 diffusion planning，把 Transformer planner 与 latent dynamics 结合。3. 设计 uncertainty-aware selector，让高分但不确定的候选自动降权。

## Links

- Paper page: https://openreview.net/forum?id=7BQkXXM8Fy
- OpenReview PDF: https://openreview.net/pdf?id=7BQkXXM8Fy
- Code: https://github.com/Josh00-Lu/DiffusionVeteran
- 本地 PDF: `[ICLR_2025] What Makes a Good Diffusion Planner for Decision Making.pdf`
