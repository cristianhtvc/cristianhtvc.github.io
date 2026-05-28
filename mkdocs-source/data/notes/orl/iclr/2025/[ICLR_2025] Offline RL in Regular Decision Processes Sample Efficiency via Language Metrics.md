---
title: "Offline RL in Regular Decision Processes: Sample Efficiency via Language Metrics"
description: "用形式语言度量和 Count-Min-Sketch 改进 RDP 离线强化学习的样本效率与空间复杂度。"
tags:
  - Offline RL
  - Regular Decision Processes
  - Non-Markovian RL
  - Automata Learning
  - Sample Complexity
---

# Offline RL in Regular Decision Processes: Sample Efficiency via Language Metrics

| 字段 | 内容 |
|---|---|
| Title | Offline RL in Regular Decision Processes: Sample Efficiency via Language Metrics |
| Year | ICLR 2025；OpenReview 发布于 2025-01-22，最终修改于 2025-05-17 |
| Source | ICLR 2025 Poster；PDF 标注为 published as a conference paper at ICLR 2025 |
| Authors | Ahana Deb, Roberto Cipollone, Anders Jonsson, Alessandro Ronca, Mohammad Sadegh Talebi |
| Affiliations | Universitat Pompeu Fabra; Leonardo S.p.A.; University of Oxford; University of Copenhagen |
| Tags | Offline RL; RDP; Non-Markovian RL; Formal Languages; PAC; Count-Min-Sketch |

**一句话概括**：本文把离线 Regular Decision Process 学习中的状态可区分性重写为形式语言上的测试问题，并引入 Count-Min-Sketch 压缩历史分布计数，从而在低语言复杂度环境中避免 RegORL 的指数级样本/空间瓶颈。

**我对这篇论文的定位**：这是一篇偏理论的 ORL 论文，重点不在 D4RL 式连续控制，而在非马尔可夫环境中如何从固定离线轨迹恢复隐藏自动机并规划。它延续 Cipollone et al. 2023 的 RegORL 路线，但指出原有 $L_\infty^p$-distinguishability 在 T-maze 等直观简单任务上会随 horizon 指数衰减。论文的新意在于：用可选择的语言族 $\mathcal{X}$ 替代“所有后缀序列”的最坏情况比较，并用 CMS 把长后缀计数从显式表格变成近似数据结构。

## 1. 第一作者相关信息

Ahana Deb 是 Universitat Pompeu Fabra AI-ML group 的博士生，导师为 Anders Jonsson。其个人主页介绍其研究兴趣包括 reinforcement learning、learning theory、representation learning，并列出本篇 ICLR 2025 论文以及后续 EWRL 2025 工作。

| Year | Title | Venue/source | Topic tag | Relation to this paper |
|---|---|---|---|---|
| 2025 | Offline RL in Regular Decision Processes: Sample Efficiency via Language Metrics | ICLR 2025 Poster | Offline RDP, language metric | 本文主体工作 |
| 2025 | Learning Compact Regular Decision Processes using Priors and Cascades | EWRL 2025 / OpenReview | Compact RDP, priors, cascades | 继续利用本文语言度量，处理带循环/更紧凑自动机 |
| 2024 | Tractable Offline Learning of Regular Decision Processes | EWRL17 / OpenReview | Offline RDP, automata learning | 本文的前身版本，提出语言伪度量和 CMS 思路 |

综合看，第一作者的研究轨迹集中在“可证明的非马尔可夫 RL + 自动机结构学习”，尤其是如何把历史依赖压缩成可规划的有限状态表示。作者信息主要来自论文首页、OpenReview、Ahana Deb 个人主页和后续 OpenReview 记录。

## 2. 研究问题

本文研究 episodic RDP 的 offline RL：给定由行为策略预先收集的 $K$ 条长度 $H$ 轨迹，既不能继续与环境交互，也不知道隐藏的自动机状态；目标是在有限样本内恢复足以规划的 RDP 表示，并输出近最优 regular policy。

RDP 是一类非马尔可夫决策过程：未来观测和奖励依赖整段历史，但这种依赖可由一个有限状态自动机捕捉。形式上，历史 $h$ 被自动机转移函数映射到隐藏状态 $u=\bar{\tau}(h)$，在 $u$ 上再满足类似 MDP 的转移/奖励结构。因此 RDP 比普通 MDP 表达力更强，又比一般 POMDP 更可学习，因为隐藏状态由历史唯一确定。

现有 RegORL 的瓶颈有两个。第一，状态区分依赖 $L_\infty^p$-distinguishability，即不同隐藏状态在后续轨迹分布上的最小可检测差异；Theorem 1 展示存在 RDP 族使该量随 $H$ 指数衰减，样本复杂度随之爆炸。第二，显式统计所有后缀轨迹的概率分布需要随 horizon 指数增长的存储。

## 3. 背景知识

**从标准 offline RL 到 RDP**：标准 offline RL 通常假设状态 $s_t$ 已经满足马尔可夫性，问题主要是行为策略和目标策略的分布偏移。RDP 中观测 $o_t$ 不够，agent 需要记住过去事件，例如“是否曾获得许可”。这里的难点不是连续动作 OOD，而是“哪两段历史应该被合并成同一个状态”。

**Regular Decision Process**：RDP 的“regular”来自形式语言理论。每个隐藏状态可看成一个等价类：两个历史若对所有未来动作-观测-奖励序列诱导相同分布，就属于同一自动机状态。学习 RDP 等价于从离线样本中发现这些等价类。

**可区分性**：若两个隐藏状态 $u,v$ 对某些未来后缀 $x$ 的概率差足够大，就能用有限样本区分。RegORL 使用非常强的全后缀比较，本文则只在语言族 $\mathcal{X}$ 上比较：

$$
\mu_{\mathcal{X}}=\min_{u\ne v}\max_{x\in\mathcal{X}}
\left|P(x\mid u)-P(x\mid v)\right|.
$$

直观地说，$\mathcal{X}$ 是一组“测试后缀”。如果这些测试已经足够区分所有自动机状态，就没有必要统计所有可能后缀。

**Count-Min-Sketch**：CMS 是流式计数数据结构。它不保存完整计数字典，而用多行哈希表保存近似计数，点查询满足“不低估、最多多出 $\varepsilon\|v\|_1$”的概率保证。本文用它压缩 RDP 学习中的后缀分布估计。

| 术语 | 直白含义 | 在本文中的作用 |
|---|---|---|
| RDP | 历史依赖可由有限自动机表示的决策过程 | 非马尔可夫 offline RL 的目标模型 |
| Regular policy | 只依赖自动机状态的策略 | 学到 RDP 后可在等价 MDP 上规划 |
| $L_\infty^p$-distinguishability | 全部后缀分布上的最坏可区分度 | RegORL 的样本复杂度瓶颈 |
| Language metric $L_{\mathcal{X}}$ | 只在测试语言族上比较分布 | 本文提升样本效率的核心 |
| CMS | 近似计数结构 | 缓解长 horizon 的内存爆炸 |

## 4. 问题分析

论文的诊断路径先从 T-maze 入手。Figure 1 中，agent 在起点得到目标方向提示，随后长走廊中的观测基本是噪声，必须记住最早的提示才能在岔路口选择正确方向。这个任务直觉上只需要一小段关键记忆，但 RegORL 若用全后缀分布比较，最小差异可能随走廊长度指数变小。

Theorem 1 正式说明：存在 RDP 族和 regular behavior policy，使 $L_\infty^p$-distinguishability 至多为 $2^{-H}$ 量级。这意味着即便真实自动机并不复杂，RegORL 的统计测试也可能被一个过于保守的复杂度参数拖垮。

第二个问题是空间。要比较所有长后缀分布，朴素实现需要维护随 $|\Gamma|^H$ 增长的计数表，其中 $\Gamma$ 是动作、观测、奖励等符号构成的字母表。CMS 不是减少要比较的后缀集合，而是减少每个分布表的存储。

## 5. 思想与方法

核心思想有两条。第一，用形式语言理论给出更细的“状态测试集”：只要某个语言族 $\mathcal{X}$ 能区分真实自动机状态，就让样本复杂度依赖 $\mu_{\mathcal{X}}$ 和 $|\mathcal{X}|$，而不是依赖全后缀空间的 $L_\infty^p$。第二，用 CMS 替换朴素计数，让算法可在更长 horizon 下运行。

方法上，论文保留 ADACT-H/RegORL 的自动机发现框架：从离线轨迹中逐步构造候选自动机状态，执行统计测试判定状态是否可区分，然后在学到的 RDP 上规划。新贡献嵌入在“测试状态是否可区分”这一步。

理论证据分为两部分。Theorem 2 给出 CMS 版本的高概率恢复保证，空间更友好但仍可能遍历大量后缀。Theorem 3 给出语言度量版本的 PAC 样本复杂度界，其关键依赖从 $\mu$ 变成 $\mu_{\mathcal{X}}$。在低语言复杂度任务中，$\mathcal{X}$ 可很小且 $\mu_{\mathcal{X}}$ 不随 $H$ 指数缩小。

## 6. 算法与伪代码

算法可理解为 ADACT-H 的两个变体：`ADACT-H + Language metric` 和 `ADACT-H + CMS`。

1. 输入离线轨迹集 $D$、置信参数 $\delta$、horizon $H$，以及语言族 $\mathcal{X}$ 或 CMS 参数。
2. 初始化一个粗糙自动机，把尚未区分的历史放入同一候选状态。
3. 对每个候选状态 $u$，从数据中估计其后缀分布。
4. 对每对候选状态 $u,v$ 执行统计测试：
   - language 版本只比较 $x\in\mathcal{X}$ 的概率差；
   - CMS 版本用 sketch 查询替代显式后缀计数。
5. 若存在测试后缀使两状态差异显著，则拆分状态；否则保持合并。
6. 重复直到自动机结构稳定。
7. 将学到的 RDP 展开成等价 MDP，执行 planning/value iteration，返回 regular policy。

关键区别在于统计测试的代价。Language metric 版本减少测试后缀数量；CMS 版本减少存储，但若仍需遍历全部后缀，时间复杂度可能没有根本改善。

## 7. 实验与消融

实验使用 5 个 RDP benchmark：Corridor、T-maze(c)、Cookie、Cheese、Mini-hall。Table 1 对比 FlexFringe、CMS 版本和 language metric 版本，报告学习到的自动机状态数、策略得分和运行时间。

主要结论如下。第一，language metric 版本在 T-maze(c)、Cookie 和 Mini-hall 等任务上能学到更有效策略，尤其 T-maze(c) 中 FlexFringe 得分为 0，而 language metric 得分达到最优 4.0。第二，CMS 版本能在部分任务上恢复高回报策略，但在 Mini-hall 上超时，说明空间压缩不等于时间问题被解决。第三，Figure 2 显示 T-maze corridor length 增大时，language metric 的时间和状态数更接近线性增长，而 CMS 版本状态数和时间增长更快。

实验支撑了论文的机制判断：当任务的关键记忆可以被短语言族捕捉时，$\mathcal{X}_{3,1}$ 这类小测试族就足够；但这也暴露局限，即语言族选择依赖先验，论文没有完全解决自动选择 $\mathcal{X}$ 的问题。

## 8. 展望

对 ORL 研究者的启发：第一，非马尔可夫 offline RL 的关键可能不是更强的神经网络记忆，而是找到可验证的历史等价表示。第二，复杂度参数的选择会决定理论界是否有实践意义。第三，形式语言和自动机学习为 reward machine、LTL 任务和长期依赖任务提供了更清晰的桥梁。第四，CMS 提醒我们 PAC 分析之外还要认真考虑空间复杂度。第五，RDP 是连接 POMDP、PSR、reward machine 的一个有用中间层。

局限和开放问题包括：语言族 $\mathcal{X}$ 需要人工设定；当前主要处理 acyclic episodic RDP；CMS 只解决空间而不完全解决时间；实验规模仍偏小；从表格/有限符号 RDP 到高维观测或连续控制仍有距离。

可追问方向：1. 设计自适应扩展语言族的 RDP 学习算法，按需增加后缀长度。2. 将语言度量用于带循环 RDP 或 omega-regular 目标，处理无限 horizon。3. 把自动机状态学习和神经表示学习结合，使 RDP 思路进入视觉/机器人任务。

## Links

- Paper page: https://openreview.net/forum?id=EW6bNEqalF
- ICLR proceedings: https://proceedings.iclr.cc/paper_files/paper/2025/hash/6e3ff406118173f18dc42a4c11c37ac9-Abstract-Conference.html
- First author profile: https://ahanadeb.com/
- Code: Ahana Deb 主页列有 code 链接；论文 PDF/OpenReview 页面未在正文首页直接给出仓库地址。
- arXiv: 未能从 ICLR OpenReview 页面确认独立 arXiv 条目。

