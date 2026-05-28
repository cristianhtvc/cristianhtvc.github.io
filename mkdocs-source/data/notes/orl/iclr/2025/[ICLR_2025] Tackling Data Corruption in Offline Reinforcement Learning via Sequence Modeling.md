---
title: "Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling"
description: "RDT 说明序列建模式 offline RL 在数据损坏下天然更稳，并用三种轻量技术增强 Decision Transformer。"
tags:
  - Offline RL
  - Robust RL
  - Decision Transformer
  - Data Corruption
  - Sequence Modeling
---

# Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling

| 字段 | 内容 |
|---|---|
| Title | Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling |
| Year | 2025 |
| Source | ICLR 2025 Poster；OpenReview 显示 Published 22 Jan 2025，Last Modified 02 Mar 2025 |
| Authors | Jiawei Xu, Rui Yang, Shuang Qiu, Feng Luo, Meng Fang, Baoxiang Wang, Lei Han |
| Affiliations | The Chinese University of Hong Kong, Shenzhen；University of Illinois Urbana-Champaign；City University of Hong Kong；Rice University；University of Liverpool；Tencent Robotics X |
| Tags | Robust Offline RL；Decision Transformer；Data Corruption；Noisy Dataset；Sequence Modeling |

**一句话概括**：本文发现 vanilla Decision Transformer 在有限且损坏的离线数据上比 TD-learning 类 robust offline RL 更稳，并提出 Robust Decision Transformer (RDT)：embedding dropout、Gaussian weighted learning 与 iterative data correction 三个简单增强。

**我对这篇论文的定位**：这是 robust offline RL 与 sequence modeling offline RL 的交叉论文。它的亮点不是把 DT 做得更复杂，而是把“数据损坏”这个此前多由 TD-learning 处理的问题重新放到序列建模视角下分析。对 ORL 来说，它提供了一个很实用的判断：当噪声来自传感器、动作日志或奖励记录时，避免 bootstrapping 的监督式序列模型可能天然更稳。

## 1. 第一作者相关信息

Jiawei Xu 来自香港中文大学（深圳），与 Baoxiang Wang 团队相关；Rui Yang 来自 UIUC，两人为共同一作。OpenReview 与论文均列出代码仓库 `https://github.com/jiawei415/RobustDecisionTransformer`。由于 Jiawei Xu 是常见姓名，作者身份应以论文中的 CUHK-Shenzhen 邮箱、OpenReview 页面和代码仓库为准。

| 年份 | 论文 | Venue/source | 主题标签 | 与本文关系 |
|---|---|---|---|---|
| 2025 | Tackling Data Corruption in Offline RL via Sequence Modeling | ICLR 2025 Poster | Robust offline RL | 本文主体 |
| 2025/2026 | ADG: Ambient Diffusion-Guided Dataset Recovery for Corruption-Robust Offline RL | OpenReview/后续相关 | Dataset recovery, robust ORL | 延续数据损坏与恢复方向 |
| 2024 | Robust IQL / data corruption robust offline RL 相关工作 | NeurIPS/CoRR 相关线索 | TD-learning robustness | 本文比较和动机中的相邻方向 |

作者团队的研究轨迹集中在两个问题：一是离线数据不干净时如何稳健学习；二是序列模型能否替代传统 Bellman bootstrapping，减少误差传播。

## 2. 研究问题

现实离线数据通常来自传感器、人类操作或日志系统，可能包含状态噪声、动作记录错误和奖励污染。标准 offline RL 已经要面对分布偏移，如果数据本身再被污染，TD target 会把错误通过 Bellman backup 传播到许多状态动作对。

论文研究三类损坏：state attack、action attack、reward attack，并区分 random corruption 与 adversarial corruption。关键设置是有限数据：当只使用完整 D4RL 数据的一小部分时，鲁棒 TD 方法（如 UWMSG、RIQL）并不能可靠解决问题。

RDT 的目标不是恢复真实 MDP，而是在损坏数据集 $D_c$ 上学到尽可能高回报且对测试扰动稳健的策略。DT 类方法把策略学习写成条件序列预测：

$$
a_t = f_\theta(R_{1:t}, s_{1:t}, a_{1:t-1}).
$$

含义：动作由上下文历史预测，而不是通过 $Q(s_t,a_t) \leftarrow r_t + \gamma Q(s_{t+1},a_{t+1})$ 的 bootstrap 更新得到。

## 3. 背景知识

传统 TD-learning 的优点是可以做动态规划式价值传播，但在损坏数据中它也会传播错误。若 $s_{t+1}$ 被污染，则 $Q(s_{t+1},a')$ 的估计错误会成为 $Q(s_t,a_t)$ 的目标；后续再反复 bootstrap，错误会扩散。

Decision Transformer 把 offline RL 变成监督学习：给定 reward-to-go、状态、历史动作，预测下一个动作。它没有显式 Bellman backup，因此单点状态损坏不容易沿时间反向污染整条价值函数。Transformer 的上下文还可以利用前后状态模式“忽略”不一致 token。

本文三种损坏可以直观理解为：状态损坏类似传感器读数错误；动作损坏类似执行器日志错记或噪声执行；奖励损坏类似人为标注或环境奖励记录错误。对于 DT，动作损坏更接近 label noise，状态损坏更接近 input noise，奖励损坏会影响条件 token。

| 术语 | 直观含义 | 在本文中的作用 |
|---|---|---|
| Data corruption | 离线轨迹中状态、动作或奖励被噪声污染 | 核心问题 |
| TD bootstrapping | 用下一状态价值构造当前价值目标 | 脆弱性来源 |
| Decision Transformer | 将 RL 写成条件序列预测 | RDT 的基础模型 |
| Embedding dropout | 随机丢弃输入 token embedding | 模拟/抵抗错误输入 |
| Gaussian weighted learning | 大误差样本降权 | 抑制疑似损坏标签 |
| Iterative data correction | 用模型预测纠正异常动作 | 从数据源减少动作污染 |

为什么动机成立：robust offline RL 过去常在 Q-learning 上加 ensemble、Huber loss 或不确定性惩罚，但这些方法仍依赖 bootstrapping。本文指出，面对有限损坏数据时，先换掉问题表述本身，可能比在 TD 框架里继续补丁更有效。

## 4. 问题分析

Figure 1 是全文的动机证据：在随机数据损坏且数据量有限的设置下，DT 和 DeFog 等序列模型明显比 TD-learning robust baseline 更稳，尤其在 state attack 下优势突出。

论文在 Section 3.2 给出的解释是：TD-learning 对 state corruption 特别敏感，因为错误下一状态会污染 bootstrapped target；sequence modeling 预测当前动作时读取历史上下文，不把未来错误状态作为递归目标，因此错误传播链短得多。

Figure 2 给出 RDT 框架。RDT 不把 reward-to-go 作为唯一预测对象，而是同时考虑动作与奖励预测，使模型能识别当前数据点是否与上下文一致。Figure 3 分别验证三项组件：embedding dropout 优于普通 dropout；Gaussian weighting 能降低大误差样本影响；iterative correction 能逐步替换明显损坏动作。

## 5. 思想与方法

RDT 的指导原则是：保留 DT 的非 bootstrapping 优势，再用鲁棒监督学习工具处理输入噪声和标签噪声。

第一，embedding dropout 直接作用于 token embedding。它让模型训练时习惯“某些状态或奖励 token 不可靠”，因此测试时遇到损坏观测也不至于崩溃。

第二，Gaussian weighted learning 根据预测误差给样本加权：

$$
w_i = \exp\left(-\frac{\ell_i}{2\sigma^2}\right),
$$

其中 $\ell_i$ 可以是动作预测误差。误差越大的样本越可能是损坏点，训练影响越小。

第三，iterative data correction 使用当前模型预测与原始动作比较，若差异过大，则用模型预测替换疑似损坏动作，再重新训练。它适合 action corruption，因为动作是监督目标。

新意在于组合视角：这些技术本身不复杂，但放在 offline RL 数据损坏问题中，形成了“序列建模天然鲁棒 + 噪声监督学习增强”的方案。

## 6. 算法与伪代码

算法名：Robust Decision Transformer (RDT)。

1. 输入损坏离线数据集 $D_c$，构造 DT 训练序列。
2. 对每个状态、动作、奖励 token 编码为 embedding，并以概率 $p$ 做 embedding dropout。
3. 用 Transformer 预测动作和奖励。
4. 计算每个训练样本的预测误差 $\ell_i$。
5. 用 Gaussian weight $w_i$ 对 loss 加权，降低大误差样本的梯度贡献。
6. 训练若干轮后，用模型预测动作 $\hat a_t$ 检查原动作 $a_t$。
7. 若 $\|a_t-\hat a_t\|$ 超过阈值或被判定为异常，则替换或修正该动作。
8. 在修正后的数据集上迭代训练，直到达到预设轮数。
9. 测试时按 DT 方式给定目标回报和历史上下文，自回归输出动作。

整体目标可写成：

$$
\min_\theta \sum_i w_i \cdot \ell(f_\theta(x_i), y_i),
\quad
w_i = \exp\left(-\frac{\ell_i}{2\sigma^2}\right).
$$

直观解释：模型不是平等相信所有离线样本，而是让与上下文严重不一致的样本少说话。

关键超参数包括 embedding dropout 概率、Gaussian weighting 的 $\sigma$、迭代纠正频率和异常阈值。论文默认 embedding dropout 使用 0.1，并在 Figure 6 做组件消融。

## 7. 实验与消融

实验覆盖 MuJoCo、Kitchen、Adroit 任务，使用 D4RL/NeoRL 等数据，并构造随机损坏、对抗损坏、混合损坏以及“训练时损坏 + 测试时观测扰动”的更难设置。评价指标是 normalized score，通常跨任务组和随机种子平均。

baseline 包括 TD3+BC、CQL、IQL、UWMSG、RIQL、DeFog、vanilla DT 等。Table 1 报告随机损坏下的详细比较，RDT 在 state/action/reward attack 中总体最好。Table 2 显示对抗损坏下 RDT 仍优于其他方法。Figure 4 报告混合损坏；Figure 5 说明在测试时观测扰动下 RDT 的鲁棒性仍更强。

消融见 Figure 6：三个组件单独加入都能改善 DT，组合后最好。Figure 7 进一步表明，仅把 Transformer backbone 加到 UWMSG 上不能复现 RDT 的收益，说明优势不只是模型容量，而是序列建模目标与鲁棒训练机制共同产生。

实验能证明 RDT 在作者设定的多类损坏下很强；仍较弱的地方是损坏模型多为人为构造，高维视觉真实传感器噪声、人类策略系统性偏差和 reward hacking 式污染还需要进一步验证。

## 8. 展望

启发：第一，robust offline RL 不一定必须从 Q-learning 出发；第二，减少 bootstrapping 可以显著降低状态噪声的传播；第三，监督学习中的 robust loss、dropout、data correction 可以直接迁移到 DT 类 ORL；第四，数据清洗和策略学习可以联合迭代。

局限：RDT 依赖 Transformer 上下文质量，极低覆盖数据下仍会失败；iterative correction 主要适合动作损坏，对奖励语义错误更难；Gaussian weighting 可能把罕见但重要的高价值样本误判为异常；实验多为模拟损坏。

后续研究方向：1. 将 RDT 与不确定性估计结合，区分“罕见高价值样本”和“损坏样本”。2. 研究真实机器人日志中的传感器漂移和动作延迟损坏。3. 把 iterative correction 扩展到 reward relabeling 或 trajectory-level denoising。

## Links

- Paper page: https://openreview.net/forum?id=phAlw3JPms
- OpenReview PDF: https://openreview.net/pdf?id=phAlw3JPms
- arXiv: https://arxiv.org/abs/2407.04285
- Code: https://github.com/jiawei415/RobustDecisionTransformer
- 本地 PDF: `[ICLR_2025] Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling.pdf`
