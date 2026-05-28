---
title: "C2IQL: Constraint-Conditioned Implicit Q-learning for Safe Offline Reinforcement Learning"
description: "C2IQL 将 IQL 扩展到安全离线 RL，用隐式约束更新、成本重构和阈值条件化同时追求奖励与安全。"
tags:
  - Safe Offline RL
  - Implicit Q-learning
  - Constrained RL
  - Cost Reconstruction
---

# C2IQL: Constraint-Conditioned Implicit Q-learning for Safe Offline Reinforcement Learning

<div class="paper-hero">
<p class="paper-meta">Zifan Liu, Xinran Li, Jun Zhang · ICML 2025 · ICML / 2025</p>
<div class="tag-row"><span>Safe Offline RL</span><span>Implicit Q-learning</span><span>Constrained RL</span><span>Cost Reconstruction</span></div>
</div>

<article class="note-body" markdown>

| 字段 | 内容 |
|---|---|
| Title | C2IQL: Constraint-Conditioned Implicit Q-learning for Safe Offline Reinforcement Learning |
| Year | 2025 |
| Source | ICML 2025, PMLR 267:38827-38841 |
| Authors | Zifan Liu, Xinran Li, Jun Zhang |
| Affiliations | The Hong Kong University of Science and Technology, Department of Electronic and Computer Engineering |
| Tags | Safe Offline RL, CMDP, IQL, Constraint Conditioning, DSRL |

**一句话概括**：C2IQL 先把 IQL 的 in-dataset 隐式更新推广到 CMDP，再用非折扣成本重构和 constraint-conditioned 输入解决安全阈值估计不准、阈值不灵活的问题。

**我对这篇论文的定位**：这是一篇 safe offline RL 方法论文，定位在 CPQ/COptiDICE/CDT/FISOR 等工作的交叉处。它继承 IQL 的优点：避免显式对数据外动作做 max；同时补上 safe RL 的约束处理，使策略能在不同 cost threshold 下调节 reward-cost trade-off。

## 1. 第一作者相关信息

第一作者 **Zifan Liu**，论文首页和 PMLR 均标注其机构为香港科技大学电子与计算机工程系。当前公开来源足以确认其为本文第一作者；未在本次阅读中系统核验其个人主页和完整发表列表。

| year | title | venue/source | topic tag | relation to this paper |
|---|---|---|---|---|
| 2025 | C2IQL: Constraint-Conditioned Implicit Q-learning for Safe Offline Reinforcement Learning | ICML / PMLR | Safe Offline RL | 本文 |

研究方向从本文看主要是安全强化学习、离线强化学习、约束优化和隐式值函数方法。

## 2. 研究问题

Safe offline RL 要在不与环境交互的前提下，从固定数据集中学习一个策略，最大化累计奖励并满足安全成本约束。CMDP 目标为：

$$
\max_\pi \mathbb{E}_{\tau\sim\pi}[R(\tau)],
\quad
\text{s.t. } \mathbb{E}_{\tau\sim\pi}[C(\tau)]\le L.
$$

其中 $R(\tau)=\sum_t r_t$，$C(\tau)=\sum_t c_t$，$L$ 是成本阈值。offline 设置下的难点是：显式策略更新可能选择数据外动作，导致奖励和成本估计都不可靠；同时多数方法用折扣成本近似原始非折扣约束，可能造成 false positive/false negative。

## 3. 背景知识

IQL 的关键是避免对任意动作做 $\max_a Q(s,a)$，只在数据动作分布内通过 expectile regression 估计较高分位的 value，再用 advantage-weighted regression 抽取策略。因此它在 offline RL 中相对稳健。

Safe RL 则通常处理 CMDP，常见方法包括 primal-dual、CPO、CPQ 等。CPQ 通过成本 Q 屏蔽不安全动作，但依赖 OOD 检测，仍可能误判。C2IQL 希望保留 IQL 的 OOD avoidance，同时实现约束惩罚更新。

| term | plain Chinese meaning | role in this paper |
|---|---|---|
| CMDP | 带成本约束的 MDP | 安全 RL 形式化 |
| IQL | 数据内隐式 Q 学习 | 避免 OOD 的基础 |
| CIQL | Constrained IQL | C2IQL 的基础版本 |
| CRM | Cost Reconstruction Model | 从折扣成本恢复非折扣累计成本 |
| constraint conditioning | 把阈值作为 value/policy 输入 | 支持动态安全要求 |
| DSRL | safe offline RL benchmark | 实验数据来源 |

**为什么动机成立**：安全约束通常是非折扣累计成本，但 Bellman backup 为稳定性常用折扣成本。相同累计成本因为发生时间不同会有不同折扣值；同一折扣值也可能对应不同真实累计成本。安全任务中这种错配会直接导致过保守或不安全。

## 4. 问题分析

作者先指出现有 SORL 的两类瓶颈：offline 角度，OOD detection/regularization 只能缓解不能完全避免 OOD；safe RL 角度，折扣成本约束不准确，固定阈值无法处理长时域中“已用预算/剩余预算”的动态变化。

Figure 1 随机生成成本轨迹，展示折扣成本 $\tilde{C}$ 与真实累计成本 $C$ 的不一致。早期集中成本可能在折扣值上过高而被错误拒绝，晚期集中成本可能折扣值低而被错误接受。CRM 重构后的成本 $\hat{C}$ 显著减少这两类错误。

IQL 的普通 reward backup 为：

$$
Q_r^\pi(s_t,a_t)=r_t+\gamma V_r^\pi(s_{t+1}).
$$

CIQL 引入约束惩罚项：

$$
Q_{r|c}^\pi(s,a)=\mathbf{1}(Q_c^\pi(s,a)\le \tilde{L})Q_r^\pi(s,a),
$$

即只让满足成本约束的动作贡献 reward value。

## 5. 思想与方法

C2IQL 的方法由三层组成。

第一层是 **CIQL**：把 CPQ 的 shield/constraint-penalized update 注入 IQL，但仍保持 expectile regression 和 in-dataset update。Theorem 1 给出隐式策略是行为策略的重加权，因此可在不显式采样新动作的情况下更新 reward value 和 cost value。

第二层是 **cost reconstruction model (CRM)**：训练一个模型 $R_c$，输入多个折扣因子下的 discounted cost values，输出非折扣累计成本估计：

$$
\hat{Q}_c^\pi(s,a)=R_c(Q_c^{\pi,\gamma_1}(s,a),\ldots,Q_c^{\pi,\gamma_m}(s,a)).
$$

第三层是 **constraint-conditioned ability**：把阈值 $\hat{L}$ 作为 value、cost value、Q 和 policy 的输入，例如 $V_r^\pi(s,\hat{L})$、$\pi(a|s,\hat{L})$。这让同一个模型能对不同安全预算调整策略。

## 6. 算法与伪代码

算法名：**C2IQL**。

1. 初始化 reward value、cost value、reward Q、cost Q、policy 和 cost reconstruction model。
2. 预设 reward 折扣因子 $\gamma$、多个 cost 折扣因子 $\{\gamma_1,\ldots,\gamma_m\}$、expectile 参数 $\kappa_1,\kappa_2$ 和阈值集合 $L$。
3. 每轮从离线数据 $D$ 采样 $(s,a,r,c,s')$，并随机采样阈值 $\hat{L}\in L$。
4. 用多个 cost Q 网络得到 $Q_c^{\pi,\gamma_i}(s,a,\hat{L})$，经 CRM 得到 $\hat{Q}_c^\pi(s,a,\hat{L})$。
5. 构造约束惩罚 reward Q：

$$
Q_{r|c}^\pi(s,a,\hat{L})
=\mathbf{1}(\hat{Q}_c^\pi(s,a,\hat{L})\le \hat{L})Q_r^\pi(s,a,\hat{L}).
$$

6. 用 expectile regression 更新 reward value。
7. 用 Theorem 1 的隐式权重更新 cost value。
8. 用 TD loss 更新 reward Q 和多个 cost Q。
9. 用加权行为克隆形式抽取策略：

$$
L_\pi=-\mathbb{E}_{(s,a)\sim D}
\left[
|\kappa_2-\mathbf{1}(Q_{r|c}^\pi(s,a,\hat{L})-V_r^\pi(s,\hat{L})<0)|
\log\pi(a|s,\hat{L})
\right].
$$

这里 $\kappa_2$ 类似控制策略离行为克隆有多近的温度参数。

## 7. 实验与消融

环境与数据：Bullet-Safety-Gym 和 SafetyGymnasium，数据格式来自 DSRL。主表覆盖 Run/Circle 两类任务，以及 Ant、Ball、Car、Drone 四类机器人。评价使用 normalized reward 和 normalized cost，成本阈值归一化为 1；每项为 10 evaluation episodes、5 seeds、3 thresholds 平均。

基线：BCQ-Lag、BEAR-Lag、COptiDICE、CPQ、FISOR、VOCE、WSAC、CDT。它们分别代表 primal-dual、constraint penalization、distribution correction、sequence modeling 与 variational/conservative safe offline RL。

主结果：Table 1 显示 C2IQL 平均 reward 0.74、平均 cost 0.76，整体安全且 reward 最高；CDT reward 0.71、cost 0.91 也强，但 Figure 2 显示 C2IQL 随 small/middle/large 阈值更灵活，CDT 在小阈值或大阈值下容易不够安全或不够进取。FISOR 很安全但过保守，平均 reward 仅约 0.31。

消融：Figure 3 比较 CIQL、C2IQL w/o CC、C2IQL w/o CR、完整 C2IQL。CIQL 因缺少 CRM 和阈值条件化，成本接近 0、reward 很低；w/o CC 更保守；w/o CR reward 较差。完整 C2IQL 最能贴近阈值并获得高奖励。

超参数：Figure 4 显示 $\kappa_1$ 影响安全性，过大容易受 lucky safe samples 影响而不安全；$\kappa_2$ 更影响 reward，较大 $\kappa_2$ 通常提高性能而对安全影响小。Table 2 显示 CRM 输入加小噪声可略微提升泛化，大噪声使策略保守。

## 8. 展望

研究启发：

1. safe offline RL 中“避免 OOD”与“满足成本约束”必须同时处理，不能只靠后验惩罚。
2. 非折扣安全约束与折扣 Bellman backup 的错配是被低估的问题。
3. 把安全预算作为条件输入，比为每个阈值单独训练模型更适合部署。
4. IQL 的隐式策略视角可推广到 constrained setting。

局限与开放问题：

1. CRM 的准确性对成本估计很关键，复杂长时域任务可能更难重构。
2. 阈值集合 $L$ 需要预设，超出训练阈值范围的泛化未充分证明。
3. 实验主要是 locomotion safety benchmark，真实医疗/自动驾驶成本定义更复杂。
4. 方法仍依赖数据覆盖足够的安全高奖励行为；若数据本身缺少可行策略，C2IQL 也难以创造。

后续想法：

1. 将 CRM 换成带不确定性的 cost reconstructor，对高不确定成本更保守。
2. 研究连续阈值外推，让 constraint-conditioned policy 支持部署时任意预算。
3. 与 diffusion/trajectory generative policies 结合，提高 safe offline RL 的多模态动作表达。

## Links

- Paper page: [PMLR](https://proceedings.mlr.press/v267/liu25ai.html)
- PDF: [PMLR PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/liu25ai/liu25ai.pdf)
- OpenReview: [OpenReview via PMLR](https://proceedings.mlr.press/v267/liu25ai.html)
- Code: 未能从论文/PMLR 公开来源确认官方代码链接
- Local PDF: `[ICML_2025] C2IQL Constraint-Conditioned Implicit Q-learning for Safe Offline Reinforcement Learning.pdf`

</article>
