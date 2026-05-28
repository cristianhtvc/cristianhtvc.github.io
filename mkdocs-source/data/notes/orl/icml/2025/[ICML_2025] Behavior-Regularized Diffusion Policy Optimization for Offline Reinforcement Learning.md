---
title: "Behavior-Regularized Diffusion Policy Optimization for Offline Reinforcement Learning"
description: "BDPO 将行为正则化写到扩散反向路径上，用可解析 pathwise KL 和双时间尺度 actor-critic 优化离线扩散策略。"
tags:
  - Offline RL
  - Diffusion Policy
  - Behavior Regularization
  - Actor-Critic
---

# Behavior-Regularized Diffusion Policy Optimization for Offline Reinforcement Learning

| 字段 | 内容 |
|---|---|
| Title | Behavior-Regularized Diffusion Policy Optimization for Offline Reinforcement Learning |
| Year | 2025 |
| Source | ICML 2025, PMLR 267:18630-18657 |
| Authors | Chen-Xiao Gao, Chenyang Wu, Mingjun Cao, Chenjun Xiao, Yang Yu, Zongzhang Zhang |
| Affiliations | Nanjing University；The Chinese University of Hong Kong, Shenzhen |
| Tags | Offline RL, Diffusion Policy, Behavior Regularization, Pathwise KL, D4RL |

**一句话概括**：BDPO 证明并实现了一种面向扩散策略的行为正则化 offline RL 框架，把 KL 正则项分解为反向扩散路径上每一步 transition kernel 的差异，并用双时间尺度 actor-critic 高效优化。

**我对这篇论文的定位**：这是扩散策略用于 offline RL 的一篇“机制补课”论文。过去很多方法要么只用扩散模型生成候选动作，要么对扩散 actor 直接回传 Q 梯度；BDPO 试图回答一个更基础的问题：如果行为正则化目标本来是 $D_{\mathrm{KL}}(\pi||\nu)$，那么当 $\pi$ 和 $\nu$ 都是扩散过程时，KL 应该怎么算、又该怎么优化。

## 1. 第一作者相关信息

第一作者 **Chen-Xiao Gao**，论文首页标注其为南京大学 National Key Laboratory for Novel Software Technology / School of Artificial Intelligence 成员。公开 PMLR、ICML virtual poster、OpenReview 均确认其为第一作者。

| year | title | venue/source | topic tag | relation to this paper |
|---|---|---|---|---|
| 2025 | Behavior-Regularized Diffusion Policy Optimization for Offline Reinforcement Learning | ICML / PMLR | Diffusion Offline RL | 本文 |

从本文和合作网络看，其研究方向集中在 offline RL、生成式策略参数化、扩散模型与强化学习优化。未在本次阅读中系统核验完整发表列表。

## 2. 研究问题

offline RL 中策略优化容易查询数据外动作并高估其价值。行为正则化通过约束策略 $\pi$ 接近行为策略 $\nu$ 来减少 OOD exploitation：

$$
\max_\pi \mathbb{E}_\pi\left[\sum_{t=0}^{\infty}\gamma^t
\left(r_t-\eta D_{\mathrm{KL}}(\pi(\cdot|s_t)||\nu(\cdot|s_t))\right)\right].
$$

问题是：扩散策略的 action log-probability 很难精确计算，直接在 clean action distribution 上算 KL 不方便。本文要解决的是：如何在扩散策略参数化下实现原则性的行为正则化，并避免全路径反传带来的高内存和高方差。

## 3. 背景知识

行为正则化 offline RL 的直觉是：离线数据以外的动作没有真实反馈，策略如果偏离行为分布太远，Q 函数可能出现危险外推。常见做法包括 BCQ、TD3+BC、AWAC、IQL 等，其中一部分显式约束策略，一部分通过 in-sample backup 避免 OOD。

扩散策略把动作生成看作从噪声 $a^N$ 逐步去噪到动作 $a^0$ 的反向 Markov 过程。优势是能表达多峰动作分布；困难是 clean action 的显式密度和 KL 不容易算。

| term | plain Chinese meaning | role in this paper |
|---|---|---|
| behavior policy $\nu$ | 离线数据采集策略 | 正则化参照 |
| diffusion policy | 多步反向去噪生成动作的策略 | actor 参数化 |
| pathwise KL | 整条扩散路径上的 KL | 本文核心正则项 |
| diffusion value $V_n$ | 扩散第 $n$ 步的中间价值 | 降低全路径反传成本 |
| LCB target | 用 ensemble 下置信界作 value target | 实践稳定技巧 |

**为什么动机成立**：离线数据常来自多个策略 checkpoint，动作分布多峰。用高斯或确定性策略表达行为分布会把多个模式平均成“中间动作”，正则化会误导策略。扩散模型能表达多峰行为，但必须有配套的 KL 与 actor-critic 训练方式。

## 4. 问题分析

论文 Figure 1 展示了 unimodal policy 的问题：当行为分布多峰时，均值或单峰高斯会把行为中心放在低质量区域，导致行为正则化方向错误。扩散策略保留多峰结构，但 clean action KL 难算。

作者的关键分解是利用扩散反向过程的 Markov 性，将整条路径 KL 写成每个反向 transition kernel 的 KL 之和：

$$
D_{\mathrm{KL}}[p_{0:N}^{\pi,s}||p_{0:N}^{\nu,s}]
=
\mathbb{E}\left[
\sum_{n=1}^N
D_{\mathrm{KL}}\left(p_{n-1|n}^{\pi,s,a^n}||p_{n-1|n}^{\nu,s,a^n}\right)
\right].
$$

每一步反向 transition 近似为高斯时，KL 有解析形式：

$$
D_{\mathrm{KL}}
=
\frac{\|\mu_{\pi,s}^n(a^n)-\mu_{\nu,s}^n(a^n)\|^2}{2\sigma_n^2}.
$$

这把“动作分布 KL”转成“每个去噪方向均值差异”的累计惩罚。

## 5. 思想与方法

BDPO 的 guiding principle 是：不要对最终动作分布硬算 KL，而是在扩散生成路径上逐步惩罚 actor diffusion direction 偏离 behavior diffusion direction。

方法包含两层 actor-critic：

第一层是环境 MDP 的 Q-value $Q^\pi(s,a^0)$，它评估最终动作执行后的回报。TD target 包含下一状态 Q 和下一步扩散路径上的累计 KL penalty。

第二层是扩散内部的 value functions $V_n^{\pi,s}(a^n)$，它估计从中间噪声动作 $a^n$ 继续去噪到最终动作时的价值与惩罚。这样策略更新只需单步 reverse transition，而不是像 Diffusion-QL 那样保留整条扩散路径的计算图。

理论上，Theorem 4.2 说明 pathwise KL-regularized RL 的最优扩散策略对应原始 clean-action KL 正则目标的最优策略；Proposition 4.4/4.5 给出 soft policy improvement / policy iteration 收敛论证。

## 6. 算法与伪代码

算法名：**BDPO**。

1. 输入离线数据集 $D$，初始化 actor diffusion policy $p^\pi$、behavior diffusion $p^\nu$、Q 网络和 diffusion value 网络。
2. 用行为克隆/扩散训练目标预训练 $p^\nu$，使其拟合数据行为分布。
3. 用 $p^\nu$ 初始化 $p^\pi$。
4. 重复训练：
5. 从 $D$ 采样 batch $(s,a,s',r)$。
6. 用环境 TD target 更新 Q networks：

$$
B^\pi Q^\pi(s,a)=r+\gamma\mathbb{E}\left[
Q^\pi(s',a'^0)-\eta\sum_{n=1}^{N}\ell_n^{\pi,s'}(a'^n)
\right].
$$

7. 为每个样本采样扩散步 $n$，并用 forward noising 得到 $a^n$。
8. 用 diffusion TD target 更新 $V_n^{\pi,s}$：

$$
T_n^\pi V_n^{\pi,s}(a^n)
=-\eta \ell_n^{\pi,s}(a^n)
+\mathbb{E}_{p_{n-1|n}^{\pi,s,a^n}}[V_{n-1}^{\pi,s}(a^{n-1})].
$$

9. 每隔若干步更新 actor，使其最大化单步去噪目标。
10. 评估时采样多个动作，论文实践中每个状态扩散 $N=10$ 个动作并选 Q 值最高者执行。

## 7. 实验与消融

实验 1：synthetic 2D datasets，包括 8gaussians、2spirals、moons。Figure 4/5 显示 BDPO 生成分布能接近由能量重加权得到的目标分布；中间 diffusion value 在低噪声步给出更清晰的局部引导。

实验 2：D4RL locomotion-v2 和 antmaze-v0。基线包括 CQL、IQL、Decision Diffuser、SfBC、IDQL-A、QGPO、SRPO、DTQL、Diffusion-QL、EDP、DAC。Table 1 显示 BDPO 在 locomotion 总分 856.7，高于 DAC 836.4、DTQL 798.3；antmaze 总分 519.6，也高于主要扩散策略基线。

消融与敏感性：Figure 6 分析 regularization strength $\eta$；过大可能过度贴近行为，过小可能不稳定。Figure 7 分析 LCB coefficient $\rho$；LCB 对 locomotion 稳定性有明显影响。附录比较 policy parameterization，说明扩散 actor 比确定性/高斯 actor 更适合多峰行为正则化。

实验能证明：pathwise KL + 双时间尺度 critic 在 D4RL 上有效且效率优于全路径反传。实验仍未完全证明：大规模视觉机器人任务或更复杂策略分布下，预训练 behavior diffusion 的误差不会主导性能。

## 8. 展望

研究启发：

1. 对扩散策略做 offline RL 时，正则化应落在生成过程而不只是最终动作。
2. 复杂策略参数化需要重新设计 actor-critic，而不是简单套用高斯策略目标。
3. diffusion value functions 是连接生成过程和 RL value backup 的有用抽象。
4. 多峰行为数据是扩散策略相对高斯策略的主要优势场景。

局限与开放问题：

1. 需要先训练准确的 behavior diffusion，若数据稀疏或高维视觉动作复杂，误差可能较大。
2. 评估时采样多个动作并用 Q 选取，会引入额外计算和 Q 选择偏差。
3. 理论假设包含 admissible policy 与行为分布支持比值有界，现实中未必满足。
4. antmaze 部分任务方差较大，长程稀疏奖励仍具挑战。

后续想法：

1. 将 BDPO 与 uncertainty-aware behavior diffusion 结合，降低行为模型误差。
2. 在机器人 imitation + offline RL 数据上测试多峰动作优势。
3. 把 pathwise KL 推广到 flow matching 或 consistency policy 等生成式 actor。

## Links

- Paper page: [PMLR](https://proceedings.mlr.press/v267/gao25q.html)
- PDF: [PMLR PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/gao25q/gao25q.pdf)
- arXiv: [2502.04778](https://arxiv.org/abs/2502.04778)
- Code: [typoverflow/flow-rl](https://github.com/typoverflow/flow-rl)
- ICML poster: [Virtual poster](https://icml.cc/virtual/2025/poster/44003)
- Local PDF: `[ICML_2025] Behavior-Regularized Diffusion Policy Optimization for Offline Reinforcement Learning.pdf`
