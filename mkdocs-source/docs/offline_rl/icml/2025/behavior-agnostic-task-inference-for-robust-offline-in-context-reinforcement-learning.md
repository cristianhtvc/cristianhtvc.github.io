---
title: "Behavior-agnostic Task Inference for Robust Offline In-context Reinforcement Learning"
description: "BATI 用动力学似然替代上下文编码器的近似贝叶斯推断，缓解 offline ICRL 中由行为策略变化造成的上下文分布偏移。"
tags:
  - Offline RL
  - In-context RL
  - Meta-RL
  - Robust Task Inference
---

# Behavior-agnostic Task Inference for Robust Offline In-context Reinforcement Learning

<div class="paper-hero">
<p class="paper-meta">Long Ma, Fangwei Zhong, Yizhou Wang · ICML 2025 · ICML / 2025</p>
<div class="tag-row"><span>Offline RL</span><span>In-context RL</span><span>Meta-RL</span><span>Robust Task Inference</span></div>
</div>

<article class="note-body" markdown>

| 字段 | 内容 |
|---|---|
| Title | Behavior-agnostic Task Inference for Robust Offline In-context Reinforcement Learning |
| Year | 2025 |
| Source | ICML 2025, PMLR 267:42293-42308 |
| Authors | Long Ma, Fangwei Zhong, Yizhou Wang |
| Affiliations | Peking University；Beijing Institute for General Artificial Intelligence；Beijing Normal University；State Key Laboratory of General Artificial Intelligence |
| Tags | Offline ICRL, Meta-RL, Context Shift, Dynamics Model, Maximum Likelihood |

**一句话概括**：BATI 认为 offline ICRL 的上下文编码器容易学到“任务与采样行为策略”的伪相关，因此用条件动力学似然 $p(X^t|X^b,M)$ 在测试时搜索任务 latent，使任务推断尽量不依赖行为策略。

**我对这篇论文的定位**：这篇文章位于 offline meta-RL / in-context RL 与鲁棒任务推断之间。它的价值不在于一个复杂策略优化器，而在于指出：很多 offline ICRL 方法本质上是在训练分布上做近似贝叶斯后验，一旦测试上下文由不同策略采集，任务 latent 会被行为偏移污染。

## 1. 第一作者相关信息

第一作者 **Long Ma**。论文首页标注其机构为北京大学前沿交叉学科研究院数据科学中心和北京通用人工智能研究院。公开项目页和 PMLR 页面均确认其为本文第一作者。

| year | title | venue/source | topic tag | relation to this paper |
|---|---|---|---|---|
| 2025 | Behavior-agnostic Task Inference for Robust Offline In-context Reinforcement Learning | ICML / PMLR | Offline ICRL | 本文 |

从本文看，Long Ma 的研究重心包括 in-context reinforcement learning、meta-RL、鲁棒任务表征与离线强化学习。未在本次阅读中系统核验其完整近五年发表列表。

## 2. 研究问题

ICRL 希望 agent 在测试时通过一段 context trajectory 推断当前任务，并在不更新参数的情况下执行合适策略。offline ICRL 进一步要求训练时只用固定离线数据，不在线采集。

本文研究的问题是：当测试时 context 是由不同于训练分布的行为策略 $\mu$ 采集时，如何稳健推断任务 $M$？形式化地，任务是 MDP $M=(S,A,P,R,\rho_0,\gamma)$，context 为

$$
X=\{(s_i,a_i,r_i,s'_i)\}_{i=0}^{C-1},
$$

其中 $X^b=\{(s_i,a_i)\}$ 表示行为部分，$X^t=\{(r_i,s'_i)\}$ 表示由环境动力学和奖励生成的结果部分。目标是学习 meta-policy $\pi_\theta:S\times X\to\Delta A$，在新任务上根据 context 行动。

## 3. 背景知识

Meta-RL 训练 agent 快速适应新任务；ICRL 则借鉴 LLM in-context learning，把“适应”变成基于上下文的前向推断，而不是梯度更新。offline ICRL 的诱人之处是无需在线训练，但固定数据会让 context distribution shift 更严重。

传统 task inference 方法通常学习一个 encoder $f_\phi(X)=Z$，让 $Z$ 包含任务信息。问题是，训练数据中的任务 $M$ 与采样策略 $\mu$ 往往相关：某类任务可能总由某类策略采集。encoder 可能学到“看到这种动作行为，就猜这个任务”，而不是根据环境反馈和动力学判断任务。

| term | plain Chinese meaning | role in this paper |
|---|---|---|
| ICRL | 用测试时上下文推断任务并行动 | 问题背景 |
| context shift | 测试 context 与训练 context 分布不同 | 本文主要失效模式 |
| $X^b$ | state-action 行为片段 | 易携带行为策略伪相关 |
| $X^t$ | reward-next-state 结果片段 | 更直接反映任务动力学/奖励 |
| BATI | Behavior-agnostic Task Inference | 本文方法 |
| Recon | 同样用重构损失但有 encoder 的基线 | 证明 encoder 是瓶颈 |

**为什么动机成立**：如果训练时“月球任务”总由穿宇航服的人采样，encoder 会把宇航服当任务线索；测试时行为换了，这条线索失效。BATI 选择依赖跳跃高度这类动力学结果，而不是衣着这类行为相关信号。

## 4. 问题分析

论文的图模型 Figure 2 表示：任务 $M$ 与采样策略 $\mu$ 共同影响 context；训练和测试时 $p(M,\mu)$ 可能变化，导致 $X^b$ 和 $X^t$ 都 shift。作者指出直接估计 $p(M|X)$ 会不可避免依赖 $p(M,\mu)$：

$$
p(M|X)\propto \int p(M,\mu)p(X|M,\mu)d\mu.
$$

这说明 learned encoder 的贝叶斯后验会被训练时任务-策略相关性污染。

作者进一步提出，如果条件化 $X^b$，则 $\mu$ 到 $X^t$ 的路径被阻断，任务可通过动力学/奖励结果推断：

$$
\arg\max_M \log p(X^t|X^b,M).
$$

Theorem 3.1 还说明，CSRO 那类同时最大化 $I(Z;M)$、最小化 $I(Z;X^b)$ 的目标在某些情况下不可兼得：

$$
I(Z;M)-I(Z;X^b)\le H(M)-I(M;X^b).
$$

如果 $M$ 与 $X^b$ 本来强相关，想同时保留任务信息又去掉行为信息会变得困难。

## 5. 思想与方法

BATI 的核心思想是：训练时不学一个从 context 到 latent 的 encoder；测试时也不让 encoder 直接读行为轨迹。它为训练任务维护 latent embedding table，并训练动力学 estimator，使 latent 能解释 $X^b$ 到 $X^t$ 的映射。测试时给定新 context，直接搜索最能解释 $X^t$ 的 latent。

训练目标为负对数似然形式的重构损失：

$$
L_{\mathrm{recon}}^{X,Z}
=\frac{(X^t-g_\psi(X^b,Z))^2}{\exp h_\psi(X^b,Z)}+h_\psi(X^b,Z).
$$

这里 $g_\psi$ 预测 reward/next-state，$h_\psi$ 表示不确定性或方差项。策略和值函数使用 IQL 作为基础 offline RL 算法，但 task embedding 的梯度只由 reconstruction loss 监督，避免策略损失直接把行为偏差写入 latent。

测试时，BATI 从训练得到的 task embedding 分布中采样或优化 $Z^\*$，使 $L_{\mathrm{recon}}$ 最小，然后将 $Z^\*$ 输入策略执行。真正的新意是把 task inference 从 amortized encoder 改成 test-time maximum-likelihood inference。

## 6. 算法与伪代码

算法名：**BATI**。

训练阶段：

1. 对每个训练任务 $M$ 初始化一个 latent distribution $Z_M$。
2. 从任务离线数据 $D_M$ 中采样 context $X=(X^b,X^t)$。
3. 采样 $Z\sim Z_M$。
4. 用 dynamics estimator $(g_\psi,h_\psi)$ 预测 $X^t$。
5. 用 $L_{\mathrm{recon}}$ 更新 task embeddings 和 dynamics estimator。
6. 同时用 IQL 训练 context-conditioned policy 与 value functions，但停止 policy/value loss 对 task embedding 的梯度。

测试阶段：

1. 收集未知任务的一段 context $X$。
2. 在 latent 空间中搜索

$$
Z^\*=\arg\min_Z L_{\mathrm{recon}}^{X,Z}.
$$

3. 将 $Z^\*$ 输入策略 $\pi_\theta(a|s,Z^\*)$。
4. 在环境中 rollout 并评估。

训练和推理差异很关键：训练不需要在线交互；测试中的“适应”是 latent 搜索，不是参数更新。

## 7. 实验与消融

环境：MuJoCo-based offline ICRL，包括 AntDir、HalfCheetahVel、HalfCheetahDir、HopperParam、WalkerParam；任务可能变化 reward function 或 dynamics parameters。作者人为构造强 context shift：训练时任务与主要行为策略强相关，测试时 context 来自与真实任务差异大的策略。

基线：UNICORN、CSRO、FOCAL、Recon、DPT，以及 Oracle。所有方法使用 IQL 作为基础 offline RL 算法。

主结果：Figure 4 显示 BATI 在所有环境中最终 episodic return 最优或明显更稳。AntDir 和 HalfCheetahDir 中若干基线随训练反而下降，说明它们逐渐学到伪相关。CSRO 在部分环境接近 BATI，但收敛慢且不稳定，在更难任务中落后明显。

消融：Recon 与 BATI 使用同样 reconstruction loss，但 Recon 仍用 encoder 推断 latent。Table 4/Figure 4 说明 Recon 在所有环境都低于 BATI，支持“不是重构目标本身，而是 encoder 后验推断方式导致问题”的判断。

噪声分析：Figure 5 和附录 Figure 7 显示 dynamics noise 增加会显著削弱基线，因为真实动力学线索变弱后，encoder 更依赖行为伪相关；BATI 更稳。context length 实验表明 BATI 随 context 增长受益，而基线没有明显改善。

## 8. 展望

研究启发：

1. offline ICRL 的 task inference 应区分行为线索和环境动力学线索。
2. amortized encoder 快，但在分布偏移下容易把训练相关性当因果任务信息。
3. test-time latent optimization 虽慢，但可能更鲁棒。
4. IQL 可作为 offline ICRL 的低层优化器，但上层 latent 推断决定泛化质量。

局限与开放问题：

1. 主要实验是 state-based MuJoCo，尚未扩展到视觉 embodied 场景。
2. 测试时 latent 搜索需要额外计算，实时控制中可能是负担。
3. 假设 $p(X^t|X^b,M)$ 足以区分任务；若任务差异主要体现在偏好或长期目标，短 context 可能不足。
4. embedding table 对训练任务离散化友好，对连续开放任务分布还需改造。

后续想法：

1. 用贝叶斯优化或粒子滤波加速 test-time latent search。
2. 将 BATI 与序列模型 policy 结合，让 latent 同时解释短期动力学和长期 reward。
3. 在视觉机器人离线数据中测试行为伪相关，例如不同 demonstrator 与任务的耦合。

## Links

- Paper page: [PMLR](https://proceedings.mlr.press/v267/ma25x.html)
- PDF: [PMLR PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/ma25x/ma25x.pdf)
- Project page: [BATI ICRL](https://sites.google.com/view/bati-icrl)
- OpenReview: [OpenReview via PMLR](https://proceedings.mlr.press/v267/ma25x.html)
- Code: 未能从论文/PMLR 公开来源确认官方代码链接
- Local PDF: `[ICML_2025] Behavior-agnostic Task Inference for Robust Offline In-context Reinforcement Learning.pdf`

</article>
