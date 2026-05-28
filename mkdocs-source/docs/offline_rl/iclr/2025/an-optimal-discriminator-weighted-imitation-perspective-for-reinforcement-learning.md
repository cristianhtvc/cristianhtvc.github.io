---
title: "An Optimal Discriminator Weighted Imitation Perspective for Reinforcement Learning"
description: "IDRL 将离线 RL 中的 Dual-RL 分布比学习解释为最优判别器加权模仿，并通过状态-动作访问分布修正与迭代数据过滤逼近专家分布。"
tags:
  - Offline RL
  - Dual RL
  - Imitation Learning
  - Distribution Ratio
  - Weighted BC
---

# An Optimal Discriminator Weighted Imitation Perspective for Reinforcement Learning

<div class="paper-hero">
<p class="paper-meta">Haoran Xu, Shuozhe Li, Harshit Sikchi, Scott Niekum, Amy Zhang · ICLR 2025 · ICLR / 2025</p>
<div class="tag-row"><span>Offline RL</span><span>Dual RL</span><span>Imitation Learning</span><span>Distribution Ratio</span><span>Weighted BC</span></div>
</div>

<article class="note-body" markdown>

| 字段 | 内容 |
|---|---|
| Title | An Optimal Discriminator Weighted Imitation Perspective for Reinforcement Learning |
| Year | ICLR 2025 |
| Source | ICLR 2025 Poster |
| Authors | Haoran Xu, Shuozhe Li, Harshit Sikchi, Scott Niekum, Amy Zhang |
| Affiliations | 1 University of Texas at Austin, 2 UMass Amherst, 3 Meta AI |
| Tags | Offline RL, Dual-RL/DICE, Discriminator-Weighted Imitation, Weighted Behavior Cloning, Distribution Shift |

**一句话概括**：这篇论文提出 Iterative Dual Reinforcement Learning (IDRL)，先纠正半梯度 Dual-RL 学到 action ratio 而非 state-action visitation ratio 的问题，再用迭代过滤把离线数据逐步推向“最优判别器加权模仿”所需的专家式支持集。

**我对这篇论文的定位**：它不是又一个简单的 offline RL 正则化技巧，而是把 Dual-RL、DICE、weighted BC 和 discriminator-weighted imitation 放到同一个视角里重新解释。它试图回答一个很实际的问题：如果 Dual-RL 理论上直接在访问分布空间里优化，为什么经验上还经常输给 ReBRAC、Diffusion-QL 这类 Primal-RL 或强策略类方法？作者的答案是：现有 Dual-RL 的 ratio 学错了，并且即便学对了也仍被数据分布正则化困住；IDRL 的价值就在于同时修正这两点。

## 1. 第一作者相关信息

**第一作者**：Haoran Xu (徐浩然)。论文 PDF 标注其单位为 University of Texas at Austin；作者个人主页显示其为 UT Austin 三年级 Ph.D. student，导师为 Amy Zhang，研究方向是“scaling foundation models to learn from experience through reinforcement learning”。

**研究轨迹概括**：Haoran Xu 的主线非常清晰：从 offline imitation learning 与 safe/offline RL 起步，逐渐进入 in-sample/offline value regularization、DICE/Dual-RL、diffusion-guided offline RL，再扩展到 online/offline/offline-to-online 统一框架与 foundation-model experience learning。本文 IDRL 可以看作他早期 discriminator-weighted offline imitation 与近年 DICE/IVR 线索的汇合：用“判别器权重”解释“访问分布比”，再用“迭代过滤”提升离线数据中的专家支持。

| year | title | venue/source | topic tag | relation to this paper |
|---|---|---|---|---|
| 2026 | Reinforcement Learning via Value Gradient Flow | ICLR 2026 | RL theory / value gradients | 后续延伸到更一般的价值梯度流视角 |
| 2025 | Uni-RL: Unifying Online and Offline RL via Implicit Value Regularization | NeurIPS 2025 Poster | IVR / online-offline unification | 延续 SQL/IVR，将 reference-policy 约束统一到 online/offline |
| 2025 | An Optimal Discriminator Weighted Imitation Perspective for Reinforcement Learning | ICLR 2025 Poster | Dual-RL / imitation / ratio correction | 本文 |
| 2025 | Learning to Achieve Goals with Belief State Transformers | ICLR 2025 | sequence modeling / goal RL | 展示其 RL 研究向序列模型和 belief state 方向扩展 |
| 2024 | Diffusion-DICE: In-Sample Diffusion Guidance for Offline Reinforcement Learning | NeurIPS 2024 | DICE / diffusion policy | 与本文共享 Dual-RL/DICE 基础，但用 diffusion policy 表达多模态动作 |
| 2024 | ODICE: Revealing the Mystery of Distribution Correction Estimation via Orthogonal-gradient Update | ICLR 2024 Spotlight | DICE / orthogonal gradient | 本文直接讨论并对比的 Dual-RL/DICE 邻近工作 |
| 2023 | Offline RL with No OOD Actions: In-Sample Learning via Implicit Value Regularization | ICLR 2023 notable/top 5% | in-sample offline RL / SQL | 本文的 weighted-BC 与 in-sample value learning 背景之一 |
| 2022 | Discriminator-Weighted Offline Imitation Learning from Suboptimal Demonstrations | ICML 2022 | offline imitation | 本文“判别器加权模仿”动机的前史 |
| 2022 | Constraints Penalized Q-Learning for Safe Offline Reinforcement Learning | AAAI 2022 | safe offline RL | 早期 offline/safe RL 方向 |

**主要研究方向综合**：第一，围绕 offline RL 的分布偏移、OOD action 与数据支持约束；第二，围绕 Dual-RL/DICE/访问分布比的理论-算法桥接；第三，将 imitation learning、diffusion/generative policy 与 RL 统一为“从离线经验中抽取高质量行为”的问题；第四，近年开始把 online、offline、offline-to-online 以及 foundation-model experience learning 放到统一框架中。

**来源**：作者主页 <https://ryanxhr.github.io/>；本文 OpenReview <https://openreview.net/forum?id=9JtG4nN7ql>。

## 2. 研究问题

这篇文章解决的是 **offline RL 中 Dual-RL 方法如何正确、稳定地从固定数据集中提取接近专家访问分布的策略**。更具体地说，作者认为 Dual-RL 的目标本来很诱人：不像 Primal-RL 那样反复对 OOD action 做 Q 评估，而是在访问分布空间中学习优化策略相对数据分布的 ratio，然后用 weighted behavior cloning 抽取显式策略。可是现有 Dual-RL 经验表现并不总是好，尤其在混杂大量低质量轨迹的数据里，weighted BC 容易被错误权重污染。

论文假设的基本设定是标准单智能体 discounted MDP：

$$
M = \langle S, A, P, d_0, r, \gamma \rangle .
$$

离线数据集为

$$
D = \{s_i, a_i, r_i, s'_i\}_{i=1}^N,
$$

没有在线交互，也没有额外专家数据。行为策略记为 $\mu$，经验访问分布记为 $d_D$。目标是从这个固定数据集里学出高回报策略 $\pi$。

核心失败模式有两个。

第一，半梯度 Dual-RL 的固定点改变了。实践中为了稳定训练，Dual-RL 往往用 $Q(s,a)$ 近似 Bellman backup，只对当前状态值做 semi-gradient 更新。作者指出这会导致方法学到的是 action distribution ratio：

$$
w^\star(a|s) = \frac{\pi^\star(a|s)}{\mu(a|s)},
$$

而不是 state-action visitation ratio：

$$
w^\star(s,a) = \frac{d^\star(s,a)}{d_D(s,a)} .
$$

第二，即便访问分布比估对了，Dual-RL 学到的仍然是相对离线数据分布正则化后的最优访问分布，而不是真正专家/最优访问分布 $d_E$。也就是说，数据正则化让它安全，但也限制它突破行为数据的低质量成分。

论文的 formal objective 来自 Dual-RL/DICE 的访问分布正则化：

$$
\pi^\star =
\arg\max_\pi
\mathbb{E}_{(s,a)\sim d_\pi}[r(s,a)]
- \alpha D_f(d_\pi \| d_D).
$$

含义：策略既要高回报，也不能让访问分布过度偏离离线数据分布。本文想做的是：保留这种 in-sample 安全性，同时通过正确 ratio 估计和迭代过滤逐步接近真正高质量支持集。

## 3. 背景知识

**Primal-RL 与 Dual-RL 的差别**：Primal-RL 是更常见的 actor-critic 视角：先估计 $Q^\pi$，再用 $Q$ 改进策略。offline 下的 CQL、TD3+BC、ReBRAC、Diffusion-QL 等大多属于这个大类，区别在于如何约束策略或惩罚价值。Dual-RL 则从另一边看 RL：策略价值也可以写成访问分布上的 reward 期望。
$$
J(\pi) =
(1-\gamma)\mathbb{E}_{s_0\sim d_0,a_0\sim\pi}[Q^\pi(s_0,a_0)]
=
\mathbb{E}_{(s,a)\sim d_\pi}[r(s,a)].
$$

前者是 value-function form，后者是 visitation-distribution form。Dual-RL 的直觉是：如果能直接学“优化策略会访问哪些 state-action”，就可以避免在训练时查询 OOD action 的 Q 值。

**访问分布 $d_\pi$**：它表示策略 $\pi$ 在环境中随时间折扣访问某个 $(s,a)$ 的概率：
$$
d_\pi(s,a) =
(1-\gamma)\sum_{t=0}^{\infty}\gamma^t
\Pr(s_t=s,a_t=a\mid \pi).
$$

在 offline RL 里，$d_D$ 是数据集行为策略诱导的经验访问分布。ratio $d^\star(s,a)/d_D(s,a)$ 就是“这个 transition 相对数据而言有多像优化策略会访问的 transition”。

**Weighted behavior cloning (weighted BC)**：普通 BC 最大化数据动作的 log likelihood。Weighted BC 则给每个 transition 一个权重：
$$
\pi =
\arg\max_\pi
\mathbb{E}_{(s,a)\sim d_D}
\left[
w^\star(s,a)\log \pi(a|s)
\right].
$$

权重大，说明这条 transition 更像高质量策略会使用；权重为 0，则可以过滤掉。本文的核心就是让这个权重尽可能接近“最优判别器”会给出的专家权重。

**Discriminator-weighted imitation 的桥**：如果有一个额外专家数据集 $D_E$，可以训练判别器区分专家分布和离线数据分布，再用
$$
w_d(s,a)=\frac{d_E(s,a)}{d_D(s,a)}
\approx \frac{d(s,a)}{1-d(s,a)}
$$

作为 BC 权重。这就是 Figure 1(b) 中的 Optimal-DWBC oracle。它表现很好，但现实中通常没有额外专家数据。IDRL 的野心是：不用 $D_E$，只靠 Dual-RL 的分布比学习和迭代过滤，逼近这个 oracle 权重。

**为什么 action ratio 不够**：$w(a|s)$ 只告诉你“在这个状态下哪个动作更像优化策略”，却不告诉你“这个状态本身是不是优化策略会到达的状态”。如果某个坏状态里存在一个看起来不错的动作，action ratio 仍可能给它正权重，BC 就会学到坏状态附近的模式。函数逼近会进一步放大这个问题，因为相似状态共享表示，坏状态的动作模式可能泛化到好状态上。

**为什么这篇文章的动机成立**：现有 offline RL 的保守性是必要的，因为完全突破数据支持会导致 OOD 估值错误；但过度保守又会把策略锚定在低质量行为分布。Dual-RL 理论上提供的是 state-action 级别访问分布约束，比 action-level 约束更精细；问题是实际半梯度训练改变了 ratio 的含义，且单次正则化只能得到“数据正则化最优”而非“真实最优”。因此，本文提出“先修正 ratio，再迭代替换数据支持”的路线，是围绕这个理论-实践落差展开的。

## 4. 问题分析

论文的分析路径分三步。

**第一步：用 oracle 实验证明“判别器加权模仿”潜力很大。** Figure 1(b) 构造了一个不可用但有诊断价值的 baseline：用离线数据 $D$ 和额外专家数据 $D_E$ 训练判别器，再用 $w_d(s,a)$ 在 $D$ 上做 weighted BC。结果显示 Optimal-DWBC 在 D4RL Mujoco、Antmaze、Kitchen 等数据集上很强，甚至超过不少 classical offline RL 方法。这说明问题不在 weighted BC 必然弱，而在如何得到正确权重。

**第二步：证明 semi-gradient Dual-RL 学错了 ratio。** Section 3.1 的 Proposition 1 指出，半梯度 Dual-RL 学到的是
$$
w^\star(a|s)=\frac{\pi^\star(a|s)}{\mu(a|s)},
$$

而不是

$$
w^\star(s,a)=\frac{d^\star(s,a)}{d_D(s,a)}.
$$

这个差异很关键：前者只在状态内归一化，无法判断状态本身是否属于专家/最优访问分布；后者同时包含状态选择与动作选择。作者进一步指出，action-ratio 过滤可能造成 fragmented dataset，导致深度 RL backup 缺失并引起过估计。

**第三步：指出单次正确 ratio 仍受正则化限制。** 即使通过修正得到了 $w(s,a)$，它仍是相对 $d_D$ 正则化后的最优访问分布比。若离线数据中大量是低质量 transition，单次 Dual-RL 仍被数据分布牵制。作者因此提出迭代：第 $k$ 次先得到正则化最优支持 $D_{k+1}$，再在这个子数据集上继续做 Dual-RL，使正则化参考分布从 $d_D$ 逐步替换为更接近高质量策略的分布。

Figure 2 的 grid-world toycase 是一个很好的机制图。用 $w(a|s)$ 过滤时，策略仍可能走偏；用修正后的 $w(s,a)$ 且迭代两次后，剩余 transition 聚焦到通往目标的关键路径。这个图不是证明，但直观展示了“状态是否值得到达”比“到达后哪个动作更好”多出来的信息。

## 5. 思想与方法

IDRL 的指导原则可以概括为：**把 offline RL 策略学习转化为“在数据内部识别并模仿最优访问分布”的问题**。这和传统 offline RL 的差别是，它不主要依赖 policy improvement 时对动作空间做优化，而是先在数据支持内学出 transition 级别的“像不像最优策略”的权重，再用 weighted BC 提取策略。

方法包含两个核心组件。

**组件 A：从 action ratio 修正到 state-action visitation ratio。** 半梯度 Dual-RL 先给出 $w^\star(a|s)$。作者把“给定 action ratio 恢复 state visitation ratio”重写为 OPE 问题，并通过 Fenchel-Rockafellar duality 得到可计算目标。Theorem 1 给出：

$$
w^\star(s)
=
\frac{d^\star(s)}{d_D(s)}
=
\max\left(
0,
(f')^{-1}
\left(
\mathbb{E}_{a\sim\mu}
\left[
w^\star(a|s)(T U^\star(s,a)-U^\star(s))
\right]
\right)
\right).
$$

直观解释：状态 $s$ 的权重取决于在行为动作分布下，用 action ratio 加权后的未来残差。如果这个状态通向更优的后续访问，它会得到更高 state weight。

由于非线性函数外面套了期望，直接样本估计会有 bias。Lemma 1 和 Theorem 2 引入辅助函数 $W(s)$，把这个非线性期望变成一个关于 $W$ 的凸优化目标。最后实际联合优化两个目标：

$$
\min_W
\mathbb{E}_{(s,a)\sim d_D}
\left[
f(W(s))
- w^\star(a|s)(T U(s,a)-U(s))W(s)
\right],
$$

$$
\min_U
\mathbb{E}_{(s,a)\sim d_D}[U(s)-T U(s,a)]
+
\mathbb{E}_{(s,a)\sim d_D}
\left[
\max(0,W(s))w^\star(a|s)(T U(s,a)-U(s))
\right].
$$

得到 $W^\star$ 后，修正后的 transition 级别权重为：

$$
w^\star(s,a)
=
w^\star(s)w^\star(a|s)
=
\max\left(
0,
W^\star(s)(f')^{-1}(Q^\star(s,a)-V^\star(s))
\right).
$$

**组件 B：迭代自蒸馏 / 数据支持过滤。** 单次 Dual-RL 学到的是相对当前数据分布正则化的最优访问分布。IDRL 利用 sparse weight 的性质：每轮保留 $w_k^\star(s,a)>0$ 的 transition，得到子数据集 $D_{k+1}$，再在 $D_{k+1}$ 上重新做 Dual-RL。这样下一轮正则化参考分布不再是原始混杂数据，而是上一轮筛出来的更高质量分布。

**新意判断**：ratio correction 的 OPE 推导是本文最技术性的部分；迭代过滤的想法本身和 self-training / curriculum / data filtering 有相似精神，但本文把它严格接到 Dual-RL 的访问分布比上，并给出 monotonic improvement 形式的分析。我的判断是，本文真正有价值的地方不只是 SOTA 表格，而是把 “DICE ratio 学习为什么经验上不够好” 这件事拆出了两个可诊断、可消融的机制。

## 6. 算法与伪代码

算法名：**Iterative Dual-RL (IDRL)**。论文 Algorithm 1 给出完整流程。按中文重写如下。

1. 初始化四个价值相关网络 $Q_{\phi_1}, V_{\phi_2}, U_{\psi_1}, W_{\psi_2}$ 和策略网络 $\pi_\theta$；设定超参数 $\alpha$ 或实践中的 $\lambda$；令初始数据集 $D_1=D$。
2. 对第 $k=1,\dots,M$ 轮迭代：
3. 从当前数据集 $D_k$ 采样 $(s,a,r,s')$。
4. 用 semi-gradient Dual-RL 目标更新 $Q$ 和 $V$：

$$
\min_V
\mathbb{E}_{(s,a)\sim d_D}
\left[
V(s)+\alpha f_p^\star((Q(s,a)-V(s))/\alpha)
\right],
$$

$$
\min_Q
\mathbb{E}_{(s,a,s')\sim d_D}
\left[
r(s,a)+\gamma V(s')-Q(s,a)
\right]^2.
$$

这一步得到 action ratio：

$$
w_k(a|s)=
\max\left(0,(f')^{-1}((Q(s,a)-V(s))/\alpha)\right).
$$

5. 固定/使用该 action ratio，从 $D_k$ 采样 $(s,a,s')$，更新 $U$ 和 $W$，求 state visitation correction。
6. 组合得到 corrected state-action ratio $w_k(s,a)$。
7. 过滤数据集：

$$
D_{k+1}
=
\{(s,a,r,s')\in D_k \mid w_k(s,a)>0\}.
$$

8. 重复直到第 $M$ 轮。
9. 在最后数据集 $D_M$ 上，用最后一轮权重 $w_M(s,a)$ 做 weighted BC，训练显式策略：

$$
\pi_\theta =
\arg\max_{\pi_\theta}
\mathbb{E}_{(s,a)\sim D_M}
\left[
w_M(s,a)\log \pi_\theta(a|s)
\right].
$$

实践细节来自 Appendix C：作者使用 Pearson $\chi^2$ divergence：

$$
f(x)=(x-1)^2,\quad
f^\star(y)=y(y/4+1),\quad
(f')^{-1}(R)=R/2+1.
$$

他们还把 $\alpha$ 改写为更容易调的 $\lambda\in(0,1)$：

$$
\min_V
\mathbb{E}
\left[
(1-\lambda)V(s)+\lambda f_p^\star(Q(s,a)-V(s))
\right].
$$

网络与训练：toycase、D4RL 主实验均使用 3-layer MLP、256 hidden units、Adam、learning rate $10^{-4}$。D4RL 每个任务做 2 次 IDRL iteration，每轮 $10^6$ steps，前 500k steps 学 action ratio，后 500k steps 学 state-action ratio。Mujoco 每 $5\times 10^4$ steps 做 10 次 evaluation，Antmaze/Kitchen 做 50 次 evaluation，均 7 seeds。Q-function 使用 target network，soft update weight $5\times 10^{-3}$。

理论结果：Theorem 3 把第 $k$ 轮的 weighted BC 看作在 $D_{k+1}$ 这个“更专家”的数据上模仿，并给出 bound：

$$
V(\pi)=
V(D_{k+1})
-
O\left(
\frac{|S|H^2}
{N_{D_{k+1}}+N_{D_k-D_{k+1}}/\max_s w^\star_{k+1}(s)}
\right).
$$

含义：过滤后数据越接近专家，$V(D_{k+1})$ 越高；但过滤太狠会减少样本数。因此 IDRL 的过滤存在“质量 vs 数量”权衡。

Theorem 4 给出单调性：

$$
V(D_{k+1})\ge V(D_k).
$$

含义：在作者假设下，每轮用正则化最优访问分布筛数据，得到的数据集期望 reward 不低于上一轮。这支撑了“迭代过滤比单纯调小正则化系数更有原则”的说法。

## 7. 实验与消融

**实验 1：D4RL benchmark。** 数据集包括 Mujoco locomotion、Antmaze navigation、Kitchen。评价指标是 D4RL normalized score，7 random seeds，报告训练末尾平均分与标准差。基线覆盖 Primal-RL 与 Dual-RL 两类：TD3+BC、CQL、ReBRAC、Diffusion-QL、X%-BC、IQL、SQL、O-DICE。X%-BC 是按 trajectory return 选 top X% 后做 BC，用来证明 transition-wise selection 比 trajectory-wise filtering 更必要。

Table 1 的主要结果：IDRL 在大多数任务上达到或接近 top score。比较突出的例子包括 halfcheetah-medium-replay 从 ReBRAC 51.0 / O-DICE 44.0 到 IDRL 58.0；antmaze-medium-play 从 ReBRAC 84.0 / O-DICE 86.0 到 IDRL 93.0；walker2d-medium-expert IDRL 113.9，高于其他方法约 110-112 区间。Kitchen 上 IDRL 也总体强，但 kitchen-complete 中 Diffusion-QL 84.0 高于 IDRL 80.5，说明 IDRL 并非每个任务都绝对第一。

**实验解释**：Mujoco 中很多数据集已有较强行为策略，强保守方法也可做得不错，因此 IDRL 通常只需 1-2 轮。Antmaze/Kitchen 需要 trajectory stitching，transition 级别识别高质量片段更重要，IDRL 的访问分布过滤优势更明显。

**消融实验。** Table 2 比较三个版本：

| Ablation | Mujoco | Antmaze | Kitchen |
|---|---:|---:|---:|
| IDRL | 90.8 | 80.3 | 74.8 |
| IDRL w/ M=1 | 73.2 | 68.8 | 70.5 |
| IDRL w/ $w^\star(a|s)$ | 56.8 | 51.3 | 67.8 |

结论很直接：只做一轮会明显掉分，说明迭代过滤有用；用 action ratio 代替 state-action ratio 掉得更厉害，说明作者提出的 ratio correction 不是可有可无的数学装饰，而是实证上最关键的组件之一。

**实验 2：corrupted demonstrations。** 作者构造更现实的混杂数据：在 D4RL Mujoco 中用 1%、5%、10% expert transitions 与大量 random transitions 混合，总 transition 数 1,000,000。比较 naive BC、Advantage-Weighted (AW)、Density-Weighted initialized with AW (DW+AW) 和 IDRL。Figure 3 显示 IDRL 在这些低专家比例数据上更稳，能逐步滤掉随机 transition 并组合稀有高质量片段。

**实现细节对解释的影响**：主实验基线结果多来自原论文而非统一重新跑，这在 offline RL 文献中常见，但会引入实现/调参差异。IDRL 本身只有一个主要超参数 $\lambda$，Table 3 给出不同任务值，大致在 0.4 到 0.8。作者说 $\lambda$ 越大越能搜索更优动作，但若 $V$ 发散则不能过大；这意味着方法虽然比 O-DICE 少一个 $\eta$，仍有稳定性调参问题。

**实验能证明什么**：实验强力支持两个机制：state-action ratio 优于 action ratio，迭代过滤优于单轮 Dual-RL。实验也说明 IDRL 在混杂低质量数据上更有优势。实验没有完全证明的是：IDRL 是否在小数据、视觉观测、离散大动作空间、多任务或真实机器人数据上仍然稳定；论文自己也承认 small data 下可能泛化不足。

## 8. 展望

**对 ORL 研究者有用的启发**：

1. Weighted BC 不是 offline RL 的弱后处理；关键是权重是否真能反映 state-action visitation quality。
2. Action-level support constraint 和 state-action visitation constraint 的差别会在混杂数据中被放大。
3. 数据过滤不只是 heuristic；如果过滤规则来自正则化访问分布优化，就可以和性能 bound 连接起来。
4. DICE/Dual-RL 的理论美感要落地，必须严肃处理半梯度、单样本 Bellman backup 和函数逼近导致的固定点变化。
5. Offline RL 和 offline imitation learning 的界限可以进一步模糊：从固定数据里找“专家式支持集”本身就是一个模仿问题。

**局限与开放问题**：

1. 多轮 IDRL 增加训练时间，尤其每轮都分 action-ratio 和 state-ratio 两阶段训练。
2. 当数据量小或覆盖极窄时，过滤可能导致样本太少，作者也承认可能有 generalization issue。
3. 过滤阈值采用 $w(s,a)>0$，在深度函数逼近和噪声 reward 下是否过硬，值得进一步研究。
4. 论文主要在 D4RL 和 synthetic corrupted demonstrations 上验证，缺乏更广泛的实验验证。
5. 方法最终策略仍是 unimodal Gaussian BC；面对强多模态动作分布，可能需要和 diffusion/generative policy 结合。

**三个后续研究想法**：

1. **Uncertainty-aware IDRL filtering**：在 $w(s,a)>0$ 外加入 ratio uncertainty 或 ensemble confidence，避免小数据下误删关键 transition。
2. **Offline-to-online IDRL**：把每轮过滤后的分布作为 online fine-tuning 的 reference policy / replay prioritization，测试是否能比静态 offline 版本更快适应。

未来连接上，IDRL 很自然地接到三个方向：offline RL 中的 data-centric filtering，DICE/IVR 的 online-offline 统一，以及 RLHF/RLVR 中“从固定样本里按优势或判别器权重重加权学习”的算法范式。尤其在大模型对齐里，policy update 也常被看作某种加权模仿；本文的 state-action visitation 视角可能为长期 credit assignment 和数据支持约束提供类比。

## Links

- Paper page / OpenReview: <https://openreview.net/forum?id=9JtG4nN7ql>
- arXiv: <https://arxiv.org/abs/2504.13368>
- Project page: <https://ryanxhr.github.io/IDRL/>
- Code: 项目页有 `Code` 链接，但截至 2026-05-27 该链接指向 `maoliyuan/diffusion-DICE-Pytorch`，README 明确是 Diffusion-DICE 的官方实现，未能从论文/公开来源确认 IDRL 的官方代码链接。
- First author homepage: <https://ryanxhr.github.io/>
- First author OpenReview profile: <https://openreview.net/profile?id=~Haoran_Xu4>
- DBLP author entry: <https://dblp1.uni-trier.de/pid/140/8357-3.html>

</article>
