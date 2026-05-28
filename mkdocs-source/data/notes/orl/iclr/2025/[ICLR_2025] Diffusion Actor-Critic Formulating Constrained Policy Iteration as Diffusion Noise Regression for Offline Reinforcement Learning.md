---
title: "Diffusion Actor-Critic: Formulating Constrained Policy Iteration as Diffusion Noise Regression for Offline Reinforcement Learning"
description: "DAC 将 KL 约束策略迭代转化为扩散噪声回归，用 soft Q-guidance 训练扩散目标策略并避免 OOD 动作。"
tags:
  - Offline RL
  - Diffusion Policy
  - Actor-Critic
  - Policy Regularization
  - Q-Ensemble
---

# Diffusion Actor-Critic: Formulating Constrained Policy Iteration as Diffusion Noise Regression for Offline Reinforcement Learning

| 字段 | 内容 |
|---|---|
| Title | Diffusion Actor-Critic: Formulating Constrained Policy Iteration as Diffusion Noise Regression for Offline Reinforcement Learning |
| Year | arXiv 2024；ICLR 2025 Conference Paper |
| Source | ICLR 2025 / arXiv 2405.20555 |
| Authors | Linjiajie Fang, Ruoxue Liu, Jing Zhang, Wenjia Wang, Bing-Yi Jing |
| Affiliations | Hong Kong University of Science and Technology; HKUST (Guangzhou); Southern University of Science and Technology |
| Tags | Offline RL; diffusion policy; constrained policy iteration; soft Q-guidance; LCB critic |

**一句话概括：** DAC 证明 KL 约束的 offline policy improvement 可以写成扩散噪声回归目标，从而直接训练一个扩散目标策略，而不是先训练行为采样器再从候选动作中筛选。

**我对这篇论文的定位：** DAC 是 diffusion policy for offline RL 中非常重要的 policy-regularized actor-critic 工作。它与 SfBC/IDQL 的区别是扩散模型不是行为策略 sampler，而是目标策略本身；与 Diffusion Q-learning 的区别是 Q guidance 作用在噪声回归目标中，并随扩散噪声尺度软化，避免对完整去噪链反向传播。

## 1. 第一作者相关信息

第一作者 **Linjiajie Fang**。论文首页显示其来自 HKUST；DBLP 记录可核验到其 2024-2025 年围绕 diffusion/offline RL 和相关机器学习方向的论文条目。本次检索未确认其个人主页，因此不补充未核实的履历。

| year | title | venue/source | topic tag | relation to this paper |
|---|---|---|---|---|
| 2025 | Diffusion Actor-Critic: Formulating Constrained Policy Iteration as Diffusion Noise Regression for Offline Reinforcement Learning | ICLR 2025 / arXiv | Offline RL, diffusion policy | 本文 |
| 2024 | Diffusion Actor-Critic: Formulating Constrained Policy Iteration as Diffusion Noise Regression for Offline Reinforcement Learning | CoRR abs/2405.20555 | Offline RL | 本文 arXiv 版本 |

从公开记录看，第一作者处在较早阶段，合作网络包括 Wenjia Wang 和 Bing-Yi Jing。本文反映的研究主线是把生成模型的 score/noise regression 形式和 offline RL 的策略正则化理论对齐。

## 2. 研究问题

offline RL 中，目标策略若选择离线数据外动作，会让 Q 函数在未覆盖区域外推并高估。policy-regularized methods 用“目标策略不要离行为策略太远”来控制这个风险。问题在于：如果行为策略由扩散模型表示，没有显式密度，如何定义并优化一个理论上合理的 KL 约束策略改进？

论文从 KL 约束策略迭代出发：

$$
\pi_{k+1}
=\arg\max_{\pi}\mathbb{E}_{s\sim D,a\sim\pi(\cdot|s)}[Q^{\pi_k}(s,a)]
\quad
\mathrm{s.t.}\quad
\mathbb{E}_{s\sim D}[D_{\mathrm{KL}}(\pi(\cdot|s)\|\pi_\beta(\cdot|s))]\le \epsilon_b .
$$

闭式解是：

$$
\pi_{k+1}^{\star}(a|s)
=\frac{1}{Z(s)}\pi_\beta(a|s)\exp\left(\frac{1}{\eta}Q^{\pi_k}(s,a)\right).
$$

难点是 $Z(s)$ 和 $\pi_\beta(a|s)$ 的密度都不可直接得到，扩散模型也不方便计算 $\log\pi_\theta(a|s)$。DAC 的核心贡献就是把这个密度问题转成噪声预测问题。

## 3. 背景知识

| term | plain Chinese meaning | role in this paper |
|---|---|---|
| Policy regularization | 限制目标策略不要远离行为策略 | 避免 OOD 动作 |
| Reverse KL constraint | $D_{\mathrm{KL}}(\pi\|\pi_\beta)$ 形式的策略约束 | DAC 推导起点 |
| Diffusion policy | 条件扩散模型生成动作 | DAC 的 actor |
| Noise regression | 训练网络预测加到动作上的高斯噪声 | DAC 的策略学习形式 |
| Soft Q-guidance | 按噪声尺度缩放的 Q 梯度引导 | DAC 的核心 actor 更新 |
| LCB Q-ensemble | 用均值减方差/标准差项构造悲观 target | 稳定 critic 和 Q-gradient |

扩散模型把干净动作 $a$ 加噪为：

$$
x_t=\sqrt{\bar{\alpha}_t}a+\sqrt{1-\bar{\alpha}_t}\epsilon,\quad \epsilon\sim\mathcal{N}(0,I).
$$

普通行为克隆式 diffusion policy 学的是 $\epsilon_\theta(x_t,s,t)\approx\epsilon$。DAC 则要学一个带 Q guidance 的目标噪声，使去噪方向既贴近行为支持，又朝高 Q 区域移动。

为什么动机成立：现有 SfBC/IDQL 用 diffusion 生成很多候选动作再筛选，推理慢；DQL 直接把 Q 目标加到 denoised action 上，需要穿过去噪链反传，且可能把动作推到行为支持外。DAC 的 soft Q-guidance 在早期噪声大时允许探索，在靠近最终动作时自动减弱，像一个“越接近数据流形越谨慎”的引导。

## 4. 问题分析

论文先指出三类 diffusion offline RL 局限：

- **行为采样器路线**：SfBC/IDQL 需要生成大量动作候选，再用 Q 选择；表达力强但推理慢。
- **denoised Q-guidance 路线**：Diffusion Q-learning 对最终 denoised action 做 Q maximization，梯度穿过完整去噪路径，训练慢且可能 OOD。
- **显式密度路线不可用**：AWAC/AWR 类方法需要 $\log\pi_\theta(a|s)$ 或 density ratio，扩散模型不方便给出密度。

Figure 1/2 的 2D bandit 例子显示，hard guidance 或 denoised guidance 可能把动作推到行为点云之外；soft guidance 能在行为支持内找到高回报、多模态动作。

理论上，DAC 把闭式最优策略的 score 写成：

$$
\nabla_a\log \pi_{k+1}^{\star}(a|s)
=\nabla_a\log \pi_\beta(a|s)+\frac{1}{\eta}\nabla_a Q^{\pi_k}(s,a).
$$

然后把它扩展到噪声空间 $x_t$。Theorem 1 说明目标 noisy score 可定义一个生成目标策略的 diffusion model；Theorem 2 进一步把训练该目标 diffusion model 转成只需要离线数据采样的 noise regression objective。

## 5. 思想与方法

DAC 的 actor 目标来自 Theorem 2：

$$
\arg\min_\theta
\mathbb{E}_{(s,a)\sim D,\epsilon,t}
\left[
\left\|
\epsilon_\theta(x_t,s,t)-\epsilon
+\frac{1}{\eta}\sqrt{1-\bar{\alpha}_t}\nabla_{x_t}Q^{\pi_k}(s,x_t)
\right\|^2
\right].
$$

这里 $\sqrt{1-\bar{\alpha}_t}$ 是 soft Q-guidance 的关键：当 $t$ 大、噪声多时，Q 梯度更能影响去噪方向；当 $t\to 0$、动作接近最终输出时，该系数趋近 0，策略更像行为克隆，减少 OOD。

等价 actor loss 可写成：

$$
L_A(\theta)=
\mathbb{E}\left[
\eta\|\epsilon_\theta(x_t,s,t)-\epsilon\|^2
+\sqrt{1-\bar{\alpha}_t}\,
\epsilon_\theta(x_t,s,t)\cdot\nabla_{x_t}Q^{\pi_k}(s,x_t)
\right].
$$

其中 $\eta$ 控制 behavior cloning 与 policy improvement 的权衡；实现中可固定，也可用 dual gradient ascent 让噪声预测误差满足阈值 $b$。

critic 方面，DAC 训练 $H$ 个 Q 网络，并用 lower confidence bound target：

$$
Q_{\mathrm{LCB}}(s',a')
=\mathbb{E}_h[Q_{\bar{\phi}_h}(s',a')]
-\rho\sqrt{\mathrm{Var}_h[Q_{\bar{\phi}_h}(s',a')]}.
$$

然后 TD loss 为：

$$
L_C(\phi_h)=
\mathbb{E}_{D,a'\sim\pi_\theta}
\left[
r+\gamma Q_{\mathrm{LCB}}(s',a')-Q_{\phi_h}(s,a)
\right]^2.
$$

Q-gradient 用 ensemble target Q 的平均梯度估计，并按 $C=\mathbb{E}_{D}|Q(s,a)|$ 做尺度归一化。

## 6. 算法与伪代码

算法名：**Diffusion Actor-Critic (DAC)**。

训练流程：

1. 初始化 diffusion policy $\epsilon_\theta$、target diffusion policy $\epsilon_{\bar{\theta}}$、$H$ 个 critic $Q_{\phi_h}$ 与 target critic、Lagrange multiplier $\eta$。
2. 从离线数据集 $D$ 采样 batch $(s,a,r,s')$。
3. 用 target diffusion policy 在 $s'$ 处采样下一个动作 $a'$。
4. 用 $Q_{\mathrm{LCB}}(s',a')$ 构造 critic target，分别更新 $H$ 个 Q 网络。
5. 对数据动作 $a$ 采样噪声 $\epsilon$ 和扩散步 $t$，得到 $x_t$。
6. 用 target Q ensemble 估计 $\nabla_{x_t}Q(s,x_t)$。
7. 用 soft Q-guidance actor loss 更新 diffusion policy。
8. 若启用可学习 $\eta$，根据 $\|\epsilon_\theta-\epsilon\|^2-b$ 做 dual ascent。
9. 用 EMA 更新 actor 和 critic target networks。

策略执行：

1. 给定状态 $s$，从 diffusion policy 采样 $N_a$ 个动作。
2. 用 Q ensemble mean 选择价值最高的动作执行。

关键实现细节：论文默认 diffusion steps $T=5$，Q ensemble size $H=10$，batch size 256；训练 2M gradient steps；每个环境 8 个独立训练过程，最后 50k steps 平均评估；硬件为 2 张 RTX 4090。

## 7. 实验与消融

主实验在 D4RL locomotion 和 AntMaze 上进行；appendix 包含 Kitchen 和 Adroit。基线包括 Onestep-RL、CQL、IQL、IVR、EQL、Diffuser、DTQL、AlignIQL、SfBC、DQL、IDQL-A 等。

主要结果：

- Table 1：DAC 在 locomotion 9 个任务总分 836.4，高于 DQL 791.2、DTQL 798.3 等强基线；在 medium 数据集上提升尤其明显，例如 hopper-medium 101.2、walker2d-medium 96.8。
- AntMaze：DAC 在 umaze 上接近满分 99.5，在 medium-play/diverse 也有竞争力；large 任务不如 IDQL-A，作者认为可能与没有做 reward shifting 有关。
- Figure 3 / Table 2：soft Q-guidance 在大多数 locomotion 任务上优于 hard guidance 和 denoised guidance；denoised guidance 常不稳定，hard guidance 在 suboptimal medium 数据上落后。
- Figure 4：Q ensemble size 至少 5 后结果较稳，size=2 某些任务下降；论文采用 10 平衡稳定性与成本。
- Appendix Table 6：LCB target 明显优于 ensemble minimum，尤其 halfcheetah-medium-expert 中 min target 仅 43.2，而 LCB 为 99.1。
- Appendix Table 8：DAC 的单步 batch update 时间显著低于 denoised Q-guidance；当 diffusion step $T=100$ 时，DQL 约比 DAC 慢 18 倍。

实验说明：DAC 的优势来自三件事的组合：扩散策略表达多模态目标策略、soft Q-guidance 保持行为支持、LCB critic 稳定 Q-gradient。弱点是仍需调 $\eta$、$b$、$\rho$，且 AntMaze large 稀疏奖励场景并未全面领先。

## 8. 展望

对 ORL 研究者的启发：

- 扩散模型可以直接表示 target policy，而不只是 behavior sampler。
- offline RL 中的策略约束可以通过 score/noise regression 实现，不必显式估计密度。
- Q-gradient guidance 的尺度设计很重要；越靠近最终动作越应该谨慎。
- LCB 比简单 ensemble minimum 更适合为扩散 actor 提供平滑梯度。

局限与开放问题：

- 仍依赖 Q 函数梯度质量；若 critic 外推差，actor guidance 仍可能被误导。
- AntMaze large 等稀疏长程任务中没有全面超过 IDQL-A。
- 每次执行仍需要 diffusion sampling 和多个 action candidate，虽比 sampler 方法少，但不是纯 one-step policy。
- 理论推导依赖 surrogate objective 和噪声空间平滑扩展，实际深度实现仍有近似。

可能后续方向：

1. **自适应 soft guidance**：让 guidance scale 随 critic uncertainty、数据密度或扩散步动态变化。
2. **DAC + trajectory planner**：把 action-level DAC 扩展到 trajectory-level diffusion，提升 AntMaze large 的 stitching 能力。
3. **preference-DAC**：用偏好学习得到的 reward/Q 替代环境奖励，服务 offline PbRL/RLHF 场景。

## Links

- OpenReview: https://openreview.net/forum?id=ldVkAO09Km
- arXiv: https://arxiv.org/abs/2405.20555
- Code: https://github.com/Fang-Lin93/DAC
- DBLP first author: https://dblp.org/pid/372/6474
