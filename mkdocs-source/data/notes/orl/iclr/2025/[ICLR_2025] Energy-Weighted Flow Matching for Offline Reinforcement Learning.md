---
title: "Energy-Weighted Flow Matching for Offline Reinforcement Learning"
description: "一篇将精确能量引导引入 flow matching 和 diffusion，并用于 Q 加权离线策略优化的论文。"
tags:
  - Offline RL
  - Flow Matching
  - Diffusion Models
  - Energy Guidance
  - QIPO
---

# Energy-Weighted Flow Matching for Offline Reinforcement Learning

| Field | 内容 |
|---|---|
| Title | Energy-Weighted Flow Matching for Offline Reinforcement Learning |
| Year | 2025 |
| Source | ICLR 2025 Poster |
| Authors | Shiyuan Zhang, Weitong Zhang, Quanquan Gu |
| Affiliations | Tsinghua University；UNC-Chapel Hill；University of California, Los Angeles |
| Tags | Energy-guided generation, flow matching, diffusion policy, offline RL, KL-regularized policy optimization |

**一句话概括**：本文提出 Energy-Weighted Flow Matching / Diffusion，用原始数据点的能量权重直接训练能量引导的生成模型，无需中间能量网络或额外 guidance 模型，并将其用于离线 RL 中的 Q-weighted Iterative Policy Optimization。

**我对这篇论文的定位**：这篇论文位于“生成式策略建模 + 离线 RL”的交叉处。与 QGPO、Guided Flows 等方法相比，它的核心不是再设计一个新的 critic，而是从生成模型理论上回答：如何精确采样 $\mu(a|s)\exp(\beta Q(s,a))$ 这种 Q 加权策略分布。其 ORL 意义在于把 KL 正则化策略改进转化为能量引导生成建模，并用 flow matching/diffusion 直接实现。

## 1. 第一作者相关信息

**Shiyuan Zhang** 是清华大学相关研究者，论文署名邮箱为 `shiyuan-21@mails.tsinghua.edu.cn`。本文合作者 Weitong Zhang 来自 UNC-Chapel Hill，Quanquan Gu 来自 UCLA。Quanquan Gu 组长期关注机器学习理论、优化、强化学习和生成模型，因此本文的理论证明和离线 RL 应用结合比较自然。

第一作者在这篇论文中体现出的研究轨迹是：从生成模型的引导问题出发，给出可证明的目标函数，再把它落到离线 RL 的策略优化。由于 Shiyuan Zhang 的公开个人主页/DBLP 覆盖有限，近期论文表只能保守列出本论文和与合作者路线直接相关的公开工作。

| Year | Title | Venue/Source | Topic tag | Relation to this paper |
|---|---|---|---|---|
| 2023 | Contrastive Energy Prediction / QGPO | NeurIPS / cited work | Energy-guided diffusion for ORL | 本文重点改进对象，需要辅助中间能量模型 |
| 2023 | Guided Flows for Offline RL | cited work | Flow matching in ORL | 本文在 flow matching 中加入精确 energy guidance |
| 2025 | Energy-Weighted Flow Matching for Offline RL | ICLR | Energy guidance + ORL | 本文主贡献 |

**研究方向综合**：本文更像一篇生成模型方法论文，而不是传统 offline RL 算法论文。它关心的问题是如何直接学习能量引导后的速度场/score，再把 Q 函数作为能量用于策略改进。

**信息来源**：论文首页署名、OpenReview 页面、论文引用和附录说明；第一作者完整发表记录未能从本地材料中全面确认。

## 2. 研究问题

本文解决两个层次的问题。

第一层是生成模型问题：给定数据分布 $p(x)$ 和能量函数 $E(x)$，希望从引导分布采样：

$$
q(x) \propto p(x)\exp(-\beta E(x)).
$$

这里 $E(x)$ 越低，样本越应该被偏好；$\beta$ 控制引导强度。已有方法通常需要辅助模型估计中间能量 $E_t(x)$ 或其梯度 $\nabla_x E_t(x)$，引入额外误差和训练成本。

第二层是离线 RL 问题：离线策略优化的 KL 正则形式常有闭式解：

$$
\pi(a|s) \propto \mu(a|s)\exp(\beta Q(s,a)),
$$

其中 $\mu$ 是行为策略或数据分布，$Q(s,a)$ 是离线 RL 学到的价值函数。这个形式正好是能量引导分布，只需令能量 $E(s,a)=-Q(s,a)$。因此问题变成：能否用 diffusion 或 flow matching 直接学习这个 Q 加权策略分布，而不训练额外 guidance 模型？

本文设置：

- 单智能体离线 RL，固定数据集 $\mathcal{D}=\{(s,a,s',r)\}$。
- 先学习行为策略生成模型，再学习或使用离线 Q 函数。
- 策略类是 diffusion / flow matching 生成式策略。
- 无在线交互，主要在 D4RL 上评估。

## 3. 背景知识

**能量引导生成模型**：普通生成模型学习 $p(x)$；能量引导生成模型希望保留数据分布支持，同时偏向低能量或高价值样本：

$$
q(x)\propto p(x)\exp(-\beta E(x)).
$$

在图像生成中，$E$ 可以来自分类器或偏好模型；在离线 RL 中，$E=-Q$，高 Q 动作得到更大权重。

**Diffusion models**：扩散模型把数据 $x_0$ 逐步加噪得到 $x_t$，再学习反向 score $\nabla_x\log p_t(x)$ 或噪声预测函数，从噪声恢复样本。若要引导生成，需要修改反向 score。

**Flow matching**：flow matching 学习速度场 $v_t(x)$，使样本沿 ODE 从噪声分布流向数据分布。它比 diffusion 更一般，optimal transport flow、rectified flow、Gaussian path 都可放在此框架下。

**Conditional Flow Matching**：直接学习 marginal velocity $u_t(x)$ 通常不可行，但可以学习条件速度场 $u_t(x|x_0)$。Lipman et al. 的结论是，条件目标与边缘目标在梯度上等价。

**为什么 auxiliary guidance 麻烦**：QGPO/CEP 等方法需要估计中间能量 $E_t(x)$ 或 force field。中间量依赖噪声时间 $t$ 和当前噪声样本，训练额外网络会带来偏差、复杂度和算力开销。

| Term | 朴素含义 | 在本文中的作用 |
|---|---|---|
| Energy $E(x)$ | 样本“不好”的程度 | 在 RL 中对应 $-Q(s,a)$ |
| Guidance scale $\beta$ | 引导强度 | 越大越偏向高 Q 动作 |
| Velocity field $u_t$ | 样本在 flow 中移动方向 | flow matching 要学习的对象 |
| Score $\nabla\log p_t$ | 扩散反向去噪方向 | diffusion 版本要学习的对象 |
| EFM | Energy-Weighted Flow Matching | 直接学习能量引导速度场 |
| QIPO | Q-weighted Iterative Policy Optimization | 本文用于离线 RL 的策略改进算法 |

**为什么这篇文章的动机成立**：离线 RL 的生成式策略常被要求既“像数据”又“偏向高回报”。KL 正则化闭式解恰好给出这种分布形式。若每次都额外训练 guidance 网络，方法会复杂且误差叠加；若能把能量权重直接放进 flow/diffusion 训练目标，就能更干净地实现 Q 加权策略。

## 4. 问题分析

论文的诊断围绕两个问题：

1. 现有 diffusion 能量引导方法虽然可以精确，但需要辅助中间能量模型。
2. flow matching 中如何精确加入 energy guidance 并不直接，因为它学习的是速度场而不是 score。

Theorem 4.1 给出能量引导速度场：

$$
\hat{u}_t(x)=
\int
p_{0t}(x_0|x)\,u_t(x|x_0)
\frac{\exp(-\beta E(x_0))}{\exp(-E_t(x))}
dx_0.
$$

直觉：在当前噪声样本 $x$ 可能对应的所有原始数据点 $x_0$ 中，低能量的 $x_0$ 权重大，因此速度场更倾向把 $x$ 带向低能量样本。

但直接用这个公式训练仍有两个障碍：

- 需要对所有可能的 $x_0$ 积分。
- 需要估计中间能量函数 $E_t(x)$。

Theorem 4.3 是核心诊断结果：可以用条件能量加权 flow matching loss 代替直接匹配 $\hat{u}_t$，而且梯度等价：

$$
\mathcal{L}_{CEFM}(\theta)=
\mathbb{E}_{t,x_0,x}
\left[
\frac{\exp(-\beta E(x_0))}
{\mathbb{E}_{\tilde{x}_0\sim p_0}[\exp(-\beta E(\tilde{x}_0))]}
\left\|v_\theta^t(x)-u_t(x|x_0)\right\|^2
\right].
$$

这一步揭示了论文的关键：不用显式学习 $E_t$，只需在训练样本上按 $\exp(-\beta E(x_0))$ 加权。

Corollary 4.8 将这一思想推广到 diffusion loss，得到 energy-weighted diffusion。Section 4.3 进一步比较 classifier guidance / classifier-free guidance：当 $\beta\ne 1$ 时，普通 CFG 对应的目标一般不是精确的 $p(x)p(c|x)^\beta$，而 energy-weighted diffusion 仍能精确匹配目标分布。Figure 1 用二维例子显示 $\beta>1$ 时 energy-weighted diffusion 更接近真分布。

## 5. 思想与方法

本文的指导原则是：**不要先估计中间能量再组合 guidance，而是直接把原始样本的能量变成训练权重，让模型自己学到引导后的 flow/score**。

方法分三层。

**第一层：Energy-Weighted Flow Matching**  
把普通 CFM 损失改为按 $\exp(-\beta E(x_0))$ 加权。低能量数据点对速度场训练贡献更大，因此生成路径整体偏向低能量区域。

**第二层：Energy-Weighted Diffusion**  
在 Gaussian diffusion path 下，条件 score 为

$$
\nabla_x\log p_{t0}(x|x_0)=-(x-\mu_t x_0)/\sigma_t^2.
$$

训练时只需对 batch 内样本计算 softmax 权重

$$
g_i=
\frac{\exp(-\beta E(x_0^i))}
{\sum_j \exp(-\beta E(x_0^j))},
$$

然后加权噪声/score matching loss。

**第三层：QIPO for offline RL**  
在离线 RL 中，将 $x$ 换成状态条件下的动作 $a$，令能量为 $-Q(s,a)$，得到权重：

$$
w(s,a)=
\frac{\exp(\beta Q(s,a))}
{\mathbb{E}_{\tilde{a}\sim\mu(\cdot|s)}[\exp(\beta Q(s,\tilde{a}))]}.
$$

于是 Q-weighted diffusion loss 为：

$$
\mathcal{L}_{QD}(\theta)=
\mathbb{E}
\left[
\frac{\exp(\beta Q(s,a))}
{\mathbb{E}_{\tilde{a}\sim\mu(\cdot|s)}[\exp(\beta Q(s,\tilde{a}))]}
\left\|
s_\theta^t(a_t;s)-\nabla_{a_t}\log p_{t0}(a_t|a)
\right\|^2
\right].
$$

这让策略生成模型直接从行为分布 $\mu(a|s)$ 变成 Q 加权分布 $\mu(a|s)\exp(\beta Q(s,a))$。

真正新的是理论上的等价加权目标以及其 flow matching 版本；借用的是 KL 正则策略改进、diffusion policy、Q-learning 和 batch 内 softmax 估计。

## 6. 算法与伪代码

本文包含两个主要算法。

**Algorithm 1：Energy-Weighted Diffusion Model**

1. 输入 score network $s_\theta^t(\cdot)$、噪声 schedule $(\mu_t,\sigma_t)$、guidance scale $\beta$、batch size $B$。
2. 对每个 batch $\{x_0^i,E(x_0^i)\}_{i=1}^B$：
   1. 计算 batch 内 guidance 权重：
      $g_i=\mathrm{softmax}(-\beta E(x_0^i))$。
   2. 采样时间 $t_i\sim U(0,1)$，采样噪声 $\epsilon_i\sim\mathcal{N}(0,I)$。
   3. 构造 $x_{t_i}=\mu_{t_i}x_0^i+\sigma_{t_i}\epsilon_i$。
   4. 用加权 diffusion loss 更新 $s_\theta$：
      $ \sum_i \lambda(t_i)g_i\|s_\theta^{t_i}(x_{t_i})+\epsilon_i/\sigma_{t_i}\|^2 $。

**Algorithm 2：QIPO diffusion 版**

1. 输入离线数据 $\mathcal{D}=\{(s,a,s',r)\}$、score model、Q function、support set size $M$、renew frequency $K_{renew}$。
2. Diffusion warm-up：先用标准 diffusion 训练行为策略模型 $\mu(a|s)$。
3. Q-learning warm-up：用离线数据训练 $Q_\phi(s,a)$。
4. Policy improvement：
   1. 对 batch 中每个状态 $s_i$，从当前 score model 中采样 support actions $\{a_{ij}\}_{j=1}^{M}$，并加入数据动作 $a_{i0}$。
   2. 计算 Q 权重：
      $g_{ij}=\mathrm{softmax}_j(\beta Q_\phi(s_i,a_{ij}))$。
   3. 对每个 action 加噪得到 $a_{ij,t}$。
   4. 用 Q-weighted diffusion loss 更新 score model。
   5. 每隔 $K_{renew}$ 重新采样 support action set。

QIPO 的迭代性质体现在：

$$
\pi_{l+1}(a|s)\propto \pi_l(a|s)\exp(\beta Q(s,a))
\propto \mu(a|s)\exp((l+1)\beta Q(s,a)).
$$

含义：每次 support set 更新后，策略都进一步向高 Q 动作集中；这比一次性设置很大的 $\beta$ 更稳健。

实现细节：

- QIPO-Diff 使用 diffusion score model。
- QIPO-OT 使用 optimal transport conditional velocity field。
- Q 函数可由任意离线 RL 算法提供，论文实现中使用 in-support softmax Q-learning 风格。
- support action set size 和 renewal period 是重要超参，Appendix E.3 做了消融。

## 7. 实验与消融

**生成模型验证**：

- Table 1 对比 guidance 方法：classifier guidance、classifier-free guidance、contrastive energy prediction、energy-weighted diffusion。本文方法同时满足 exact guidance 和 no auxiliary model。
- Figure 1：二维分类引导例子显示，当 $\beta>1$ 时，CFG 生成分布偏离目标，而 energy-weighted diffusion 更接近真实 $p(x)p(c|x)^\beta$。
- Appendix C 进一步用 8-Gaussian 等例子展示不同能量景观下的采样效果。

**离线 RL 主实验**：

- Benchmark：D4RL。
- 方法：QIPO-Diff、QIPO-OT。
- Baselines：SfBC、QGPO、IDQL、SRPO、Guided Flows，Appendix Table 4 还包含 BEAR、TD3+BC、IQL、Diffuser、Diffusion-QL 等。
- Table 2：QIPO-Diff 和 QIPO-OT 在多个 D4RL 任务上整体优于主要生成式 baseline；QIPO-OT 相对 Guided Flows 在多任务上更高。
- Table 3：相比 QGPO，QIPO-OT 动作选择耗时降低约 63.68%，QIPO-Diff 降低约 25.57%，因为不需要每次反传中间 energy guidance。

**附录结果与消融**：

- Table 4：加入更多传统 offline RL baseline 后，QIPO-Diff 仍有竞争力；$\beta=10$ 的高引导版本也被报告。
- Figure 4：单任务动作选择耗时对比，QIPO-OT 通常更快。
- Figure 5：support action set size $M=16,32,64$，更大 $M$ 通常能提高候选覆盖，但计算更重。
- Figure 6：support action set renewal frequency $K_{renew}$ 影响学习稳定性和引导更新速度。
- Figure 9 / Table 5：还测试了图像/分子属性引导，说明方法不只局限于 RL。
- Figure 10/11：比较 QIPO-Diff 与 QIPO-OT 的 behavior model 学习损失。

**实验解释边界**：

论文证明了 energy-weighted 目标在理论上可精确匹配引导分布，并在 D4RL 上有效。但最终性能仍依赖 Q 函数质量。如果离线 Q 在 OOD 动作上不可靠，Q 加权生成模型仍可能放大错误价值估计。因此 QIPO 解决的是“如何采样 Q 加权分布”，不完全解决“Q 是否可信”。

## 8. 展望

**对 ORL 研究者的启发**：

1. KL 正则离线策略改进天然是能量引导生成问题。
2. 生成式策略的关键不只是 expressive policy，还包括 guidance 是否精确、是否引入辅助误差。
3. Flow matching 可以作为 diffusion policy 的更一般替代，尤其适合研究不同生成路径。
4. Batch 内 softmax 能量加权是一个简单但理论上有力的技巧。
5. 迭代增大有效 guidance scale 可能比一次性大 $\beta$ 更稳。

**局限与开放问题**：

1. QIPO 依赖离线 Q 函数，Q 的外推错误仍可能被放大。
2. batch 内权重估计会受 batch size 和 support set 覆盖影响。
3. 高 $\beta$ 会导致权重集中，可能带来方差和模式坍缩问题。
4. 实验仍主要是 D4RL，真实机器人/工业任务验证不足。
5. Flow matching 与 diffusion 版本的实际选择标准仍需更系统分析。

**后续研究想法**：

1. **不确定性感知 QIPO**：把 Q ensemble 不确定性加入能量，避免高不确定动作被过度加权。
2. **自适应 guidance scale**：根据 effective sample size 或 Q 分布自动调节 $\beta$，缓解权重坍缩。
3. **模型型 ORL 中的 trajectory-level EFM**：将能量加权从单步动作扩展到整条轨迹，结合 return 或 safety energy 进行规划。

## Links

- Paper page: https://openreview.net/forum?id=HA0oLUvuGI
- arXiv: 未能从论文 PDF 正文确认 arXiv 链接
- Code: 未能从论文和公开页面确认官方代码仓库；论文称 supplementary materials 中包含实现以保证复现
- Related dataset in appendix: https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset
