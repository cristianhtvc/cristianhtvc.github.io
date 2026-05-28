---
title: "ContraDiff: Planning Towards High Return States via Contrastive Learning"
description: "ContraDiff 在扩散轨迹规划中引入按回报定义的对比约束，让低回报轨迹作为负样本参与训练，帮助策略远离低回报状态。"
tags:
  - Offline RL
  - Diffusion Planning
  - Contrastive Learning
  - Dataset Imbalance
  - Trajectory Generation
---

# ContraDiff: Planning Towards High Return States via Contrastive Learning

| 字段 | 内容 |
|---|---|
| Title | ContraDiff: Planning Towards High Return States via Contrastive Learning |
| Year | 2025 |
| Source | ICLR 2025 Poster |
| Authors | Yixiang Shan, Zhengbang Zhu, Ting Long, Qifan Liang, Yi Chang, Weinan Zhang, Liang Yin |
| Affiliations | Jilin University; Shanghai Jiao Tong University |
| Tags | Offline RL; diffusion planner; contrastive learning; low-return trajectories; D4RL |

**一句话概括：** ContraDiff 把离线数据中的高回报状态当正样本、低回报状态当负样本，在 Diffuser 式轨迹生成过程中用对比学习把规划轨迹拉向高回报区域、推离低回报区域。

**我对这篇论文的定位：** 这是一篇面向“低高回报比例数据集”的 diffusion planning 方法。它不是传统的 CQL/IQL 价值悲观路线，也不是只给高回报样本加权的 AW/RW 路线，而是把低回报数据显式转化为负向训练信号。放在 ORL 版图里，它连接了 Diffuser/Decision Diffuser、trajectory reweighting、contrastive RL 和数据不均衡问题。

## 1. 第一作者相关信息

第一作者 **Yixiang Shan**。论文首页给出的单位为 Jilin University，邮箱为 `shanyx22@mails.jlu.edu.cn`。

## 2. 研究问题

论文研究的是 **高回报轨迹稀缺时，扩散式离线 RL 如何利用低回报轨迹**。

在标准 offline RL 中，离线数据集 $D=\{(s_t,a_t,r_t,s_{t+1})\}$ 固定，agent 不能再与环境交互。Diffuser 类方法会学习一个轨迹生成模型，在测试时从当前状态开始生成后续轨迹并执行第一个动作。问题是：许多真实数据集不是由专家大量采集，而是含有大量低回报轨迹，少量高回报轨迹只覆盖很窄的区域。**若只重加权高回报样本，模型可能在 agent 进入低回报区域时不知道如何脱离**。

论文把从时刻 $t$ 开始的折扣回报记为：

$$
v_t=\sum_{i\ge t}\gamma^{i-t}r_i .
$$

其中 $v_t$ 用于判断状态 $s_t$ 是“高回报状态”还是“低回报状态”。目标仍是学习策略 $\pi_\theta$，最大化离线训练后在线评估的期望轨迹回报：

$$
\pi_\theta=\arg\max_\theta \mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)].
$$

核心难点不只是分布偏移，而是 **数据回报分布不均衡**：低回报样本很多，但传统方法没有把它们作为“应该远离的区域”来用。

## 3. 背景知识

离线 RL 中，distribution shift 指学到的策略可能访问数据没有覆盖的状态-动作区域，导致模型或价值函数外推错误。Diffusion planning 的思路不同于值函数方法：它把轨迹 $\tau=(s_t,a_t,\ldots,s_{t+H},a_{t+H})$ 当作生成对象，通过去噪从随机噪声中生成未来轨迹，再执行轨迹的第一个动作。

Diffusion model 的简化训练目标可写成：

$$
L_{\mathrm{diff}}=\mathbb{E}_{x_0,i}\left[\lVert x_0-\psi_\theta(x_i,i)\rVert^2\right].
$$

在 RL 里，$x_0$ 可对应一段真实轨迹，$\psi_\theta$ 是从第 $i$ 步加噪轨迹恢复干净轨迹的网络。

Contrastive learning 的基本思想是：给定 anchor，把它拉近正样本，推远负样本。ContraDiff 的关键区别是，正负样本不是由数据增强或同轨迹/异轨迹定义，而是由 **return** 定义：高回报状态为正，低回报状态为负。

| term | plain Chinese meaning | role in this paper |
|---|---|---|
| Diffuser | 用扩散模型生成未来轨迹并取第一步动作 | ContraDiff 的 planning backbone |
| High-return state | 从该状态往后的折扣回报较高 | 对比学习正样本 |
| Low-return state | 从该状态往后的折扣回报较低 | 对比学习负样本 |
| SR | sampling according to return | 只按回报采样正负状态 |
| SRD | sampling according to return and dynamic consistency | 同时考虑回报和可转移性 |
| Contrastive module | 约束生成轨迹的状态分布 | 本文主要新增组件 |

为什么动机成立：如果 agent 已经处在低回报区域，单纯“多模仿高回报轨迹”未必告诉它如何逃离。低回报数据虽然不是好示范，却标记了危险区域；把它们作为负样本，能给生成轨迹一个“反作用力”。

## 4. 问题分析

论文的诊断从 Figure 1 开始：Maze2d 中高回报轨迹比例很低。高回报轨迹充足时，Diffuser 类方法能学到通往目标的模式；高回报轨迹稀缺时，生成轨迹容易偏向低回报区域。已有 reweighting 方法只是提高高回报样本权重，但如果 agent 从低回报状态 $s_t$ 出发，**数据里没有从 $s_t$ 到目标的高回报路径，单靠高回报样本权重不会产生新的逃离信号**。

作者因此提出两个观察：

1. 高回报轨迹中的状态表示“值得靠近的区域”。
2. 低回报轨迹中的状态表示“容易导致失败的区域”。

这排除了把低回报轨迹简单丢弃或降权的做法。ContraDiff 要做的是：在轨迹生成模型内部，让每个生成状态同时受到正样本吸引和负样本排斥。

论文没有给 regret bound 或收敛性定理，主要证据来自机制分析、27 个构造 sub-optimal 数据集实验、低回报区域逃离可视化、以及消融实验。

## 5. 思想与方法

ContraDiff 由两个模块组成。

**Planning Module**：沿用 Diffuser 风格，从当前状态 $s_t$ 出发生成长度为 $H$ 的未来轨迹。初始化时除当前状态外，其余轨迹元素为高斯噪声：

$$
\hat{\tau}_t^N=\{(s_t,\hat{a}_t^N),(\hat{s}_{t+1}^N,\hat{a}_{t+1}^N),\ldots,(\hat{s}_{t+H}^N,\hat{a}_{t+H}^N)\}.
$$

反向去噪时使用 return predictor $J_\phi$ 做 guidance：

$$
p_\theta(\hat{\tau}^{i-1}_t|\hat{\tau}^{i}_t)
=\mathcal{N}\left(\mu_\theta(\hat{\tau}^{i}_t,i)+\alpha\nabla J_\phi(\hat{\tau}^{i}_t,i),\beta_i I\right).
$$

直观含义：扩散模型生成可行动的未来轨迹，return predictor 让轨迹偏向高回报。

**Contrastive Module**：从生成轨迹的重构状态中抽取 anchor，并从离线数据中采样正负状态。论文给出两种采样策略：

- **SR**：只根据状态回报采样，$v_i$ 越高越可能为正样本，越低越可能为负样本。
- **SRD**：先对状态做 MiniBatch K-Means 聚类，只从当前状态所在簇的后继状态集合中选正样本，从而提升动态一致性；负样本仍可来自全体状态。

对比损失使用多正样本/多负样本形式，并刻意从分母去掉正样本项，增强负样本利用：

$$
L_h^i
=-\log
\frac{\sum_{k=1}^{K}\exp(\mathrm{sim}(f(\hat{s}_{h}^{i,0}),f(s_{h,k}^{+}))/T)}
{\sum_{k=1}^{K}\exp(\mathrm{sim}(f(\hat{s}_{h}^{i,0}),f(s_{h,k}^{-}))/T)} .
$$

其中 $f$ 是投影函数，$T$ 是温度，$\mathrm{sim}$ 是余弦相似度。这个目标会提升生成状态与高回报状态的相似度，同时降低其与低回报状态的相似度。

## 6. 算法与伪代码

算法名：**ContraDiff**，包含 **ContraDiff-SR** 和 **ContraDiff-SRD** 两个版本。

训练过程：

1. 从离线数据集 $D$ 采样轨迹片段 $\tau_t$ 和回报 $v_t$。
2. 采样扩散步 $i\in[1,N]$，对轨迹加噪得到 $\tau_t^i$。
3. 用轨迹生成网络 $\psi_\theta(\tau_t^i,i)$ 重构干净轨迹 $\hat{\tau}_t^{i,0}$。
4. 计算扩散重构损失：

$$
L_d=\mathbb{E}_{\tau_t,i}\left[\lVert\tau_t-\psi_\theta(\tau_t^i,i)\rVert^2\right].
$$

5. 用 return predictor $J_\phi$ 预测轨迹回报，计算：

$$
L_v=\mathbb{E}_{\tau_t,i}\left[\lVert v_t-J_\phi(\tau_t^i,i)\rVert^2\right].
$$

6. 对重构轨迹中的每个未来状态，按 SR 或 SRD 采样正负状态集合，计算加权对比损失：

$$
L_c=\mathbb{E}_{i}\left[\sum_{h=t}^{t+H}\frac{1}{h+1}L_h^i\right].
$$

7. 总目标：

$$
L=L_d+L_v+\lambda_c L_c.
$$

规划/执行过程：

1. 观察当前状态 $s_t$。
2. 初始化 $\hat{\tau}_t^N$，固定第一项状态为 $s_t$。
3. 从 $N$ 到 1 逐步去噪，并用 $J_\phi$ 做 return guidance。
4. 得到 $\hat{\tau}_t^0$ 后，执行其中的第一个动作 $\hat{a}_t$。
5. 与环境交互进入下一状态，重复以上过程。

实现细节：论文使用 U-Net 作为去噪网络和 return predictor，用带 Sigmoid 的线性层作为 projector $f$；训练硬件为 4 张 NVIDIA A40。

## 7. 实验与消融

实验主要评估 ContraDiff 在高回报样本不足场景下是否优于普通离线 RL 和重采样方法。

数据集：作者在 D4RL locomotion 中构造 27 个 mixed sub-optimal datasets。环境包括 HalfCheetah、Hopper、Walker2d；数据组合包括 M-Exp、MR-Exp、Rand-Exp，每组混入 Expert 轨迹比例为 0.1、0.2、0.3。论文还在 appendix 报告 regular D4RL、Maze2d 和 Kitchen。

主要基线：

- 普通 offline RL / planning：CQL、IQL、DT、TT、MOPO、CDE、Decision Stacks、ReDiffuser、Diffuser、Decision Diffuser。
- 重采样或重加权：AW、RW、AW-DW、U-DW 与 Diffuser backbone 结合。

关键结果：

- Table 1：在 27 个 sub-optimal 数据集上，ContraDiff-SR/SRD 在 25 个设置中取得最优或次优，说明回报不均衡越强时越有优势。
- Table 2：在同一 Diffuser backbone 下，ContraDiff 大多超过 AW/RW/AW-DW/U-DW，说明低回报负样本提供的信息不是简单重加权能替代的。
- Figure 3：Walker2d-Rand-Exp-0.3 中，给定接近摔倒的初始姿态，Diffuser 很快结束，AW 延长交互但仍失败，ContraDiff 能恢复并持续交互超过 60 步。
- Figure 4：消融显示只用高回报样本的 ContraDiff-N 在 9 个任务中都不如完整 ContraDiff；去掉对比模块的 ContraDiff-C 也全面下降。
- Figure 5/6：ContraDiff 生成和访问的状态更偏向高奖励区域，并在 OOD 状态圈出的区域获得更高 reward。
- Appendix Table 3/4：regular D4RL 中 ContraDiff-SR 在 Maze2d 和部分 locomotion 上也表现强，但在高质量 Med-Expert 场景中优势不如 sub-optimal 场景明显。

实验解释：论文有力证明“低回报样本不是废数据”。但它主要依赖启发式对比学习，没有理论保证；SR 和 SRD 的选择也需要结合数据分布判断。

## 8. 展望

对 ORL 研究者的启发：

- 数据不均衡可以不只靠 reweighting，还可以转化为正负样本结构。
- 低回报轨迹对 policy learning 有“边界”信息，特别适合用于生成模型的分布塑形。
- 对比学习不必只学 representation，也可以直接约束生成式规划的状态分布。
- SRD 提醒我们：把高回报状态当正样本时，需要考虑动态可达性。

局限与开放问题：

- 缺少理论保证，尤其是对比约束是否会引入不可达目标状态。
- SR/SRD 和聚类超参数会影响正样本质量。
- 当前主要验证在 D4RL locomotion、Maze2d、Kitchen，真实机器人数据仍需测试。
- 依赖扩散规划，推理成本比一步策略更高。

可能后续方向：

1. **可达性约束对比学习**：用 learned dynamics 或 reachability model 替代 K-Means，让正样本既高回报又动态可达。
2. **风险敏感 ContraDiff**：把低回报负样本扩展为安全约束或 CVaR 风险区域，适合医疗和机器人安全任务。
3. **与价值悲观结合**：将 contrastive state guidance 和 conservative Q-learning 结合，分别处理状态级失败区域和动作级 OOD 过估计。

## Links

- OpenReview/PDF: https://openreview.net/forum?id=XMOaOigOQo
- arXiv: https://arxiv.org/abs/2402.02772
- Code: https://github.com/Looomo/contradiff
