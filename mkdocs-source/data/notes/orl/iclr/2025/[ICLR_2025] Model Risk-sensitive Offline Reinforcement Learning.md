---
title: "Model Risk-sensitive Offline Reinforcement Learning"
description: "将金融中的模型风险引入风险敏感离线强化学习，用最坏可置信回报分布的风险替代单一估计风险，并用 critic ensemble 与 Fourier-IQN 实现。"
tags:
  - Offline RL
  - Risk-sensitive RL
  - Distributional RL
  - Model Risk
  - Spectral Risk Measures
---

# Model Risk-sensitive Offline Reinforcement Learning

| 字段 | 内容 |
|---|---|
| Title | Model Risk-sensitive Offline Reinforcement Learning |
| Year | 2025 |
| Source | ICLR 2025 Poster / Published as a conference paper |
| Authors | Gwangpyo Yoo, Honguk Woo |
| Affiliations | Department of Computer Science and Engineering, Sungkyunkwan University |
| Tags | Risk-sensitive RL; Offline RL; Distributional RL; Model Risk; IQN; Fourier Features |
| Local PDF | `[ICLR_2025] Model Risk-sensitive Offline Reinforcement Learning.pdf` |

**一句话概括**：MR-IQN 把风险敏感 offline RL 的目标从“最小化估计回报分布的风险”改成“最小化一组可置信替代分布中的最坏风险”，并用 critic ensemble 自动确定替代分布集合，用 Fourier feature IQN 提升尾部分位数估计精度。

**我对这篇论文的定位**：这篇文章位于风险敏感 RL、分布式 RL 与离线 RL 的交叉点。它不是简单换一个 CVaR 估计器，而是从金融数学的 model risk 出发，指出离线 RL 中的风险估计本身也有模型误差，因此应该对“风险估计会错”再做一层鲁棒化。方法工程上建立在 TD3+BC/IQN/TQC 一类成熟组件之上，概念贡献则是把 model risk 正式搬进 offline RL。

## 1. 第一作者相关信息

**Gwangpyo Yoo** 是 Sungkyunkwan University 计算机科学与工程方向研究者，本文单位与邮箱来自论文首页 `{necrocathy, hwoo}@skku.edu`。共同作者 **Honguk Woo** 是通讯作者。公开来源显示 Gwangpyo Yoo 的研究主题集中在风险敏感强化学习、风险条件化策略和分布式价值估计。

他的研究轨迹比较清晰：先研究如何让策略适配不同风险度量，再在本文中进一步处理离线风险估计对模型误差的敏感性。两篇工作之间的连续性很强：AAAI 2024 的 Risk-Conditioned RL 关注“风险偏好可调”，ICLR 2025 的 MR-IQN 关注“风险估计不可信时如何决策”。

| 年份 | 标题 | Venue/Source | 主题标签 | 与本文关系 |
|---|---|---|---|---|
| 2024 | Risk-Conditioned Reinforcement Learning: A Generalized Approach for Adapting to Varying Risk Measures | AAAI 2024 | Risk-conditioned RL | 为不同风险度量训练可调策略，是本文风险敏感路线的前序工作 |
| 2025 | Model Risk-sensitive Offline Reinforcement Learning | ICLR 2025 Poster | Offline RL, Model Risk | 本文，把 model risk 引入 offline RL |

主要研究方向可概括为：**用分布式 RL 学回报分布，再让策略对不同风险偏好或模型误差更稳健**。来源包括论文 PDF、OpenReview 页面和论文参考文献；由于公开检索不一定覆盖全部出版物，这里不声称完整列表。

## 2. 研究问题

本文处理的具体问题是：**在风险敏感离线 RL 中，风险度量高度依赖回报分布的尾部，而离线训练导致回报分布估计容易有偏；小的分位数误差会被 CVaR/Wang 等风险度量放大，从而让策略做出错误的风险判断**。

标准风险敏感 RL 可写作：

$$
\pi^\star=\arg\min_\pi H_\phi(Z^\pi(s,a)),
$$

其中 $Z^\pi$ 是策略回报随机变量，$H_\phi$ 是用户指定的风险度量。离线场景中 $Z^\pi$ 由 critic 估计而来，容易受到分布偏移、OOD 动作、尾部样本不足和神经网络分位数估计偏差影响。

本文假设单智能体 MDP，奖励可以是随机变量，离线数据固定；算法基于 distributional critic 学习回报分布的 quantile function，并在 TD3+BC 风格 actor-critic 框架中优化风险。它的目标不是提升平均回报，而是在金融和自动驾驶等风险敏感任务中降低尾部风险。

## 3. 背景知识

**风险敏感 RL** 不只看期望回报，而看回报分布的某个风险函数。在金融、自动驾驶、医疗等任务中，平均表现不错但偶尔极差的策略不可接受。

**谱风险度量** 用分位数函数的加权积分表示：

$$
H_\phi(Z)=-\int_0^1 F_Z^{-1}(p)\phi(p)\,dp.
$$

这里 $F_Z^{-1}(p)$ 是回报分布的 $p$ 分位数，$\phi(p)$ 是权重函数。若 $\phi$ 更重视低分位，那么差结果的影响更大。CV@R 是典型例子：关注最差 $\alpha$ 部分的平均；Wang risk measure 则来自保险/期权定价语境。

**IQN** 通过输入分位水平 $p$，输出对应分位值 $F_Z^{-1}(p;s,a)$，因此很适合计算谱风险。但 IQN 的难点是尾部和高频分位函数可能学不准，尤其在离线数据稀疏时。

**模型风险** 来自金融风险管理。传统风险敏感 RL 相信一个估计分布 $\hat{Z}$，而 model risk 问的是：如果这个分布估计错了，在所有“仍然合理”的替代分布中，最坏风险是多少？

$$
MR_\phi(Z;\mu,\sigma,\varepsilon)
=\sup_{X\in\mathcal{M}(Z;\mu,\sigma,\varepsilon)}H_\phi(X).
$$

其中 $\mathcal{M}$ 是可置信替代场景集合，约束替代分布的均值、方差以及与参考分布的 Wasserstein-2 距离。直觉是：不要只相信 critic 给出的单一分布，而要为 critic 可能错的方向预留余量。

| 术语 | 直观含义 | 在本文中的作用 |
|---|---|---|
| Aleatoric risk | 回报随机性本身带来的风险 | 策略要最小化的对象 |
| Spectral risk | 对回报分位数加权的风险 | 统一 CV@R 与 Wang 风险 |
| Model risk | 在可置信替代分布中的最坏风险 | 本文的核心优化目标 |
| IQN | 学习回报分位数函数的 critic | 计算风险和模型风险的基础 |
| Critic ensemble | 多个分布式 critic | 自动估计 $\mu,\sigma,\varepsilon$ |
| Fourier feature | 高频输入映射 | 缓解分位数函数学习的 spectral bias |
| TQC | 截断分位数 critic ensemble | 控制过估计，并构造 ensemble reference |

本文动机成立的关键是：风险度量通常刻意放大尾部，而尾部恰好是离线 RL 最难估计的部分。因此“更准的风险估计”还不够，还要考虑“估计器错了时风险会如何恶化”。

## 4. 问题分析

Figure 1 是全文的诊断图：左侧传统风险敏感 offline RL 直接最小化 critic 估计分布下的风险；右侧 MR-IQN 先构造一组 plausible alternative scenarios，再取其中最坏风险。这个图表达的不是算法细节，而是目标函数层面的改变。

论文的分析路径有三步：

1. Section 3 定义谱风险和 model risk，把优化目标从 $H_\phi(Z^\pi)$ 扩展到 $MR_\phi(Z^\pi;\mu,\sigma,\varepsilon)$。
2. Section 4.1 用 critic ensemble criterion 解决 $\mu,\sigma,\varepsilon$ 在 RL 中未知的问题。
3. Section 4.2 用 Fourier feature quantile regression 解决 IQN 分位数估计的 spectral bias。

最关键的变量是 $\varepsilon$。如果 critic 之间差异大，说明不确定性高，替代分布集合应扩大；如果 critic 一致，集合应收缩。本文通过每个 critic 与 ensemble quantile mixture 的 Wasserstein-2 距离确定 $\varepsilon$，避免手动加入新超参数。

Figure 4 支持另一个诊断：普通 IQN 学习分位数函数时误差和 quantile crossing 明显，Full Fourier IQN 将 Wasserstein-1 距离从 2.23 降至 0.06，quantile crossing 从 6.07% 降至 0.01%。这说明 risk/model risk 计算的瓶颈不只是目标函数，也是 quantile network 的函数逼近能力。

## 5. 思想与方法

MR-IQN 的指导原则是：**风险敏感策略不应只防环境尾部风险，也应防 critic 对尾部风险的建模错误**。这类似 robust optimization，但鲁棒集合不是人工设置，而由 critic ensemble 的不一致性自动给出。

方法名 **MR-IQN**，主要包含四个组件。

**组件 1：Model risk actor objective**。在 TD3+BC 风格框架中，actor 同时最小化 model risk 和行为克隆损失：

$$
L(\pi)=\lambda_{q.learn}\cdot MR_\phi(Z^\pi(s,a);\mu,\sigma,\varepsilon)+(a-a_D)^2.
$$

第一项让动作在最坏可置信风险下仍安全，第二项限制策略不要偏离离线数据支持。

**组件 2：Critic-ensemble criterion**。它自动确定 model risk 所需参数：用最保守的 critic 均值作为 $\mu$，用最大标准差作为 $\sigma$，用 critic 与 ensemble 分位数混合之间的最大 Wasserstein-2 距离作为 $\varepsilon$。这把“critic 分歧”转化为“可置信替代场景集合大小”。

**组件 3：Fourier feature IQN**。普通 IQN 已经用 cosine embedding 表示 quantile level，但状态/动作输入本身也可能需要高频表示。论文在 $(s,a)$ 与 quantile level 上加入 Fourier feature，提升分位数函数尤其是尾部的拟合能力。

**组件 4：TQC-style quantile mixture**。将多个 critic 的分位数混合、排序、截断后得到 reference distribution $F_{ens}^{-1}$，既减轻过估计，也为 $\varepsilon$ 的计算提供稳定参考。

这些组件分别对应问题：model risk 处理风险估计错误，ensemble criterion 处理替代集合参数未知，Fourier feature 处理分位数拟合偏差，BC/TQC 处理离线分布偏移与过估计。

## 6. 算法与伪代码

**算法名**：MR-IQN with Critic-Ensemble Model Risk。

可按 Algorithm 1 重写为：

1. 对 batch 中每个 $(s,a)$，从多个 IQN critic 中采样分位水平 $p_j\sim U[0,1]$，得到每个 critic 的 quantile estimates。
2. 计算每个 critic 的回报均值和标准差。
3. 取 $\mu=\min_i \mathbb{E}[Z_i]$，取 $\sigma=\max_i \mathrm{std}(Z_i)$，形成保守统计量。
4. 将所有 critic 的分位数混合并排序，经过 TQC 截断得到 reference quantile function $F_{ens}^{-1}$。
5. 对每个 critic 计算与 $F_{ens}^{-1}$ 的 Wasserstein-2 距离平方 $\varepsilon_i$。
6. 取 $\varepsilon=\max_i\varepsilon_i$。
7. 根据论文 Corollary 1 / Eq. 15 计算 $MR_\phi(Z;\mu,\sigma,\varepsilon)$。
8. 用 quantile regression loss 训练 critic，用 model risk + BC loss 更新 actor。

训练/推理区分：critic 训练仍是分布式 Bellman 学习；actor 更新时不最大化均值或单一风险，而使用 model risk。风险度量由任务指定，实验主要使用 CV@R(50%) 和 Wang(-0.5)。

重要实现细节包括：ensemble critic 数量、TQC 截断比例、Fourier feature 初始化 $\sigma_{ff}$、IQN quantile samples 数量。Appendix Figure B 显示 critic 数量在 5 到 7 左右较稳定；Appendix Table D 给出 finance/self-driving/D4RL 的超参数。

## 7. 实验与消融

实验覆盖金融、自动驾驶和附录 D4RL。指标主要是平均回报 Mean 和负风险 $-H_\phi(Z^\pi)$；在风险最小化语境中，负风险越高通常表示风险越低或表现越安全。

**Finance scenarios**：Table 1 在 Stock 和 Forex/Trader simulator 中比较 ORAAC、CODAC、IQN-TD3+BC 和 MR-IQN。论文摘要与 Section 5 报告，相比最强风险敏感 offline RL baseline，MR-IQN 在高不确定环境中降低风险 11.2%-18.5%。Forex 高杠杆场景尤其能体现 model risk 的优势，因为尾部和分布偏移更强。

**Self-driving scenarios**：Table 2 使用 AirSim 自动驾驶环境，报告 Mean、负风险、Success 和 Collision。MR-IQN 在 CV@R 和 Wang 风险设置下都显著降低碰撞相关风险，传统 ORAAC/CODAC 往往或过保守、或受估计误差影响。

**Ablation**：Table 3 在 Forex 环境中逐项移除 model risk、Fourier feature、TQC。作者报告 Fourier feature 带来 8.6%-19.8% 增益，TQC 带来 19.8%-41.3% 增益；model risk 与其他组件组合时效果最好。这说明本文不是单一公式贡献，分位数估计质量同样关键。

**D4RL appendix**：Appendix A.1 显示 MR-IQN 在 D4RL MuJoCo 上平均约 68.59，与风险敏感 model-based baseline 1R2R 的 67.35-85.44 区间可比，但不是全面 SOTA。这个结果帮助界定方法适用范围：MR-IQN 的主战场是风险敏感任务，而非标准平均回报 benchmark。

实验能证明：model risk 目标在高不确定尾部风险任务中有优势，Fourier-IQN/TQC 对稳定计算很重要。实验尚不能完全证明：model risk 是所有风险敏感 offline RL 场景中最优目标；其优势依赖 spectral risk measure，且真实系统测试规模有限。

## 8. 展望

对 ORL 研究者的启发：

1. 风险敏感 offline RL 不应只研究“风险度量怎么优化”，还要研究“风险估计错了怎么办”。
2. Distributional critic 的函数逼近误差会直接影响决策安全，尾部分位数质量需要被单独评估。
3. Ensemble disagreement 可以从启发式 uncertainty penalty 转化为鲁棒集合大小，这比手调惩罚系数更自然。
4. 金融数学中的 robust/distortion/model risk 概念仍有许多可迁移到 RL 的空间。

局限和开放问题：

1. 方法局限于 spectral risk measures，暂不覆盖 entropic risk、动态风险等更广类别。
2. Model risk 的替代分布集合由 critic ensemble 诱导，ensemble 共同偏差时仍可能低估风险。
3. 计算成本高于普通 TD3+BC/IQN，尤其是多个 critic 的 quantile mixing 与 Wasserstein 距离。
4. 风险降低与平均回报之间的权衡需要更系统的 Pareto 分析。

可能的后续研究：

1. **Coverage-aware model risk**：将离线数据覆盖度或 density ratio 纳入 $\varepsilon$，让 OOD 区域的可置信集合更大。
2. **Dynamic model risk RL**：把静态谱风险扩展到时间一致的动态风险度量，适配多阶段安全约束。
3. **Model risk for preference/RLHF**：在人类偏好奖励模型不可靠时，用 model risk 处理 reward model tail error。

## Links

- Paper page: https://openreview.net/forum?id=h6k4809xVV
- PDF: https://openreview.net/pdf?id=h6k4809xVV
- Code: 未能从论文和公开来源确认官方代码链接
- First-author related paper: https://ojs.aaai.org/index.php/AAAI/article/view/29049
- Source notes: OpenReview and the local PDF were used for venue status, author list, formulas, Algorithm 1, Figures 1/4/5 and Tables 1-3.
