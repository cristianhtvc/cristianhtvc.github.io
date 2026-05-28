---
title: "Multi-level Certified Defense Against Poisoning Attacks in Offline Reinforcement Learning"
description: "用差分隐私的 outcomes guarantee 为离线 RL 数据投毒提供动作级与策略级双层认证防御。"
tags:
  - Offline RL
  - Certified Defense
  - Data Poisoning
  - Differential Privacy
  - Safety
---

# Multi-level Certified Defense Against Poisoning Attacks in Offline Reinforcement Learning

| 字段 | 内容 |
|---|---|
| Title | Multi-level Certified Defense Against Poisoning Attacks in Offline Reinforcement Learning |
| Year | 2025 |
| Source | ICLR 2025 Poster；OpenReview 显示 Published: 2025-01-22, Last Modified: 2025-02-28 |
| Authors | Shijie Liu, Andrew C. Cullen, Paul Montague, Sarah Erfani, Benjamin I. P. Rubinstein |
| Affiliations | University of Melbourne, School of Computing and Information Systems; Defence Science and Technology Group, Adelaide |
| Tags | Offline RL Security; Certified Robustness; Poisoning Attack; Differential Privacy; Action-Level Certification; Policy-Level Certification |

**一句话概括**：MuCD 把 DP-based randomized training 引入离线 RL 投毒防御，同时给出动作级稳定性和策略级期望累计回报下界认证，把 COPA 只能处理离散动作、确定环境、单轨迹认证的适用范围扩展到更一般的 offline RL 设置。

**我对这篇论文的定位**：这是一篇 offline RL safety/security 论文，而不是传统意义上的新 offline RL 优化算法。它的核心贡献是“认证框架”：训练过程必须是满足 DP guarantee 的随机化算法，随后利用 DP 的 post-processing 和 outcomes guarantee 推导投毒半径下的可认证性能下界。它与 COPA 的关系很直接：继承“可计算认证”的目标，但用 DP 替代 COPA 的树搜索，从而支持连续动作和随机环境。

## 1. 第一作者相关信息

**Shijie Liu** 是 University of Melbourne 的 postdoc；OpenReview 个人页显示其 2024 年起任 postdoc，PhD 阶段也在 Melbourne，研究标签包括 Deep Learning、Adversarial Machine Learning 和 Reinforcement Learning。PDF 第一页给出其单位为 University of Melbourne，并标注通讯邮箱。

其研究轨迹主要围绕 adversarial ML、certified robustness 和安全机器学习展开。本文把其团队在随机平滑/认证防御中的工具迁移到 offline RL 投毒场景，并把“分类模型输出稳定性”的认证思路推广到“RL 策略行为与期望回报”的认证。

| 年份 | 论文 | Venue/Source | 主题标签 | 与本文关系 |
|---|---|---|---|---|
| 2025 | Multi-level Certified Defense Against Poisoning Attacks in Offline Reinforcement Learning | ICLR 2025 Poster | Offline RL Poisoning Defense | 本文；多层级认证防御 |
| 2025 | Fortifying Time Series: DTW-Certified Robust Anomaly Detection | NeurIPS 2025 Poster | Certified Robustness | 同属认证鲁棒性方向 |
| 2023 | Et Tu Certifications: Robustness Certificates Yield Better Adversarial Examples | AAAI 2023 | Certified Defense / Attacks | 研究认证机制的边界与攻击面 |
| 2022 | Double Bubble, Toil and Trouble: Enhancing Certified Robustness through Transitivity | NeurIPS 2022 | Certified Robustness | 团队早期认证鲁棒性工作 |

**主要方向概括**：认证鲁棒性、对抗机器学习、差分隐私和 RL 安全。出版列表以 OpenReview profile、DBLP 与论文页可核验条目为准。

## 2. 研究问题

离线 RL 的训练数据常来自外部日志、未知行为策略或第三方系统。攻击者如果能修改部分轨迹或 transition，就可能让学到的策略在部署时做出危险动作。经验防御通常只能抵抗已知攻击；认证防御则要回答更强的问题：在任意不超过预算 $r$ 的数据投毒下，策略行为或性能是否仍有可证明下界？

论文区分两类攻击粒度：

1. **Transition-level poisoning**：数据收集完成后，攻击者独立修改若干 transition。
2. **Trajectory-level poisoning**：攻击发生在数据收集过程中，被污染的轨迹可能整条受影响。

论文也定义两类认证目标：

1. **Policy-level robustness**：给出投毒后策略的期望累计回报下界 $J_r(\tilde{\pi})$。
2. **Action-level robustness**：在某个测试状态 $s_t$，证明投毒后策略选择的动作仍不变。

形式化地，随机训练算法 $M$ 用干净数据 $D$ 输出策略 $\pi=M(D)$；投毒数据 $\tilde D\in B(D,r)$ 输出 $\tilde\pi=M(\tilde D)$。认证目标是对所有这样的 $\tilde D$ 给出性能或动作稳定性保证。

## 3. 背景知识

**离线 RL 投毒攻击**：攻击者不需要干扰在线交互，只需污染训练日志，例如改状态、动作、奖励或下一状态。因为 RL 的数据是序列化的，早期 transition 的修改可能影响后续轨迹分布，所以投毒风险比普通监督学习更复杂。

**Certified defense**：不是说“某个攻击没打过”，而是给出最坏情况保证。例如“只要投毒轨迹不超过 $r$ 条，期望回报下界仍不少于某个值”。

**Differential Privacy (DP)**：DP 原本刻画模型输出对单个数据点变化的不敏感性。这里的关键转译是：如果训练过程对数据变化不敏感，那么攻击者改少量数据也不能大幅改变策略输出分布。论文使用 ADP 和 RDP 两种机制，并通过 group privacy 扩展到最多 $r$ 个数据点差异。

**Outcomes guarantee**：论文 Definition 3.5 把 DP 写成函数族 $K$ 对输出事件概率的约束。对任意输出集合 $S$，干净数据与投毒数据上的输出概率被 $K$ 互相限制。这个概率界再通过 Lemma 4.1 扩展为期望值界。

| 术语 | 直观含义 | 在本文中的作用 |
|---|---|---|
| Trajectory-level poisoning | 攻击整条轨迹或收集过程 | 对应 DP-FedAvg 式轨迹级隐私单元 |
| Transition-level poisoning | 攻击单个 transition | 对应 SGM/DP-SGD 式 transition 级隐私单元 |
| ADP | $(\epsilon,\delta)$ approximate DP | 给出较通用但相对松的认证界 |
| RDP | Renyi DP | 在深度网络多轮迭代中常给出更紧隐私损失累积 |
| Policy-level certificate | 期望累计回报下界 | 评估整体策略抗投毒能力 |
| Action-level certificate | 单状态动作稳定性 | 评估关键状态是否仍选择同一动作 |

**为什么动机成立**：COPA 虽然是实用认证防御，但依赖 exhaustive tree search，所以只能处理离散动作和确定环境，并且轨迹长度稍大就爆炸。DP 的 post-processing property 与 RL 的“训练后再评估/决策”结构天然匹配：只要训练器是 DP 的，后续计算策略回报、Q 值动作偏好等都不破坏认证。

## 4. 问题分析

论文的关键分析是把 DP 概率输出界变成 RL 回报和动作的认证界。

Lemma 4.1 对有界输出 $M(D)\in[0,b]$ 给出期望 outcomes guarantee。ADP 情况下，如果 $M$ 满足 $(K,r)$ outcomes guarantee，则

$$
e^{-\epsilon}\left(\mathbb{E}[M(D)]-b\delta\right)
\le
\mathbb{E}[M(\tilde D)]
\le
e^{\epsilon}\mathbb{E}[M(D)]+b\delta.
$$

含义：投毒数据下的输出期望不能偏离干净数据太多；偏离幅度由隐私预算 $\epsilon,\delta$ 和输出上界 $b$ 控制。

Theorem 4.2 把 $M$ 的输出解释为策略，再把策略映射到累计回报 $C(\pi)$。由于 $C(M(D))$ 是 DP 输出的后处理，仍满足同样 guarantee，因此得到 policy-level 下界：

$$
J(\tilde{\pi}) \ge J_r(\tilde{\pi})
= e^{-\epsilon}\left(J(\pi)-b\delta\right)
$$

用于 ADP；RDP 对应另一个幂形式下界。实际计算 $J(\pi)$ 时，作者用 $p=50$ 个随机化训练出来的 policy instances，各评估若干 episode，形成经验 CDF，并用 DKW 不等式给出 CDF 置信带。

Theorem 4.4 则处理 action-level。对每个动作 $a_l$ 定义 inferred score：

$$
I_{a_l}(s_t,\pi)
=
\Pr\left[\arg\max_{a_i} Q_{\pi}(s_t,a_i)=a_l\right].
$$

如果干净策略选择动作 $a_t$，并且存在 $K_1,K_2$ 使

$$
K_1^{-1}(I_{a_t}(s_t,\pi))
>
\max_{a_l\ne a_t} K_2(I_{a_l}(s_t,\pi)),
$$

那么投毒半径 $r$ 内动作不变。直观上，赢家动作的下界仍高于所有输家动作的上界，投毒也翻不了盘。

## 5. 思想与方法

方法名是 **MuCD: Multi-level Certified Defenses**。

指导原则是：先把训练算法随机化并确保 DP，再把 DP 的稳定性转化为 RL 层面的可认证稳定性。它不是靠检测攻击样本，而是让训练输出对少量样本变化天然不敏感。

方法由三部分组成：

1. **DP randomized training**：训练 $p$ 个 policy instances。transition-level 使用 Sampled Gaussian Mechanism (SGM)，trajectory-level 改造 DP-FedAvg。
2. **Policy-level certification**：用 Theorem 4.2 计算投毒规模 $r$ 下的期望累计回报下界。
3. **Action-level certification**：用 inferred scores 和 Theorem 4.4 对每个状态计算最大可容忍投毒半径 $r_t$。

它的新意不是提出一个新的 offline RL policy optimizer，而是把 DP 认证从分类任务推广到 RL 的两个层级：单状态动作和整体期望回报。与 COPA 相比，MuCD 避免树搜索，因此能覆盖随机环境、连续动作和更长轨迹。

证据强度：理论上主要依赖 DP post-processing、Fubini/Hölder 推导的期望界和 inferred score 的概率界；实验上在 Atari 和 MuJoCo 上验证认证半径与 clean performance 的 tradeoff。

## 6. 算法与伪代码

**MuCD 工作流**

1. 选择威胁模型：transition-level 或 trajectory-level。
2. 根据威胁模型选择 DP 训练器：
   - transition-level：SGM，对 batch 内样本梯度裁剪并加高斯噪声；
   - trajectory-level：DP-FedAvg，把整条轨迹当作隐私单元，做轨迹级裁剪和加噪。
3. 用相同 DP 机制训练 $p$ 个随机 policy instances $\hat{\pi}_1,\dots,\hat{\pi}_p$。
4. Policy-level：
   - 每个 policy instance 运行若干 episode，收集累计回报；
   - 构造经验 CDF；
   - 用 DKW 不等式得到 $J(\pi)$ 的下界；
   - 代入 ADP/RDP 公式得到 $J_r(\tilde{\pi})$。
5. Action-level：
   - 对状态 $s_t$，统计 $p$ 个 policy instances 的动作选择，估计每个动作的 inferred score；
   - 用 SIMUEM 给 inferred score 上下置信界；
   - 对投毒半径 $r$ 做二分搜索，找到满足 Theorem 4.4 条件的最大 $r_t$；
   - 计算稳定率

$$
\mathrm{Stability\ Ratio}
=
\frac{1}{H}\sum_{t=0}^{H-1}\mathbf{1}[r_t\ge \bar r].
$$

**SGM 的核心步骤**（Appendix Algorithm 1）：采样 mini-batch、计算梯度、按阈值 $C$ 裁剪、加入 $\mathcal{N}(0,\sigma^2C^2I)$ 噪声、更新参数。$\sigma$ 越大，认证越强，但 clean performance 通常下降。

**DP-FedAvg 的核心步骤**（Appendix Algorithm 2）：采样若干轨迹，每条轨迹本地训练若干步，裁剪“轨迹级模型更新”，再对平均更新加高斯噪声。

重要超参数包括噪声倍数 $\sigma$、policy instances 数 $p$、评估次数 $m$、置信参数 $\delta$、裁剪阈值 $C$ 和采样率 $q$。论文实验中 $p=50$，每个游戏数据约 2M transitions，batch size 为 32，$\delta=0.001$。

## 7. 实验与消融

**实验环境**：Atari Freeway、Breakout，以及连续动作 MuJoCo HalfCheetah。离线数据来自 D4RL；训练算法包括 DQN、C51、IQL；实现使用 PyTorch、Opacus 和 A100 GPU。

**Action-level robustness**：Figure 1 和 Table 1 比较 Freeway/Breakout 上的稳定率和平均最大可容忍投毒半径。RDP 下，Freeway 中 DQN 在 $\sigma=2.0$ 时平均回报 16.6，transition-level mean radii 145.5、trajectory-level mean radii 58.7；COPA 在相近回报下 trajectory radii 约 10.1。Breakout 中 $\sigma=2.0$ 时 radius 明显上升，但 C51 clean reward 下降较大，体现认证-性能 tradeoff。

**Policy-level robustness**：Figure 2 给出 Freeway/Breakout 的 $J_r$ 曲线，Figure 3 给出 HalfCheetah 连续动作环境。RDP 曲线普遍比 ADP 更紧，说明 RDP 对深度网络迭代组合的隐私损失刻画更有利。

**与 COPA 对比**：COPA 因树搜索限制，Freeway 只能处理 400 步、Breakout 75 步；默认轨迹分别约 2000 和 600 步。论文报告，在“性能下降不超过 50%”的策略级认证中，COPA 在 Freeway 仅允许约 0.008% 轨迹被投毒，而 MuCD 达到 7.17%；Breakout 中 COPA 为 0.0075%，MuCD 为 2.05%。

**实验解读**：MuCD 证明了 DP 认证可以覆盖比 COPA 更一般的 RL 设置；但它依赖 DP 训练，噪声带来 clean performance 损失。表中也能看到 $\sigma$ 增大时认证半径提升、平均回报下降的典型 tradeoff。

## 8. 展望

对 ORL 研究者的启发：

1. 离线 RL 安全不应只看平均回报，还要给出投毒预算下的可认证下界。
2. DP 的 post-processing property 很适合 RL，因为策略评估、动作投票和回报计算都可视为训练输出的后处理。
3. Action-level 与 policy-level 认证回答的问题不同，安全关键状态更需要 action-level 证书。
4. RDP 在深度 RL 训练中的隐私预算累积通常比 ADP 更实用。

局限与开放问题：

1. DP 噪声损害 clean performance，尤其在 Breakout 的 C51 上较明显。
2. 认证半径依赖训练器满足 DP；现成 offline RL 算法需要改造。
3. Policy-level 认证需要大量评估 episode 来估计 CDF。
4. 论文没有给出针对更强自适应投毒攻击的完整实证攻击-防御曲线，附录只做补充测试。

可能的后续研究：

1. **DP-IQL/CQL 的性能保持**：研究更适合 offline RL 的低噪声 DP optimizer，减少认证代价。
2. **关键状态认证调度**：只对安全关键状态进行高精度 action-level certification，降低评估成本。
3. **认证与数据清洗结合**：先用异常检测降低投毒预算，再用 MuCD 给剩余风险认证。

## Links

- Paper page: [OpenReview](https://openreview.net/forum?id=X2x2DuGIbx)
- PDF: [OpenReview PDF](https://openreview.net/pdf?id=X2x2DuGIbx)
- arXiv: [arXiv:2505.20621](https://arxiv.org/abs/2505.20621)
- Code: 未能从论文或公开来源确认官方代码链接
- First author profile: [Shijie Liu on OpenReview](https://openreview.net/profile?id=~Shijie_Liu4)
