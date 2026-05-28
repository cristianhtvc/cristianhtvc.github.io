---
title: "Preference Elicitation for Offline Reinforcement Learning"
description: "在完全离线设定下主动选择偏好查询轨迹，用 Sim-OPRL 结合悲观模型学习与乐观偏好采样。"
tags:
  - Offline RL
  - Preference Learning
  - Active Learning
  - Model-based RL
---

# Preference Elicitation for Offline Reinforcement Learning

<div class="paper-hero">
<p class="paper-meta">Alizée Pace, Bernhard Schölkopf, Gunnar Rätsch, Giorgia Ramponi · ICLR 2025 · ICLR / 2025</p>
<div class="tag-row"><span>Offline RL</span><span>Preference Learning</span><span>Active Learning</span><span>Model-based RL</span></div>
</div>

<article class="note-body" markdown>

| 字段 | 内容 |
|---|---|
| Title | Preference Elicitation for Offline Reinforcement Learning |
| Year | arXiv v1: 2024-06-26；ICLR 2025 Poster；OpenReview 发布于 2025-01-22，最终修改于 2025-02-28 |
| Source | ICLR 2025 Poster；此前有 ICML 2024 Workshop MHFAIA 版本 |
| Authors | Alizée Pace, Bernhard Schölkopf, Gunnar Rätsch, Giorgia Ramponi |
| Affiliations | ETH AI Center / ETH Zürich; MPI for Intelligent Systems; ELLIS Institute Tübingen; University of Zürich |
| Tags | Offline PbRL; Preference Elicitation; Active Learning; Pessimism; Optimism; Model-based RL |

**一句话概括**：Sim-OPRL 在没有环境交互、没有显式奖励的条件下，用离线数据学习动力学模型，再在模型中生成接近最优策略的模拟 rollouts 来主动查询偏好，从而比只在离线轨迹池中采样偏好更省查询。

**我对这篇论文的定位**：这是 offline RL 与 preference-based RL 的交叉论文，关注的不是“给定偏好数据如何训练策略”，而是“完全离线时应该问人类哪些轨迹对”。它把 offline RL 的悲观性和 preference elicitation 的乐观探索结合起来：输出策略时对 OOD 悲观，采集偏好时对潜在最优策略乐观。

## 1. 第一作者相关信息

Alizée Pace 是 ETH Zürich / ETH AI Center 和 MPI-IS 的机器学习博士生，个人主页显示其研究兴趣包括 offline RL、preference-based RL、RLHF、causal inference 和 clinical time-series。主页还记录本文于 2025-01-22 被 ICLR 2025 接收，并提到其 2023-2024 年在 Google Gemini 团队从事 RLHF/reward modelling 相关实习。

| Year | Title | Venue/source | Topic tag | Relation to this paper |
|---|---|---|---|---|
| 2025 | Preference Elicitation for Offline Reinforcement Learning | ICLR 2025 Poster | Offline PbRL, active preference | 本文主体工作 |
| 2024 | Preference Elicitation for Offline Reinforcement Learning | arXiv / ICML 2024 Workshop MHFAIA | Workshop version | 本文早期版本 |
| 2024 | 个人主页列出与 Hugo Yèche、Bernhard Schölkopf、Gunnar Rätsch、Guy Tennenholtz 合作论文 | Author homepage | Offline RL / decision support | 与其临床决策和 offline RL 方向相关，但本笔记不展开未核验细节 |

整体轨迹看，第一作者将 offline decision making、偏好反馈和真实临床/决策支持场景联系在一起。本文正好解决两个现实瓶颈：不能在线试错、奖励函数也难以手写。

## 2. 研究问题

标准 offline RL 假设离线数据已经带奖励；标准 preference-based RL 不需要手写奖励，但通常要在线与环境交互收集轨迹和偏好。本文把两者同时设为困难模式：只有离线轨迹 $D_{\mathrm{offline}}$，没有奖励函数，不能与真实环境交互，但可以向人类查询少量轨迹偏好。

核心问题是：为了用尽量少的偏好查询学到最好的离线策略，应该查询哪些轨迹对？如果只在离线数据集中随机或按不确定性选轨迹，可能浪费查询预算在低价值或与最优策略无关的行为上。Sim-OPRL 的答案是：先学模型，再在模型中生成接近 offline optimal policy 的 rollouts，优先问这些更有决策价值的比较。

形式目标是 $(\epsilon,\delta,N_p)$-correct：用 $N_p$ 次偏好查询得到策略 $\hat{\pi}$，使其与真实最优策略的价值差以至少 $1-\delta$ 的概率不超过 $\epsilon$。由于离线数据覆盖有限，论文引入 $\epsilon_T$ 表示即便知道真实奖励也无法突破的 inherent offline suboptimality。

## 3. 背景知识

**Offline RL 的悲观性**：离线数据不覆盖的 state-action 不可靠，因此输出策略时要惩罚转移模型不确定性，避免学到“模型幻想中很好、真实环境中危险”的行为。

**Preference-based RL**：奖励不是数值标签，而来自轨迹比较。论文采用 Bradley-Terry 模型：

$$
P_R(\tau_1 \succ \tau_2)=
\frac{\exp(R(\tau_1))}{\exp(R(\tau_1))+\exp(R(\tau_2))}
=\sigma(R(\tau_1)-R(\tau_2)).
$$

含义：若轨迹 $\tau_1$ 的累计奖励高于 $\tau_2$，人类更可能偏好 $\tau_1$。

**Preference elicitation**：不是被动接受偏好数据，而是主动选择要问的轨迹对。在线 PbRL 可以让环境生成新轨迹；本文不能接触环境，只能在离线数据或学习模型中构造查询。

**悲观与乐观的冲突**：offline RL 要保守，active learning 要探索。Sim-OPRL 的关键是分工：学习/输出策略时使用 pessimism；选择偏好查询时使用 optimism，去覆盖可能最优但尚不确定的策略。

| 术语 | 直白含义 | 在本文中的作用 |
|---|---|---|
| $D_{\mathrm{offline}}$ | 无奖励离线轨迹数据 | 学动力学模型和限制策略覆盖 |
| $D_{\mathrm{pref}}$ | 人类偏好查询数据 | 学奖励模型 |
| OPRL | 只在离线轨迹池里问偏好 | 本文分析和比较的基线 |
| Sim-OPRL | 在学习模型里生成 rollouts 再问偏好 | 本文主算法 |
| $C_T$ | 最优策略被离线数据覆盖程度 | 决定模型误差影响 |
| $C_R$ | 偏好数据对最优策略奖励的覆盖 | 决定偏好查询效率 |

## 4. 问题分析

论文先指出既有 OPRL 的局限：如果偏好查询只能从 $D_{\mathrm{offline}}$ 中抽两条轨迹，那么查询分布被行为策略锁死。行为策略若很差或轨迹池缺少最优附近行为，偏好模型即使学得很准，也未必对最终策略有用。

Theorem 5.1 给出 OPRL 的样本复杂度分析，其性能依赖 transition concentrability 和 preference concentrability。直观地说，离线数据要覆盖最优策略，偏好查询也要覆盖最优策略相关轨迹；如果只在离线池中问，第二个覆盖条件可能很差。

Sim-OPRL 的分析见 Theorem 6.1。通过在学习到的 pessimistic transition model 中 rollout near-optimal policies，偏好查询更集中到可能影响最终策略选择的轨迹上。论文因此把 preference sample complexity 从“被离线轨迹池质量强约束”转向“由模拟 rollouts 对 offline optimal policy 的覆盖决定”。

## 5. 思想与方法

Sim-OPRL 的指导原则是：用离线数据学习“可信环境模型”，用模型生成“值得问的人类比较”。方法由三层组成。

第一层是模型学习。用 $D_{\mathrm{offline}}$ 估计 transition model $\hat{T}$，并通过模型集合/置信集得到不确定性 $u_T(s,a)$。输出策略时使用 pessimistic reward：

$$
\hat{R}(\tau)-u_R(\tau)-u_T(\tau),
$$

避免选择模型和奖励都不确定的轨迹。

第二层是偏好学习。随着 $D_{\mathrm{pref}}$ 增加，用最大似然训练奖励模型 $\hat{R}$，并用 reward ensemble 估计偏好不确定性 $u_{PR}(\tau_1,\tau_2)$。

第三层是主动查询。Sim-OPRL 先构造 near-optimal policy set $\Pi_{\mathrm{offline}}$，再选能最大化偏好不确定性的 exploratory policies，rollout 出两条模拟轨迹并询问人类偏好。这里的 optimism 不是让最终策略冒险，而是让查询更有信息量。

## 6. 算法与伪代码

Algorithm 1 给出通用 offline preference-based RL 框架，Algorithm 2/3 给出 Sim-OPRL 查询流程。

1. 输入无奖励离线轨迹 $D_{\mathrm{offline}}$、偏好预算 $N_p$ 和置信参数。
2. 用 $D_{\mathrm{offline}}$ 最大似然训练 transition model $\hat{T}$，构造 transition uncertainty $u_T$。
3. 初始化偏好数据集 $D_{\mathrm{pref}}=\emptyset$。
4. 重复直到偏好预算用完：
   - 根据当前 reward ensemble 和 pessimistic model 构造 near-optimal policy set；
   - 从中选择两条能最大化 preference uncertainty 的策略；
   - 在 learned model 中 rollout 得到 $\tau_1,\tau_2$；
   - 向人类查询 $\tau_1 \succ \tau_2$ 或反向标签；
   - 更新 $D_{\mathrm{pref}}$；
   - 重新训练/更新 reward ensemble 和不确定性。
5. 最后在 pessimistic learned MDP 中规划，输出保守的最优策略。

实践实现中，reward uncertainty 由 bootstrap ensemble 近似；StarMDP/Gridworld 用精确规划或线性规划，HalfCheetah 用 SAC 训练策略；偏好查询可按 batch 采样以降低计算成本。

## 7. 实验与消融

实验比较 Sim-OPRL、OPRL Uniform、OPRL Uncertainty，以及在线可交互上界 PbOP。环境包括 StarMDP、Gridworld、Sepsis simulator 和 D4RL HalfCheetah。Figure 1 显示在 fully offline 方法中，Sim-OPRL 通常以更少偏好查询达到更高 normalized return；Table 2 汇总达到目标 suboptimality 所需的 $N_p$，Sim-OPRL 在多个环境中需要的偏好数最少。

Figure 2 做算法消融：去掉最终输出策略的 pessimism 会因 OOD 风险导致性能下降；rollout 时去掉 pessimism 会把偏好预算浪费在模型不可信区域；去掉 optimism 则退化成不够主动的信息采集。Figure 3 进一步验证理论预期：离线数据数量不足或行为策略远离最优时，需要更多偏好查询。

附录 Table 5/6 在 D4RL HalfCheetah 上比较不同 offline dataset optimality。结论与理论一致：数据越接近最优，偏好查询越省；固定偏好预算下，Sim-OPRL 往往比 OPRL variants 得到更好策略。

实验能证明 Sim-OPRL 的 preference query 更有效，尤其当离线轨迹池包含大量低价值行为时。弱点是人类偏好标签多由模拟/真实奖励 oracle 生成，真实专家反馈噪声和长轨迹可理解性仍需进一步测试。

## 8. 展望

启发：第一，offline PbRL 的关键瓶颈可能是“问什么”，而不仅是“怎么从偏好训练 reward”。第二，悲观和乐观并不矛盾，可以分别用于安全决策和信息采集。第三，model-based rollouts 在不能真实交互时提供了一种主动偏好查询代理。第四，覆盖度系数 $C_T,C_R$ 是理解 offline preference sample complexity 的核心。

局限：Sim-OPRL 依赖学习模型质量；模拟轨迹若不符合人类可评估格式，真实标注会困难；长 horizon 偏好可能违反 Bradley-Terry 简化假设；大规模连续控制中 near-optimal policy set 的构造较近似；真实医疗等高风险场景需要更严格验证。

后续方向：1. 与语言模型/视觉模型结合，把模拟轨迹转成可解释摘要后再让专家比较。2. 用主动学习选择“短片段”而非整条轨迹，降低标注负担。3. 将 Sim-OPRL 与 RLHF 中的 offline preference optimization 结合，研究 LLM 对齐中的 model-based query synthesis。

## Links

- Paper page: https://openreview.net/forum?id=2pJpFtdVNe
- ICLR proceedings: https://proceedings.iclr.cc/paper_files/paper/2025/hash/7b86ad1216ffa3951575717f526e6efb-Abstract-Conference.html
- arXiv: https://arxiv.org/abs/2406.18450
- First author profile: https://alizeepace.com/
- MPI-IS publication record: https://is.mpg.de/publications/pacschratram25
- Code: 未能从论文/OpenReview/arXiv 页面确认官方代码仓库。

</article>
