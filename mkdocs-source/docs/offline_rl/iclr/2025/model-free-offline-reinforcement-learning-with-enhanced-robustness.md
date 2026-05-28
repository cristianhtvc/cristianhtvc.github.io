---
title: "Model-Free Offline Reinforcement Learning with Enhanced Robustness"
description: "提出 double-pessimism 原则，用 model-free Q-learning 同时处理离线数据不足和部署环境模型偏差。"
tags:
  - Offline RL
  - Robust RL
  - Model-Free RL
  - Pessimism
  - Sample Complexity
---

# Model-Free Offline Reinforcement Learning with Enhanced Robustness

<div class="paper-hero">
<p class="paper-meta">Chi Zhang, Zain Ulabedeen Farhat, George K. Atia, Yue Wang · ICLR 2025 · ICLR / 2025</p>
<div class="tag-row"><span>Offline RL</span><span>Robust RL</span><span>Model-Free RL</span><span>Pessimism</span><span>Sample Complexity</span></div>
</div>

<article class="note-body" markdown>

| 字段 | 内容 |
|---|---|
| Title | Model-Free Offline Reinforcement Learning with Enhanced Robustness |
| Year | 2025 |
| Source | ICLR 2025 Poster；OpenReview 显示 Published: 2025-01-22, Last Modified: 2025-03-14 |
| Authors | Chi Zhang, Zain Ulabedeen Farhat, George K. Atia, Yue Wang |
| Affiliations | University of Central Florida, Department of Electrical and Computer Engineering; Department of Computer Science |
| Tags | Offline RL; Robust RL; Model-Free RL; Double Pessimism; Robust MDP; Sample Complexity |

**一句话概括**：这篇论文把鲁棒离线 RL 中的两种风险分开处理：用普通离线 RL 的悲观项处理有限数据估计误差，再用面向不确定性集合的额外悲观项处理部署环境与数据环境之间的 model mismatch，从而得到第一个带样本复杂度分析的 model-free offline robust Q-learning 框架。

**我对这篇论文的定位**：它处在 offline RL、robust MDP 和 model-free 理论交叉处。已有 robust offline RL 多是 model-based，需要估计并存储完整转移核，空间通常为 $O(S^2A)$；本文试图把鲁棒性从“规划完整模型”改写为“在 Q-learning 目标里加一项可由样本估计的鲁棒惩罚”。因此它的贡献更偏理论与算法原则，而不是大规模深度 RL benchmark 上的新 SOTA。

## 1. 第一作者相关信息

**Chi Zhang** 是 University of Central Florida ECE 的 postdoc；OpenReview 个人页显示其 2024 年起在 UCF，研究标签包括 Reinforcement Learning、Dynamic Optimization、Robust Control 等。论文 PDF 第一页也给出 UCF ECE/CS 机构和邮箱。

从公开资料看，Chi Zhang 近年的研究轨迹从动态优化、鲁棒控制扩展到鲁棒/迁移强化学习：本文关注 model mismatch 下的离线鲁棒性；2025 年 ICML poster 则把 pessimism principle 用到 zero-shot transfer RL；其合作者 Yue Wang、George K. Atia 也活跃在 robust/offline RL 理论方向。

| 年份 | 论文 | Venue/Source | 主题标签 | 与本文关系 |
|---|---|---|---|---|
| 2025 | Model-Free Offline Reinforcement Learning with Enhanced Robustness | ICLR 2025 Poster | Offline Robust RL | 本文；提出 double-pessimism model-free 算法 |
| 2025 | Pessimism Principle Can Be Effective: Towards a Framework for Zero-Shot Transfer Reinforcement Learning | ICML 2025 Poster | Zero-shot Transfer RL | 延续“悲观原则处理环境转移风险”的思路 |
| 2025/2026 | Provably Sample-Efficient Robust Reinforcement Learning with Average Reward | OpenReview profile listed withdrawn ICLR 2026 submission | Average-Reward Robust RL | 同一团队的鲁棒 RL 理论延伸，状态需以 OpenReview 为准 |

**主要方向概括**：离线/鲁棒 RL 的样本复杂度、pessimism-based learning、环境扰动和 transfer/generalization。作者出版记录存在同名歧义，本节只采用 PDF、OpenReview 个人页和论文页可核验的信息。

## 2. 研究问题

论文要解决的具体问题是：在离线数据只来自名义环境 $P$ 的情况下，如何学到一个在部署时面对转移核扰动仍然表现好的策略，并且不要像 model-based robust RL 那样显式估计 $P(s' \mid s,a)$。

形式化上，作者把问题写成 robust MDP。对于每个状态动作对 $(s,a)$，部署转移核属于一个围绕名义转移核的集合：

$$
\mathcal{P}_{s,a}=\{P_{s,a}+q \in \Delta(S): q \in \mathcal{Q}_{s,a}\}.
$$

目标不是最大化名义环境价值，而是最大化最坏情况下价值：

$$
\max_{\pi} V^{\pi}(\rho), \quad
V^{\pi}(s)=\inf_{P'\in \mathcal{P}} V^{\pi,P'}(s).
$$

这件事难在有两种不确定性叠在一起：离线数据覆盖不足导致 Q 估计容易乐观；部署环境可能偏离名义环境，单纯覆盖数据分布仍不够。已有离线 RL 常处理第一种，已有 robust RL 常处理第二种但依赖完整模型。本文的假设是有限状态动作空间、tabular setting，分别讨论 finite-horizon 和 discounted infinite-horizon；主理论针对 $l_\alpha$-norm uncertainty set，并在附录讨论 $\chi^2$ divergence case study。

## 3. 背景知识

**标准 offline RL**：数据集由行为策略 $\mu$ 预先收集，训练时不能再与环境交互。由于目标策略可能选择数据中很少出现的动作，Q 函数会在 OOD 区域过估计，常见做法是 pessimism：对低覆盖 $(s,a)$ 加惩罚，使策略更偏向数据支持内的动作。

**Robust MDP**：环境转移不是固定的 $P$，而是属于不确定性集合 $\mathcal{P}$。策略价值按最坏转移核评估。直觉是：如果现实环境有 sim-to-real gap、非平稳性或参数扰动，鲁棒策略不追求名义环境最高分，而追求扰动下仍不崩。

**Model-based vs model-free robust offline RL**：model-based 方法要估计 $\hat P_{s,a}$ 并进行 robust planning，空间一般是 $O(S^2A)$；model-free 方法只维护 $Q(s,a)$、访问计数和样本后继状态，空间可降到 $O(SA)$。本文的核心问题就是：能否不用完整转移模型，也能得到 robust Bellman update 的保守下界？

**Double pessimism 的含义**：第一层 pessimism 是离线学习的统计惩罚 $b_n$，随访问次数 $n$ 增大而减小；第二层 pessimism 是模型偏差惩罚 $\kappa_{s,a}(V)$，用于近似 worst-case transition 对价值的影响。二者分别对应“我没有足够数据”和“现实模型可能变了”。

| 术语 | 直观含义 | 在本文中的作用 |
|---|---|---|
| Robust MDP | 转移核可在集合内变化的 MDP | 刻画部署环境与数据环境不一致 |
| $l_\alpha$ uncertainty set | 用 $l_\alpha$ 范数限制转移概率扰动大小 | 主理论分析对象 |
| Pessimism bonus $b_n$ | 对低访问次数状态动作保守估值 | 处理有限离线数据误差 |
| $\kappa_{s,a}(V)$ | 对模型扰动导致的最坏价值损失加惩罚 | 处理 model mismatch |
| Partial coverage | 行为策略覆盖最优鲁棒策略所需区域 | 样本复杂度定理的关键条件 |
| Burn-in cost | 与精度 $\epsilon$ 无关的样本门槛 | 定理 2 中出现的固定启动成本 |

**为什么这篇文章的动机成立**：如果只做普通 offline pessimism，学到的策略可能在名义环境可靠，但参数扰动后性能明显下降；如果做传统 robust planning，又要存储和优化完整模型，在大状态空间和复杂动力学中不现实。本文把 robust planning 里的 worst-case correction 变成一个可插入 Q-learning target 的惩罚项，因此试图同时保住鲁棒性和可扩展性。

## 4. 问题分析

论文的诊断路径有三步。

第一，Section 1 指出 robustness-scalability tradeoff：model-free offline RL 可扩展但通常假设训练环境等于部署环境；robust offline RL 能处理模型偏差但多为 model-based，内存和计算开销高。

第二，Section 3-5 把 robust Bellman 更新拆成可由样本估计的形式。普通离线 Q-learning target 大致是 $r+\gamma V(s')-b_n$；robust 版本还需要 worst-case transition 的下界。作者定义惩罚函数 $\kappa$，要求满足一种保守关系，使得

$$
\mathbb{E}_{s'\sim P_{s,a}}\left[V(s')-\kappa_{s,a}(V)\right]
\leq
\inf_{P'\in \mathcal{P}_{s,a}}\mathbb{E}_{s'\sim P'}[V(s')].
$$

直观解释：名义样本后继价值减去 $\kappa$ 后，应不高于不确定性集合中的最坏后继价值。这样即使没有显式模型，单样本 TD target 仍然是 robust value 的保守估计。

第三，Theorem 2 和 Theorem 3 给出样本复杂度。有限时域中，Algorithm 1 在 $l_\alpha$ uncertainty set 下得到约

$$
\tilde{O}\left(\sqrt{\frac{H^6 S C^\star}{T}}\right)
$$

的次优性界，因此达到 $\epsilon$-optimal 需要约 $\tilde{O}(H^6SC^\star/\epsilon^2)$ 样本。无限时域中，Theorem 3 给出与 model-free non-robust offline RL 相匹配量级的样本复杂度，主项约为

$$
\tilde{O}\left(\frac{S C^\star}{(1-\gamma)^5\epsilon^2}\right).
$$

其中 $C^\star$ 是覆盖/集中系数，衡量离线数据对最优鲁棒策略相关状态动作的覆盖程度。Table 1 进一步比较：本文内存 $O(SA)$，而 Shi & Chi 2022、Blanchet et al. 2023 需要 $O(S^2A)$；Blanchet et al. 还涉及 NP-hard 计算。

## 5. 思想与方法

方法名可概括为 **Double-Pessimism Robust Q-Learning**。

核心思想是把鲁棒性拆成两个可解释惩罚项：

$$
Q(s,a)\leftarrow (1-\eta)Q(s,a)
+\eta\left[r+\gamma V(s')-\gamma\kappa_{s,a}(V)-b_n\right].
$$

$b_n$ 是数据悲观项：访问次数越少，惩罚越大，防止有限样本下的乐观外推。$\kappa_{s,a}(V)$ 是模型悲观项：利用不确定性集合结构对后继价值做最坏情况修正，防止策略只在名义模型上好看。

对 $l_\alpha$-norm uncertainty set，Lemma 1 给出可用惩罚：

$$
\kappa_{s,a}(V)=R_{s,a}\min_{w\in\mathbb{R}}\|we-V\|_\beta,
\quad
\frac{1}{\alpha}+\frac{1}{\beta}=1.
$$

含义是：如果部署转移概率能在半径 $R_{s,a}$ 内移动，那么最坏情形会把概率质量推向低价值状态；$\kappa$ 用 $V$ 的跨度/离散程度估计这种最坏损失。

真正新的是：作者不是重新发明 pessimistic Q-learning，而是把 robust MDP 的支持函数/惩罚项以 model-free 方式接入更新，并证明在 tabular offline setting 中可以达到接近已有最优量级的样本复杂度。证据强度上，理论是主贡献；实验主要验证趋势，而不是覆盖现代深度 offline RL 的完整生态。

## 6. 算法与伪代码

**Algorithm 1: finite-horizon Double-Pessimism Q-Learning**

1. 输入离线轨迹数据 $D$、置信参数 $\delta$、不确定性半径 $R$ 和惩罚函数 $\kappa$。
2. 初始化所有 $Q_h(s,a)=0$、$V_h(s)=0$、访问计数 $N_h(s,a)=0$。
3. 对每条离线 episode，从 $h=1$ 到 $H$ 扫描样本 $(s_h,a_h,r_h,s_{h+1})$。
4. 更新访问计数 $N_h(s_h,a_h)$，令当前次数为 $n$。
5. 设学习率 $\eta_n=(H+1)/(H+n)$。
6. 构造数据悲观项

$$
b_n=c_b\sqrt{\frac{H^3\log^2(SAKH/\delta)}{n}}.
$$

7. 用双重悲观 target 更新：

$$
Q_h(s_h,a_h)\leftarrow (1-\eta_n)Q_h(s_h,a_h)
+\eta_n\left[r_h+V_{h+1}(s_{h+1})-\kappa_{h,s_h,a_h}(V_{h+1})-b_n\right].
$$

8. 更新 $V_h(s_h)=\max\{V_h(s_h),\max_a Q_h(s_h,a)\}$，保持价值估计单调保守。
9. 每轮后令 $\hat{\pi}_h(s)=\arg\max_a Q_h(s,a)$。

**Infinite-horizon 版本**在 Appendix E.1 的 Algorithm 3 中给出，思想相同：沿一条长离线轨迹异步更新 $Q(s,a)$，使用折扣因子 $\gamma$、访问次数学习率和双重惩罚。

**$\chi^2$ divergence case study** 在 Appendix C.3 和 Algorithm 2 中展示。对

$$
D_{\chi^2}(P_{s,a}+q \| P_{s,a})\le R_{s,a},
$$

最坏修正与方差有关：

$$
\min_q \sum_i q_i V_i
=
-\sqrt{R_{s,a}\mathrm{Var}_{P_{s,a}}(V)}.
$$

因此需要两个后继样本估计方差，但仍不需要显式存储完整 $P_{s,a}$。

重要实现点：理论算法是 tabular 的；附录还把 double-pessimism 思想接到 CQL 上做 CartPole case study，但这不是主理论覆盖的深度函数逼近结果。

## 7. 实验与消融

论文实验分两类。

**Garnet 随机 MDP**：Section 8.1 使用 Garnet $G(20,30,20)$、$G(30,50,30)$、$G(50,100,50)$，$l_\infty$ uncertainty radius 在 $[0.1,0.5]$。Figure 1 显示，随着数据规模增大，double-pessimism 的 robust optimality gap 持续低于 single-pessimism baseline，并更接近最优鲁棒值。

**Classic Control 参数扰动**：Section 8.2 在 MountainCar、CartPole 等 OpenAI Gym 环境中扰动动力学参数，例如 MountainCar 的 gravity/force、CartPole 的 pole length。Figure 2 和 Appendix Figure 4 显示，在随机参数扰动下，double-pessimism 的奖励分布整体优于 single-pessimism，说明额外模型悲观项确实提升了部署扰动下的稳定性。

**附录实验**：Appendix B 还包括 Frozen-Lake、Taxi、American Option，以及 $\chi^2$ uncertainty set 的 Garnet 和模拟环境结果。Figure 7-8 显示 $\chi^2$ case 中 double-pessimism 也优于 single-pessimism。

**实验解释**：这些实验支持“模型偏差悲观项有用”这个机制判断；但实验主要是 tabular/Classic Control 和算法思想验证。它还没有证明在 D4RL、NeoRL 或机器人高维连续控制中直接优于 CQL/IQL/TD3+BC 等深度 offline RL 方法。

## 8. 展望

对 ORL 研究者有用的启发：

1. 鲁棒离线学习不一定必须显式建模完整转移核；若能找到合适的 uncertainty penalty，model-free 更新也能做 worst-case correction。
2. “数据不足”和“环境偏差”应分开建模。把二者都塞进一个保守系数，通常会牺牲解释性和可调性。
3. Robust MDP 的不确定性集合选择很关键；$l_\alpha$、$\chi^2$、KL 等集合会导致不同的 $\kappa$ 形式和不同计算代价。
4. 理论上好的 tabular penalty 如何迁移到神经网络、连续动作和大规模 benchmark，是下一步真正难点。

局限与开放问题：

1. 主理论基于 tabular setting，函数逼近部分只是 case study。
2. $\kappa$ 依赖不确定性半径 $R$，实际任务中如何校准 $R$ 没有充分讨论。
3. 有些 penalty 仍需 $O(S)$ 操作，超大状态空间下需要近似或表示学习。
4. 保守项过强可能损失名义性能，论文实验主要展示鲁棒性收益，对 tradeoff 的系统调参分析还不够。

可能的后续研究：

1. **Deep double-pessimism CQL/IQL**：把 $\kappa$ 设计成可学习不确定性 head，研究连续动作 benchmark 中是否稳定。
2. **数据驱动半径估计**：从环境扰动样本、domain randomization 或 OPE 误差中估计 $R_{s,a}$，减少手工设定。
3. **distributionally robust sequence modeling**：把 double-pessimism penalty 接到 Decision Transformer 或 diffusion planner 的 trajectory value 中，处理长时域分布偏移。

## Links

- Paper page: [OpenReview](https://openreview.net/forum?id=QyVLJ7EnAC)
- PDF: [OpenReview PDF](https://openreview.net/pdf?id=QyVLJ7EnAC)
- arXiv: 未能从论文或公开来源确认独立 arXiv 链接
- Code: 未能从论文或公开来源确认官方代码链接
- First author profile: [Chi Zhang on OpenReview](https://openreview.net/profile?id=~Chi_Zhang61)

</article>
