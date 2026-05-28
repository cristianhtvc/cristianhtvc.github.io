---
title: "Model-based RL as a Minimalist Approach to Horizon-Free and Second-Order Bounds"
description: "证明最标准的 MLE 学模型加乐观/悲观规划，已经足以在在线和离线 RL 中得到近似 horizon-free 与 second-order 的理论界。"
tags:
  - Model-based RL
  - RL Theory
  - Offline RL
  - Horizon-free Bounds
  - Second-order Bounds
---

# Model-based RL as a Minimalist Approach to Horizon-Free and Second-Order Bounds

<div class="paper-hero">
<p class="paper-meta">Zhiyong Wang, Dongruo Zhou, John C. S. Lui, Wen Sun · ICLR 2025 · ICLR / 2025</p>
<div class="tag-row"><span>Model-based RL</span><span>RL Theory</span><span>Offline RL</span><span>Horizon-free Bounds</span><span>Second-order Bounds</span></div>
</div>

<article class="note-body" markdown>

| 字段 | 内容 |
|---|---|
| Title | Model-based RL as a Minimalist Approach to Horizon-Free and Second-Order Bounds |
| Year | 2025；arXiv 首版 2024-08-16 |
| Source | ICLR 2025 / Published as a conference paper |
| Authors | Zhiyong Wang, Dongruo Zhou, John C. S. Lui, Wen Sun |
| Affiliations | The Chinese University of Hong Kong; Indiana University Bloomington; Cornell University |
| Tags | Model-based RL; RL Theory; Horizon-free; Second-order Bound; Optimism; Pessimism; MLE |
| Local PDF | `[ICLR_2025] Model-based RL as a Minimalist Approach to Horizon-Free and Second-Order Bounds.pdf` |

**一句话概括**：本文证明在轨迹总奖励归一化、转移时间齐次和模型可实现等条件下，最朴素的 MLE 学转移模型加在线乐观规划/离线悲观规划，就能获得近乎无 horizon 多项式依赖且依赖回报方差的 second-order regret 与 sample complexity bounds。

**我对这篇论文的定位**：这是一篇 RL 理论论文，不是提出新实用算法。它的贡献在于重新分析标准 model-based RL：过去 horizon-free 和 second-order bounds 往往依赖分布式 RL、方差加权回归或专门置信区间设计；本文证明普通 MLE version space + optimism/pessimism 已经足够。对 offline RL 读者而言，它重要在于给 CPPO-LR 式悲观 model-based offline RL 一个更精细、更少 horizon 依赖的保证。

## 1. 第一作者相关信息

**Zhiyong Wang** 在论文发表时标注为 The Chinese University of Hong Kong，邮箱为 `zywang21@cse.cuhk.edu.hk`。他的研究方向集中在 RL theory，尤其是 horizon-free、first/second-order regret bound、distributional RL 和 model-based RL 的统计复杂度。

从论文参考文献和作者署名可以看出，他的研究轨迹围绕一个连续问题展开：RL 的样本复杂度是否真的必须因为长 horizon 而变差？以及能否给出依赖具体问题实例方差/最优值的更精细界？本文把这一问题从 model-free/distributional 分析推进到最标准的 model-based MLE 框架。

| 年份 | 标题 | Venue/Source | 主题标签 | 与本文关系 |
|---|---|---|---|---|
| 2023 | First-order regret bounds in reinforcement learning | 论文引用中的前序工作 | First-order bound | 研究 regret 依赖最优值而非 worst-case 的精细界 |
| 2024 | Distributional RL achieves first-order and second-order regret bounds | AISTATS 2024 / 论文引用 | Distributional RL, Second-order | 用 distributional RL 得到一阶/二阶界，本文说明 MLE-MBRL 也可做到 |
| 2025 | Model-based RL as a Minimalist Approach to Horizon-Free and Second-Order Bounds | ICLR 2025 | MLE MBRL, Horizon-free | 本文，用极简 MBRL 框架统一在线和离线结果 |

主要研究方向可以概括为：**用更精细的概率分析说明 RL 的统计难度可随实例性质缩放，而不必总是按 worst-case horizon 多项式增长**。来源包括论文 PDF、arXiv 元数据和论文参考文献；公开个人主页信息未作为强断言使用。

## 2. 研究问题

本文研究的具体问题是：**标准 model-based RL 是否已经足够获得 horizon-free 和 second-order guarantees，还是必须设计更复杂的算法？**

在线设置中，agent 交互 $K$ 个 episode，目标是控制 regret：

$$
\sum_{k=0}^{K-1}(V^{\pi^\star}-V^{\pi_k}).
$$

离线设置中，agent 只有固定的 $K$ 条轨迹，目标是学到策略 $\hat{\pi}$，使其与 comparator policy $\pi^\star$ 的 gap 小：

$$
V^{\pi^\star}-V^{\hat{\pi}}.
$$

论文假设有限时域 time-homogeneous MDP，转移 $P^\star$ 属于模型类 $\mathcal{P}$，奖励已知且任意轨迹总奖励归一化到 $[0,1]$。这点非常关键：value 范围不随 $H$ 增长，horizon-free 分析才可能成立。在线算法使用 optimism，离线算法使用 pessimism；二者都先用 MLE 构造 version space。

## 3. 背景知识

**Horizon-free bounds** 指 regret 或 sample complexity 对 horizon $H$ 没有显式多项式依赖。传统 RL 界常有 $H$、$H^2$ 或更高次依赖，让长时域问题看起来统计上更难。本文追问：在合适归一化下，长 horizon 规划本身是否真是统计瓶颈？

**Second-order bounds** 是 instance-dependent bounds，主项依赖策略回报方差，而不是简单依赖 episode 数：

$$
\tilde{O}\left(\sqrt{\sum_k \mathrm{VaR}^{\pi_k}\cdot \mathrm{complexity}}+\mathrm{complexity}\right).
$$

如果环境近似确定、回报方差低，second-order bound 会显著小于 worst-case $\sqrt{K}$ 型界。它还可推出 first-order bound，即依赖 $V^{\pi^\star}$。

**MLE version space** 是 model-based RL 的标准工具：在已收集 transition 上最大化似然，保留所有 log-likelihood 距最优值不超过阈值 $\beta$ 的模型。这个集合以高概率包含真实模型 $P^\star$。

**乐观/悲观规划**：在线学习中在置信集合里选最乐观模型和策略来探索；离线学习中对每个策略取置信集合内最悲观模型，再选悲观值最高的策略。

**Eluder dimension 与 concentrability**：在线部分用 $\ell_1$ Eluder dimension 衡量模型类下未探索区域的复杂度；离线部分用单策略 concentrability 衡量数据对 comparator policy 的覆盖。

| 术语 | 直观含义 | 在本文中的作用 |
|---|---|---|
| MLE | 最大似然学习转移模型 | 构造 version space |
| Version space | 与数据相容的一组模型 | 乐观/悲观规划的模型集合 |
| O-MBRL | 在线乐观 model-based RL | 获得 regret bound |
| CPPO-LR | 离线悲观 model-based RL | 获得 performance gap bound |
| Triangular discrimination | 与 squared Hellinger 等价的散度 | 将均值差绑定到方差与模型误差 |
| $\ell_1$ Eluder dimension | 函数类外推复杂度 | 在线探索复杂度 |
| Single-policy concentrability | 离线数据覆盖 comparator 的程度 | 离线样本复杂度 |

本文动机成立，是因为许多已有 horizon-free/second-order 结果依赖更复杂的算法，而 MBRL 的最朴素形式在实践和理论中都常见。如果只需更细分析即可获得这些界，那么算法设计的重心会发生变化。

## 4. 问题分析

论文的诊断是：朴素分析会把 simulation lemma 中每个时间步的误差直接相加，从而付出 $H$ 甚至多项式 $H$ 代价；要去掉 horizon 依赖，必须更精细地利用回报方差和 MLE 的 Hellinger 型泛化界。

关键技术是 triangular discrimination 与 mean-to-variance lemma。粗略地说，两个分布下函数期望的差可以由“函数方差 × 分布差异”控制，而不是由函数最大范围直接控制。这样就能让界依赖 $\mathrm{VaR}^{\pi}$，并避免每个时间步都付一个最坏 $H$。

在线部分的证明路径是：

1. MLE 保证真实模型 $P^\star$ 在所有 version space 中。
2. 乐观规划保证选出的模型/策略价值至少不低于真实最优。
3. 用 Eluder dimension 将训练分布上的 Hellinger 误差转移到测试轨迹分布。
4. 用 variance conversion lemma 把乐观模型下的方差转回真实模型下的回报方差。

离线部分的诊断类似，但 optimism 换成 pessimism，探索复杂度换成覆盖条件。Definition 3 的单策略覆盖系数 $C^{\pi^\star}$ 描述离线数据是否覆盖 comparator policy 会访问的状态动作。

## 5. 思想与方法

本文的方法思想可以总结为：**算法保持极简，分析做精细**。

在线算法 O-MBRL：

$$
\hat{\mathcal{P}}_k=
\left\{
P\in\mathcal{P}:
\sum_{(s,a,s')\in D}\log P(s'\mid s,a)
\ge
\max_{\tilde{P}\in\mathcal{P}}\sum_{(s,a,s')\in D}\log \tilde{P}(s'\mid s,a)-\beta
\right\}.
$$

然后选

$$
(\pi_k,\hat{P}_k)=\arg\max_{\pi\in\Pi,P\in\hat{\mathcal{P}}_k}V^\pi_{0;P}(s_0).
$$

离线算法 CPPO-LR：

$$
\hat{\pi}=\arg\max_{\pi\in\Pi}\min_{P\in\hat{\mathcal{P}}}V^\pi_{0;P}(s_0).
$$

新意不在算法公式，而在说明这些标准对象已经足够获得：

1. 在线 finite model class 的 near horizon-free second-order regret。
2. 离线 finite model class 的完全 horizon-free performance gap。
3. deterministic transition 下的更快 rate。
4. infinite model class 下用 bracketing number 替代 $|\mathcal{P}|$。

从 ORL 角度看，悲观规划的作用是避免离线数据覆盖不足时选择高估模型；本文进一步证明，如果覆盖 comparator policy，gap 可随 $\mathrm{VaR}^{\pi^\star}$ 变小。

## 6. 算法与伪代码

**Algorithm 1：Optimistic Model-based RL (O-MBRL)**。

1. 输入模型类 $\mathcal{P}$、置信参数 $\delta$、阈值 $\beta$。
2. 初始化策略 $\pi^0$，数据集 $D=\emptyset$。
3. 第 $k$ 个 episode，执行当前策略 $\pi^k$ 收集轨迹并拆成 $(s,a,s')$ 加入 $D$。
4. 用 MLE likelihood threshold 构造 version space $\hat{\mathcal{P}}_k$。
5. 在 $\hat{\mathcal{P}}_k$ 中做 optimistic planning，选择最大 $V^\pi_{0;P}(s_0)$ 的 $(\pi_k,\hat{P}_k)$。
6. 重复直到 $K$ 个 episode 完成。

**Algorithm 2：CPPO-LR / Constrained Pessimistic Policy Optimization with Likelihood-Ratio constraints**。

1. 输入离线 transition 数据 $D=\{(s,a,s')\}$、模型类 $\mathcal{P}$、策略类 $\Pi$、阈值 $\beta$。
2. 用离线数据构造 version space $\hat{\mathcal{P}}$。
3. 输出

$$
\hat{\pi}=\arg\max_{\pi\in\Pi}\min_{P\in\hat{\mathcal{P}}}V^\pi_{0;P}(s_0).
$$

核心定理：

**Theorem 1（在线 finite $\mathcal{P}$）** 给出

$$
\sum_{k=0}^{K-1}(V^{\pi^\star}-V^{\pi_k})
\le
\tilde{O}\left(
\sqrt{\sum_k \mathrm{VaR}^{\pi_k}\, d_{RL}}
+d_{RL}
\right),
$$

其中 $d_{RL}=D_E^1(\Psi,S\times A,1/KH)$，省略了对数项。该界没有 $H$ 的显式多项式依赖。

**Theorem 2（离线 finite $\mathcal{P}$）** 给出

$$
V^{\pi^\star}-V^{\hat{\pi}}
\le
O\left(
\sqrt{\frac{C^{\pi^\star}\mathrm{VaR}^{\pi^\star}\log(|\mathcal{P}|/\delta)}{K}}
+
\frac{C^{\pi^\star}\log(|\mathcal{P}|/\delta)}{K}
\right).
$$

这在 finite model class 离线情形下完全没有 $\log H$ 依赖。若真实转移确定，$\mathrm{VaR}$ 项消失，可得到 $O(C^{\pi^\star}\log(|\mathcal{P}|/\delta)/K)$ 的更快 rate。

## 7. 实验与消融

本文没有经验实验，Section 4/5 是理论结果与 proof sketch。这里的“验证”来自定理和与已有 bounds 的比较。

**在线结果**：Theorem 1 说明 O-MBRL 在 finite model class 下得到 near horizon-free second-order regret；Corollary 1 推出 first-order regret；Corollary 2 在 deterministic transition 下得到仅对 $K$ poly-log 的 regret；Corollary 3 用 bracketing number 扩展到 infinite class；Example 1 给 tabular MDP 的具体形式。

**离线结果**：Theorem 2 分析 Uehara & Sun (2021) 的 CPPO-LR，但给出更强界：相比原结果减少 horizon 依赖，并引入 $\mathrm{VaR}^{\pi^\star}$。Corollary 4 在 deterministic transition 下得到 $1/K$ rate；Corollary 5 处理 infinite class；Example 3 给 tabular MDP 显式 gap。

**与 prior work 的差异**：Huang et al. (2024) 等工作需要方差估计和方差加权回归；Wang et al. (2023/2024a) 用 distributional RL 得到一阶/二阶界。本文说明标准 MLE-MBRL 也能做到类似统计性质，算法更少改造。

理论结论的适用边界也要写清：奖励需要轨迹级归一化；转移是 time-homogeneous；realizability $P^\star\in\mathcal{P}$；离线部分需要单策略覆盖；许多结果对计算 oracle 默认可解，实际复杂策略/模型类下的规划计算没有解决。

## 8. 展望

对 ORL 研究者的启发：

1. Offline RL 的悲观 model-based 方法不只是工程保守技巧，也能得到非常精细的 horizon-free theory。
2. 长 horizon 的统计难度在合适归一化和模型设定下可能没有直觉中那么可怕。
3. Second-order bound 对安全、确定性或低方差任务尤其有意义，因为 worst-case 界会过度悲观。
4. 一些“新算法设计”可能可以被“旧算法新分析”替代。

局限与开放问题：

1. Realizability 和精确规划 oracle 假设很强，深度 MBRL 中通常不成立。
2. 轨迹总奖励归一化到 $[0,1]$ 是 horizon-free 的核心，但和许多实际 benchmark 的 step reward 设定需要转换。
3. Offline 覆盖条件是单策略 comparator 覆盖，不是任意最优策略都自动满足。
4. Infinite class 结果仍有 $\log H$ 或 bracketing complexity，完全 horizon-free 只在有限类离线结果中最干净。

可能的后续研究：

1. **Approximate planning version**：分析 planning oracle 近似误差对 second-order/horizon-free bound 的影响。
2. **Misspecified model class**：放宽 $P^\star\in\mathcal{P}$，加入模型偏差项，贴近深度 world model。
3. **Practical offline MBRL diagnostics**：把 $\mathrm{VaR}^{\pi}$ 和 concentrability 估计转成 benchmark 上的可观测诊断指标。

## Links

- arXiv: https://arxiv.org/abs/2408.08994
- PDF: https://arxiv.org/pdf/2408.08994
- Paper page: ICLR 2025 / OpenReview 页面可由标题检索；本地 PDF 标注为 Published as a conference paper at ICLR 2025
- Code: 理论论文，未能从论文和公开来源确认官方代码链接
- Source notes: arXiv API confirms title, authors, abstract and first upload date; local PDF was used for Algorithms 1-2, Theorem 1/2, corollaries, assumptions and proof sketch.

</article>
