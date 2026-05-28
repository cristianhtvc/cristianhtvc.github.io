---
title: "Offline RL with Smooth OOD Generalization in Convex Hull and its Neighborhood"
description: "用 CHN 安全区域和 Smooth Bellman Operator 缓解离线 RL 中对 OOD 动作的过度保守。"
tags:
  - Offline RL
  - OOD Generalization
  - Q Function
  - Smooth Bellman Operator
---

# Offline RL with Smooth OOD Generalization in Convex Hull and its Neighborhood

| 字段 | 内容 |
|---|---|
| Title | Offline RL with Smooth OOD Generalization in Convex Hull and its Neighborhood |
| Year | ICLR 2025；arXiv v1 提交于 2025-06-10 |
| Source | ICLR 2025 Poster；OpenReview 发布于 2025-01-22，最终修改于 2025-03-01 |
| Authors | Qingmao Yao, Zhichao Lei, Tianyuan Chen, Ziyue Yuan, Xuefan Chen, Jianxiang Liu, Faguo Wu, Xiao Zhang |
| Affiliations | Beihang University; Zhongguancun Laboratory; Hangzhou International Innovation Institute of Beihang University |
| Tags | Offline RL; OOD Action; CHN; Smooth Bellman Operator; TD3+BC |

**一句话概括**：SQOG 把离线 RL 中的 OOD 动作划分为“CHN 内可平滑泛化”和“CHN 外不可靠”两类，并用邻近 in-sample Q 值构造 Smooth Bellman Operator，从而缓解传统保守方法的 over-constraint。

**我对这篇论文的定位**：本文属于 model-free offline RL 中 critic 泛化问题的工作，特别针对策略约束和价值惩罚方法把 OOD 动作一概压低导致的 Q 值低估。它并不否认 OOD 风险，而是提出一个几何边界 CHN，允许在数据凸包及邻域内做有限度泛化。相较 CQL/IQL/EDAC 的“避免或惩罚 OOD”，它更像是“局部插值式的 OOD 价值修复”。

## 1. 第一作者相关信息

Qingmao Yao 的 OpenReview 资料显示其 2021-2025 年为北京航空航天大学本科生，2025 年起为北京大学博士生，研究兴趣包含 Reinforcement Learning、Generative AI 和 Autobidding。本文是其 ICLR 2025 Poster 工作。

| Year | Title | Venue/source | Topic tag | Relation to this paper |
|---|---|---|---|---|
| 2025 | Offline RL with Smooth OOD Generalization in Convex Hull and its Neighborhood | ICLR 2025 Poster / arXiv | Offline RL, OOD generalization | 本文主体工作 |
| 2026 | Iterative Scarcity-Guided Exploration: Bootstrapping Generative Auto-bidding from Narrow Support | ICLR 2026 Workshop AIMS / OpenReview | Generative auto-bidding, narrow support | 延续“窄支撑/数据覆盖不足下如何探索”的兴趣 |

第一作者可公开核验的论文记录较少，因此不要过度推断其完整研究谱系。团队中的 Tianyuan Chen 个人主页显示其关注 offline RL 和 uncertainty quantification，与本文的 Q 泛化/不确定性边界主题一致。

## 2. 研究问题

离线 RL 的经典问题是 distribution shift：目标策略可能输出数据集中没有的动作，critic 对这些 OOD 动作的 bootstrapping 容易产生外推误差。多数算法因此采取保守策略：限制 actor 靠近行为策略，或者直接惩罚 OOD 动作的 Q 值。

本文指出一个相反但常见的失败模式：保守过头。所谓 over-constraint，是指 OOD 区域的 Q 值被系统性压低，导致策略无法利用数据支撑附近的潜在高价值动作。核心问题因此变成：能否只在安全的 OOD 区域内提升 Q 函数泛化，而不在完全未知区域任意外推？

论文设定是标准离线 MDP，数据集 $D=\{(s,a,r,s',d)\}$ 由行为策略生成；方法是 model-free actor-critic，实践上基于 TD3+BC，并通过额外 critic loss 实现 OOD 平滑泛化。

## 3. 背景知识

**离线 RL 与 OOD action**：在线 RL 可以试错，离线 RL 只能使用固定数据。若 actor 输出动作 $a'$ 在状态 $s'$ 的数据支持之外，critic 对 $Q(s',a')$ 的估计可能没有真实样本支撑。

**保守性和 over-constraint**：CQL 等方法通过降低 OOD Q 值避免过估计；IQL 等 in-sample 方法甚至只在数据动作上做值学习。这能提高安全性，但也可能把“数据附近、实际可行且高价值”的动作错杀。

**凸包与 CHN**：给定数据中的 state-action 点，凸包 $\mathrm{Conv}(D)$ 包含这些点的所有凸组合；CHN 是凸包加外部邻域：

$$
\mathrm{CHN}(D)=\mathrm{Conv}(D)\cup N(\mathrm{Conv}(D)).
$$

直观地说，凸包内是插值区域，邻域允许少量外推；CHN 外则是距离数据支撑太远的区域。论文 Proposition 1、2 分别给出 CHN 内 Q 值差异可控和 Q 函数一致连续的安全保证。

**Smooth Bellman Operator**：SBO 的作用是用邻居数据动作的 Q 值来更新 CHN 内 OOD 动作，使 OOD Q 值朝局部 in-sample 估计靠近。

| 术语 | 直白含义 | 在本文中的作用 |
|---|---|---|
| OOD action | 数据中没有或概率极低的动作 | 离线 critic 的主要风险来源 |
| Over-constraint | 把 OOD 价值压得过低 | 本文要缓解的失败模式 |
| CHN | 数据凸包及其邻域 | 划定可安全泛化区域 |
| SBO | 平滑贝尔曼算子 | 用邻居 in-sample Q 修复 OOD Q |
| SQOG | 实用算法 | TD3+BC + OOD generalization loss |

## 4. 问题分析

论文先用 Figure 1 做 sanity check：在两个关键状态上比较 TD3+BC 与 SQOG 的 critic Q 值和 Monte Carlo true Q。TD3+BC 的 Q 值被限制在很窄区间，无法贴近真实高价值区域；SQOG 更接近 MC 估计，说明它不是单纯抬高所有 OOD，而是在 CHN 内做平滑修正。

理论分析围绕 CHN 和 SBO 展开。Proposition 1 依赖 NTK regime，说明 CHN 内点到数据投影的 Q 差异可被距离控制；Proposition 2 说明连续 Q 函数在紧致 CHN 上一致连续。Theorem 1 论证 empirical Bellman operator 在策略约束前提下能近似真实 Bellman operator。Theorem 2 表明 SBO 对 in-sample evaluation 的影响很小；Theorem 3 表明对 CHN 内 OOD evaluation，SBO 可以缓解低估或高估；Proposition 4 给出 $\gamma$-contraction 收敛性质。

## 5. 思想与方法

SQOG 的指导原则是“局部安全泛化”：不要把所有 OOD 都当作危险，也不要无约束外推。具体机制是：

1. 通过 CHN 区分安全 OOD 和危险 OOD。
2. 对 CHN 内 OOD 动作，找到邻近 in-sample state-action。
3. 用邻居 Q 值构造平滑目标，让 OOD Q 值逐步靠近局部真实趋势。
4. 维持 TD3+BC 的 actor 约束，使策略输出不至于跑到 CHN 外。

方法真正新的部分是 SBO/LOG，而 TD3+BC 骨架、双 Q、target network、policy delay 等训练套路是借用成熟 actor-critic 实现。证据强度上，CHN 安全性和 SBO 收敛有理论分析；D4RL 表现和 Q-value sanity check 是实践证据；但 CHN 的高维几何近似仍带启发式成分。

## 6. 算法与伪代码

算法名称：Smooth Q-function OOD Generalization，简称 SQOG。

1. 初始化双 Q 网络、actor、target networks，并保留 TD3+BC 风格 actor loss。
2. 从离线数据中采样 batch $(s,a,r,s')$。
3. 用 target actor 生成下一步候选动作 $a'=\pi_{\bar{\phi}}(s')+\epsilon$。
4. 判断 $a'$ 是否位于 CHN 内；实践中用局部近邻距离近似 CHN 判断。
5. Critic 的第一部分执行标准 TD target：

$$
y=r+\gamma \min_i Q_{\bar{\theta}_i}(s',a').
$$

6. Critic 的第二部分加入 OOD generalization loss：对 CHN 内 OOD 动作，把 $Q(s',a')$ 拉向其邻近 in-sample 动作的加权 Q 值。
7. Actor 更新仍最大化 Q 并加入 BC 正则。
8. 推理时直接使用 actor，不需要额外做 KNN/SBO 查询。

论文 Algorithm 1 给出 SQOG 主流程，Eq. (13)(14) 对应实践 critic loss；Appendix D 的 Table 9 说明理论 SBO 与实践 loss 的对应关系。

## 7. 实验与消融

实验覆盖 D4RL MuJoCo、Maze2d 和 Adroit。Table 1 的 MuJoCo 结果显示 SQOG 总体优于 BC、TD3+BC、CQL、IQL、DOGE、MCQ 等基线；论文特别强调 SQOG 相比 TD3+BC 增加很小计算成本，却显著提升 normalized score。Appendix Table 2 进一步展示 Maze2d 和 Adroit 上的结果，SQOG 在多类数据集上保持较强表现。

消融包括：Figure 4 对 $\beta$ 和噪声类型的敏感性分析；Appendix Table 3-5 测试不同 $\alpha$、Gaussian noise scale/clip 和噪声分布；Appendix Table 10 把 SBO 插入 BRAC，显示 BRAC+SBO 明显优于 BRAC，支持“SBO 是 policy-constraint 方法的插件”这一说法。

实验能证明的是：在 D4RL 任务中，SBO 风格的局部 OOD Q 修正确实能缓解部分 conservative critic 的价值低估，并提升策略表现。它尚未完全证明的是：CHN 在高维状态-动作空间中总能可靠刻画安全泛化边界。

## 8. 展望

研究启发：第一，OOD 不应只有“惩罚/避免”一种处理方式，局部插值可能是第三条路。第二，critic 泛化能力本身值得被显式设计，而不是只靠 actor 约束兜底。第三，KNN/局部几何工具在 offline RL 中仍然有生命力。第四，SBO 插件化思路可尝试接入 BRAC、diffusion policy、IQL policy extraction 等框架。

局限：CHN 的实际判断是近似且依赖阈值；理论用到 NTK 和连续性假设，深度网络实践中并不严格满足；主要处理 action OOD，对 state OOD 帮助有限；高维动作空间的近邻质量可能下降；实验仍以 D4RL 为主。

后续方向：1. 用 VAE/normalizing flow/score model 学习数据支撑流形，替代几何凸包近似。2. 将 SBO 融入 diffusion actor 或 flow matching policy 的去噪过程。3. 同时建模 state-action CHN，处理更完整的分布偏移。

## Links

- Paper page: https://openreview.net/forum?id=eY5JNJE56i
- ICLR poster: https://iclr.cc/virtual/2025/poster/28920
- arXiv: https://arxiv.org/abs/2506.08417
- Code: https://github.com/yqpqry/SQOG
- First author OpenReview: https://openreview.net/profile?id=~Qingmao_Yao1

