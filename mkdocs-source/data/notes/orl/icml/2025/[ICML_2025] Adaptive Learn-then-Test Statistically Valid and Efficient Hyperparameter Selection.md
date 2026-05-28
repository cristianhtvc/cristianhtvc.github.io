---
title: "Adaptive Learn-then-Test: Statistically Valid and Efficient Hyperparameter Selection"
description: "aLTT 用 e-process 做自适应多重假设检验，在保持有限样本风险保证的同时减少昂贵测试轮数。"
tags:
  - Offline RL
  - Hyperparameter Selection
  - Statistical Guarantees
  - Multiple Testing
---

# Adaptive Learn-then-Test: Statistically Valid and Efficient Hyperparameter Selection

| 字段 | 内容 |
|---|---|
| Title | Adaptive Learn-then-Test: Statistically Valid and Efficient Hyperparameter Selection |
| Year | 2025 |
| Source | ICML 2025, PMLR 267:74018-74036 |
| Authors | Matteo Zecchin, Sangwoo Park, Osvaldo Simeone |
| Affiliations | King’s College London, Centre for Intelligent Information Processing Systems, Department of Engineering |
| Tags | Reliable Policy Selection, Offline RL Validation, e-process, FWER, FDR, Prompt Selection |

**一句话概括**：aLTT 把 LTT 的一次性 p-value 多重检验改造成基于 e-process 的顺序自适应检验，从而在在线策略验证、prompt 选择等昂贵测试场景中更快找到满足总体风险约束的超参数集合。

**我对这篇论文的定位**：它不是一篇新的 offline RL 算法论文，而是面向“离线训练后如何可靠选策略/选超参”的统计校准论文。对 ORL 研究者的价值在于：离线数据上的 OPE 或验证分数经常不可靠，真实部署前的在线测试又昂贵，aLTT 提供了一种带 FWER/FDR 有限样本保证的顺序测试框架。

## 1. 第一作者相关信息

第一作者 **Matteo Zecchin**，论文首页标注其机构为 King’s College London。其公开研究线与可靠 AI、通信/控制系统、统计学习和安全校准有关；本文把 e-process 与 sequential multiple testing 用于超参数选择。

| year | title | venue/source | topic tag | relation to this paper |
|---|---|---|---|---|
| 2025 | Adaptive Learn-then-Test | ICML / PMLR | Reliable selection | 本文 |
| 2024 | 论文中引用 Zecchin et al. 2024 的控制策略安全校准相关工作 | cited in paper | Safe calibration | 本文动机来源之一 |

公开来源足以确认本文、机构和代码库；未在本次阅读中系统核验 Matteo Zecchin 的完整发表列表。

## 2. 研究问题

给定离散候选超参数集合 $\Lambda=\{\lambda_1,\ldots,\lambda_N\}$，每个 $\lambda$ 对应一个 AI app、prompt、控制策略或 offline RL 策略。目标不是找经验风险最低者，而是找出满足总体风险阈值 $\alpha$ 的可靠集合：

$$
\Lambda_{\mathrm{rel}}=\{\lambda\in\Lambda:R(\lambda)\le \alpha\}.
$$

问题在于总体分布 $P_Z$ 不知道，只能通过 held-out data 或真实环境测试估计风险。传统 LTT 需要预先固定测试过程以避免 p-hacking，不能根据已收集证据自适应选择下一个要测的超参数。aLTT 要解决的是：在保持 FWER 或 FDR 控制的同时，允许顺序、自适应、可提前停止的测试。

## 3. 背景知识

LTT (learn-then-test) 将超参数选择视为多重假设检验。对每个候选 $\lambda_i$，原假设是“它不可靠”，即风险不满足阈值。通过控制 FWER 或 FDR，LTT 保证选出来的可靠集合中错误纳入坏超参的概率或比例受控。

FWER 控制更严格：要求“至少选错一个不可靠超参”的概率不超过 $\delta$。FDR 控制更宽松：要求被选集合中错误发现比例的期望不超过 $\delta$。在策略部署里，FWER 更像安全审查，FDR 更像允许少量错误的批量筛选。

e-value 与 e-process 是本文的核心。e-value 可理解为反驳原假设的证据量；e-process 是随时间更新且对任意停止时间都有效的证据过程。因此它天然支持“看到证据后决定下一步测谁、何时停”。

| term | plain Chinese meaning | role in this paper |
|---|---|---|
| population risk | 在真实分布上的期望风险 | 可靠性目标 |
| LTT | 先训练候选，再做固定多重检验 | 对比方法 |
| e-process | 任意停止时间仍有效的顺序证据 | aLTT 的技术核心 |
| acquisition policy | 每轮选择哪些超参数测试 | 允许自适应 |
| TPR | 找到真实可靠超参的比例 | 衡量测试是否有信息量 |

**为什么动机成立**：在 offline RL 中，策略训练可完全离线，但部署前往往仍要用少量在线 episode 验证策略是否可靠。若像 LTT 一样平均测试所有候选，会浪费大量真实交互；若随意自适应测试，又破坏统计保证。aLTT 正是在二者之间找平衡。

## 4. 问题分析

论文先指出 LTT 的限制：它依赖 p-value-based MHT，测试计划需要非自适应；这使得其在测试成本高或有安全风险时效率较低。然后作者引入 e-process 的 anytime-valid 性质：给定 e-process $E_{i,t}$，可构造 anytime-valid p-value：

$$
P_{i,t}=\frac{1}{\max_{\tau\le t}E_{i,\tau}}.
$$

含义：只要证据过程本身有效，即使测试轮数 $T$ 是根据历史证据动态决定的，最终得到的 $P_{i,T}$ 仍可用于 FWER 控制。

论文的诊断由 Figure 2/3 支撑：在 HalfCheetah 的 offline RL policy selection 中，aLTT 的 true positive rate 随测试轮数迅速上升，而 LTT 在结束前没有输出有效集合；当 $\epsilon$-greedy acquisition 更依赖证据时，TPR 从约 0.32/0.4 提升到约 0.85。

## 5. 思想与方法

aLTT 的 guiding principle 是：把“是否可靠”的统计检验写成可顺序累积的财富过程，然后让数据自适应地决定下一轮测哪个候选。

方法组件：

1. 为每个候选超参数维护 e-process $E_{i,t}$。
2. 每轮 acquisition policy $Q_t$ 根据已有证据选出待测子集 $I_t$。
3. 获得风险估计 $R(\lambda_i,Z_i^t)$ 后更新 e-process。
4. 若做 FWER 控制，则由 e-process 转成 anytime p-value，再用 FWER 选择规则。
5. 若做 FDR 控制，可直接用 eBH 等 e-value 规则。
6. 当可靠集合大小达到下限 $d$ 或达到最大轮数 $t_{\max}$ 时停止。

新意主要在把 LTT 与 e-process-based MHT 结合，使 adaptive acquisition 和 adaptive stopping 合法化。它借用了 e-process/eBH 的统计工具，但在超参数选择、offline RL 在线策略验证、prompt 工程上给出了完整应用。

## 6. 算法与伪代码

算法名：**Adaptive Learn-Then-Test (aLTT)**。

1. 输入候选集合 $\Lambda$、可靠性阈值 $\alpha$、错误容忍 $\delta$、acquisition policy、FWER/FDR 选择规则、betting strategy、$t_{\max}$ 和最小集合大小 $d$。
2. 初始化 $t=1$，每个候选的初始证据 $E_{i,0}=1$。
3. 在第 $t$ 轮，用 $Q_t(E_{t-1})$ 选择待测集合 $I_t$。
4. 对 $I_t$ 中候选获取风险估计。
5. 用

$$
E_{i,t}=
\begin{cases}
(1+\mu_i^t(\alpha-R(\lambda_i,Z_i^t)))E_{i,t-1}, & \lambda_i\in I_t,\\
E_{i,t-1}, & \lambda_i\notin I_t
\end{cases}
$$

更新证据。直觉是：当观测风险低于阈值 $\alpha$，支持“可靠”的财富增加；风险高则财富减少。

6. FWER 模式下先算 anytime p-values，再调用 FWER selection；FDR 模式下用 e-values 调用 FDR selection。
7. 若可靠集合足够大或达到预算，返回 $\hat{\Lambda}_{\mathrm{aLTT}}$。

定理 4.1 说明：在给定 FWER/FDR-controlling selection rule 时，aLTT 返回的最终集合满足对应的 $(\alpha,\delta)$ 控制。

## 7. 实验与消融

实验 1：online policy selection for offline RL。环境为 OpenAI Gym MuJoCo HalfCheetah，候选策略由 TD3+BC 训练得到，超参数 $\lambda$ 在 $[0.25,5]$ 网格上取 20 个候选；目标可靠性 $\alpha=0.57$，$\delta=0.1$，校准上限 $T=5000$。

结果：Figure 2 显示 aLTT 可在校准过程中持续输出可靠集合，而 LTT 只有到末尾才输出。证据驱动更强的 $\epsilon$-greedy acquisition 带来更高 TPR；FDR 控制下通常比 FWER 找到更多可靠候选，因为约束更宽松。Figure 3 显示经验 FWER/FDR 随目标 $\delta$ 增大而增大，但保持在目标线以下。

实验 2：automated prompt engineering。候选 prompts 由 Llama3.3 70B 生成，目标模型为 Llama3 8B Instruct，任务来自 instruction induction dataset。Figure 4/5 及附录结果显示，aLTT 能以更少测试找到更多可靠 prompts，且选出的最短 instruction 更短。

附录还包含无线资源分配实验，显示 aLTT 可用于多约束策略选择。对 ORL 最相关的是第 5.1 节，因为它提供了“离线训练策略的少量在线安全验证”模板。

## 8. 展望

研究启发：

1. Offline RL 的模型选择/策略选择不应只看离线验证分数，必须考虑真实部署风险。
2. 顺序统计工具可以把有限在线测试预算用在更有希望的候选上。
3. FWER 与 FDR 对应不同安全偏好，应根据应用风险选取。

局限与开放问题：

1. aLTT 选择的是离散候选集合，连续超参数仍需先离散化或由 HPO 生成候选。
2. 实验中的 online policy testing 仍需要真实交互，aLTT 只是减少而非消除成本。
3. 风险函数需归一化且可观测；复杂 RL 安全指标可能不易直接写成 $R(\lambda,Z)\in[0,1]$。
4. 对强相关候选、大规模候选池和非平稳环境的实践性能还可进一步研究。

后续想法：

1. 将 aLTT 与 OPE 结合，先用 OPE 排序候选，再用 e-process 进行在线确认。
2. 为 safe offline RL 设计多指标 aLTT，同时约束 reward shortfall、cost violation 和 tail risk。
3. 在 offline-to-online fine-tuning 中动态决定何时停止某个策略的危险验证。

## Links

- Paper page: [PMLR](https://proceedings.mlr.press/v267/zecchin25a.html)
- PDF: [PMLR PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/zecchin25a/zecchin25a.pdf)
- OpenReview: [OpenReview entry via PMLR](https://proceedings.mlr.press/v267/zecchin25a.html)
- Code: [kclip/aLTT](https://github.com/kclip/aLTT)
- Local PDF: `[ICML_2025] Adaptive Learn-then-Test Statistically Valid and Efficient Hyperparameter Selection.pdf`
