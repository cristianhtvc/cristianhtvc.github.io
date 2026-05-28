---
title: "Cross-Domain Offline Policy Adaptation with Optimal Transport and Dataset Constraint"
description: "OTDF 用最优传输筛选源域转移，并用目标域数据集约束防止策略偏向错误动力学。"
tags:
  - Offline RL
  - Domain Adaptation
  - Optimal Transport
  - Dynamics Shift
  - Dataset Constraint
---

# Cross-Domain Offline Policy Adaptation with Optimal Transport and Dataset Constraint

<div class="paper-hero">
<p class="paper-meta">Jiafei Lyu, Mengbei Yan, Zhongjian Qiao, Runze Liu, Xiaoteng Ma, Deheng Ye, Jingwen Yang, Zongqing Lu, Xiu Li · ICLR 2025 · ICLR / 2025</p>
<div class="tag-row"><span>Offline RL</span><span>Domain Adaptation</span><span>Optimal Transport</span><span>Dynamics Shift</span><span>Dataset Constraint</span></div>
</div>

<article class="note-body" markdown>

| 字段 | 内容 |
|---|---|
| Title | Cross-Domain Offline Policy Adaptation with Optimal Transport and Dataset Constraint |
| Year | 2025 |
| Source | ICLR 2025 Conference Paper |
| Authors | Jiafei Lyu, Mengbei Yan, Zhongjian Qiao, Runze Liu, Xiaoteng Ma, Deheng Ye, Jingwen Yang, Zongqing Lu, Xiu Li |
| Affiliations | Tsinghua University; Tencent; Peking University; BAAI，见论文首页 |
| Tags | Cross-domain offline RL; optimal transport; data filtering; support constraint |

**一句话概括：** 论文提出 OTDF：先用 optimal transport 对齐源域和目标域 transition，过滤与目标动力学不相似的源域样本，再加入目标域 dataset constraint 防止学到的策略偏向源域行为分布。

**我对这篇论文的定位：** 这是 cross-domain offline RL / offline policy adaptation 的数据层方法。它不训练 domain classifier 或 contrastive dynamics encoder，而是用 OT 这种非神经网络的分布匹配工具，在目标域样本很少时做源域数据选择。

## 1. 第一作者相关信息

第一作者 **Jiafei Lyu**。论文首页显示其来自 Tsinghua Shenzhen International Graduate School，且工作期间有 Tencent internship 标注。公开主页中可核验到本文 ICLR 2025 记录和项目/代码链接；本次未完整整理其全部近期论文。

| year | title | venue/source | topic tag | relation to this paper |
|---|---|---|---|---|
| 2025 | Cross-Domain Offline Policy Adaptation with Optimal Transport and Dataset Constraint | ICLR 2025 | Cross-domain Offline RL | 本文 |

从本文和作者列表看，该工作延续了 domain adaptation / off-dynamics RL 方向：在目标域数据很少、源域数据丰富但动力学有偏差时，如何安全地借用源域数据。

## 2. 研究问题

问题设定是 **cross-domain offline RL**：有源域离线数据 $D_{\mathrm{src}}$ 和少量目标域离线数据 $D_{\mathrm{tar}}$。两个域状态/动作空间一致，奖励函数一致，但转移动力学不同：

$$
M_{\mathrm{src}}=(S,A,P_{\mathrm{src}},r,\gamma),\quad
M_{\mathrm{tar}}=(S,A,P_{\mathrm{tar}},r,\gamma).
$$

目标是在目标域获得高性能策略。直接合并源域和目标域数据会有风险，因为源域 transition 可能来自不同动力学；只用目标域数据又太少。论文问的是：如何在有限目标域数据下可靠地选择能帮助目标域学习的源域 transition？

## 3. 背景知识

| term | plain Chinese meaning | role in this paper |
|---|---|---|
| Cross-domain offline RL | 两个域都只有离线数据，用源域帮助目标域 | 主问题 |
| Dynamics mismatch | 同一动作在两个域导致不同下一状态 | 源域数据有害的原因 |
| Optimal transport | 计算两个离散分布之间的最小搬运代价 | 用于对齐源/目标 transition |
| Wasserstein distance | OT 诱导的分布距离 | 衡量数据分布差异 |
| Dataset constraint | 策略动作应落在目标域数据支持内 | 防止策略被源域带偏 |
| IQL backbone | 一种 in-sample offline RL 方法 | 论文实际实现的底座 |

offline RL 本来就依赖数据覆盖。cross-domain 情况下，覆盖问题变成两层：源域数据多但动力学可能错，目标域数据少但可信。OTDF 的思路是把 transition 写成 $u=(s,a,s')$，比较源域 $u$ 与目标域 $u'$ 的相似度；越像目标域 transition 的源域样本越应被保留。

## 4. 问题分析

论文从 performance bound 出发，说明目标域真实性能和经验源域性能之间的差异受三类因素影响：动力学 mismatch、策略分布偏差、以及数据集经验 MDP 的偏差。Theorem 3.1 给出这一分解后，作者把方法目标落到两个可控项：

1. 降低源域 transition 与目标域 transition 的动力学差异。
2. 限制最终策略不要偏离目标域数据支持。

OT 对齐的核心是求源域和目标域 transition 分布的 coupling：

$$
W(\mu_s,\mu_t)=\min_{\Gamma}\sum_{i,j} C(u_i,u'_j)\Gamma_{ij}.
$$

随后对每个源域样本计算它和目标域数据的 OT deviation $d(u_i)$。论文 Theorem 3.2 说明在 cost 有界时，该量能作为 total variation discrepancy 的代理。

## 5. 思想与方法

OTDF 包含两个机制。

**Selective source data sharing**：用 Sinkhorn 求解 entropy-regularized OT，得到每个源域 transition 与目标域数据的相似度。训练 Q 时只保留 batch 中 deviation 排名前 $\kappa\%$ 的源域样本，并按 $\exp(\alpha d)$ 加权。

**Target-domain dataset regularization**：训练一个目标域行为策略近似器 $\hat{\pi}_{\mathrm{tar}}$，在 policy objective 中加入约束，使 learned policy 的动作更可能落在目标域数据支持内：

$$
\hat{L}_\pi=L_\pi-\beta\,\mathbb{E}_{s\sim D_{\mathrm{src}}\cup D_{\mathrm{tar}}}
\log \hat{\pi}_{\mathrm{tar}}(\pi(\cdot|s)|s).
$$

直观上，第一项决定“哪些源域数据可信”，第二项决定“策略最后应更像目标域可执行动作”。

## 6. 算法与伪代码

算法名：**Optimal Transport Data Filtering (OTDF)**，实际版本是 OTDF+IQL。

1. 输入源域数据 $D_{\mathrm{src}}$、目标域数据 $D_{\mathrm{tar}}$、筛选比例 $\kappa$。
2. 将 transition 表示为 $u=(s,a,s')$。
3. 用 Sinkhorn/OTT-JAX 求解源域和目标域 transition 的 OT coupling。
4. 为每个源域样本计算 deviation $d(u)$，并写回源域数据。
5. 训练目标域 CVAE/behavior model $\hat{\pi}_{\mathrm{tar}}$。
6. 每次更新时采样一半源域、一半目标域 batch。
7. 对源域 batch 按 $d$ 排序，仅保留 top $\kappa\%$，并用归一化后的 $\exp(d)$ 加权 critic loss。
8. 用 IQL 更新 value/critic。
9. 更新 policy 时加入目标域 dataset regularization。

论文强调 OT 可以预先计算：用 OTT-JAX 在 1M 源域 transition 和约 5000 目标域 transition 下，约 5 分钟内完成 deviation 计算。

## 7. 实验与消融

实验构造三类 dynamics shift：gravity shift、kinematic shift、morphology shift；环境包括 halfcheetah、hopper、walker2d、ant。源域用 D4RL MuJoCo v2，目标域在修改动力学环境中采集约 5000 transitions。源/目标数据质量包括 medium、medium-replay、medium-expert、expert 等组合。

对比方法包括 IQL、DARA、BOSA、SRPO、IGDF 等。结果摘要：

- Table 1/2 主结果显示，OTDF 在多种 shift 和数据质量组合上显著提升目标域 normalized score。
- Figure 3 的 $\beta$ 参数研究表明 dataset regularization 太弱会让策略偏向源域，太强会压制源域知识，论文多采用 0.5。
- Appendix Table 7 在 morphology shift + expert source 下显示 OTDF 总分 393.0，高于 IQL/DARA/BOSA/SRPO/IGDF。
- Figure 5 消融说明仅筛选不加源域样本权重会变差，说明 adaptive weighting 不只是装饰项。

局限也很明确：源/目标状态空间或动作空间不同则 OTDF 不适用；实验仍是模拟环境，真实机器人迁移尚未验证。

## 8. 展望

启发：

- cross-domain offline RL 不应简单合并数据；需要 transition-level 的动力学相似性判断。
- 当目标域数据很少时，非参数/弱参数工具如 OT 可能比 domain classifier 更稳。
- policy regularization 应指向目标域支持，而不是源+目标混合支持。

局限：

- OT cost 的设计会强烈影响筛选效果。
- 状态/动作空间必须一致。
- 大规模高维视觉数据下，直接在原始 transition 空间做 OT 可能不可行。

后续方向：

- 在 learned representation 上做 OTDF，扩展到视觉/语言条件任务。
- 将 OTDF 与 model-based dynamics adaptation 结合。
- 研究在线少量 fine-tuning 时是否可动态更新 OT coupling。

## Links

- Project/page: https://z0ngqing.github.io/publication/otdf/
- OpenReview PDF: https://openreview.net/pdf?id=LRrbD8EZJl
- Code: https://github.com/dmksjfl/OTDF

</article>
