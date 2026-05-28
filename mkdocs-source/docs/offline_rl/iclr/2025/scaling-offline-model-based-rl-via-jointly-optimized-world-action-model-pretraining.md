---
title: "Scaling Offline Model-Based RL via Jointly-Optimized World-Action Model Pretraining"
description: "JOWA 用共享 Transformer 同时预训练世界模型和动作价值模型，探索 Atari 多任务 offline model-based RL 的 scaling。"
tags:
  - Offline RL
  - Model-Based RL
  - World Model
  - Atari
  - Scaling
---

# Scaling Offline Model-Based RL via Jointly-Optimized World-Action Model Pretraining

<div class="paper-hero">
<p class="paper-meta">Jie Cheng, Ruixi Qiao, Yingwei Ma, Binhua Li, Gang Xiong, Qinghai Miao, Yongbin Li, Yisheng Lv · ICLR 2025 · ICLR / 2025</p>
<div class="tag-row"><span>Offline RL</span><span>Model-Based RL</span><span>World Model</span><span>Atari</span><span>Scaling</span></div>
</div>

<article class="note-body" markdown>

| 字段 | 内容 |
|---|---|
| Title | Scaling Offline Model-Based RL via Jointly-Optimized World-Action Model Pretraining |
| Year | 2025 |
| Source | ICLR 2025 Poster；OpenReview 显示 Published 22 Jan 2025，Last Modified 24 Mar 2025 |
| Authors | Jie Cheng, Ruixi Qiao, Yingwei Ma, Binhua Li, Gang Xiong, Qinghai Miao, Yongbin Li, Yisheng Lv |
| Affiliations | Institute of Automation, Chinese Academy of Sciences；University of Chinese Academy of Sciences；Alibaba Group |
| Tags | Offline Model-Based RL；World Model；Distributional CQL；Atari；Scaling Law |

**一句话概括**：JOWA 将视觉世界模型和动作价值模型放进共享 Transformer backbone 中联合预训练，用世界模型监督信号稳定大模型 TD-learning，并在推理时用 beam-search planning 补偿 Q 值误差。

**我对这篇论文的定位**：这是 offline RL scaling 方向的一篇代表性工作。它试图回答“能否像语言/视觉模型那样扩大 RL 模型和数据规模”，并把答案放在多 Atari 游戏、图像观测、世界模型与 Q-learning 结合的框架中。对 ORL 研究者来说，它的重要性在于展示：model-based 表征学习可以成为大规模 TD-learning 的稳定器。

## 1. 第一作者相关信息

Jie Cheng 来自中国科学院自动化研究所/中国科学院大学，与 Yisheng Lv 团队相关；Yongbin Li 来自阿里巴巴集团，Yisheng Lv 与 Yongbin Li 为共同通讯作者。OpenReview 页面列出 JOWA 为 ICLR 2025 Poster，官方仓库为 `https://github.com/CJReinforce/JOWA`。

| 年份 | 论文 | Venue/source | 主题标签 | 与本文关系 |
|---|---|---|---|---|
| 2025 | Scaling Offline Model-Based RL via JOWA | ICLR 2025 Poster | Large-scale offline MBRL | 本文主体 |
| 2024 | RIME / preference RL 相关工作 | 团队近作 | Preference RL, robust learning | 第一作者相关研究线之一 |
| 2024 | 视觉语言/多模态理解相关工作 | 团队近作 | Representation learning | 与大模型表征学习背景相关 |

该团队的研究兴趣横跨离线 RL、偏好学习、大规模预训练和智能交通/决策系统。JOWA 是把“世界模型预训练”推进到 offline RL scaling 的尝试。

## 2. 研究问题

离线 RL 的长期目标之一是从大规模异构数据中训练通用决策智能体。但以 TD-learning 为核心的 offline RL 在模型变大时容易不稳定，现有多任务方法又常退化为 conditional behavior cloning，依赖专家轨迹且泛化有限。

JOWA 研究的问题是：能否在多 Atari 游戏的图像观测离线数据上，联合训练一个既能预测未来、又能估计动作价值的大模型，并让模型容量扩大带来稳定收益？

设置上，JOWA 使用多游戏 Atari 离线数据（论文摘要称 6B tokens），图像观测经 VQ-VAE 离散化为 token，Transformer backbone 同时服务于 world head 与 Q/action head。训练是 offline-only；迁移到新游戏时使用少量离线微调数据。

## 3. 背景知识

World model 学习环境动态，例如给定过去观测和动作，预测下一帧、奖励和终止信号。它提供的是监督学习信号，通常比纯 TD target 更稳定。

Distributional RL 不只估计回报期望 $Q(s,a)$，而是估计回报分布 $Z(s,a)$。C51 将回报分布离散到固定 atoms 上，用分类式目标训练。JOWA 的 action model 采用 distributional CQL 风格，以便在离线数据上保守估计动作价值。

CQL 的核心是压低 OOD 动作价值、抬高数据动作价值：

$$
\mathcal{L}_{\mathrm{CQL}} = \alpha\left(\mathbb{E}_s \log\sum_a \exp Q(s,a) - \mathbb{E}_{(s,a)\sim D} Q(s,a)\right) + \mathcal{L}_{\mathrm{TD}}.
$$

含义：不要让数据外动作因为估值错误变得过于诱人。

| 术语 | 直观含义 | 在本文中的作用 |
|---|---|---|
| World model | 预测下一观测、奖励、终止 | 稳定表征与支持规划 |
| Action model | 估计动作价值/Q 分布 | 负责决策 |
| Shared backbone | 世界模型和 Q 模型共用 Transformer | 联合优化核心 |
| VQ-VAE tokens | 图像压缩后的离散视觉 token | 让 Atari 图像可被 Transformer 处理 |
| Beam search planning | 推理时搜索多步候选动作 | 补偿 Q 值估计误差 |
| Scaling law | 模型越大性能越好 | JOWA 的主要实验证据 |

为什么动机成立：纯 Q-learning 的梯度信号稀疏且高方差，模型变大后更容易发散或过拟合；世界模型的下一帧预测提供密集监督，能塑造共享表示，从而让大模型 TD-learning 更稳定。

## 4. 问题分析

Table 1 对比多任务 offline RL 方法，指出许多 prior 要么只学表征不直接学可泛化决策头，要么依赖专家轨迹。JOWA 的诊断是：需要同时预训练 representation 和 decision-making ability。

Figure 1 展示 JOWA 架构：VQ-VAE 把 Atari 帧压成 token，Transformer backbone 处理历史 token 和动作，world head 预测下一观测/reward/done，action head 输出 distributional Q。

Theorem 4.1 给出 beam search planning 的非正式保证：在一定假设下，搜索得到的动作价值可优于单步 Q 估计，从而缓解 Q-value estimation error。这个理论不是完整解决离线分布偏移，而是解释为什么推理时规划能补偿模型误差。

## 5. 思想与方法

JOWA 的指导原则是“用世界模型监督信号托住大规模 TD-learning”。世界模型学到的视觉动态、奖励和终止结构会反向塑造 Transformer 表示，使 action model 不必仅依赖不稳定的 Bellman 误差信号。

方法包含三部分：第一阶段训练或使用 VQ-VAE tokenizer；第二阶段在多游戏数据上联合训练 world-action Transformer；第三阶段测试时结合 Q head 和 world model 做 beam search planning。

新意在于 joint optimization。它不同于先训练 world model 再单独规划，也不同于只做多任务 BC；JOWA 让 world loss 和 action loss 同时作用于共享 backbone。

需要注意一个版本差异：本地 PDF 摘要写“outperforming ... by 71.4% on average”，OpenReview 当前摘要写“31.6% on average”。笔记中以表格和官方页面为准，并保留该差异作为版本核查点。

## 6. 算法与伪代码

算法名：JOWA (Jointly-Optimized World-Action model)。

1. 收集多 Atari 游戏离线数据，按帧构造轨迹。
2. 用 VQ-VAE 将 $84\times84$ 图像压缩为离散 token。
3. 将历史观测 token、动作、奖励输入 Transformer backbone。
4. World head 预测下一观测 token、奖励和终止信号，形成 $\mathcal{L}_{world}$。
5. Action head 输出 distributional Q，使用 CQL/TD 目标形成 $\mathcal{L}_{action}$。
6. 联合优化：

$$
\mathcal{L}_{JOWA}=\mathcal{L}_{world}+\lambda \mathcal{L}_{action}.
$$

7. 测试时，对候选动作使用 world model 展开若干步。
8. 用 beam search 保留高价值候选路径。
9. 执行最优路径的第一个动作，然后下一步重新规划。
10. 对新游戏，用约 5k transitions 少量离线数据微调。

直观解释：训练时用世界预测学“环境怎么变”，用 action loss 学“该做什么”；推理时让模型短暂想象未来，用搜索修正单步 Q 的短视或误差。

## 7. 实验与消融

实验使用 15 个 Atari 游戏预训练，论文摘要称 150M 参数模型在仅 10% subsampled offline data 下达到 78.9% human-level performance。Table 2 给出 15 游戏主结果；Figure 2 展示模型从 40M 到 70M 到 150M 的 scaling trend。

迁移实验见 Table 3：在 unseen games 上使用 5k transitions（约 4 条轨迹）微调，评估 DQN-normalized score，显示 JOWA 比重置 Q head 或纯表征迁移更样本高效。

Table 4-6 做关键消融：planning 对性能和速度的影响、world loss/action loss 的组合、不同训练损失的贡献。Table 5 报告 beam search 在若干游戏上提升裸 JOWA 表现，说明推理时规划不是装饰模块。

实验强证据是：JOWA 在 Atari 多任务图像观测上观察到模型规模扩大带来的收益，并能少样本迁移。弱点是：任务仍属于离散动作 Atari，连续控制、真实机器人和更复杂奖励结构中是否同样 scaling 尚未证明；world model 预测误差在长 horizon 中仍可能累积。

## 8. 展望

启发：第一，大规模 offline RL 可能需要辅助监督任务稳定表示；第二，world model 不只是规划器，也可以是 TD-learning 的正则器；第三，推理时搜索可作为 Q 值误差修正机制；第四，tokenized 视觉观测是连接 RL 与序列模型 scaling 的可行接口。

局限：Atari 的离散动作和视觉结构相对规则；beam search 依赖 world model 精度；CQL 保守性和 world loss 权重需要调参；版本间摘要数字存在差异，复现实验应以官方代码和最终表格为准。

后续方向：1. 将 JOWA 扩展到连续动作，通过 latent action tokenizer 或 diffusion action head 做规划。2. 研究 world model uncertainty 在 beam search 中的惩罚项。3. 在机器人多任务数据上验证 shared world-action pretraining 是否同样产生 scaling。

## Links

- Paper page: https://openreview.net/forum?id=T1OvCSFaum
- OpenReview PDF: https://openreview.net/pdf?id=T1OvCSFaum
- Code: https://github.com/CJReinforce/JOWA
- 本地 PDF: `[ICLR_2025] Scaling Offline Model-Based RL via Jointly-Optimized World-Action Model Pretraining.pdf`

</article>
