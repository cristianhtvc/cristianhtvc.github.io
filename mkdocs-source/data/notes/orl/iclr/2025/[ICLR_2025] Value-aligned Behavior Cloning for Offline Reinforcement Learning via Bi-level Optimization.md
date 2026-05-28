---
title: "Value-aligned Behavior Cloning for Offline Reinforcement Learning via Bi-level Optimization"
description: "VACO 用双层优化学习样本权重，让行为克隆在保持数据支持的同时对齐预训练价值函数。"
tags:
  - Offline RL
  - Behavior Cloning
  - Bi-level Optimization
  - Value Alignment
  - D4RL
---

# Value-aligned Behavior Cloning for Offline Reinforcement Learning via Bi-level Optimization

| 字段 | 内容 |
|---|---|
| Title | Value-aligned Behavior Cloning for Offline Reinforcement Learning via Bi-level Optimization |
| Year | 2025 |
| Source | ICLR 2025 Conference Paper |
| Authors | Xingyu Jiang, Ning Gao, Xiuhui Zhang, Hongkun Dou, Yue Deng |
| Affiliations | Beihang University; Beijing Zhongguancun Academy，见论文首页 |
| Tags | Offline RL; behavior cloning; value alignment; bi-level optimization; meta-scoring |

**一句话概括：** VACO 用一个 meta-scoring network 为离线样本分配权重，把行为克隆改成 value-aligned weighted BC，从而同时缓解 OOD 动作风险和普通 BC 不区分数据质量的问题。

**我对这篇论文的定位：** 这是 offline RL 中“BC + value guidance”路线的工作，邻近 TD3+BC、IQL/AWR、PLAS/EDP、Decision Transformer 等策略抽取方法。它的重点不是重新设计 Q-learning，而是让行为克隆阶段知道哪些数据动作更值得模仿。

## 1. 第一作者相关信息

第一作者 **Xingyu Jiang**。论文首页显示其单位为 Beihang University。公开信息中本次主要核验到本文 ICLR 2025 记录；未能完整确认其个人主页和全部近期论文。

| year | title | venue/source | topic tag | relation to this paper |
|---|---|---|---|---|
| 2025 | Value-aligned Behavior Cloning for Offline Reinforcement Learning via Bi-level Optimization | ICLR 2025 | Offline RL, BC | 本文 |

从本文看，作者关心的是 offline RL 中 policy extraction 的一个具体矛盾：BC 稳定但盲目，value maximization 有用但容易 OOD；双层优化试图在两者之间建立可学习的接口。

## 2. 研究问题

offline RL 有两个常见风险：

1. **OOD overestimation**：策略选出数据外动作，Q 函数错误高估。
2. **Value alignment failure**：纯 BC 只模仿数据，不知道哪些数据动作高价值。

VACO 的问题是：如何在不离开数据支持太远的情况下，让 BC 更偏向高价值样本？

普通 BC 目标是：

$$
J_{\mathrm{BC}}(\theta)=\mathbb{E}_{(s,a)\sim D}\left[\|\pi_\theta(s)-a\|^2\right].
$$

它的问题是所有样本权重相同，专家动作、次优动作和失败动作都被同样模仿。

## 3. 背景知识

| term | plain Chinese meaning | role in this paper |
|---|---|---|
| Behavior Cloning | 监督学习模仿数据动作 | VACO 的内层学习 |
| Value alignment | 策略动作应与高 Q 动作一致 | VACO 的外层目标 |
| Meta-scoring network | 给样本分配可学习权重的网络 | VACO 核心组件 |
| Bi-level optimization | 内层学策略，外层学权重/超参数 | 方法框架 |
| Controlled noise | 给 policy 输入加逐渐减小的噪声 | 有限探索和鲁棒性 |
| In-sample domain | 离线数据覆盖的动作区域 | VACO 避免 OOD 的依托 |

BC 的优势是不会主动跑到数据集之外，因此天然安全；缺点是数据质量差时性能差。TD3+BC 这类方法把 Q maximization 和 BC regularization 加在一起，但权重通常固定，无法细粒度决定哪些样本应强模仿。VACO 用 meta-scoring network 让每个 $(s,a,Q(s,a))$ 都有自己的权重。

## 4. 问题分析

论文 Figure 1 把 offline RL 的问题拆成两块：OOD 区域价值容易被高估；in-sample 区域内部也有好动作和坏动作，BC 可能收敛到次优数据动作。也就是说，即使完全避免 OOD，策略抽取仍可能因“模仿错对象”而失败。

如果直接令权重网络最小化 weighted BC loss，它可能把所有权重设为 0，出现平凡解。因此 VACO 把权重网络的学习放到外层，用 value maximization 约束它：权重应使内层 BC 学出的策略有更高预估回报。

## 5. 思想与方法

VACO 的 weighted BC loss 为：

$$
J_{\mathrm{BC}}^w(\theta)
=\mathbb{E}_{(s,a)\sim D}
\left[w_\alpha(s,a,Q_\psi(s,a))\cdot \|\pi_\theta(s)-a\|^2\right].
$$

内层优化策略 $\pi_\theta$，外层优化权重网络 $w_\alpha$：

$$
\min_\alpha J_\pi(\theta)
=\mathbb{E}_{s\sim D}\left[-Q_\psi(s,\pi_\theta(s+\mathcal{N}(0,\sigma)))\right],
\quad
\theta^\star(\alpha)=\arg\min_\theta J_{\mathrm{BC}}^w(\theta).
$$

直观解释：权重网络要给样本打分，使得“按这些权重克隆出来的策略”在预训练 Q 函数下更高价值。逐渐减小的噪声让外层在训练早期有一点局部探索，但不会像在线 RL 那样离开数据分布太远。

## 6. 算法与伪代码

算法名：**VACO: Value-aligned Behavior Cloning via Bi-level Optimization**。

1. 输入固定离线数据集 $D$，初始化 value network $Q_\psi$、policy $\pi_\theta$、meta-scoring network $w_\alpha$。
2. **Value training phase**：按 IQL 式 TD learning 预训练 value/critic。
3. **Bi-level phase**：
4. 采样 batch $(s,a,r,s')$。
5. 固定 $w_\alpha$，用 weighted BC 更新 policy $\pi_\theta$。
6. 固定 $\pi_\theta$，用 value-alignment objective 更新 $w_\alpha$。
7. 外层输入加入逐渐减小的 Gaussian noise。
8. 测试时只保留 policy，meta-scoring network 不参与推理。

Appendix 还给出把 value update 合并进同一循环的 Algorithm 2，说明 VACO 可替换不同 value loss 或 BC 类 policy extraction loss。

## 7. 实验与消融

实验包括 D4RL MuJoCo 和 AntMaze。基线分为经典方法 BC/TD3/CQL，显式正则 TD3+BC/IQL/PRDC/TD7/A2PO，隐式正则 MOPO/PLAS/EDP，以及 return-conditioned DT/DS4/DC 等。

主要结果：

- Table 1：MuJoCo locomotion 上，VACO 在多数数据质量设置中达到最好或接近最好，平均分高于若干强基线。
- Table 2：AntMaze 上，VACO 平均 76.7，高于 IQL 58.3、CQL 50.6、EDP 73.4，在 large-play / large-diverse 上也更强。
- Figure 3：与启发式样本权重相比，meta-scoring 明显更好，说明简单按 value 或 advantage 加权不够。
- Figure 4：去掉 meta-scoring 输入中的 value 或 state 会显著降分；逐渐减小噪声对 halfcheetah 等任务有帮助。

实验的强项是覆盖了多类 policy extraction baseline；弱点是 value function 质量对外层优化很关键，若 Q 本身偏差严重，权重网络会被错误价值引导。

## 8. 展望

启发：

- BC 不该被视为“低阶 baseline”，它可以通过可学习权重承担更强的 policy extraction。
- value alignment 的关键不只是加一个 Q maximization 项，而是让模仿目标本身被价值塑形。
- 训练期复杂组件可以不出现在测试期，这对机器人部署很友好。

局限：

- 双层优化实现复杂，稳定性依赖近似求解。
- Q 估计错误会直接影响 meta weights。
- 论文主要在 D4RL 上验证，真实人类数据或多模态数据仍未知。

后续方向：

- 将 meta-scoring 与 uncertainty-aware critic 结合，避免被高估 Q 误导。
- 用 trajectory-level 权重扩展 VACO，使其更适合长程任务。
- 在 offline PbRL 中用偏好模型替代数值 Q 来学习 imitation weights。

## Links

- Paper page: https://proceedings.iclr.cc/paper_files/paper/2025/hash/9a8647d31d5c48f7ca38f9b3de0054f1-Abstract-Conference.html
- OpenReview PDF: https://openreview.net/pdf?id=UcgXih5Wf4
- Code: 未能从论文和公开检索中确认官方代码链接
