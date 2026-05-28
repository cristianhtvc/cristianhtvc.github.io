---
title: "Latent Safety-Constrained Policy Approach for Safe Offline Reinforcement Learning"
description: "> **论文标题**: Latent Safety-Constrained Policy Approach for Safe Offline Reinforcement Learning"
tags:
  - Offline RL
---

# Latent Safety-Constrained Policy Approach for Safe Offline Reinforcement Learning

<div class="paper-hero">
<p class="paper-meta">详见笔记 · ICLR 2025 · ICLR / 2025</p>
<div class="tag-row"><span>Offline RL</span></div>
</div>

<article class="note-body" markdown>

﻿# Latent Safety-Constrained Policy Approach (LSPC) 深度阅读笔记

> **论文标题**: Latent Safety-Constrained Policy Approach for Safe Offline Reinforcement Learning
> **会议**: ICLR 2025 (已接受)
> **作者**: Prajwal Koirala, Zhanhong Jiang, Soumik Sarkar, Cody Fleming
> **机构**: Iowa State University, Ames, Iowa, USA
> **资源**: [OpenReview](https://openreview.net/forum?id=bDt5qc7TfO) · [arXiv:2412.08794](https://arxiv.org/abs/2412.08794) · [Code](https://github.com/prajwalkoirala/LSPC)
> **阅读日期**: 2026-05-27

---

## 0. 一句话总结

LSPC 利用**条件变分自编码器（CVAE）**在隐空间（latent space）中建模安全约束，推导出"保守安全策略"后，将 safe offline RL 形式化为**约束奖励回报最大化问题**，通过 reward-Advantage Weighted Regression 在隐约束空间中优化，在保持安全合规的同时最大化奖励，并提供了性能界和样本复杂度理论保证。

---

## 1. 定位 (Positioning)

本文属于 **Safe Offline Reinforcement Learning（安全离线强化学习）** 领域，位于 offline RL 与 safe RL 的交汇点。

**在安全离线 RL 版图中的位置**:
- **BC-Safe**：仅在数据集的"安全子集"上做行为克隆——没有奖励信号引导优化，保守但不追求最优
- **CDT** (Conditioned Decision Transformer)：以目标和约束成本为条件做序列建模——提示条件（prompt）选择困难，安全与奖励难以同时满足
- **CPQ** (Conservative Policy Q-learning)：在 Q 学习中对成本函数施加保守估计——偏保守
- **FISOR** (Feasibility Informed Safe Offline RL)：通过可行性引导实现硬约束满足——过度保守，奖励性能受限
- **VOCE**：变分优化+保守估计——偏向价值约束

**LSPC 的差异化**：
1. **隐空间安全建模**：通过 CVAE 将安全约束编码到 latent space 中，形成"隐式安全优先约束"（Latent Safety-Prioritized Constraints）
2. **解耦安全与奖励**：先用 CVAE+Cost-AWR 推导保守安全策略 π_s，再在隐约束空间中做 reward maximization
3. **理论+实践桥梁**：提供了 performance bound 和 sample complexity 分析，并在多种 benchmark 上验证（含自动驾驶 Metadrive）

---

## 2. 第一作者简介

**Prajwal Koirala**，Iowa State University 博士生，导师 Cody Fleming（机械工程/AI系统安全方向）。

研究聚焦于离线强化学习、安全RL、自动驾驶。近5年代表论文：

| 年份 | 论文 | 会议/期刊 |
|------|------|-----------|
| 2025 | **LSPC: Latent Safety-Constrained Policy for Safe Offline RL** | ICLR 2025 |
| 2025 | LexiSafe: Offline Safe RL with Lexicographic Safety-Reward Hierarchy | CoRR |
| 2025 | Feasibility Informed Advantage Weighted Regression (FAWAC) | CDC 2025 |
| 2024 | Solving Offline RL with Decision Tree Regression | CoRL 2024 |
| 2024 | F1Tenth Autonomous Racing with Offline RL Methods | ITSC 2024 |
| 2024 | Reframing Offline RL as a Regression Problem | CoRR |
| 2025 | Flow-Based Single-Step Completion for Efficient Policy Learning | CoRR |

> 来源: DBLP (pid: 367/3704)。Koirala 是活跃的 junior researcher，已有 ICLR/CoRL/CDC/ITSC 等多个顶级会议论文，研究方向围绕 offline RL 与 safe RL 交叉。合作网络以 ISU 团队为核心。

---

## 3. 核心问题

### 问题陈述
在**安全离线强化学习**中，目标是从静态数据集中学习一个策略，该策略**最大化累积奖励**同时**严格满足安全约束**（成本不超过阈值 κ）。现有方法面临"安全-奖励"的平衡难题：要么过度保守导致低奖励，要么奖励优化导致安全违规。

### 问题的数学形式：Constrained MDP (CMDP)

$$\max_{\pi} \mathbb{E}\left[\sum_{t=0}^T \gamma^t r(s_t, a_t)\right] \quad \text{s.t.} \quad \mathbb{E}\left[\sum_{t=0}^T \gamma^t c(s_t, a_t)\right] \leq \kappa$$

其中 $c(s_t, a_t)$ 是每步的安全成本，$\kappa$ 是成本阈值。离线的额外挑战在于：数据分布偏移导致价值函数（奖励 Q 和成本 Q）估计不准。

### 现有方法的局限

| 方法 | 思路 | 局限 |
|------|------|------|
| BC-Safe | 仅在安全子集上行为克隆 | 依赖大量安全数据；无视奖励标签；换 κ 需重新过滤数据 |
| CDT | 序列建模 + 条件生成 | 提示条件选择和解释困难；安全与奖励条件难以同时满足 |
| CPQ | 对成本 Q 施加保守偏置 | 仍偏保守 |
| FISOR | 硬约束 + 可行性引导 | 过度保守，奖励性能严重受限 |

**核心矛盾**：如何在保证安全约束的同时尽可能优化奖励？如何让安全约束可以随 κ 调整而不需要重新训练？

### 论据来源
- Table 1 的全面比较：FISOR 在低 κ 下安全但奖励极低（如 Metadrive EasySparse: 0.45, 对比 LSPC-O 的 0.94）；CDT 则在低 κ 时安全违规频繁
- 实验覆盖 3 个 domain (Metadrive, Safety Gym, Bullet Safety Gym)、多个 κ 阈值和 random seeds

---

## 4. 前置知识

### 4.1 条件变分自编码器 (CVAE)
CVAE 是 VAE 的条件扩展，目标是在给定条件 y 下最大化条件似然 p_θ(x|y)。训练时最大化变分下界（ELBO）：

$$\max_{\alpha,\beta} \mathbb{E}_{z \sim q_\alpha}[\log p_\beta(x|y,z)] - D_{KL}[q_\alpha(z|x,y) \| p(z|x,y)]$$

其中 q_α 是编码器（encoder），p_β 是解码器（decoder），z 是隐变量。解码器 p_β(x|y,z) 在给定先验 z ~ N(0,I) 时生成符合条件分布的样本。

**在 LSPC 中的作用**：将状态-动作对 (s,a) 编码为隐变量 z，解码器生成符合安全约束的动作。

### 4.2 Advantage Weighted Regression (AWR)
AWR 是一种从离线数据中提取策略的方法，通过**优势加权**最大化动作的对数似然：

$$\pi = \arg\max_\pi \mathbb{E}_{(s,a)\sim D}\left[\exp(\lambda \cdot A(s,a)) \cdot \log \pi(a|s)\right]$$

其中 A(s,a) = Q(s,a) - V(s) 是优势函数，λ 是逆温度参数。当 A 为正（动作优于平均），该动作被"加权鼓励"；当 A 为负，该动作被弱化。

**在 LSPC 中的作用**：
- **Cost-AWR**：用成本优势 A^c（负值意味着安全）加权，提取安全策略 π_s
- **Reward-AWR**：用奖励优势 A^r 在隐安全约束空间中提取最优策略

### 4.3 Implicit Q-Learning (IQL)
IQL 通过**不对称损失**学习价值函数，不需要采样 OOD 动作：
- 对 Q 函数使用 expectile 回归（对高值不敏感，对低值敏感——避免低估）
- 对 V 函数同样使用 expectile 回归

**在 LSPC 中的作用**：同时学习奖励价值 $(Q_r, V_r)$ 和成本价值 $(Q_c, V_c)$，分别用于奖励和成本的 AWR。

### 4.4 隐空间约束 (Latent Space Constraints)
不同于在原始动作空间中添加显式约束（如 KL 散度惩罚），隐空间约束将约束编码到低维 latent space 中。好处：
- 数据结构更紧凑、更易处理
- CVAE 的解码器天然地只生成"数据集内"的动作，无需额外 OOD 惩罚
- 约束的强度可通过 latent space 的缩放来调节

---

## 5. 思想与方法

### 5.1 核心思想

LSPC 的核心洞察是**两阶段解耦**：

1. **先学安全**：用 CVAE 建模行为策略 → 用 Cost-AWR 从行为策略中提取"保守安全策略" π_s → 隐空间 z 中编码了安全约束
2. **再学最优**：在 π_s 的隐约束空间中，用 Reward-AWR 最大化奖励，得到 LSPC-O 策略

这个框架的关键优势是：
- 安全策略 π_s 独立于奖励最大化，保证有"安全底线"
- 隐空间约束强度可调（通过限制 z 的采样范围），天然适配不同 κ
- CVAE 的解码器作为生成模型，保证了动作保持在数据支撑集内，无需显式 OOD 惩罚

### 5.2 方法细节

**阶段一：学习保守安全策略 π_s**

1. 用 CVAE (encoder α, decoder β) 在数据集 D 上建模行为策略 π_b：
   - 最大化 ELBO: log π_b(a|s) ≥ E_z[log p_β(a|s,z)] - D_KL[q_α(z|s,a) || p(z|s,a)]
   - 使编码器/解码器能够重建数据集中的动作

2. 用 IQL 学习成本价值函数 $(Q^c_\psi, V^c_\eta)$

3. 用 Cost-AWR 从行为策略中提取安全策略：
   $$\pi_s = \arg\max_\pi \mathbb{E}_{(s,a)\sim D}\left[\exp(\lambda(V^c_\eta(s) - Q^c_\psi(s,a))) \cdot \log \pi(a|s)\right]$$
   - 成本优势 A^c = V^c - Q^c：当某动作的成本低于平均值时，A^c 为正，该动作被加权鼓励
   - 在 CVAE 中，这相当于在隐空间中对编码器 α 进行 AWR 微调

**阶段二：约束奖励优化 (LSPC-O)**

4. 用 IQL 学习奖励价值函数 $(Q^r_\phi, V^r_\xi)$

5. 在隐安全约束空间中用 Reward-AWR 优化策略 π：
   $$\pi = \arg\max_\pi \mathbb{E}_{(s,a)\sim D}\left[\exp(\tau(Q^r_\phi(s,a) - V^r_\xi(s))) \cdot \log \pi(a|s)\right]$$
   - 但此时策略 π 受限于隐安全约束：解码器输入 z 的采样范围由安全策略 π_s 的编码器输出的分布决定
   - 即：z 从 q_α(z|s, a^safe) 中采样，再通过解码器生成动作

**两个变体**:
- **LSPC-S**：仅用 Cost-AWR，最大化安全性，用于安全优先场景
- **LSPC-O**：Cost-AWR + Reward-AWR，同时优化安全与奖励

### 5.3 为什么能解耦安全与奖励？

**机制链条**:
1. **CVAE 建模行为策略** → 解码器 p_β(a|s,z) 天然只生成数据支撑集内的动作 → 无需额外 OOD 惩罚
2. **Cost-AWR 推导 π_s** → 通过成本优势加权，从行为策略中筛选出安全动作的分布
3. **隐空间安全约束** → π_s 的编码器定义了"安全 z 区域"，后续优化限制在此区域内
4. **Reward-AWR 在约束内优化** → 在安全区域中选择高奖励动作

**证据强度**: 理论分析（Theorem 1 & 2）给出了性能界和约束违反界的上界；实验（Table 1, Figure 2）显示 LSPC-O 在多数任务上同时实现安全和最高奖励；CVAE 的 OOD 抑制通过隐空间可视化得到验证。

---

## 6. 算法与伪代码

### 算法名称: Latent Safety-Prioritized Constraints (LSPC)

### 训练流程:

```
算法: LSPC

输入: 离线数据集 D (含状态、动作、奖励、成本标签), 成本阈值 κ,
      逆温度参数 λ (cost), τ (reward)

阶段1: 训练价值函数
  同时训练 (无需 OOD 采样):
    - 奖励 IQL: (Q^r_ϕ, V^r_ξ)  ← expectile regression on D
    - 成本 IQL: (Q^c_ψ, V^c_η)  ← expectile regression on D

阶段2: 学习保守安全策略 π_s (CVAE + Cost-AWR)
  2.1 训练 CVAE (encoder α, decoder β):
      最大化: E_z[log p_β(a|s,z)] - D_KL[q_α(z|s,a) || N(0,I)]
      目的: 编码器/解码器能重建数据集动作

  2.2 用 Cost-AWR 微调编码器 α_s (推导 π_s):
      L_cost_awr = -E_{(s,a)~D}[ exp(λ(V^c_η(s) - Q^c_ψ(s,a))) · log π_s(a|s) ]
      等价于: 鼓励 α_s 对低成本动作输出更高的对数似然

  2.3 LSPC-S 策略 (安全策略):
      z ~ N(0,I) 或 z ~ q_αs(z|s, a_safe)
      a = p_β(a|s, z)  ← 解码器生成安全动作

阶段3: 约束奖励优化 (LSPC-O)
  3.1 用 Reward-AWR 在隐约束空间内优化:
      L_reward_awr = -E_{(s,a)~D}[ exp(τ(Q^r_ϕ(s,a) - V^r_ξ(s))) · log π(a|s) ]

  3.2 约束施加:
      - 解码器 β 保持不变 (保证动作在支撑集内)
      - 编码器在"安全区域"中采样 z
      - 可选: 对 z 的采样加额外限制 (如缩小方差)

  3.3 LSPC-O 推理:
      z ~ N(0, σ²I) (σ < 1 可控制保守程度)
      a = p_β(a|s, z)

输出: LSPC-S (安全优先) 或 LSPC-O (安全+奖励最优)
```

### 关键超参数:
| 参数 | 含义 | 影响 |
|------|------|------|
| λ | Cost-AWR 逆温度 | 越大越保守 |
| τ | Reward-AWR 逆温度 | 越大奖励优化越激进 |
| latent dim | 隐空间维度 | 编码能力 vs 过拟合 |
| σ | 隐空间采样方差 | 越小越保守 (LSPC-O) |

### 理论-实践差距:
- Theorem 1 & 2 给出了性能上界，但依赖于 Assumption 1 & 2（D_KL 假设），且上界带有 (1-γ)² 因子
- 理论分析中的 π_s 是基于真实成本的"真正安全策略"，实践中 Cost-AWR 的推导质量依赖于 IQL 对成本函数的估计精度

---

## 7. 实验与消融

### 7.1 实验设置
- **基准**: DSRL benchmark (Liu et al., 2023a)
- **环境** (3 domains):
  - **Metadrive**: 自动驾驶——Easy/Medium/Hard × Sparse/Mean/Dense = 9 种配置
  - **Safety Gymnasium**: CarButton, CarGoal, CarPush, SwimmerVel, HopperVel, HalfCheetahVel, Walker2dVel, AntVel 等
  - **Bullet Safety Gym**: BallRun, CarRun, DroneRun, AntRun, BallCircle, CarCircle, DroneCircle, AntCircle
- **Baselines**: BC-Safe, CDT, CPQ, FISOR (4个)
- **评估指标**: 归一化奖励回报 + 归一化成本回报；成本 < 1 视为安全
- **评估协议**: 每种方法在 3 个不同 κ 阈值、3 个 random seed 下评估

### 7.2 主要结果 (Table 1)

**关键发现**:

- **LSPC-O 在 27 个测试配置中平均奖励 0.67**，远超 BC-Safe (0.18)、CDT (0.58)、CPQ (0.42)、FISOR (0.80? 但看原文 Table 1，FISOR 虽然有些任务奖励不错，但安全违规严重)
- **LSPC-S 平均成本 0.17**，是安全合规最稳定的方法
- **Metadrive**（自动驾驶）: LSPC-O 在 Easy/Medium/Hard 的所有密度下均安全且奖励最优
- **Safety Gym**: LSPC-O 在 AntVel 上达到 0.95 奖励（成本 0.07），在 HalfCheetahVel 上达到 0.79 奖励（成本 0.01）
- **Bullet Safety Gym**: LSPC-O 在 CarRun 上达到 0.72 奖励（成本 0.00）

**与其他方法的对比亮点**:
- FISOR 虽然成本控制极好（经常 0.00），但奖励普遍被 LSPC-O 超越
- CDT 在低 κ 阈值下频繁出现成本超标的不可靠问题
- BC-Safe 无奖励优化，纯安全策略性能上限低

### 7.3 训练过程分析 (Figure 2)

以 Pybullet CarRun 和 Metadrive EasySparse 为例的训练曲线显示：
- LSPC-O 在**早期快速收敛到安全区域**（约 0.2M steps 即满足成本约束）
- 此后**奖励持续稳定增长**而不牺牲安全
- LSPC-S 奖励低于 LSPC-O 但成本更低且更早收敛
- 对比之下，BC-Safe 奖励停滞（无优化信号），CPQ 奖励震荡，FISOR 安全但奖励低

### 7.4 可视化分析 (Figure 3-5)

- **隐空间可视化 (Figure 3)**：CVAE 的 latent space 中，安全样本和不安全样本形成可分离的簇——验证了隐空间编码安全信息的能力
- **动作分布 (Figure 4)**：LSPC-O 生成的行动在数据支撑集内（CVAE 解码器保证），且集中于高奖励区域
- **成本预测 (Figure 5)**：IQL 学习的成本函数在数据集中可靠，外推误差被 CVAE 的生成约束抑制

### 7.5 迁移实验 (Figure 12, Metadrive Transfer)

在**零样本迁移**（训练于某一环境，测试于另一环境）中：
- Hard Dense 训练的 agent 转移到更简单环境时**奖励甚至超过源环境**
- Easy Sparse 训练的 agent 转移到更复杂环境时性能下降，但**安全始终维持**
- 这验证了 LSPC 的安全约束在不同分布下具有泛化能力

---

## 8. 展望

### 研究启发

1. **隐空间约束 = 自然 OOD 屏障**：CVAE 解码器的生成范围天然受限于训练分布的支持集，这使得 LSPC 无需显式 KL/MMD 惩罚即可保证 OOD 安全——这是一种更优雅的行为正则化方式。

2. **安全-奖励的两阶段解耦**：先推导安全底线，再从安全底线出发做奖励优化。这种"先保命再求优"的策略可推广到其他约束 RL 问题（如资源约束、公平性约束）。

3. **Cost-AWR 是安全策略提取的自然语言**：通过成本优势函数加权，Cost-AWR 天然地从混合数据中"筛选"出安全动作——无需显式过滤数据子集。

4. **理论-算法的有机结合**：Theorem 1-4 的性能界和样本复杂度分析直接与 LSPC 的架构设计挂钩（π_s 替换 π_b 收紧 KL 上界），这是少见的"算法设计紧跟理论指导"的案例。

5. **自动驾驶是 Safe Offline RL 的理想场景**：Metadrive 提供了真实的自动驾驶安全约束（碰撞=成本），离线数据（事故数据）天然适合 safe offline RL 范式。

### 局限性

1. **CVAE 容量限制**：CVAE 的生成质量决定了安全策略的表达力上限。在极高维动作空间或极度多模态的行为数据中，CVAE 可能难以准确建模。
2. **成本函数必须可标注**：LSPC 假设离线数据包含成本标签，而实际应用中安全成本的定义和标注可能困难（需要领域专家人工标注或仿真器）。
3. **IQL 的保守性隐式而非显式**：IQL 通过 in-sample 学习避免 OOD，但没有显式的悲观惩罚，在极端数据稀缺时可能仍会过估。
4. **隐空间约束强度的调控缺乏自动化**：σ 的调节需要人工设定，缺乏自适应机制来匹配不同的 κ。
5. **理论假设强**：Assumption 1 & 2 的 D_KL 上界假设在数据覆盖差的时候可能不成立。

### 后续研究方向

1. **自适应隐空间约束调节**：设计一种根据 κ 自动调整 latent space 约束强度（σ）的机制，实现无需重训练即可切换安全等级。
2. **Online fine-tuning 版本**：在离线训练基础上加入在线微调，利用 CVAE 的安全约束作为"安全屏障"，允许有限探索。
3. **多目标安全约束扩展**：将 LSPC 框架扩展到多维度成本约束（如同时限制碰撞率、能耗、急刹车次数），在隐空间中编码多维约束结构。

---

## 链接

- **OpenReview**: [https://openreview.net/forum?id=bDt5qc7TfO](https://openreview.net/forum?id=bDt5qc7TfO)
- **arXiv**: [https://arxiv.org/abs/2412.08794](https://arxiv.org/abs/2412.08794)
- **代码**: [https://github.com/prajwalkoirala/LSPC](https://github.com/prajwalkoirala/LSPC)
- **DBLP**: [https://dblp.org/rec/conf/iclr/KoiralaJSF25](https://dblp.org/rec/conf/iclr/KoiralaJSF25)

</article>
