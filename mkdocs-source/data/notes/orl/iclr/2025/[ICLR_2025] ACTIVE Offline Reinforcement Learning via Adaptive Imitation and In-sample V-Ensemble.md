---
title: "ACTIVE: Offline Reinforcement Learning via Adaptive Imitation and In-sample V-Ensemble"
description: "ACTIVE 用 V 函数集成抑制 in-sample offline RL 的初始价值误差传播，并用自适应克隆温度缓解过度保守。"
tags:
  - Offline RL
  - In-sample Learning
  - Implicit Q-Learning
  - Ensemble
  - Policy Extraction
---

# ACTIVE: Offline Reinforcement Learning via Adaptive Imitation and In-sample V-Ensemble

| 字段 | 内容 |
|---|---|
| Title | ACTIVE: Offline Reinforcement Learning via Adaptive Imitation and In-sample V-Ensemble |
| Year | 2025 |
| Source | ICLR 2025 Conference Paper |
| Authors | Tianyuan Chen, Ronglong Cai, Faguo Wu, Xiao Zhang |
| Affiliations | Beihang University; Zhongguancun Laboratory 等，见论文首页 |
| Tags | Offline RL; In-sample value learning; IQL/SQL; V-ensemble; adaptive imitation |

**一句话概括：** 这篇论文认为 IQL/SQL 一类 in-sample 方法的问题不只是“不看 OOD 动作”，还会让价值函数过拟合早期 Q 误差；ACTIVE 通过 V-ensemble 和自适应克隆温度，在保持 in-sample 安全性的同时减少过度正则化。

**我对这篇论文的定位：** ACTIVE 是 IQL 系列的“误差传播修补”工作。它不走 CQL/EDAC 那种显式 OOD 惩罚路线，而是在隐式 TD backup 内部处理 V 函数过拟合和策略抽取过保守的问题。它适合放在 IQL、SQL、EQL、MSG、RORL 这一条 in-sample / ensemble offline RL 线索里读。

## 1. 第一作者相关信息

第一作者 **Tianyuan Chen**。论文首页显示其单位为 Beihang University 相关实验室。

## 2. 研究问题

论文研究的是 **in-sample offline RL 的过度正则化与价值误差传播问题**。IQL 的核心安全性来自不显式对策略动作做 TD backup，而是用数据内动作上的 expectile value 近似 in-sample maximum。但作者指出，当数据偏次优或需要轨迹拼接时，这类方法往往不得不用较强的隐式正则化；如果把 expectile 设得更激进，V 函数会捕捉早期 Q 估计误差，并通过 bootstrapping 扩散。

设离线数据集为 $D=\{(s,a,r,s')\}$，行为策略为 $\mu(a|s)$。IQL 的 value update 形式为：

$$
L_V(\psi)=\mathbb{E}_{(s,a)\sim D}\left[L_2^\tau(Q_{\bar{\theta}}(s,a)-V_\psi(s))\right].
$$

直观含义：$V(s)$ 不是对当前策略动作求期望，而是在数据内动作上做 expectile regression；$\tau$ 越大，越接近数据内最大 Q。问题在于，若某些数据内动作的早期 $Q$ 被高估，较大的 $\tau$ 会放大这些异常值。

## 3. 背景知识

| term | plain Chinese meaning | role in this paper |
|---|---|---|
| Offline RL | 只能用固定数据集学习策略，不能再和环境交互 | 总问题设定 |
| Distribution shift | 学到的策略访问的数据分布偏离数据集支持 | offline RL 的核心风险 |
| In-sample learning | TD backup 只依赖数据内动作或数据内 value | IQL/SQL 的安全机制 |
| Expectile regression | 用不对称平方损失估计分布高分位附近的值 | IQL 的 $V$ 学习方式 |
| AWR | 按 advantage 加权克隆数据动作 | IQL 的策略抽取机制 |
| V-ensemble | 多个 $V$ 网络估计数据内不确定性 | ACTIVE 的主要组件 |
| Adaptive cloning temperature | 自动调节 imitation 强度 | ACTIVE 缓解过保守的策略抽取 |

标准 offline RL 会担心策略选出数据集中没有覆盖的动作，导致 Q 函数在 OOD 区域乱泛化。IQL 的思路更保守：不直接用 $\pi(s')$ 采样的动作做 TD target，而用 $V(s')$ 表示数据内可达动作的高价值估计。这样避免了最危险的 OOD action query。

但本文指出，**in-sample 不等于不会过估计**。因为 $Q$ 和 $V$ 都是函数逼近器，初始误差可能在数据内动作上出现；若 $V$ 过度拟合这些高估值，下一轮 Q target 又会继承该误差。于是 in-sample 方法仍可能出现灾难性过估计，只是来源从 OOD query 变成了 in-sample value overfitting。

## 4. 问题分析

论文的诊断路径分两步。

第一，作者分析 generalized IQL：不同的 convex loss $f$ 对应不同的隐式 actor $\pi_{\mathrm{imp}}$。这说明 IQL/SQL 不是没有策略约束，**而是通过 $V$ update 间接定义了一个隐式策略分布**。若正则化太强，策略抽取太像数据；若正则化太弱，$V$ 又容易追逐高估 Q。

第二，作者用 NTK 风格分析说明 V-ensemble 可以抵消初始误差项。论文 Theorem 4.5 将 Q 的迭代估计分解为奖励传播项、初始误差项与 ensemble 产生的惩罚项。关键直觉是：多个 $V$ 网络对同一状态的估计分歧反映了 in-sample epistemic uncertainty；用 ensemble 的保守聚合做 target，可以压低被单个 V 网络放大的初始高估。

实验上，Figure 1 和 Figure 3 展示 IQL/SQL 在较大 expectile 或较小正则化系数下可能不稳定，而 IVE 允许使用更激进的 expectile 来做更强的次优动作过滤。

## 5. 思想与方法

ACTIVE 有两个核心组件。

1. **In-sample V-Ensemble (IVE)**：训练多个 $V_i(s)$，在 critic target 和 advantage estimation 中使用 ensemble 的保守聚合，例如最小值或分位数。这相当于把“不确定的高 value”从 bootstrapping target 中扣掉，减少初始误差传播。

2. **Adaptive Cloning Temperature (ACT)**：把策略抽取写成一个带平均 log-likelihood 约束的优化问题，并用 dual gradient descent 自动调节 imitation 强度。这样在数据质量较差时，策略可以减少对全体行为动作的平均模仿，而更关注高 advantage 的动作。

策略损失的形式可概括为：

$$
L_\pi(\phi)
=-\mathbb{E}_{s\sim D,a\sim \pi_\phi}[Q_{\bar{\theta}}(s,a)]
-\beta\mathbb{E}_{(s,a)\sim D}[w(s,a)\log \pi_\phi(a|s)].
$$

其中 $w(s,a)$ 由 $Q(s,a)$ 与 V-ensemble 估计的 advantage 得到；$\beta$ 由 dual update 自适应调整。直观上，第一项鼓励策略找高价值动作，第二项把策略拉回数据支持，但拉回强度不是固定超参数。

## 6. 算法与伪代码

算法名：**ACTIVE: Actor-Critic with Temperature adjustment and In-sample Value Ensemble**。论文给出 ACTIVE-I 和 ACTIVE-S 两类实例，分别可理解为基于 IQL/SQL 风格的版本。

1. 初始化 policy $\pi_\phi$、critic $Q_\theta$、target critic $Q_{\bar{\theta}}$、$m$ 个 value 网络 $\{V_{\psi_i}\}_{i=1}^m$、dual temperature $\beta$。
2. 从离线数据集 $D$ 采样 batch。
3. 对每个 $V_{\psi_i}$ 执行 generalized IQL value update。
4. 用 ensemble 聚合后的 $V$ 构造 critic target，并更新 $Q_\theta$。
5. 用 EMA 更新 target critic。
6. 根据数据动作的平均 log-likelihood 与目标阈值 $H_D$ 更新 dual temperature。
7. 用 value-maximization + weighted behavior cloning 更新 policy。
8. 训练结束后部署 $\pi_\phi$。

实现上最敏感的是 ensemble size、expectile/regularization 选择、以及 ensemble 聚合分位数 $E_c[V]$。论文经验上常用较小 ensemble，运行时间只比 IQL/SQL 稍慢。

## 7. 实验与消融

实验使用 D4RL Gym MuJoCo、AntMaze 和 Kitchen。对比方法包括 CQL、TD3+BC、IQL、SQL，以及 ensemble 类方法 RORL、MSG、SAC-N。

主要结论：

- Table 2 显示 ACTIVE 通常优于对应 in-sample baseline，尤其在 suboptimal/diverse 数据集和需要 stitching 的 AntMaze、Kitchen 上更明显。
- Table 1 在 AntMaze 上与 RORL/MSG 比较，ACTIVE 不依赖显式 CQL 类正则也能取得有竞争力结果。
- Figure 3 说明 V-ensemble 让较大 expectile 更稳定；这正对应论文的核心诊断。
- Figure 6 的消融显示：IVE 对 AntMaze 这类 compositional dataset 更重要，ACT 对 MuJoCo 的 suboptimal dataset 更有帮助。
- Table 3 显示 ACTIVE 的运行开销相比 IQL/SQL 有增加但不大：1M steps 约 25-26 分钟，而 IQL/SQL 约 22-23 分钟。

需要谨慎的地方是：理论分析包含较强假设，特别是假设隐式策略在 value learning 过程中固定；实际深度 RL 中该假设并不完全成立。

## 8. 展望

对 ORL 研究者的启发：

- in-sample 方法的安全性不只取决于是否查询 OOD action，还取决于 value target 如何传播早期误差。
- ensemble 不一定只用于 OOD action penalty，也可以用于数据内 epistemic uncertainty。
- 策略抽取阶段是 offline RL 的独立瓶颈；固定 AWR 温度可能是过度保守的重要来源。

局限与问题：

- 理论对实际 IQL 动态做了简化。
- ensemble 虽然开销不大，但仍增加实现复杂度。
- 如何与 out-of-sample generalization 结合，论文只在 Discussion 中提出，未系统解决。

可能后续方向：

- 将 IVE 与 CQL/EDAC 类显式 pessimism 结合，区分 in-sample 和 OOD 两类不确定性。
- 研究 adaptive cloning target $H_D$ 的自动选择，减少数据集级调参。
- 分析 V-ensemble 的分歧是否能预测 trajectory stitching 成功率。

## Links

- Paper page: https://proceedings.iclr.cc/paper_files/paper/2025/hash/c06f788963f0ce069f5b2dbf83fe7822-Abstract-Conference.html
- DBLP: https://dblp.org/rec/conf/iclr/ChenCWZ25
- Code: 未能从论文和公开检索中确认官方代码链接
